"""
Risk management for position sizing and trade validation.
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import config
from trading.portfolio import PortfolioManager
from analysis.signal_generator import TradingSignal, SignalType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeParams:
    """Parameters for executing a trade."""
    symbol: str
    market: str
    side: str  # 'buy' or 'sell'
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_value: float
    risk_amount: float
    risk_percent: float


class RiskManager:
    """Manage trading risk and position sizing."""

    def __init__(self, portfolio: PortfolioManager):
        """
        Initialize risk manager.

        Args:
            portfolio: Portfolio manager instance
        """
        self.portfolio = portfolio

    def validate_trade(self, signal: TradingSignal) -> Tuple[bool, str]:
        """
        Validate if a trade can be executed.

        Args:
            signal: Trading signal to validate

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if already have position
        if self.portfolio.get_position(signal.symbol):
            return False, f"Already have position in {signal.symbol}"

        # Check position limit
        if not self.portfolio.can_open_position():
            return False, f"Maximum positions ({config.MAX_POSITIONS}) reached"

        # Check drawdown limit
        if self.portfolio.is_max_drawdown_exceeded():
            return False, f"Maximum drawdown ({config.MAX_DRAWDOWN * 100}%) exceeded"

        # Check daily loss limit
        if self.portfolio.is_daily_loss_exceeded():
            return False, f"Daily loss limit ({config.MAX_DAILY_LOSS * 100}%) exceeded"

        # Check minimum confidence (configurable, default 65%)
        min_confidence = getattr(config, 'MIN_CONFIDENCE', 0.65)
        if signal.confidence < min_confidence:
            return False, f"Low confidence ({signal.confidence * 100:.0f}% < {min_confidence * 100:.0f}% required)"

        # Check stop loss is set
        if signal.stop_loss is None:
            return False, "No stop loss defined"

        # Check risk/reward ratio (configurable, default 2:1)
        min_rr = getattr(config, 'MIN_RISK_REWARD', 2.0)
        if signal.take_profit is not None:
            if signal.signal_type == SignalType.BUY:
                risk = signal.price - signal.stop_loss
                reward = signal.take_profit - signal.price
            else:
                risk = signal.stop_loss - signal.price
                reward = signal.price - signal.take_profit

            if risk <= 0:
                return False, "Invalid stop loss (risk <= 0)"

            risk_reward = reward / risk if risk > 0 else 0
            if risk_reward < min_rr:
                return False, f"Poor risk/reward ratio ({risk_reward:.2f} < {min_rr:.1f} required)"

        # Check volume confirmation if required
        if getattr(config, 'REQUIRE_VOLUME_CONFIRMATION', True):
            volume_confirmed = signal.indicators.get('volume_confirmed', False)
            if not volume_confirmed:
                logger.debug("Skipping %s: volume confirmation required", signal.symbol)
                return False, "Volume below average (confirmation required)"

        # Check trend alignment if required
        if getattr(config, 'REQUIRE_TREND_ALIGNMENT', True):
            trend_aligned = signal.indicators.get('trend_aligned', False)
            if not trend_aligned:
                logger.debug("Skipping %s: trend alignment required", signal.symbol)
                return False, "Price not aligned with EMA 21 trend"

        return True, "Trade validated (all criteria met)"

    def calculate_position_size(
        self,
        signal: TradingSignal,
        atr: Optional[float] = None
    ) -> TradeParams:
        """
        Calculate position size based on risk parameters.

        Uses ATR-based position sizing when available, otherwise uses
        fixed percentage of equity.

        Args:
            signal: Trading signal
            atr: Average True Range (optional)

        Returns:
            TradeParams with calculated values
        """
        equity = self.portfolio.get_equity()
        max_position_value = equity * config.POSITION_SIZE_PERCENT

        entry_price = signal.price
        stop_loss = signal.stop_loss

        # Calculate risk per share/unit
        if signal.signal_type == SignalType.BUY:
            risk_per_unit = entry_price - stop_loss
        else:
            risk_per_unit = stop_loss - entry_price

        risk_per_unit = abs(risk_per_unit)

        # Risk amount (2% of equity for risk-based sizing)
        risk_percent = 0.02  # 2% risk per trade
        risk_amount = equity * risk_percent

        # Calculate quantity based on risk
        if risk_per_unit > 0:
            qty_by_risk = risk_amount / risk_per_unit
            qty_by_value = max_position_value / entry_price
            qty = min(qty_by_risk, qty_by_value)
        else:
            qty = max_position_value / entry_price

        # Calculate actual position value
        position_value = qty * entry_price
        actual_risk = qty * risk_per_unit

        # Determine side
        side = "buy" if signal.signal_type == SignalType.BUY else "sell"

        # Calculate take profit
        take_profit = signal.take_profit
        if take_profit is None:
            # Default to 3:1 risk/reward
            if signal.signal_type == SignalType.BUY:
                take_profit = entry_price + (risk_per_unit * config.TAKE_PROFIT_MULTIPLIER)
            else:
                take_profit = entry_price - (risk_per_unit * config.TAKE_PROFIT_MULTIPLIER)

        return TradeParams(
            symbol=signal.symbol,
            market=signal.market,
            side=side,
            qty=qty,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_value=position_value,
            risk_amount=actual_risk,
            risk_percent=actual_risk / equity * 100
        )

    def adjust_for_volatility(
        self,
        params: TradeParams,
        atr: float,
        avg_atr: float
    ) -> TradeParams:
        """
        Adjust position size for current volatility.

        Reduces position size when volatility is higher than average.

        Args:
            params: Original trade parameters
            atr: Current ATR
            avg_atr: Average ATR

        Returns:
            Adjusted TradeParams
        """
        if avg_atr <= 0 or atr <= 0:
            return params

        volatility_ratio = atr / avg_atr

        if volatility_ratio > 1.5:
            # High volatility - reduce position by 30%
            adjustment = 0.7
            logger.info("High volatility (%.2fx avg), reducing position by 30%%", volatility_ratio)
        elif volatility_ratio > 1.2:
            # Elevated volatility - reduce by 15%
            adjustment = 0.85
            logger.info("Elevated volatility (%.2fx avg), reducing position by 15%%", volatility_ratio)
        elif volatility_ratio < 0.7:
            # Low volatility - can slightly increase (10%)
            adjustment = 1.1
            logger.info("Low volatility (%.2fx avg), increasing position by 10%%", volatility_ratio)
        else:
            adjustment = 1.0

        adjusted_qty = params.qty * adjustment
        adjusted_value = adjusted_qty * params.entry_price
        adjusted_risk = adjusted_qty * abs(params.entry_price - params.stop_loss)

        return TradeParams(
            symbol=params.symbol,
            market=params.market,
            side=params.side,
            qty=adjusted_qty,
            entry_price=params.entry_price,
            stop_loss=params.stop_loss,
            take_profit=params.take_profit,
            position_value=adjusted_value,
            risk_amount=adjusted_risk,
            risk_percent=adjusted_risk / self.portfolio.get_equity() * 100
        )

    def adjust_stop_loss_atr(
        self,
        entry_price: float,
        atr: float,
        side: str,
        multiplier: float = None
    ) -> float:
        """
        Calculate ATR-based stop loss.

        Args:
            entry_price: Entry price
            atr: Average True Range
            side: 'buy' or 'sell'
            multiplier: ATR multiplier (default from config)

        Returns:
            Stop loss price
        """
        multiplier = multiplier or config.STOP_LOSS_ATR_MULTIPLIER
        atr_distance = atr * multiplier

        if side == "buy":
            return entry_price - atr_distance
        else:
            return entry_price + atr_distance

    def get_trade_summary(self, params: TradeParams) -> str:
        """Get formatted trade summary."""
        return (
            f"\n{'=' * 40}\n"
            f"TRADE PARAMETERS\n"
            f"{'=' * 40}\n"
            f"Symbol: {params.symbol} ({params.market})\n"
            f"Side: {params.side.upper()}\n"
            f"Quantity: {params.qty:.4f}\n"
            f"Entry: ${params.entry_price:.4f}\n"
            f"Stop Loss: ${params.stop_loss:.4f}\n"
            f"Take Profit: ${params.take_profit:.4f}\n"
            f"Position Value: ${params.position_value:.2f}\n"
            f"Risk Amount: ${params.risk_amount:.2f} ({params.risk_percent:.2f}%)\n"
            f"{'=' * 40}\n"
        )

    def get_portfolio_risk_summary(self) -> Dict[str, Any]:
        """Get current portfolio risk metrics."""
        positions = self.portfolio.get_all_positions()
        equity = self.portfolio.get_equity()

        total_exposure = sum(
            (p.current_price or p.entry_price) * p.qty
            for p in positions
        )

        total_risk = sum(
            abs((p.current_price or p.entry_price) - (p.stop_loss or p.entry_price)) * p.qty
            for p in positions
            if p.stop_loss
        )

        return {
            "equity": equity,
            "total_exposure": total_exposure,
            "exposure_percent": total_exposure / equity * 100 if equity > 0 else 0,
            "total_at_risk": total_risk,
            "risk_percent": total_risk / equity * 100 if equity > 0 else 0,
            "position_count": len(positions),
            "available_slots": config.MAX_POSITIONS - len(positions),
            "max_drawdown": self.portfolio.max_drawdown * 100,
            "daily_pnl": self.portfolio.daily_pnl,
        }
