"""
Portfolio management for tracking positions and performance.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    market: str  # 'stock' or 'crypto'
    side: str  # 'long' or 'short'
    qty: float
    entry_price: float
    entry_time: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    highest_price: Optional[float] = None  # Highest price since entry (for trailing stop)
    initial_stop_loss: Optional[float] = None  # Original stop loss before trailing
    trailing_stop_active: bool = False  # Whether trailing stop has been activated


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    market: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_percent: float
    reason: str


class PortfolioManager:
    """Manage portfolio, positions, and performance tracking."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        data_path: str = "portfolio_data.json"
    ):
        """
        Initialize portfolio manager.

        Args:
            initial_capital: Starting capital
            data_path: Path to persist portfolio data
        """
        self.initial_capital = initial_capital
        self.data_path = Path(data_path)
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.cash = initial_capital

        # Performance metrics
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.daily_start_equity = initial_capital
        self.daily_pnl = 0.0

        self._load_data()

    def _load_data(self):
        """Load portfolio data from file."""
        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)

                self.cash = data.get("cash", self.initial_capital)
                self.peak_equity = data.get("peak_equity", self.initial_capital)
                self.max_drawdown = data.get("max_drawdown", 0.0)

                for pos_data in data.get("positions", []):
                    pos = Position(**pos_data)
                    self.positions[pos.symbol] = pos

                for trade_data in data.get("trade_history", []):
                    self.trade_history.append(Trade(**trade_data))

                logger.info("Loaded portfolio: $%.2f cash, %d positions",
                           self.cash, len(self.positions))
            except Exception as e:
                logger.error("Error loading portfolio data: %s", e)

    def _save_data(self):
        """Save portfolio data to file."""
        try:
            data = {
                "cash": self.cash,
                "peak_equity": self.peak_equity,
                "max_drawdown": self.max_drawdown,
                "positions": [asdict(p) for p in self.positions.values()],
                "trade_history": [asdict(t) for t in self.trade_history[-100:]],  # Keep last 100
            }
            with open(self.data_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Error saving portfolio data: %s", e)

    def get_equity(self) -> float:
        """Calculate total equity (cash + positions value)."""
        positions_value = sum(
            (p.current_price or p.entry_price) * p.qty
            for p in self.positions.values()
        )
        return self.cash + positions_value

    def get_available_cash(self) -> float:
        """Get available cash for trading."""
        return self.cash

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return len(self.positions) < config.MAX_POSITIONS

    def get_position_size_dollars(self) -> float:
        """Calculate dollar amount for new position."""
        equity = self.get_equity()
        return equity * config.POSITION_SIZE_PERCENT

    def open_position(
        self,
        symbol: str,
        market: str,
        side: str,
        qty: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Position:
        """
        Open a new position.

        Args:
            symbol: Asset symbol
            market: 'stock' or 'crypto'
            side: 'long' or 'short'
            qty: Quantity
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            New Position object
        """
        if symbol in self.positions:
            raise ValueError(f"Position already exists for {symbol}")

        if not self.can_open_position():
            raise ValueError(f"Maximum positions ({config.MAX_POSITIONS}) reached")

        position_value = qty * price
        if position_value > self.cash:
            raise ValueError(f"Insufficient cash: ${self.cash:.2f} < ${position_value:.2f}")

        position = Position(
            symbol=symbol,
            market=market,
            side=side,
            qty=qty,
            entry_price=price,
            entry_time=datetime.now().isoformat(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=price,
            unrealized_pnl=0.0,
            highest_price=price,  # Initialize to entry price
            initial_stop_loss=stop_loss,  # Store original stop loss
            trailing_stop_active=False
        )

        self.positions[symbol] = position
        self.cash -= position_value

        logger.info(
            "Opened position: %s %s %.4f @ $%.4f (SL: %s, TP: %s)",
            side, symbol, qty, price,
            f"${stop_loss:.4f}" if stop_loss else "None",
            f"${take_profit:.4f}" if take_profit else "None"
        )

        self._save_data()
        return position

    def close_position(
        self,
        symbol: str,
        price: float,
        reason: str = "Manual close"
    ) -> Optional[Trade]:
        """
        Close an existing position.

        Args:
            symbol: Asset symbol
            price: Exit price
            reason: Reason for closing

        Returns:
            Trade object if successful
        """
        position = self.positions.get(symbol)
        if not position:
            logger.warning("No position found for %s", symbol)
            return None

        # Calculate PnL
        if position.side == "long":
            pnl = (price - position.entry_price) * position.qty
        else:
            pnl = (position.entry_price - price) * position.qty

        pnl_percent = pnl / (position.entry_price * position.qty) * 100

        trade = Trade(
            symbol=symbol,
            market=position.market,
            side=position.side,
            qty=position.qty,
            entry_price=position.entry_price,
            exit_price=price,
            entry_time=position.entry_time,
            exit_time=datetime.now().isoformat(),
            pnl=pnl,
            pnl_percent=pnl_percent,
            reason=reason
        )

        # Update cash
        self.cash += price * position.qty

        # Remove position
        del self.positions[symbol]
        self.trade_history.append(trade)

        logger.info(
            "Closed position: %s @ $%.4f | PnL: $%.2f (%.2f%%) | %s",
            symbol, price, pnl, pnl_percent, reason
        )

        # Update drawdown metrics
        self._update_metrics()
        self._save_data()

        return trade

    def update_position_price(self, symbol: str, current_price: float):
        """Update current price for a position."""
        position = self.positions.get(symbol)
        if position:
            position.current_price = current_price
            if position.side == "long":
                position.unrealized_pnl = (current_price - position.entry_price) * position.qty
                # Update highest price for trailing stop
                if position.highest_price is None or current_price > position.highest_price:
                    position.highest_price = current_price
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.qty
                # For shorts, track lowest price
                if position.highest_price is None or current_price < position.highest_price:
                    position.highest_price = current_price

    def update_trailing_stop(self, symbol: str, atr: float) -> bool:
        """
        Update trailing stop loss for a position.

        Args:
            symbol: Asset symbol
            atr: Current ATR value for the asset

        Returns:
            True if stop loss was updated
        """
        if not getattr(config, 'TRAILING_STOP_ENABLED', False):
            return False

        position = self.positions.get(symbol)
        if not position or position.current_price is None:
            return False

        entry_price = position.entry_price
        current_price = position.current_price
        highest_price = position.highest_price or current_price

        # Calculate profit percentage
        if position.side == "long":
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price

        # Check if trailing stop should be activated
        activation_pct = getattr(config, 'TRAILING_STOP_ACTIVATION_PERCENT', 0.02)
        if profit_pct < activation_pct:
            return False  # Not enough profit yet to activate trailing stop

        # Trailing stop is now active
        if not position.trailing_stop_active:
            position.trailing_stop_active = True
            logger.info("Trailing stop activated for %s at %.2f%% profit", symbol, profit_pct * 100)

        # Calculate new trailing stop
        trail_multiplier = getattr(config, 'TRAILING_STOP_ATR_MULTIPLIER', 1.5)
        min_profit_lock = getattr(config, 'TRAILING_STOP_MIN_PROFIT_LOCK', 0.005)

        if position.side == "long":
            # For longs: trail below the highest price
            new_stop = highest_price - (atr * trail_multiplier)
            # Ensure we lock in minimum profit
            min_stop = entry_price * (1 + min_profit_lock)
            new_stop = max(new_stop, min_stop)

            # Only move stop up, never down
            if position.stop_loss is None or new_stop > position.stop_loss:
                old_stop = position.stop_loss
                position.stop_loss = new_stop
                logger.info(
                    "Trailing stop updated for %s: $%.4f -> $%.4f (highest: $%.4f, profit: %.2f%%)",
                    symbol, old_stop or 0, new_stop, highest_price, profit_pct * 100
                )
                self._save_data()
                return True
        else:
            # For shorts: trail above the lowest price
            new_stop = highest_price + (atr * trail_multiplier)
            # Ensure we lock in minimum profit
            max_stop = entry_price * (1 - min_profit_lock)
            new_stop = min(new_stop, max_stop)

            # Only move stop down, never up (for shorts)
            if position.stop_loss is None or new_stop < position.stop_loss:
                old_stop = position.stop_loss
                position.stop_loss = new_stop
                logger.info(
                    "Trailing stop updated for %s: $%.4f -> $%.4f (lowest: $%.4f, profit: %.2f%%)",
                    symbol, old_stop or 0, new_stop, highest_price, profit_pct * 100
                )
                self._save_data()
                return True

        return False

    def update_all_prices(self, prices: Dict[str, float]):
        """Update prices for all positions."""
        for symbol, price in prices.items():
            self.update_position_price(symbol, price)

    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if position hit stop loss or take profit.

        Args:
            symbol: Asset symbol
            current_price: Current price

        Returns:
            'stop_loss', 'take_profit', or None
        """
        position = self.positions.get(symbol)
        if not position:
            return None

        if position.side == "long":
            if position.stop_loss and current_price <= position.stop_loss:
                return "stop_loss"
            if position.take_profit and current_price >= position.take_profit:
                return "take_profit"
        else:  # short
            if position.stop_loss and current_price >= position.stop_loss:
                return "stop_loss"
            if position.take_profit and current_price <= position.take_profit:
                return "take_profit"

        return None

    def _update_metrics(self):
        """Update performance metrics."""
        equity = self.get_equity()

        # Update peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        else:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)

        # Update daily PnL
        self.daily_pnl = equity - self.daily_start_equity

    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of each day)."""
        self.daily_start_equity = self.get_equity()
        self.daily_pnl = 0.0
        logger.info("Daily metrics reset. Starting equity: $%.2f", self.daily_start_equity)

    def is_max_drawdown_exceeded(self) -> bool:
        """Check if maximum drawdown limit exceeded."""
        return self.max_drawdown >= config.MAX_DRAWDOWN

    def is_daily_loss_exceeded(self) -> bool:
        """Check if daily loss limit exceeded."""
        if self.daily_start_equity == 0:
            return False
        daily_loss_pct = -self.daily_pnl / self.daily_start_equity
        return daily_loss_pct >= config.MAX_DAILY_LOSS

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        equity = self.get_equity()
        total_return = (equity - self.initial_capital) / self.initial_capital * 100

        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl < 0]

        win_rate = len(winning_trades) / len(self.trade_history) * 100 if self.trade_history else 0

        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else float('inf')

        return {
            "initial_capital": self.initial_capital,
            "current_equity": equity,
            "cash": self.cash,
            "total_return_pct": total_return,
            "max_drawdown_pct": self.max_drawdown * 100,
            "daily_pnl": self.daily_pnl,
            "open_positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate_pct": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
        }

    def get_summary_string(self) -> str:
        """Get formatted performance summary."""
        stats = self.get_performance_summary()
        return (
            f"\n{'=' * 50}\n"
            f"PORTFOLIO SUMMARY\n"
            f"{'=' * 50}\n"
            f"Equity: ${stats['current_equity']:,.2f} ({stats['total_return_pct']:+.2f}%)\n"
            f"Cash: ${stats['cash']:,.2f}\n"
            f"Open Positions: {stats['open_positions']}\n"
            f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%\n"
            f"Daily PnL: ${stats['daily_pnl']:+,.2f}\n"
            f"\nTrade Statistics:\n"
            f"Total Trades: {stats['total_trades']}\n"
            f"Win Rate: {stats['win_rate_pct']:.1f}%\n"
            f"Avg Win: ${stats['avg_win']:,.2f}\n"
            f"Avg Loss: ${stats['avg_loss']:,.2f}\n"
            f"Profit Factor: {stats['profit_factor']:.2f}\n"
            f"{'=' * 50}\n"
        )
