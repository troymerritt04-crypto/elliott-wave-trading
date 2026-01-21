"""
Trade execution logic.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from data.alpaca_client import AlpacaClient
from data.binance_client import BinanceClient
from trading.portfolio import PortfolioManager, Position, Trade
from trading.risk_manager import RiskManager, TradeParams
from analysis.signal_generator import TradingSignal, SignalType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    success: bool
    order_id: Optional[str]
    symbol: str
    side: str
    qty: float
    price: float
    message: str


class TradeExecutor:
    """Execute trades across different brokers."""

    def __init__(
        self,
        portfolio: PortfolioManager,
        risk_manager: RiskManager,
        alpaca_client: Optional[AlpacaClient] = None,
        binance_client: Optional[BinanceClient] = None
    ):
        """
        Initialize trade executor.

        Args:
            portfolio: Portfolio manager instance
            risk_manager: Risk manager instance
            alpaca_client: Alpaca client for stocks
            binance_client: Binance client for crypto
        """
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.alpaca = alpaca_client
        self.binance = binance_client

    def execute_signal(self, signal: TradingSignal) -> ExecutionResult:
        """
        Execute a trading signal.

        Args:
            signal: Trading signal to execute

        Returns:
            ExecutionResult with execution details
        """
        # Validate trade
        is_valid, reason = self.risk_manager.validate_trade(signal)
        if not is_valid:
            logger.warning("Trade rejected for %s: %s", signal.symbol, reason)
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=signal.symbol,
                side=signal.signal_type.value,
                qty=0,
                price=signal.price,
                message=reason
            )

        # Calculate position size
        params = self.risk_manager.calculate_position_size(signal)

        # Execute based on market
        if signal.market == "stock":
            return self._execute_stock_trade(params)
        elif signal.market == "crypto":
            return self._execute_crypto_trade(params)
        else:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=signal.symbol,
                side=params.side,
                qty=params.qty,
                price=params.entry_price,
                message=f"Unknown market: {signal.market}"
            )

    def _execute_stock_trade(self, params: TradeParams) -> ExecutionResult:
        """Execute stock trade via Alpaca."""
        if not self.alpaca:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=params.symbol,
                side=params.side,
                qty=params.qty,
                price=params.entry_price,
                message="Alpaca client not initialized"
            )

        try:
            # Check if market is open
            if not self.alpaca.is_market_open():
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    symbol=params.symbol,
                    side=params.side,
                    qty=params.qty,
                    price=params.entry_price,
                    message="Stock market is closed"
                )

            # Round quantity to whole shares for stocks
            qty = int(params.qty)
            if qty < 1:
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    symbol=params.symbol,
                    side=params.side,
                    qty=qty,
                    price=params.entry_price,
                    message="Quantity too small (< 1 share)"
                )

            # Place market order
            order = self.alpaca.place_market_order(
                symbol=params.symbol,
                qty=qty,
                side=params.side
            )

            # Record in portfolio
            self.portfolio.open_position(
                symbol=params.symbol,
                market="stock",
                side="long" if params.side == "buy" else "short",
                qty=qty,
                price=params.entry_price,
                stop_loss=params.stop_loss,
                take_profit=params.take_profit
            )

            logger.info(
                "Stock order executed: %s %d %s @ ~$%.2f",
                params.side.upper(), qty, params.symbol, params.entry_price
            )

            return ExecutionResult(
                success=True,
                order_id=order.get("id"),
                symbol=params.symbol,
                side=params.side,
                qty=qty,
                price=params.entry_price,
                message="Order executed successfully"
            )

        except Exception as e:
            logger.error("Stock trade execution failed: %s", e)
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=params.symbol,
                side=params.side,
                qty=params.qty,
                price=params.entry_price,
                message=f"Execution error: {str(e)}"
            )

    def _execute_crypto_trade(self, params: TradeParams) -> ExecutionResult:
        """Execute crypto trade via Binance."""
        if not self.binance:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=params.symbol,
                side=params.side,
                qty=params.qty,
                price=params.entry_price,
                message="Binance client not initialized"
            )

        try:
            # Round quantity per exchange rules
            qty = self.binance.round_quantity(params.symbol, params.qty)

            # Check minimum notional
            symbol_info = self.binance.get_symbol_info(params.symbol)
            notional = qty * params.entry_price
            if notional < symbol_info.get("min_notional", 0):
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    symbol=params.symbol,
                    side=params.side,
                    qty=qty,
                    price=params.entry_price,
                    message=f"Order value ${notional:.2f} below minimum"
                )

            # Place market order
            order = self.binance.place_market_order(
                symbol=params.symbol,
                qty=qty,
                side=params.side
            )

            # Get actual fill price
            fill_price = order.get("avg_price", params.entry_price)

            # Record in portfolio
            self.portfolio.open_position(
                symbol=params.symbol,
                market="crypto",
                side="long" if params.side == "buy" else "short",
                qty=float(order.get("qty", qty)),
                price=fill_price,
                stop_loss=params.stop_loss,
                take_profit=params.take_profit
            )

            logger.info(
                "Crypto order executed: %s %.6f %s @ $%.4f",
                params.side.upper(), qty, params.symbol, fill_price
            )

            return ExecutionResult(
                success=True,
                order_id=order.get("id"),
                symbol=params.symbol,
                side=params.side,
                qty=float(order.get("qty", qty)),
                price=fill_price,
                message="Order executed successfully"
            )

        except Exception as e:
            logger.error("Crypto trade execution failed: %s", e)
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=params.symbol,
                side=params.side,
                qty=params.qty,
                price=params.entry_price,
                message=f"Execution error: {str(e)}"
            )

    def close_position(
        self,
        symbol: str,
        market: str,
        reason: str = "Manual close"
    ) -> ExecutionResult:
        """
        Close an existing position.

        Args:
            symbol: Asset symbol
            market: 'stock' or 'crypto'
            reason: Reason for closing

        Returns:
            ExecutionResult
        """
        position = self.portfolio.get_position(symbol)
        if not position:
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side="sell",
                qty=0,
                price=0,
                message=f"No position found for {symbol}"
            )

        try:
            if market == "stock" and self.alpaca:
                order = self.alpaca.close_position(symbol)
                exit_price = position.current_price or position.entry_price
            elif market == "crypto" and self.binance:
                order = self.binance.close_position(symbol)
                exit_price = order.get("avg_price") if order else position.entry_price
            else:
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    symbol=symbol,
                    side="sell",
                    qty=position.qty,
                    price=position.entry_price,
                    message="No broker client available"
                )

            if order:
                # Record close in portfolio
                trade = self.portfolio.close_position(symbol, exit_price, reason)

                return ExecutionResult(
                    success=True,
                    order_id=order.get("id"),
                    symbol=symbol,
                    side="sell" if position.side == "long" else "buy",
                    qty=position.qty,
                    price=exit_price,
                    message=f"Position closed: {reason}"
                )
            else:
                return ExecutionResult(
                    success=False,
                    order_id=None,
                    symbol=symbol,
                    side="sell",
                    qty=position.qty,
                    price=position.entry_price,
                    message="Failed to close position with broker"
                )

        except Exception as e:
            logger.error("Failed to close position %s: %s", symbol, e)
            return ExecutionResult(
                success=False,
                order_id=None,
                symbol=symbol,
                side="sell",
                qty=position.qty,
                price=position.entry_price,
                message=f"Close error: {str(e)}"
            )

    def check_and_close_stopped_positions(
        self,
        prices: Dict[str, Dict[str, float]]
    ) -> List[ExecutionResult]:
        """
        Check all positions for stop loss / take profit hits.

        Args:
            prices: Dict of {symbol: {'price': float, 'market': str}}

        Returns:
            List of ExecutionResults for closed positions
        """
        results = []

        for symbol, data in prices.items():
            price = data["price"]
            market = data["market"]

            # Update position price
            self.portfolio.update_position_price(symbol, price)

            # Check stops
            trigger = self.portfolio.check_stop_loss_take_profit(symbol, price)
            if trigger:
                logger.info("%s triggered for %s at $%.4f", trigger.upper(), symbol, price)
                result = self.close_position(symbol, market, reason=trigger)
                results.append(result)

        return results

    def sync_with_broker(self):
        """Sync portfolio with broker positions."""
        synced = 0

        # Sync Alpaca positions
        if self.alpaca:
            try:
                broker_positions = self.alpaca.get_positions()
                for bp in broker_positions:
                    pos = self.portfolio.get_position(bp["symbol"])
                    if pos:
                        self.portfolio.update_position_price(
                            bp["symbol"],
                            bp["current_price"]
                        )
                        synced += 1
            except Exception as e:
                logger.error("Failed to sync Alpaca positions: %s", e)

        # Sync Binance positions
        if self.binance:
            try:
                broker_positions = self.binance.get_positions()
                for bp in broker_positions:
                    pos = self.portfolio.get_position(bp["symbol"])
                    if pos:
                        self.portfolio.update_position_price(
                            bp["symbol"],
                            bp["current_price"]
                        )
                        synced += 1
            except Exception as e:
                logger.error("Failed to sync Binance positions: %s", e)

        logger.info("Synced %d positions with brokers", synced)
