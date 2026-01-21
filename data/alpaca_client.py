"""
Alpaca API client for stock trading.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class AlpacaClient:
    """Client for interacting with Alpaca API for stocks trading."""

    def __init__(self):
        """Initialize Alpaca client with API credentials."""
        self.trading_client = TradingClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            paper=config.PAPER_TRADING
        )
        self.data_client = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY
        )
        logger.info("Alpaca client initialized (paper=%s)", config.PAPER_TRADING)

    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "day_trade_count": account.daytrade_count,
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        positions = self.trading_client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": "long" if float(p.qty) > 0 else "short",
                "entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
            for p in positions
        ]

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol."""
        try:
            p = self.trading_client.get_open_position(symbol)
            return {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": "long" if float(p.qty) > 0 else "short",
                "entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
        except Exception:
            return None

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 100
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 1d)
            days: Number of days of history

        Returns:
            DataFrame with OHLCV data
        """
        tf_map = {
            "1m": TimeFrame(1, TimeFrameUnit.Minute),
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "1h": TimeFrame(1, TimeFrameUnit.Hour),
            "4h": TimeFrame(4, TimeFrameUnit.Hour),
            "1d": TimeFrame(1, TimeFrameUnit.Day),
        }

        tf = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Hour))
        end = datetime.now()
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end
        )

        bars = self.data_client.get_stock_bars(request)
        df = bars.df

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")

        df = df.reset_index()
        df.columns = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        logger.debug("Fetched %d bars for %s", len(df), symbol)
        return df

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        bars = self.get_historical_bars(symbol, "1m", days=1)
        if bars.empty:
            raise ValueError(f"No price data for {symbol}")
        return float(bars["close"].iloc[-1])

    def place_market_order(
        self,
        symbol: str,
        qty: float,
        side: str
    ) -> Dict[str, Any]:
        """
        Place a market order.

        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'

        Returns:
            Order details
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY
        )

        order = self.trading_client.submit_order(request)
        logger.info("Market order placed: %s %s %s", side, qty, symbol)

        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side.value,
            "status": order.status.value,
            "type": "market",
        }

    def place_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float
    ) -> Dict[str, Any]:
        """
        Place a limit order.

        Args:
            symbol: Stock symbol
            qty: Quantity to trade
            side: 'buy' or 'sell'
            limit_price: Limit price

        Returns:
            Order details
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price
        )

        order = self.trading_client.submit_order(request)
        logger.info("Limit order placed: %s %s %s @ %.2f", side, qty, symbol, limit_price)

        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side.value,
            "status": order.status.value,
            "type": "limit",
            "limit_price": limit_price,
        }

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info("Order cancelled: %s", order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Close entire position for a symbol."""
        try:
            order = self.trading_client.close_position(symbol)
            logger.info("Position closed: %s", symbol)
            return {
                "id": str(order.id),
                "symbol": order.symbol,
                "status": order.status.value,
            }
        except Exception as e:
            logger.error("Failed to close position %s: %s", symbol, e)
            return None

    def is_market_open(self) -> bool:
        """Check if the stock market is currently open."""
        clock = self.trading_client.get_clock()
        return clock.is_open
