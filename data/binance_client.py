"""
Binance API client for cryptocurrency trading.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class BinanceClient:
    """Client for interacting with Binance API for crypto trading."""

    def __init__(self):
        """Initialize Binance client with API credentials."""
        if config.BINANCE_TESTNET:
            self.client = Client(
                api_key=config.BINANCE_API_KEY,
                api_secret=config.BINANCE_SECRET_KEY,
                testnet=True
            )
            logger.info("Binance client initialized (testnet=True)")
        else:
            self.client = Client(
                api_key=config.BINANCE_API_KEY,
                api_secret=config.BINANCE_SECRET_KEY
            )
            logger.info("Binance client initialized (testnet=False)")

    def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        account = self.client.get_account()
        balances = {
            b["asset"]: {
                "free": float(b["free"]),
                "locked": float(b["locked"]),
                "total": float(b["free"]) + float(b["locked"])
            }
            for b in account["balances"]
            if float(b["free"]) > 0 or float(b["locked"]) > 0
        }
        return {
            "balances": balances,
            "can_trade": account["canTrade"],
        }

    def get_usdt_balance(self) -> float:
        """Get available USDT balance."""
        account = self.get_account()
        return account["balances"].get("USDT", {}).get("free", 0.0)

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions (non-USDT balances)."""
        account = self.get_account()
        positions = []

        for asset, balance in account["balances"].items():
            if asset in ["USDT", "BUSD", "USDC"]:
                continue
            if balance["total"] > 0:
                symbol = f"{asset}USDT"
                try:
                    price = self.get_latest_price(symbol)
                    positions.append({
                        "symbol": symbol,
                        "asset": asset,
                        "qty": balance["total"],
                        "current_price": price,
                        "market_value": balance["total"] * price,
                    })
                except Exception:
                    continue

        return positions

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        days: int = 100
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            days: Number of days of history

        Returns:
            DataFrame with OHLCV data
        """
        tf_map = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
        }

        interval = tf_map.get(timeframe, Client.KLINE_INTERVAL_1HOUR)
        start_time = datetime.now() - timedelta(days=days)
        start_str = start_time.strftime("%d %b %Y")

        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_str
        )

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df.set_index("timestamp")
        logger.debug("Fetched %d bars for %s", len(df), symbol)
        return df

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get trading rules for a symbol."""
        info = self.client.get_symbol_info(symbol)
        if not info:
            raise ValueError(f"Symbol {symbol} not found")

        filters = {f["filterType"]: f for f in info["filters"]}

        lot_size = filters.get("LOT_SIZE", {})
        price_filter = filters.get("PRICE_FILTER", {})
        min_notional = filters.get("NOTIONAL", {})

        return {
            "symbol": symbol,
            "base_asset": info["baseAsset"],
            "quote_asset": info["quoteAsset"],
            "min_qty": float(lot_size.get("minQty", 0)),
            "max_qty": float(lot_size.get("maxQty", 0)),
            "step_size": float(lot_size.get("stepSize", 0)),
            "min_price": float(price_filter.get("minPrice", 0)),
            "tick_size": float(price_filter.get("tickSize", 0)),
            "min_notional": float(min_notional.get("minNotional", 0)),
        }

    def round_quantity(self, symbol: str, qty: float) -> float:
        """Round quantity to valid step size for symbol."""
        info = self.get_symbol_info(symbol)
        step_size = info["step_size"]
        if step_size == 0:
            return qty
        precision = len(str(step_size).rstrip("0").split(".")[-1])
        return round(qty - (qty % step_size), precision)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to valid tick size for symbol."""
        info = self.get_symbol_info(symbol)
        tick_size = info["tick_size"]
        if tick_size == 0:
            return price
        precision = len(str(tick_size).rstrip("0").split(".")[-1])
        return round(price - (price % tick_size), precision)

    def place_market_order(
        self,
        symbol: str,
        qty: float,
        side: str
    ) -> Dict[str, Any]:
        """
        Place a market order.

        Args:
            symbol: Trading pair
            qty: Quantity to trade
            side: 'buy' or 'sell'

        Returns:
            Order details
        """
        qty = self.round_quantity(symbol, qty)
        order_side = SIDE_BUY if side.lower() == "buy" else SIDE_SELL

        order = self.client.create_order(
            symbol=symbol,
            side=order_side,
            type=ORDER_TYPE_MARKET,
            quantity=qty
        )

        logger.info("Market order placed: %s %s %s", side, qty, symbol)

        fills = order.get("fills", [])
        avg_price = 0.0
        if fills:
            total_qty = sum(float(f["qty"]) for f in fills)
            avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty

        return {
            "id": str(order["orderId"]),
            "symbol": order["symbol"],
            "qty": float(order["executedQty"]),
            "side": order["side"].lower(),
            "status": order["status"],
            "type": "market",
            "avg_price": avg_price,
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
            symbol: Trading pair
            qty: Quantity to trade
            side: 'buy' or 'sell'
            limit_price: Limit price

        Returns:
            Order details
        """
        qty = self.round_quantity(symbol, qty)
        limit_price = self.round_price(symbol, limit_price)
        order_side = SIDE_BUY if side.lower() == "buy" else SIDE_SELL

        order = self.client.create_order(
            symbol=symbol,
            side=order_side,
            type=ORDER_TYPE_LIMIT,
            timeInForce="GTC",
            quantity=qty,
            price=limit_price
        )

        logger.info("Limit order placed: %s %s %s @ %.8f", side, qty, symbol, limit_price)

        return {
            "id": str(order["orderId"]),
            "symbol": order["symbol"],
            "qty": float(order["origQty"]),
            "side": order["side"].lower(),
            "status": order["status"],
            "type": "limit",
            "limit_price": limit_price,
        }

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self.client.cancel_order(symbol=symbol, orderId=int(order_id))
            logger.info("Order cancelled: %s %s", symbol, order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Close entire position for a symbol (sell all holdings)."""
        positions = self.get_positions()
        position = next((p for p in positions if p["symbol"] == symbol), None)

        if not position:
            logger.warning("No position found for %s", symbol)
            return None

        return self.place_market_order(symbol, position["qty"], "sell")
