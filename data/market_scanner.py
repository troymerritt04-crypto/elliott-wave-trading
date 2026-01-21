"""
Market scanner for finding tradeable assets.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from data.alpaca_client import AlpacaClient
from data.binance_client import BinanceClient
from utils.logger import get_logger

logger = get_logger(__name__)


class MarketScanner:
    """Scans markets for tradeable assets based on criteria."""

    def __init__(
        self,
        alpaca_client: Optional[AlpacaClient] = None,
        binance_client: Optional[BinanceClient] = None
    ):
        """
        Initialize market scanner.

        Args:
            alpaca_client: Alpaca client instance
            binance_client: Binance client instance
        """
        self.alpaca = alpaca_client
        self.binance = binance_client

    def scan_stocks(
        self,
        symbols: Optional[List[str]] = None,
        min_volume: float = 1_000_000,
        min_price: float = 5.0,
        max_price: float = 500.0
    ) -> List[Dict[str, Any]]:
        """
        Scan stock symbols for tradeable candidates.

        Args:
            symbols: List of symbols to scan (defaults to watchlist)
            min_volume: Minimum average daily volume
            min_price: Minimum stock price
            max_price: Maximum stock price

        Returns:
            List of tradeable stock candidates with metadata
        """
        if not self.alpaca:
            logger.warning("Alpaca client not initialized")
            return []

        symbols = symbols or config.STOCK_WATCHLIST
        candidates = []

        def scan_symbol(symbol: str) -> Optional[Dict[str, Any]]:
            try:
                bars = self.alpaca.get_historical_bars(symbol, "1d", days=30)
                if bars.empty:
                    return None

                latest_price = bars["close"].iloc[-1]
                avg_volume = bars["volume"].mean()

                if latest_price < min_price or latest_price > max_price:
                    return None
                if avg_volume < min_volume:
                    return None

                volatility = bars["close"].pct_change().std() * 100

                return {
                    "symbol": symbol,
                    "market": "stock",
                    "price": latest_price,
                    "avg_volume": avg_volume,
                    "volatility": volatility,
                }
            except Exception as e:
                logger.debug("Error scanning %s: %s", symbol, e)
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(scan_symbol, s): s for s in symbols}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    candidates.append(result)

        candidates.sort(key=lambda x: x["volatility"], reverse=True)
        logger.info("Found %d stock candidates from %d symbols", len(candidates), len(symbols))
        return candidates

    def scan_crypto(
        self,
        symbols: Optional[List[str]] = None,
        min_volume_usdt: float = 10_000_000,
        min_price: float = 0.0001
    ) -> List[Dict[str, Any]]:
        """
        Scan crypto symbols for tradeable candidates.

        Args:
            symbols: List of symbols to scan (defaults to watchlist)
            min_volume_usdt: Minimum 24h volume in USDT
            min_price: Minimum price

        Returns:
            List of tradeable crypto candidates with metadata
        """
        if not self.binance:
            logger.warning("Binance client not initialized")
            return []

        symbols = symbols or config.CRYPTO_WATCHLIST
        candidates = []

        def scan_symbol(symbol: str) -> Optional[Dict[str, Any]]:
            try:
                bars = self.binance.get_historical_bars(symbol, "1d", days=30)
                if bars.empty:
                    return None

                latest_price = bars["close"].iloc[-1]
                if latest_price < min_price:
                    return None

                avg_volume = bars["volume"].mean()
                volume_usdt = avg_volume * latest_price
                if volume_usdt < min_volume_usdt:
                    return None

                volatility = bars["close"].pct_change().std() * 100

                return {
                    "symbol": symbol,
                    "market": "crypto",
                    "price": latest_price,
                    "avg_volume": avg_volume,
                    "volume_usdt": volume_usdt,
                    "volatility": volatility,
                }
            except Exception as e:
                logger.debug("Error scanning %s: %s", symbol, e)
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(scan_symbol, s): s for s in symbols}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    candidates.append(result)

        candidates.sort(key=lambda x: x["volatility"], reverse=True)
        logger.info("Found %d crypto candidates from %d symbols", len(candidates), len(symbols))
        return candidates

    def scan_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan both stock and crypto markets.

        Returns:
            Dictionary with 'stocks' and 'crypto' candidate lists
        """
        return {
            "stocks": self.scan_stocks(),
            "crypto": self.scan_crypto(),
        }

    def get_market_data(
        self,
        symbol: str,
        market: str,
        timeframe: str = "1h",
        days: int = 100
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol.

        Args:
            symbol: Asset symbol
            market: 'stock' or 'crypto'
            timeframe: Candle timeframe
            days: Number of days of history

        Returns:
            DataFrame with OHLCV data
        """
        if market == "stock" and self.alpaca:
            return self.alpaca.get_historical_bars(symbol, timeframe, days)
        elif market == "crypto" and self.binance:
            return self.binance.get_historical_bars(symbol, timeframe, days)
        else:
            raise ValueError(f"Cannot fetch data for {symbol} ({market})")

    def get_latest_price(self, symbol: str, market: str) -> float:
        """
        Get latest price for a symbol.

        Args:
            symbol: Asset symbol
            market: 'stock' or 'crypto'

        Returns:
            Latest price
        """
        if market == "stock" and self.alpaca:
            return self.alpaca.get_latest_price(symbol)
        elif market == "crypto" and self.binance:
            return self.binance.get_latest_price(symbol)
        else:
            raise ValueError(f"Cannot fetch price for {symbol} ({market})")
