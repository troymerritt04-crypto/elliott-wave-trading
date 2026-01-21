"""
Technical indicators for market analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple

import config


class Indicators:
    """Technical indicator calculations."""

    @staticmethod
    def rsi(data: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            data: DataFrame with 'close' column
            period: RSI period (default from config)

        Returns:
            Series with RSI values
        """
        period = period or config.RSI_PERIOD
        close = data["close"]

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def macd(
        data: pd.DataFrame,
        fast: int = None,
        slow: int = None,
        signal: int = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            data: DataFrame with 'close' column
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        fast = fast or config.MACD_FAST
        slow = slow or config.MACD_SLOW
        signal = signal or config.MACD_SIGNAL

        close = data["close"]

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def ema(data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Args:
            data: DataFrame with 'close' column
            period: EMA period

        Returns:
            Series with EMA values
        """
        return data["close"].ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            data: DataFrame with 'close' column
            period: SMA period

        Returns:
            Series with SMA values
        """
        return data["close"].rolling(window=period).mean()

    @staticmethod
    def atr(data: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (default from config)

        Returns:
            Series with ATR values
        """
        period = period or config.ATR_PERIOD

        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def bollinger_bands(
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            data: DataFrame with 'close' column
            period: SMA period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        close = data["close"]

        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def volume_sma(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Volume Simple Moving Average.

        Args:
            data: DataFrame with 'volume' column
            period: SMA period

        Returns:
            Series with volume SMA values
        """
        return data["volume"].rolling(window=period).mean()

    @staticmethod
    def rsi_divergence(
        data: pd.DataFrame,
        rsi: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        Detect RSI divergence (bullish or bearish).

        Args:
            data: DataFrame with 'close' column
            rsi: RSI series
            lookback: Lookback period for divergence detection

        Returns:
            Series with divergence signals: 1 (bullish), -1 (bearish), 0 (none)
        """
        close = data["close"]
        divergence = pd.Series(0, index=data.index)

        for i in range(lookback, len(data)):
            window_close = close.iloc[i - lookback:i + 1]
            window_rsi = rsi.iloc[i - lookback:i + 1]

            # Find recent lows
            close_min_idx = window_close.idxmin()
            rsi_min_idx = window_rsi.idxmin()

            # Bullish divergence: price makes lower low, RSI makes higher low
            if (close.iloc[i] < window_close.iloc[0] and
                rsi.iloc[i] > window_rsi.iloc[0] and
                    close.iloc[i] <= close.loc[close_min_idx] * 1.02):
                divergence.iloc[i] = 1

            # Find recent highs
            close_max_idx = window_close.idxmax()
            rsi_max_idx = window_rsi.idxmax()

            # Bearish divergence: price makes higher high, RSI makes lower high
            if (close.iloc[i] > window_close.iloc[0] and
                rsi.iloc[i] < window_rsi.iloc[0] and
                    close.iloc[i] >= close.loc[close_max_idx] * 0.98):
                divergence.iloc[i] = -1

        return divergence

    @staticmethod
    def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add all standard indicators to a DataFrame.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        df = data.copy()

        # RSI
        df["rsi"] = Indicators.rsi(df)

        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = Indicators.macd(df)

        # EMAs
        df["ema_9"] = Indicators.ema(df, config.EMA_SHORT)
        df["ema_21"] = Indicators.ema(df, config.EMA_MID)
        df["ema_50"] = Indicators.ema(df, config.EMA_LONG)

        # ATR
        df["atr"] = Indicators.atr(df)

        # Bollinger Bands
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = Indicators.bollinger_bands(df)

        # Volume SMA
        df["volume_sma"] = Indicators.volume_sma(df)

        # RSI divergence
        df["rsi_divergence"] = Indicators.rsi_divergence(df, df["rsi"])

        return df
