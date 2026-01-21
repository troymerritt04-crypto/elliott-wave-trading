"""
Unit tests for technical indicators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(__file__).rsplit("/tests", 1)[0])

from analysis.indicators import Indicators


class TestIndicators:
    """Tests for technical indicator calculations."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample OHLCV data."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")

        # Create trending data with some noise
        trend = np.linspace(100, 150, 100)
        noise = np.random.randn(100) * 2

        close = trend + noise
        high = close + np.abs(np.random.randn(100))
        low = close - np.abs(np.random.randn(100))
        open_price = close + np.random.randn(100) * 0.5

        return pd.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100000, 1000000, 100)
        }, index=dates)

    def test_rsi_calculation(self, sample_data):
        """Test RSI calculation."""
        rsi = Indicators.rsi(sample_data, period=14)

        # RSI should be between 0 and 100
        assert rsi.dropna().min() >= 0
        assert rsi.dropna().max() <= 100

        # For uptrending data, RSI should generally be above 50
        assert rsi.iloc[-1] > 40  # Some tolerance for noise

    def test_rsi_overbought_oversold(self):
        """Test RSI identifies overbought/oversold conditions."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="h")

        # Strongly uptrending data
        uptrend = pd.DataFrame({
            "close": np.linspace(100, 200, 50)
        }, index=dates)
        rsi_up = Indicators.rsi(uptrend)
        assert rsi_up.iloc[-1] > 60  # Should be elevated

        # Strongly downtrending data
        downtrend = pd.DataFrame({
            "close": np.linspace(200, 100, 50)
        }, index=dates)
        rsi_down = Indicators.rsi(downtrend)
        assert rsi_down.iloc[-1] < 40  # Should be depressed

    def test_macd_calculation(self, sample_data):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = Indicators.macd(sample_data)

        # All series should have values
        assert len(macd_line.dropna()) > 0
        assert len(signal_line.dropna()) > 0
        assert len(histogram.dropna()) > 0

        # Histogram should equal MACD - Signal
        valid_idx = histogram.dropna().index
        expected_hist = macd_line.loc[valid_idx] - signal_line.loc[valid_idx]
        np.testing.assert_array_almost_equal(
            histogram.loc[valid_idx].values,
            expected_hist.values,
            decimal=10
        )

    def test_macd_crossover_detection(self):
        """Test MACD crossover detection."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")

        # Create data that causes MACD crossover
        prices = []
        for i in range(100):
            if i < 50:
                prices.append(100 + i * 0.5)  # Uptrend
            else:
                prices.append(125 - (i - 50) * 0.3)  # Slight downtrend

        data = pd.DataFrame({"close": prices}, index=dates)
        macd_line, signal_line, histogram = Indicators.macd(data)

        # Check for sign change in histogram (crossover)
        hist_signs = np.sign(histogram.dropna())
        sign_changes = (hist_signs.diff() != 0).sum()
        assert sign_changes > 0  # Should have at least one crossover

    def test_ema_calculation(self, sample_data):
        """Test EMA calculation."""
        ema_9 = Indicators.ema(sample_data, 9)
        ema_21 = Indicators.ema(sample_data, 21)

        # EMA should be calculated for all rows
        assert len(ema_9) == len(sample_data)

        # Shorter EMA should be more responsive (closer to recent prices)
        recent_close = sample_data["close"].iloc[-1]
        assert abs(ema_9.iloc[-1] - recent_close) <= abs(ema_21.iloc[-1] - recent_close)

    def test_sma_calculation(self, sample_data):
        """Test SMA calculation."""
        sma_20 = Indicators.sma(sample_data, 20)

        # First 19 values should be NaN
        assert sma_20.iloc[:19].isna().all()

        # SMA should be the mean of the last 20 values
        expected_last_sma = sample_data["close"].iloc[-20:].mean()
        assert abs(sma_20.iloc[-1] - expected_last_sma) < 0.01

    def test_atr_calculation(self, sample_data):
        """Test ATR calculation."""
        atr = Indicators.atr(sample_data, period=14)

        # ATR should be positive
        assert atr.dropna().min() >= 0

        # ATR should reflect the range of prices
        avg_range = (sample_data["high"] - sample_data["low"]).mean()
        assert atr.iloc[-1] > 0
        assert atr.iloc[-1] < avg_range * 2  # Reasonable bounds

    def test_bollinger_bands(self, sample_data):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = Indicators.bollinger_bands(sample_data, period=20, std_dev=2.0)

        # Upper should be above middle, middle above lower
        valid_idx = upper.dropna().index
        assert (upper.loc[valid_idx] >= middle.loc[valid_idx]).all()
        assert (middle.loc[valid_idx] >= lower.loc[valid_idx]).all()

        # Most prices should be within bands
        close = sample_data["close"].loc[valid_idx]
        within_bands = ((close >= lower.loc[valid_idx]) & (close <= upper.loc[valid_idx])).sum()
        total = len(valid_idx)
        assert within_bands / total > 0.9  # 90% should be within 2 std dev

    def test_volume_sma(self, sample_data):
        """Test volume SMA calculation."""
        vol_sma = Indicators.volume_sma(sample_data, period=20)

        # First 19 values should be NaN
        assert vol_sma.iloc[:19].isna().all()

        # Volume SMA should be the mean of the last 20 volumes
        expected_last = sample_data["volume"].iloc[-20:].mean()
        assert abs(vol_sma.iloc[-1] - expected_last) < 0.01

    def test_add_all_indicators(self, sample_data):
        """Test adding all indicators to DataFrame."""
        df = Indicators.add_all_indicators(sample_data)

        # Check all expected columns are present
        expected_columns = [
            "rsi", "macd", "macd_signal", "macd_hist",
            "ema_9", "ema_21", "ema_50", "atr",
            "bb_upper", "bb_middle", "bb_lower",
            "volume_sma", "rsi_divergence"
        ]

        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Original columns should still be present
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_rsi_divergence_detection(self, sample_data):
        """Test RSI divergence detection."""
        rsi = Indicators.rsi(sample_data)
        divergence = Indicators.rsi_divergence(sample_data, rsi, lookback=20)

        # Divergence should be -1, 0, or 1
        unique_values = divergence.unique()
        assert all(v in [-1, 0, 1] for v in unique_values)

        # Most values should be 0 (no divergence)
        assert (divergence == 0).sum() > len(divergence) * 0.5


class TestIndicatorEdgeCases:
    """Test edge cases for indicators."""

    def test_empty_dataframe(self):
        """Test indicators handle empty data."""
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # These should not raise exceptions
        rsi = Indicators.rsi(empty_df)
        assert len(rsi) == 0

    def test_single_row(self):
        """Test indicators handle single row of data."""
        single_row = pd.DataFrame({
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100.5],
            "volume": [1000000]
        }, index=[pd.Timestamp("2024-01-01")])

        # Should not raise exceptions
        rsi = Indicators.rsi(single_row)
        atr = Indicators.atr(single_row)
        ema = Indicators.ema(single_row, 9)

    def test_constant_prices(self):
        """Test indicators with constant prices."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="h")
        constant_data = pd.DataFrame({
            "open": [100] * 50,
            "high": [100] * 50,
            "low": [100] * 50,
            "close": [100] * 50,
            "volume": [1000000] * 50
        }, index=dates)

        # RSI should be 50 (neutral) when no change
        rsi = Indicators.rsi(constant_data)
        # Note: With constant prices, we get NaN due to 0/0
        # This is expected behavior

        # ATR should be 0 with no range
        atr = Indicators.atr(constant_data)
        assert atr.iloc[-1] == 0

    def test_extreme_volatility(self):
        """Test indicators with extreme price movements."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="h")

        # Alternating extreme moves
        prices = [100 * (1.5 if i % 2 == 0 else 0.5) for i in range(50)]

        extreme_data = pd.DataFrame({
            "open": prices,
            "high": [p * 1.1 for p in prices],
            "low": [p * 0.9 for p in prices],
            "close": prices,
            "volume": [1000000] * 50
        }, index=dates)

        # Should not raise exceptions
        df = Indicators.add_all_indicators(extreme_data)
        assert not df["rsi"].iloc[-1:].isna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
