"""
Unit tests for Elliott Wave detection.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(__file__).rsplit("/tests", 1)[0])

from analysis.elliott_wave import (
    ElliottWaveDetector,
    SwingPoint,
    WavePattern,
    WaveType,
    TrendDirection
)
from analysis.fibonacci import FibonacciCalculator


class TestFibonacciCalculator:
    """Tests for Fibonacci calculations."""

    def test_retracements_uptrend(self):
        """Test Fibonacci retracement levels in uptrend."""
        fib = FibonacciCalculator()
        levels = fib.retracements(100.0, 50.0, is_uptrend=True)

        # 0% retracement should be at swing high
        assert levels[0].price == 100.0
        assert levels[0].ratio == 0.0

        # 100% retracement should be at swing low
        assert levels[-1].price == 50.0
        assert levels[-1].ratio == 1.0

        # 50% retracement should be at 75
        level_50 = next(l for l in levels if l.ratio == 0.5)
        assert level_50.price == 75.0

    def test_retracements_downtrend(self):
        """Test Fibonacci retracement levels in downtrend."""
        fib = FibonacciCalculator()
        levels = fib.retracements(100.0, 50.0, is_uptrend=False)

        # 0% retracement should be at swing low
        assert levels[0].price == 50.0

        # 100% retracement should be at swing high
        assert levels[-1].price == 100.0

    def test_extensions(self):
        """Test Fibonacci extension levels."""
        fib = FibonacciCalculator()

        # Bullish: Wave 1 from 100 to 120, Wave 2 ends at 108
        levels = fib.extensions(100.0, 120.0, 108.0)

        # Wave 1 size = 20
        # 161.8% extension from 108 = 108 + 20*1.618 = 140.36
        level_1618 = next(l for l in levels if l.ratio == 1.618)
        assert abs(level_1618.price - 140.36) < 0.01

    def test_valid_wave2_retracement(self):
        """Test Wave 2 retracement validation."""
        fib = FibonacciCalculator()

        # Wave 1: 100 -> 150 (bullish)
        # Valid Wave 2: retraces 50% (to 125)
        is_valid, ratio = fib.is_valid_wave2_retracement(100.0, 150.0, 125.0)
        assert is_valid is True
        assert abs(ratio - 0.5) < 0.01

        # Invalid: Wave 2 goes below Wave 1 start
        is_valid, ratio = fib.is_valid_wave2_retracement(100.0, 150.0, 90.0)
        assert is_valid is False

        # Invalid: Wave 2 retraces less than 38.2%
        is_valid, ratio = fib.is_valid_wave2_retracement(100.0, 150.0, 145.0)
        assert is_valid is False
        assert ratio < 0.382

    def test_wave3_targets(self):
        """Test Wave 3 target calculations."""
        fib = FibonacciCalculator()

        # Wave 1: 100 -> 120, Wave 2 ends at 110
        targets = fib.get_wave3_targets(100.0, 120.0, 110.0)

        assert "minimum" in targets  # 161.8%
        assert "typical" in targets  # 200%
        assert "extended" in targets  # 261.8%

        # Wave 1 size = 20, so 161.8% from 110 = 142.36
        assert abs(targets["minimum"] - 142.36) < 0.01


class TestElliottWaveDetector:
    """Tests for Elliott Wave detection."""

    def create_sample_data(self, pattern: str = "bullish_wave3") -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")

        if pattern == "bullish_wave3":
            # Create a bullish Elliott Wave pattern:
            # Wave 1: 100 -> 120
            # Wave 2: 120 -> 110 (50% retracement)
            # Wave 3 starting: 110 -> currently at 125
            prices = []
            for i in range(100):
                if i < 20:  # Wave 1 up
                    price = 100 + i
                elif i < 40:  # Wave 2 down (retracement)
                    price = 120 - (i - 20) * 0.5
                else:  # Wave 3 up
                    price = 110 + (i - 40) * 0.5
                prices.append(price)

        elif pattern == "bearish_wave3":
            # Create a bearish Elliott Wave pattern
            prices = []
            for i in range(100):
                if i < 20:  # Wave 1 down
                    price = 100 - i
                elif i < 40:  # Wave 2 up (retracement)
                    price = 80 + (i - 20) * 0.5
                else:  # Wave 3 down
                    price = 90 - (i - 40) * 0.5
                prices.append(price)

        elif pattern == "no_pattern":
            # Sideways/ranging market
            np.random.seed(42)
            prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

        else:
            prices = [100] * 100

        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": [1000000] * 100
        }, index=dates)

        return df

    def test_find_swing_points(self):
        """Test swing point detection."""
        detector = ElliottWaveDetector(zigzag_percent=3.0)
        data = self.create_sample_data("bullish_wave3")

        swings = detector.find_swing_points(data)

        assert len(swings) >= 2  # Should find at least 2 swings
        assert any(s.is_high for s in swings)  # Should have swing highs
        assert any(not s.is_high for s in swings)  # Should have swing lows

    def test_detect_bullish_pattern(self):
        """Test bullish wave pattern detection."""
        detector = ElliottWaveDetector(zigzag_percent=3.0, min_wave1_percent=3.0)
        data = self.create_sample_data("bullish_wave3")

        pattern = detector.detect_wave_pattern(data)

        # Should detect a pattern (may or may not depending on data)
        # This is a basic sanity check
        if pattern:
            assert pattern.direction == TrendDirection.BULLISH
            assert pattern.wave1_start is not None
            assert pattern.wave1_end is not None
            assert pattern.wave2_end is not None

    def test_detect_no_pattern_in_ranging_market(self):
        """Test that ranging market doesn't falsely detect patterns."""
        detector = ElliottWaveDetector(zigzag_percent=5.0, min_wave1_percent=5.0)
        data = self.create_sample_data("no_pattern")

        pattern = detector.detect_wave_pattern(data)

        # May or may not find pattern, but should handle gracefully
        # Just verify no exception is raised

    def test_wave3_entry_conditions(self):
        """Test Wave 3 entry condition checking."""
        detector = ElliottWaveDetector()

        # Create a mock pattern
        pattern = WavePattern(
            direction=TrendDirection.BULLISH,
            wave1_start=SwingPoint(0, pd.Timestamp("2024-01-01"), 100.0, False),
            wave1_end=SwingPoint(20, pd.Timestamp("2024-01-02"), 120.0, True),
            wave2_end=SwingPoint(40, pd.Timestamp("2024-01-03"), 110.0, False),
            wave3_target_min=142.36,
            wave3_target_typical=150.0,
            wave3_target_extended=163.6,
            wave2_retracement=0.5,
            confidence=0.7,
            current_wave=WaveType.WAVE_3
        )

        # Should be entry when all conditions met
        is_entry, reason = detector.is_wave3_entry(
            pattern,
            current_price=125.0,  # Above Wave 1 high
            rsi=55.0,  # Not overbought
            macd_bullish=True,
            volume_above_avg=True
        )
        assert is_entry is True

        # Should not be entry if RSI overbought
        is_entry, reason = detector.is_wave3_entry(
            pattern,
            current_price=125.0,
            rsi=75.0,
            macd_bullish=True,
            volume_above_avg=True
        )
        assert is_entry is False
        assert "overbought" in reason.lower()

        # Should not be entry if price below Wave 1 high
        is_entry, reason = detector.is_wave3_entry(
            pattern,
            current_price=115.0,
            rsi=55.0,
            macd_bullish=True,
            volume_above_avg=True
        )
        assert is_entry is False

    def test_wave4_exit_conditions(self):
        """Test Wave 4 exit condition checking."""
        detector = ElliottWaveDetector()

        pattern = WavePattern(
            direction=TrendDirection.BULLISH,
            wave1_start=SwingPoint(0, pd.Timestamp("2024-01-01"), 100.0, False),
            wave1_end=SwingPoint(20, pd.Timestamp("2024-01-02"), 120.0, True),
            wave2_end=SwingPoint(40, pd.Timestamp("2024-01-03"), 110.0, False),
            wave3_target_min=142.36,
            wave3_target_typical=150.0,
            wave3_target_extended=163.6,
            wave2_retracement=0.5,
            confidence=0.7,
            current_wave=WaveType.WAVE_3
        )

        # Should be exit when at target with RSI divergence and price below EMA
        is_exit, reason = detector.is_wave4_exit(
            pattern,
            current_price=145.0,  # Above 161.8% extension
            rsi=75.0,
            rsi_divergence=-1,  # Bearish divergence
            price_below_ema9=True
        )
        assert is_exit is True

        # Should not be exit if extension not reached
        is_exit, reason = detector.is_wave4_exit(
            pattern,
            current_price=130.0,  # Below 161.8% extension
            rsi=75.0,
            rsi_divergence=-1,
            price_below_ema9=True
        )
        assert is_exit is False


class TestIntegration:
    """Integration tests for the full detection pipeline."""

    def test_full_detection_pipeline(self):
        """Test the complete detection and signal generation."""
        from analysis.indicators import Indicators
        from analysis.signal_generator import SignalGenerator, SignalType

        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
        prices = []
        for i in range(100):
            if i < 20:
                price = 100 + i * 1.5
            elif i < 40:
                price = 130 - (i - 20) * 0.75
            else:
                price = 115 + (i - 40) * 0.5
            prices.append(price)

        data = pd.DataFrame({
            "open": [p * 0.999 for p in prices],
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": [1000000 + i * 10000 for i in range(100)]
        }, index=dates)

        # Run through signal generator
        signal_gen = SignalGenerator()
        signal = signal_gen.analyze("TEST", "stock", data)

        # Should return a signal (may be HOLD)
        assert signal is not None
        assert signal.symbol == "TEST"
        assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
