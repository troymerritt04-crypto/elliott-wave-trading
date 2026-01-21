"""
Elliott Wave pattern detection.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import config
from analysis.fibonacci import FibonacciCalculator
from utils.logger import get_logger

logger = get_logger(__name__)


class WaveType(Enum):
    """Elliott Wave types."""
    WAVE_1 = 1
    WAVE_2 = 2
    WAVE_3 = 3
    WAVE_4 = 4
    WAVE_5 = 5
    WAVE_A = "A"
    WAVE_B = "B"
    WAVE_C = "C"


class TrendDirection(Enum):
    """Trend direction."""
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class SwingPoint:
    """A swing high or low point."""
    index: int
    timestamp: pd.Timestamp
    price: float
    is_high: bool


@dataclass
class WavePattern:
    """Detected Elliott Wave pattern."""
    direction: TrendDirection
    wave1_start: SwingPoint
    wave1_end: SwingPoint
    wave2_end: SwingPoint
    wave3_target_min: float
    wave3_target_typical: float
    wave3_target_extended: float
    wave2_retracement: float
    confidence: float
    current_wave: WaveType


class ElliottWaveDetector:
    """Detect Elliott Wave patterns in price data."""

    def __init__(
        self,
        zigzag_percent: float = None,
        min_wave1_percent: float = None
    ):
        """
        Initialize detector.

        Args:
            zigzag_percent: Minimum % move for zigzag swing
            min_wave1_percent: Minimum % move for valid Wave 1
        """
        self.zigzag_percent = zigzag_percent or config.ZIGZAG_PERCENT
        self.min_wave1_percent = min_wave1_percent or config.MIN_WAVE1_PERCENT
        self.fib = FibonacciCalculator()

    def find_swing_points(self, data: pd.DataFrame) -> List[SwingPoint]:
        """
        Find swing highs and lows using zigzag method.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            List of SwingPoint objects
        """
        highs = data["high"].values
        lows = data["low"].values
        timestamps = data.index

        swing_points = []
        last_swing_high = highs[0]
        last_swing_low = lows[0]
        last_swing_idx = 0
        last_swing_is_high = True
        trend_up = True

        threshold = self.zigzag_percent / 100

        for i in range(1, len(data)):
            if trend_up:
                if highs[i] > last_swing_high:
                    last_swing_high = highs[i]
                    last_swing_idx = i
                elif lows[i] < last_swing_high * (1 - threshold):
                    # Confirmed swing high
                    swing_points.append(SwingPoint(
                        index=last_swing_idx,
                        timestamp=timestamps[last_swing_idx],
                        price=last_swing_high,
                        is_high=True
                    ))
                    last_swing_low = lows[i]
                    last_swing_idx = i
                    trend_up = False
            else:
                if lows[i] < last_swing_low:
                    last_swing_low = lows[i]
                    last_swing_idx = i
                elif highs[i] > last_swing_low * (1 + threshold):
                    # Confirmed swing low
                    swing_points.append(SwingPoint(
                        index=last_swing_idx,
                        timestamp=timestamps[last_swing_idx],
                        price=last_swing_low,
                        is_high=False
                    ))
                    last_swing_high = highs[i]
                    last_swing_idx = i
                    trend_up = True

        # Add final swing if significant
        if len(swing_points) > 0:
            last_sp = swing_points[-1]
            if trend_up and last_swing_high > last_sp.price * (1 + threshold / 2):
                swing_points.append(SwingPoint(
                    index=last_swing_idx,
                    timestamp=timestamps[last_swing_idx],
                    price=last_swing_high,
                    is_high=True
                ))
            elif not trend_up and last_swing_low < last_sp.price * (1 - threshold / 2):
                swing_points.append(SwingPoint(
                    index=last_swing_idx,
                    timestamp=timestamps[last_swing_idx],
                    price=last_swing_low,
                    is_high=False
                ))

        return swing_points

    def detect_wave_pattern(
        self,
        data: pd.DataFrame,
        swing_points: Optional[List[SwingPoint]] = None
    ) -> Optional[WavePattern]:
        """
        Detect Elliott Wave pattern suitable for Wave 3 entry.

        Looks for completed Wave 1 and Wave 2, ready for Wave 3.

        Args:
            data: DataFrame with OHLCV data
            swing_points: Pre-calculated swing points (optional)

        Returns:
            WavePattern if detected, None otherwise
        """
        if swing_points is None:
            swing_points = self.find_swing_points(data)

        if len(swing_points) < 3:
            return None

        # Try to find bullish pattern (more recent swings first)
        bullish = self._detect_bullish_pattern(swing_points, data)
        if bullish:
            return bullish

        # Try to find bearish pattern
        bearish = self._detect_bearish_pattern(swing_points, data)
        if bearish:
            return bearish

        return None

    def _detect_bullish_pattern(
        self,
        swing_points: List[SwingPoint],
        data: pd.DataFrame
    ) -> Optional[WavePattern]:
        """Detect bullish Wave 1-2 completion."""

        # Look for pattern: Low -> High -> Low (Wave 1 up, Wave 2 retracement)
        for i in range(len(swing_points) - 3, -1, -1):
            if i + 2 >= len(swing_points):
                continue

            sp1 = swing_points[i]      # Potential Wave 1 start
            sp2 = swing_points[i + 1]  # Potential Wave 1 end
            sp3 = swing_points[i + 2]  # Potential Wave 2 end

            # Need Low -> High -> Low pattern
            if sp1.is_high or not sp2.is_high or sp3.is_high:
                continue

            # Wave 1 must be significant upward move
            wave1_pct = (sp2.price - sp1.price) / sp1.price * 100
            if wave1_pct < self.min_wave1_percent:
                continue

            # Validate Wave 2 retracement
            is_valid, retracement = self.fib.is_valid_wave2_retracement(
                sp1.price, sp2.price, sp3.price
            )

            if not is_valid:
                continue

            # Check current price is near Wave 2 end or starting Wave 3
            current_price = data["close"].iloc[-1]
            if current_price < sp3.price:
                continue  # Still in Wave 2 or invalid

            # Calculate Wave 3 targets
            targets = self.fib.get_wave3_targets(sp1.price, sp2.price, sp3.price)

            # Calculate confidence based on retracement quality
            confidence = self._calculate_confidence(retracement, wave1_pct)

            # Determine current wave
            extension = self.fib.get_extension_ratio(
                sp1.price, sp2.price, sp3.price, current_price
            )

            if extension < 0.1:
                current_wave = WaveType.WAVE_2  # Still near Wave 2 end
            elif extension < 1.618:
                current_wave = WaveType.WAVE_3  # In Wave 3
            else:
                current_wave = WaveType.WAVE_4  # Likely in Wave 4

            return WavePattern(
                direction=TrendDirection.BULLISH,
                wave1_start=sp1,
                wave1_end=sp2,
                wave2_end=sp3,
                wave3_target_min=targets.get("minimum", 0),
                wave3_target_typical=targets.get("typical", 0),
                wave3_target_extended=targets.get("extended", 0),
                wave2_retracement=retracement,
                confidence=confidence,
                current_wave=current_wave
            )

        return None

    def _detect_bearish_pattern(
        self,
        swing_points: List[SwingPoint],
        data: pd.DataFrame
    ) -> Optional[WavePattern]:
        """Detect bearish Wave 1-2 completion."""

        # Look for pattern: High -> Low -> High (Wave 1 down, Wave 2 retracement)
        for i in range(len(swing_points) - 3, -1, -1):
            if i + 2 >= len(swing_points):
                continue

            sp1 = swing_points[i]
            sp2 = swing_points[i + 1]
            sp3 = swing_points[i + 2]

            # Need High -> Low -> High pattern
            if not sp1.is_high or sp2.is_high or not sp3.is_high:
                continue

            # Wave 1 must be significant downward move
            wave1_pct = (sp1.price - sp2.price) / sp1.price * 100
            if wave1_pct < self.min_wave1_percent:
                continue

            # Validate Wave 2 retracement (bearish)
            is_valid, retracement = self.fib.is_valid_wave2_retracement(
                sp1.price, sp2.price, sp3.price
            )

            if not is_valid:
                continue

            # Check current price is near Wave 2 end or starting Wave 3
            current_price = data["close"].iloc[-1]
            if current_price > sp3.price:
                continue

            # Calculate Wave 3 targets (bearish - going down)
            targets = self.fib.get_wave3_targets(sp1.price, sp2.price, sp3.price)

            confidence = self._calculate_confidence(retracement, wave1_pct)

            extension = self.fib.get_extension_ratio(
                sp1.price, sp2.price, sp3.price, current_price
            )

            if extension < 0.1:
                current_wave = WaveType.WAVE_2
            elif extension < 1.618:
                current_wave = WaveType.WAVE_3
            else:
                current_wave = WaveType.WAVE_4

            return WavePattern(
                direction=TrendDirection.BEARISH,
                wave1_start=sp1,
                wave1_end=sp2,
                wave2_end=sp3,
                wave3_target_min=targets.get("minimum", 0),
                wave3_target_typical=targets.get("typical", 0),
                wave3_target_extended=targets.get("extended", 0),
                wave2_retracement=retracement,
                confidence=confidence,
                current_wave=current_wave
            )

        return None

    def _calculate_confidence(
        self,
        retracement: float,
        wave1_pct: float
    ) -> float:
        """
        Calculate pattern confidence score.

        Args:
            retracement: Wave 2 retracement ratio
            wave1_pct: Wave 1 percentage move

        Returns:
            Confidence score 0-1
        """
        confidence = 0.5

        # Ideal Wave 2 retracement is 50-61.8%
        if 0.5 <= retracement <= 0.618:
            confidence += 0.25
        elif 0.382 <= retracement <= 0.786:
            confidence += 0.1

        # Stronger Wave 1 = higher confidence
        if wave1_pct >= 10:
            confidence += 0.15
        elif wave1_pct >= 5:
            confidence += 0.1

        return min(confidence, 1.0)

    def is_wave3_entry(
        self,
        pattern: WavePattern,
        current_price: float,
        rsi: float,
        macd_bullish: bool,
        volume_above_avg: bool
    ) -> Tuple[bool, str]:
        """
        Check if conditions are right for Wave 3 entry.

        Args:
            pattern: Detected wave pattern
            current_price: Current price
            rsi: Current RSI value
            macd_bullish: True if MACD is bullish
            volume_above_avg: True if volume above average

        Returns:
            Tuple of (is_entry, reason)
        """
        if pattern.current_wave != WaveType.WAVE_3:
            return False, "Not in Wave 3"

        is_bullish = pattern.direction == TrendDirection.BULLISH

        # Check price broke Wave 1 high/low
        if is_bullish:
            if current_price <= pattern.wave1_end.price:
                return False, "Price below Wave 1 high"
            if rsi >= config.RSI_OVERBOUGHT:
                return False, f"RSI overbought ({rsi:.1f})"
        else:
            if current_price >= pattern.wave1_end.price:
                return False, "Price above Wave 1 low"
            if rsi <= config.RSI_OVERSOLD:
                return False, f"RSI oversold ({rsi:.1f})"

        # MACD confirmation
        if is_bullish and not macd_bullish:
            return False, "MACD not bullish"
        if not is_bullish and macd_bullish:
            return False, "MACD not bearish"

        # Volume confirmation preferred but not required
        reason = "Wave 3 entry"
        if volume_above_avg:
            reason += " (volume confirmed)"

        return True, reason

    def is_wave4_exit(
        self,
        pattern: WavePattern,
        current_price: float,
        rsi: float,
        rsi_divergence: int,
        price_below_ema9: bool
    ) -> Tuple[bool, str]:
        """
        Check if conditions indicate Wave 3 completion / Wave 4 start.

        Args:
            pattern: Detected wave pattern
            current_price: Current price
            rsi: Current RSI value
            rsi_divergence: 1 = bullish, -1 = bearish, 0 = none
            price_below_ema9: True if price below EMA 9

        Returns:
            Tuple of (is_exit, reason)
        """
        is_bullish = pattern.direction == TrendDirection.BULLISH

        # Check if reached minimum Wave 3 target
        extension = self.fib.get_extension_ratio(
            pattern.wave1_start.price,
            pattern.wave1_end.price,
            pattern.wave2_end.price,
            current_price
        )

        if extension < config.WAVE3_MIN_EXTENSION:
            return False, f"Extension only {extension:.2f}x (need {config.WAVE3_MIN_EXTENSION}x)"

        reasons = []

        # RSI overbought/oversold
        if is_bullish and rsi >= config.RSI_OVERBOUGHT:
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif not is_bullish and rsi <= config.RSI_OVERSOLD:
            reasons.append(f"RSI oversold ({rsi:.1f})")

        # RSI divergence
        if is_bullish and rsi_divergence == -1:
            reasons.append("Bearish RSI divergence")
        elif not is_bullish and rsi_divergence == 1:
            reasons.append("Bullish RSI divergence")

        # Price below/above EMA 9
        if is_bullish and price_below_ema9:
            reasons.append("Price below EMA 9")
        elif not is_bullish and not price_below_ema9:
            reasons.append("Price above EMA 9")

        # Need at least 2 signals to confirm exit
        if len(reasons) >= 2:
            return True, f"Wave 4 entry: {', '.join(reasons)}"
        elif len(reasons) == 1:
            return False, f"Partial signal: {reasons[0]}"

        return False, "No exit signals"
