"""
Fibonacci retracement and extension calculations.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FibLevel:
    """Fibonacci level with price and ratio."""
    ratio: float
    price: float
    label: str


class FibonacciCalculator:
    """Calculate Fibonacci retracements and extensions."""

    # Standard Fibonacci retracement levels
    RETRACEMENT_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    # Standard Fibonacci extension levels
    EXTENSION_LEVELS = [1.0, 1.272, 1.414, 1.618, 2.0, 2.272, 2.618, 3.0, 3.618, 4.236]

    @staticmethod
    def retracements(
        swing_high: float,
        swing_low: float,
        is_uptrend: bool = True
    ) -> List[FibLevel]:
        """
        Calculate Fibonacci retracement levels.

        Args:
            swing_high: The swing high price
            swing_low: The swing low price
            is_uptrend: True if measuring retracement in uptrend (from high)

        Returns:
            List of FibLevel objects with price levels
        """
        diff = swing_high - swing_low
        levels = []

        for ratio in FibonacciCalculator.RETRACEMENT_LEVELS:
            if is_uptrend:
                # In uptrend, retracement goes down from high
                price = swing_high - (diff * ratio)
            else:
                # In downtrend, retracement goes up from low
                price = swing_low + (diff * ratio)

            label = f"{ratio * 100:.1f}%"
            levels.append(FibLevel(ratio=ratio, price=price, label=label))

        return levels

    @staticmethod
    def extensions(
        wave_start: float,
        wave_end: float,
        correction_end: float
    ) -> List[FibLevel]:
        """
        Calculate Fibonacci extension levels.

        For Elliott Wave, this calculates potential Wave 3 targets based on Wave 1.

        Args:
            wave_start: Start price of Wave 1
            wave_end: End price of Wave 1
            correction_end: End price of Wave 2 (correction)

        Returns:
            List of FibLevel objects with price levels
        """
        wave_size = abs(wave_end - wave_start)
        is_bullish = wave_end > wave_start
        levels = []

        for ratio in FibonacciCalculator.EXTENSION_LEVELS:
            if is_bullish:
                price = correction_end + (wave_size * ratio)
            else:
                price = correction_end - (wave_size * ratio)

            label = f"{ratio * 100:.1f}%"
            levels.append(FibLevel(ratio=ratio, price=price, label=label))

        return levels

    @staticmethod
    def get_retracement_ratio(
        swing_high: float,
        swing_low: float,
        current_price: float
    ) -> float:
        """
        Calculate how much price has retraced from a swing.

        Args:
            swing_high: The swing high price
            swing_low: The swing low price
            current_price: Current price to measure

        Returns:
            Retracement ratio (0.0 to 1.0+)
        """
        diff = swing_high - swing_low
        if diff == 0:
            return 0.0

        # Measure retracement from high
        retracement = (swing_high - current_price) / diff
        return retracement

    @staticmethod
    def get_extension_ratio(
        wave_start: float,
        wave_end: float,
        correction_end: float,
        current_price: float
    ) -> float:
        """
        Calculate current extension ratio.

        Args:
            wave_start: Start price of Wave 1
            wave_end: End price of Wave 1
            correction_end: End price of Wave 2
            current_price: Current price

        Returns:
            Extension ratio relative to Wave 1 size
        """
        wave_size = abs(wave_end - wave_start)
        if wave_size == 0:
            return 0.0

        is_bullish = wave_end > wave_start

        if is_bullish:
            extension = (current_price - correction_end) / wave_size
        else:
            extension = (correction_end - current_price) / wave_size

        return extension

    @staticmethod
    def is_valid_wave2_retracement(
        wave1_start: float,
        wave1_end: float,
        wave2_end: float
    ) -> Tuple[bool, float]:
        """
        Check if Wave 2 retracement is valid per Elliott Wave rules.

        Wave 2 must:
        - Retrace at least 38.2% of Wave 1
        - Retrace at most 78.6% of Wave 1 (ideally, rarely 99%)
        - Never exceed Wave 1 start

        Args:
            wave1_start: Start price of Wave 1
            wave1_end: End price of Wave 1
            wave2_end: End price of Wave 2

        Returns:
            Tuple of (is_valid, retracement_ratio)
        """
        is_bullish = wave1_end > wave1_start
        wave1_size = abs(wave1_end - wave1_start)

        if wave1_size == 0:
            return False, 0.0

        if is_bullish:
            retracement = (wave1_end - wave2_end) / wave1_size
            # Wave 2 cannot go below Wave 1 start
            if wave2_end < wave1_start:
                return False, retracement
        else:
            retracement = (wave2_end - wave1_end) / wave1_size
            # Wave 2 cannot go above Wave 1 start
            if wave2_end > wave1_start:
                return False, retracement

        is_valid = 0.382 <= retracement <= 0.786
        return is_valid, retracement

    @staticmethod
    def get_wave3_targets(
        wave1_start: float,
        wave1_end: float,
        wave2_end: float
    ) -> Dict[str, float]:
        """
        Calculate Wave 3 price targets.

        Args:
            wave1_start: Start price of Wave 1
            wave1_end: End price of Wave 1
            wave2_end: End price of Wave 2

        Returns:
            Dictionary with target names and prices
        """
        extensions = FibonacciCalculator.extensions(
            wave1_start, wave1_end, wave2_end
        )

        targets = {}
        key_ratios = {1.618: "minimum", 2.0: "typical", 2.618: "extended"}

        for level in extensions:
            if level.ratio in key_ratios:
                targets[key_ratios[level.ratio]] = level.price

        return targets

    @staticmethod
    def get_wave4_retracement_levels(
        wave3_start: float,
        wave3_end: float
    ) -> Dict[str, float]:
        """
        Calculate potential Wave 4 retracement levels.

        Wave 4 typically retraces 38.2% of Wave 3 and must not overlap
        Wave 1 territory.

        Args:
            wave3_start: Start price of Wave 3 (Wave 2 end)
            wave3_end: End price of Wave 3

        Returns:
            Dictionary with retracement level names and prices
        """
        levels = FibonacciCalculator.retracements(
            swing_high=max(wave3_start, wave3_end),
            swing_low=min(wave3_start, wave3_end),
            is_uptrend=wave3_end > wave3_start
        )

        result = {}
        key_ratios = {0.236: "shallow", 0.382: "typical", 0.5: "deep"}

        for level in levels:
            if level.ratio in key_ratios:
                result[key_ratios[level.ratio]] = level.price

        return result
