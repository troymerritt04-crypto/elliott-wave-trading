"""
Signal generator combining Elliott Wave and technical indicators.
"""

import pandas as pd
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import config
from analysis.indicators import Indicators
from analysis.elliott_wave import ElliottWaveDetector, WavePattern, WaveType, TrendDirection
from utils.logger import get_logger

logger = get_logger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    signal_type: SignalType
    symbol: str
    market: str
    price: float
    confidence: float
    reason: str
    wave_pattern: Optional[WavePattern]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    indicators: Dict[str, float]


class SignalGenerator:
    """Generate trading signals from analysis."""

    def __init__(self):
        """Initialize signal generator."""
        self.wave_detector = ElliottWaveDetector()
        self.indicators = Indicators()

    def analyze(
        self,
        symbol: str,
        market: str,
        data: pd.DataFrame
    ) -> TradingSignal:
        """
        Analyze price data and generate trading signal.

        Args:
            symbol: Asset symbol
            market: 'stock' or 'crypto'
            data: DataFrame with OHLCV data

        Returns:
            TradingSignal with recommendation
        """
        if len(data) < 50:
            return self._hold_signal(symbol, market, data, "Insufficient data")

        # Add indicators
        df = Indicators.add_all_indicators(data)
        current = df.iloc[-1]
        current_price = current["close"]

        # Get indicator values
        rsi = current["rsi"]
        macd = current["macd"]
        macd_signal = current["macd_signal"]
        macd_hist = current["macd_hist"]
        ema_9 = current["ema_9"]
        ema_21 = current["ema_21"]
        ema_50 = current["ema_50"]
        atr = current["atr"]
        volume = current["volume"]
        volume_sma = current["volume_sma"]
        rsi_div = current["rsi_divergence"]

        # Check volume and trend confirmation for risk manager
        volume_confirmed = volume > volume_sma * 1.1  # Volume 10% above average
        trend_aligned_bullish = current_price > ema_50  # Price above EMA 50 for longs
        trend_aligned_bearish = current_price < ema_50  # Price below EMA 50 for shorts

        indicator_values = {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "ema_9": ema_9,
            "ema_21": ema_21,
            "ema_50": ema_50,
            "atr": atr,
            "volume_confirmed": volume_confirmed,
        }

        # Detect Elliott Wave pattern
        pattern = self.wave_detector.detect_wave_pattern(df)

        if pattern is None:
            return self._hold_signal(
                symbol, market, data, "No wave pattern detected",
                indicators=indicator_values
            )

        # Determine conditions
        macd_bullish = macd > macd_signal
        volume_above_avg = volume > volume_sma
        price_below_ema9 = current_price < ema_9

        # Check for buy signal (Wave 3 entry)
        is_entry, entry_reason = self.wave_detector.is_wave3_entry(
            pattern, current_price, rsi, macd_bullish, volume_above_avg
        )

        if is_entry:
            # Calculate stop loss and take profit
            if pattern.direction == TrendDirection.BULLISH:
                stop_loss = pattern.wave2_end.price - (atr * config.STOP_LOSS_ATR_MULTIPLIER)
                take_profit = pattern.wave3_target_min
                trend_aligned = trend_aligned_bullish
            else:
                stop_loss = pattern.wave2_end.price + (atr * config.STOP_LOSS_ATR_MULTIPLIER)
                take_profit = pattern.wave3_target_min
                trend_aligned = trend_aligned_bearish

            signal_type = SignalType.BUY if pattern.direction == TrendDirection.BULLISH else SignalType.SELL

            # Add trend alignment to indicators for risk manager validation
            entry_indicators = indicator_values.copy()
            entry_indicators["trend_aligned"] = trend_aligned

            return TradingSignal(
                signal_type=signal_type,
                symbol=symbol,
                market=market,
                price=current_price,
                confidence=pattern.confidence,
                reason=entry_reason,
                wave_pattern=pattern,
                stop_loss=stop_loss,
                take_profit=take_profit,
                indicators=entry_indicators
            )

        # Check for sell signal (Wave 4 exit)
        is_exit, exit_reason = self.wave_detector.is_wave4_exit(
            pattern, current_price, rsi, int(rsi_div), price_below_ema9
        )

        if is_exit:
            signal_type = SignalType.SELL if pattern.direction == TrendDirection.BULLISH else SignalType.BUY

            # Exit signals bypass volume/trend checks - always allow closing positions
            exit_indicators = indicator_values.copy()
            exit_indicators["trend_aligned"] = True
            exit_indicators["volume_confirmed"] = True

            return TradingSignal(
                signal_type=signal_type,
                symbol=symbol,
                market=market,
                price=current_price,
                confidence=pattern.confidence * 0.8,  # Slightly lower confidence for exits
                reason=exit_reason,
                wave_pattern=pattern,
                stop_loss=None,
                take_profit=None,
                indicators=exit_indicators
            )

        # No actionable signal
        wave_status = f"Wave {pattern.current_wave.value}"
        return self._hold_signal(
            symbol, market, data,
            f"{wave_status}: waiting for entry/exit conditions",
            pattern=pattern,
            indicators=indicator_values
        )

    def scan_for_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        market: str
    ) -> List[TradingSignal]:
        """
        Scan multiple symbols for signals.

        Args:
            market_data: Dictionary of symbol -> DataFrame
            market: 'stock' or 'crypto'

        Returns:
            List of actionable signals (non-HOLD)
        """
        signals = []

        for symbol, data in market_data.items():
            try:
                signal = self.analyze(symbol, market, data)
                if signal.signal_type != SignalType.HOLD:
                    signals.append(signal)
                    logger.info(
                        "Signal: %s %s - %s (%.0f%% confidence)",
                        signal.signal_type.value.upper(),
                        symbol,
                        signal.reason,
                        signal.confidence * 100
                    )
            except Exception as e:
                logger.error("Error analyzing %s: %s", symbol, e)

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def _hold_signal(
        self,
        symbol: str,
        market: str,
        data: pd.DataFrame,
        reason: str,
        pattern: Optional[WavePattern] = None,
        indicators: Optional[Dict[str, float]] = None
    ) -> TradingSignal:
        """Create a HOLD signal."""
        current_price = data["close"].iloc[-1] if not data.empty else 0

        return TradingSignal(
            signal_type=SignalType.HOLD,
            symbol=symbol,
            market=market,
            price=current_price,
            confidence=0.0,
            reason=reason,
            wave_pattern=pattern,
            stop_loss=None,
            take_profit=None,
            indicators=indicators or {}
        )

    def get_signal_summary(self, signal: TradingSignal) -> str:
        """
        Get a human-readable summary of a signal.

        Args:
            signal: Trading signal

        Returns:
            Summary string
        """
        lines = [
            f"{'=' * 50}",
            f"SIGNAL: {signal.signal_type.value.upper()} {signal.symbol}",
            f"Market: {signal.market}",
            f"Price: ${signal.price:.4f}",
            f"Confidence: {signal.confidence * 100:.0f}%",
            f"Reason: {signal.reason}",
        ]

        if signal.stop_loss:
            lines.append(f"Stop Loss: ${signal.stop_loss:.4f}")
        if signal.take_profit:
            lines.append(f"Take Profit: ${signal.take_profit:.4f}")

        if signal.wave_pattern:
            wp = signal.wave_pattern
            lines.extend([
                f"Wave Pattern: {wp.direction.value}",
                f"Current Wave: {wp.current_wave.value}",
                f"Wave 2 Retracement: {wp.wave2_retracement * 100:.1f}%",
                f"Wave 3 Targets: ${wp.wave3_target_min:.4f} / ${wp.wave3_target_typical:.4f}",
            ])

        if signal.indicators:
            lines.append("Indicators:")
            lines.append(f"  RSI: {signal.indicators.get('rsi', 0):.1f}")
            lines.append(f"  MACD: {signal.indicators.get('macd', 0):.4f}")

        lines.append(f"{'=' * 50}")
        return "\n".join(lines)
