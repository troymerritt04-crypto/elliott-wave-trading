#!/usr/bin/env python3
"""
Elliott Wave Momentum Trading System

Main entry point and scheduler for the trading system.
Scans markets for Elliott Wave patterns and executes trades
during Wave 3, exiting during Wave 4.
"""

import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional

import schedule

import config
from data.alpaca_client import AlpacaClient
from data.binance_client import BinanceClient
from data.market_scanner import MarketScanner
from analysis.signal_generator import SignalGenerator, SignalType
from trading.portfolio import PortfolioManager
from trading.risk_manager import RiskManager
from trading.executor import TradeExecutor
from utils.logger import setup_logger, get_logger, TradeLogger

# Initialize logger
setup_logger()
logger = get_logger(__name__)


class TradingSystem:
    """Main trading system orchestrator."""

    def __init__(self):
        """Initialize the trading system."""
        logger.info("=" * 60)
        logger.info("Elliott Wave Momentum Trading System")
        logger.info("=" * 60)
        logger.info("Mode: %s", "PAPER" if config.PAPER_TRADING else "LIVE")

        # Initialize clients
        self.alpaca: Optional[AlpacaClient] = None
        self.binance: Optional[BinanceClient] = None

        try:
            if config.ALPACA_API_KEY:
                self.alpaca = AlpacaClient()
                logger.info("Alpaca client initialized")
        except Exception as e:
            logger.warning("Failed to initialize Alpaca: %s", e)

        try:
            if config.BINANCE_API_KEY:
                self.binance = BinanceClient()
                logger.info("Binance client initialized")
        except Exception as e:
            logger.warning("Failed to initialize Binance: %s", e)

        # Initialize components
        self.scanner = MarketScanner(self.alpaca, self.binance)
        self.signal_generator = SignalGenerator()
        self.portfolio = PortfolioManager()
        self.risk_manager = RiskManager(self.portfolio)
        self.executor = TradeExecutor(
            self.portfolio,
            self.risk_manager,
            self.alpaca,
            self.binance
        )
        self.trade_logger = TradeLogger()

        # State
        self.running = False
        self.last_scan_time: Optional[datetime] = None

        logger.info("Trading system initialized")
        logger.info(self.portfolio.get_summary_string())

    def scan_and_trade(self):
        """Main trading loop - scan for signals and execute trades."""
        logger.info("-" * 40)
        logger.info("Starting market scan...")
        self.last_scan_time = datetime.now()

        try:
            # Check risk limits first
            if self.portfolio.is_max_drawdown_exceeded():
                logger.warning("Max drawdown exceeded - trading paused")
                return

            if self.portfolio.is_daily_loss_exceeded():
                logger.warning("Daily loss limit exceeded - trading paused")
                return

            # Sync positions with brokers
            self.executor.sync_with_broker()

            # Check existing positions for stops
            self._check_position_stops()

            # Scan for new opportunities
            self._scan_stocks()
            self._scan_crypto()

            # Log portfolio snapshot
            self.trade_logger.log_portfolio_snapshot(
                self.portfolio.get_performance_summary()
            )

            logger.info("Scan complete. Next scan in %d minutes.",
                       config.SCAN_INTERVAL_MINUTES)

        except Exception as e:
            logger.error("Error in scan_and_trade: %s", e, exc_info=True)

    def _check_position_stops(self):
        """Check all positions for stop loss / take profit."""
        positions = self.portfolio.get_all_positions()
        if not positions:
            return

        prices = {}

        for pos in positions:
            try:
                price = self.scanner.get_latest_price(pos.symbol, pos.market)
                prices[pos.symbol] = {"price": price, "market": pos.market}
            except Exception as e:
                logger.debug("Could not get price for %s: %s", pos.symbol, e)

        if prices:
            results = self.executor.check_and_close_stopped_positions(prices)
            for result in results:
                if result.success:
                    self.trade_logger.log_trade(
                        action="CLOSE",
                        symbol=result.symbol,
                        side=result.side,
                        qty=result.qty,
                        price=result.price,
                        reason=result.message
                    )

    def _scan_stocks(self):
        """Scan stock market for opportunities."""
        if not self.alpaca:
            return

        if not self.alpaca.is_market_open():
            logger.info("Stock market closed")
            return

        try:
            candidates = self.scanner.scan_stocks()
            self._process_candidates(candidates, "stock")
        except Exception as e:
            logger.error("Stock scan error: %s", e)

    def _scan_crypto(self):
        """Scan crypto market for opportunities."""
        if not self.binance:
            return

        try:
            candidates = self.scanner.scan_crypto()
            self._process_candidates(candidates, "crypto")
        except Exception as e:
            logger.error("Crypto scan error: %s", e)

    def _process_candidates(self, candidates: list, market: str):
        """Process scan candidates and generate signals."""
        if not candidates:
            return

        for candidate in candidates[:10]:  # Limit to top 10
            symbol = candidate["symbol"]

            # Skip if already have position
            if self.portfolio.get_position(symbol):
                continue

            # Skip if at position limit
            if not self.portfolio.can_open_position():
                logger.info("Position limit reached, skipping remaining candidates")
                break

            try:
                # Get historical data
                data = self.scanner.get_market_data(
                    symbol, market, config.TIMEFRAME, days=100
                )

                if data.empty or len(data) < 50:
                    continue

                # Generate signal
                signal = self.signal_generator.analyze(symbol, market, data)

                # Log signal
                self.trade_logger.log_signal(
                    symbol=symbol,
                    signal_type=signal.signal_type.value,
                    price=signal.price,
                    confidence=signal.confidence,
                    reason=signal.reason
                )

                # Execute if actionable
                if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                    logger.info("\n%s", self.signal_generator.get_signal_summary(signal))

                    result = self.executor.execute_signal(signal)

                    if result.success:
                        self.trade_logger.log_trade(
                            action="OPEN",
                            symbol=result.symbol,
                            side=result.side,
                            qty=result.qty,
                            price=result.price,
                            reason=signal.reason,
                            confidence=f"{signal.confidence * 100:.0f}%"
                        )
                    else:
                        logger.warning(
                            "Trade execution failed for %s: %s",
                            symbol, result.message
                        )

            except Exception as e:
                logger.error("Error processing %s: %s", symbol, e)

    def run(self):
        """Run the trading system with scheduler."""
        self.running = True

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        logger.info("Starting trading system scheduler...")

        # Run initial scan
        self.scan_and_trade()

        # Schedule regular scans
        schedule.every(config.SCAN_INTERVAL_MINUTES).minutes.do(self.scan_and_trade)

        # Schedule daily reset
        schedule.every().day.at("00:00").do(self.portfolio.reset_daily_metrics)

        # Run scheduler loop
        while self.running:
            schedule.run_pending()
            time.sleep(1)

        logger.info("Trading system stopped")

    def _shutdown(self, signum, frame):
        """Handle shutdown signal."""
        logger.info("Shutdown signal received...")
        self.running = False

    def run_once(self):
        """Run a single scan (for testing/debugging)."""
        logger.info("Running single scan...")
        self.scan_and_trade()
        logger.info(self.portfolio.get_summary_string())


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Elliott Wave Momentum Trading System"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single scan and exit"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show portfolio status and exit"
    )

    args = parser.parse_args()

    system = TradingSystem()

    if args.status:
        print(system.portfolio.get_summary_string())
        risk_summary = system.risk_manager.get_portfolio_risk_summary()
        print("\nRisk Summary:")
        print(f"  Total Exposure: ${risk_summary['total_exposure']:,.2f} "
              f"({risk_summary['exposure_percent']:.1f}%)")
        print(f"  Total at Risk: ${risk_summary['total_at_risk']:,.2f} "
              f"({risk_summary['risk_percent']:.1f}%)")
        print(f"  Available Slots: {risk_summary['available_slots']}")
    elif args.once:
        system.run_once()
    else:
        system.run()


if __name__ == "__main__":
    main()
