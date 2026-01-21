"""
Logging utilities for the trading system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import config


def setup_logger(
    name: str = "elliott_wave",
    level: str = None,
    log_file: str = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up and configure a logger.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: File to write logs to
        console: Whether to also log to console

    Returns:
        Configured logger instance
    """
    level = level or config.LOG_LEVEL
    log_file = log_file or config.LOG_FILE

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (uses module name if None)

    Returns:
        Logger instance
    """
    if name is None:
        name = "elliott_wave"

    logger = logging.getLogger(name)

    # If logger has no handlers, set it up
    if not logger.handlers:
        setup_logger(name)

    return logger


class TradeLogger:
    """Specialized logger for trade events."""

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize trade logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.trade_log_file = self.log_dir / "trades.log"
        self.signal_log_file = self.log_dir / "signals.log"

        self.logger = get_logger("trades")

    def log_trade(
        self,
        action: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reason: str,
        **kwargs
    ):
        """
        Log a trade event.

        Args:
            action: 'OPEN' or 'CLOSE'
            symbol: Asset symbol
            side: 'buy' or 'sell'
            qty: Trade quantity
            price: Trade price
            reason: Trade reason
            **kwargs: Additional fields
        """
        timestamp = datetime.now().isoformat()

        log_entry = (
            f"{timestamp} | {action} | {symbol} | "
            f"{side.upper()} | qty={qty:.6f} | price=${price:.4f} | "
            f"reason={reason}"
        )

        for key, value in kwargs.items():
            log_entry += f" | {key}={value}"

        self.logger.info(log_entry)

        # Also write to trade log file
        with open(self.trade_log_file, "a") as f:
            f.write(log_entry + "\n")

    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        price: float,
        confidence: float,
        reason: str,
        **kwargs
    ):
        """
        Log a trading signal.

        Args:
            symbol: Asset symbol
            signal_type: Signal type (BUY, SELL, HOLD)
            price: Current price
            confidence: Signal confidence
            reason: Signal reason
            **kwargs: Additional fields
        """
        timestamp = datetime.now().isoformat()

        log_entry = (
            f"{timestamp} | SIGNAL | {symbol} | "
            f"{signal_type} | price=${price:.4f} | "
            f"confidence={confidence * 100:.0f}% | reason={reason}"
        )

        for key, value in kwargs.items():
            log_entry += f" | {key}={value}"

        self.logger.info(log_entry)

        # Also write to signal log file
        with open(self.signal_log_file, "a") as f:
            f.write(log_entry + "\n")

    def log_portfolio_snapshot(self, portfolio_data: dict):
        """
        Log portfolio snapshot.

        Args:
            portfolio_data: Portfolio summary data
        """
        timestamp = datetime.now().isoformat()

        snapshot_file = self.log_dir / "portfolio_snapshots.log"

        log_entry = (
            f"{timestamp} | SNAPSHOT | "
            f"equity=${portfolio_data.get('current_equity', 0):,.2f} | "
            f"positions={portfolio_data.get('open_positions', 0)} | "
            f"return={portfolio_data.get('total_return_pct', 0):+.2f}% | "
            f"drawdown={portfolio_data.get('max_drawdown_pct', 0):.2f}%"
        )

        self.logger.info(log_entry)

        with open(snapshot_file, "a") as f:
            f.write(log_entry + "\n")
