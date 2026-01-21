"""
Configuration settings for the Elliott Wave Trading System.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Trading Mode
PAPER_TRADING = True

# Alpaca Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Binance Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# Portfolio Settings
MAX_POSITIONS = 20
POSITION_SIZE_PERCENT = 0.04  # 4% per position
MAX_DRAWDOWN = 0.15  # 15% max portfolio drawdown
MAX_DAILY_LOSS = 0.05  # 5% max single-day loss

# Scanning Settings
SCAN_INTERVAL_MINUTES = 15
TIMEFRAME = "1h"  # Primary analysis timeframe

# Elliott Wave Settings
ZIGZAG_PERCENT = 5.0  # Minimum % move for zigzag swing detection
MIN_WAVE1_PERCENT = 3.0  # Minimum % move for valid Wave 1
WAVE2_MIN_RETRACE = 0.382  # Wave 2 minimum retracement of Wave 1
WAVE2_MAX_RETRACE = 0.786  # Wave 2 maximum retracement of Wave 1
WAVE3_MIN_EXTENSION = 1.618  # Wave 3 minimum Fib extension
WAVE3_TARGET_EXTENSION = 2.618  # Wave 3 typical target

# Technical Indicator Settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 9
EMA_MID = 21
EMA_LONG = 50
ATR_PERIOD = 14

# Risk Management
STOP_LOSS_ATR_MULTIPLIER = 2.0
TAKE_PROFIT_MULTIPLIER = 3.0  # Risk/reward ratio

# Watchlists
CRYPTO_WATCHLIST = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT"
]

STOCK_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AMD", "JPM", "V",
    "SPY", "QQQ", "IWM"
]

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "trading.log"
