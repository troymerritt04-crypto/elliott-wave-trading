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
MAX_POSITIONS = 8  # Conservative: fewer, higher-conviction positions
POSITION_SIZE_PERCENT = 0.03  # 3% per position (conservative)
MAX_DRAWDOWN = 0.12  # 12% max portfolio drawdown
MAX_DAILY_LOSS = 0.04  # 4% max single-day loss

# Scanning Settings
SCAN_INTERVAL_MINUTES = 1  # Scan every 1 minute
TIMEFRAME = "1h"  # Primary analysis timeframe

# Elliott Wave Settings
ZIGZAG_PERCENT = 7.0  # Higher threshold filters out noise
MIN_WAVE1_PERCENT = 5.0  # Require stronger Wave 1 moves
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
STOP_LOSS_ATR_MULTIPLIER = 1.0  # Tighter stop loss for better R/R
TAKE_PROFIT_MULTIPLIER = 3.0  # Risk/reward ratio
MIN_CONFIDENCE = 0.65  # Minimum 65% confidence to enter trade
MIN_RISK_REWARD = 1.5  # Minimum 1.5:1 risk/reward ratio
REQUIRE_VOLUME_CONFIRMATION = False  # Disabled - don't require volume confirmation
REQUIRE_TREND_ALIGNMENT = True  # Price must align with EMA 50

# Watchlists
CRYPTO_WATCHLIST = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT"
]

STOCK_WATCHLIST = [
    # Mega Cap Tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    # Semiconductors
    "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON",
    # Software & Cloud
    "CRM", "ADBE", "NOW", "INTU", "SNOW", "PLTR", "PANW", "CRWD", "ZS", "DDOG",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V", "MA", "PYPL",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "DHR", "BMY", "AMGN",
    # Consumer
    "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "BKNG",
    # Industrial & Defense
    "CAT", "DE", "BA", "RTX", "LMT", "GE", "HON", "UPS", "UNP", "MMM",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
    # Communication
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "WBD", "PARA", "FOX",
    # Small/Mid Cap Tech (<$10B)
    "SMCI", "APP", "RKLB", "IONQ", "SOUN", "BIGC", "DCBO", "ALTR", "AI", "PATH",
    "GTLB", "CFLT", "ESTC", "NEWR", "TENB", "JAMF", "RPD", "FRSH", "BRZE", "PCOR",
    # Small/Mid Cap Semiconductors
    "WOLF", "ACLS", "CRUS", "DIOD", "POWI", "SLAB", "SITM", "FORM", "AOSL", "INDI",
    # Small/Mid Cap Biotech & Healthcare
    "EXAS", "RARE", "SRPT", "ALNY", "NBIX", "PCVX", "RYTM", "KRYS", "VERA", "IMVT",
    "TGTX", "AXSM", "CPRX", "FOLD", "ARVN", "KYMR", "RCKT", "APLS", "BEAM", "VERV",
    # Small/Mid Cap Consumer & Retail
    "BOOT", "SHAK", "WING", "CAVA", "BROS", "TOST", "DTC", "CURV", "ONON", "BIRK",
    "FIGS", "XPOF", "PRPL", "LOVE", "LE", "RENT", "POSH", "REAL", "CVNA", "WRBY",
    # Small/Mid Cap Industrials
    "ASPN", "ACHR", "JOBY", "LILM", "EVTL", "RDW", "ASTR", "LUNR", "ASTS", "SATL",
    "BLDE", "POWW", "AXON", "TDG", "HEI", "OSIS", "KTOS", "RCAT", "UMAC", "PRTS",
    # Small/Mid Cap Fintech & Financial
    "AFRM", "UPST", "SOFI", "LC", "HOOD", "RELY", "BILL", "TOST", "MQ", "FOUR",
    "COIN", "MARA", "RIOT", "CLSK", "CIFR", "HUT", "BTBT", "CORZ", "IREN", "WULF",
    # Small/Mid Cap Energy & Clean Tech
    "RUN", "NOVA", "ARRY", "STEM", "CHPT", "EVGO", "BLNK", "DCFC", "PTRA", "ARVL",
    "LAZR", "INVZ", "OUST", "LIDR", "AEVA", "CPTN", "VLDR", "HSAI", "FFIE", "GOEV",
    # Small/Mid Cap Misc Growth
    "CELH", "MNST", "OLPX", "ELF", "HIMS", "PTON", "CHWY", "W", "ETSY", "PINS",
    "SNAP", "MTCH", "BMBL", "RBLX", "U", "TTWO", "EA", "ZNGA", "DKNG", "PENN",
]

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "trading.log"
