import math
import os
from dotenv import load_dotenv

load_dotenv()

# ── Kalshi credentials ──
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_key.pem")
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# ── Trading mode ──
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"

# ── Remote control (bot exposes this; dashboard connects to it) ──
BOT_BIND_HOST = os.getenv("BOT_BIND_HOST", "0.0.0.0")   # bind address for the bot's control server
BOT_CONNECT_HOST = os.getenv("BOT_CONNECT_HOST", "localhost")  # address the dashboard uses to reach the bot
BOT_PORT = int(os.getenv("BOT_PORT", "8377"))  # port for the bot's control server

# ── Strategy parameters ──
ENTRY_MIN = 92   # cents (YES price lower bound)
ENTRY_MAX = 96   # cents (YES price upper bound)
ENTRY_HOUR_LOCAL = 16  # 4 PM local
MAX_CONTRACTS = 50
POLL_INTERVAL = 60  # 1 minute in seconds

# ── Weather thresholds ──
MAX_HUMIDITY = 60.0      # percent
MAX_CLOUD_COVER = 60.0   # percent

# ── Fee model ──
TAKER_FEE_RATE = 0.07  # 7%


def kalshi_taker_fee(price_cents: int, contracts: int = 1) -> int:
    """Kalshi taker fee: ceil(0.07 * C * P * (1-P)), in cents."""
    p = price_cents / 100
    return math.ceil(TAKER_FEE_RATE * contracts * p * (1 - p) * 100)


# ── Cities: series ticker → (display name, timezone, latitude, longitude) ──
CITIES = {
    "KXHIGHNY":     ("New York",        "US/Eastern",   40.71, -74.01),
    "KXHIGHCHI":    ("Chicago",         "US/Central",   41.88, -87.63),
    "KXHIGHMIA":    ("Miami",           "US/Eastern",   25.76, -80.19),
    "KXHIGHTBOS":   ("Boston",          "US/Eastern",   42.36, -71.06),
    "KXHIGHLAX":    ("Los Angeles",     "US/Pacific",   34.05, -118.24),
    "KXHIGHAUS":    ("Austin",          "US/Central",   30.27, -97.74),
    "KXHIGHTSFO":   ("San Francisco",   "US/Pacific",   37.77, -122.42),
    "KXHIGHTDAL":   ("Dallas",          "US/Central",   32.78, -96.80),
    "KXHIGHPHIL":   ("Philadelphia",    "US/Eastern",   39.95, -75.17),
    "KXHIGHTPHX":   ("Phoenix",         "US/Arizona",   33.45, -112.07),
    "KXHIGHTOKC":   ("Oklahoma City",   "US/Central",   35.47, -97.52),
    "KXHIGHDEN":    ("Denver",          "US/Mountain",  39.74, -104.98),
    "KXHIGHTDC":    ("Washington DC",   "US/Eastern",   38.91, -77.04),
    "KXHIGHTSATX":  ("San Antonio",     "US/Central",   29.42, -98.49),
    "KXHIGHHOU":    ("Houston",         "US/Central",   29.76, -95.37),
    "KXHIGHTMIN":   ("Minneapolis",     "US/Central",   44.98, -93.27),
    "KXHIGHTATL":   ("Atlanta",         "US/Eastern",   33.75, -84.39),
    "KXHIGHTSEA":   ("Seattle",         "US/Pacific",   47.61, -122.33),
    "KXHIGHTLV":    ("Las Vegas",       "US/Pacific",   36.17, -115.14),
    "KXHIGHTNOLA":  ("New Orleans",     "US/Central",   29.95, -90.07),
}
