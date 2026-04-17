"""
Live Kalshi market data scraper.

Snapshots all 20 cities' temperature bucket markets (prices, orderbook, volume)
every 15 minutes. Designed to be run via cron:

    */15 * * * * cd /path/to/kalshi-weather/neural-net && python scraper_kalshi.py

Data is appended to data/kalshi_markets/live_snapshots.csv
"""
import csv
import logging
import os
import sys
from datetime import date, datetime, timezone

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalshi_client import KalshiClient, compute_microprice
from market_discovery import date_token
import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Parent config for credentials
parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, parent_dir)
from dotenv import load_dotenv
load_dotenv(os.path.join(parent_dir, ".env"))

KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH",
                                     os.path.join(parent_dir, "kalshi_key.pem"))

OUT_DIR = os.path.join(cfg.DATA_DIR, "kalshi_markets")
OUT_FILE = os.path.join(OUT_DIR, "live_snapshots.csv")

FIELDNAMES = [
    "snapshot_ts", "date", "ticker", "series_ticker", "city",
    "subtitle", "floor_strike", "cap_strike",
    "yes_bid", "yes_ask", "no_bid", "no_ask",
    "last_price", "volume", "open_interest",
    "microprice", "spread",
    "status", "result",
]


def snapshot_all_cities():
    """Fetch and record market data for all cities for today's date."""
    if not KALSHI_API_KEY_ID:
        log.error("KALSHI_API_KEY_ID not set")
        return

    client = KalshiClient(KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH)
    today = date.today()
    ts = datetime.now(timezone.utc).isoformat()

    os.makedirs(OUT_DIR, exist_ok=True)
    file_exists = os.path.exists(OUT_FILE)

    rows_written = 0
    with open(OUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for series_ticker, (city_name, tz, lat, lon) in cfg.CITIES.items():
            event_ticker = f"{series_ticker}-{date_token(today)}"
            try:
                markets = client.get_markets(event_ticker)
            except Exception as e:
                log.warning("Failed to fetch markets for %s: %s", event_ticker, e)
                continue

            for market in markets:
                ticker = market.get("ticker", "")
                yes_bid = market.get("yes_bid_dollars")
                yes_ask = market.get("yes_ask_dollars")
                no_bid = market.get("no_bid_dollars")
                no_ask = market.get("no_ask_dollars")

                # Compute microprice from orderbook
                mp = None
                try:
                    ob = client.get_orderbook(ticker)
                    mp = compute_microprice(ob, market)
                except Exception:
                    pass

                # Compute spread
                spread = None
                if yes_bid is not None and yes_ask is not None:
                    try:
                        spread = round(float(yes_ask) - float(yes_bid), 4)
                    except (ValueError, TypeError):
                        pass

                row = {
                    "snapshot_ts": ts,
                    "date": today.isoformat(),
                    "ticker": ticker,
                    "series_ticker": series_ticker,
                    "city": city_name,
                    "subtitle": market.get("subtitle", ""),
                    "floor_strike": market.get("floor_strike"),
                    "cap_strike": market.get("cap_strike"),
                    "yes_bid": yes_bid,
                    "yes_ask": yes_ask,
                    "no_bid": no_bid,
                    "no_ask": no_ask,
                    "last_price": market.get("last_price_dollars"),
                    "volume": market.get("volume"),
                    "open_interest": market.get("open_interest"),
                    "microprice": mp,
                    "spread": spread,
                    "status": market.get("status"),
                    "result": market.get("result"),
                }
                writer.writerow(row)
                rows_written += 1

    log.info("Snapshot complete: %d market rows written at %s", rows_written, ts)


if __name__ == "__main__":
    snapshot_all_cities()
