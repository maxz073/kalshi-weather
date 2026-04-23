"""
Backfill historical volume and open_interest data for KXHIGH weather contracts.

Uses two data sources discovered via API probing:
1. /events/{event_ticker} - returns final volume_fp and open_interest_fp per market
2. /markets/trades?ticker=... - returns individual trades with timestamps for
   aggregation into 30-minute volume buckets (4pm-8pm local per city)

Usage:
    python backfill_kalshi.py
    python backfill_kalshi.py --start-date 2026-03-01 --end-date 2026-04-19
    python backfill_kalshi.py --cities KXHIGHNY,KXHIGHCHI

Output: neural-net/data/kalshi_markets/historical_volume.csv
"""
import argparse
import csv
import os
import sys
import time
from datetime import date, datetime, timedelta

import pytz
from dotenv import load_dotenv

load_dotenv()

import config
from kalshi_client import KalshiClient
from market_discovery import date_token

KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_key.pem")

OUT_DIR = os.path.join("neural-net", "data", "kalshi_markets")
OUT_FILE = os.path.join(OUT_DIR, "historical_volume.csv")

FIELDNAMES = [
    "snapshot_ts", "local_time", "date", "city", "series_ticker", "ticker",
    "subtitle", "floor_strike", "cap_strike",
    "volume_in_slot", "volume_92_97", "cumulative_volume", "open_interest",
    "status", "result",
]


def get_utc_window(target_date, tz_name):
    """Return (start_ts, end_ts) as epoch seconds for 4pm-8pm local."""
    tz = pytz.timezone(tz_name)
    local_4pm = tz.localize(datetime(target_date.year, target_date.month, target_date.day, 16, 0))
    local_8pm = tz.localize(datetime(target_date.year, target_date.month, target_date.day, 20, 0))
    return int(local_4pm.timestamp()), int(local_8pm.timestamp())


def load_completed(out_file):
    """Load set of (date, series_ticker) pairs already in the output file."""
    completed = set()
    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["date"], row["series_ticker"]))
    return completed


def fetch_all_trades(client, ticker, min_ts, max_ts):
    """Fetch all trades for a ticker in a time window using cursor pagination."""
    all_trades = []
    cursor = None

    while True:
        params = {"ticker": ticker, "limit": 1000}
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor

        try:
            result = client.get("/markets/trades", params=params)
        except Exception:
            break

        trades = result.get("trades", [])
        all_trades.extend(trades)

        cursor = result.get("cursor")
        if not cursor or not trades:
            break

        time.sleep(0.2)

    return all_trades


def aggregate_trades_to_slots(trades, min_ts, max_ts, interval_min=30):
    """Aggregate trades into 30-min volume buckets using count_fp."""
    slots = {}
    for ts in range(min_ts, max_ts, interval_min * 60):
        slots[ts] = {"volume": 0.0, "volume_92_97": 0.0, "trade_count": 0}

    for trade in trades:
        created_time = trade.get("created_time")
        if not created_time:
            continue

        try:
            trade_ts = int(datetime.fromisoformat(
                created_time.replace("Z", "+00:00")).timestamp())
        except (ValueError, TypeError):
            continue

        # Find which slot this belongs to
        if trade_ts < min_ts or trade_ts >= max_ts:
            continue

        slot_ts = min_ts + ((trade_ts - min_ts) // (interval_min * 60)) * (interval_min * 60)
        if slot_ts in slots:
            count = float(trade.get("count_fp", "0") or "0")
            slots[slot_ts]["volume"] += count
            slots[slot_ts]["trade_count"] += 1

            # Filter for trades in 92-97 cent range
            yes_price = trade.get("yes_price_dollars")
            if yes_price is not None:
                price_cents = round(float(yes_price) * 100)
                if 92 <= price_cents <= 97:
                    slots[slot_ts]["volume_92_97"] += count

    return slots


def get_event_markets(client, event_ticker):
    """Fetch markets via /events/{event_ticker} which includes volume_fp and open_interest_fp."""
    try:
        result = client.get(f"/events/{event_ticker}")
        return result.get("markets", [])
    except Exception:
        return []


def backfill_city_date(client, series_ticker, city_name, target_date, writer):
    """Backfill one city for one date: trade-level 30-min aggregation + final OI."""
    event_ticker = f"{series_ticker}-{date_token(target_date)}"
    tz_name = config.CITIES[series_ticker][1]
    min_ts, max_ts = get_utc_window(target_date, tz_name)
    tz = pytz.timezone(tz_name)

    # Get markets from events endpoint (has volume_fp, open_interest_fp)
    markets = get_event_markets(client, event_ticker)
    if not markets:
        return 0

    rows_written = 0

    for market in markets:
        ticker = market.get("ticker", "")
        final_volume = market.get("volume_fp")
        final_oi = market.get("open_interest_fp")

        time.sleep(0.3)

        # Fetch trades in the 4pm-8pm window
        trades = fetch_all_trades(client, ticker, min_ts, max_ts)

        # Aggregate into 30-min slots
        slots = aggregate_trades_to_slots(trades, min_ts, max_ts)

        # Write a row for each 30-min slot
        for slot_ts in sorted(slots.keys()):
            data = slots[slot_ts]
            snap_dt = datetime.fromtimestamp(slot_ts, tz=pytz.utc).astimezone(tz)

            row = {
                "snapshot_ts": snap_dt.isoformat(),
                "local_time": snap_dt.strftime("%Y-%m-%d %H:%M"),
                "date": target_date.isoformat(),
                "city": city_name,
                "series_ticker": series_ticker,
                "ticker": ticker,
                "subtitle": market.get("subtitle", ""),
                "floor_strike": market.get("floor_strike") or market.get("cap_strike"),
                "cap_strike": market.get("cap_strike"),
                "volume_in_slot": round(data["volume"], 2),
                "volume_92_97": round(data["volume_92_97"], 2),
                "cumulative_volume": final_volume,
                "open_interest": final_oi,
                "status": market.get("status"),
                "result": market.get("result"),
            }
            writer.writerow(row)
            rows_written += 1

    return rows_written


def main():
    parser = argparse.ArgumentParser(description="Backfill KXHIGH volume/OI data")
    parser.add_argument("--start-date", default="2026-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD), defaults to yesterday")
    parser.add_argument("--cities", default=None, help="Comma-separated series tickers (e.g. KXHIGHNY,KXHIGHCHI)")
    args = parser.parse_args()

    if not KALSHI_API_KEY_ID:
        print("ERROR: KALSHI_API_KEY_ID not set in .env")
        sys.exit(1)

    client = KalshiClient(KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH)

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date) if args.end_date else date.today() - timedelta(days=1)

    cities = config.CITIES
    if args.cities:
        filter_set = set(args.cities.split(","))
        cities = {k: v for k, v in cities.items() if k in filter_set}

    print(f"Backfill: {start} to {end}, {len(cities)} cities")
    print(f"Output: {OUT_FILE}")

    # Load resume state
    completed = load_completed(OUT_FILE)
    print(f"Already completed: {len(completed)} (date, city) pairs")
    print()

    # Iterate dates
    os.makedirs(OUT_DIR, exist_ok=True)
    file_exists = os.path.exists(OUT_FILE) and os.path.getsize(OUT_FILE) > 0

    total_rows = 0
    skipped = 0
    current_date = start

    with open(OUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        while current_date <= end:
            for series_ticker, (city_name, tz_name, lat, lon) in cities.items():
                key = (current_date.isoformat(), series_ticker)
                if key in completed:
                    skipped += 1
                    continue

                print(f"  {current_date} | {city_name:<15}", end="", flush=True)
                time.sleep(0.3)

                try:
                    rows = backfill_city_date(client, series_ticker, city_name, current_date, writer)
                    total_rows += rows
                    print(f" | {rows} rows")
                except Exception as e:
                    print(f" | ERROR: {e}")

            current_date += timedelta(days=1)
            f.flush()

    print(f"\nDone. Total rows written: {total_rows}, skipped: {skipped}")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
