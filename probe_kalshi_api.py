"""
Probe Kalshi API for historical market data endpoints.

Tests various endpoints to discover what historical volume/open_interest
data is available for KXHIGH weather contracts.

Usage:
    python probe_kalshi_api.py
"""
import os
import sys
import time
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

import config
from kalshi_client import KalshiClient
from market_discovery import date_token

KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_key.pem")


def probe_endpoint(client, method, endpoint, params=None, label=""):
    """Try an endpoint and report results."""
    print(f"\n{'='*60}")
    print(f"PROBE: {label}")
    print(f"  {method} {endpoint}")
    if params:
        print(f"  params: {params}")
    print("-" * 60)

    try:
        if method == "GET":
            result = client.get(endpoint, params=params)
        else:
            print("  Unsupported method")
            return None

        # Print structure
        if isinstance(result, dict):
            print(f"  Status: 200 OK")
            print(f"  Top-level keys: {list(result.keys())}")
            for key, val in result.items():
                if isinstance(val, list):
                    print(f"    {key}: list[{len(val)}]")
                    if val:
                        if isinstance(val[0], dict):
                            print(f"      Item keys: {list(val[0].keys())}")
                            # Show first item
                            for k, v in val[0].items():
                                print(f"        {k}: {repr(v)[:80]}")
                        else:
                            print(f"      First item: {repr(val[0])[:100]}")
                elif isinstance(val, dict):
                    print(f"    {key}: dict with keys {list(val.keys())[:10]}")
                else:
                    print(f"    {key}: {repr(val)[:100]}")
        else:
            print(f"  Status: 200 OK")
            print(f"  Response type: {type(result).__name__}")
            print(f"  Value: {repr(result)[:200]}")

        return result

    except Exception as e:
        status = ""
        if hasattr(e, "response") and e.response is not None:
            status = f" (HTTP {e.response.status_code})"
            try:
                body = e.response.text[:200]
                print(f"  Status: FAILED{status}")
                print(f"  Body: {body}")
            except:
                print(f"  Status: FAILED{status}")
        else:
            print(f"  Status: FAILED - {type(e).__name__}: {e}")
        return None


def main():
    if not KALSHI_API_KEY_ID:
        print("ERROR: KALSHI_API_KEY_ID not set in .env")
        sys.exit(1)

    client = KalshiClient(KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH)

    # Verify auth
    print("Verifying authentication...")
    try:
        balance = client.get_balance()
        print(f"  Auth OK. Balance: {balance}")
    except Exception as e:
        print(f"  Auth FAILED: {e}")
        sys.exit(1)

    # Test dates: yesterday, 1 week ago, 1 month ago, Jan 2026
    today = date.today()
    test_dates = {
        "yesterday": today - timedelta(days=1),
        "1_week_ago": today - timedelta(days=7),
        "1_month_ago": today - timedelta(days=30),
        "jan_2026": date(2026, 1, 15),
    }

    # Use New York as test city
    series = "KXHIGHNY"

    print("\n" + "=" * 60)
    print("DISCOVERING TEST TICKERS")
    print("=" * 60)

    test_tickers = {}
    for label, d in test_dates.items():
        event_ticker = f"{series}-{date_token(d)}"
        print(f"\n  {label} ({d}): event={event_ticker}")
        time.sleep(0.5)
        try:
            markets = client.get_markets(event_ticker)
            if markets:
                ticker = markets[0]["ticker"]
                test_tickers[label] = ticker
                print(f"    Found {len(markets)} markets. Using: {ticker}")
                print(f"    volume={markets[0].get('volume')}, "
                      f"open_interest={markets[0].get('open_interest')}, "
                      f"status={markets[0].get('status')}")
            else:
                print(f"    No markets found")
        except Exception as e:
            print(f"    Failed: {e}")

    if not test_tickers:
        print("\nNo valid tickers found. Cannot proceed with probing.")
        sys.exit(1)

    # Pick the most recent valid ticker for detailed probing
    probe_ticker = None
    for label in ["yesterday", "1_week_ago", "1_month_ago", "jan_2026"]:
        if label in test_tickers:
            probe_ticker = test_tickers[label]
            probe_label = label
            break

    print(f"\nUsing ticker for detailed probes: {probe_ticker} ({probe_label})")

    # ── Probe 1: /markets/{ticker}/history ──
    time.sleep(0.5)
    probe_endpoint(client, "GET", f"/markets/{probe_ticker}/history",
                   label=f"/markets/{{ticker}}/history (no params)")

    time.sleep(0.5)
    # With period_interval
    now_ts = int(time.time())
    week_ago_ts = now_ts - 7 * 86400
    probe_endpoint(client, "GET", f"/markets/{probe_ticker}/history",
                   params={"min_ts": week_ago_ts, "max_ts": now_ts, "period_interval": 30},
                   label=f"/markets/{{ticker}}/history (period_interval=30)")

    time.sleep(0.5)
    probe_endpoint(client, "GET", f"/markets/{probe_ticker}/history",
                   params={"min_ts": week_ago_ts, "max_ts": now_ts, "period_interval": 1},
                   label=f"/markets/{{ticker}}/history (period_interval=1)")

    # ── Probe 2: /markets/{ticker}/trades ──
    time.sleep(0.5)
    probe_endpoint(client, "GET", f"/markets/{probe_ticker}/trades",
                   label=f"/markets/{{ticker}}/trades")

    time.sleep(0.5)
    probe_endpoint(client, "GET", "/markets/trades",
                   params={"ticker": probe_ticker, "limit": 10},
                   label="/markets/trades?ticker=...")

    # ── Probe 3: /events/{event_ticker} ──
    for label, d in test_dates.items():
        if label in test_tickers:
            event_ticker = f"{series}-{date_token(d)}"
            time.sleep(0.5)
            probe_endpoint(client, "GET", f"/events/{event_ticker}",
                           label=f"/events/{{event_ticker}} ({label})")
            break

    # ── Probe 4: /series/{series_ticker} ──
    time.sleep(0.5)
    probe_endpoint(client, "GET", f"/series/{series}",
                   label=f"/series/{series}")

    # ── Probe 5: Check settled market still returns volume/OI ──
    # Try the oldest ticker we found
    for label in ["jan_2026", "1_month_ago", "1_week_ago", "yesterday"]:
        if label in test_tickers:
            old_ticker = test_tickers[label]
            time.sleep(0.5)
            probe_endpoint(client, "GET", f"/markets/{old_ticker}",
                           label=f"/markets/{{ticker}} - settled ({label})")
            break

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PROBE COMPLETE")
    print("=" * 60)
    print("\nReview the output above to determine which endpoints")
    print("return historical volume and open_interest data.")
    print("\nNext steps:")
    print("  - If /markets/{ticker}/history works: use it in backfill_kalshi.py")
    print("  - If /markets/trades works: aggregate trades into 30-min buckets")
    print("  - If only /markets/{ticker} works: can get final volume/OI per market")


if __name__ == "__main__":
    main()
