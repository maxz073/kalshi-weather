import logging
import sys
import time
from datetime import date

import config
from kalshi_client import KalshiClient
from market_discovery import discover_city_markets, find_entry_market
from weather_client import get_current_weather
from strategy import get_local_hour, should_enter
from trader import get_today_contract_count, execute_trade

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("weather_trading.log"),
    ],
)
log = logging.getLogger("main")


def run_cycle(client: KalshiClient):
    """Check all cities once and trade where conditions are met."""
    today = date.today()
    mode = config.TRADING_MODE

    for series, (city_name, tz_name, lat, lon) in config.CITIES.items():
        try:
            # 1. Time check
            local_hour = get_local_hour(tz_name)
            if local_hour < config.ENTRY_HOUR_LOCAL:
                log.debug("%-16s  skip — local hour %d < %d", city_name, local_hour, config.ENTRY_HOUR_LOCAL)
                continue

            # 2. Daily limit check
            already = get_today_contract_count(series)
            if already >= config.MAX_CONTRACTS:
                log.debug("%-16s  skip — already %d contracts today", city_name, already)
                continue

            # 3. Discover markets & find qualifying price
            markets = discover_city_markets(client, series, today)
            entry_market = find_entry_market(markets)
            if entry_market is None:
                log.debug("%-16s  skip — no market with YES ask in %d-%d¢", city_name, config.ENTRY_MIN, config.ENTRY_MAX)
                continue

            ticker = entry_market["ticker"]
            yes_price = entry_market["yes_ask"]

            # 4. Weather check
            weather = get_current_weather(lat, lon)
            if weather is None:
                log.warning("%-16s  skip — weather data unavailable", city_name)
                continue

            humidity = weather["humidity"]
            cloud_cover = weather["cloud_cover"]

            # 5. Entry evaluation
            if not should_enter(local_hour, yes_price, humidity, cloud_cover):
                log.info(
                    "%-16s  no entry — price=%d¢  humidity=%.0f%%  cloud=%.0f%%",
                    city_name, yes_price, humidity, cloud_cover,
                )
                continue

            # 6. Trade
            log.info(
                "%-16s  ENTRY — %s  price=%d¢  humidity=%.0f%%  cloud=%.0f%%",
                city_name, ticker, yes_price, humidity, cloud_cover,
            )
            execute_trade(client, ticker, city_name, series, yes_price, mode)

        except Exception:
            log.exception("Error processing %s", city_name)


def main():
    # Validate config
    if not config.KALSHI_API_KEY_ID:
        log.error("KALSHI_API_KEY_ID not set in .env")
        sys.exit(1)

    mode = config.TRADING_MODE
    if mode not in ("paper", "live"):
        log.error("TRADING_MODE must be 'paper' or 'live', got '%s'", mode)
        sys.exit(1)

    # Init client
    client = KalshiClient(config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)

    # Validate credentials
    try:
        balance = client.get_balance()
        log.info("Connected — balance: %s  mode: %s", balance, mode)
    except Exception:
        log.exception("Failed to connect to Kalshi API — check credentials")
        sys.exit(1)

    log.info("Monitoring %d cities every %ds", len(config.CITIES), config.POLL_INTERVAL)

    # Main loop
    while True:
        try:
            log.info("── cycle start ──")
            run_cycle(client)
            log.info("── cycle end — sleeping %ds ──", config.POLL_INTERVAL)
        except Exception:
            log.exception("Cycle failed")
        time.sleep(config.POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Shutting down")
