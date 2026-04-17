import logging
import os
import signal
import sys
import threading
import time
from datetime import date
from http.server import HTTPServer, BaseHTTPRequestHandler

import config
from kalshi_client import KalshiClient
from market_discovery import discover_city_markets, find_entry_market
from weather_client import get_current_weather
from strategy import get_local_hour, should_enter
from trader import execute_trade

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


# ── Remote control server ─────────────────────────────────────────────

class ControlHandler(BaseHTTPRequestHandler):
    """Tiny HTTP handler that accepts POST /shutdown and POST /status."""

    def do_POST(self):
        if self.path == "/shutdown":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Shutting down...\n")
            log.warning("Remote shutdown received — exiting")
            # Kill the whole process after responding
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0)), daemon=True).start()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"running | mode={config.TRADING_MODE}\n".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default stderr logging; use our logger instead
        log.debug("Control server: %s", format % args)


def start_control_server():
    """Run the control HTTP server in a daemon thread."""
    try:
        server = HTTPServer((config.BOT_BIND_HOST, config.BOT_PORT), ControlHandler)
    except OSError as e:
        log.error("Cannot start control server on %s:%d — %s", config.BOT_BIND_HOST, config.BOT_PORT, e)
        log.error("Dashboard kill/status will not work. Is another instance running?")
        return None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info("Control server listening on %s:%d", config.BOT_BIND_HOST, config.BOT_PORT)
    return server


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

            # 2. Discover markets & find qualifying price
            markets = discover_city_markets(client, series, today)
            entry_market = find_entry_market(client, markets)
            if entry_market is None:
                log.debug("%-16s  skip — no market with YES ask in %d-%d¢", city_name, config.ENTRY_MIN, config.ENTRY_MAX)
                continue

            ticker = entry_market["ticker"]
            yes_price = entry_market["yes_ask"]

            # 3. Weather check
            weather = get_current_weather(lat, lon)
            if weather is None:
                log.warning("%-16s  skip — weather data unavailable", city_name)
                continue

            humidity = weather["humidity"]
            cloud_cover = weather["cloud_cover"]

            # 4. Entry evaluation
            if not should_enter(local_hour, yes_price, humidity, cloud_cover):
                log.info(
                    "%-16s  no entry — price=%d¢  humidity=%.0f%%  cloud=%.0f%%",
                    city_name, yes_price, humidity, cloud_cover,
                )
                continue

            # 5. Inventory check — buy only what's needed to reach MAX_POSITION_SIZE
            positions = client.get_positions().get("market_positions", [])
            current_qty = 0
            for p in positions:
                if p.get("ticker") == ticker:
                    current_qty = int(float(p.get("position_fp", 0)))
                    break

            needed = config.MAX_POSITION_SIZE - current_qty
            if needed <= 0:
                log.info("%-16s  skip — already at max position (%d) in %s", city_name, current_qty, ticker)
                continue

            # 6. Trade
            log.info(
                "%-16s  ENTRY — %s  qty=%d (have %d, target %d)  price=%d¢  humidity=%.0f%%  cloud=%.0f%%",
                city_name, ticker, needed, current_qty, config.MAX_POSITION_SIZE, yes_price, humidity, cloud_cover,
            )
            execute_trade(client, ticker, city_name, series, yes_price, needed, mode)

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

    # Start remote control server
    start_control_server()

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
