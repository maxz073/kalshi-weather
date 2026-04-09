"""
Live terminal dashboard for the Kalshi weather trading bot.

Usage:
    python dashboard.py              # refreshes every 30s
    python dashboard.py --interval 10  # refresh every 10s
    python dashboard.py --bot-host 192.168.1.50  # connect to bot on another machine
"""

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime, timezone

import requests
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config
from kalshi_client import KalshiClient

TRADES_FILE = "trades.json"
ALERT_THRESHOLD = 60  # cents — flag positions whose YES price drops below this


# ── Bot remote control ────────────────────────────────────────────────

def get_bot_url(bot_host: str, bot_port: int) -> str:
    return f"http://{bot_host}:{bot_port}"


def check_bot_status(bot_host: str, bot_port: int) -> str:
    """Ping the bot's control server. Returns status string."""
    try:
        resp = requests.get(f"{get_bot_url(bot_host, bot_port)}/status", timeout=3)
        if resp.status_code == 200:
            return resp.text.strip()
        return "unknown"
    except requests.exceptions.ConnectionError:
        return "offline"
    except Exception as e:
        return f"error: {e}"


def send_kill_signal(bot_host: str, bot_port: int) -> str:
    """Send shutdown command to the bot. Returns result message."""
    try:
        resp = requests.post(f"{get_bot_url(bot_host, bot_port)}/shutdown", timeout=5)
        if resp.status_code == 200:
            return "Shutdown signal sent successfully"
        return f"Unexpected response: {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return "Failed — bot is not reachable"
    except Exception as e:
        return f"Failed — {e}"


# ── Keyboard listener ─────────────────────────────────────────────────

class KeyListener:
    """Non-blocking single-keypress reader for macOS/Linux."""

    def __init__(self):
        self.last_key = None
        self._lock = threading.Lock()
        self._old_settings = None
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._running:
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        self.last_key = ch.lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_settings)

    def consume(self) -> str | None:
        """Return the last key pressed and clear it, or None."""
        with self._lock:
            key = self.last_key
            self.last_key = None
            return key

    def stop(self):
        self._running = False


# ── Data fetching ─────────────────────────────────────────────────────

def fetch_balance(client: KalshiClient) -> dict:
    try:
        return client.get_balance()
    except Exception as e:
        return {"error": str(e)}


def fetch_positions(client: KalshiClient) -> list[dict]:
    try:
        data = client.get_positions()
        return data.get("market_positions", [])
    except Exception as e:
        return [{"error": str(e)}]


def fetch_market_price(client: KalshiClient, ticker: str) -> dict | None:
    try:
        return client.get_market(ticker)
    except Exception:
        return None


def load_trades() -> list[dict]:
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r") as f:
            data = f.read()
            return json.loads(data) if data.strip() else []
    except (json.JSONDecodeError, OSError):
        return []


# ── Panel builders ────────────────────────────────────────────────────

def build_header(mode: str, last_refresh: str, bot_status: str) -> Panel:
    title = Text()
    title.append("KALSHI WEATHER TRADING DASHBOARD", style="bold cyan")
    title.append("  |  mode: ", style="dim")
    mode_style = "bold green" if mode == "paper" else "bold red"
    title.append(mode.upper(), style=mode_style)
    title.append("  |  bot: ", style="dim")
    if "running" in bot_status:
        title.append("ONLINE", style="bold green")
    elif bot_status == "offline":
        title.append("OFFLINE", style="bold red")
    else:
        title.append(bot_status.upper(), style="bold yellow")
    title.append("  |  ", style="dim")
    title.append(last_refresh, style="dim")
    title.append("  |  ", style="dim")
    title.append("[K] Kill Bot  [Q] Quit", style="bold yellow")
    return Panel(title, style="cyan", height=3)


def build_balance_panel(balance_data: dict) -> Panel:
    if "error" in balance_data:
        return Panel(Text(f"Error: {balance_data['error']}", style="red"), title="Portfolio")

    # Kalshi returns balance in cents
    balance_cents = balance_data.get("balance", 0)
    payout_cents = balance_data.get("payout", 0)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("label", style="dim")
    table.add_column("value", style="bold")

    table.add_row("Cash Balance", f"${balance_cents / 100:,.2f}")
    table.add_row("Portfolio Payout", f"${payout_cents / 100:,.2f}")
    table.add_row("Total Value", f"${(balance_cents + payout_cents) / 100:,.2f}")

    return Panel(table, title="Portfolio", border_style="green")


def build_positions_panel(positions: list[dict], market_prices: dict) -> Panel:
    if positions and "error" in positions[0]:
        return Panel(Text(f"Error: {positions[0]['error']}", style="red"), title="Open Positions")

    if not positions:
        return Panel(Text("No open positions", style="dim"), title="Open Positions", border_style="blue")

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Side", justify="center")
    table.add_column("Qty", justify="right")
    table.add_column("Avg Price", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Status", justify="center")

    for pos in positions:
        ticker = pos.get("ticker", "?")
        quantity = pos.get("total_traded", 0)
        # Positions can have yes or no side quantities
        yes_qty = pos.get("yes_number", 0)
        no_qty = pos.get("no_number", 0)

        if yes_qty > 0:
            side = "YES"
            qty = yes_qty
        elif no_qty > 0:
            side = "NO"
            qty = no_qty
        else:
            continue  # no position

        # Average entry from position data
        resting_cost = pos.get("market_exposure", 0)  # cents total cost
        avg_price = resting_cost / qty if qty else 0

        # Current market price
        mkt = market_prices.get(ticker)
        if mkt:
            if side == "YES":
                current = mkt.get("yes_ask") or mkt.get("last_price") or 0
            else:
                current = mkt.get("no_ask") or (100 - (mkt.get("last_price") or 0))
        else:
            current = 0

        # P&L estimate (current value - cost)
        if side == "YES":
            current_value = current * qty
        else:
            current_value = current * qty
        pnl = current_value - resting_cost if resting_cost else 0

        # Style P&L
        if pnl > 0:
            pnl_text = Text(f"+${pnl / 100:,.2f}", style="green")
        elif pnl < 0:
            pnl_text = Text(f"-${abs(pnl) / 100:,.2f}", style="red")
        else:
            pnl_text = Text(f"${pnl / 100:,.2f}", style="dim")

        # Alert if price below threshold
        if current > 0 and current < ALERT_THRESHOLD:
            status = Text("BELOW 60", style="bold red")
            current_text = Text(f"{current}c", style="bold red")
        else:
            status = Text("OK", style="green")
            current_text = Text(f"{current}c" if current else "—", style="white")

        side_style = "green" if side == "YES" else "red"

        table.add_row(
            ticker,
            Text(side, style=side_style),
            str(qty),
            f"{avg_price:.0f}c" if avg_price else "—",
            current_text,
            pnl_text,
            status,
        )

    return Panel(table, title="Open Positions", border_style="blue")


def build_alerts_panel(positions: list[dict], market_prices: dict) -> Panel:
    alerts = []

    if positions and "error" not in positions[0]:
        for pos in positions:
            ticker = pos.get("ticker", "?")
            yes_qty = pos.get("yes_number", 0)
            no_qty = pos.get("no_number", 0)
            if yes_qty == 0 and no_qty == 0:
                continue

            mkt = market_prices.get(ticker)
            if not mkt:
                continue

            if yes_qty > 0:
                price = mkt.get("yes_ask") or mkt.get("last_price") or 0
                if 0 < price < ALERT_THRESHOLD:
                    alerts.append(
                        Text.assemble(
                            ("  WARNING  ", "bold white on red"),
                            (f"  {ticker}  YES @ {price}c — below {ALERT_THRESHOLD}c threshold", "red"),
                        )
                    )
            if no_qty > 0:
                price = mkt.get("no_ask") or (100 - (mkt.get("last_price") or 0))
                if 0 < price < ALERT_THRESHOLD:
                    alerts.append(
                        Text.assemble(
                            ("  WARNING  ", "bold white on red"),
                            (f"  {ticker}  NO @ {price}c — below {ALERT_THRESHOLD}c threshold", "red"),
                        )
                    )

    if not alerts:
        content = Text("No alerts — all positions above 60c", style="green")
    else:
        content = Text("\n")
        for a in alerts:
            content.append_text(a)
            content.append("\n")

    return Panel(content, title="Alerts", border_style="red" if alerts else "green")


def build_trades_panel() -> Panel:
    trades = load_trades()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_trades = [t for t in trades if t.get("timestamp", "").startswith(today_str)]

    if not today_trades:
        return Panel(Text("No trades today", style="dim"), title="Today's Trades", border_style="yellow")

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Time (UTC)", style="dim", no_wrap=True)
    table.add_column("City")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Fee", justify="right")
    table.add_column("Mode", justify="center")

    for t in today_trades[-10:]:  # show last 10
        ts = t.get("timestamp", "")
        time_str = ts[11:19] if len(ts) > 19 else ts
        mode_style = "green" if t.get("mode") == "paper" else "red"
        table.add_row(
            time_str,
            t.get("city", "?"),
            t.get("ticker", "?"),
            str(t.get("contracts", 0)),
            f"{t.get('entry_price_cents', 0)}c",
            f"{t.get('fee_cents', 0)}c",
            Text(t.get("mode", "?").upper(), style=mode_style),
        )

    total_contracts = sum(t.get("contracts", 0) for t in today_trades)
    total_cost = sum(t.get("entry_price_cents", 0) * t.get("contracts", 0) for t in today_trades)
    total_fees = sum(t.get("fee_cents", 0) for t in today_trades)

    table.add_section()
    table.add_row(
        "", "TOTAL", "",
        str(total_contracts),
        f"${total_cost / 100:,.2f}",
        f"${total_fees / 100:,.2f}",
        "",
    )

    return Panel(table, title="Today's Trades", border_style="yellow")


# ── Main dashboard ────────────────────────────────────────────────────

def build_dashboard(client: KalshiClient, bot_status: str, kill_msg: str = "") -> Layout:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    balance_data = fetch_balance(client)
    positions = fetch_positions(client)

    # Fetch current prices for all open positions
    market_prices = {}
    for pos in positions:
        if isinstance(pos, dict) and "error" not in pos:
            ticker = pos.get("ticker", "")
            if ticker:
                mkt = fetch_market_price(client, ticker)
                if mkt:
                    market_prices[ticker] = mkt

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=9),
        Layout(name="middle"),
        Layout(name="bottom"),
    )
    layout["top"].split_row(
        Layout(name="balance"),
        Layout(name="alerts"),
    )

    layout["header"].update(build_header(config.TRADING_MODE, now, bot_status))
    layout["balance"].update(build_balance_panel(balance_data))

    # If there's a kill message, show it in the alerts panel temporarily
    if kill_msg:
        layout["alerts"].update(
            Panel(Text(kill_msg, style="bold yellow"), title="Kill Signal", border_style="yellow")
        )
    else:
        layout["alerts"].update(build_alerts_panel(positions, market_prices))

    layout["middle"].update(build_positions_panel(positions, market_prices))
    layout["bottom"].update(build_trades_panel())

    return layout


def main():
    parser = argparse.ArgumentParser(description="Kalshi weather trading dashboard")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    parser.add_argument(
        "--bot-host", type=str,
        default=os.getenv("BOT_HOST", "localhost"),
        help="Hostname/IP of the machine running main.py (default: localhost or BOT_HOST env var)",
    )
    parser.add_argument(
        "--bot-port", type=int,
        default=config.BOT_PORT,
        help=f"Control server port (default: {config.BOT_PORT})",
    )
    args = parser.parse_args()

    if not config.KALSHI_API_KEY_ID:
        print("Error: KALSHI_API_KEY_ID not set in .env")
        sys.exit(1)

    console = Console()
    client = KalshiClient(config.KALSHI_API_KEY_ID, config.KALSHI_PRIVATE_KEY_PATH)

    # Validate Kalshi connection
    try:
        client.get_balance()
    except Exception as e:
        console.print(f"[red]Failed to connect to Kalshi API: {e}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Dashboard starting — refreshing every {args.interval}s[/cyan]")
    console.print(f"[cyan]Bot control target: {args.bot_host}:{args.bot_port}[/cyan]")
    console.print(f"[yellow]Press K to kill the bot, Q to quit the dashboard[/yellow]\n")
    time.sleep(1)

    keys = KeyListener()
    bot_status = check_bot_status(args.bot_host, args.bot_port)
    kill_msg = ""

    with Live(
        build_dashboard(client, bot_status),
        console=console,
        refresh_per_second=0.5,
        screen=True,
    ) as live:
        while True:
            # Check for keypresses every 0.5s within the refresh interval
            elapsed = 0.0
            while elapsed < args.interval:
                time.sleep(0.5)
                elapsed += 0.5

                key = keys.consume()
                if key == "q":
                    keys.stop()
                    live.stop()
                    console.print("\n[cyan]Dashboard closed.[/cyan]")
                    sys.exit(0)
                elif key == "k":
                    kill_msg = send_kill_signal(args.bot_host, args.bot_port)
                    bot_status = check_bot_status(args.bot_host, args.bot_port)
                    live.update(build_dashboard(client, bot_status, kill_msg))

            # Full refresh
            try:
                bot_status = check_bot_status(args.bot_host, args.bot_port)
                # Clear kill message after one full refresh cycle
                kill_msg = ""
                live.update(build_dashboard(client, bot_status))
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDashboard closed.")
