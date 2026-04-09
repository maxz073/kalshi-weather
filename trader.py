import datetime
import json
import logging
import os

import config

logger = logging.getLogger(__name__)

TRADES_FILE = "trades.json"


def load_trades() -> list[dict]:
    """Read and parse trades.json. Return empty list if file doesn't exist or is empty."""
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r") as f:
            data = f.read()
            if not data.strip():
                return []
            return json.loads(data)
    except (json.JSONDecodeError, OSError):
        return []


def save_trades(trades: list[dict]):
    """Write trades list to trades.json with indent=2."""
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


def get_today_trades(series: str) -> list[dict]:
    """Return trades matching this series ticker and today's date."""
    today_str = datetime.date.today().isoformat()
    trades = load_trades()
    return [
        t for t in trades
        if t.get("series") == series
        and t.get("timestamp", "").startswith(today_str)
    ]


def get_today_contract_count(series: str) -> int:
    """Sum up contracts from today's trades for the given series."""
    return sum(t.get("contracts", 0) for t in get_today_trades(series))


def record_trade(trade_data: dict):
    """Load existing trades, append trade_data, and save back."""
    trades = load_trades()
    trades.append(trade_data)
    save_trades(trades)


def execute_trade(
    client,
    ticker: str,
    city_name: str,
    series: str,
    yes_price_cents: int,
    mode: str,
):
    """Execute (or paper-trade) a YES order and record it."""
    contracts = config.MAX_CONTRACTS
    fee = config.kalshi_taker_fee(yes_price_cents, contracts)

    trade_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "city": city_name,
        "series": series,
        "ticker": ticker,
        "side": "yes",
        "contracts": contracts,
        "entry_price_cents": yes_price_cents,
        "fee_cents": fee,
        "mode": mode,
    }

    if mode == "paper":
        logger.info(
            "[PAPER] %s | %s | %d contracts @ %dc | fee %dc",
            city_name, ticker, contracts, yes_price_cents, fee,
        )
        record_trade(trade_data)
    elif mode == "live":
        logger.info(
            "[LIVE] %s | %s | %d contracts @ %dc | fee %dc",
            city_name, ticker, contracts, yes_price_cents, fee,
        )
        client.post_market_order(ticker, "yes", contracts)
        record_trade(trade_data)
