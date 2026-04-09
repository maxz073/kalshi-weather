import logging
from datetime import date

import config

logger = logging.getLogger(__name__)


def date_token(d: date) -> str:
    """Convert a date to the Kalshi event ticker format, e.g. 26APR09."""
    return d.strftime("%y%b%d").upper()


def discover_city_markets(client, series_ticker: str, target_date: date) -> list[dict]:
    """Fetch all contracts for a city's temperature event on the given date."""
    event_ticker = f"{series_ticker}-{date_token(target_date)}"
    markets = client.get_markets(event_ticker)
    logger.info("Found %d markets for %s", len(markets), event_ticker)
    return markets


def find_entry_market(
    markets: list[dict],
    entry_min: int = config.ENTRY_MIN,
    entry_max: int = config.ENTRY_MAX,
) -> dict | None:
    """Return the first active market whose YES ask price falls in [entry_min, entry_max] cents."""
    for market in markets:
        if market.get("status") != "active":
            continue
        yes_ask = market.get("yes_ask")
        if yes_ask is not None and entry_min <= yes_ask <= entry_max:
            return market
    return None
