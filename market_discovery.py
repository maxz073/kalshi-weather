import logging
from datetime import date

import config
from kalshi_client import compute_microprice

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
    client,
    markets: list[dict],
    entry_min: int = config.ENTRY_MIN,
    entry_max: int = config.ENTRY_MAX,
) -> dict | None:
    """Return the first active market whose microprice is in [entry_min, entry_max]
    and whose YES ask is at most entry_max cents."""
    for market in markets:
        status = market.get("status")
        if status not in ("active", "open"):
            logger.debug("Skipping %s — status=%s", market.get("ticker"), status)
            continue

        yes_ask_str = market.get("yes_ask_dollars")
        if yes_ask_str is None:
            continue
        yes_ask = int(float(yes_ask_str) * 100)

        # Ask must be at most entry_max (96c) to be purchasable
        if yes_ask > entry_max:
            logger.debug("Skipping %s — yes_ask %dc > %dc", market.get("ticker"), yes_ask, entry_max)
            continue

        # Compute microprice for fair value check
        ticker = market.get("ticker")
        try:
            orderbook = client.get_orderbook(ticker)
            mp = compute_microprice(orderbook, market)
        except Exception:
            logger.warning("Could not fetch orderbook for %s, falling back to ask", ticker)
            mp = None

        if mp is None:
            # Fall back to ask price if orderbook is unavailable
            mp = yes_ask

        if entry_min <= mp <= entry_max:
            market["yes_ask"] = yes_ask
            market["microprice"] = mp
            logger.info("%s microprice=%dc yes_ask=%dc — qualifies", ticker, mp, yes_ask)
            return market
        else:
            logger.debug("Skipping %s — microprice %dc outside %d-%dc", ticker, mp, entry_min, entry_max)

    return None
