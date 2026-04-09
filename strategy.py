import datetime
import pytz
import config


def get_local_hour(timezone_name: str) -> int:
    """Return the current hour (0-23) in the given timezone."""
    tz = pytz.timezone(timezone_name)
    return datetime.datetime.now(tz).hour


def should_enter(
    local_hour: int,
    yes_price_cents: int,
    humidity: float,
    cloud_cover: float,
) -> bool:
    """Return True only when all four entry conditions are met."""
    return (
        local_hour >= config.ENTRY_HOUR_LOCAL
        and config.ENTRY_MIN <= yes_price_cents <= config.ENTRY_MAX
        and humidity <= config.MAX_HUMIDITY
        and cloud_cover <= config.MAX_CLOUD_COVER
    )
