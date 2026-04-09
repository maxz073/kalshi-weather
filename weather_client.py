import logging
import requests

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 5  # seconds


def get_current_weather(lat: float, lon: float) -> dict | None:
    """Fetch current humidity and cloud cover from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "relative_humidity_2m,cloud_cover",
    }
    for attempt in range(2):
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            current = resp.json()["current"]
            return {
                "humidity": float(current["relative_humidity_2m"]),
                "cloud_cover": float(current["cloud_cover"]),
            }
        except requests.exceptions.Timeout:
            if attempt == 0:
                logger.warning("Timeout fetching weather (lat=%.2f, lon=%.2f), retrying...", lat, lon)
                continue
            logger.warning("Timeout fetching weather after retry (lat=%.2f, lon=%.2f)", lat, lon)
            return None
        except Exception as e:
            logger.warning("Error fetching weather (lat=%.2f, lon=%.2f): %s", lat, lon, e)
            return None
