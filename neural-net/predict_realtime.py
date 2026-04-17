"""
Real-time temperature prediction for all 20 Kalshi weather cities.

Loads the 3 trained models + ensemble, fetches live data from Open-Meteo,
NWS (via ACIS), and (optionally) Kalshi, and outputs predicted NWS daily
recorded high temperature with uncertainty and Kalshi bucket fair values.

Target: NWS daily recorded high — the official Kalshi settlement value.

Usage:
    python predict_realtime.py              # predict for today
    python predict_realtime.py 2026-04-20   # predict for a specific date

No look-ahead bias: only uses data available before prediction time.
"""
import argparse
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import torch

import config as cfg
import feature_utils as fu
import training
import evaluation as ev

# Allow imports from parent for Kalshi client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ACIS_URL = "https://data.rcc-acis.org/StnData"

FORECAST_MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "jma_seamless"]
FORECAST_COLS = [f"fcst_{m}" for m in FORECAST_MODELS]

CHECKPOINT_DIR = cfg.CHECKPOINT_DIR
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ── Data fetching (live) ─────────────────────────────────────────────

def fetch_recent_nws_daily(station_id: str, n_days: int = 35) -> pd.DataFrame:
    """Fetch recent NWS daily recorded highs from ACIS.
    Used for lag features in Model 2 (up to 28 days) + rolling windows.
    """
    end = date.today() - timedelta(days=1)  # yesterday
    start = end - timedelta(days=n_days)
    payload = {
        "sid": station_id,
        "sdate": start.isoformat(),
        "edate": end.isoformat(),
        "elems": [{"name": "maxt"}],
        "output": "json",
    }
    resp = requests.post(ACIS_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for entry in data.get("data", []):
        date_str = entry[0]
        val = entry[1]
        if isinstance(val, (int, float)):
            rows.append({"date": date_str, "nws_high": float(val)})
        elif isinstance(val, str) and val not in ("M", "T", "S", ""):
            try:
                rows.append({"date": date_str, "nws_high": float(val)})
            except ValueError:
                rows.append({"date": date_str, "nws_high": np.nan})
        else:
            rows.append({"date": date_str, "nws_high": np.nan})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_recent_daily(lat: float, lon: float, tz: str, n_days: int = 35) -> pd.DataFrame:
    """Fetch the last n_days of daily weather from Open-Meteo archive.
    Used for meteorological features (NOT the target — NWS high is the target).
    """
    end = date.today() - timedelta(days=1)  # yesterday (latest completed day)
    start = end - timedelta(days=n_days)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,dewpoint_2m_mean,"
                 "surface_pressure_mean,cloud_cover_mean,wind_speed_10m_max,"
                 "wind_direction_10m_dominant,precipitation_sum,snowfall_sum",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": tz,
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["daily"])
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_recent_hourly(lat: float, lon: float, tz: str) -> dict:
    """Fetch yesterday's hourly temps (for temp path features)."""
    yesterday = date.today() - timedelta(days=1)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": yesterday.isoformat(), "end_date": yesterday.isoformat(),
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": tz,
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    hourly = pd.DataFrame(data["hourly"])
    hourly["hour"] = pd.to_datetime(hourly["time"]).dt.hour
    temps = {}
    for h in [6, 9, 12, 15]:
        row = hourly[hourly["hour"] == h]
        temps[f"temp_{h}"] = float(row["temperature_2m"].iloc[0]) if len(row) else np.nan
    temps["temp_path_range"] = max(temps.values()) - min(temps.values()) if temps else 0.0
    return temps


def fetch_forecasts_for_date(lat: float, lon: float, tz: str, target_date: date) -> dict:
    """Fetch forecast daily max temp from all 5 models for the target date."""
    # Try the current forecast API first (for today/tomorrow)
    forecasts = {}
    for model in FORECAST_MODELS:
        params = {
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": tz,
            "models": model,
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
        }
        try:
            # Try current forecast endpoint first
            resp = requests.get(FORECAST_URL, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                temps = daily.get("temperature_2m_max", [])
                if temps and temps[0] is not None:
                    forecasts[f"fcst_{model}"] = float(temps[0])
                    continue
            # Fall back to historical forecast API
            resp = requests.get(HIST_FORECAST_URL, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("daily", {})
                temps = daily.get("temperature_2m_max", [])
                if temps and temps[0] is not None:
                    forecasts[f"fcst_{model}"] = float(temps[0])
        except Exception as e:
            log.warning("Forecast fetch failed for model %s: %s", model, e)
        time.sleep(0.3)
    return forecasts


def fetch_climate_indices() -> dict:
    """Fetch latest climate indices (ENSO, AO, NAO, PNA).
    Returns most recent values, forward-filled.
    """
    indices = {}

    # Try to load from cached files first
    climate_dir = os.path.join(cfg.DATA_DIR, "climate_indices")
    for name, col in [("enso_oni", "enso_oni"), ("ao", "ao"), ("nao", "nao"), ("pna", "pna")]:
        path = os.path.join(climate_dir, f"{name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date")
            if col in df.columns and len(df) > 0:
                indices[col] = float(df[col].dropna().iloc[-1])
    return indices


# ── Feature builders for each model ──────────────────────────────────

def build_model1_features(
    ticker: str, target_date: date,
    forecasts: dict, recent_daily: pd.DataFrame, hourly_temps: dict,
) -> tuple[np.ndarray, int]:
    """Build Model 1 feature vector for a single city-day.
    Returns (features_array, city_idx).
    """
    city_idx = cfg.TICKER_TO_IDX[ticker]

    # Forecast values
    fcst_vals = [forecasts.get(c, np.nan) for c in FORECAST_COLS]
    fcst_arr = np.array(fcst_vals)
    # Fill missing forecasts with mean of available
    if np.any(np.isnan(fcst_arr)):
        mean_val = np.nanmean(fcst_arr)
        fcst_arr = np.where(np.isnan(fcst_arr), mean_val, fcst_arr)

    forecast_mean = np.mean(fcst_arr)
    forecast_std = np.std(fcst_arr)
    forecast_range = np.max(fcst_arr) - np.min(fcst_arr)
    q75, q25 = np.percentile(fcst_arr, [75, 25]) if len(fcst_arr) >= 4 else (fcst_arr.max(), fcst_arr.min())
    forecast_iqr = q75 - q25

    # Pairwise spreads (vs GFS)
    gfs = fcst_arr[0]
    spread_ecmwf = fcst_arr[1] - gfs
    spread_icon = fcst_arr[2] - gfs
    spread_gem = fcst_arr[3] - gfs
    spread_jma = fcst_arr[4] - gfs

    # Lagged meteorological (yesterday's values — last row of recent_daily)
    yesterday = recent_daily.iloc[-1] if len(recent_daily) > 0 else {}
    cloud_lag1 = float(yesterday.get("cloud_cover_mean", 0) or 0)
    dewpoint_lag1 = float(yesterday.get("dewpoint_2m_mean", 0) or 0)
    wind_lag1 = float(yesterday.get("wind_speed_10m_max", 0) or 0)
    pressure_lag1 = float(yesterday.get("surface_pressure_mean", 0) or 0)
    precip_lag1 = float(yesterday.get("precipitation_sum", 0) or 0)

    # Lagged hourly temp path (yesterday)
    t6 = hourly_temps.get("temp_6", 0.0) or 0.0
    t9 = hourly_temps.get("temp_9", 0.0) or 0.0
    t12 = hourly_temps.get("temp_12", 0.0) or 0.0
    t15 = hourly_temps.get("temp_15", 0.0) or 0.0
    t_range = hourly_temps.get("temp_path_range", 0.0) or 0.0

    # Yesterday's low
    tmin_lag1 = float(yesterday.get("temperature_2m_min", 0) or 0)

    # Rolling forecast bias (use recent residuals if we have prior actuals + forecasts)
    # Approximate: compute from recent_daily vs forecast_mean (use 0 if unavailable)
    roll7 = roll14 = roll30 = 0.0

    # City static
    static = fu.city_static_features(ticker)
    lat = static["lat"]
    lon = static["lon"]
    elev = static["elevation_ft"]
    coastal = static["coastal"]
    cont = static["continentality"]

    # Calendar
    doy = target_date.timetuple().tm_yday
    month = target_date.month
    sin_doy, cos_doy = fu.sin_cos_encode(np.array([doy], dtype=float), 365.25)
    sin_month, cos_month = fu.sin_cos_encode(np.array([month], dtype=float), 12)

    features = np.array([
        *fcst_arr,                                          # 5 forecast highs
        forecast_mean, forecast_std, forecast_range, forecast_iqr,  # 4 ensemble stats
        spread_ecmwf, spread_icon, spread_gem, spread_jma,  # 4 spreads
        cloud_lag1, dewpoint_lag1, wind_lag1, pressure_lag1, precip_lag1,  # 5 meteo lag1
        t6, t9, t12, t15, t_range,                          # 5 hourly path lag1
        tmin_lag1,                                           # 1 min temp lag1
        roll7, roll14, roll30,                               # 3 rolling bias
        lat, lon, elev, coastal, cont,                       # 5 static
        sin_doy[0], cos_doy[0], sin_month[0], cos_month[0], # 4 calendar
    ], dtype=np.float32)

    return features, city_idx, forecast_mean


def build_model2_features(
    ticker: str, target_date: date,
    recent_daily: pd.DataFrame,
    all_city_recent: dict[str, pd.DataFrame],
    climate: dict,
    nws_recent: pd.DataFrame | None = None,
    all_city_nws: dict[str, pd.DataFrame] | None = None,
) -> tuple[np.ndarray, int]:
    """Build Model 2 feature vector for a single city-day.
    Uses NWS recorded highs for temperature lag/rolling features.
    """
    city_idx = cfg.TICKER_TO_IDX[ticker]
    rd = recent_daily.copy()

    # Calendar
    doy = target_date.timetuple().tm_yday
    month = target_date.month
    woy = target_date.isocalendar()[1]
    sin_doy, cos_doy = fu.sin_cos_encode(np.array([doy], dtype=float), 365.25)
    sin_month, cos_month = fu.sin_cos_encode(np.array([month], dtype=float), 12)
    sin_woy, cos_woy = fu.sin_cos_encode(np.array([woy], dtype=float), 52)

    # Own-city lagged NWS highs (fall back to Open-Meteo if NWS unavailable)
    if nws_recent is not None and len(nws_recent) > 0:
        tmax = nws_recent["nws_high"].dropna().values
    else:
        tmax = rd["temperature_2m_max"].values
    lags = {}
    for lag in [1, 2, 3, 5, 7, 14, 21, 28]:
        lags[f"lag{lag}"] = float(tmax[-lag]) if len(tmax) >= lag else 0.0

    # Rolling stats (computed on available history, excluding today)
    def safe_rolling(arr, w, stat="mean"):
        if len(arr) < 1:
            return 0.0
        window = arr[-w:] if len(arr) >= w else arr
        if stat == "mean":
            return float(np.mean(window))
        elif stat == "std":
            return float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
        elif stat == "max":
            return float(np.max(window))
        elif stat == "min":
            return float(np.min(window))

    roll_feats = {
        "roll3_mean": safe_rolling(tmax, 3),
        "roll7_mean": safe_rolling(tmax, 7),
        "roll14_mean": safe_rolling(tmax, 14),
        "roll30_mean": safe_rolling(tmax, 30),
        "roll7_std": safe_rolling(tmax, 7, "std"),
        "roll14_std": safe_rolling(tmax, 14, "std"),
        "roll7_max": safe_rolling(tmax, 7, "max"),
        "roll7_min": safe_rolling(tmax, 7, "min"),
    }

    # Min temp rolling (Open-Meteo — NWS low not fetched separately)
    tmin = rd["temperature_2m_min"].values if "temperature_2m_min" in rd.columns else np.zeros(len(rd))
    roll_feats["tmin_roll7_mean"] = safe_rolling(tmin, 7)

    # Diurnal range rolling (NWS high - Open-Meteo low)
    if nws_recent is not None and len(nws_recent) > 0:
        # Align NWS highs with Open-Meteo lows by date
        nws_vals = nws_recent["nws_high"].dropna().values
        min_len = min(len(nws_vals), len(tmin))
        diurnal = nws_vals[-min_len:] - tmin[-min_len:]
    else:
        diurnal = rd["temperature_2m_max"].values - tmin
    roll_feats["diurnal_roll7_mean"] = safe_rolling(diurnal, 7)

    # Neighbor features (using NWS highs where available)
    all_nws = all_city_nws or {}
    neighbors = cfg.NEIGHBORS.get(ticker, [])
    nb_feats = [float(len(neighbors))]  # n_neighbors
    for slot in range(4):
        if slot < len(neighbors):
            nb_ticker = neighbors[slot]
            nb_nws = all_nws.get(nb_ticker)
            if nb_nws is not None and len(nb_nws) > 0:
                nb_tmax = nb_nws["nws_high"].dropna().values
            else:
                nb_rd = all_city_recent.get(nb_ticker)
                nb_tmax = nb_rd["temperature_2m_max"].values if nb_rd is not None and len(nb_rd) > 0 else np.array([])
            if len(nb_tmax) > 0:
                nb_feats.append(float(nb_tmax[-1]) if len(nb_tmax) >= 1 else 0.0)
                nb_feats.append(float(nb_tmax[-2]) if len(nb_tmax) >= 2 else 0.0)
                nb_feats.append(float(nb_tmax[-3]) if len(nb_tmax) >= 3 else 0.0)
                nb_feats.append(safe_rolling(nb_tmax, 3))
            else:
                nb_feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            nb_feats.extend([0.0, 0.0, 0.0, 0.0])

    # Climate indices
    enso = climate.get("enso_oni", 0.0)
    ao = climate.get("ao", 0.0)
    nao = climate.get("nao", 0.0)
    pna = climate.get("pna", 0.0)

    # Synoptic lags
    synoptic_vars = ["dewpoint_2m_mean", "surface_pressure_mean", "cloud_cover_mean",
                     "wind_speed_10m_max", "precipitation_sum", "snowfall_sum"]
    syn_feats = []
    for var in synoptic_vars:
        vals = rd[var].values if var in rd.columns else np.zeros(len(rd))
        syn_feats.append(float(vals[-1]) if len(vals) >= 1 else 0.0)  # lag1
        syn_feats.append(float(vals[-2]) if len(vals) >= 2 else 0.0)  # lag2

    # Static
    static = fu.city_static_features(ticker)

    features = np.array([
        sin_doy[0], cos_doy[0], sin_month[0], cos_month[0], sin_woy[0], cos_woy[0],
        *[lags[f"lag{l}"] for l in [1, 2, 3, 5, 7, 14, 21, 28]],
        *[roll_feats[k] for k in ["roll3_mean", "roll7_mean", "roll14_mean", "roll30_mean",
                                   "roll7_std", "roll14_std", "roll7_max", "roll7_min",
                                   "tmin_roll7_mean", "diurnal_roll7_mean"]],
        *nb_feats,
        enso, ao, nao, pna,
        *syn_feats,
        static["lat"], static["lon"], static["elevation_ft"],
        static["coastal"], static["desert"], static["continentality"],
    ], dtype=np.float32)

    return features, city_idx


def build_model3_features(
    ticker: str, target_date: date,
    forecast_mean: float,
    market_data: dict | None = None,
) -> tuple[np.ndarray, int]:
    """Build Model 3 feature vector. Uses live market data if available,
    otherwise uses forecast-derived defaults (low confidence).
    """
    city_idx = cfg.TICKER_TO_IDX[ticker]
    now = datetime.now()
    local_hour = now.hour  # approximate

    if market_data and market_data.get("buckets"):
        # Real market data available
        buckets = market_data["buckets"]  # list of dicts with price, spread, volume
        bucket_prices = [b.get("price", 0.5) for b in buckets]
        bucket_spreads = [b.get("spread", 0.05) for b in buckets]
        bucket_volumes = [b.get("volume", 0) for b in buckets]
        # Pad/truncate to 6 buckets
        while len(bucket_prices) < 6:
            bucket_prices.append(0.0)
            bucket_spreads.append(0.0)
            bucket_volumes.append(0.0)
        bucket_prices = bucket_prices[:6]
        bucket_spreads = bucket_spreads[:6]
        bucket_volumes = bucket_volumes[:6]

        bp = np.array(bucket_prices)
        total_p = bp.sum()
        if total_p > 0:
            probs = bp / total_p
        else:
            probs = np.ones(6) / 6

        mids = np.array([forecast_mean - 12.5, forecast_mean - 7.5, forecast_mean - 2.5,
                         forecast_mean + 2.5, forecast_mean + 7.5, forecast_mean + 12.5])
        implied_exp = float(np.dot(probs, mids))
        implied_var = float(np.dot(probs, (mids - implied_exp) ** 2))
        imp_std = np.sqrt(implied_var) if implied_var > 0 else 1.0
        implied_skew = float(np.dot(probs, ((mids - implied_exp) / imp_std) ** 3)) if imp_std > 0 else 0.0
        implied_kurt = float(np.dot(probs, ((mids - implied_exp) / imp_std) ** 4) - 3.0) if imp_std > 0 else 0.0
        upper_tail = float(probs[4] + probs[5])
        lower_tail = float(probs[0] + probs[1])
        modal_bucket = int(np.argmax(probs))

        momentum = market_data.get("momentum", {})
        pc1h = momentum.get("price_change_1h", 0.0)
        pc3h = momentum.get("price_change_3h", 0.0)
        intra_vol = momentum.get("intraday_vol", 0.0)
        open_to_now = momentum.get("open_to_now", 0.0)

        avg_spread = float(np.mean(bucket_spreads))
        avg_price = float(np.mean(bucket_prices)) if np.mean(bucket_prices) > 0.01 else 0.5
        avg_norm_spread = avg_spread / avg_price
        imbalance = market_data.get("bid_ask_imbalance", 0.5)
        total_vol = float(sum(bucket_volumes))
    else:
        # No market data — use forecast-derived defaults
        # Uniform-ish distribution centered on forecast
        bucket_prices = [0.05, 0.15, 0.30, 0.30, 0.15, 0.05]
        bucket_spreads = [0.05] * 6
        bucket_volumes = [50.0] * 6

        implied_exp = forecast_mean
        implied_var = 25.0  # wide uncertainty
        implied_skew = 0.0
        implied_kurt = 0.0
        upper_tail = 0.20
        lower_tail = 0.20
        modal_bucket = 2

        pc1h = pc3h = intra_vol = open_to_now = 0.0
        avg_norm_spread = 0.1
        imbalance = 0.5
        total_vol = 300.0

    # Cross-market (use defaults — populated later in batch if desired)
    nb_imp_avg = implied_exp  # no neighbor info yet
    nb_pc_avg = 0.0
    n_nb = float(len(cfg.NEIGHBORS.get(ticker, [])))
    nb_imp_spread = 0.0
    nb_pc_spread = 0.0
    nb_coverage = n_nb / 4.0

    # Divergence
    div_raw = implied_exp - forecast_mean
    div_z = div_raw / 5.0  # approximate z-score

    # Timing
    minutes_to_close = max(0, (21 - local_hour) * 60)
    sin_h, cos_h = fu.sin_cos_encode(np.array([local_hour], dtype=float), 24)
    frac_day = local_hour / 24.0

    # Calendar
    doy = target_date.timetuple().tm_yday
    month = target_date.month
    sin_doy, cos_doy = fu.sin_cos_encode(np.array([doy], dtype=float), 365.25)
    sin_month, cos_month = fu.sin_cos_encode(np.array([month], dtype=float), 12)

    # Static
    static = fu.city_static_features(ticker)

    features = np.array([
        *bucket_prices, *bucket_spreads, *bucket_volumes,  # 18
        implied_exp, implied_var, implied_skew, implied_kurt,
        upper_tail, lower_tail, modal_bucket,               # 7
        pc1h, pc3h, intra_vol, open_to_now,                 # 4
        nb_imp_avg, nb_pc_avg, n_nb,
        nb_imp_spread, nb_pc_spread, nb_coverage,           # 6
        div_raw, div_z,                                      # 2
        avg_norm_spread, imbalance, total_vol,               # 3
        minutes_to_close, sin_h[0], cos_h[0], frac_day,     # 4
        sin_doy[0], cos_doy[0], sin_month[0], cos_month[0], # 4
        static["lat"], static["lon"], static["elevation_ft"],
        static["coastal"], static["desert"], static["continentality"],  # 6
    ], dtype=np.float32)

    return features, city_idx


# ── Model loading ────────────────────────────────────────────────────

def load_models():
    """Load all 3 base models + ensemble weights + scalers."""
    log.info("Loading models from %s...", CHECKPOINT_DIR)

    # Model 1
    scaler1 = fu.ScalerWrapper()
    scaler1.load(os.path.join(CHECKPOINT_DIR, "model1_scaler.pkl"))
    n1 = len(scaler1.columns)
    model1 = training.TemperatureMLP(
        n_continuous=n1, **{k: v for k, v in cfg.MODEL1_HP.items()
                            if k in ["hidden_dims", "dropout", "city_embed_dim"]})
    model1.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "model1_best.pt"),
                                       weights_only=True, map_location=DEVICE))
    model1.to(DEVICE).eval()

    # Model 2
    scaler2 = fu.ScalerWrapper()
    scaler2.load(os.path.join(CHECKPOINT_DIR, "model2_scaler.pkl"))
    n2 = len(scaler2.columns)
    model2 = training.TemperatureMLP(
        n_continuous=n2, **{k: v for k, v in cfg.MODEL2_HP.items()
                            if k in ["hidden_dims", "dropout", "city_embed_dim"]})
    model2.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "model2_best.pt"),
                                       weights_only=True, map_location=DEVICE))
    model2.to(DEVICE).eval()

    # Model 3
    scaler3 = fu.ScalerWrapper()
    scaler3.load(os.path.join(CHECKPOINT_DIR, "model3_scaler.pkl"))
    n3 = len(scaler3.columns)
    model3 = training.TemperatureMLP(
        n_continuous=n3, use_layer_norm=True,
        **{k: v for k, v in cfg.MODEL3_HP.items()
           if k in ["hidden_dims", "dropout", "city_embed_dim"]})
    model3.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "model3_best.pt"),
                                       weights_only=True, map_location=DEVICE))
    model3.to(DEVICE).eval()

    # Ensemble weight network
    from ensemble import DynamicWeightNet, ensemble_predict
    weight_net = DynamicWeightNet()
    weight_net.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "ensemble_weights.pt"),
                                           weights_only=True, map_location=DEVICE))
    weight_net.to(DEVICE).eval()

    log.info("All models loaded. Device: %s", DEVICE)
    return model1, scaler1, model2, scaler2, model3, scaler3, weight_net


# ── Inference for one city ───────────────────────────────────────────

@torch.no_grad()
def predict_city(
    ticker: str, target_date: date,
    model1, scaler1, model2, scaler2, model3, scaler3, weight_net,
    recent_daily: pd.DataFrame,
    all_city_recent: dict,
    forecasts: dict,
    hourly_temps: dict,
    climate: dict,
    market_data: dict | None = None,
    nws_recent: pd.DataFrame | None = None,
    all_city_nws: dict | None = None,
) -> dict:
    """Run all 3 models + ensemble for a single city.
    Predicts the NWS daily recorded high temperature.
    """
    from ensemble import ensemble_predict

    city_name = cfg.CITIES[ticker][0]

    # Model 1
    f1, cidx1, fcst_mean = build_model1_features(ticker, target_date, forecasts, recent_daily, hourly_temps)
    f1_df = pd.DataFrame([f1], columns=scaler1.columns)
    f1_scaled = scaler1.transform(f1_df)[scaler1.columns].values[0]
    x1 = torch.tensor(f1_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    c1 = torch.tensor([cidx1], dtype=torch.long).to(DEVICE)
    mu1_resid, s1 = model1(x1, c1)
    mu1 = fcst_mean + mu1_resid.item()
    s1 = s1.item()

    # Model 2
    f2, cidx2 = build_model2_features(
        ticker, target_date, recent_daily, all_city_recent, climate,
        nws_recent=nws_recent, all_city_nws=all_city_nws,
    )
    f2_df = pd.DataFrame([f2], columns=scaler2.columns)
    f2_scaled = scaler2.transform(f2_df)[scaler2.columns].values[0]
    x2 = torch.tensor(f2_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    c2 = torch.tensor([cidx2], dtype=torch.long).to(DEVICE)
    mu2, s2 = model2(x2, c2)
    mu2, s2 = mu2.item(), s2.item()

    # Model 3
    f3, cidx3 = build_model3_features(ticker, target_date, fcst_mean, market_data)
    f3_df = pd.DataFrame([f3], columns=scaler3.columns)
    f3_scaled = scaler3.transform(f3_df)[scaler3.columns].values[0]
    x3 = torch.tensor(f3_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    c3 = torch.tensor([cidx3], dtype=torch.long).to(DEVICE)
    mu3, s3 = model3(x3, c3)
    mu3, s3 = mu3.item(), s3.item()

    # Ensemble context
    doy = target_date.timetuple().tm_yday
    sin_doy, cos_doy = fu.sin_cos_encode(np.array([doy], dtype=float), 365.25)
    fcst_spread_norm = (s1 - 1.0) / 1.0  # rough normalization

    has_market = 1.0 if (market_data and market_data.get("buckets")) else 0.0
    context = torch.tensor([[
        0.5,               # minutes_to_close_norm
        fcst_spread_norm,  # forecast_spread_norm
        0.0 if not has_market else 0.5,  # market_liquidity_norm
        1.0,               # forecast_available
        has_market,         # market_available
        sin_doy[0], cos_doy[0],
        0.5,               # hours_since_open_norm
    ]], dtype=torch.float32).to(DEVICE)

    t_mu1 = torch.tensor([mu1], dtype=torch.float32).to(DEVICE)
    t_s1 = torch.tensor([s1], dtype=torch.float32).to(DEVICE)
    t_mu2 = torch.tensor([mu2], dtype=torch.float32).to(DEVICE)
    t_s2 = torch.tensor([s2], dtype=torch.float32).to(DEVICE)
    t_mu3 = torch.tensor([mu3], dtype=torch.float32).to(DEVICE)
    t_s3 = torch.tensor([s3], dtype=torch.float32).to(DEVICE)

    import torch.nn.functional as F
    weights = weight_net(context)
    ens_mu, ens_sigma = ensemble_predict(t_mu1, t_s1, t_mu2, t_s2, t_mu3, t_s3, weights)

    ens_mu = ens_mu.item()
    ens_sigma = ens_sigma.item()
    w = weights[0].cpu().numpy()

    # Bucket probabilities
    bucket_edges = [
        (float("-inf"), fcst_mean - 10),
        (fcst_mean - 10, fcst_mean - 5),
        (fcst_mean - 5, fcst_mean),
        (fcst_mean, fcst_mean + 5),
        (fcst_mean + 5, fcst_mean + 10),
        (fcst_mean + 10, float("inf")),
    ]
    bucket_probs = ev.gaussian_bucket_probs(ens_mu, ens_sigma, bucket_edges)
    fair_values = [ev.fair_value_cents(p) for p in bucket_probs]

    return {
        "ticker": ticker,
        "city": city_name,
        "target_date": target_date.isoformat(),
        "forecast_mean": round(fcst_mean, 1),
        "model1_mu": round(mu1, 1),
        "model1_sigma": round(s1, 2),
        "model2_mu": round(mu2, 1),
        "model2_sigma": round(s2, 2),
        "model3_mu": round(mu3, 1),
        "model3_sigma": round(s3, 2),
        "ensemble_mu": round(ens_mu, 1),
        "ensemble_sigma": round(ens_sigma, 2),
        "ci_90_low": round(ens_mu - 1.645 * ens_sigma, 1),
        "ci_90_high": round(ens_mu + 1.645 * ens_sigma, 1),
        "weights": {"forecast": round(w[0], 3), "historical": round(w[1], 3), "market": round(w[2], 3)},
        "bucket_probs": [round(p, 4) for p in bucket_probs],
        "fair_values_cents": fair_values,
    }


# ── Main ─────────────────────────────────────────────────────────────

def run(target_date: date | None = None):
    """Run predictions for all 20 cities."""
    if target_date is None:
        target_date = date.today()

    log.info("Predicting NWS daily recorded high for %s", target_date)

    # Load models
    model1, scaler1, model2, scaler2, model3, scaler3, weight_net = load_models()

    # Fetch climate indices (cached, monthly)
    climate = fetch_climate_indices()
    log.info("Climate indices: %s", climate)

    # Fetch data for all cities
    all_city_recent = {}
    all_city_nws = {}
    results = []

    for ticker, (city_name, tz, lat, lon) in cfg.CITIES.items():
        try:
            log.info("Processing %s...", city_name)

            # Fetch recent daily weather (35 days for lag-28 + rolling-30)
            recent_daily = fetch_recent_daily(lat, lon, tz, n_days=35)
            all_city_recent[ticker] = recent_daily

            # Fetch recent NWS daily highs for lag features
            station_id = cfg.NWS_STATIONS.get(ticker)
            if station_id:
                try:
                    nws_recent = fetch_recent_nws_daily(station_id, n_days=35)
                    all_city_nws[ticker] = nws_recent
                except Exception as e:
                    log.warning("  %s: NWS fetch failed, using Open-Meteo fallback: %s", city_name, e)

            # Fetch yesterday's hourly temps
            hourly_temps = fetch_recent_hourly(lat, lon, tz)

            # Fetch forecasts for target date
            forecasts = fetch_forecasts_for_date(lat, lon, tz, target_date)
            if not forecasts:
                log.warning("  %s: no forecasts available, skipping", city_name)
                continue

            time.sleep(0.5)  # rate limit

        except Exception as e:
            log.error("  %s: data fetch failed: %s", city_name, e)
            continue

    # Second pass: predict (needs all_city_recent for neighbor features)
    for ticker, (city_name, tz, lat, lon) in cfg.CITIES.items():
        if ticker not in all_city_recent:
            continue
        try:
            recent_daily = all_city_recent[ticker]
            nws_recent = all_city_nws.get(ticker)
            hourly_temps = fetch_recent_hourly(lat, lon, tz)
            forecasts = fetch_forecasts_for_date(lat, lon, tz, target_date)
            if not forecasts:
                continue

            result = predict_city(
                ticker, target_date,
                model1, scaler1, model2, scaler2, model3, scaler3, weight_net,
                recent_daily, all_city_recent, forecasts, hourly_temps, climate,
                nws_recent=nws_recent, all_city_nws=all_city_nws,
            )
            results.append(result)

        except Exception as e:
            log.error("  %s: prediction failed: %s", city_name, e)
            import traceback
            traceback.print_exc()
            continue

    # Display results
    if not results:
        log.error("No predictions produced.")
        return []

    print(f"\n{'='*80}")
    print(f"  NWS Daily Recorded High Predictions for {target_date}")
    print(f"{'='*80}\n")
    print(f"{'City':<16} {'Fcst':>6} {'M1':>6} {'M2':>6} {'M3':>6} {'Ensemble':>8} {'90% CI':>14} {'Weights (F/H/M)':>18}")
    print(f"{'-'*16} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*14} {'-'*18}")

    for r in sorted(results, key=lambda x: x["city"]):
        w = r["weights"]
        print(f"{r['city']:<16} {r['forecast_mean']:>5.1f}F {r['model1_mu']:>5.1f} {r['model2_mu']:>5.1f} "
              f"{r['model3_mu']:>5.1f} {r['ensemble_mu']:>7.1f}F  "
              f"[{r['ci_90_low']:>5.1f}, {r['ci_90_high']:>5.1f}]  "
              f"{w['forecast']:.2f}/{w['historical']:.2f}/{w['market']:.2f}")

    print(f"\n{'='*80}\n")

    # Save to JSON
    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return super().default(obj)

    out_path = os.path.join(os.path.dirname(__file__), f"predictions_{target_date}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log.info("Predictions saved to %s", out_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time temperature prediction")
    parser.add_argument("date", nargs="?", default=None, help="Target date (YYYY-MM-DD)")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else None
    run(target)
