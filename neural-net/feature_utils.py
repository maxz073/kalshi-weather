"""
Shared feature engineering utilities.
"""
import math
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as cfg


# ── Cyclical encoding ────────────────────────────────────────────────

def sin_cos_encode(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Encode a cyclical feature as (sin, cos) pair."""
    angle = 2 * math.pi * values / period
    return np.sin(angle), np.cos(angle)


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add sin/cos day-of-year, month, week-of-year features."""
    doy = df[date_col].dt.dayofyear.values.astype(float)
    month = df[date_col].dt.month.values.astype(float)
    woy = df[date_col].dt.isocalendar().week.values.astype(float)

    df["sin_doy"], df["cos_doy"] = sin_cos_encode(doy, 365.25)
    df["sin_month"], df["cos_month"] = sin_cos_encode(month, 12)
    df["sin_woy"], df["cos_woy"] = sin_cos_encode(woy, 52)
    return df


# ── City features ────────────────────────────────────────────────────

def city_static_features(ticker: str) -> dict:
    """Return static metadata for a city."""
    meta = cfg.CITY_META[ticker]
    _, _, lat, lon = cfg.CITIES[ticker]
    return {
        "lat": lat,
        "lon": lon,
        "elevation_ft": meta["elevation_ft"],
        "coastal": float(meta["coastal"]),
        "desert": float(meta["desert"]),
        "continentality": meta["continentality"],
    }


def add_city_static_features(df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
    """Add static city features (lat, lon, elevation, coastal, desert, continentality)."""
    static_rows = []
    for ticker in df[ticker_col]:
        static_rows.append(city_static_features(ticker))
    static_df = pd.DataFrame(static_rows, index=df.index)
    return pd.concat([df, static_df], axis=1)


def add_city_index(df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
    """Add integer city index for embedding lookup."""
    df["city_idx"] = df[ticker_col].map(cfg.TICKER_TO_IDX)
    return df


# ── Scaler wrapper ───────────────────────────────────────────────────

class ScalerWrapper:
    """Wraps sklearn StandardScaler with save/load and train-only fitting."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.columns = None

    def fit(self, df: pd.DataFrame, columns: list[str]):
        self.columns = columns
        self.scaler.fit(df[columns].values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.columns] = self.scaler.transform(df[self.columns].values)
        return df

    def fit_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        self.fit(df, columns)
        return self.transform(df)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "columns": self.columns}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.scaler = d["scaler"]
        self.columns = d["columns"]
        return self


# ── Climatological normals ───────────────────────────────────────────

def compute_climatological_normals(df: pd.DataFrame, train_end: pd.Timestamp,
                                    temp_col: str = "nws_high",
                                    ticker_col: str = "ticker",
                                    date_col: str = "date",
                                    smooth_window: int = 15) -> pd.DataFrame:
    """Compute per-city per-DOY smoothed average temperature from training data."""
    train = df[df[date_col] <= train_end].copy()
    train["doy"] = train[date_col].dt.dayofyear

    normals = train.groupby([ticker_col, "doy"])[temp_col].mean().reset_index()
    normals.rename(columns={temp_col: "clim_normal"}, inplace=True)

    # Smooth with rolling window (circular for DOY wrap-around)
    smoothed = []
    for ticker in normals[ticker_col].unique():
        city_normals = normals[normals[ticker_col] == ticker].sort_values("doy")
        # Pad for circular rolling
        padded = pd.concat([city_normals.tail(smooth_window),
                            city_normals,
                            city_normals.head(smooth_window)])
        padded["clim_normal"] = padded["clim_normal"].rolling(smooth_window, center=True).mean()
        smoothed.append(padded.iloc[smooth_window:-smooth_window])

    return pd.concat(smoothed, ignore_index=True)


# ── Lag / rolling helpers ────────────────────────────────────────────

def add_lags(df: pd.DataFrame, col: str, lags: list[int],
             group_col: str = "ticker") -> pd.DataFrame:
    """Add lagged columns grouped by city."""
    for lag in lags:
        df[f"{col}_lag{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df


def add_rolling(df: pd.DataFrame, col: str, windows: list[int],
                stats: list[str] = None, group_col: str = "ticker") -> pd.DataFrame:
    """Add rolling statistics grouped by city. Stats: 'mean', 'std', 'max', 'min'."""
    if stats is None:
        stats = ["mean"]
    for w in windows:
        grouped = df.groupby(group_col)[col]
        for stat in stats:
            shifted = grouped.shift(1)  # avoid leaking current day
            rolling = shifted.rolling(w, min_periods=1)
            df[f"{col}_roll{w}_{stat}"] = getattr(rolling, stat)()
    return df


# ── Train/val/test split ─────────────────────────────────────────────

def split_data(df: pd.DataFrame, date_col: str = "date"):
    """Split DataFrame into train/val/test by date."""
    train = df[(df[date_col] >= pd.Timestamp(cfg.TRAIN_START)) &
               (df[date_col] <= pd.Timestamp(cfg.TRAIN_END))].copy()
    val = df[(df[date_col] >= pd.Timestamp(cfg.VAL_START)) &
             (df[date_col] <= pd.Timestamp(cfg.VAL_END))].copy()
    test = df[(df[date_col] >= pd.Timestamp(cfg.TEST_START)) &
              (df[date_col] <= pd.Timestamp(cfg.TEST_END))].copy()
    return train, val, test
