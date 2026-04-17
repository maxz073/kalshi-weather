"""
Model 1 — Forecast Bias-Correction.

Predicts the residual (nws_recorded_high - forecast_mean) using:
  - Raw forecast model outputs + ensemble statistics
  - Meteorological proxy features from archive data
  - Hourly temperature path features
  - Rolling historical forecast bias per city
  - City static features + calendar encoding
  - City embedding

Target: NWS daily recorded high (official Kalshi settlement value).
At inference time: predicted_actual = forecast_mean + predicted_residual
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

# Shared modules
import config as cfg
import data_fetch
import feature_utils
import training
import evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FORECAST_COLS = [
    "fcst_gfs_seamless",
    "fcst_ecmwf_ifs025",
    "fcst_icon_seamless",
    "fcst_gem_seamless",
    "fcst_jma_seamless",
]


# ── Feature engineering ─────────────────────────────────────────────

def _build_hourly_temp_path(hourly: pd.DataFrame) -> pd.DataFrame:
    """Extract per-city-day temperature at 6am, 9am, noon, 3pm and diurnal range."""
    log.info("Building hourly temperature path features...")
    df = hourly.copy()
    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.normalize()

    # Filter to hours of interest
    target_hours = [6, 9, 12, 15]
    df = df[df["hour"].isin(target_hours)]

    # Pivot: one column per hour
    pivot = df.pivot_table(
        index=["date", "ticker"],
        columns="hour",
        values="temperature_2m",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot.rename(
        columns={6: "temp_6am", 9: "temp_9am", 12: "temp_noon", 15: "temp_3pm"},
        inplace=True,
    )

    # Diurnal range from the four sampled hours
    hour_cols = ["temp_6am", "temp_9am", "temp_noon", "temp_3pm"]
    pivot["temp_path_range"] = pivot[hour_cols].max(axis=1) - pivot[hour_cols].min(axis=1)

    return pivot


def build_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build the full feature matrix for Model 1.

    Returns
    -------
    train, val, test : pd.DataFrame
        Each contains continuous features, city_idx, y_resid, date, ticker, forecast_mean.
    cont_cols : list[str]
        Names of continuous feature columns (everything the scaler should touch).
    """
    log.info("=== Model 1: Building features ===")

    # ── Load raw data ───────────────────────────────────────────────
    archive_daily = data_fetch.load_archive_daily()
    archive_hourly = data_fetch.load_archive_hourly()
    forecasts = data_fetch.load_forecasts()
    nws_daily = data_fetch.load_nws_daily()
    log.info("Loaded archive_daily=%d, archive_hourly=%d, forecasts=%d, nws_daily=%d",
             len(archive_daily), len(archive_hourly), len(forecasts), len(nws_daily))

    # ── Merge forecasts with archive (features) and NWS (target) ────
    df = forecasts.merge(archive_daily, on=["date", "ticker"], how="inner")
    df = df.merge(nws_daily[["date", "ticker", "nws_high"]], on=["date", "ticker"], how="inner")
    log.info("After merge forecast+archive+nws: %d rows", len(df))

    # ── (a) Forecast ensemble statistics ────────────────────────────
    fcst_values = df[FORECAST_COLS]
    # Fill per-city mean for any NaN forecast columns
    for col in FORECAST_COLS:
        city_means = df.groupby("ticker")[col].transform("mean")
        df[col] = df[col].fillna(city_means)

    fcst_values = df[FORECAST_COLS]
    df["forecast_mean"] = fcst_values.mean(axis=1)
    df["forecast_std"] = fcst_values.std(axis=1)
    df["forecast_range"] = fcst_values.max(axis=1) - fcst_values.min(axis=1)
    # IQR (75th - 25th percentile across 5 models)
    df["forecast_iqr"] = fcst_values.quantile(0.75, axis=1) - fcst_values.quantile(0.25, axis=1)

    # ── (b) Target: residual (NWS recorded high minus forecast mean) ─
    df["y_resid"] = df["nws_high"] - df["forecast_mean"]

    # ── (c) Pairwise spreads ────────────────────────────────────────
    df["spread_ecmwf_gfs"] = df["fcst_ecmwf_ifs025"] - df["fcst_gfs_seamless"]
    df["spread_icon_gfs"] = df["fcst_icon_seamless"] - df["fcst_gfs_seamless"]
    df["spread_gem_gfs"] = df["fcst_gem_seamless"] - df["fcst_gfs_seamless"]
    df["spread_jma_gfs"] = df["fcst_jma_seamless"] - df["fcst_gfs_seamless"]

    # ── (d) Lagged meteorological features (yesterday's weather) ─────
    # IMPORTANT: Same-day archive values (cloud_cover_mean, etc.) are
    # end-of-day aggregates — NOT available at trading time. Use lag-1
    # (yesterday's values) to avoid look-ahead bias.
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    meteo_raw = [
        "cloud_cover_mean", "dewpoint_2m_mean", "wind_speed_10m_max",
        "surface_pressure_mean", "precipitation_sum",
    ]
    for col in meteo_raw:
        df[f"{col}_lag1"] = df.groupby("ticker")[col].shift(1)
    meteo_cols = [f"{c}_lag1" for c in meteo_raw]

    # ── (e) Lagged temperature path (yesterday's hourly temps) ──────
    # IMPORTANT: Same-day hourly temps (especially noon/3pm) leak the
    # outcome. Use yesterday's hourly path + yesterday's overnight low
    # as proxy for current thermal regime.
    hourly_path = _build_hourly_temp_path(archive_hourly)
    hourly_path_cols = ["temp_6am", "temp_9am", "temp_noon", "temp_3pm", "temp_path_range"]
    hourly_path = hourly_path.rename(columns={c: f"{c}_lag1" for c in hourly_path_cols})
    df = df.merge(hourly_path, on=["date", "ticker"], how="left")
    # Shift the merged columns to get yesterday's values
    for col in [f"{c}_lag1" for c in hourly_path_cols]:
        df[col] = df.groupby("ticker")[col].shift(1)
    # Yesterday's overnight low
    df["temperature_2m_min_lag1"] = df.groupby("ticker")["temperature_2m_min"].shift(1)

    # ── (f) Rolling forecast bias (shifted by 1 to avoid leaking) ──
    df = feature_utils.add_rolling(
        df, col="y_resid", windows=[7, 14, 30],
        stats=["mean"], group_col="ticker",
    )

    # ── (g) City static features ────────────────────────────────────
    df = feature_utils.add_city_static_features(df)

    # ── (h) Calendar features ───────────────────────────────────────
    df = feature_utils.add_calendar_features(df)

    # ── (i) City index for embedding ────────────────────────────────
    df = feature_utils.add_city_index(df)

    # ── Drop rows where NWS high is NaN ─────────────────────────────
    before = len(df)
    df = df.dropna(subset=["nws_high"]).reset_index(drop=True)
    log.info("Dropped %d rows with NaN NWS high (remaining: %d)", before - len(df), len(df))

    # ── Define continuous feature columns ───────────────────────────
    # Forecast highs
    cont_cols = list(FORECAST_COLS)
    # Ensemble stats
    cont_cols += ["forecast_mean", "forecast_std", "forecast_range", "forecast_iqr"]
    # Pairwise spreads
    cont_cols += ["spread_ecmwf_gfs", "spread_icon_gfs", "spread_gem_gfs", "spread_jma_gfs"]
    # Lagged meteorological (yesterday's weather)
    cont_cols += meteo_cols
    # Lagged hourly path (yesterday's temps)
    cont_cols += [f"{c}_lag1" for c in ["temp_6am", "temp_9am", "temp_noon", "temp_3pm", "temp_path_range"]]
    # Yesterday's overnight low
    cont_cols += ["temperature_2m_min_lag1"]
    # Rolling bias
    cont_cols += ["y_resid_roll7_mean", "y_resid_roll14_mean", "y_resid_roll30_mean"]
    # City static
    cont_cols += ["lat", "lon", "elevation_ft", "coastal", "continentality"]
    # Calendar
    cont_cols += ["sin_doy", "cos_doy", "sin_month", "cos_month"]

    # Fill remaining NaNs in continuous features with 0 (rolling features will have NaN at start)
    df[cont_cols] = df[cont_cols].fillna(0.0)

    log.info("Total continuous features: %d", len(cont_cols))

    # ── Split ───────────────────────────────────────────────────────
    train, val, test = feature_utils.split_data(df)
    log.info("Split sizes — train: %d, val: %d, test: %d", len(train), len(val), len(test))

    return train, val, test, cont_cols


# ── Training ────────────────────────────────────────────────────────

def train():
    """Build features, train Model 1, evaluate on test set."""
    train_df, val_df, test_df, cont_cols = build_features()

    # Save unscaled values needed for evaluation before scaling
    val_forecast_mean_raw = val_df["forecast_mean"].values.copy()
    val_actual_temp_raw = val_df["nws_high"].values.copy()
    test_forecast_mean_raw = test_df["forecast_mean"].values.copy()
    test_actual_temp_raw = test_df["nws_high"].values.copy()

    # ── Scale continuous features (fit on train only) ───────────────
    scaler = feature_utils.ScalerWrapper()
    train_df = scaler.fit_transform(train_df, cont_cols)
    val_df = scaler.transform(val_df)
    test_df = scaler.transform(test_df)

    # Save scaler for later inference
    scaler_path = os.path.join(cfg.CHECKPOINT_DIR, "model1_scaler.pkl")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    scaler.save(scaler_path)
    log.info("Scaler saved to %s", scaler_path)

    # ── Create TensorDatasets ───────────────────────────────────────
    train_ds = training.make_dataset(
        train_df[cont_cols].values,
        train_df["city_idx"].values,
        train_df["y_resid"].values,
    )
    val_ds = training.make_dataset(
        val_df[cont_cols].values,
        val_df["city_idx"].values,
        val_df["y_resid"].values,
    )
    test_ds = training.make_dataset(
        test_df[cont_cols].values,
        test_df["city_idx"].values,
        test_df["y_resid"].values,
    )

    hp = cfg.MODEL1_HP
    train_loader = training.make_loader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_loader = training.make_loader(val_ds, batch_size=hp["batch_size"], shuffle=False)
    test_loader = training.make_loader(test_ds, batch_size=hp["batch_size"], shuffle=False)

    # ── Build model ─────────────────────────────────────────────────
    model = training.TemperatureMLP(
        n_continuous=len(cont_cols),
        n_cities=cfg.N_CITIES,
        city_embed_dim=hp["city_embed_dim"],
        hidden_dims=hp["hidden_dims"],
        dropout=hp["dropout"],
    )
    log.info("Model 1 architecture:\n%s", model)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Total parameters: %d", n_params)

    # ── Train ───────────────────────────────────────────────────────
    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "model1_best.pt")
    history = training.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        hp=hp,
        checkpoint_path=checkpoint_path,
    )

    # ── Evaluate on test set ────────────────────────────────────────
    log.info("=== Model 1: Test set evaluation ===")
    mu_resid, sigma_resid = training.predict(model, test_loader)

    # Convert residual predictions back to actual temperature
    # Use the raw (unscaled) forecast_mean and actual temp saved before scaling
    test_df = test_df.copy()
    test_df["pred_resid"] = mu_resid
    test_df["pred_sigma"] = sigma_resid
    test_df["pred_temp"] = test_forecast_mean_raw + mu_resid
    test_df["actual_temp"] = test_actual_temp_raw

    # Overall metrics
    overall = evaluation.compute_metrics(test_df["actual_temp"].values, test_df["pred_temp"].values)
    log.info("Overall test metrics:")
    for k, v in overall.items():
        log.info("  %s: %.4f", k, v)

    # Per-city metrics
    city_metrics = evaluation.metrics_by_city(test_df, "actual_temp", "pred_temp")
    log.info("Per-city test MAE:\n%s", city_metrics[["city", "mae", "rmse", "bias"]].to_string(index=False))

    # Calibration check on residual
    cal = evaluation.calibration_check(test_df["y_resid"].values, mu_resid, sigma_resid)
    log.info("Calibration:\n%s", cal.to_string(index=False))

    # Residual metrics (what the model directly predicts)
    resid_metrics = evaluation.compute_metrics(test_df["y_resid"].values, mu_resid)
    log.info("Residual prediction metrics:")
    for k, v in resid_metrics.items():
        log.info("  %s: %.4f", k, v)

    # ── Save val + test predictions for ensemble ───────────────────
    # Val predictions
    val_loader_eval = training.make_loader(
        training.make_dataset(val_df[cont_cols].values, val_df["city_idx"].values, val_df["y_resid"].values),
        batch_size=cfg.MODEL1_HP["batch_size"], shuffle=False)
    mu_val_r, sigma_val_r = training.predict(model, val_loader_eval)
    for split_name, split_df, mu_r, sigma_r, fcst_raw, actual_raw in [
        ("val", val_df, mu_val_r, sigma_val_r, val_forecast_mean_raw, val_actual_temp_raw),
        ("test", test_df, mu_resid, sigma_resid, test_forecast_mean_raw, test_actual_temp_raw),
    ]:
        pred_df = pd.DataFrame({
            "date": split_df["date"].values,
            "ticker": split_df["ticker"].values,
            "mu": fcst_raw + mu_r,  # convert residual back to temp
            "sigma": sigma_r,
            "y_true": actual_raw,
        })
        path = os.path.join(cfg.CHECKPOINT_DIR, f"model1_preds_{split_name}.csv")
        pred_df.to_csv(path, index=False)
        log.info("Saved %s predictions to %s", split_name, path)

    return model, history, test_df


# ── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
