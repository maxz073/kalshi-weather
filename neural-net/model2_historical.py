"""
Model 2 — Historical / Regime model.

Predicts the NWS daily recorded high temperature using lagged own-city NWS highs,
neighbor-city NWS highs, climate regime indices (ENSO, AO, NAO, PNA), synoptic
weather lags (from Open-Meteo), and city static features.

Target: NWS daily recorded high (official Kalshi settlement value).
Trained via heteroscedastic Gaussian NLL on the shared TemperatureMLP
architecture (256-128-64, GELU).
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

# Ensure neural-net directory is on the path for sibling imports
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
import data_fetch
import feature_utils as fu
import training
import evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Maximum neighbor slots — cities with fewer neighbors get zero-padded
MAX_NEIGHBORS = 4
NEIGHBOR_FEATS_PER_SLOT = 4  # lag1, lag2, lag3, roll3avg


# ── Feature engineering ─────────────────────────────────────────────

def _build_neighbor_features(archive: pd.DataFrame) -> pd.DataFrame:
    """Build neighbor-city lagged temperature features (uses nws_high).

    For each (date, ticker) row, look up config.NEIGHBORS to find that city's
    neighbors, then attach lag-1, lag-2, lag-3, and 3-day rolling average of
    each neighbor's nws_high.  Cities with fewer than MAX_NEIGHBORS
    neighbors have extra slots zero-filled.  An 'n_neighbors' count feature is
    also added.

    Returns a DataFrame indexed like the input with neighbor feature columns.
    """
    # Pivot to wide: one column per city with nws_high
    wide = archive.pivot_table(
        index="date", columns="ticker", values="nws_high"
    ).sort_index()

    # Pre-compute lag and rolling for every city in one pass
    lag1 = wide.shift(1)
    lag2 = wide.shift(2)
    lag3 = wide.shift(3)
    roll3 = wide.shift(1).rolling(3, min_periods=1).mean()

    records = []
    for _, row in archive[["date", "ticker"]].iterrows():
        dt = row["date"]
        ticker = row["ticker"]
        neighbors = cfg.NEIGHBORS.get(ticker, [])
        rec = {"date": dt, "ticker": ticker, "n_neighbors": len(neighbors)}

        for slot in range(MAX_NEIGHBORS):
            if slot < len(neighbors):
                nb = neighbors[slot]
                rec[f"nb{slot}_lag1"] = lag1.at[dt, nb] if nb in lag1.columns else 0.0
                rec[f"nb{slot}_lag2"] = lag2.at[dt, nb] if nb in lag2.columns else 0.0
                rec[f"nb{slot}_lag3"] = lag3.at[dt, nb] if nb in lag3.columns else 0.0
                rec[f"nb{slot}_roll3avg"] = roll3.at[dt, nb] if nb in roll3.columns else 0.0
            else:
                rec[f"nb{slot}_lag1"] = 0.0
                rec[f"nb{slot}_lag2"] = 0.0
                rec[f"nb{slot}_lag3"] = 0.0
                rec[f"nb{slot}_roll3avg"] = 0.0
        records.append(rec)

    return pd.DataFrame(records)


def _build_neighbor_features_fast(archive: pd.DataFrame) -> pd.DataFrame:
    """Vectorised version of neighbor feature construction (uses nws_high)."""
    wide = archive.pivot_table(
        index="date", columns="ticker", values="nws_high"
    ).sort_index()

    lag1 = wide.shift(1)
    lag2 = wide.shift(2)
    lag3 = wide.shift(3)
    roll3 = wide.shift(1).rolling(3, min_periods=1).mean()

    # Map each (date, ticker) back to the lag tables
    df = archive[["date", "ticker"]].copy()

    for slot in range(MAX_NEIGHBORS):
        df[f"nb{slot}_lag1"] = 0.0
        df[f"nb{slot}_lag2"] = 0.0
        df[f"nb{slot}_lag3"] = 0.0
        df[f"nb{slot}_roll3avg"] = 0.0

    df["n_neighbors"] = df["ticker"].map(lambda t: len(cfg.NEIGHBORS.get(t, [])))

    # Process one city at a time (vectorised within each city)
    for ticker in cfg.CITY_TICKERS:
        mask = df["ticker"] == ticker
        dates = df.loc[mask, "date"]
        neighbors = cfg.NEIGHBORS.get(ticker, [])

        for slot in range(min(len(neighbors), MAX_NEIGHBORS)):
            nb = neighbors[slot]
            if nb not in lag1.columns:
                continue
            # Use reindex to align dates
            df.loc[mask, f"nb{slot}_lag1"] = lag1.loc[dates.values, nb].values
            df.loc[mask, f"nb{slot}_lag2"] = lag2.loc[dates.values, nb].values
            df.loc[mask, f"nb{slot}_lag3"] = lag3.loc[dates.values, nb].values
            df.loc[mask, f"nb{slot}_roll3avg"] = roll3.loc[dates.values, nb].values

    return df


def build_features():
    """Full feature pipeline for Model 2.

    Returns
    -------
    train, val, test : pd.DataFrame
        Each split with features and target column.
    feature_cols : list[str]
        Continuous feature column names (for scaling / model input).
    scaler : fu.ScalerWrapper
        Fitted on training data.
    """
    log.info("Loading archive daily data...")
    archive = data_fetch.load_archive_daily()
    archive["date"] = pd.to_datetime(archive["date"])

    log.info("Loading NWS daily highs (target)...")
    nws = data_fetch.load_nws_daily()
    nws["date"] = pd.to_datetime(nws["date"])
    archive = archive.merge(nws[["date", "ticker", "nws_high"]], on=["date", "ticker"], how="inner")
    archive = archive.sort_values(["ticker", "date"]).reset_index(drop=True)

    log.info("Loading climate indices...")
    climate = data_fetch.load_climate_indices()
    climate["date"] = pd.to_datetime(climate["date"])

    # ── (a) Calendar / seasonality features ──
    df = fu.add_calendar_features(archive.copy())

    # ── (b) Own-city lagged NWS highs ──
    lags = [1, 2, 3, 5, 7, 14, 21, 28]
    df = fu.add_lags(df, "nws_high", lags)

    # ── (c) Own-city rolling stats (on NWS high) ──
    # Rolling mean 3,7,14,30
    df = fu.add_rolling(df, "nws_high", [3, 7, 14, 30], stats=["mean"])
    # Rolling std 7,14
    df = fu.add_rolling(df, "nws_high", [7, 14], stats=["std"])
    # Rolling max/min 7
    df = fu.add_rolling(df, "nws_high", [7], stats=["max", "min"])
    # Rolling mean of temperature_2m_min over 7 days (Open-Meteo, no NWS equivalent)
    df = fu.add_rolling(df, "temperature_2m_min", [7], stats=["mean"])
    # Diurnal range rolling mean over 7 days (NWS high - Open-Meteo low)
    df["diurnal_range"] = df["nws_high"] - df["temperature_2m_min"]
    df = fu.add_rolling(df, "diurnal_range", [7], stats=["mean"])

    # ── (d) Neighbor-city lagged temps ──
    log.info("Building neighbor features (vectorised)...")
    nb_df = _build_neighbor_features_fast(archive)
    # Merge on date + ticker
    df = df.merge(nb_df, on=["date", "ticker"], how="left")

    # ── (e) Climate regime indices ──
    log.info("Merging climate indices...")
    climate_cols = [c for c in climate.columns if c != "date"]
    df = df.merge(climate, on="date", how="left")
    # Forward-fill any remaining NaN in climate columns
    for c in climate_cols:
        df[c] = df[c].ffill()

    # ── (f) Synoptic weather lags ──
    synoptic_vars = [
        "dewpoint_2m_mean", "surface_pressure_mean", "cloud_cover_mean",
        "wind_speed_10m_max", "precipitation_sum", "snowfall_sum",
    ]
    for var in synoptic_vars:
        if var in df.columns:
            df = fu.add_lags(df, var, [1, 2])

    # ── (g) Static city features ──
    df = fu.add_city_static_features(df)

    # ── (h) City index for embedding ──
    df = fu.add_city_index(df)

    # ── Define target (NWS daily recorded high) ──
    df["y"] = df["nws_high"]

    # ── Assemble feature column list ──
    calendar_cols = ["sin_doy", "cos_doy", "sin_month", "cos_month", "sin_woy", "cos_woy"]

    lag_cols = [f"nws_high_lag{l}" for l in lags]

    rolling_cols = (
        [f"nws_high_roll{w}_mean" for w in [3, 7, 14, 30]]
        + [f"nws_high_roll{w}_std" for w in [7, 14]]
        + ["nws_high_roll7_max", "nws_high_roll7_min"]
        + ["temperature_2m_min_roll7_mean"]
        + ["diurnal_range_roll7_mean"]
    )

    neighbor_cols = ["n_neighbors"]
    for slot in range(MAX_NEIGHBORS):
        neighbor_cols += [
            f"nb{slot}_lag1", f"nb{slot}_lag2", f"nb{slot}_lag3", f"nb{slot}_roll3avg",
        ]

    synoptic_lag_cols = []
    for var in synoptic_vars:
        for lag in [1, 2]:
            col_name = f"{var}_lag{lag}"
            if col_name in df.columns:
                synoptic_lag_cols.append(col_name)

    static_cols = ["lat", "lon", "elevation_ft", "coastal", "desert", "continentality"]

    feature_cols = (
        calendar_cols + lag_cols + rolling_cols + neighbor_cols
        + climate_cols + synoptic_lag_cols + static_cols
    )

    log.info("Total continuous features: %d", len(feature_cols))

    # ── Drop rows with NaN in any feature or target ──
    subset = feature_cols + ["y"]
    before = len(df)
    df = df.dropna(subset=subset).reset_index(drop=True)
    log.info("Dropped %d rows with NaN (%.1f%%)", before - len(df), 100 * (before - len(df)) / before)

    # ── Train / val / test split ──
    train, val, test = fu.split_data(df)
    log.info("Split sizes — train: %d  val: %d  test: %d", len(train), len(val), len(test))

    # ── Scale continuous features (fit on train) ──
    scaler = fu.ScalerWrapper()
    train = scaler.fit_transform(train, feature_cols)
    val = scaler.transform(val)
    test = scaler.transform(test)

    return train, val, test, feature_cols, scaler


# ── Training ────────────────────────────────────────────────────────

def train():
    """Build features, train Model 2, and evaluate on test set."""
    train_df, val_df, test_df, feature_cols, scaler = build_features()
    hp = cfg.MODEL2_HP

    # Create datasets
    train_ds = training.make_dataset(
        train_df[feature_cols].values,
        train_df["city_idx"].values,
        train_df["y"].values,
    )
    val_ds = training.make_dataset(
        val_df[feature_cols].values,
        val_df["city_idx"].values,
        val_df["y"].values,
    )
    test_ds = training.make_dataset(
        test_df[feature_cols].values,
        test_df["city_idx"].values,
        test_df["y"].values,
    )

    train_loader = training.make_loader(train_ds, hp["batch_size"], shuffle=True)
    val_loader = training.make_loader(val_ds, hp["batch_size"], shuffle=False)
    test_loader = training.make_loader(test_ds, hp["batch_size"], shuffle=False)

    # Build model
    model = training.TemperatureMLP(
        n_continuous=len(feature_cols),
        n_cities=cfg.N_CITIES,
        city_embed_dim=hp["city_embed_dim"],
        hidden_dims=hp["hidden_dims"],
        dropout=hp["dropout"],
    )
    log.info("Model 2 architecture:\n%s", model)

    # Train
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "model2_best.pt")
    history = training.train_model(model, train_loader, val_loader, hp, ckpt_path)

    # Save scaler alongside checkpoint
    scaler.save(os.path.join(cfg.CHECKPOINT_DIR, "model2_scaler.pkl"))

    # ── Evaluate on test set ──
    mu, sigma = training.predict(model, test_loader)
    y_true = test_df["y"].values

    metrics = evaluation.compute_metrics(y_true, mu)
    log.info("Model 2 — Test metrics:")
    for k, v in metrics.items():
        log.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    # Per-city breakdown
    test_df = test_df.copy()
    test_df["y_pred"] = mu
    test_df["sigma"] = sigma
    city_metrics = evaluation.metrics_by_city(test_df, "y", "y_pred")
    log.info("Per-city MAE:\n%s", city_metrics[["city", "mae", "rmse", "bias"]].to_string(index=False))

    # Calibration
    cal = evaluation.calibration_check(y_true, mu, sigma)
    log.info("Calibration:\n%s", cal.to_string(index=False))

    # ── Save val + test predictions for ensemble ──
    for split_name, split_loader, split_df_raw in [
        ("val", val_loader, val_df),
        ("test", test_loader, test_df),
    ]:
        mu_s, sigma_s = training.predict(model, split_loader)
        pred_df = pd.DataFrame({
            "date": split_df_raw["date"].values,
            "ticker": split_df_raw["ticker"].values,
            "mu": mu_s,
            "sigma": sigma_s,
            "y_true": split_df_raw["y"].values,
        })
        path = os.path.join(cfg.CHECKPOINT_DIR, f"model2_preds_{split_name}.csv")
        pred_df.to_csv(path, index=False)
        log.info("Saved %s predictions to %s", split_name, path)

    return model, history, test_df, scaler


# ── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
