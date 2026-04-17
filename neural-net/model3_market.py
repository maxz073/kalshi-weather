"""
Model 3 -- Market / Microstructure.

Predicts the NWS daily recorded high temperature directly from synthetic (or
future real) Kalshi market microstructure data.  Because Kalshi exposes no
historical market-price API, we *generate* synthetic market snapshots for every
city-day in the 2022-2026 window, reverse-engineering plausible bucket prices,
spreads, volumes, and momentum from the NWS recorded high + forecast data.

Target: NWS daily recorded high (official Kalshi settlement value).

Feature groups (~65 continuous + city embedding):
  (a) Per-bucket features (18): mid_price, spread, volume for 6 buckets
  (b) Implied distribution (7)
  (c) Market momentum (4)
  (d) Cross-market / neighbor (6)
  (e) Forecast-vs-market divergence (2)
  (f) Liquidity (3)
  (g) Timing (4)
  (h) Calendar (4)
  (i) City static (6)
  (j) City index for embedding

Architecture: TemperatureMLP with LayerNorm (config.MODEL3_HP, use_layer_norm=True).
"""
import os
import sys
import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm

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

MARKET_DIR = os.path.join(cfg.DATA_DIR, "kalshi_markets")
SYNTHETIC_CSV = os.path.join(MARKET_DIR, "synthetic_markets.csv")

N_BUCKETS = 6
# Bucket offsets relative to forecast mean (in Fahrenheit)
BUCKET_OFFSETS = [
    (-np.inf, -10),
    (-10, -5),
    (-5, 0),
    (0, 5),
    (5, 10),
    (10, np.inf),
]


# ── Synthetic market data generation ───────────────────────────────

def _bucket_edges_for_forecast(forecast_mean: float) -> list[tuple[float, float]]:
    """Return absolute (low, high) temperature edges for the 6 buckets."""
    edges = []
    for lo, hi in BUCKET_OFFSETS:
        abs_lo = -999.0 if np.isinf(lo) and lo < 0 else forecast_mean + lo
        abs_hi = 999.0 if np.isinf(hi) and hi > 0 else forecast_mean + hi
        edges.append((abs_lo, abs_hi))
    return edges


def _bucket_midpoints(forecast_mean: float) -> np.ndarray:
    """Return midpoint temperatures for the 6 buckets (tails clamped)."""
    mids = []
    for lo, hi in BUCKET_OFFSETS:
        if np.isinf(lo):
            mids.append(forecast_mean + (-10 + (-5)) / 2)  # -12.5 offset
        elif np.isinf(hi):
            mids.append(forecast_mean + (10 + 5) / 2)       # +12.5 offset
        else:
            mids.append(forecast_mean + (lo + hi) / 2)
    return np.array(mids)


def generate_synthetic_markets(force: bool = False) -> pd.DataFrame:
    """
    Generate synthetic Kalshi-like market snapshots for every city-day.

    Uses actual temperature + forecast data to reverse-engineer plausible
    bucket prices, spreads, volumes, and momentum.  Deterministic via seed.
    """
    if os.path.exists(SYNTHETIC_CSV) and not force:
        log.info("Synthetic market data already exists at %s", SYNTHETIC_CSV)
        return pd.read_csv(SYNTHETIC_CSV, parse_dates=["date"])

    log.info("=== Generating synthetic market data ===")
    np.random.seed(42)

    # Load raw data
    archive_daily = data_fetch.load_archive_daily()
    forecasts = data_fetch.load_forecasts()
    nws_daily = data_fetch.load_nws_daily()

    # Merge on (date, ticker) — use NWS recorded high as the actual temperature
    df = forecasts.merge(nws_daily[["date", "ticker", "nws_high"]],
                         on=["date", "ticker"], how="inner")
    log.info("Merged rows for synthetic generation: %d", len(df))

    # Forecast mean across available models
    fcst_cols = [c for c in df.columns if c.startswith("fcst_")]
    df["forecast_mean"] = df[fcst_cols].mean(axis=1)
    # Fill NaN forecasts with column mean per city
    for col in fcst_cols:
        city_means = df.groupby("ticker")[col].transform("mean")
        df[col] = df[col].fillna(city_means)
    df["forecast_mean"] = df[fcst_cols].mean(axis=1)

    actual = df["nws_high"].values
    fcst_mean = df["forecast_mean"].values
    tickers = df["ticker"].values
    dates = df["date"].values
    n = len(df)

    # ── Per-row synthetic generation ─────────────────────────────────
    # Pre-allocate output arrays
    # Per-bucket: mid_price, spread, volume (6 * 3 = 18)
    bucket_prices = np.zeros((n, N_BUCKETS))
    bucket_spreads = np.zeros((n, N_BUCKETS))
    bucket_volumes = np.zeros((n, N_BUCKETS))

    # Implied distribution stats
    implied_exp = np.zeros(n)
    implied_var = np.zeros(n)
    implied_skew = np.zeros(n)
    implied_kurt = np.zeros(n)
    upper_tail = np.zeros(n)
    lower_tail = np.zeros(n)
    modal_bucket = np.zeros(n, dtype=int)

    # Momentum
    price_change_1h = np.zeros(n)
    price_change_3h = np.zeros(n)
    intraday_vol = np.zeros(n)
    open_to_now = np.zeros(n)

    # Liquidity
    avg_norm_spread = np.zeros(n)
    bid_ask_imbalance = np.zeros(n)
    total_volume = np.zeros(n)

    # Timing
    snapshot_hour = np.zeros(n)
    minutes_to_close = np.zeros(n)

    # Divergence
    divergence_raw = np.zeros(n)

    for i in range(n):
        fm = fcst_mean[i]
        act = actual[i]

        # (a) Simulate market distribution centered on FORECAST (not actual)
        # Real markets are informed by forecasts + trader views, NOT the outcome.
        # Add random offset to simulate market disagreement with forecasts.
        market_center = fm + np.random.normal(0, 2.0)  # market deviates from forecast
        market_std = np.random.uniform(3.0, 6.0)
        edges = _bucket_edges_for_forecast(fm)

        market_probs = np.array([
            sp_norm.cdf(hi, loc=market_center, scale=market_std) -
            sp_norm.cdf(lo, loc=market_center, scale=market_std)
            for lo, hi in edges
        ])
        market_probs = np.clip(market_probs, 0.001, None)
        market_probs /= market_probs.sum()  # normalize

        # (b) Market prices = market_prob + noise
        noise = np.random.uniform(-0.05, 0.05, N_BUCKETS)
        mkt_prices = np.clip(market_probs + noise, 0.01, 0.99)
        bucket_prices[i] = mkt_prices

        # (c) Bid-ask spread (wider for tail buckets)
        base_spread = np.random.uniform(0.02, 0.08, N_BUCKETS)
        # Tails get wider spreads
        base_spread[0] *= 1.3
        base_spread[-1] *= 1.3
        bucket_spreads[i] = base_spread

        # (d) Volume (higher for center buckets)
        center_boost = np.array([0.3, 0.7, 1.0, 1.0, 0.7, 0.3])
        vol = np.random.uniform(10, 500, N_BUCKETS) * center_boost
        bucket_volumes[i] = vol

        # (e) Implied distribution stats from market prices
        mids = _bucket_midpoints(fm)
        # Normalize market prices to probabilities
        mkt_prob = mkt_prices / mkt_prices.sum()
        imp_mean = np.dot(mkt_prob, mids)
        implied_exp[i] = imp_mean

        imp_var = np.dot(mkt_prob, (mids - imp_mean) ** 2)
        implied_var[i] = imp_var

        # Skewness
        if imp_var > 0:
            imp_std = np.sqrt(imp_var)
            implied_skew[i] = np.dot(mkt_prob, ((mids - imp_mean) / imp_std) ** 3)
            implied_kurt[i] = np.dot(mkt_prob, ((mids - imp_mean) / imp_std) ** 4) - 3.0
        else:
            implied_skew[i] = 0.0
            implied_kurt[i] = 0.0

        # Tail probabilities
        upper_tail[i] = mkt_prob[4] + mkt_prob[5]  # above forecast+5
        lower_tail[i] = mkt_prob[0] + mkt_prob[1]  # below forecast-5
        modal_bucket[i] = int(np.argmax(mkt_prob))

        # (f) Momentum: small random walk changes
        price_change_1h[i] = np.random.normal(0, 0.01)
        price_change_3h[i] = np.random.normal(0, 0.02)
        intraday_vol[i] = np.abs(np.random.normal(0, 0.015))
        open_to_now[i] = np.random.normal(0, 0.025)

        # (g) Liquidity
        avg_norm_spread[i] = bucket_spreads[i].mean() / np.clip(mkt_prices.mean(), 0.01, None)
        bid_ask_imbalance[i] = np.random.uniform(0.3, 0.7)
        total_volume[i] = vol.sum()

        # (h) Timing: snapshot hour between 10 and 20 local
        snap_h = np.random.uniform(10, 20)
        snapshot_hour[i] = snap_h
        minutes_to_close[i] = max(0, (21.0 - snap_h) * 60)

        # (i) Divergence
        divergence_raw[i] = imp_mean - fm

    # Z-score the divergence
    div_mean = divergence_raw.mean()
    div_std = divergence_raw.std()
    divergence_zscore = (divergence_raw - div_mean) / (div_std + 1e-8)

    # ── Cross-market / neighbor features ─────────────────────────────
    # Build a lookup: (ticker, date) -> implied_exp and price_change_1h
    log.info("Computing cross-market neighbor features...")
    lookup_imp = {}
    lookup_pc1h = {}
    for i in range(n):
        key = (tickers[i], dates[i])
        lookup_imp[key] = implied_exp[i]
        lookup_pc1h[key] = price_change_1h[i]

    max_neighbors = 4
    neighbor_imp_avg = np.zeros(n)
    neighbor_pc_avg = np.zeros(n)
    n_neighbors = np.zeros(n)

    for i in range(n):
        tk = tickers[i]
        dt = dates[i]
        nbrs = cfg.NEIGHBORS.get(tk, [])[:max_neighbors]
        imp_vals = []
        pc_vals = []
        for nb in nbrs:
            key = (nb, dt)
            if key in lookup_imp:
                imp_vals.append(lookup_imp[key])
                pc_vals.append(lookup_pc1h[key])
        if imp_vals:
            neighbor_imp_avg[i] = np.mean(imp_vals)
            neighbor_pc_avg[i] = np.mean(pc_vals)
            n_neighbors[i] = len(imp_vals)

    # ── Assemble output DataFrame ────────────────────────────────────
    log.info("Assembling synthetic market DataFrame...")
    out = pd.DataFrame({
        "date": dates,
        "ticker": tickers,
        "nws_high": actual,
        "forecast_mean": fcst_mean,
    })

    # Per-bucket features
    for b in range(N_BUCKETS):
        out[f"bucket_{b}_price"] = bucket_prices[:, b]
        out[f"bucket_{b}_spread"] = bucket_spreads[:, b]
        out[f"bucket_{b}_volume"] = bucket_volumes[:, b]

    # Implied distribution
    out["implied_expected_temp"] = implied_exp
    out["implied_variance"] = implied_var
    out["implied_skew"] = implied_skew
    out["implied_kurtosis"] = implied_kurt
    out["upper_tail_prob"] = upper_tail
    out["lower_tail_prob"] = lower_tail
    out["modal_bucket_idx"] = modal_bucket

    # Momentum
    out["price_change_1h"] = price_change_1h
    out["price_change_3h"] = price_change_3h
    out["intraday_vol"] = intraday_vol
    out["open_to_now_change"] = open_to_now

    # Cross-market
    out["neighbor_imp_avg"] = neighbor_imp_avg
    out["neighbor_pc_avg"] = neighbor_pc_avg
    out["n_neighbors"] = n_neighbors

    # Divergence
    out["divergence_raw"] = divergence_raw
    out["divergence_zscore"] = divergence_zscore

    # Liquidity
    out["avg_normalized_spread"] = avg_norm_spread
    out["bid_ask_imbalance"] = bid_ask_imbalance
    out["total_volume"] = total_volume

    # Timing
    out["snapshot_hour"] = snapshot_hour
    out["minutes_to_close"] = minutes_to_close
    sin_hour = np.sin(2 * math.pi * snapshot_hour / 24)
    cos_hour = np.cos(2 * math.pi * snapshot_hour / 24)
    out["sin_hour"] = sin_hour
    out["cos_hour"] = cos_hour
    out["fraction_of_day"] = snapshot_hour / 24.0

    # Save
    os.makedirs(MARKET_DIR, exist_ok=True)
    out.to_csv(SYNTHETIC_CSV, index=False)
    log.info("Synthetic market data saved to %s  (%d rows)", SYNTHETIC_CSV, len(out))

    # Quick sanity check: correlation of implied_expected_temp with actual
    corr = np.corrcoef(out["implied_expected_temp"].values, out["nws_high"].values)[0, 1]
    log.info("Sanity check: corr(implied_expected_temp, actual) = %.4f", corr)

    return out


# ── Feature engineering ────────────────────────────────────────────

def build_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build the full feature matrix for Model 3.

    Returns
    -------
    train, val, test : pd.DataFrame
        Each contains continuous features, city_idx, nws_high, date, ticker.
    cont_cols : list[str]
        Names of continuous feature columns for the scaler.
    """
    log.info("=== Model 3: Building features ===")

    # Load synthetic market data
    if os.path.exists(SYNTHETIC_CSV):
        df = pd.read_csv(SYNTHETIC_CSV, parse_dates=["date"])
        log.info("Loaded synthetic market data: %d rows", len(df))
    else:
        df = generate_synthetic_markets()

    # ── (a) Per-bucket features are already columns ────────────────
    bucket_price_cols = [f"bucket_{b}_price" for b in range(N_BUCKETS)]
    bucket_spread_cols = [f"bucket_{b}_spread" for b in range(N_BUCKETS)]
    bucket_volume_cols = [f"bucket_{b}_volume" for b in range(N_BUCKETS)]

    # ── (b) Implied distribution stats are already columns ─────────
    implied_cols = [
        "implied_expected_temp", "implied_variance", "implied_skew",
        "implied_kurtosis", "upper_tail_prob", "lower_tail_prob",
        "modal_bucket_idx",
    ]

    # ── (c) Market momentum ────────────────────────────────────────
    momentum_cols = [
        "price_change_1h", "price_change_3h", "intraday_vol", "open_to_now_change",
    ]

    # ── (d) Cross-market neighbor features ─────────────────────────
    # Expand neighbor_imp_avg into individual slots (up to 4) + avg pc + count
    # For simplicity: use the averaged values + n_neighbors
    cross_cols = ["neighbor_imp_avg", "neighbor_pc_avg", "n_neighbors"]

    # Pad cross-market to 6 features: add 3 dummy columns
    # neighbor_imp_avg already summarizes up to 4 neighbors; add variance proxy
    df["neighbor_imp_spread"] = df["neighbor_imp_avg"] - df["implied_expected_temp"]
    df["neighbor_pc_spread"] = df["neighbor_pc_avg"] - df["price_change_1h"]
    df["neighbor_coverage"] = df["n_neighbors"] / 4.0  # fraction of max neighbors
    cross_cols += ["neighbor_imp_spread", "neighbor_pc_spread", "neighbor_coverage"]

    # ── (e) Divergence ─────────────────────────────────────────────
    divergence_cols = ["divergence_raw", "divergence_zscore"]

    # ── (f) Liquidity ──────────────────────────────────────────────
    liquidity_cols = ["avg_normalized_spread", "bid_ask_imbalance", "total_volume"]

    # ── (g) Timing ─────────────────────────────────────────────────
    timing_cols = ["minutes_to_close", "sin_hour", "cos_hour", "fraction_of_day"]

    # ── (h) Calendar features ──────────────────────────────────────
    df = feature_utils.add_calendar_features(df)
    calendar_cols = ["sin_doy", "cos_doy", "sin_month", "cos_month"]

    # ── (i) City static features ───────────────────────────────────
    df = feature_utils.add_city_static_features(df)
    city_static_cols = ["lat", "lon", "elevation_ft", "coastal", "desert", "continentality"]

    # ── (j) City index for embedding ───────────────────────────────
    df = feature_utils.add_city_index(df)

    # ── Assemble continuous feature list ───────────────────────────
    cont_cols = (
        bucket_price_cols       # 6
        + bucket_spread_cols    # 6
        + bucket_volume_cols    # 6
        + implied_cols          # 7
        + momentum_cols         # 4
        + cross_cols            # 6
        + divergence_cols       # 2
        + liquidity_cols        # 3
        + timing_cols           # 4
        + calendar_cols         # 4
        + city_static_cols      # 6
    )
    # Total: 6+6+6+7+4+6+2+3+4+4+6 = 54 continuous
    # (The specification targets ~65; we can count them accurately.)
    log.info("Total continuous features: %d", len(cont_cols))

    # ── Drop rows with NaN NWS high ─────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["nws_high"]).reset_index(drop=True)
    log.info("Dropped %d rows with NaN NWS high (remaining: %d)", before - len(df), len(df))

    # Fill remaining NaNs (neighbor features may be 0 for isolated cities)
    df[cont_cols] = df[cont_cols].fillna(0.0)

    # ── Split ──────────────────────────────────────────────────────
    train, val, test = feature_utils.split_data(df)
    log.info("Split sizes -- train: %d, val: %d, test: %d", len(train), len(val), len(test))

    return train, val, test, cont_cols


# ── Training ───────────────────────────────────────────────────────

def train():
    """Generate synthetic data if needed, build features, train Model 3, evaluate."""
    # Step 1: ensure synthetic market data exists
    if not os.path.exists(SYNTHETIC_CSV):
        generate_synthetic_markets()

    # Step 2: build features
    train_df, val_df, test_df, cont_cols = build_features()

    # Save unscaled actuals before scaling
    val_df_raw_actual = val_df["nws_high"].values.copy()
    test_df_raw_actual = test_df["nws_high"].values.copy()

    # ── Scale continuous features (fit on train only) ──────────────
    scaler = feature_utils.ScalerWrapper()
    train_df = scaler.fit_transform(train_df, cont_cols)
    val_df = scaler.transform(val_df)
    test_df = scaler.transform(test_df)

    # Save scaler
    scaler_path = os.path.join(cfg.CHECKPOINT_DIR, "model3_scaler.pkl")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    scaler.save(scaler_path)
    log.info("Scaler saved to %s", scaler_path)

    # ── Create TensorDatasets ──────────────────────────────────────
    train_ds = training.make_dataset(
        train_df[cont_cols].values,
        train_df["city_idx"].values,
        train_df["nws_high"].values,
    )
    val_ds = training.make_dataset(
        val_df[cont_cols].values,
        val_df["city_idx"].values,
        val_df["nws_high"].values,
    )
    test_ds = training.make_dataset(
        test_df[cont_cols].values,
        test_df["city_idx"].values,
        test_df["nws_high"].values,
    )

    hp = cfg.MODEL3_HP
    train_loader = training.make_loader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_loader = training.make_loader(val_ds, batch_size=hp["batch_size"], shuffle=False)
    test_loader = training.make_loader(test_ds, batch_size=hp["batch_size"], shuffle=False)

    # ── Build model (LayerNorm for market regime shifts) ───────────
    model = training.TemperatureMLP(
        n_continuous=len(cont_cols),
        n_cities=cfg.N_CITIES,
        city_embed_dim=hp["city_embed_dim"],
        hidden_dims=hp["hidden_dims"],
        dropout=hp["dropout"],
        use_layer_norm=True,
    )
    log.info("Model 3 architecture:\n%s", model)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Total parameters: %d", n_params)

    # ── Train ──────────────────────────────────────────────────────
    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "model3_best.pt")
    history = training.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        hp=hp,
        checkpoint_path=checkpoint_path,
    )

    # ── Evaluate on test set ───────────────────────────────────────
    log.info("=== Model 3: Test set evaluation ===")
    mu_pred, sigma_pred = training.predict(model, test_loader)

    test_eval = test_df.copy()
    test_eval["pred_temp"] = mu_pred
    test_eval["pred_sigma"] = sigma_pred
    test_eval["actual_temp"] = test_df_raw_actual

    # Overall metrics
    overall = evaluation.compute_metrics(test_eval["actual_temp"].values, test_eval["pred_temp"].values)
    log.info("Overall test metrics:")
    for k, v in overall.items():
        log.info("  %s: %.4f", k, v)

    # Per-city metrics
    city_metrics = evaluation.metrics_by_city(test_eval, "actual_temp", "pred_temp")
    log.info("Per-city test MAE:\n%s",
             city_metrics[["city", "mae", "rmse", "bias"]].to_string(index=False))

    # Calibration check
    cal = evaluation.calibration_check(
        test_eval["actual_temp"].values, mu_pred, sigma_pred,
    )
    log.info("Calibration:\n%s", cal.to_string(index=False))

    # ── Save val + test predictions for ensemble ──
    val_actual = val_df_raw_actual
    test_actual = test_df_raw_actual
    for split_name, split_loader, split_df, actual_vals in [
        ("val", val_loader, val_df, val_actual),
        ("test", test_loader, test_df, test_actual),
    ]:
        mu_s, sigma_s = training.predict(model, split_loader)
        pred_df = pd.DataFrame({
            "date": split_df["date"].values,
            "ticker": split_df["ticker"].values,
            "mu": mu_s,
            "sigma": sigma_s,
            "y_true": actual_vals,
        })
        path = os.path.join(cfg.CHECKPOINT_DIR, f"model3_preds_{split_name}.csv")
        pred_df.to_csv(path, index=False)
        log.info("Saved %s predictions to %s", split_name, path)

    return model, history, test_eval


# ── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
