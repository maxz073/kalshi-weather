"""
Compare all forecast models against NWS daily recorded high temperatures.

Models compared:
  - 5 NWP forecast models (GFS, ECMWF, ICON, GEM, JMA) + their ensemble mean
  - Neural net Model 1 (Forecast Bias-Correction)
  - Neural net Model 2 (Historical / Regime)
  - Neural net Model 3 (Market / Microstructure)
  - Neural net Ensemble (Dynamic-weight combination of Models 1-3)

Metrics: MAE, RMSE, Bias (mean error), correlation, % within 1/2/3°F
"""
import os
import logging

import numpy as np
import pandas as pd

import config as cfg
import data_fetch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred):
    """Compute comparison metrics between true and predicted values."""
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "MAE": np.mean(abs_err),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "Bias": np.mean(err),
        "Corr": np.corrcoef(y_true, y_pred)[0, 1],
        "Within_1F": np.mean(abs_err <= 1.0) * 100,
        "Within_2F": np.mean(abs_err <= 2.0) * 100,
        "Within_3F": np.mean(abs_err <= 3.0) * 100,
        "P50_err": np.median(abs_err),
        "P90_err": np.percentile(abs_err, 90),
        "P95_err": np.percentile(abs_err, 95),
        "N": len(y_true),
    }


def run_comparison():
    # ── Load data ────────────────────────────────────────────────────
    log.info("Loading NWS daily highs...")
    nws = data_fetch.load_nws_daily()
    nws["date"] = pd.to_datetime(nws["date"])

    log.info("Loading forecast data...")
    forecasts = data_fetch.load_forecasts()
    forecasts["date"] = pd.to_datetime(forecasts["date"])

    # Merge NWS + forecasts
    df = nws.merge(forecasts, on=["date", "ticker"], how="inner")
    log.info("Merged NWS + forecasts: %d rows", len(df))

    # Forecast columns
    fcst_cols = [c for c in df.columns if c.startswith("fcst_")]
    # Fill NaN forecasts per city
    for col in fcst_cols:
        city_means = df.groupby("ticker")[col].transform("mean")
        df[col] = df[col].fillna(city_means)
    df["fcst_ensemble_mean"] = df[fcst_cols].mean(axis=1)

    # Drop rows with NaN NWS high
    df = df.dropna(subset=["nws_high"]).reset_index(drop=True)

    # ── Load neural net predictions ──────────────────────────────────
    nn_models = {}
    ckpt = cfg.CHECKPOINT_DIR
    for name, filename in [
        ("NN Model 1 (Forecast)", "model1_preds_test.csv"),
        ("NN Model 2 (Historical)", "model2_preds_test.csv"),
        ("NN Model 3 (Market)", "model3_preds_test.csv"),
    ]:
        path = os.path.join(ckpt, filename)
        if os.path.exists(path):
            pred_df = pd.read_csv(path, parse_dates=["date"])
            nn_models[name] = pred_df
            log.info("Loaded %s: %d rows", name, len(pred_df))

    ensemble_path = os.path.join(ckpt, "ensemble_preds_test.csv")
    if os.path.exists(ensemble_path):
        ens_df = pd.read_csv(ensemble_path, parse_dates=["date"])
        nn_models["NN Ensemble"] = ens_df
        log.info("Loaded NN Ensemble: %d rows", len(ens_df))

    # Also load val predictions for a larger evaluation set
    nn_models_val = {}
    for name, filename in [
        ("NN Model 1 (Forecast)", "model1_preds_val.csv"),
        ("NN Model 2 (Historical)", "model2_preds_val.csv"),
        ("NN Model 3 (Market)", "model3_preds_val.csv"),
    ]:
        path = os.path.join(ckpt, filename)
        if os.path.exists(path):
            pred_df = pd.read_csv(path, parse_dates=["date"])
            nn_models_val[name] = pred_df

    # ── Define date splits ───────────────────────────────────────────
    splits = {
        "Full (2022-2026)": (pd.Timestamp(cfg.TRAIN_START), pd.Timestamp(cfg.TEST_END)),
        "Train (2022-2024)": (pd.Timestamp(cfg.TRAIN_START), pd.Timestamp(cfg.TRAIN_END)),
        "Val (2025)": (pd.Timestamp(cfg.VAL_START), pd.Timestamp(cfg.VAL_END)),
        "Test (2026)": (pd.Timestamp(cfg.TEST_START), pd.Timestamp(cfg.TEST_END)),
    }

    # ── Compare NWP models across all splits ─────────────────────────
    nwp_models = fcst_cols + ["fcst_ensemble_mean"]
    nwp_display = {
        "fcst_gfs_seamless": "GFS",
        "fcst_ecmwf_ifs025": "ECMWF",
        "fcst_icon_seamless": "ICON",
        "fcst_gem_seamless": "GEM",
        "fcst_jma_seamless": "JMA",
        "fcst_ensemble_mean": "NWP Ens. Mean",
    }

    for split_name, (start, end) in splits.items():
        split_df = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(split_df) == 0:
            continue

        print(f"\n{'='*100}")
        print(f"  {split_name}  —  {len(split_df)} city-days")
        print(f"{'='*100}")
        print(f"{'Model':<25} {'MAE':>6} {'RMSE':>6} {'Bias':>7} {'Corr':>6} "
              f"{'≤1°F':>6} {'≤2°F':>6} {'≤3°F':>6} {'P50':>5} {'P90':>5} {'P95':>5}")
        print(f"{'-'*25} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*5}")

        results = []

        # NWP models (available for all splits)
        for col in nwp_models:
            mask = split_df[col].notna()
            if mask.sum() < 10:
                continue
            m = compute_metrics(split_df.loc[mask, "nws_high"].values,
                                split_df.loc[mask, col].values)
            name = nwp_display.get(col, col)
            results.append((name, m))

        # Neural net models (only for val/test splits where predictions exist)
        if split_name.startswith("Val"):
            for name, pred_df in nn_models_val.items():
                merged = pred_df.merge(nws[["date", "ticker", "nws_high"]],
                                       on=["date", "ticker"], how="inner")
                merged = merged.dropna(subset=["nws_high", "mu"])
                if len(merged) >= 10:
                    m = compute_metrics(merged["nws_high"].values, merged["mu"].values)
                    results.append((name, m))

        if split_name.startswith("Test"):
            for name, pred_df in nn_models.items():
                if "ensemble_mu" in pred_df.columns:
                    mu_col = "ensemble_mu"
                else:
                    mu_col = "mu"
                merged = pred_df.merge(nws[["date", "ticker", "nws_high"]],
                                       on=["date", "ticker"], how="inner")
                merged = merged.dropna(subset=["nws_high", mu_col])
                if len(merged) >= 10:
                    m = compute_metrics(merged["nws_high"].values, merged[mu_col].values)
                    results.append((name, m))

        # Sort by MAE
        results.sort(key=lambda x: x[1]["MAE"])

        for name, m in results:
            print(f"{name:<25} {m['MAE']:>5.2f}F {m['RMSE']:>5.2f}F {m['Bias']:>+6.2f}F "
                  f"{m['Corr']:>5.3f} {m['Within_1F']:>5.1f}% {m['Within_2F']:>5.1f}% "
                  f"{m['Within_3F']:>5.1f}% {m['P50_err']:>4.1f}F {m['P90_err']:>4.1f}F {m['P95_err']:>4.1f}F")

    # ── Per-city breakdown (test set) ────────────────────────────────
    test_df = df[(df["date"] >= pd.Timestamp(cfg.TEST_START)) &
                 (df["date"] <= pd.Timestamp(cfg.TEST_END))]

    if len(test_df) > 0:
        print(f"\n{'='*100}")
        print(f"  Per-City MAE on Test Set (2026)  —  NWP models vs NWS recorded high")
        print(f"{'='*100}")

        # Build per-city MAE for each model
        city_results = []
        for ticker in sorted(cfg.CITY_TICKERS):
            city_name = cfg.CITIES[ticker][0]
            city_df = test_df[test_df["ticker"] == ticker]
            if len(city_df) == 0:
                continue
            row = {"City": city_name}
            for col in nwp_models:
                mask = city_df[col].notna()
                if mask.sum() > 0:
                    mae = np.mean(np.abs(city_df.loc[mask, col].values -
                                         city_df.loc[mask, "nws_high"].values))
                    name = nwp_display.get(col, col)
                    row[name] = mae

            # Add NN ensemble if available
            if "NN Ensemble" in nn_models:
                ens = nn_models["NN Ensemble"]
                city_ens = ens[ens["ticker"] == ticker].merge(
                    nws[["date", "ticker", "nws_high"]], on=["date", "ticker"], how="inner"
                ).dropna(subset=["nws_high", "ensemble_mu"])
                if len(city_ens) > 0:
                    row["NN Ensemble"] = np.mean(np.abs(
                        city_ens["ensemble_mu"].values - city_ens["nws_high"].values))

            city_results.append(row)

        city_df_out = pd.DataFrame(city_results)
        # Format
        model_cols = [c for c in city_df_out.columns if c != "City"]
        header = f"{'City':<16}" + "".join(f"{c:>14}" for c in model_cols)
        print(header)
        print("-" * len(header))
        for _, row in city_df_out.iterrows():
            line = f"{row['City']:<16}"
            for c in model_cols:
                val = row.get(c, np.nan)
                if pd.notna(val):
                    line += f"{val:>13.2f}F"
                else:
                    line += f"{'—':>14}"
            print(line)

        # Print averages
        print("-" * len(header))
        line = f"{'AVERAGE':<16}"
        for c in model_cols:
            vals = city_df_out[c].dropna()
            line += f"{vals.mean():>13.2f}F" if len(vals) > 0 else f"{'—':>14}"
        print(line)

    # ── Open-Meteo reanalysis vs NWS comparison ─────────────────────
    # Show how much the old target (Open-Meteo temperature_2m_max) differs from NWS
    log.info("Loading Open-Meteo archive for reanalysis comparison...")
    archive = data_fetch.load_archive_daily()
    archive["date"] = pd.to_datetime(archive["date"])
    compare = nws.merge(archive[["date", "ticker", "temperature_2m_max"]],
                        on=["date", "ticker"], how="inner")
    compare = compare.dropna(subset=["nws_high", "temperature_2m_max"])

    print(f"\n{'='*100}")
    print(f"  Open-Meteo Reanalysis vs NWS Recorded High (old target vs true target)")
    print(f"{'='*100}")

    overall = compute_metrics(compare["nws_high"].values,
                              compare["temperature_2m_max"].values)
    print(f"\nOverall ({len(compare)} city-days):")
    print(f"  MAE:  {overall['MAE']:.2f}°F")
    print(f"  RMSE: {overall['RMSE']:.2f}°F")
    print(f"  Bias: {overall['Bias']:+.2f}°F  (positive = Open-Meteo runs hot)")
    print(f"  Corr: {overall['Corr']:.4f}")
    print(f"  Within 1°F: {overall['Within_1F']:.1f}%")
    print(f"  Within 2°F: {overall['Within_2F']:.1f}%")
    print(f"  Within 3°F: {overall['Within_3F']:.1f}%")

    print(f"\nPer-city Open-Meteo vs NWS discrepancy (MAE / Bias):")
    print(f"{'City':<16} {'MAE':>7} {'Bias':>8} {'RMSE':>7}")
    print(f"{'-'*16} {'-'*7} {'-'*8} {'-'*7}")
    for ticker in sorted(cfg.CITY_TICKERS):
        city_name = cfg.CITIES[ticker][0]
        city_comp = compare[compare["ticker"] == ticker]
        if len(city_comp) == 0:
            continue
        m = compute_metrics(city_comp["nws_high"].values,
                            city_comp["temperature_2m_max"].values)
        print(f"{city_name:<16} {m['MAE']:>6.2f}F {m['Bias']:>+7.2f}F {m['RMSE']:>6.2f}F")


if __name__ == "__main__":
    run_comparison()
