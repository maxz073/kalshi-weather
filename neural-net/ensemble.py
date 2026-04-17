"""
Dynamic-weight ensemble combining 3 base models + Kalshi bucket pricer.

Training strategy (stacking):
  - Base models trained on 2022-2024, frozen here.
  - Ensemble weight network trained on 2025 validation set.
  - Evaluated on 2026 test set.
"""
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm

import config as cfg
import evaluation as ev

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Dynamic weight network ───────────────────────────────────────────

class DynamicWeightNet(nn.Module):
    """Small MLP that outputs 3 softmax weights given context features."""

    def __init__(self, context_dim: int = cfg.ENSEMBLE_HP["context_dim"],
                 hidden_dim: int = cfg.ENSEMBLE_HP["hidden_dim"]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 3) softmax weights."""
        return F.softmax(self.net(context), dim=1)


# ── Ensemble combiner ────────────────────────────────────────────────

def ensemble_predict(
    mu1: torch.Tensor, s1: torch.Tensor,
    mu2: torch.Tensor, s2: torch.Tensor,
    mu3: torch.Tensor, s3: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine 3 Gaussian predictions via mixture of Gaussians.

    weights: (batch, 3) softmax weights
    Returns: (ensemble_mu, ensemble_sigma)
    """
    w1, w2, w3 = weights[:, 0], weights[:, 1], weights[:, 2]

    # Mixture mean
    mu_ens = w1 * mu1 + w2 * mu2 + w3 * mu3

    # Mixture variance = E[sigma^2] + E[mu^2] - (E[mu])^2
    var_ens = (
        w1 * (s1 ** 2 + mu1 ** 2) +
        w2 * (s2 ** 2 + mu2 ** 2) +
        w3 * (s3 ** 2 + mu3 ** 2)
    ) - mu_ens ** 2

    sigma_ens = torch.sqrt(var_ens.clamp(min=1e-6))
    return mu_ens, sigma_ens


# ── Context feature builder ─────────────────────────────────────────

def build_context_features(df: pd.DataFrame) -> np.ndarray:
    """Build the 8-dim context vector for the weight network.

    Expected columns in df:
      - forecast_spread: std of forecast models (high = uncertain forecast)
      - market_liquidity: total volume or similar (high = trust market more)
      - forecast_available: 1 if forecast data present, else 0
      - market_available: 1 if market data present, else 0
      - minutes_to_close: minutes until market settlement
      - sin_doy, cos_doy: seasonal encoding
      - hours_since_open: hours since market open
    """
    context_cols = [
        "minutes_to_close_norm", "forecast_spread_norm",
        "market_liquidity_norm", "forecast_available",
        "market_available", "sin_doy", "cos_doy",
        "hours_since_open_norm",
    ]

    # Fill missing context columns with defaults
    if "minutes_to_close_norm" not in df.columns:
        df["minutes_to_close_norm"] = 0.5  # midday default
    if "forecast_spread_norm" not in df.columns:
        df["forecast_spread_norm"] = 0.0
    if "market_liquidity_norm" not in df.columns:
        df["market_liquidity_norm"] = 0.0
    if "forecast_available" not in df.columns:
        df["forecast_available"] = 1.0
    if "market_available" not in df.columns:
        df["market_available"] = 0.0  # synthetic data
    if "hours_since_open_norm" not in df.columns:
        df["hours_since_open_norm"] = 0.5

    return df[context_cols].fillna(0).values.astype(np.float32)


# ── Bucket pricer ────────────────────────────────────────────────────

def compute_bucket_fair_values(
    mu: float,
    sigma: float,
    bucket_edges: list[tuple[float, float]],
    fee_rate: float = cfg.TAKER_FEE_RATE,
) -> list[dict]:
    """Convert ensemble (mu, sigma) into fair value for each Kalshi bucket.

    bucket_edges: list of (lower_bound, upper_bound) in Fahrenheit.
    Returns list of dicts with keys: lower, upper, prob, fair_yes_cents, fair_no_cents.
    """
    results = []
    probs = ev.gaussian_bucket_probs(mu, sigma, bucket_edges)
    for (low, high), prob in zip(bucket_edges, probs):
        yes_cents = ev.fair_value_cents(prob, fee_rate)
        no_cents = ev.fair_value_cents(1.0 - prob, fee_rate)
        results.append({
            "lower": low,
            "upper": high,
            "prob": round(prob, 4),
            "fair_yes_cents": yes_cents,
            "fair_no_cents": no_cents,
        })
    return results


# ── Training the ensemble ────────────────────────────────────────────

def gaussian_nll(mu, sigma, y):
    var = sigma ** 2 + 1e-6
    return 0.5 * (torch.log(var) + (y - mu) ** 2 / var).mean()


def train_ensemble(
    preds_val: dict,
    y_val: np.ndarray,
    context_val: np.ndarray,
    preds_test: dict,
    y_test: np.ndarray,
    context_test: np.ndarray,
):
    """Train the dynamic weight network on validation data.

    preds_val / preds_test: dict with keys 'mu1','s1','mu2','s2','mu3','s3' (numpy arrays)
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    hp = cfg.ENSEMBLE_HP

    # Build tensors for val
    val_mu1 = torch.tensor(preds_val["mu1"], dtype=torch.float32)
    val_s1 = torch.tensor(preds_val["s1"], dtype=torch.float32)
    val_mu2 = torch.tensor(preds_val["mu2"], dtype=torch.float32)
    val_s2 = torch.tensor(preds_val["s2"], dtype=torch.float32)
    val_mu3 = torch.tensor(preds_val["mu3"], dtype=torch.float32)
    val_s3 = torch.tensor(preds_val["s3"], dtype=torch.float32)
    val_ctx = torch.tensor(context_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32)

    val_ds = TensorDataset(val_mu1, val_s1, val_mu2, val_s2, val_mu3, val_s3, val_ctx, val_y)
    val_loader = DataLoader(val_ds, batch_size=hp["batch_size"], shuffle=True)

    # Init weight network
    weight_net = DynamicWeightNet(context_dim=context_val.shape[1]).to(device)
    optimizer = torch.optim.AdamW(weight_net.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"])

    best_loss = float("inf")
    patience_counter = 0
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "ensemble_weights.pt")
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(hp["epochs"]):
        weight_net.train()
        losses = []
        for m1, s1, m2, s2, m3, s3, ctx, y in val_loader:
            m1, s1 = m1.to(device), s1.to(device)
            m2, s2 = m2.to(device), s2.to(device)
            m3, s3 = m3.to(device), s3.to(device)
            ctx, y = ctx.to(device), y.to(device)

            weights = weight_net(ctx)
            mu_ens, sigma_ens = ensemble_predict(m1, s1, m2, s2, m3, s3, weights)
            loss = gaussian_nll(mu_ens, sigma_ens, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        epoch_loss = np.mean(losses)

        if (epoch + 1) % 10 == 0:
            log.info("Ensemble epoch %d/%d  loss=%.4f", epoch + 1, hp["epochs"], epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(weight_net.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= hp["patience"]:
                log.info("Ensemble early stopping at epoch %d", epoch + 1)
                break

    weight_net.load_state_dict(torch.load(ckpt_path, weights_only=True))
    log.info("Ensemble training complete. Best loss: %.4f", best_loss)

    # ── Evaluate on test set ─────────────────────────────────────────
    weight_net.eval()
    with torch.no_grad():
        t_mu1 = torch.tensor(preds_test["mu1"], dtype=torch.float32).to(device)
        t_s1 = torch.tensor(preds_test["s1"], dtype=torch.float32).to(device)
        t_mu2 = torch.tensor(preds_test["mu2"], dtype=torch.float32).to(device)
        t_s2 = torch.tensor(preds_test["s2"], dtype=torch.float32).to(device)
        t_mu3 = torch.tensor(preds_test["mu3"], dtype=torch.float32).to(device)
        t_s3 = torch.tensor(preds_test["s3"], dtype=torch.float32).to(device)
        t_ctx = torch.tensor(context_test, dtype=torch.float32).to(device)

        weights = weight_net(t_ctx)
        mu_ens, sigma_ens = ensemble_predict(t_mu1, t_s1, t_mu2, t_s2, t_mu3, t_s3, weights)

    mu_np = mu_ens.cpu().numpy()
    sigma_np = sigma_ens.cpu().numpy()
    weights_np = weights.cpu().numpy()

    metrics = ev.compute_metrics(y_test, mu_np)
    log.info("Ensemble test metrics: %s", metrics)

    return weight_net, mu_np, sigma_np, weights_np, metrics


# ── Full pipeline ────────────────────────────────────────────────────

def run_ensemble():
    """Load all 3 base model predictions and train the ensemble.

    Expects pre-computed prediction CSVs from each model:
      checkpoints/model{1,2,3}_preds_{val,test}.csv
    with columns: ticker, date, mu, sigma, y_true
    """
    log.info("Loading base model predictions...")

    preds = {}
    for split in ["val", "test"]:
        # Load all 3 model predictions and merge on (date, ticker)
        dfs = []
        for m in [1, 2, 3]:
            path = os.path.join(cfg.CHECKPOINT_DIR, f"model{m}_preds_{split}.csv")
            if not os.path.exists(path):
                log.error("Missing predictions: %s — run model%d training first", path, m)
                return
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.rename(columns={"mu": f"mu{m}", "sigma": f"s{m}"})
            if m == 1:
                dfs.append(df[["date", "ticker", f"mu{m}", f"s{m}", "y_true"]])
            else:
                dfs.append(df[["date", "ticker", f"mu{m}", f"s{m}"]])

        # Inner join on (date, ticker) to align all predictions
        merged = dfs[0]
        for extra in dfs[1:]:
            merged = merged.merge(extra, on=["date", "ticker"], how="inner")
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        log.info("Aligned %s predictions: %d rows", split, len(merged))

        split_preds = {
            "mu1": merged["mu1"].values, "s1": merged["s1"].values,
            "mu2": merged["mu2"].values, "s2": merged["s2"].values,
            "mu3": merged["mu3"].values, "s3": merged["s3"].values,
            "y_true": merged["y_true"].values,
            "_df": merged,
        }
        preds[split] = split_preds

    # Build context features
    from feature_utils import sin_cos_encode
    for split in ["val", "test"]:
        df = preds[split]["_df"]
        doy = pd.to_datetime(df["date"]).dt.dayofyear.values.astype(float)
        sin_doy, cos_doy = sin_cos_encode(doy, 365.25)

        # Forecast spread (from model 1 predictions — use sigma as proxy)
        fcst_spread = preds[split]["s1"]
        fcst_spread_norm = (fcst_spread - fcst_spread.mean()) / (fcst_spread.std() + 1e-6)

        context = np.column_stack([
            np.full(len(df), 0.5),       # minutes_to_close_norm (default)
            fcst_spread_norm,             # forecast_spread_norm
            np.full(len(df), 0.0),       # market_liquidity_norm
            np.ones(len(df)),            # forecast_available
            np.zeros(len(df)),           # market_available (synthetic)
            sin_doy,
            cos_doy,
            np.full(len(df), 0.5),       # hours_since_open_norm
        ]).astype(np.float32)

        preds[split]["context"] = context

    # Train ensemble on val, evaluate on test
    weight_net, mu_test, sigma_test, weights_test, metrics = train_ensemble(
        preds_val={k: v for k, v in preds["val"].items() if k != "_df"},
        y_val=preds["val"]["y_true"],
        context_val=preds["val"]["context"],
        preds_test={k: v for k, v in preds["test"].items() if k != "_df"},
        y_test=preds["test"]["y_true"],
        context_test=preds["test"]["context"],
    )

    # Save ensemble predictions
    test_df = preds["test"]["_df"].copy()
    test_df["ensemble_mu"] = mu_test
    test_df["ensemble_sigma"] = sigma_test
    test_df["w1_forecast"] = weights_test[:, 0]
    test_df["w2_historical"] = weights_test[:, 1]
    test_df["w3_market"] = weights_test[:, 2]

    out_path = os.path.join(cfg.CHECKPOINT_DIR, "ensemble_preds_test.csv")
    test_df.to_csv(out_path, index=False)
    log.info("Ensemble predictions saved to %s", out_path)

    # Print per-city metrics
    test_df["y_pred"] = mu_test
    city_metrics = ev.metrics_by_city(test_df, "y_true", "y_pred")
    log.info("\nPer-city ensemble metrics:\n%s", city_metrics.to_string(index=False))

    return weight_net, test_df


if __name__ == "__main__":
    run_ensemble()
