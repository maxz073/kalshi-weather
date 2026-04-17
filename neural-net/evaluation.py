"""
Shared evaluation utilities for all models and ensemble.
"""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config as cfg


# ── Metrics ──────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, bias, R2, correlation."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": r2_score(y_true, y_pred),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "n": len(y_true),
    }


def metrics_by_city(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                    ticker_col: str = "ticker") -> pd.DataFrame:
    """Compute metrics per city."""
    rows = []
    for ticker in df[ticker_col].unique():
        mask = df[ticker_col] == ticker
        city_name = cfg.CITIES.get(ticker, (ticker,))[0]
        m = compute_metrics(df.loc[mask, y_true_col].values, df.loc[mask, y_pred_col].values)
        m["city"] = city_name
        m["ticker"] = ticker
        rows.append(m)
    return pd.DataFrame(rows).sort_values("mae")


# ── Calibration ──────────────────────────────────────────────────────

def calibration_check(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                      quantiles: list[float] = None) -> pd.DataFrame:
    """Check if predicted uncertainty is calibrated.
    For each quantile q, what fraction of actuals fall within the q-CI?
    """
    if quantiles is None:
        quantiles = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
    rows = []
    for q in quantiles:
        z = norm.ppf(0.5 + q / 2)
        lower = mu - z * sigma
        upper = mu + z * sigma
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        rows.append({"target_coverage": q, "actual_coverage": coverage})
    return pd.DataFrame(rows)


# ── Bucket probabilities ────────────────────────────────────────────

def gaussian_bucket_probs(mu: float, sigma: float,
                          bucket_edges: list[tuple[float, float]]) -> list[float]:
    """Compute probability mass in each temperature bucket using Gaussian CDF."""
    probs = []
    for low, high in bucket_edges:
        p_high = norm.cdf(high, loc=mu, scale=sigma) if high < float("inf") else 1.0
        p_low = norm.cdf(low, loc=mu, scale=sigma) if low > float("-inf") else 0.0
        probs.append(max(0.0, p_high - p_low))
    return probs


def fair_value_cents(prob: float, fee_rate: float = cfg.TAKER_FEE_RATE) -> int:
    """Convert probability to fair YES price in cents, adjusted for Kalshi taker fee."""
    raw = prob * 100
    fee = fee_rate * raw * (100 - raw) / 100
    return max(1, min(99, round(raw - fee / 2)))


# ── Plotting ─────────────────────────────────────────────────────────

def plot_calibration(cal_df: pd.DataFrame, title: str = "Calibration Plot"):
    """Plot predicted vs actual coverage."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.plot(cal_df["target_coverage"], cal_df["actual_coverage"], "o-", label="Model")
    ax.set_xlabel("Target Coverage")
    ax.set_ylabel("Actual Coverage")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residuals"):
    """Histogram of residuals."""
    resid = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(resid, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Residual (pred - actual) [F]")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{title} — Histogram")

    from scipy.stats import probplot
    probplot(resid, dist="norm", plot=axes[1])
    axes[1].set_title(f"{title} — Q-Q Plot")
    plt.tight_layout()
    return fig


def plot_timeseries(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                    date_col: str = "date", ticker_col: str = "ticker",
                    cities: list[str] = None, title: str = "Predicted vs Actual"):
    """Time series of predicted vs actual for selected cities."""
    if cities is None:
        cities = ["KXHIGHNY", "KXHIGHTPHX", "KXHIGHMIA"]
    n = len(cities)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, ticker in zip(axes, cities):
        city_df = df[df[ticker_col] == ticker].sort_values(date_col)
        city_name = cfg.CITIES.get(ticker, (ticker,))[0]
        ax.plot(city_df[date_col], city_df[y_true_col], label="Actual", alpha=0.8)
        ax.plot(city_df[date_col], city_df[y_pred_col], label="Predicted", alpha=0.8)
        ax.set_ylabel("Max Temp (F)")
        ax.set_title(f"{city_name}")
        ax.legend()
    axes[-1].set_xlabel("Date")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_mae_heatmap(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                     date_col: str = "date", ticker_col: str = "ticker",
                     title: str = "MAE by City and Month"):
    """Heatmap of MAE per city per month."""
    df = df.copy()
    df["month"] = df[date_col].dt.month
    df["abs_error"] = np.abs(df[y_pred_col] - df[y_true_col])

    pivot = df.pivot_table(values="abs_error", index=ticker_col, columns="month", aggfunc="mean")
    # Map tickers to city names
    pivot.index = [cfg.CITIES.get(t, (t,))[0] for t in pivot.index]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("City")
    plt.tight_layout()
    return fig
