"""
Shared training infrastructure for all 3 models.
"""
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

import config as cfg

log = logging.getLogger(__name__)


# ── Loss function ────────────────────────────────────────────────────

def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of heteroscedastic Gaussian.
    mu, sigma, y: all (batch,) tensors.
    """
    var = sigma ** 2 + 1e-6
    return 0.5 * (torch.log(var) + (y - mu) ** 2 / var).mean()


# ── Dataset helpers ──────────────────────────────────────────────────

def make_dataset(features: np.ndarray, city_idx: np.ndarray, targets: np.ndarray) -> TensorDataset:
    """Create a TensorDataset from numpy arrays."""
    return TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(city_idx, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float32),
    )


def make_loader(dataset: TensorDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ── Generic MLP builder ─────────────────────────────────────────────

class TemperatureMLP(nn.Module):
    """MLP with city embedding, heteroscedastic output (mu, sigma)."""

    def __init__(self, n_continuous: int, n_cities: int = cfg.N_CITIES,
                 city_embed_dim: int = 8, hidden_dims: list[int] = None,
                 dropout: list[float] = None, use_layer_norm: bool = False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        if dropout is None:
            dropout = [0.2, 0.15, 0.0]

        self.city_embed = nn.Embedding(n_cities, city_embed_dim)
        input_dim = n_continuous + city_embed_dim

        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            else:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            if i < len(dropout) and dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 2))  # [mu, log_sigma]
        self.net = nn.Sequential(*layers)

    def forward(self, x_continuous: torch.Tensor, city_idx: torch.Tensor):
        emb = self.city_embed(city_idx)
        x = torch.cat([x_continuous, emb], dim=1)
        out = self.net(x)
        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-3  # ensure positive
        return mu, sigma


# ── Training loop ────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hp: dict,
    checkpoint_path: str,
    device: str = None,
) -> dict:
    """
    Train with AdamW, cosine annealing, early stopping, gradient clipping.
    Returns training history dict.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log.info("Training on device: %s", device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"], eta_min=hp["lr"] / 100)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = hp.get("patience", 20)
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(hp["epochs"]):
        # Train
        model.train()
        train_losses = []
        for x, city_idx, y in train_loader:
            x, city_idx, y = x.to(device), city_idx.to(device), y.to(device)
            mu, sigma = model(x, city_idx)
            loss = gaussian_nll_loss(mu, sigma, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, city_idx, y in val_loader:
                x, city_idx, y = x.to(device), city_idx.to(device), y.to(device)
                mu, sigma = model(x, city_idx)
                loss = gaussian_nll_loss(mu, sigma, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info("Epoch %d/%d  train=%.4f  val=%.4f  lr=%.2e",
                     epoch + 1, hp["epochs"], train_loss, val_loss, lr)

        # Early stopping + checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info("Early stopping at epoch %d (best val=%.4f)", epoch + 1, best_val_loss)
                break

    # Load best
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    log.info("Training complete. Best val loss: %.4f", best_val_loss)
    return history


# ── Inference ────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str = None) -> tuple[np.ndarray, np.ndarray]:
    """Run inference, return (mu_array, sigma_array)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    mus, sigmas = [], []
    for x, city_idx, _ in loader:
        x, city_idx = x.to(device), city_idx.to(device)
        mu, sigma = model(x, city_idx)
        mus.append(mu.cpu().numpy())
        sigmas.append(sigma.cpu().numpy())
    return np.concatenate(mus), np.concatenate(sigmas)
