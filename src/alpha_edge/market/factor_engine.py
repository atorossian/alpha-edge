# factor_engine.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from alpha_edge.core.schemas import PCAModel


def fit_pca_model(
    asset_rets: pd.DataFrame,
    k: int = 5,
) -> PCAModel:
    """
    Fit a simple PCA factor model on daily asset returns.

    asset_rets: DataFrame (T x N) with no missing values (caller should dropna)
    Model:
      X = (R - mu)
      X ≈ F @ L.T + eps
    """
    if asset_rets.empty:
        raise ValueError("asset_rets is empty")

    X = asset_rets.to_numpy(dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 50 or X.shape[1] < 2:
        raise ValueError("asset_rets must be (T x N) with enough rows and N>=2")

    T, N = X.shape
    k = int(min(max(1, k), N))

    mu = X.mean(axis=0)
    Xc = X - mu

    # SVD-based PCA
    # Xc = U S Vt; principal directions are rows of Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    # Loadings: N x K
    loadings = Vt[:k, :].T.copy()

    # Factor returns: T x K (scores)
    factor_returns = Xc @ loadings

    # Residuals: T x N
    resid = Xc - (factor_returns @ loadings.T)

    return PCAModel(
        tickers=list(asset_rets.columns),
        mu=mu,
        loadings=loadings,
        factor_returns=factor_returns,
        resid=resid,
    )


def sample_portfolio_returns_paths_pca(
    *,
    model: PCAModel,
    weights: dict[str, float],
    n_paths: int,
    n_days: int,
    seed: int | None = None,
    # optional: (min,max) for stationary-like blocks on factor returns/residuals
    block_size: int | tuple[int, int] | None = None,
    weight_mode: str = "long_only",
) -> np.ndarray:
    """
    Generate portfolio return paths using:
      r_asset_t = mu + f_t @ L.T + eps_t
      r_port_t  = w' r_asset_t

    Supports:
      - long_only: normalize by sum(weights)
      - gross_signed / long_short: normalize by sum(abs(weights))

    Returns: R_port_paths shape (n_paths, n_days)
    """
    rng = np.random.default_rng(seed)

    tickers = model.tickers

    # align weights to model tickers
    w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)

    if weight_mode in {"gross_signed", "long_short"}:
        denom = float(np.sum(np.abs(w)))
        if not np.isfinite(denom) or denom <= 0:
            raise ValueError("Gross weight exposure <= 0 after aligning to PCA tickers")
        w = w / denom
    else:
        denom = float(np.sum(w))
        if not np.isfinite(denom) or denom <= 0:
            raise ValueError("Weights sum <= 0 after aligning to PCA tickers")
        w = w / denom

    F = model.factor_returns   # (T, K)
    E = model.resid            # (T, N)
    mu = model.mu              # (N,)
    L = model.loadings         # (N, K)

    T_hist = F.shape[0]

    # --- sample indices for time steps (iid or simple blocks) ---
    if block_size is None:
        idx = rng.integers(0, T_hist, size=(n_paths, n_days))
    else:
        if isinstance(block_size, tuple):
            # approximate persistence: variable block lengths
            bmin, bmax = int(block_size[0]), int(block_size[1])
            if bmin < 1 or bmax < bmin:
                raise ValueError("block_size tuple must be (min>=1, max>=min)")
            avg = 0.5 * (bmin + bmax)
            p_restart = 1.0 / avg
            cur = rng.integers(0, T_hist, size=n_paths, dtype=np.int64)
            idx = np.empty((n_paths, n_days), dtype=np.int64)
            idx[:, 0] = cur
            for t in range(1, n_days):
                restart = (rng.random(n_paths) < p_restart) | (cur >= T_hist - 1)
                new_draws = rng.integers(0, T_hist, size=n_paths, dtype=np.int64)
                cur = np.where(restart, new_draws, cur + 1)
                idx[:, t] = cur
        else:
            B = int(block_size)
            if B < 1:
                raise ValueError("block_size must be >= 1")
            n_blocks = (n_days + B - 1) // B
            starts = rng.integers(0, max(1, T_hist - B), size=(n_paths, n_blocks))
            offsets = np.arange(B, dtype=np.int64)[None, None, :]
            idx = (starts[:, :, None] + offsets).reshape(n_paths, n_blocks * B)[:, :n_days]

    # sample factor returns and residuals using same idx
    F_s = F[idx]  # (P, D, K)
    E_s = E[idx]  # (P, D, N)

    # reconstruct asset returns (P, D, N)
    R_assets = mu[None, None, :] + (F_s @ L.T) + E_s

    # collapse to portfolio returns (P, D)
    R_port = (R_assets * w[None, None, :]).sum(axis=2)
    return R_port.astype(np.float64)


def sample_iid_returns(
    port_rets: np.ndarray,
    n_paths: int,
    n_days: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    IID bootstrap of daily returns.
    Returns R with shape (n_paths, n_days).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(port_rets, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("port_rets must be 1D with enough history")

    idx = rng.integers(0, x.size, size=(n_paths, n_days), dtype=np.int64)
    return x[idx]

def sample_block_bootstrap_returns(
    port_rets: np.ndarray,
    n_paths: int,
    n_days: int,
    block_size: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    """
    Fixed-length block bootstrap.
    Samples contiguous blocks of length B from history and stitches until n_days.
    Returns R with shape (n_paths, n_days).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(port_rets, dtype=np.float64)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("port_rets must be 1D with enough history")

    B = int(block_size)
    if B < 1:
        raise ValueError("block_size must be >= 1")

    N = x.size
    # Need starts in [0, N-B] to keep blocks contiguous
    max_start = max(1, N - B)
    n_blocks = (n_days + B - 1) // B

    starts = rng.integers(0, max_start, size=(n_paths, n_blocks), dtype=np.int64)
    offsets = np.arange(B, dtype=np.int64)[None, None, :]
    idx = (starts[:, :, None] + offsets).reshape(n_paths, n_blocks * B)[:, :n_days]

    return x[idx]