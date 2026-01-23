# tune_score_weights.py
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from alpha_edge.core.schemas import ScoreConfig


# ---- small helpers ----

def _eval_features(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    equity0: float,
    notional: float,
    goals: Tuple[float, float, float],
    main_goal: float,
    lw_cov: pd.DataFrame | None,
    days: int,
    n_paths: int,
    seed: int,
    block_size: int | tuple[int, int] | None,
    path_source: str,
    pca_k: int,
    cfg: ScoreConfig,
    spec_df_full: pd.DataFrame | None = None,  # NEW
) -> tuple[float, float, dict, float]:
    """
    Return (p_main, ruin, penalties, score) under cfg for a given portfolio weights.
    Rebuild penalties + score by reusing optimizer_engine.compute_penalties + score_candidate.

    spec_df_full: precomputed spectral profile DF indexed by ticker (hf/mf/lf/entropy)
                 to avoid recomputing FFT repeatedly during tuning.
    """
    # Import here to avoid circular imports
    from optimizer_engine import compute_penalties, score_candidate
    from mc_engine import simulate_leveraged_paths_vectorized
    from factor_engine import fit_pca_model, sample_portfolio_returns_paths_pca

    goals = tuple(float(g) for g in goals)
    tickers = [t for t in weights.keys() if t in returns.columns]
    if not tickers:
        raise ValueError("No overlap between weights and returns columns")

    w_vec = np.array([float(weights[t]) for t in tickers], dtype=float)
    if w_vec.sum() <= 0:
        raise ValueError("Weights must sum to a positive number")
    w_vec = w_vec / w_vec.sum()

    rets_assets = returns[tickers].dropna(how="any")
    port_rets = (rets_assets * w_vec).sum(axis=1).dropna()
    if len(port_rets) < 50:
        raise ValueError("Not enough data for this candidate")

    # stats_for_pen (same as evaluate_portfolio)
    cum = (1.0 + port_rets).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    max_dd = float(dd.min())

    var_95 = float(np.percentile(port_rets, 5))
    cvar_95 = float(port_rets[port_rets <= var_95].mean())
    stats_for_pen = {"cvar_95": cvar_95, "max_drawdown": max_dd}

    # MC
    if path_source == "bootstrap":
        mc = simulate_leveraged_paths_vectorized(
            port_rets=port_rets.to_numpy(dtype=np.float64),
            notional=notional,
            equity0=equity0,
            goals=list(goals),
            n_days=days,
            n_paths=n_paths,
            seed=seed,
            block_size=block_size,
        )

    elif path_source == "pca":
        pca_model = fit_pca_model(rets_assets, k=int(pca_k))
        R_port_paths = sample_portfolio_returns_paths_pca(
            model=pca_model,
            weights=weights,
            n_paths=n_paths,
            n_days=days,
            seed=seed,
            block_size=block_size,
        )
        mc = simulate_leveraged_paths_vectorized(
            port_rets=None,              # ignored when precomputed_r is provided
            precomputed_r=R_port_paths,  # (n_paths, n_days)
            notional=notional,
            equity0=equity0,
            goals=list(goals),
            n_days=days,
            n_paths=n_paths,
            seed=seed,
            block_size=None,             # irrelevant when precomputed_r is used
        )

    else:
        raise ValueError(path_source)

    ruin = float(mc["ruin_prob"])
    p_main = float(mc["p_hit"].get(float(main_goal), 0.0))

    penalties = compute_penalties(
        w=w_vec,
        stats=stats_for_pen,
        mc=mc,
        asset_rets=rets_assets,
        goals=list(goals),
        cfg=cfg,
        spec_df_full=spec_df_full,  # NEW
    )

    score = float(score_candidate(p_main, ruin, penalties, cfg))
    return p_main, ruin, penalties, score


def _fit_lambdas_nonneg_ridge(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """
    Solve min ||Xb - y||^2 + ridge||b||^2 s.t. b>=0 using projected gradient.
    X shape: (n, d), y shape: (n,)
    """
    n, d = X.shape
    b = np.zeros(d, dtype=np.float64)

    # Lipschitz-ish step
    L = (np.linalg.norm(X, 2) ** 2) / max(1, n) + ridge
    lr = 1.0 / max(L, 1e-8)

    for _ in range(2000):
        grad = (2.0 / n) * (X.T @ (X @ b - y)) + 2.0 * ridge * b
        b = b - lr * grad
        b = np.maximum(b, 0.0)

    return b


def tune_score_config(
    returns: pd.DataFrame,
    lw_cov: pd.DataFrame | None,
    candidate_weights: List[Dict[str, float]],
    equity0: float,
    notional: float,
    goals: Tuple[float, float, float] = (800.0, 1200.0, 2000.0),
    main_goal: float = 2000.0,
    *,
    cheap_paths: int = 4000,
    truth_paths: int = 60000,
    days: int = 252,
    block_size: int | tuple[int, int] | None = (8, 12),
    path_source: str = "bootstrap",
    pca_k: int = 5,
    seed0: int = 123,
) -> ScoreConfig:
    """
    Learn lambdas so cheap-score approximates truth-score.

    We fit:
        y = (p_main_cheap - p_main_truth)  ≈  λ_ruin*ruin + Σ λ_k * pen_k
    using the *cheap* penalties/ruin as explanatory variables.

    This encourages lambdas that penalize candidates where cheap MC is overly optimistic vs truth.
    """
    base_cfg = ScoreConfig()  # caps used inside penalties; lambdas will be overwritten

    # Precompute spectral profiles once for tuning (huge speedup)
    from optimizer_engine import _spectral_profiles_df
    returns_clean = returns.dropna(how="any")
    spec_df_full = _spectral_profiles_df(returns_clean, bands_days=base_cfg.fft_bands_days)

    feature_cols = [
        "ruin",
        "cvar",
        "mdd",
        "hhi",
        "avg_corr",
        "time",
        "hf_ratio",
        "freq_overlap",
        "spec_entropy",
    ]

    rows = []
    for i, w in enumerate(candidate_weights):
        seed = seed0 + i

        # cheap
        p_c, r_c, pen_c, _ = _eval_features(
            returns, w, equity0, notional, goals, main_goal, lw_cov,
            days, cheap_paths, seed, block_size, path_source, pca_k, base_cfg,
            spec_df_full=spec_df_full,
        )

        # truth
        p_t, r_t, pen_t, _ = _eval_features(
            returns, w, equity0, notional, goals, main_goal, lw_cov,
            days, truth_paths, seed, block_size, path_source, pca_k, base_cfg,
            spec_df_full=spec_df_full,
        )

        y = float(p_c - p_t)

        rows.append({
            "y": y,
            "ruin": float(r_c),
            "cvar": float(pen_c.get("cvar", 0.0)),
            "mdd": float(pen_c.get("mdd", 0.0)),
            "hhi": float(pen_c.get("hhi", 0.0)),
            "avg_corr": float(pen_c.get("avg_corr", 0.0)),
            "time": float(pen_c.get("time", 0.0)),
            "hf_ratio": float(pen_c.get("hf_ratio", 0.0)),
            "freq_overlap": float(pen_c.get("freq_overlap", 0.0)),
            "spec_entropy": float(pen_c.get("spec_entropy", 0.0)),
        })

    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()

    print("y mean/std:", df["y"].mean(), df["y"].std())
    print("y pct > 0:", (df["y"] > 0).mean())
    for col in feature_cols:
        print(col, "mean:", df[col].mean(), "pct>0:", (df[col] > 0).mean())

    if len(df) < 20:
        raise RuntimeError("Not enough valid training rows to fit lambdas.")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)

    b = _fit_lambdas_nonneg_ridge(X, y, ridge=1e-3)

    # Build config (copy caps etc.) and set learned lambdas
    cfg = ScoreConfig(**asdict(base_cfg))

    # Map fitted coefficients to config lambdas
    # Order must match feature_cols above
    cfg.lambda_ruin = float(b[0])
    cfg.lambda_cvar = float(b[1])
    cfg.lambda_mdd  = float(b[2])
    cfg.lambda_conc = float(b[3])
    cfg.lambda_corr = float(b[4])
    cfg.lambda_time = float(b[5])

    cfg.lambda_hf_ratio = float(b[6])
    cfg.lambda_freq_overlap = float(b[7])
    cfg.lambda_spec_entropy = float(b[8])

    return cfg


if __name__ == "__main__":
    raise SystemExit("Import this and call tune_score_config(...) from your pipeline.")
