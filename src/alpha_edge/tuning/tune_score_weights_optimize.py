# tune_score_weights_optimize.py
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from alpha_edge.core.schemas import ScoreConfig
from alpha_edge.portfolio.optimizer_engine import evaluate_portfolio_candidate, _spectral_profiles_df


def _split_returns_time(
    returns: pd.DataFrame,
    train_frac: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = returns.dropna(how="any")
    n = len(returns)
    if n < 200:
        raise ValueError("Not enough rows to split returns")
    cut = int(n * train_frac)
    return returns.iloc[:cut], returns.iloc[cut:]


def _evaluate_pool(
    returns: pd.DataFrame,
    lw_cov: pd.DataFrame | None,
    pool: List[Dict[str, float]],
    equity0: float,
    notional: float,
    goals: Tuple[float, float, float],
    main_goal: float,
    cfg: ScoreConfig,
    *,
    spec_df_full: pd.DataFrame | None = None,
    days: int = 252,
    n_paths: int = 20000,
    seed0: int = 123,
    path_source: str = "bootstrap",
    pca_k: int = 5,
    block_size: int | tuple[int, int] | None = (8, 12),
) -> list:
    if spec_df_full is None:
        spec_df_full = _spectral_profiles_df(returns, bands_days=cfg.fft_bands_days)

    out = []
    for i, w in enumerate(pool):
        try:
            m = evaluate_portfolio_candidate(
                returns=returns,
                weights=w,
                equity0=equity0,
                notional=notional,
                goals=list(goals),
                main_goal=main_goal,
                lw_cov=lw_cov,
                days=days,
                n_paths=n_paths,
                score_config=cfg,
                mc_seed=seed0 + i,
                path_source=path_source,
                pca_k=pca_k,
                block_size=block_size,
                spec_df_full=spec_df_full,
            )
            out.append(m)
        except Exception:
            continue
    return out


def _objective_from_metrics(
    metrics: list,
    *,
    main_goal: float,
    ruin_cap: float,
    top_k: int = 5,
    alpha_ruin: float = 0.5,
) -> float:
    if not metrics:
        return -1e9

    metrics = sorted(metrics, key=lambda m: m.score, reverse=True)

    feasible = [m for m in metrics if float(m.ruin_prob_1y) <= float(ruin_cap)]
    if not feasible:
        best_ruin = float(min(m.ruin_prob_1y for m in metrics))
        return -1000.0 - 2000.0 * float(best_ruin - ruin_cap)

    top = feasible[: max(1, min(top_k, len(feasible)))]

    vals = []
    for m in top:
        g1, g2, g3 = m.goals
        if float(main_goal) == float(g1):
            p_main = float(m.p_hit_goal_1_1y)
        elif float(main_goal) == float(g2):
            p_main = float(m.p_hit_goal_2_1y)
        else:
            p_main = float(m.p_hit_goal_3_1y)

        vals.append(p_main - alpha_ruin * float(m.ruin_prob_1y))

    return float(np.mean(vals))


def _sample_lambdas(rng: np.random.Generator, base: ScoreConfig) -> ScoreConfig:
    cfg = ScoreConfig(**asdict(base))

    def logu(lo, hi):
        x = rng.uniform(np.log(lo), np.log(hi))
        return float(np.exp(x))

    # core penalties
    cfg.lambda_ruin = logu(1e-3, 5.0)
    cfg.lambda_mdd = logu(1e-3, 5.0)
    cfg.lambda_cvar = logu(1e-3, 5.0)
    cfg.lambda_conc = logu(1e-3, 5.0)
    cfg.lambda_corr = logu(1e-3, 5.0)
    cfg.lambda_time = logu(1e-3, 5.0)

    # FFT penalties (smaller typical scale)
    cfg.lambda_hf_ratio = logu(1e-4, 1.0)
    cfg.lambda_freq_overlap = logu(1e-4, 1.0)
    cfg.lambda_spec_entropy = logu(1e-4, 1.0)

    return cfg


def tune_lambdas_by_optimization(
    returns: pd.DataFrame,
    lw_cov: pd.DataFrame | None,
    candidate_pool: List[Dict[str, float]],
    equity0: float,
    notional: float,
    goals: Tuple[float, float, float] = (800.0, 1200.0, 2000.0),
    main_goal: float = 2000.0,
    *,
    train_frac: float = 0.7,
    days: int = 252,
    n_paths_train: int = 5000,
    n_paths_valid: int = 20000,
    path_source: str = "bootstrap",
    pca_k: int = 5,
    block_size: int | tuple[int, int] | None = (8, 12),
    n_trials: int = 40,
    pool_sample_size: int = 500,     # NEW: evaluate only this many per trial
    shortlist_size: int = 60,        # NEW: validate only top shortlist
    top_k: int = 5,
    ruin_cap: float = 0.10,
    alpha_ruin: float = 0.5,
    seed: int = 123,
) -> tuple[ScoreConfig, dict]:
    rng = np.random.default_rng(seed)

    train_rets, valid_rets = _split_returns_time(returns, train_frac=train_frac)

    base = ScoreConfig()

    # Precompute FFT profiles once per split for speed
    spec_train = _spectral_profiles_df(train_rets, bands_days=base.fft_bands_days)
    spec_valid = _spectral_profiles_df(valid_rets, bands_days=base.fft_bands_days)

    best_cfg: ScoreConfig | None = None
    best_obj = -1e18
    best_info: dict = {}

    n_pool = len(candidate_pool)
    if n_pool < 50:
        raise ValueError("candidate_pool too small; need at least ~50")

    for t in range(n_trials):
        cfg = _sample_lambdas(rng, base)

        # sample a subset of candidates for this trial (massive speedup)
        m = min(pool_sample_size, n_pool)
        idx = rng.choice(n_pool, size=m, replace=False)
        subset = [candidate_pool[i] for i in idx]

        train_metrics = _evaluate_pool(
            returns=train_rets,
            lw_cov=lw_cov,
            pool=subset,
            equity0=equity0,
            notional=notional,
            goals=goals,
            main_goal=main_goal,
            cfg=cfg,
            spec_df_full=spec_train,
            days=days,
            n_paths=n_paths_train,
            seed0=seed * 1000 + t * 10,
            path_source=path_source,
            pca_k=pca_k,
            block_size=block_size,
        )
        if not train_metrics:
            continue

        train_metrics.sort(key=lambda m: m.score, reverse=True)
        shortlist = train_metrics[: min(shortlist_size, len(train_metrics))]
        shortlist_weights = [m.weights for m in shortlist]

        valid_metrics = _evaluate_pool(
            returns=valid_rets,
            lw_cov=lw_cov,
            pool=shortlist_weights,
            equity0=equity0,
            notional=notional,
            goals=goals,
            main_goal=main_goal,
            cfg=cfg,
            spec_df_full=spec_valid,
            days=days,
            n_paths=n_paths_valid,
            seed0=seed * 2000 + t * 10,
            path_source=path_source,
            pca_k=pca_k,
            block_size=block_size,
        )

        obj = _objective_from_metrics(
            valid_metrics,
            main_goal=main_goal,
            ruin_cap=ruin_cap,
            top_k=top_k,
            alpha_ruin=alpha_ruin,
        )

        if obj > best_obj:
            best_obj = obj
            best_cfg = cfg
            best_info = {
                "trial": t,
                "objective": obj,
                "subset_size": m,
                "shortlist_size": len(shortlist_weights),
                "valid_evaluated": len(valid_metrics),
                "best_valid_score": float(max((mm.score for mm in valid_metrics), default=float("-inf"))),
                "best_valid_ruin": float(min((mm.ruin_prob_1y for mm in valid_metrics), default=float("nan"))),
            }

        if (t + 1) % 10 == 0:
            print(f"[tune] trial {t+1}/{n_trials} best_obj={best_obj:.6f}")

    if best_cfg is None:
        raise RuntimeError("No valid config found during tuning")

    return best_cfg, best_info
