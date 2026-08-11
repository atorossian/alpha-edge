# tune_score_weights_optimize.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from alpha_edge.core.schemas import ScoreConfig
from alpha_edge.portfolio.optimizer_engine import evaluate_portfolio_candidate, _spectral_profiles_df


Candidate = Dict[str, Any]


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


def _is_weight_dict(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    if "weights" in x:
        return False
    # Candidate weights are ticker -> numeric mappings.
    for v in x.values():
        try:
            float(v)
        except Exception:
            return False
    return True


def _normalize_candidate(x: Any, *, default_label: str | None = None) -> Candidate:
    """
    Normalize tuning candidates.

    Supported inputs:
      1. legacy: {ticker: weight}
      2. executable-aware: {
             "label": str,
             "weights": {ticker: realized_executable_weight},
             "notional": realized_executable_gross_notional,
             "execution_quality": {...},
             ...
         }
    """
    if _is_weight_dict(x):
        return {
            "label": default_label or "candidate",
            "weights": {str(k): float(v) for k, v in x.items()},
        }

    if isinstance(x, dict) and isinstance(x.get("weights"), dict):
        out = dict(x)
        out["label"] = str(out.get("label") or default_label or "candidate")
        out["weights"] = {str(k): float(v) for k, v in out["weights"].items()}
        if out.get("notional") is not None:
            out["notional"] = float(out["notional"])
        return out

    raise TypeError(f"Unsupported candidate format: {type(x)!r}")


def _candidate_pool_specs(candidate_pool: list[Any]) -> list[Candidate]:
    return [_normalize_candidate(c, default_label=f"candidate_{i}") for i, c in enumerate(candidate_pool)]


def _evaluate_pool(
    returns: pd.DataFrame,
    lw_cov: pd.DataFrame | None,
    pool: list[Any],
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
    weight_mode: str = "long_short",
) -> list[dict[str, Any]]:
    if spec_df_full is None:
        spec_df_full = _spectral_profiles_df(returns, bands_days=cfg.fft_bands_days)

    out: list[dict[str, Any]] = []
    for i, raw_c in enumerate(pool):
        try:
            c = _normalize_candidate(raw_c, default_label=f"candidate_{i}")
            candidate_notional = float(c.get("notional") or notional)
            if not np.isfinite(candidate_notional) or candidate_notional <= 0:
                candidate_notional = float(notional)

            m = evaluate_portfolio_candidate(
                returns=returns,
                weights=c["weights"],
                equity0=equity0,
                notional=candidate_notional,
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
                weight_mode=str(c.get("weight_mode") or weight_mode),
            )
            out.append({"metrics": m, "candidate": c})
        except Exception:
            continue
    return out


def _main_goal_probability(m: Any, main_goal: float) -> float:
    g1, g2, g3 = m.goals
    if float(main_goal) == float(g1):
        return float(m.p_hit_goal_1_1y)
    if float(main_goal) == float(g2):
        return float(m.p_hit_goal_2_1y)
    return float(m.p_hit_goal_3_1y)


def _metric_float(m: Any, name: str, default: float = 0.0) -> float:
    try:
        v = float(getattr(m, name))
    except Exception:
        return float(default)
    return v if np.isfinite(v) else float(default)


def _objective_from_metrics(
    metrics: list[Any],
    *,
    main_goal: float,
    ruin_cap: float,
    top_k: int = 5,
    alpha_ruin: float = 0.5,
    alpha_stability: float = 0.35,
    alpha_cdar: float = 0.25,
    alpha_path_mdd: float = 0.25,
    alpha_breach: float = 0.40,
    alpha_underwater: float = 0.10,
    alpha_ttr: float = 0.10,
) -> float:
    """
    Validation objective for tuning score lambdas.

    The optimizer still sorts by candidate score under the sampled ScoreConfig,
    but the objective rewards the out-of-sample properties we actually want:
    goal probability with low ruin and good path stability.
    """
    if not metrics:
        return -1e9

    metrics = sorted(metrics, key=lambda m: float(m.score), reverse=True)
    feasible = [m for m in metrics if float(m.ruin_prob_1y) <= float(ruin_cap)]
    if not feasible:
        best_ruin = float(min(m.ruin_prob_1y for m in metrics))
        return -1000.0 - 2000.0 * float(best_ruin - ruin_cap)

    top = feasible[: max(1, min(top_k, len(feasible)))]

    vals: list[float] = []
    for m in top:
        p_main = _main_goal_probability(m, main_goal)
        ruin = float(m.ruin_prob_1y)
        stability = _metric_float(m, "stability_energy", 0.0)
        cdar = _metric_float(m, "cdar_95", 0.0)
        path_mdd = _metric_float(m, "path_mdd_mean", 0.0)
        breach = _metric_float(m, "p_dd_breach", 0.0)
        underwater = _metric_float(m, "underwater_mean", 0.0)
        ttr_days = _metric_float(m, "ttr_mean_days", 0.0)
        ttr_norm = ttr_days / float(max(1, 252))

        vals.append(
            p_main
            - float(alpha_ruin) * ruin
            - float(alpha_stability) * stability
            - float(alpha_cdar) * cdar
            - float(alpha_path_mdd) * path_mdd
            - float(alpha_breach) * breach
            - float(alpha_underwater) * underwater
            - float(alpha_ttr) * ttr_norm
        )

    return float(np.mean(vals))


def _sample_lambdas(rng: np.random.Generator, base: ScoreConfig) -> ScoreConfig:
    cfg = ScoreConfig(**asdict(base))

    def logu(lo: float, hi: float) -> float:
        x = rng.uniform(np.log(lo), np.log(hi))
        return float(np.exp(x))

    # Core penalties.
    cfg.lambda_ruin = logu(1e-3, 5.0)
    cfg.lambda_mdd = logu(1e-3, 5.0)
    cfg.lambda_cvar = logu(1e-3, 5.0)
    cfg.lambda_conc = logu(1e-3, 5.0)
    cfg.lambda_corr = logu(1e-3, 5.0)
    cfg.lambda_time = logu(1e-3, 5.0)

    # FFT penalties.
    cfg.lambda_hf_ratio = logu(1e-4, 1.0)
    cfg.lambda_freq_overlap = logu(1e-4, 1.0)
    cfg.lambda_spec_entropy = logu(1e-4, 1.0)

    # Stability penalties, if the active ScoreConfig schema supports them.
    for field, lo, hi in [
        ("lambda_stability_energy", 1e-3, 5.0),
        ("lambda_path_mdd_mean", 1e-3, 5.0),
        ("lambda_cdar_95", 1e-3, 5.0),
        ("lambda_p_dd_breach", 1e-3, 5.0),
        ("lambda_underwater", 1e-3, 5.0),
        ("lambda_ttr", 1e-3, 5.0),
    ]:
        if hasattr(cfg, field):
            setattr(cfg, field, logu(lo, hi))

    return cfg


def tune_lambdas_by_optimization(
    returns: pd.DataFrame,
    lw_cov: pd.DataFrame | None,
    candidate_pool: list[Any],
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
    pool_sample_size: int = 500,
    shortlist_size: int = 60,
    top_k: int = 5,
    ruin_cap: float = 0.10,
    alpha_ruin: float = 0.5,
    alpha_stability: float = 0.35,
    alpha_cdar: float = 0.25,
    alpha_path_mdd: float = 0.25,
    alpha_breach: float = 0.40,
    alpha_underwater: float = 0.10,
    alpha_ttr: float = 0.10,
    weight_mode: str = "long_short",
    seed: int = 123,
) -> tuple[ScoreConfig, dict]:
    rng = np.random.default_rng(seed)

    train_rets, valid_rets = _split_returns_time(returns, train_frac=train_frac)

    base = ScoreConfig()

    # Precompute FFT profiles once per split for speed.
    spec_train = _spectral_profiles_df(train_rets, bands_days=base.fft_bands_days)
    spec_valid = _spectral_profiles_df(valid_rets, bands_days=base.fft_bands_days)

    best_cfg: ScoreConfig | None = None
    best_obj = -1e18
    best_info: dict = {}

    candidate_specs = _candidate_pool_specs(candidate_pool)
    n_pool = len(candidate_specs)
    if n_pool < 50:
        raise ValueError(f"candidate_pool too small; need at least ~50; got {n_pool}")

    executable_aware = any("source_weights" in c or "execution_quality" in c for c in candidate_specs)

    for t in range(n_trials):
        cfg = _sample_lambdas(rng, base)

        m = min(int(pool_sample_size), n_pool)
        idx = rng.choice(n_pool, size=m, replace=False)
        subset = [candidate_specs[i] for i in idx]

        train_rows = _evaluate_pool(
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
            weight_mode=weight_mode,
        )
        if not train_rows:
            continue

        train_rows.sort(key=lambda row: float(row["metrics"].score), reverse=True)
        shortlist_rows = train_rows[: min(int(shortlist_size), len(train_rows))]
        shortlist_candidates = [row["candidate"] for row in shortlist_rows]

        valid_rows = _evaluate_pool(
            returns=valid_rets,
            lw_cov=lw_cov,
            pool=shortlist_candidates,
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
            weight_mode=weight_mode,
        )
        valid_metrics = [row["metrics"] for row in valid_rows]

        obj = _objective_from_metrics(
            valid_metrics,
            main_goal=main_goal,
            ruin_cap=ruin_cap,
            top_k=top_k,
            alpha_ruin=alpha_ruin,
            alpha_stability=alpha_stability,
            alpha_cdar=alpha_cdar,
            alpha_path_mdd=alpha_path_mdd,
            alpha_breach=alpha_breach,
            alpha_underwater=alpha_underwater,
            alpha_ttr=alpha_ttr,
        )

        if obj > best_obj:
            best_obj = obj
            best_cfg = cfg
            best_valid = max(valid_metrics, key=lambda mm: float(mm.score), default=None)
            best_info = {
                "trial": int(t),
                "objective": float(obj),
                "subset_size": int(m),
                "shortlist_size": int(len(shortlist_candidates)),
                "valid_evaluated": int(len(valid_metrics)),
                "weight_mode": str(weight_mode),
                "executable_aware_candidate_pool": bool(executable_aware),
                "best_valid_score": float(max((mm.score for mm in valid_metrics), default=float("-inf"))),
                "best_valid_ruin": float(min((mm.ruin_prob_1y for mm in valid_metrics), default=float("nan"))),
                "best_valid_stability_energy": float(getattr(best_valid, "stability_energy", float("nan"))) if best_valid is not None else float("nan"),
                "best_valid_path_mdd_mean": float(getattr(best_valid, "path_mdd_mean", float("nan"))) if best_valid is not None else float("nan"),
                "best_valid_cdar_95": float(getattr(best_valid, "cdar_95", float("nan"))) if best_valid is not None else float("nan"),
                "best_valid_p_dd_breach": float(getattr(best_valid, "p_dd_breach", float("nan"))) if best_valid is not None else float("nan"),
                "best_valid_underwater_mean": float(getattr(best_valid, "underwater_mean", float("nan"))) if best_valid is not None else float("nan"),
                "best_valid_ttr_mean_days": float(getattr(best_valid, "ttr_mean_days", float("nan"))) if best_valid is not None else float("nan"),
                "objective_alphas": {
                    "alpha_ruin": float(alpha_ruin),
                    "alpha_stability": float(alpha_stability),
                    "alpha_cdar": float(alpha_cdar),
                    "alpha_path_mdd": float(alpha_path_mdd),
                    "alpha_breach": float(alpha_breach),
                    "alpha_underwater": float(alpha_underwater),
                    "alpha_ttr": float(alpha_ttr),
                },
            }

        if (t + 1) % 10 == 0:
            print(f"[tune] trial {t + 1}/{n_trials} best_obj={best_obj:.6f}")

    if best_cfg is None:
        raise RuntimeError("No valid config found during tuning")

    return best_cfg, best_info
