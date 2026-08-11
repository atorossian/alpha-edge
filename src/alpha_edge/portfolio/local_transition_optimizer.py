from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from alpha_edge.core.schemas import (
    LocalTransitionOptimizerConfig,
    LocalTransitionOptimizerResult,
    LocalTransitionCandidate,
)
from alpha_edge.portfolio.portfolio_search import (
    evaluate_weights_for_search,
    refine_portfolio_annealing,
)


def weight_turnover(
    current: dict[str, float],
    target: dict[str, float],
) -> float:
    keys = set(current.keys()) | set(target.keys())
    return float(
        0.5 * sum(
            abs(float(target.get(k, 0.0)) - float(current.get(k, 0.0)))
            for k in keys
        )
    )


def delta_weights(
    current: dict[str, float],
    target: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(current.keys()) | set(target.keys()))
    return {
        k: float(float(target.get(k, 0.0)) - float(current.get(k, 0.0)))
        for k in keys
        if abs(float(target.get(k, 0.0)) - float(current.get(k, 0.0))) > 1e-10
    }


def run_local_transition_optimizer(
    *,
    as_of: str,
    returns: pd.DataFrame,
    universe,
    current_weights: dict[str, float],
    equity0: float,
    notional: float,
    goals: tuple[float, float, float],
    main_goal: float,
    score_config,
    cfg: LocalTransitionOptimizerConfig | None = None,
    lw_cov: pd.DataFrame | None = None,
) -> LocalTransitionOptimizerResult:
    if cfg is None:
        cfg = LocalTransitionOptimizerConfig()

    rng = np.random.default_rng(int(cfg.random_seed))

    # 1. Evaluate current live portfolio using same engine as portfolio search.
    current_metrics = evaluate_weights_for_search(
        returns=returns,
        weights=current_weights,
        equity0=float(equity0),
        notional=float(notional),
        goals=list(goals),
        main_goal=float(main_goal),
        score_config=score_config,
        n_paths=int(cfg.n_paths_current),
        mc_seed=int(cfg.random_seed),
        block_size=cfg.block_size,
        weight_mode=str(cfg.weight_mode),
    )

    # 2. Run existing simulated annealing from portfolio_search.py.
    best_local = refine_portfolio_annealing(
        base_metrics=current_metrics,
        returns=returns,
        universe=universe,
        lw_cov=lw_cov,
        equity0=float(equity0),
        notional=float(notional),
        goals=list(goals),
        main_goal=float(main_goal),
        score_config=score_config,
        max_assets=int(cfg.max_assets),
        min_assets=int(cfg.min_assets),
        n_steps=int(cfg.anneal_steps),
        temp_start=float(cfg.temp_start),
        temp_end=float(cfg.temp_end),
        n_paths_init=int(cfg.n_paths_init),
        n_paths_final=int(cfg.n_paths_final),
        rng=rng,
        path_source=str(cfg.path_source),
        pca_k=cfg.pca_k,
        block_size=cfg.block_size,
        weight_mode=str(cfg.weight_mode),
    )

    turnover = weight_turnover(current_metrics.weights, best_local.weights)
    score_improvement = float(best_local.score) - float(current_metrics.score)

    candidate = LocalTransitionCandidate(
        weights={k: float(v) for k, v in best_local.weights.items()},
        score=float(best_local.score),
        health_score=None,
        turnover=float(turnover),
        score_improvement=float(score_improvement),
        health_improvement=None,
        delta_weights=delta_weights(current_metrics.weights, best_local.weights),
        metrics=asdict(best_local),
    )

    diagnostics: dict[str, Any] = {
        "method": "refine_portfolio_annealing",
        "source": "alpha_edge.portfolio.portfolio_search",
        "current_metrics": asdict(current_metrics),
        "best_local_metrics": asdict(best_local),
    }

    if turnover > float(cfg.max_turnover):
        return LocalTransitionOptimizerResult(
            as_of=str(as_of),
            recommendation="HOLD",
            reason=(
                f"Best local candidate turnover {turnover:.2%} exceeds "
                f"max_turnover {float(cfg.max_turnover):.2%}."
            ),
            current_weights={k: float(v) for k, v in current_metrics.weights.items()},
            current_score=float(current_metrics.score),
            current_health_score=None,
            best_candidate=candidate,
            candidates_evaluated=int(cfg.anneal_steps),
            candidates_accepted_by_turnover=0,
            config=cfg,
            diagnostics=diagnostics,
        )

    if score_improvement < float(cfg.min_score_improvement):
        return LocalTransitionOptimizerResult(
            as_of=str(as_of),
            recommendation="HOLD",
            reason=(
                f"Best local candidate score improvement {score_improvement:.4f} "
                f"is below min_score_improvement {float(cfg.min_score_improvement):.4f}."
            ),
            current_weights={k: float(v) for k, v in current_metrics.weights.items()},
            current_score=float(current_metrics.score),
            current_health_score=None,
            best_candidate=candidate,
            candidates_evaluated=int(cfg.anneal_steps),
            candidates_accepted_by_turnover=1,
            config=cfg,
            diagnostics=diagnostics,
        )

    return LocalTransitionOptimizerResult(
        as_of=str(as_of),
        recommendation="LOCAL_REBALANCE_RECOMMENDED",
        reason=(
            f"Existing annealing improved score by {score_improvement:.4f} "
            f"with turnover {turnover:.2%}."
        ),
        current_weights={k: float(v) for k, v in current_metrics.weights.items()},
        current_score=float(current_metrics.score),
        current_health_score=None,
        best_candidate=candidate,
        candidates_evaluated=int(cfg.anneal_steps),
        candidates_accepted_by_turnover=1,
        config=cfg,
        diagnostics=diagnostics,
    )