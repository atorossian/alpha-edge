from __future__ import annotations
from typing import Any, Dict
from alpha_edge.jobs.run_portfolio_search import run_portfolio_search_asof
from alpha_edge.core.schemas import EvalMetrics

def portfolio_search_via_prod(
    *,
    returns_wide,
    as_of: str,
    equity0: float,
    notional: float,
    goals,
    main_goal: float,
    universe_csv: str,
) -> Dict[str, Any]:
    out = run_portfolio_search_asof(
        as_of=as_of,
        equity0=equity0,
        goals=goals,
        main_goal=main_goal,
        universe_csv=universe_csv,
        use_market_hmm=True,
        override_target_leverage=None,
        write_outputs=False,          # IMPORTANT in backtest
        run_dt=as_of,                 # keep run_dt aligned for reproducibility
        cache_min_years=5.0,
    )

    em = EvalMetrics(**out["best_refined"])
    return {"weights": dict(em.weights), "eval_metrics": em, "meta": out}
