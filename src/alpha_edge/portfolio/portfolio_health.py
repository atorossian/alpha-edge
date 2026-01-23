# portfolio_health.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import pandas as pd

from alpha_edge.core.schemas import PortfolioHealth, EvalMetrics
from alpha_edge.portfolio.report_engine import extract_p_hit_map  # if you want to reuse it


def build_portfolio_health(
    eval_metrics: EvalMetrics,
    as_of: pd.Timestamp,
    benchmark_ann_return: float | None = None,
) -> PortfolioHealth:
    """
    Turn today's EvalMetrics into a PortfolioHealth snapshot.
    """
    alpha = 0.0 if benchmark_ann_return is None else float(eval_metrics.ann_return - benchmark_ann_return)

    # map goals -> hit probs
    p_hit = extract_p_hit_map(eval_metrics, list(eval_metrics.goals))
    p_hit_main = float(p_hit.get(float(eval_metrics.main_goal), 0.0))

    return PortfolioHealth(
        date=as_of,
        score=float(eval_metrics.score),
        main_goal=float(eval_metrics.main_goal),
        p_hit_main_goal=p_hit_main,
        ruin_prob=float(eval_metrics.ruin_prob_1y),
        ann_return=float(eval_metrics.ann_return),
        sharpe=float(eval_metrics.sharpe),
        alpha_vs_bench=float(alpha),
    )


def should_reoptimize(
    baseline: PortfolioHealth,
    current: PortfolioHealth,
    *,
    max_score_drop: float = 0.15,   # 15%
    max_p_main_drop: float = 0.10,   # 10 percentage points
    min_alpha: float = -0.05,       # -5% annualized vs benchmark
) -> bool:
    if current.score < baseline.score * (1 - max_score_drop):
        return True
    if current.p_hit_main_goal < baseline.p_hit_main_goal - max_p_main_drop:
        return True
    if current.alpha_vs_bench < min_alpha:
        return True
    return False


# -------- baseline persistence (simple JSON on disk) --------

def _health_to_json(h: PortfolioHealth) -> dict:
    d = asdict(h)
    d["date"] = h.date.strftime("%Y-%m-%d")
    return d


def _health_from_json(d: dict) -> PortfolioHealth:
    d = dict(d)
    d["date"] = pd.to_datetime(d["date"])
    return PortfolioHealth(**d)


def save_baseline_health(h: PortfolioHealth, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_health_to_json(h), f, indent=2)


def load_baseline_health(path: str | Path) -> PortfolioHealth | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _health_from_json(data)
