# portfolio_health.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pandas as pd

from report_engine import PortfolioMetrics


@dataclass
class PortfolioHealth:
    date: pd.Timestamp
    score: float
    p_hit_2000: float
    ruin_prob: float
    ann_return: float
    sharpe: float
    alpha_vs_bench: float  # for now we set this to 0.0


def build_portfolio_health(
    metrics: PortfolioMetrics,
    as_of: pd.Timestamp,
    lambda_ruin: float = 0.5,
) -> PortfolioHealth:
    """
    Build a compact 'health' snapshot from today's PortfolioMetrics.
    Score = P(hit 2000) - lambda_ruin * P(ruin).
    """
    score = metrics.p_hit_2000_1y - lambda_ruin * metrics.ruin_prob_1y

    return PortfolioHealth(
        date=as_of,
        score=score,
        p_hit_2000=metrics.p_hit_2000_1y,
        ruin_prob=metrics.ruin_prob_1y,
        ann_return=metrics.ann_return,
        sharpe=metrics.sharpe,
        alpha_vs_bench=0.0,  # we can wire a real benchmark later
    )


def should_reoptimize(
    baseline: PortfolioHealth,
    current: PortfolioHealth,
    *,
    max_score_drop: float = 0.15,   # 15% drop in score
    max_p2000_drop: float = 0.10,   # 10 percentage points
    min_alpha: float = -0.05,       # -5% alpha, currently unused (alpha_vs_bench=0)
) -> bool:
    # 1) score deterioration
    if current.score < baseline.score * (1 - max_score_drop):
        return True

    # 2) P(hit 2000) deterioration (absolute drop)
    if current.p_hit_2000 < baseline.p_hit_2000 - max_p2000_drop:
        return True

    # 3) alpha deterioration (will matter once we compute alpha_vs_bench)
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
