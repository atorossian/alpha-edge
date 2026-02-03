# portfolio_health.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from alpha_edge.core.schemas import PortfolioHealth, EvalMetrics
from alpha_edge.portfolio.goal_probs import extract_p_hit_map
from alpha_edge.portfolio.alpha_report import compute_alpha_report



# -----------------------------
# Health builder + reopt logic
# -----------------------------
def build_portfolio_health(
    eval_metrics: EvalMetrics,
    as_of: pd.Timestamp,
    benchmark_ann_return: float | None = None,
    *,
    port_rets: pd.Series | None = None,
    bench_rets: pd.Series | None = None,
    regime_labels: pd.Series | None = None,
) -> PortfolioHealth:
    """
    - excess_return_* are diagnostics (mean(port-bench)*252), NOT alpha
    - alpha_* are CAPM alpha (intercept), annualized
    """
    p_hit = extract_p_hit_map(eval_metrics, list(eval_metrics.goals))
    p_hit_main = float(p_hit.get(float(eval_metrics.main_goal), 0.0))

    alpha_report = None

    # Defaults
    excess_1y = float("nan")
    excess_3m = float("nan")

    alpha_1y = beta_1y = r2_1y = ir_1y = te_1y = float("nan")
    alpha_3m = beta_3m = r2_3m = ir_3m = te_3m = float("nan")

    if port_rets is not None and bench_rets is not None:
        alpha_report = compute_alpha_report(
            port_rets=port_rets,
            bench_rets=bench_rets,
            regime_labels=regime_labels,
        )
        if alpha_report.get("ok"):
            w1 = (alpha_report.get("windows") or {}).get("1y") or {}
            w3 = (alpha_report.get("windows") or {}).get("3m") or {}

            # Excess returns (diagnostics)
            excess_1y = float(w1.get("excess_return_ann", float("nan")))
            excess_3m = float(w3.get("excess_return_ann", float("nan")))

            # CAPM (true alpha)
            capm1 = w1.get("capm", {}) or {}
            alpha_1y = float(capm1.get("alpha_ann", float("nan")))
            beta_1y  = float(capm1.get("beta", float("nan")))
            r2_1y    = float(capm1.get("r2", float("nan")))

            capm3 = w3.get("capm", {}) or {}
            alpha_3m = float(capm3.get("alpha_ann", float("nan")))
            beta_3m  = float(capm3.get("beta", float("nan")))
            r2_3m    = float(capm3.get("r2", float("nan")))

            info1 = w1.get("info", {}) or {}
            ir_1y = float(info1.get("info_ratio", float("nan")))
            te_1y = float(info1.get("tracking_error_ann", float("nan")))

            info3 = w3.get("info", {}) or {}
            ir_3m = float(info3.get("info_ratio", float("nan")))
            te_3m = float(info3.get("tracking_error_ann", float("nan")))

    # Fallback for excess_1y if benchmark not available
    if not np.isfinite(excess_1y):
        if benchmark_ann_return is None:
            excess_1y = 0.0
        else:
            excess_1y = float(eval_metrics.ann_return - benchmark_ann_return)

    alpha_report_json = json.dumps(alpha_report, ensure_ascii=False) if alpha_report is not None else None

    return PortfolioHealth(
        date=as_of,
        score=float(eval_metrics.score),
        main_goal=float(eval_metrics.main_goal),
        p_hit_main_goal=p_hit_main,
        ruin_prob=float(eval_metrics.ruin_prob_1y),
        ann_return=float(eval_metrics.ann_return),
        sharpe=float(eval_metrics.sharpe),

        # legacy field: keep it, but it is EXCESS RETURN, not alpha
        alpha_vs_bench=float(excess_1y),

        alpha_report_json=alpha_report_json,

        # 1y diagnostics
        alpha_1y=(None if not np.isfinite(alpha_1y) else float(alpha_1y)),
        beta_1y=(None if not np.isfinite(beta_1y) else float(beta_1y)),
        r2_1y=(None if not np.isfinite(r2_1y) else float(r2_1y)),
        info_ratio_1y=(None if not np.isfinite(ir_1y) else float(ir_1y)),
        tracking_error_1y=(None if not np.isfinite(te_1y) else float(te_1y)),
        excess_return_1y=(None if not np.isfinite(excess_1y) else float(excess_1y)),

        # 3m diagnostics
        alpha_3m=(None if not np.isfinite(alpha_3m) else float(alpha_3m)),
        beta_3m=(None if not np.isfinite(beta_3m) else float(beta_3m)),
        r2_3m=(None if not np.isfinite(r2_3m) else float(r2_3m)),
        info_ratio_3m=(None if not np.isfinite(ir_3m) else float(ir_3m)),
        tracking_error_3m=(None if not np.isfinite(te_3m) else float(te_3m)),
        excess_return_3m=(None if not np.isfinite(excess_3m) else float(excess_3m)),
    )



def should_reoptimize(
    baseline: PortfolioHealth,
    current: PortfolioHealth,
    *,
    max_score_drop: float = 0.15,
    max_p_main_drop: float = 0.10,
    min_excess_1y: float = -0.05,
    min_capm_alpha_1y: float = -0.05,
    min_info_ratio_1y: float = -0.25,
) -> bool:
    if current.score < baseline.score * (1.0 - max_score_drop):
        return True
    if current.p_hit_main_goal < baseline.p_hit_main_goal - max_p_main_drop:
        return True

    # Always enforce excess-return sanity (available via fallback)
    if current.alpha_vs_bench < float(min_excess_1y):
        return True

    # If we have CAPM alpha / IR, enforce them too
    if current.alpha_1y is not None and current.alpha_1y < float(min_capm_alpha_1y):
        return True
    if current.info_ratio_1y is not None and current.info_ratio_1y < float(min_info_ratio_1y):
        return True

    return False


# -------- baseline persistence --------
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
