# goal_probs.py
from __future__ import annotations

from typing import Dict
from alpha_edge.core.schemas import EvalMetrics

def extract_p_hit_map(eval_metrics: EvalMetrics, goals: list[float]) -> Dict[float, float]:
    """
    Centralized helper (moved out of report_engine to avoid circular imports).
    Returns mapping {goal_value -> prob_hit_1y}.
    Adjust this if your EvalMetrics fields differ.
    """
    # Typical mapping for your 3-goal schema:
    # goals = [g1, g2, g3]
    g1, g2, g3 = goals
    return {
        float(g1): float(getattr(eval_metrics, "p_hit_goal_1_1y", 0.0)),
        float(g2): float(getattr(eval_metrics, "p_hit_goal_2_1y", 0.0)),
        float(g3): float(getattr(eval_metrics, "p_hit_goal_3_1y", 0.0)),
    }
