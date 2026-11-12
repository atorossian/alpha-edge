from __future__ import annotations

from dataclasses import dataclass

from alpha_edge.core.schemas import ScoreConfig
from alpha_edge.portfolio.evaluation_service import compute_portfolio_health_score


@dataclass
class FakeMetrics:
    p_hit_goal_1_1y: float = 0.40
    p_hit_goal_2_1y: float = 0.75
    p_hit_goal_3_1y: float = 0.20
    ruin_prob_1y: float = 0.01
    max_drawdown: float = -0.10
    cvar_95: float = -0.01
    stability_energy: float = 0.10
    path_mdd_mean: float = 0.12
    cdar_95: float = 0.18
    p_dd_breach: float = 0.05
    underwater_mean: float = 0.20
    ttr_mean_days: float = 40.0
    score: float = -1.23


def test_health_score_separates_optimizer_score_from_human_score():
    payload = compute_portfolio_health_score(
        final_metrics=FakeMetrics(),
        execution_quality={
            "deployment_ratio": 1.0,
            "cash_weight": 0.0,
            "weight_drift_l1": 0.0,
            "dropped_theoretical_weight": 0.0,
        },
        score_cfg=ScoreConfig(),
        goals=(7500.0, 10000.0, 12500.0),
        main_goal=10000.0,
        max_cash_weight=0.20,
        min_deployment_ratio=1.0,
        max_executable_mdd=0.40,
        max_executable_cdar_95=0.60,
        max_stability_energy=2.0,
        max_dropped_weight=1e-12,
        max_weight_drift_l1=1e-12,
        metadata={"producer": "unit_test"},
    )

    assert 0.0 <= payload["health_score"] <= 100.0
    assert payload["optimizer_score"] == -1.23
    assert payload["raw_optimizer_score"] == -1.23
    assert payload["metadata"]["producer"] == "unit_test"
    assert payload["schema_version"] == "portfolio_health_score_v2"


def test_main_goal_probability_uses_matching_goal_bucket():
    payload = compute_portfolio_health_score(
        final_metrics=FakeMetrics(),
        execution_quality={
            "deployment_ratio": 1.0,
            "cash_weight": 0.0,
            "weight_drift_l1": 0.0,
            "dropped_theoretical_weight": 0.0,
        },
        score_cfg=ScoreConfig(),
        goals=(7500.0, 10000.0, 12500.0),
        main_goal=10000.0,
        max_cash_weight=0.20,
        min_deployment_ratio=1.0,
        max_executable_mdd=0.40,
        max_executable_cdar_95=0.60,
        max_stability_energy=2.0,
        max_dropped_weight=1e-12,
        max_weight_drift_l1=1e-12,
    )

    assert payload["component_details"]["goal_probability"] == 0.75
