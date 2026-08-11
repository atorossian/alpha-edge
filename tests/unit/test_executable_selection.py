from __future__ import annotations

from types import SimpleNamespace


from alpha_edge.jobs.run_portfolio_search import (
    _build_executable_selection_payload,
    compute_execution_quality,
    validate_final_executable,
)
from alpha_edge.core.schemas import ScoreConfig


def _metrics(*, score: float, ruin: float, mdd: float, p_main: float = 0.8, stability: float = 1.0):
    return SimpleNamespace(
        score=score,
        ruin_prob_1y=ruin,
        max_drawdown=mdd,
        cvar_95=-0.01,
        p_hit_goal_1_1y=p_main,
        p_hit_goal_2_1y=0.60,
        p_hit_goal_3_1y=0.40,
        stability_energy=stability,
        path_mdd_mean=0.15,
        cdar_95=0.30,
        p_dd_breach=0.10,
        underwater_mean=0.60,
        ttr_mean_days=80.0,
        weights={"AAA": 0.6, "BBB": -0.4},
    )


def test_final_executable_validation_accepts_good_candidate():
    theoretical = _metrics(score=0.50, ruin=0.0, mdd=-0.12, p_main=0.82, stability=1.0)
    final = _metrics(score=0.48, ruin=0.0, mdd=-0.13, p_main=0.80, stability=1.1)
    eq = compute_execution_quality(
        theoretical_metrics=theoretical,
        final_metrics=final,
        theoretical_weights={"AAA": 0.6, "BBB": -0.4},
        executable_weights={"AAA": 0.59, "BBB": -0.41},
        realized_weights_with_cash={"AAA": 0.58, "BBB": -0.40, "CASH": 0.02},
        shares={"AAA": 10, "BBB": -5},
        target_notional=1000,
        executable_gross_notional=980,
        cash_left=20,
        goals=(35000, 37000, 40000),
        main_goal=35000,
    )
    validation = validate_final_executable(
        theoretical_metrics=theoretical,
        final_metrics=final,
        execution_quality=eq,
        score_cfg=ScoreConfig(),
        goals=(35000, 37000, 40000),
        main_goal=35000,
        min_health_score=60.0,
        min_executable_score=None,
        max_score_drop=0.25,
        max_ruin_increase=0.03,
        max_p_main_drop=0.15,
        max_cash_weight=0.05,
        min_deployment_ratio=0.95,
        max_executable_mdd=0.40,
        max_executable_cdar_95=0.60,
        max_stability_energy=2.0,
        max_dropped_weight=0.04,
        max_weight_drift_l1=0.15,
    )
    assert validation["passed"] is True
    assert validation["status"] == "accepted"


def test_final_executable_validation_rejects_bad_candidate():
    theoretical = _metrics(score=0.60, ruin=0.0, mdd=-0.12, p_main=0.85, stability=1.0)
    final = _metrics(score=-0.10, ruin=0.06, mdd=-0.55, p_main=0.70, stability=3.0)
    final.cdar_95 = 0.95
    eq = compute_execution_quality(
        theoretical_metrics=theoretical,
        final_metrics=final,
        theoretical_weights={"AAA": 0.6, "BBB": -0.4},
        executable_weights={"AAA": 1.0},
        realized_weights_with_cash={"AAA": 0.96, "CASH": 0.04},
        shares={"AAA": 10, "BBB": 0},
        target_notional=1000,
        executable_gross_notional=960,
        cash_left=40,
        goals=(35000, 37000, 40000),
        main_goal=35000,
    )
    validation = validate_final_executable(
        theoretical_metrics=theoretical,
        final_metrics=final,
        execution_quality=eq,
        score_cfg=ScoreConfig(),
        goals=(35000, 37000, 40000),
        main_goal=35000,
        min_health_score=60.0,
        min_executable_score=None,
        max_score_drop=0.25,
        max_ruin_increase=0.03,
        max_p_main_drop=0.15,
        max_cash_weight=0.05,
        min_deployment_ratio=0.95,
        max_executable_mdd=0.40,
        max_executable_cdar_95=0.60,
        max_stability_energy=2.0,
        max_dropped_weight=0.04,
        max_weight_drift_l1=0.15,
    )
    assert validation["passed"] is False
    assert validation["status"] == "rejected"
    assert any("health score" in r or "max drawdown" in r for r in validation["reasons"])
    assert any("max drawdown" in r for r in validation["reasons"])


def test_executable_selection_payload_counts_and_selected_status():
    payload = _build_executable_selection_payload(
        executable_candidate_summaries=[
            {"label": "ga_best", "passed": True, "status": "accepted", "score": 0.5, "health_score": 75.0},
            {"label": "annealed", "passed": False, "status": "rejected", "score": -0.1, "health_score": 45.0},
        ],
        selected_candidate_label="ga_best",
        final_validation={"passed": True, "status": "accepted"},
        candidate_errors=[{"label": "bad", "error": "boom"}],
        executable_selection_top_k=25,
    )
    assert payload["schema_version"] == "executable_selection_v1"
    assert payload["candidate_count"] == 2
    assert payload["accepted_count"] == 1
    assert payload["rejected_count"] == 1
    assert payload["error_count"] == 1
    assert payload["selected_label"] == "ga_best"
    assert payload["selected_passed"] is True
