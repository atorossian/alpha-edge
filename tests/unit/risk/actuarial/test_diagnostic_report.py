# tests/unit/risk/actuarial/test_diagnostic_report.py
from __future__ import annotations

import pytest

from alpha_edge.core.schemas import (
    ActuarialRiskResult,
    ActuarialRiskConfig,
    CapitalAdequacyConfig,
    DrawdownBreachConfig,
    GoalConfig,
    RecoveryConfig,
    RuinConfig,
    SurvivalConfig,
)
from alpha_edge.risk.actuarial.diagnostic_report import (
    build_actuarial_diagnostic_report,
    evaluate_many_portfolio_search_actuarial_diagnostics,
    evaluate_portfolio_search_actuarial_diagnostic,
)
from alpha_edge.risk.actuarial.engine import evaluate_actuarial_risk


def _base_config() -> ActuarialRiskConfig:
    return ActuarialRiskConfig(
        initial_value=100.0,
        horizon_days=2,
        ruin=RuinConfig(threshold_mode="absolute", threshold_value=70.0),
        drawdown=DrawdownBreachConfig(drawdown_limit_pct=0.30),
        goal=GoalConfig(enabled=True, goal_value=115.0),
        recovery=RecoveryConfig(enabled=True, recovery_level=1.0),
        survival=SurvivalConfig(horizons_days=[1, 2]),
        capital_adequacy=CapitalAdequacyConfig(
            enabled=True,
            target_ruin_probability=0.20,
            target_drawdown_breach_probability=0.30,
            current_leverage=1.0,
            max_allowed_leverage=2.0,
        ),
    )


def test_build_actuarial_diagnostic_report_pass_case() -> None:
    paths = [
        [100.0, 105.0, 115.0],
        [100.0, 106.0, 116.0],
        [100.0, 104.0, 117.0],
    ]

    result = evaluate_actuarial_risk(paths, config=_base_config())

    report = build_actuarial_diagnostic_report(
        result,
        portfolio_id="p1",
        run_id="r1",
        source="unit_test",
    )

    assert report.portfolio_id == "p1"
    assert report.run_id == "r1"
    assert report.source == "unit_test"
    assert report.verdict == "pass"
    assert report.risk_flags == []

    assert report.headline_metrics["ruin_probability"] == pytest.approx(0.0)
    assert report.headline_metrics["goal_probability"] == pytest.approx(1.0)
    assert report.headline_metrics["safe_leverage_estimate"] == pytest.approx(2.0)

    d = report.to_dict()
    assert d["portfolio_id"] == "p1"
    assert d["metadata"]["version"] == "v1_diagnostic_report"


def test_build_actuarial_diagnostic_report_fail_case() -> None:
    paths = [
        [100.0, 80.0, 60.0],
        [100.0, 85.0, 65.0],
        [100.0, 110.0, 120.0],
    ]

    result = evaluate_actuarial_risk(paths, config=_base_config())

    report = build_actuarial_diagnostic_report(
        result,
        portfolio_id="p_bad",
        run_id="r1",
        source="unit_test",
    )

    assert report.verdict == "fail"

    assert "ruin_probability_above_target" in report.risk_flags
    assert "drawdown_breach_probability_above_target" in report.risk_flags
    assert report.headline_metrics["ruin_probability"] == pytest.approx(2 / 3)


def test_evaluate_portfolio_search_actuarial_diagnostic() -> None:
    portfolio_result = {
        "portfolio_id": "p1",
        "run_id": "r1",
        "equity_paths": [
            [100.0, 105.0, 115.0],
            [100.0, 80.0, 60.0],
            [100.0, 106.0, 116.0],
        ],
    }

    report = evaluate_portfolio_search_actuarial_diagnostic(
        portfolio_result,
        config=_base_config(),
        portfolio_id="p1",
        run_id="r1",
    )

    assert report.portfolio_id == "p1"
    assert report.run_id == "r1"
    assert report.source == "portfolio_search"
    assert report.metadata["integration_step"] == "portfolio_search_diagnostic"

    assert report.headline_metrics["ruin_probability"] == pytest.approx(1 / 3)
    assert report.result["n_paths"] == 3


def test_evaluate_many_portfolio_search_actuarial_diagnostics() -> None:
    portfolios = [
        {
            "portfolio_id": "p_good",
            "run_id": "r1",
            "equity_paths": [
                [100.0, 105.0, 115.0],
                [100.0, 106.0, 116.0],
            ],
        },
        {
            "portfolio_id": "p_bad",
            "run_id": "r1",
            "equity_paths": [
                [100.0, 80.0, 60.0],
                [100.0, 85.0, 65.0],
            ],
        },
    ]

    reports = evaluate_many_portfolio_search_actuarial_diagnostics(
        portfolios,
        config=_base_config(),
    )

    assert len(reports) == 2

    assert reports[0].portfolio_id == "p_good"
    assert reports[1].portfolio_id == "p_bad"

    assert reports[0].headline_metrics["ruin_probability"] == pytest.approx(0.0)
    assert reports[1].headline_metrics["ruin_probability"] == pytest.approx(1.0)

    assert reports[1].verdict == "fail"


def test_diagnostic_report_contains_detail_metrics() -> None:
    paths = [
        [100.0, 105.0, 115.0],
        [100.0, 80.0, 60.0],
        [100.0, 106.0, 116.0],
    ]

    result = evaluate_actuarial_risk(paths, config=_base_config())
    report = build_actuarial_diagnostic_report(result)

    assert "expected_max_drawdown" in report.detail_metrics
    assert "cvar_max_drawdown_95" in report.detail_metrics
    assert "capital_required" in report.detail_metrics
    assert "safe_leverage_estimate" in report.detail_metrics


def test_safe_leverage_below_current_leverage_is_warn_not_fail() -> None:
    cfg = ActuarialRiskConfig(
        initial_value=100.0,
        horizon_days=2,
        ruin=RuinConfig(threshold_mode="absolute", threshold_value=50.0),
        drawdown=DrawdownBreachConfig(drawdown_limit_pct=0.30),
        goal=GoalConfig(enabled=True, goal_value=105.0),
        recovery=RecoveryConfig(enabled=True, recovery_level=1.0),
        survival=SurvivalConfig(horizons_days=[1, 2]),
        capital_adequacy=CapitalAdequacyConfig(
            enabled=True,
            target_ruin_probability=0.05,
            target_drawdown_breach_probability=0.20,
            current_leverage=4.0,
            max_allowed_leverage=2.0,
        ),
    )

    paths = [
        [100.0, 103.0, 108.0],
        [100.0, 104.0, 109.0],
        [100.0, 102.0, 107.0],
    ]

    result = evaluate_actuarial_risk(paths, config=cfg)
    report = build_actuarial_diagnostic_report(result)

    assert report.risk_grade == "A"
    assert "safe_leverage_below_current_leverage" in report.risk_flags
    assert report.verdict == "warn"


def test_tail_drawdown_cvar_breach_is_warn_not_fail_when_no_hard_flags() -> None:
    result = ActuarialRiskResult(
        initial_value=100.0,
        horizon_days=2,
        n_paths=100,
        ruin_threshold=50.0,
        ruin_probability=0.0,
        drawdown_limit_pct=0.30,
        drawdown_breach_probability=0.0,
        cvar_max_drawdown_95=-0.35,
        capital_buffer_gap=50.0,
        solvency_ratio=2.0,
        safe_leverage_estimate=1.0,
        risk_grade="A",
        metadata={
            "config": {
                "capital_adequacy": {
                    "target_ruin_probability": 0.05,
                    "target_drawdown_breach_probability": 0.20,
                    "current_leverage": 1.0,
                },
                "drawdown": {
                    "drawdown_limit_pct": 0.30,
                },
            }
        },
    )

    report = build_actuarial_diagnostic_report(result)

    assert "tail_drawdown_cvar_breaches_limit" in report.risk_flags
    assert report.verdict == "warn"
