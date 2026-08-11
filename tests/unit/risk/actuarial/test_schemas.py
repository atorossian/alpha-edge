# tests/unit/risk/actuarial/test_schemas.py
from __future__ import annotations

import pytest

from alpha_edge.core.schemas import (
    ActuarialRiskConfig,
    ActuarialRiskResult,
    CapitalAdequacyConfig,
    DrawdownBreachConfig,
    GoalConfig,
    RuinConfig,
    SurvivalConfig,
    SurvivalCurvePoint,
)


def test_capital_adequacy_config_allows_current_leverage_above_max_allowed() -> None:
    cfg = CapitalAdequacyConfig(
        current_leverage=6.19,
        max_allowed_leverage=2.0,
    )

    assert cfg.validate() is cfg


def test_default_actuarial_risk_config_validates() -> None:
    cfg = ActuarialRiskConfig(
        initial_value=32_000.0,
        horizon_days=252,
        survival=SurvivalConfig(horizons_days=[21, 63, 126, 252]),
    )

    out = cfg.validate()

    assert out.initial_value == 32_000.0
    assert out.horizon_days == 252
    assert out.ruin_threshold_value() == 16_000.0


def test_absolute_ruin_threshold_resolves_directly() -> None:
    cfg = ActuarialRiskConfig(
        initial_value=32_000.0,
        horizon_days=252,
        ruin=RuinConfig(threshold_mode="absolute", threshold_value=10_000.0),
        survival=SurvivalConfig(horizons_days=[21, 252]),
    )

    cfg.validate()

    assert cfg.ruin_threshold_value() == 10_000.0


def test_fractional_ruin_threshold_cannot_exceed_one() -> None:
    cfg = ActuarialRiskConfig(
        initial_value=32_000.0,
        horizon_days=252,
        ruin=RuinConfig(threshold_mode="fraction_of_initial", threshold_value=1.25),
        survival=SurvivalConfig(horizons_days=[21, 252]),
    )

    with pytest.raises(ValueError, match="threshold_value must be <= 1"):
        cfg.validate()


def test_goal_requires_goal_value_when_enabled() -> None:
    cfg = ActuarialRiskConfig(
        initial_value=32_000.0,
        horizon_days=252,
        goal=GoalConfig(enabled=True, goal_value=None),
        survival=SurvivalConfig(horizons_days=[21, 252]),
    )

    with pytest.raises(ValueError, match="goal_value is required"):
        cfg.validate()


def test_goal_can_be_disabled_without_goal_value() -> None:
    cfg = ActuarialRiskConfig(
        initial_value=32_000.0,
        horizon_days=252,
        goal=GoalConfig(enabled=False, goal_value=None),
        survival=SurvivalConfig(horizons_days=[21, 252]),
    )

    cfg.validate()

    assert cfg.goal.enabled is False
    assert cfg.goal.goal_value is None


def test_drawdown_limit_must_be_probability_like() -> None:
    cfg = ActuarialRiskConfig(
        initial_value=32_000.0,
        horizon_days=252,
        drawdown=DrawdownBreachConfig(drawdown_limit_pct=1.50),
        survival=SurvivalConfig(horizons_days=[21, 252]),
    )

    with pytest.raises(ValueError, match="drawdown_limit_pct"):
        cfg.validate()


def test_survival_horizons_must_be_sorted() -> None:
    cfg = ActuarialRiskConfig(
        initial_value=32_000.0,
        horizon_days=252,
        survival=SurvivalConfig(horizons_days=[63, 21, 252]),
    )

    with pytest.raises(ValueError, match="sorted ascending"):
        cfg.validate()


def test_survival_horizons_cannot_exceed_main_horizon() -> None:
    cfg = ActuarialRiskConfig(
        initial_value=32_000.0,
        horizon_days=252,
        survival=SurvivalConfig(horizons_days=[21, 252, 756]),
    )

    with pytest.raises(ValueError, match="cannot exceed config.horizon_days"):
        cfg.validate()


def test_actuarial_risk_config_allows_current_leverage_above_policy_cap() -> None:
    cfg = ActuarialRiskConfig(
        initial_value=32_000.0,
        horizon_days=252,
        capital_adequacy=CapitalAdequacyConfig(
            current_leverage=2.50,
            max_allowed_leverage=2.00,
        ),
        survival=SurvivalConfig(horizons_days=[21, 252]),
    )

    assert cfg.validate() is cfg


def test_result_schema_validates() -> None:
    result = ActuarialRiskResult(
        initial_value=32_000.0,
        horizon_days=252,
        n_paths=10_000,
        ruin_threshold=16_000.0,
        ruin_probability=0.04,
        drawdown_limit_pct=0.30,
        drawdown_breach_probability=0.18,
        goal_value=50_000.0,
        goal_probability=0.22,
        risk_grade="B",
        survival_curve=[
            SurvivalCurvePoint(
                horizon_days=21,
                survival_probability=0.99,
                event_probability=0.01,
            ),
            SurvivalCurvePoint(
                horizon_days=252,
                survival_probability=0.96,
                event_probability=0.04,
            ),
        ],
    )

    out = result.validate()

    assert out.n_paths == 10_000
    assert out.ruin_probability == 0.04
    assert out.survival_curve[0].horizon_days == 21


def test_result_rejects_invalid_probability() -> None:
    result = ActuarialRiskResult(
        initial_value=32_000.0,
        horizon_days=252,
        n_paths=10_000,
        ruin_probability=1.25,
    )

    with pytest.raises(ValueError, match="ruin_probability"):
        result.validate()


def test_result_to_dict_serializes_survival_curve() -> None:
    result = ActuarialRiskResult(
        initial_value=32_000.0,
        horizon_days=252,
        n_paths=100,
        survival_curve=[
            SurvivalCurvePoint(
                horizon_days=252,
                survival_probability=0.95,
                event_probability=0.05,
            )
        ],
    )

    d = result.to_dict()

    assert d["initial_value"] == 32_000.0
    assert d["survival_curve"][0]["horizon_days"] == 252
    assert d["survival_curve"][0]["survival_probability"] == 0.95
