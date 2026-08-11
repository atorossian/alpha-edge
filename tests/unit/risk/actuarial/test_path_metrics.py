# tests/unit/risk/actuarial/test_path_metrics.py
from __future__ import annotations

import numpy as np
import pytest

from alpha_edge.risk.actuarial.engine import evaluate_actuarial_risk
from alpha_edge.risk.actuarial.path_metrics import (
    calculate_cvar_max_drawdown,
    calculate_drawdown_breach_probability,
    calculate_goal_probability,
    calculate_max_drawdowns,
    calculate_probability_goal_before_ruin,
    calculate_recovery_metrics,
    calculate_ruin_probability,
    calculate_survival_curve,
    first_hit_times,
    validate_equity_paths,
)
from alpha_edge.core.schemas import (
    ActuarialRiskConfig,
    DrawdownBreachConfig,
    GoalConfig,
    RecoveryConfig,
    RuinConfig,
    SurvivalConfig,
)


def test_validate_equity_paths_accepts_valid_2d_paths() -> None:
    paths = np.array(
        [
            [100.0, 101.0, 102.0],
            [100.0, 99.0, 98.0],
        ]
    )

    out = validate_equity_paths(paths, horizon_days=2, initial_value=100.0)

    assert out.shape == (2, 3)


def test_validate_equity_paths_rejects_wrong_initial_value() -> None:
    paths = np.array(
        [
            [100.0, 101.0, 102.0],
            [99.0, 99.0, 98.0],
        ]
    )

    with pytest.raises(ValueError, match="column 0 must match"):
        validate_equity_paths(paths, horizon_days=2, initial_value=100.0)


def test_first_hit_times_returns_nan_for_paths_without_event() -> None:
    paths = np.array(
        [
            [100.0, 90.0, 80.0],
            [100.0, 110.0, 120.0],
        ]
    )

    out = first_hit_times(paths, predicate=lambda x: x <= 85.0)

    assert out[0] == 2.0
    assert np.isnan(out[1])


def test_calculate_ruin_probability() -> None:
    paths = np.array(
        [
            [100.0, 90.0, 80.0, 70.0],
            [100.0, 95.0, 91.0, 89.0],
            [100.0, 105.0, 110.0, 115.0],
        ]
    )

    result = calculate_ruin_probability(paths, ruin_threshold=90.0)

    assert result.event_probability == pytest.approx(2 / 3)
    assert result.expected_time_days == pytest.approx(2.0)
    assert result.median_time_days == pytest.approx(2.0)


def test_calculate_goal_probability() -> None:
    paths = np.array(
        [
            [100.0, 110.0, 120.0],
            [100.0, 101.0, 102.0],
            [100.0, 130.0, 140.0],
        ]
    )

    result = calculate_goal_probability(paths, goal_value=120.0)

    assert result.event_probability == pytest.approx(2 / 3)
    assert result.median_time_days == pytest.approx(1.5)


def test_probability_goal_before_ruin() -> None:
    goal_times = np.array([2.0, np.nan, 1.0, 3.0, 2.0])
    ruin_times = np.array([np.nan, 1.0, 2.0, 2.0, 2.0])

    out = calculate_probability_goal_before_ruin(
        goal_first_times=goal_times,
        ruin_first_times=ruin_times,
    )

    # success:
    # path 0: goal, no ruin
    # path 2: goal before ruin
    # path 4: goal at same time as ruin -> not before
    assert out == pytest.approx(2 / 5)


def test_calculate_max_drawdowns() -> None:
    paths = np.array(
        [
            [100.0, 120.0, 90.0, 130.0],
            [100.0, 90.0, 80.0, 70.0],
        ]
    )

    max_dd = calculate_max_drawdowns(paths)

    assert max_dd[0] == pytest.approx(-0.25)
    assert max_dd[1] == pytest.approx(-0.30)


def test_drawdown_breach_probability() -> None:
    paths = np.array(
        [
            [100.0, 120.0, 90.0, 130.0],  # -25%
            [100.0, 90.0, 80.0, 70.0],  # -30%
            [100.0, 110.0, 105.0, 115.0],  # mild
        ]
    )

    out = calculate_drawdown_breach_probability(
        paths,
        drawdown_limit_pct=0.30,
    )

    assert out == pytest.approx(1 / 3)


def test_cvar_max_drawdown_uses_worst_tail() -> None:
    max_dd = np.array([-0.10, -0.20, -0.30, -0.40, -0.50])

    out = calculate_cvar_max_drawdown(max_dd, alpha=0.80)

    # Worst 20% means the most negative drawdown here.
    assert out == pytest.approx(-0.50)


def test_survival_curve_from_ruin_times() -> None:
    first_times = np.array([5.0, 10.0, np.nan, 20.0])

    curve = calculate_survival_curve(
        first_event_times=first_times,
        horizons_days=[5, 10, 15, 20],
    )

    assert curve[0].event_probability == pytest.approx(1 / 4)
    assert curve[0].survival_probability == pytest.approx(3 / 4)

    assert curve[1].event_probability == pytest.approx(2 / 4)
    assert curve[1].survival_probability == pytest.approx(2 / 4)

    assert curve[2].event_probability == pytest.approx(2 / 4)
    assert curve[2].survival_probability == pytest.approx(2 / 4)

    assert curve[3].event_probability == pytest.approx(3 / 4)
    assert curve[3].survival_probability == pytest.approx(1 / 4)


def test_recovery_metrics_after_drawdown_breach() -> None:
    paths = np.array(
        [
            [100.0, 120.0, 90.0, 120.0],  # breach -25%, recovers to prior peak
            [100.0, 120.0, 90.0, 95.0],  # breach -25%, no recovery
            [100.0, 105.0, 104.0, 106.0],  # no breach
        ]
    )

    out = calculate_recovery_metrics(
        paths,
        drawdown_limit_pct=0.20,
        recovery_level=1.0,
    )

    assert out.recovery_probability == pytest.approx(1 / 2)
    assert out.median_recovery_time_days == pytest.approx(1.0)


def test_evaluate_actuarial_risk_end_to_end_path_metrics() -> None:
    paths = np.array(
        [
            [100.0, 110.0, 120.0, 130.0, 140.0],  # reaches goal
            [100.0, 90.0, 80.0, 70.0, 60.0],  # ruin at t=3 if threshold=70
            [100.0, 105.0, 95.0, 83.0, 115.0],  # drawdown breach and recovery
            [100.0, 99.0, 98.0, 97.0, 96.0],  # slow loss, no ruin
        ]
    )

    cfg = ActuarialRiskConfig(
        initial_value=100.0,
        horizon_days=4,
        ruin=RuinConfig(threshold_mode="absolute", threshold_value=70.0),
        drawdown=DrawdownBreachConfig(drawdown_limit_pct=0.20),
        goal=GoalConfig(enabled=True, goal_value=130.0),
        recovery=RecoveryConfig(enabled=True, recovery_level=1.0),
        survival=SurvivalConfig(horizons_days=[1, 2, 3, 4]),
    )

    result = evaluate_actuarial_risk(paths, config=cfg)

    assert result.initial_value == 100.0
    assert result.horizon_days == 4
    assert result.n_paths == 4

    assert result.ruin_threshold == 70.0
    assert result.ruin_probability == pytest.approx(1 / 4)
    assert result.median_time_to_ruin_days == pytest.approx(3.0)

    assert result.goal_probability == pytest.approx(1 / 4)
    assert result.median_time_to_goal_days == pytest.approx(3.0)
    assert result.probability_goal_before_ruin == pytest.approx(1 / 4)

    assert result.drawdown_breach_probability == pytest.approx(2 / 4)
    assert result.recovery_probability == pytest.approx(1 / 2)

    assert len(result.survival_curve) == 4
    assert result.survival_curve[2].event_probability == pytest.approx(1 / 4)

    assert result.risk_grade in {"A", "B", "C", "D", "F"}

    assert result.capital_required == pytest.approx(36.55)
    assert result.capital_buffer_gap == pytest.approx(63.45)
    assert result.solvency_ratio == pytest.approx(100.0 / 36.55)
    assert result.safe_leverage_estimate == pytest.approx(2.0)


def test_evaluate_actuarial_risk_warns_when_targets_are_exceeded() -> None:
    paths = np.array(
        [
            [100.0, 80.0, 60.0],
            [100.0, 80.0, 60.0],
            [100.0, 100.0, 100.0],
        ]
    )

    cfg = ActuarialRiskConfig(
        initial_value=100.0,
        horizon_days=2,
        ruin=RuinConfig(threshold_mode="absolute", threshold_value=70.0),
        drawdown=DrawdownBreachConfig(drawdown_limit_pct=0.20),
        survival=SurvivalConfig(horizons_days=[1, 2]),
    )

    result = evaluate_actuarial_risk(paths, config=cfg)

    assert result.ruin_probability == pytest.approx(2 / 3)
    assert result.drawdown_breach_probability == pytest.approx(2 / 3)
    assert "Ruin probability exceeds target_ruin_probability." in result.warnings
    assert "Drawdown breach probability exceeds target_drawdown_breach_probability." in result.warnings
