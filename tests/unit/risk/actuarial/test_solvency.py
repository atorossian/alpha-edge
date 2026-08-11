# tests/unit/risk/actuarial/test_solvency.py
from __future__ import annotations

import numpy as np
import pytest

from alpha_edge.core.schemas import (
    ActuarialRiskConfig,
    CapitalAdequacyConfig,
    DrawdownBreachConfig,
    RuinConfig,
    SurvivalConfig,
)
from alpha_edge.risk.actuarial.engine import evaluate_actuarial_risk
from alpha_edge.risk.actuarial.solvency import (
    calculate_capital_required_from_losses,
    calculate_path_capital_losses,
    calculate_solvent_capital_ratio,
    estimate_safe_leverage,
    evaluate_capital_adequacy,
)


def test_calculate_path_capital_losses() -> None:
    paths = np.array(
        [
            [100.0, 95.0, 90.0],
            [100.0, 110.0, 105.0],
            [100.0, 80.0, 120.0],
        ]
    )

    losses = calculate_path_capital_losses(paths, initial_value=100.0)

    assert losses.tolist() == pytest.approx([10.0, 0.0, 20.0])


def test_calculate_capital_required_from_losses_uses_quantile() -> None:
    losses = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    required = calculate_capital_required_from_losses(
        losses,
        target_ruin_probability=0.20,
        min_solvent_capital_ratio=1.0,
    )

    assert required == pytest.approx(32.0)


def test_calculate_capital_required_applies_min_solvent_ratio() -> None:
    losses = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    required = calculate_capital_required_from_losses(
        losses,
        target_ruin_probability=0.20,
        min_solvent_capital_ratio=1.25,
    )

    assert required == pytest.approx(40.0)


def test_calculate_solvent_capital_ratio() -> None:
    ratio = calculate_solvent_capital_ratio(
        current_capital=100.0,
        capital_required=50.0,
    )

    assert ratio == pytest.approx(2.0)


def test_calculate_solvent_capital_ratio_returns_none_when_required_is_zero() -> None:
    ratio = calculate_solvent_capital_ratio(
        current_capital=100.0,
        capital_required=0.0,
    )

    assert ratio is None


def test_estimate_safe_leverage_caps_at_max_allowed() -> None:
    safe = estimate_safe_leverage(
        current_leverage=1.0,
        solvency_ratio=3.0,
        max_allowed_leverage=2.0,
    )

    assert safe == pytest.approx(2.0)


def test_estimate_safe_leverage_can_reduce_below_current_leverage() -> None:
    safe = estimate_safe_leverage(
        current_leverage=2.0,
        solvency_ratio=0.50,
        max_allowed_leverage=2.0,
    )

    assert safe == pytest.approx(1.0)


def test_evaluate_capital_adequacy_basic() -> None:
    paths = np.array(
        [
            [100.0, 95.0, 90.0],
            [100.0, 110.0, 105.0],
            [100.0, 80.0, 120.0],
            [100.0, 70.0, 90.0],
            [100.0, 60.0, 80.0],
        ]
    )

    cfg = CapitalAdequacyConfig(
        enabled=True,
        target_ruin_probability=0.20,
        min_solvent_capital_ratio=1.0,
        current_leverage=1.0,
        max_allowed_leverage=2.0,
    )

    result = evaluate_capital_adequacy(
        paths,
        initial_value=100.0,
        config=cfg,
    )

    assert result.capital_required is not None
    assert result.capital_buffer_gap is not None
    assert result.solvency_ratio is not None
    assert result.safe_leverage_estimate is not None

    assert result.capital_required == pytest.approx(32.0)
    assert result.capital_buffer_gap == pytest.approx(68.0)
    assert result.solvency_ratio == pytest.approx(100.0 / 32.0)
    assert result.safe_leverage_estimate == pytest.approx(2.0)


def test_evaluate_capital_adequacy_disabled_returns_empty_result() -> None:
    paths = np.array(
        [
            [100.0, 95.0, 90.0],
            [100.0, 110.0, 105.0],
        ]
    )

    cfg = CapitalAdequacyConfig(enabled=False)

    result = evaluate_capital_adequacy(
        paths,
        initial_value=100.0,
        config=cfg,
    )

    assert result.capital_required is None
    assert result.capital_buffer_gap is None
    assert result.solvency_ratio is None
    assert result.safe_leverage_estimate is None
    assert result.warnings == []


def test_evaluate_actuarial_risk_populates_capital_fields() -> None:
    paths = np.array(
        [
            [100.0, 95.0, 90.0],
            [100.0, 110.0, 105.0],
            [100.0, 80.0, 120.0],
            [100.0, 70.0, 90.0],
            [100.0, 60.0, 80.0],
        ]
    )

    cfg = ActuarialRiskConfig(
        initial_value=100.0,
        horizon_days=2,
        ruin=RuinConfig(threshold_mode="absolute", threshold_value=60.0),
        drawdown=DrawdownBreachConfig(drawdown_limit_pct=0.30),
        survival=SurvivalConfig(horizons_days=[1, 2]),
        capital_adequacy=CapitalAdequacyConfig(
            enabled=True,
            target_ruin_probability=0.20,
            min_solvent_capital_ratio=1.0,
            current_leverage=1.0,
            max_allowed_leverage=2.0,
        ),
    )

    result = evaluate_actuarial_risk(paths, config=cfg)

    assert result.capital_required == pytest.approx(32.0)
    assert result.capital_buffer_gap == pytest.approx(68.0)
    assert result.solvency_ratio == pytest.approx(100.0 / 32.0)
    assert result.safe_leverage_estimate == pytest.approx(2.0)

    assert result.metadata["version"] == "v1_path_metrics_and_capital_adequacy"


def test_evaluate_actuarial_risk_warns_when_capital_buffer_is_negative() -> None:
    paths = np.array(
        [
            [100.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
        ]
    )

    cfg = ActuarialRiskConfig(
        initial_value=100.0,
        horizon_days=1,
        ruin=RuinConfig(threshold_mode="absolute", threshold_value=10.0),
        drawdown=DrawdownBreachConfig(drawdown_limit_pct=0.90),
        survival=SurvivalConfig(horizons_days=[1]),
        capital_adequacy=CapitalAdequacyConfig(
            enabled=True,
            target_ruin_probability=0.50,
            min_solvent_capital_ratio=1.25,
            current_leverage=2.0,
            max_allowed_leverage=2.0,
        ),
    )

    result = evaluate_actuarial_risk(paths, config=cfg)

    assert result.capital_required is not None
    assert result.capital_required > 100.0
    assert result.capital_buffer_gap is not None
    assert result.capital_buffer_gap < 0.0
    assert result.solvency_ratio is not None
    assert result.solvency_ratio < 1.0
    assert result.safe_leverage_estimate is not None
    assert result.safe_leverage_estimate < 2.0

    assert "Capital buffer gap is negative under the actuarial capital model." in result.warnings
    assert "Solvency ratio is below 1.0." in result.warnings
    assert "Safe leverage estimate is below current leverage." in result.warnings


def test_safe_leverage_can_flag_current_leverage_above_allowed_cap() -> None:
    paths = np.array(
        [
            [100.0, 95.0, 90.0],
            [100.0, 105.0, 110.0],
            [100.0, 90.0, 80.0],
            [100.0, 85.0, 75.0],
        ]
    )

    cfg = CapitalAdequacyConfig(
        enabled=True,
        target_ruin_probability=0.20,
        current_leverage=6.19,
        max_allowed_leverage=2.0,
    )

    result = evaluate_capital_adequacy(
        paths,
        initial_value=100.0,
        config=cfg,
    )

    assert result.safe_leverage_estimate is not None
    assert result.safe_leverage_estimate <= 2.0
    assert "safe leverage" in " ".join(result.warnings).lower()
