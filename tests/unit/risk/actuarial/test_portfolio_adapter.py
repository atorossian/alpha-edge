# tests/unit/risk/actuarial/test_portfolio_adapter.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from alpha_edge.core.schemas import (
    ActuarialRiskConfig,
    CapitalAdequacyConfig,
    DrawdownBreachConfig,
    GoalConfig,
    RecoveryConfig,
    RuinConfig,
    SurvivalConfig,
)
from alpha_edge.risk.actuarial.portfolio_adapter import (
    build_actuarial_config_from_portfolio_context,
    evaluate_many_portfolio_search_actuarial_risks,
    evaluate_portfolio_search_actuarial_risk,
    extract_equity_paths_from_portfolio_result,
    normalize_equity_paths,
)


@dataclass
class DummyPortfolioResult:
    portfolio_id: str
    run_id: str
    equity_paths: list[list[float]]


def test_normalize_equity_paths_accepts_numpy_array() -> None:
    paths = np.array(
        [
            [100.0, 105.0, 110.0],
            [100.0, 95.0, 90.0],
        ]
    )

    out = normalize_equity_paths(paths)

    assert out.shape == (2, 3)
    assert out.dtype == float


def test_normalize_equity_paths_accepts_dataframe() -> None:
    df = pd.DataFrame(
        [
            [100.0, 105.0, 110.0],
            [100.0, 95.0, 90.0],
        ]
    )

    out = normalize_equity_paths(df)

    assert out.shape == (2, 3)
    assert out[0, 0] == pytest.approx(100.0)


def test_extract_equity_paths_from_mapping_default_key() -> None:
    portfolio_result = {
        "portfolio_id": "p1",
        "equity_paths": [
            [100.0, 105.0, 110.0],
            [100.0, 95.0, 90.0],
        ],
    }

    out = extract_equity_paths_from_portfolio_result(portfolio_result)

    assert out.shape == (2, 3)


def test_extract_equity_paths_from_mapping_custom_key() -> None:
    portfolio_result = {
        "portfolio_id": "p1",
        "simulation": {
            "equity_paths": [
                [100.0, 105.0, 110.0],
                [100.0, 95.0, 90.0],
            ]
        },
    }

    out = extract_equity_paths_from_portfolio_result(
        portfolio_result,
        equity_paths_key="simulation.equity_paths",
    )

    assert out.shape == (2, 3)


def test_extract_equity_paths_from_dataclass() -> None:
    portfolio_result = DummyPortfolioResult(
        portfolio_id="p1",
        run_id="r1",
        equity_paths=[
            [100.0, 105.0, 110.0],
            [100.0, 95.0, 90.0],
        ],
    )

    out = extract_equity_paths_from_portfolio_result(portfolio_result)

    assert out.shape == (2, 3)


def test_extract_equity_paths_raises_when_missing() -> None:
    portfolio_result = {
        "portfolio_id": "p1",
        "weights": {"A": 0.5, "B": 0.5},
    }

    with pytest.raises(KeyError, match="Could not find simulated equity paths"):
        extract_equity_paths_from_portfolio_result(portfolio_result)


def test_build_actuarial_config_from_portfolio_context_without_base_config() -> None:
    cfg = build_actuarial_config_from_portfolio_context(
        initial_value=100.0,
        horizon_days=2,
        metadata={"portfolio_id": "p1"},
    )

    assert cfg.initial_value == 100.0
    assert cfg.horizon_days == 2
    assert cfg.metadata["portfolio_id"] == "p1"


def test_build_actuarial_config_from_portfolio_context_preserves_base_settings() -> None:
    base = ActuarialRiskConfig(
        initial_value=32_000.0,
        horizon_days=252,
        ruin=RuinConfig(threshold_mode="fraction_of_initial", threshold_value=0.40),
        drawdown=DrawdownBreachConfig(drawdown_limit_pct=0.25),
        goal=GoalConfig(enabled=True, goal_value=50_000.0),
        recovery=RecoveryConfig(enabled=True, recovery_level=1.0),
        survival=SurvivalConfig(horizons_days=[1, 2]),
        capital_adequacy=CapitalAdequacyConfig(
            enabled=True,
            target_ruin_probability=0.10,
            current_leverage=1.0,
            max_allowed_leverage=2.0,
        ),
        metadata={"base": "yes"},
    )

    cfg = build_actuarial_config_from_portfolio_context(
        initial_value=100.0,
        horizon_days=2,
        base_config=base,
        metadata={"portfolio_id": "p1"},
    )

    assert cfg.initial_value == 100.0
    assert cfg.horizon_days == 2
    assert cfg.ruin.threshold_value == 0.40
    assert cfg.drawdown.drawdown_limit_pct == 0.25
    assert cfg.goal.goal_value == 50_000.0
    assert cfg.metadata["base"] == "yes"
    assert cfg.metadata["portfolio_id"] == "p1"


def test_evaluate_portfolio_search_actuarial_risk() -> None:
    portfolio_result = {
        "portfolio_id": "p1",
        "run_id": "r1",
        "equity_paths": [
            [100.0, 110.0, 120.0],
            [100.0, 90.0, 60.0],
            [100.0, 105.0, 115.0],
        ],
    }

    cfg = ActuarialRiskConfig(
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
            current_leverage=1.0,
            max_allowed_leverage=2.0,
        ),
    )

    result = evaluate_portfolio_search_actuarial_risk(
        portfolio_result,
        config=cfg,
        portfolio_id="p1",
        run_id="r1",
    )

    assert result.n_paths == 3
    assert result.ruin_probability == pytest.approx(1 / 3)
    assert result.goal_probability == pytest.approx(2 / 3)
    assert result.drawdown_breach_probability == pytest.approx(1 / 3)

    assert result.capital_required is not None
    assert result.safe_leverage_estimate is not None

    assert result.metadata["integration"]["source"] == "portfolio_search"
    assert result.metadata["integration"]["portfolio_id"] == "p1"
    assert result.metadata["integration"]["run_id"] == "r1"


def test_evaluate_many_portfolio_search_actuarial_risks() -> None:
    portfolios = [
        {
            "portfolio_id": "p1",
            "run_id": "r1",
            "equity_paths": [
                [100.0, 110.0, 120.0],
                [100.0, 105.0, 115.0],
            ],
        },
        {
            "portfolio_id": "p2",
            "run_id": "r1",
            "equity_paths": [
                [100.0, 90.0, 60.0],
                [100.0, 95.0, 80.0],
            ],
        },
    ]

    cfg = ActuarialRiskConfig(
        initial_value=100.0,
        horizon_days=2,
        ruin=RuinConfig(threshold_mode="absolute", threshold_value=70.0),
        drawdown=DrawdownBreachConfig(drawdown_limit_pct=0.30),
        survival=SurvivalConfig(horizons_days=[1, 2]),
    )

    results = evaluate_many_portfolio_search_actuarial_risks(
        portfolios,
        config=cfg,
    )

    assert len(results) == 2
    assert results[0].metadata["integration"]["portfolio_id"] == "p1"
    assert results[1].metadata["integration"]["portfolio_id"] == "p2"

    assert results[0].ruin_probability == pytest.approx(0.0)
    assert results[1].ruin_probability == pytest.approx(1 / 2)
