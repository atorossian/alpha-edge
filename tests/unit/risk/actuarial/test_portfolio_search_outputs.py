# tests/unit/risk/actuarial/test_portfolio_search_output.py
from __future__ import annotations

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
from alpha_edge.risk.actuarial.portfolio_search_output import (
    attach_actuarial_diagnostic_to_output_payload,
    build_portfolio_search_actuarial_diagnostic_section,
    maybe_print_actuarial_diagnostic_section,
)


def _config() -> ActuarialRiskConfig:
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


def test_build_portfolio_search_actuarial_diagnostic_section() -> None:
    portfolio_result = {
        "portfolio_id": "p1",
        "run_id": "r1",
        "equity_paths": [
            [100.0, 105.0, 115.0],
            [100.0, 80.0, 60.0],
            [100.0, 106.0, 116.0],
        ],
    }

    report, terminal_text, json_block = build_portfolio_search_actuarial_diagnostic_section(
        portfolio_result,
        config=_config(),
        portfolio_id="p1",
        run_id="r1",
    )

    assert report.portfolio_id == "p1"
    assert report.run_id == "r1"
    assert report.source == "portfolio_search"

    assert "ACTUARIAL RISK DIAGNOSTICS" in terminal_text
    assert "Ruin probability:" in terminal_text
    assert "Safe leverage estimate:" in terminal_text

    assert json_block["portfolio_id"] == "p1"
    assert json_block["run_id"] == "r1"
    assert json_block["source"] == "portfolio_search"
    assert json_block["headline_metrics"]["ruin_probability"] == pytest.approx(1 / 3)


def test_attach_actuarial_diagnostic_to_output_payload() -> None:
    payload = {
        "run_id": "r1",
        "selected_portfolio": {
            "assets": ["A", "B"],
        },
    }

    diagnostic_block = {
        "verdict": "warn",
        "risk_grade": "C",
        "headline_metrics": {
            "ruin_probability": 0.03,
        },
    }

    out = attach_actuarial_diagnostic_to_output_payload(
        payload,
        diagnostic_block=diagnostic_block,
    )

    assert "actuarial_diagnostics" in out
    assert out["actuarial_diagnostics"]["verdict"] == "warn"

    # Original object should not be mutated.
    assert "actuarial_diagnostics" not in payload


def test_attach_actuarial_diagnostic_to_output_payload_custom_key() -> None:
    payload = {"run_id": "r1"}
    diagnostic_block = {"verdict": "pass"}

    out = attach_actuarial_diagnostic_to_output_payload(
        payload,
        diagnostic_block=diagnostic_block,
        key="selected_portfolio_actuarial_diagnostics",
    )

    assert out["selected_portfolio_actuarial_diagnostics"]["verdict"] == "pass"


def test_maybe_print_actuarial_diagnostic_section_enabled(capsys) -> None:
    maybe_print_actuarial_diagnostic_section(
        "ACTUARIAL RISK DIAGNOSTICS\nexample",
        enabled=True,
    )

    captured = capsys.readouterr()

    assert "ACTUARIAL RISK DIAGNOSTICS" in captured.out
    assert "example" in captured.out


def test_maybe_print_actuarial_diagnostic_section_disabled(capsys) -> None:
    maybe_print_actuarial_diagnostic_section(
        "ACTUARIAL RISK DIAGNOSTICS\nexample",
        enabled=False,
    )

    captured = capsys.readouterr()

    assert captured.out == ""
