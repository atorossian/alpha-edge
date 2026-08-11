# tests/unit/risk/actuarial/test_formatters.py
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
from alpha_edge.risk.actuarial.diagnostic_report import (
    build_actuarial_diagnostic_report,
)
from alpha_edge.risk.actuarial.engine import evaluate_actuarial_risk
from alpha_edge.risk.actuarial.formatters import (
    diagnostic_report_to_json_block,
    format_actuarial_diagnostic_report,
    format_actuarial_risk_result_compact,
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


def _result():
    paths = [
        [100.0, 105.0, 115.0],
        [100.0, 80.0, 60.0],
        [100.0, 106.0, 116.0],
    ]
    return evaluate_actuarial_risk(paths, config=_config())


def test_format_actuarial_risk_result_compact() -> None:
    result = _result()

    text = format_actuarial_risk_result_compact(result)

    assert "ACTUARIAL RISK SUMMARY" in text
    assert "Ruin probability:" in text
    assert "Drawdown breach probability:" in text
    assert "Safe leverage estimate:" in text


def test_format_actuarial_diagnostic_report() -> None:
    result = _result()
    report = build_actuarial_diagnostic_report(
        result,
        portfolio_id="p1",
        run_id="r1",
        source="unit_test",
    )

    text = format_actuarial_diagnostic_report(report)

    assert "ACTUARIAL RISK DIAGNOSTICS" in text
    assert "Portfolio ID:" in text
    assert "p1" in text
    assert "Run ID:" in text
    assert "r1" in text
    assert "Headline metrics:" in text
    assert "Detail metrics:" in text
    assert "Risk flags:" in text
    assert "Warnings:" in text


def test_format_actuarial_diagnostic_report_can_hide_details() -> None:
    result = _result()
    report = build_actuarial_diagnostic_report(result)

    text = format_actuarial_diagnostic_report(
        report,
        include_detail_metrics=False,
        include_flags=False,
        include_warnings=False,
    )

    assert "Headline metrics:" in text
    assert "Detail metrics:" not in text
    assert "Risk flags:" not in text
    assert "Warnings:" not in text


def test_diagnostic_report_to_json_block() -> None:
    result = _result()
    report = build_actuarial_diagnostic_report(
        result,
        portfolio_id="p1",
        run_id="r1",
        source="unit_test",
    )

    block = diagnostic_report_to_json_block(report)

    assert block["portfolio_id"] == "p1"
    assert block["run_id"] == "r1"
    assert block["source"] == "unit_test"
    assert block["verdict"] in {"pass", "warn", "fail"}
    assert "headline_metrics" in block
    assert "detail_metrics" in block
    assert "result" not in block

    assert block["headline_metrics"]["ruin_probability"] == pytest.approx(1 / 3)
