# tests/unit/risk/actuarial/test_diagnostic_persistence.py
from __future__ import annotations

import csv
import json
from pathlib import Path

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
from alpha_edge.risk.actuarial.diagnostic_persistence import (
    build_actuarial_diagnostic_batch_report,
    diagnostic_report_to_summary_row,
    evaluate_and_write_portfolio_search_actuarial_diagnostics,
    write_actuarial_diagnostic_batch_report,
)
from alpha_edge.risk.actuarial.diagnostic_report import (
    evaluate_many_portfolio_search_actuarial_diagnostics,
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


def _portfolio_results() -> list[dict]:
    return [
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


def test_diagnostic_report_to_summary_row() -> None:
    reports = evaluate_many_portfolio_search_actuarial_diagnostics(
        _portfolio_results(),
        config=_config(),
    )

    row = diagnostic_report_to_summary_row(reports[0])

    assert row["portfolio_id"] == "p_good"
    assert row["run_id"] == "r1"
    assert row["source"] == "portfolio_search"
    assert row["ruin_probability"] == pytest.approx(0.0)
    assert "safe_leverage_estimate" in row


def test_build_actuarial_diagnostic_batch_report() -> None:
    reports = evaluate_many_portfolio_search_actuarial_diagnostics(
        _portfolio_results(),
        config=_config(),
    )

    batch = build_actuarial_diagnostic_batch_report(
        reports,
        run_id="r1",
        source="portfolio_search",
        metadata={"test": "yes"},
    )

    assert batch.run_id == "r1"
    assert batch.source == "portfolio_search"
    assert batch.n_reports == 2
    assert len(batch.reports) == 2
    assert len(batch.summary_rows) == 2
    assert batch.metadata["test"] == "yes"


def test_write_actuarial_diagnostic_batch_report(tmp_path: Path) -> None:
    reports = evaluate_many_portfolio_search_actuarial_diagnostics(
        _portfolio_results(),
        config=_config(),
    )

    batch = build_actuarial_diagnostic_batch_report(
        reports,
        run_id="r1",
        source="portfolio_search",
    )

    paths = write_actuarial_diagnostic_batch_report(
        batch,
        output_dir=tmp_path,
        prefix="test_actuarial",
    )

    json_path = Path(paths["json"])
    csv_path = Path(paths["summary_csv"])
    manifest_path = Path(paths["manifest_json"])

    assert json_path.exists()
    assert csv_path.exists()
    assert manifest_path.exists()

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["run_id"] == "r1"
    assert payload["n_reports"] == 2
    assert len(payload["reports"]) == 2
    assert len(payload["summary_rows"]) == 2

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["run_id"] == "r1"
    assert manifest["n_reports"] == 2

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["portfolio_id"] == "p_good"
    assert rows[1]["portfolio_id"] == "p_bad"


def test_evaluate_and_write_portfolio_search_actuarial_diagnostics(tmp_path: Path) -> None:
    paths = evaluate_and_write_portfolio_search_actuarial_diagnostics(
        _portfolio_results(),
        config=_config(),
        output_dir=tmp_path,
        run_id="r1",
        prefix="portfolio_search_actuarial",
        metadata={"integration": "unit_test"},
    )

    json_path = Path(paths["json"])
    csv_path = Path(paths["summary_csv"])
    manifest_path = Path(paths["manifest_json"])

    assert json_path.exists()
    assert csv_path.exists()
    assert manifest_path.exists()

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["run_id"] == "r1"
    assert payload["source"] == "portfolio_search"
    assert payload["n_reports"] == 2
    assert payload["metadata"]["integration"] == "unit_test"

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["portfolio_id"] == "p_good"
    assert rows[1]["portfolio_id"] == "p_bad"
