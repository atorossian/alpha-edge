# src/alpha_edge/risk/actuarial/diagnostic_persistence.py
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from alpha_edge.core.schemas import (
    ActuarialDiagnosticBatchReport,
    ActuarialDiagnosticReport,
    ActuarialRiskConfig,
)
from alpha_edge.risk.actuarial.diagnostic_report import (
    evaluate_many_portfolio_search_actuarial_diagnostics,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """
    Convert numpy/pandas-ish scalar values into JSON-safe Python values.
    """
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]

    return value


def diagnostic_report_to_summary_row(
    report: ActuarialDiagnosticReport,
) -> dict[str, Any]:
    """
    Flatten an ActuarialDiagnosticReport into a CSV-friendly row.

    This intentionally keeps only the key fields needed for quick inspection.
    The full report remains available in the JSON artifact.
    """
    headline = report.headline_metrics
    detail = report.detail_metrics

    return {
        "portfolio_id": report.portfolio_id,
        "run_id": report.run_id,
        "source": report.source,
        "verdict": report.verdict,
        "risk_grade": report.risk_grade,
        "risk_flags": "|".join(report.risk_flags),
        "warnings": "|".join(report.warnings),
        "initial_value": detail.get("initial_value"),
        "horizon_days": detail.get("horizon_days"),
        "n_paths": detail.get("n_paths"),
        "ruin_threshold": detail.get("ruin_threshold"),
        "ruin_probability": headline.get("ruin_probability"),
        "expected_time_to_ruin_days": detail.get("expected_time_to_ruin_days"),
        "median_time_to_ruin_days": detail.get("median_time_to_ruin_days"),
        "drawdown_limit_pct": detail.get("drawdown_limit_pct"),
        "drawdown_breach_probability": headline.get("drawdown_breach_probability"),
        "expected_max_drawdown": detail.get("expected_max_drawdown"),
        "median_max_drawdown": detail.get("median_max_drawdown"),
        "cvar_max_drawdown_95": detail.get("cvar_max_drawdown_95"),
        "goal_value": detail.get("goal_value"),
        "goal_probability": headline.get("goal_probability"),
        "probability_goal_before_ruin": headline.get("probability_goal_before_ruin"),
        "recovery_probability": detail.get("recovery_probability"),
        "median_recovery_time_days": detail.get("median_recovery_time_days"),
        "capital_required": detail.get("capital_required"),
        "capital_buffer_gap": headline.get("capital_buffer_gap"),
        "solvency_ratio": headline.get("solvency_ratio"),
        "safe_leverage_estimate": headline.get("safe_leverage_estimate"),
    }


def build_actuarial_diagnostic_batch_report(
    reports: list[ActuarialDiagnosticReport],
    *,
    run_id: Optional[str] = None,
    source: str = "portfolio_search",
    metadata: Optional[dict[str, Any]] = None,
) -> ActuarialDiagnosticBatchReport:
    """
    Build a batch-level report from many individual diagnostic reports.
    """
    report_dicts = [_json_safe(r.to_dict()) for r in reports]
    summary_rows = [_json_safe(diagnostic_report_to_summary_row(r)) for r in reports]

    resolved_run_id = run_id
    if resolved_run_id is None:
        for report in reports:
            if report.run_id is not None:
                resolved_run_id = report.run_id
                break

    batch_metadata: dict[str, Any] = {
        "module": "alpha_edge.risk.actuarial",
        "version": "v1_diagnostic_persistence",
        "created_at_utc": _utc_now_iso(),
    }
    if metadata:
        batch_metadata.update(metadata)

    return ActuarialDiagnosticBatchReport(
        run_id=resolved_run_id,
        source=source,
        n_reports=len(reports),
        reports=report_dicts,
        summary_rows=summary_rows,
        metadata=batch_metadata,
    ).validate()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(_json_safe(row))


def write_actuarial_diagnostic_batch_report(
    batch: ActuarialDiagnosticBatchReport,
    *,
    output_dir: str | Path,
    prefix: str = "actuarial_diagnostics",
) -> dict[str, str]:
    """
    Persist an actuarial diagnostic batch report to local files.

    Writes:
      - {prefix}.json
      - {prefix}_summary.csv
      - {prefix}_manifest.json

    Returns file paths as strings.
    """
    batch = batch.validate()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{prefix}.json"
    csv_path = out / f"{prefix}_summary.csv"
    manifest_path = out / f"{prefix}_manifest.json"

    batch_payload = batch.to_dict()

    manifest = {
        "run_id": batch.run_id,
        "source": batch.source,
        "n_reports": batch.n_reports,
        "created_at_utc": _utc_now_iso(),
        "files": {
            "json": str(json_path),
            "summary_csv": str(csv_path),
            "manifest_json": str(manifest_path),
        },
        "metadata": batch.metadata,
    }

    _write_json(json_path, batch_payload)
    _write_csv(csv_path, batch.summary_rows)
    _write_json(manifest_path, manifest)

    return {
        "json": str(json_path),
        "summary_csv": str(csv_path),
        "manifest_json": str(manifest_path),
    }


def evaluate_and_write_portfolio_search_actuarial_diagnostics(
    portfolio_results: list[object],
    *,
    config: ActuarialRiskConfig,
    output_dir: str | Path,
    equity_paths_key: Optional[str] = None,
    run_id: Optional[str] = None,
    prefix: str = "actuarial_diagnostics",
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """
    Evaluate portfolio-search results and persist diagnostic artifacts.

    This is the Step 7 integration point.

    It intentionally does not:
      - mutate portfolio search outputs,
      - alter scores,
      - quarantine candidates,
      - write to S3.
    """
    reports = evaluate_many_portfolio_search_actuarial_diagnostics(
        portfolio_results,
        config=config,
        equity_paths_key=equity_paths_key,
    )

    batch = build_actuarial_diagnostic_batch_report(
        reports,
        run_id=run_id,
        source="portfolio_search",
        metadata=metadata,
    )

    return write_actuarial_diagnostic_batch_report(
        batch,
        output_dir=output_dir,
        prefix=prefix,
    )