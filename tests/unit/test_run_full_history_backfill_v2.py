"""Full-history backfill driver must be resumable and preserve batch identity."""

import hashlib
import json

import pandas as pd
import pytest

from alpha_edge.market import run_full_history_backfill_v2 as backfill


def fixture_plan(tmp_path):
    plan = tmp_path / "plan"
    plan.mkdir()
    a = pd.DataFrame([{"asset_id": "A", "yahoo_ticker": "GLD", "currency": "USD", "calendar_id": "NYSE"}])
    a.to_csv(plan / "cohort_0001.csv", index=False)
    fx = pd.DataFrame(columns=["currency", "provider_symbol", "direction"])
    fx.to_csv(plan / "fx_config.csv", index=False)
    files = {
        name: hashlib.sha256((plan / name).read_bytes()).hexdigest() for name in ["cohort_0001.csv", "fx_config.csv"]
    }
    (plan / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "alpha_edge_lake_v2_universe_preflight_1",
                "cohorts": 1,
                "assets_included": 1,
                "files_sha256": files,
            }
        )
    )
    return plan


def test_dry_run_is_offline_and_creates_no_files(tmp_path):
    plan = fixture_plan(tmp_path)
    out = tmp_path / "output"
    summary = backfill.run_full(plan_dir=plan, output=out, end="2026-11-10")
    assert summary["status"] == "dry_run"
    assert summary["assets"] == 1
    assert not out.exists()


def test_full_history_run_can_resume_without_redownloading(tmp_path, monkeypatch):
    plan = fixture_plan(tmp_path)
    counts = {"acquire": 0, "audit": 0}

    def fake_cohort(**kwargs):
        assert kwargs["full_history"] is True
        assert kwargs["start"] is None
        counts["acquire"] += 1
        batch_id = f"{kwargs['batch_prefix']}-0001"
        path = kwargs["output"] / "bronze_batches" / batch_id
        path.mkdir(parents=True)
        (path / "manifest.json").write_text("{}")
        report = {
            "state": "complete",
            "plan_manifest_sha256": hashlib.sha256((plan / "manifest.json").read_bytes()).hexdigest(),
            "requested_end_exclusive": "2026-11-10",
        }
        (kwargs["output"] / "manifest.json").write_text(json.dumps(report))
        return report

    def fake_audit(bronze, asset, fx, output):
        counts["audit"] += 1
        output.mkdir()

    monkeypatch.setattr(backfill, "run_cohort", fake_cohort)
    monkeypatch.setattr(backfill, "audit_missing", fake_audit)
    output = tmp_path / "output"
    first = backfill.run_full(plan_dir=plan, output=output, end="2026-11-10", execute=True)
    second = backfill.run_full(plan_dir=plan, output=output, end="2026-11-10", execute=True)
    assert first["cohorts_successful"] == second["cohorts_successful"] == 1
    assert counts == {"acquire": 1, "audit": 1}
    assert second["s3_writes"] == 0
    with pytest.raises(ValueError, match="different plan or end"):
        backfill.run_full(plan_dir=plan, output=output, end="2026-11-09", execute=True)
