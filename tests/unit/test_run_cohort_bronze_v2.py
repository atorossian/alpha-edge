"""Staged cohorts require verified plans and explicit local-only execution."""
import hashlib
import json

import pandas as pd
import pytest

from alpha_edge.market.run_cohort_bronze_v2 import (
    CohortRunError, load_plan, run,
)


def plan(tmp_path):
    folder = tmp_path / "plan"
    folder.mkdir()
    files = {}
    cohorts = [
        pd.DataFrame([{"asset_id": "A", "yahoo_ticker": "GLD", "currency": "USD",
                       "calendar_id": "NYSE"}]),
        pd.DataFrame([{"asset_id": "B", "yahoo_ticker": "KESKOB.HE", "currency": "EUR",
                       "calendar_id": "XHEL"}]),
    ]
    for i, cohort in enumerate(cohorts, 1):
        name = f"cohort_{i:04d}.csv"
        cohort.to_csv(folder / name, index=False)
        files[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    fx = pd.DataFrame([{"currency": "EUR", "provider_symbol": "EURUSD=X", "direction": "direct"}])
    fx.to_csv(folder / "fx_config.csv", index=False)
    files["fx_config.csv"] = hashlib.sha256((folder / "fx_config.csv").read_bytes()).hexdigest()
    (folder / "manifest.json").write_text(json.dumps({
        "schema": "alpha_edge_lake_v2_universe_preflight_1",
        "cohorts": 2, "assets_included": 2, "files_sha256": files
    }))
    return folder


def test_dry_run_makes_no_network_calls(tmp_path):
    folder = plan(tmp_path)
    def fail(**kwargs):
        raise AssertionError("No source access allowed")
    result = run(plan_dir=folder, output=tmp_path / "run", batch_prefix="test",
                 start="2025-01-01", end="2025-01-10", runner=fail)
    assert result["status"] == "dry_run"
    assert result["planned_assets"] == 1
    assert not (tmp_path / "run").exists()


def test_explicit_execution_is_bounded_and_s3_free(tmp_path):
    folder = plan(tmp_path)
    calls = []
    def fake(**kwargs):
        calls.append(kwargs)
        assert kwargs["upload"] is False
        return {"local_path": "fake"}
    result = run(plan_dir=folder, output=tmp_path / "run", batch_prefix="test",
                 start="2025-01-01", end="2025-01-10",
                 max_cohorts=1, execute=True, retries=0, runner=fake)
    assert result["acquired_cohorts"] == 1
    assert len(calls) == 1 and calls[0]["batch_id"] == "test-0001"
    assert result["s3_writes"] == 0
    with pytest.raises(FileExistsError):
        run(plan_dir=folder, output=tmp_path / "run", batch_prefix="other",
            start="2025-01-01", end="2025-01-10",
            execute=True, retries=0, runner=fake)


def test_corrupted_plan_fails_before_execution(tmp_path):
    folder = plan(tmp_path)
    with (folder / "cohort_0001.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(CohortRunError, match="checksum"):
        load_plan(folder)


def test_failed_cohort_does_not_claim_success(tmp_path):
    folder = plan(tmp_path)
    def fail(**kwargs):
        raise RuntimeError("provider rate limited")
    report = run(plan_dir=folder, output=tmp_path / "run",
                 batch_prefix="test", start="2025-01-01", end="2025-01-10",
                 execute=True, retries=1, delay_seconds=0,
                 runner=fail, sleeper=lambda _: None)
    assert report["state"] == "partial_failed"
    assert report["failed_cohorts"] == 1
