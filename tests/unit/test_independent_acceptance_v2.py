"""Independent acceptance contracts: no v1 inputs, fail-closed quality gate."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_edge.market.independent_acceptance_v2 import AcceptanceError, validate


def _write(df, directory, name):
    p = directory / name
    df.to_parquet(p, index=False)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fixture(tmp_path, *, mutate_gold=False, quarantine=False):
    bronze_dir = tmp_path / "bronze"
    candidate_dir = tmp_path / "candidate"
    bronze_dir.mkdir()
    candidate_dir.mkdir()
    raw = pd.DataFrame([
        dict(asset_id="A", provider_symbol="ABC", currency_original="USD",
             session_date=date, open=price, high=price+1, low=price-1,
             close=price, adj_close=price, volume=100)
        for date, price in [("2026-10-01", 10.), ("2026-10-02", 11.)]
    ])
    bronze_files = {"source_ohlcv.parquet": _write(raw, bronze_dir, "source_ohlcv.parquet"),
                    "source_fx_quotes.parquet": _write(pd.DataFrame(columns=[
                        "provider_symbol", "session_date", "quote_close"
                    ]), bronze_dir, "source_fx_quotes.parquet")}
    bronze_manifest = {"schema_version": "bronze-v2-1", "batch_id": "batch1",
                       "files": {key: {"sha256": digest} for key, digest in bronze_files.items()}}
    (bronze_dir / "manifest.json").write_text(json.dumps(bronze_manifest))
    g = raw.copy()
    g["fx_date"] = pd.to_datetime(g["session_date"])
    g["fx_to_usd"] = 1.0
    g["quality_status"] = "quarantined" if quarantine else "accepted"
    g["quality_reason"] = "invalid" if quarantine else ""
    g["close_raw_usd"] = float("nan") if quarantine else g["close"]
    g["close_adjusted_usd"] = float("nan") if quarantine else g["adj_close"]
    silver = g.copy()
    g["session_date"] = pd.to_datetime(g["session_date"])
    prev = g["close_adjusted_usd"].shift()
    g["ret_simple_close_adjusted_usd"] = g["close_adjusted_usd"] / prev - 1
    import numpy as np
    g["ret_log_close_adjusted_usd"] = np.log(g["close_adjusted_usd"] / prev)
    g["return_quality_status"] = ["unavailable", "unavailable" if quarantine else "accepted"]
    if mutate_gold:
        g.loc[1, "close_adjusted_usd"] = 900.0
    fx = pd.DataFrame(columns=["currency_original", "fx_date", "fx_to_usd",
                                "quote_close", "fx_quote_direction", "fx_quote_symbol"])
    issues = pd.DataFrame(columns=["asset_id", "session_date", "issue"])
    evidence = pd.DataFrame(columns=["fx_validation_status"])
    files = {}
    for name, frame in [
        ("silver.parquet", silver), ("gold.parquet", g), ("bronze_validated.parquet", raw),
        ("fx_normalized.parquet", fx), ("fx_evidence.parquet", evidence),
        ("session_issues.parquet", issues),
    ]:
        files[name] = _write(frame, candidate_dir, name)
    summary = candidate_dir / "asset_quality_summary.csv"
    summary.write_text("asset_id,rows\nA,2\n")
    files[summary.name] = hashlib.sha256(summary.read_bytes()).hexdigest()
    manifest = {
        "schema": "market_data_lake_v2_silver_gold_candidate_1",
        "publication_state": "candidate_only",
        "legacy_production_unchanged": True,
        "bronze_manifest_sha256": hashlib.sha256((bronze_dir / "manifest.json").read_bytes()).hexdigest(),
        "bronze_batch_id": "batch1", "source_rows": 2,
        "candidate_input_rows": 2, "silver_rows": 2, "gold_rows": 2,
        "files_sha256": files, "acceptance": "initial_contract_pass",
        "manual_correction_rows": 0,
    }
    (candidate_dir / "manifest.json").write_text(json.dumps(manifest))
    return candidate_dir, bronze_dir


def test_clean_usd_observations_pass_internal_integrity_but_not_external_gates(tmp_path):
    candidate, bronze = _fixture(tmp_path)
    failures, report = validate(candidate_dir=candidate, bronze_dir=bronze)
    assert failures.empty
    assert report["legacy_v1_used_as_reference"] is False
    assert report["internal_integrity_status"] == "passed"
    assert report["acceptance"] == "review_required"
    assert report["production_eligible"] is False


def test_detect_tampered_usd_conversion_with_rehashed_candidate(tmp_path):
    candidate, bronze = _fixture(tmp_path, mutate_gold=True)
    failures, report = validate(candidate_dir=candidate, bronze_dir=bronze)
    assert "usd_arithmetic_close_adjusted_usd" in set(failures["check"])
    assert report["acceptance"] == "blocked"


def test_quarantine_does_not_count_as_approved(tmp_path):
    candidate, bronze = _fixture(tmp_path, quarantine=True)
    failures, report = validate(candidate_dir=candidate, bronze_dir=bronze)
    assert failures.empty
    assert report["blocker_counts"]["quarantined_prices"] == 2
    assert report["acceptance"] == "blocked"


def test_bronze_manifest_identity_is_mandatory(tmp_path):
    candidate, bronze = _fixture(tmp_path)
    with (bronze / "manifest.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(AcceptanceError, match="digest"):
        validate(candidate_dir=candidate, bronze_dir=bronze)
