"""Financial acceptance is a separate, fail-closed diagnostic gate."""

import boto3
import pytest
from moto import mock_aws

from alpha_edge.market import audit_silver_financial_acceptance_v3 as audit
from alpha_edge.market.build_silver_adjusted_v3 import reconstruct
from alpha_edge.market.lakehouse_v3.storage import digest, json_bytes, parquet_bytes, put_immutable
from test_build_silver_adjusted_v3 import fixture
from test_consolidated_split_decisions_v3 import decision


DAY = "2025-04-02"
SHA = "a" * 64


def candidate(status=None):
    bars, cal, findings, bundle = fixture()
    bars.loc[bars.session_date.eq(DAY), "stock_splits"] = 2.0
    standard = (
        {}
        if status
        else {
            DAY: {
                "ratio": 2.0,
                "symbol": "A",
                "bronze_sha256": SHA,
                "approval_version": "v",
                "large_return_review_dates": [],
            }
        }
    )
    decisions = {DAY: decision(DAY, status)} if status else {}
    bundle.update(split_basis_approved={"a": standard}, residual_decisions={"a": decisions}, residual_version="r")
    frame, _ = reconstruct(bars, cal, findings, bundle, asset_id="a", symbol="A", source_key="key", sha=SHA)
    return frame, bars, cal, findings, standard, decisions


def assess(values):
    frame, raw, cal, findings, standard, decisions = values
    return audit.assess(
        frame,
        raw,
        standard,
        decisions,
        aid="a",
        symbol="A",
        sha=SHA,
        residual_version="r",
        calendar=cal,
        findings=findings,
    )


def test_standard_basis_pass_is_not_economic_or_quarantine_release():
    values = candidate()
    summary, events = assess(values)
    assert not summary["failure_sessions"]
    assert events[0]["outcome"] == "price_basis_and_return_pass"
    assert events[0]["economic_acceptance"] is False
    assert events[0]["quarantine_release_eligible"] is False


def test_factor_tampering_and_gap_bridging_detected():
    values = candidate()
    frame = values[0]
    frame.loc[frame.session_date.eq(DAY), "candidate_adjustment_factor"] = 1.5
    summary, events = assess(values)
    assert summary["arithmetic_failures"]["adjustment_factor"] == 1
    assert events[0]["outcome"] == "price_basis_return_review_required"
    frame.loc[frame.session_date.eq("2025-04-04"), "candidate_price_return"] = 103 / 102 - 1
    _, _, _, summary = audit.arithmetic_audit(frame, values[2], values[3])
    assert summary["arithmetic_failures"]["price_return_availability"] == 1


def test_close_only_and_deliberate_quarantine_remain_distinct():
    values = candidate("close_only_pending_integration")
    frame = values[0]
    mask = frame.session_date.eq(DAY)
    frame.loc[mask, "candidate_close_only_basis_reverified"] = True
    frame.loc[mask, "candidate_close_only_price_return"] = 102 / 101 - 1
    frame.loc[mask, "candidate_close_only_return_review"] = True
    _, events = assess(values)
    assert events[0]["outcome"] == "close_only_price_return_pass"
    assert not events[0]["quarantine_release_eligible"]
    _, held = assess(candidate("quarantined"))
    assert held[0]["outcome"] == "retain_documented_quarantine"


def test_registry_version_mismatch_fails():
    values = candidate()
    values[0].loc[values[0].session_date.eq(DAY), "candidate_split_basis_approval_version"] = "wrong"
    with pytest.raises(ValueError, match="approval version"):
        assess(values)


@mock_aws
def test_full_audit_writes_only_audit_objects_and_resumes(monkeypatch):
    frame, raw, cal, findings, standard, decisions = candidate()
    s3 = boto3.client("s3", region_name="us-east-1")
    bucket = "test-bucket"
    s3.create_bucket(Bucket=bucket)

    def store(name, f):
        blob = parquet_bytes(f)
        return put_immutable(
            s3, bucket, f"market/lake_v3/test/{name}.parquet", blob, content_type="application/vnd.apache.parquet"
        )

    raw_ref = store("raw", raw)
    cal_ref = store("cal", cal)
    findings_ref = store("findings", findings)
    blob = parquet_bytes(frame)
    run = "candidate-test"
    root = f"{audit.OUTPUT_ROOT}/runs/{run}"
    output = put_immutable(
        s3,
        bucket,
        f"{root}/assets/{digest(b'a')}/prices-{digest(blob)}.parquet",
        blob,
        content_type="application/vnd.apache.parquet",
    )
    # The same immutable Bronze hash must be present in the approval.
    standard[DAY]["bronze_sha256"] = raw_ref["sha256"]
    plan = {
        "consolidated_version": "i",
        "bundle_version": "b",
        "split_basis_approvals": ["v"],
        "deterministic_basis_approvals": [],
        "expected_split_approvals": 1,
        "residual_version": "r",
        "expected_residual_count": 0,
        "eodhd_run_id": "eod",
    }
    manifest = {
        "schema": "alpha_edge_silver_adjusted_candidate_v1",
        "run_id": run,
        "scope": "full_universe",
        "production_eligible": False,
        "quarantine_released": False,
        "execution_plan_version": "p",
        "consolidated_index_version": "i",
        "registries": {"integration": "b"},
        "split_basis_approvals": {"standard": ["v"], "deterministic": []},
        "residual_decisions": {"version": "r"},
        "assets": [{"asset_id": "a", "output": output, "candidate_rows": len(frame), "source_rows": len(raw)}],
        "statistics": {
            "assets": 1,
            "candidate_rows": len(frame),
            "candidate_returns": int(frame.candidate_total_return.notna().sum()),
            "quarantined_rows": int(frame.historical_quarantine.sum()),
            "potential_returns_before": int(frame.candidate_return_potential_before_split_basis_review.sum()),
            "potential_returns_after": int(frame.candidate_return_potential_after_split_basis_review.sum()),
        },
    }
    ref = put_immutable(s3, bucket, f"{root}/manifest.json", json_bytes(manifest), content_type="application/json")
    monkeypatch.setattr(audit.residual, "load_execution_plan", lambda *a: plan)
    monkeypatch.setattr(
        audit,
        "load_consolidated",
        lambda *a: {
            "source_fingerprint": "f",
            "assets": {"a": {"outputs": {"candidate": cal_ref, "findings": findings_ref}}},
        },
    )
    monkeypatch.setattr(audit, "load_universe", lambda *a: ([{"asset_id": "a", "yahoo_ticker": "A"}], None))
    monkeypatch.setattr(audit, "load_approved_basis", lambda *a: {"a": standard})
    monkeypatch.setattr(audit.residual, "load", lambda *a, **kw: {})
    monkeypatch.setattr(audit, "consolidated_receipt", lambda *a: {"data": raw_ref})
    args = dict(candidate_run=run, manifest_sha256=ref["sha256"], audit_run="audit-test", execute=True)
    out = audit.audit(s3, bucket, **args)
    assert out["statistics"]["events"] == 1
    assert out["statistics"]["outcomes"] == {"price_basis_and_return_pass": 1}
    assert not out["quarantine_released"]

    def no_read(*a):
        raise AssertionError("Should resume from audited checkpoint")

    monkeypatch.setattr(audit, "verified_frame", no_read)
    assert audit.audit(s3, bucket, **args) == out
