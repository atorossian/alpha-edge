"""Single-asset acquisition is immutable, isolated and never auto-approved."""

import io
import json

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from alpha_edge.market import asset_level_recovery_v2 as recovery


class FakeS3:
    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType, IfNoneMatch):
        assert IfNoneMatch == "*"
        if Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body


def test_session_audit_exposes_gaps_and_never_approves_endpoints():
    bars = pd.DataFrame(
        [
            {"asset_id": "A", "provider_symbol": "A", "session_date": "2026-11-01",
             "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100,
             "dividends": 0, "stock_splits": 0},
            {"asset_id": "A", "provider_symbol": "A", "session_date": "2026-11-03",
             "open": 11, "high": 12, "low": 10, "close": 11, "volume": 100,
             "dividends": 1, "stock_splits": 0},
        ]
    )
    findings, summary = recovery.audit_sessions_and_actions(
        bars, asset_id="A", calendar_id="24/7"
    )
    assert summary["missing_internal_sessions"] == 1
    assert summary["action_observation_counts"]["dividends"] == 1
    assert not summary["historical_coverage_approved"]
    assert not summary["listing_dates_independently_verified"]
    assert findings["session_date"].eq("2026-11-02").any()


def test_duplicate_observed_sessions_fail_closed():
    bars = pd.DataFrame(
        [
            {"asset_id": "A", "provider_symbol": "A", "session_date": "2026-11-01",
             "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
            {"asset_id": "A", "provider_symbol": "A", "session_date": "2026-11-01",
             "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100},
        ]
    )
    with pytest.raises(recovery.AssetRecoveryError, match="Duplicate observed sessions"):
        recovery.audit_sessions_and_actions(bars, asset_id="A", calendar_id="24/7")


def test_recovery_is_dry_run_by_default(monkeypatch):
    monkeypatch.setattr(
        recovery, "_original_asset",
        lambda *args: (
            {"original_cohort": 3, "asset_id": "A", "yahoo_ticker": "A",
             "currency": "USD", "calendar_id": "24/7"},
            {"plan_sha256": "f" * 64},
            {3: "missing"},
        ),
    )
    s3 = FakeS3()
    result = recovery.recover_asset(s3, "bucket", "archive", "A", "2026-11-10")
    assert result["status"] == "dry_run"
    assert s3.objects == {}


def test_existing_cohort_cannot_be_recovered_twice(monkeypatch):
    monkeypatch.setattr(
        recovery, "_original_asset",
        lambda *args: (
            {"original_cohort": 3, "asset_id": "A", "yahoo_ticker": "A",
             "currency": "USD", "calendar_id": "24/7"},
            {"plan_sha256": "f" * 64},
            {3: "marker_present_unverified"},
        ),
    )
    with pytest.raises(recovery.AssetRecoveryError, match="not missing"):
        recovery.recover_asset(FakeS3(), "bucket", "archive", "A", "2026-11-10", execute=True)


def test_acquisition_is_idempotent_and_recovered_bronze_can_be_audited(monkeypatch):
    monkeypatch.setattr(
        recovery, "_original_asset",
        lambda *args: (
            {"original_cohort": 3, "asset_id": "A", "yahoo_ticker": "A",
             "currency": "USD", "calendar_id": "24/7"},
            {"plan_sha256": "f" * 64},
            {3: "missing"},
        ),
    )
    calls = []

    def fake_acquire(asset_id, symbol, *, start, end, batch_id, ticker_factory, currency_hint, full_history):
        calls.append((asset_id, symbol, end))
        assert start is None and full_history
        assert currency_hint == "USD"
        frame = pd.DataFrame(
            [{"asset_id": "A", "provider_symbol": "A", "session_date": "2026-11-01",
              "open": 10, "high": 10, "low": 10, "close": 10, "volume": 123,
              "dividends": 0, "stock_splits": 0}]
        )
        return frame, {"provider": "test", "retrieved_at_utc": "2026-11-11T00:00:00+00:00"}

    monkeypatch.setattr(recovery, "acquire_yahoo_bars", fake_acquire)
    store = FakeS3()
    first = recovery.recover_asset(store, "bucket", "archive", "A", "2026-11-10", execute=True)
    assert first["status"] == "published_research_candidate_not_approved"
    second = recovery.recover_asset(store, "bucket", "archive", "A", "2026-11-10", execute=True)
    assert second["status"] == "verified_existing_research_candidate"
    assert len(calls) == 1

    result = recovery.audit_recovered_asset(store, "bucket", "A", first["run_id"], publish=True)
    assert result["historical_coverage_approved"] is False
    assert result["evidence"]["status"] == "verified_immutable_s3"
    assert any(k.endswith("/coverage.json") for k in store.objects)



def test_failed_provider_acquisition_is_recorded_but_not_approved(monkeypatch):
    monkeypatch.setattr(
        recovery, "_original_asset",
        lambda *args: (
            {"original_cohort": 3, "asset_id": "A", "yahoo_ticker": "A",
             "currency": "USD", "calendar_id": "24/7"},
            {"plan_sha256": "f" * 64},
            {3: "missing"},
        ),
    )

    def fail(*args, **kwargs):
        raise RuntimeError("provider history unavailable")

    monkeypatch.setattr(recovery, "acquire_yahoo_bars", fail)
    store = FakeS3()
    result = recovery.recover_asset(store, "bucket", "archive", "A", "2026-11-10", execute=True)
    assert result["status"] == "acquisition_failed"
    assert result["production_eligible"] is False
    assert any(key.endswith("/failure.json") for key in store.objects)
    assert all(not key.endswith("/source_ohlcv.parquet") for key in store.objects)


def test_unverified_corporate_action_observations_cannot_approve_prices():
    bars = pd.DataFrame(
        [{"asset_id": "A", "provider_symbol": "A", "session_date": "2026-11-01",
          "open": 10, "high": 10, "low": 10, "close": 10, "volume": 100,
          "dividends": 0.5, "stock_splits": 2}]
    )
    findings, summary = recovery.audit_sessions_and_actions(
        bars, asset_id="A", calendar_id="24/7",
        declared_listing_start="2026-11-01", declared_listing_end="2026-11-01",
    )
    assert summary["action_observation_counts"] == {"dividends": 1, "stock_splits": 1}
    assert summary["listed_history_endpoints_declared"]
    assert not summary["listing_dates_independently_verified"]
    assert not summary["corporate_actions_independently_verified"]
    assert not summary["historical_coverage_approved"]


def _seed_batch_queue(store, *, mode, plan_id="a" * 64, reconciliation_id="b" * 64):
    from alpha_edge.market.universe_registry_v2 import _sha

    filename = "missing_cohort_asset_recovery.jsonl" if mode == "recover" else "historical_coverage_queue.jsonl"
    task = {"asset_id": "A", "original_cohort": 3, "original_provider_symbol": "A"}
    payload = (json.dumps(task, sort_keys=True) + "\n").encode("utf-8")
    store.objects[f"dev/market/lake_v2/asset_plans/run-{plan_id}/{filename}"] = payload
    store.objects[f"dev/market/lake_v2/asset_plans/run-{plan_id}/manifest.json"] = json.dumps(
        {
            "schema": "aea87_asset_level_planning_v1",
            "run_id": plan_id,
            "artifact_sha256": {filename: _sha(payload)},
            "source_fingerprints": {"reconciliation_id": reconciliation_id},
        }
    ).encode()
    store.objects[
        f"dev/market/lake_v2/reconciliation/run-{reconciliation_id}/manifest.json"
    ] = json.dumps(
        {"source_fingerprints": {"original_plan": {"plan_sha256": "f" * 64}}}
    ).encode()
    return plan_id


def test_bounded_batch_dry_run_has_no_s3_writes():
    store = FakeS3()
    plan_id = _seed_batch_queue(store, mode="recover")
    before = set(store.objects)
    result = recovery.run_asset_batch(
        store, "bucket", "archive", plan_id, "2026-11-10",
        mode="recover", start_index=0, max_assets=1, execute=False,
    )
    assert result["selected_asset_ids"] == ["A"]
    assert result["s3_writes"] == 0
    assert set(store.objects) == before


def test_bounded_batch_reuses_plan_once_and_publishes_results(monkeypatch):
    store = FakeS3()
    plan_id = _seed_batch_queue(store, mode="recover")
    seen = []

    def load_plan(*args):
        seen.append("plan")
        return pd.DataFrame([{
            "asset_id": "A", "original_cohort": 3, "yahoo_ticker": "A",
            "currency": "USD", "calendar_id": "24/7",
        }]), {"plan_sha256": "f" * 64, "expected_cohorts": 3}

    monkeypatch.setattr(recovery, "load_archived_plan", load_plan)
    monkeypatch.setattr(recovery, "cohort_commit_presence", lambda *args: {3: "missing"})

    def fake_recover(*args, **kwargs):
        assert kwargs["_context"] is not None
        return {"status": "verified_existing_research_candidate",
                "evidence": {"s3_uri": "s3://bucket/verified"}}

    monkeypatch.setattr(recovery, "recover_asset", fake_recover)
    result = recovery.run_asset_batch(
        store, "bucket", "archive", plan_id, "2026-11-10",
        mode="recover", start_index=0, max_assets=1, execute=True,
    )
    assert seen == ["plan"]
    assert result["status"] == "batch_recorded"
    assert result["outcome_counts"] == {"verified_existing_research_candidate": 1}
    assert any(k.endswith("/batch_results.jsonl") for k in store.objects)
    assert result["production_eligible"] is False


def test_bounded_audit_identifies_unrecovered_cohort(monkeypatch):
    store = FakeS3()
    plan_id = _seed_batch_queue(store, mode="audit")
    monkeypatch.setattr(
        recovery, "load_archived_plan",
        lambda *args: (
            pd.DataFrame([{
                "asset_id": "A", "original_cohort": 3, "yahoo_ticker": "A",
                "currency": "USD", "calendar_id": "24/7",
            }]),
            {"plan_sha256": "f" * 64, "expected_cohorts": 3},
        ),
    )
    monkeypatch.setattr(recovery, "cohort_commit_presence", lambda *args: {3: "missing"})
    result = recovery.run_asset_batch(
        store, "bucket", "archive", plan_id, "2026-11-10",
        mode="audit", start_index=0, max_assets=1, execute=True,
    )
    assert result["status"] == "batch_pending_recovery"
    assert result["outcome_counts"] == {"awaiting_individual_source_recovery": 1}
    assert result["production_eligible"] is False


def test_bounded_batch_refuses_tampered_queue():
    store = FakeS3()
    plan_id = _seed_batch_queue(store, mode="recover")
    key = next(k for k in store.objects if k.endswith("missing_cohort_asset_recovery.jsonl"))
    store.objects[key] = b'{"asset_id": "UNKNOWN"}\n'
    with pytest.raises(recovery.AssetRecoveryError, match="checksum mismatch"):
        recovery.run_asset_batch(
            store, "bucket", "archive", plan_id, "2026-11-10",
            mode="recover", start_index=0, max_assets=1, execute=False,
        )
