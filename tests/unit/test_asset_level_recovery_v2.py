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
