"""Incremental pipeline refuses unapproved baselines and leaves prod pointer alone."""
import io
import json

import pytest
from botocore.exceptions import ClientError

from alpha_edge.market import run_daily_pipeline_v2 as daily
from alpha_edge.market.production_gold_release_v2 import CURRENT, encoded, sha


class S3:
    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body


def test_no_approved_gold_baseline_fails_without_any_writes():
    s3 = S3()
    with pytest.raises(daily.IncrementalError, match="Missing required"):
        daily.run(s3=s3, bucket="alpha-edge-algo", as_of="2026-11-09",
                  execute=False)
    assert not s3.objects


def test_invalid_date_does_not_initiate_ingestion(monkeypatch):
    """An invalid date must be rejected before any S3 read or write."""
    from unittest.mock import Mock

    s3 = S3()
    get_object = Mock(side_effect=AssertionError("S3 read before date validation"))
    put_object = Mock(side_effect=AssertionError("S3 write before date validation"))
    monkeypatch.setattr(s3, "get_object", get_object)
    monkeypatch.setattr(s3, "put_object", put_object)

    with pytest.raises(daily.IncrementalError, match="Invalid as_of"):
        daily.run(
            s3=s3,
            bucket="alpha-edge-algo",
            as_of="2026-02-30",
            execute=False,
            run_id="invalid_date_test",
        )

    get_object.assert_not_called()
    put_object.assert_not_called()
    assert s3.objects == {}


def test_merge_preserves_pre_overlap_history_and_detects_revisions():
    import pandas as pd
    old = pd.DataFrame({
        "asset_id": ["A", "A", "A"],
        "session_date": ["2026-11-02", "2026-11-10", "2026-11-11"],
        "open": [10., 11., 12.], "close": [10., 11., 12.],
    })
    new = pd.DataFrame({
        "asset_id": ["A", "A"],
        "session_date": ["2026-11-10", "2026-11-12"],
        "open": [11., 13.], "close": [12., 13.],
    })
    merged, revisions = daily._merge(
        old, new, keys=("asset_id", "session_date"), overlap_start="2026-11-05")
    assert revisions == 1
    assert merged["session_date"].tolist() == [
        "2026-11-02", "2026-11-10", "2026-11-11", "2026-11-12"]
    assert float(merged.loc[merged["session_date"].eq("2026-11-10"), "close"].iloc[0]) == 12.
    assert float(merged.loc[merged["session_date"].eq("2026-11-02"), "close"].iloc[0]) == 10.


def test_merge_rejects_source_dates_outside_overlap():
    import pandas as pd
    frame = pd.DataFrame({"asset_id": ["A"], "session_date": ["2026-11-01"], "close": [1]})
    with pytest.raises(daily.IncrementalError, match="before requested overlap"):
        daily._merge(frame, frame, keys=("asset_id", "session_date"),
                     overlap_start="2026-11-05")


def test_pinned_current_release_dry_run_never_writes(monkeypatch):
    s3 = S3()
    release_key = "market/lake_v2/releases/candidates/base_release.json"
    release = {"schema": "alpha_edge_v2_release_candidate_1",
               "release_id": "base_release", "as_of": "2026-11-08",
               "expected_cohorts": 1, "expected_assets": 1, "cohorts": []}
    raw = encoded(release)
    s3.objects[release_key] = raw
    s3.objects[CURRENT] = encoded({
        "schema": "alpha_edge_v2_release_pointer_1",
        "release_id": "base_release", "release_key": release_key,
        "release_sha256": sha(raw), "as_of": "2026-11-08",
    })
    # Acceptance is owned by release controller; fixture omits large S3 tables.
    monkeypatch.setattr(daily, "validate_release", lambda *a, **kw: {"cohorts": 1})
    state = daily.run(s3=s3, bucket="alpha-edge-algo",
                      as_of="2026-11-09", execute=False)
    assert state["status"] == "dry_run"
    assert state["retrieval_end_exclusive"] == "2026-11-10"
    assert state["gold_activated"] is False
    assert len(s3.objects) == 2


def test_real_daily_pipeline_writes_only_s3_candidates_and_audit(monkeypatch):
    import pandas as pd
    import io
    from alpha_edge.market.ingest_bronze_yahoo_v2 import _parquet_bytes

    class FakeTicker:
        def __init__(self, symbol):
            self.fast_info = {"currency": "USD", "exchange": "NYQ"}
        def history(self, **kw):
            assert kw["start"] == "2026-09-27"
            assert kw["end"] == "2026-10-08"
            index = pd.date_range("2026-10-05", periods=3, freq="D", tz="UTC")
            return pd.DataFrame({
                "Open": [100., 101., 102.],
                "High": [101., 102., 103.],
                "Low": [99., 100., 101.],
                "Close": [100., 101., 102.],
                "Adj Close": [100., 101., 102.],
                "Volume": [100, 110, 120],
            }, index=index)

    s3 = S3()
    historic = pd.DataFrame({
        "asset_id": ["A", "A"], "provider_symbol": ["SPY", "SPY"],
        "currency_original": ["USD", "USD"],
        "session_date": ["2026-10-05", "2026-10-06"],
        "open": [100., 101.], "high": [101., 102.],
        "low": [99., 100.], "close": [100., 101.],
        "adj_close": [100., 101.], "volume": [100, 110],
        "source_timestamp": ["2026-10-05T00:00:00+00:00",
                             "2026-10-06T00:00:00+00:00"],
    })
    fx = pd.DataFrame(columns=[
        "provider_symbol", "session_date", "quote_close"])
    source = {}
    for name, frame in (("bronze", historic), ("fx", fx), ("gold", historic)):
        key = f"market/lake_v2/releases/data/base/{name}.parquet"
        blob = _parquet_bytes(frame)
        s3.objects[key] = blob
        source[name] = {"key": key, "sha256": sha(blob)}
    release_key = "market/lake_v2/releases/candidates/base_release.json"
    release = {
        "schema": "alpha_edge_v2_release_candidate_1",
        "release_id": "base_release", "as_of": "2026-10-06",
        "expected_cohorts": 1, "expected_assets": 1,
        "cohorts": [{
            "cohort": 1, "assets": ["A"],
            "registry": [{"asset_id": "A", "yahoo_ticker": "SPY",
                          "currency": "USD", "calendar_id": "NYSE"}],
            "fx_mapping": [], **source,
        }],
    }
    raw = encoded(release)
    s3.objects[release_key] = raw
    s3.objects[CURRENT] = encoded({
        "schema": "alpha_edge_v2_release_pointer_1",
        "release_id": "base_release", "release_key": release_key,
        "release_sha256": sha(raw), "as_of": "2026-10-06",
    })
    monkeypatch.setattr(daily, "validate_release", lambda *a, **kw: {"cohorts": 1})
    events = []
    def recorder(cfg, **kw):
        events.append(kw)
    original_pointer = s3.objects[CURRENT]
    result = daily.run(
        s3=s3, bucket="alpha-edge-algo", as_of="2026-10-07",
        execute=True, confirm_prod_write=True, run_id="daily_test_001",
        ticker_factory=FakeTicker, audit_writer=recorder,
    )
    assert result["cohorts_staged"] == 1
    assert result["gold_activated"] is False
    assert result["persistent_local_files"] == 0
    assert s3.objects[CURRENT] == original_pointer
    assert events[-1]["status"] == "review_required"
    acceptance_key = ("market/lake_v2/daily/candidates/"
                      "dt=2026-10-07/run=daily_test_001/"
                      "cohort-0001/acceptance.json")
    report = json.loads(s3.objects[acceptance_key])
    assert report["internal_integrity_status"] == "passed"
    assert report["acceptance"] == "review_required"
    assert "market/lake_v2/releases/candidates/daily_test_001.json" in s3.objects


def test_production_preflight_failure_is_recorded_as_s3_audit_event():
    s3 = S3()
    events = []
    def recorder(cfg, **kw):
        events.append(kw)
    with pytest.raises(daily.IncrementalError, match="Missing required"):
        daily.run(s3=s3, bucket="alpha-edge-algo", as_of="2026-11-09",
                  execute=True, confirm_prod_write=True,
                  run_id="daily_audit_failure_01", audit_writer=recorder)
    assert events and events[-1]["status"] == "failed"
    assert events[-1]["run_id"] == "daily_audit_failure_01"
    assert not s3.objects
