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
        daily.run(s3=s3, bucket="alpha-edge-algo", as_of="2026-11-11",
                  execute=False)
    assert not s3.objects


def test_invalid_date_does_not_initiate_ingestion():
    s3 = S3()
    with pytest.raises(daily.IncrementalError, match="Invalid as_of"):
        daily._day("2026-02-30")


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
               "release_id": "base_release", "as_of": "2026-11-10",
               "expected_cohorts": 1, "expected_assets": 1, "cohorts": []}
    raw = encoded(release)
    s3.objects[release_key] = raw
    s3.objects[CURRENT] = encoded({
        "schema": "alpha_edge_v2_release_pointer_1",
        "release_id": "base_release", "release_key": release_key,
        "release_sha256": sha(raw), "as_of": "2026-11-10",
    })
    # Acceptance is owned by release controller; fixture omits large S3 tables.
    monkeypatch.setattr(daily, "validate_release", lambda *a, **kw: {"cohorts": 1})
    state = daily.run(s3=s3, bucket="alpha-edge-algo",
                      as_of="2026-11-11", execute=False)
    assert state["status"] == "dry_run"
    assert state["retrieval_end_exclusive"] == "2026-11-12"
    assert state["gold_activated"] is False
    assert len(s3.objects) == 2
