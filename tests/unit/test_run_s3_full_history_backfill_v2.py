"""S3 research backfill verifies bytes and never changes production pointers."""
import hashlib
import io
import json

import pytest
from botocore.exceptions import ClientError

from alpha_edge.market.run_s3_full_history_backfill_v2 import (
    upload_verified, verify_commit, S3BackfillError,
)


class FakeS3:
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

    def upload_file(self, Filename, Bucket, Key):
        self.objects[Key] = open(Filename, "rb").read()


def test_s3_commit_contains_verified_objects_and_resumes(tmp_path):
    root = tmp_path / "ephemeral"
    root.mkdir()
    (root / "bronze").mkdir()
    (root / "bronze" / "bar.parquet").write_bytes(b"immutable source")
    (root / "audit.csv").write_text("asset,missing\nA,0\n")
    s3 = FakeS3()
    prefix = "dev/market/lake_v2/backfills/abc/end-2026-11-10/cohort-0001"
    result = upload_verified(
        s3, bucket="alpha-edge-algo", prefix=prefix,
        local_dir=root, plan_hash="a" * 64, end="2026-11-10",
    )
    assert len(result["files"]) == 2
    assert verify_commit(s3, "alpha-edge-algo", prefix, "a" * 64) == result
    assert all("/current.json" not in k and k.startswith("dev/market/") for k in s3.objects)
    with pytest.raises(S3BackfillError, match="already committed"):
        upload_verified(s3, bucket="alpha-edge-algo", prefix=prefix,
                        local_dir=root, plan_hash="a" * 64, end="2026-11-10")


def test_s3_tampering_detected_before_cohort_is_reused(tmp_path):
    root = tmp_path / "ephemeral"
    root.mkdir()
    (root / "bars").write_bytes(b"first")
    s3 = FakeS3()
    prefix = "dev/market/lake_v2/backfills/test/end-2026-11-10/cohort-0001"
    upload_verified(s3, bucket="alpha-edge-algo", prefix=prefix,
                    local_dir=root, plan_hash="b" * 64, end="2026-11-10")
    s3.objects[f"{prefix}/files/bars"] = b"tampered"
    with pytest.raises(S3BackfillError, match="checksum mismatch"):
        verify_commit(s3, "alpha-edge-algo", prefix, "b" * 64)


def test_full_s3_runner_uses_ephemeral_disk_and_skips_verified_cohort(tmp_path, monkeypatch):
    from alpha_edge.market import run_s3_full_history_backfill_v2 as module
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "cohort_0001.csv").write_text(
        "asset_id,yahoo_ticker,currency,calendar_id\nA,GLD,USD,NYSE\n")
    (plan / "fx_config.csv").write_text("currency,provider_symbol,direction\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in plan.iterdir()}
    (plan / "manifest.json").write_text(json.dumps({
        "schema": "alpha_edge_lake_v2_universe_preflight_1",
        "cohorts": 1, "assets_included": 1, "files_sha256": hashes,
    }))
    seen_temp = []
    count = {"acquire": 0, "audit": 0, "silver": 0, "acceptance": 0}
    def fake_cohort(**kw):
        count["acquire"] += 1
        assert kw["full_history"] is True and kw["start"] is None
        root = kw["output"] / "bronze_batches" / "s3bf-0001-0001"
        root.mkdir(parents=True)
        (root / "manifest.json").write_text("{}")
        seen_temp.append(kw["output"].parent)
        return {"state": "complete"}
    def fake_audit(*args):
        count["audit"] += 1
        args[-1].mkdir()
        (args[-1] / "audit.csv").write_text("field,issues\n")
    def fake_silver(*args, **kwargs):
        count["silver"] += 1
        kwargs["output"].mkdir()
        (kwargs["output"] / "gold.parquet").write_bytes(b"gold")
    def fake_acceptance(**kw):
        count["acceptance"] += 1
        kw["output"].mkdir()
        (kw["output"] / "report.json").write_text('{"acceptance":"blocked"}')
    monkeypatch.setattr(module, "run_cohort", fake_cohort)
    monkeypatch.setattr(module, "audit_missing", fake_audit)
    monkeypatch.setattr(module, "standardize", fake_silver)
    monkeypatch.setattr(module, "acceptance", fake_acceptance)
    s3 = FakeS3()
    kw = dict(plan_dir=plan, bucket="test", end="2026-11-10",
              execute=True, start_cohort=1, max_cohorts=1, s3=s3)
    first = module.run(**kw)
    second = module.run(**kw)
    assert first["s3_verified_cohorts"] == second["s3_verified_cohorts"] == 1
    assert count == {"acquire": 1, "audit": 1, "silver": 1, "acceptance": 1}
    assert all(not path.exists() for path in seen_temp)
    assert not any("/current.json" in path for path in s3.objects)
