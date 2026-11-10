"""Production Bronze publication is immutable and never changes any consumer pointer."""
import io
import hashlib
import json

from botocore.exceptions import ClientError

from alpha_edge.market.publish_production_bronze_v2 import publish_one, run


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


def test_promotes_immutable_raw_bronze_without_approving_gold():
    s3 = FakeS3()
    plan = "a" * 24
    source = f"dev/market/lake_v2/backfills/{plan}/end-2026-11-10/cohort-0001"
    batch = "execution/bronze_batches/s3bf-0001-0001"
    bars = b"immutable raw history"
    fx = b"immutable raw fx"
    acquisition = {
        "schema_version": "bronze-v2-1",
        "batch_id": "s3bf-0001-0001",
        "files": {
            "source_ohlcv.parquet": {"sha256": hashlib.sha256(bars).hexdigest()},
            "source_fx_quotes.parquet": {"sha256": hashlib.sha256(fx).hexdigest()},
        },
    }
    manifest = json.dumps(acquisition).encode()
    files = {
        f"{batch}/source_ohlcv.parquet": bars,
        f"{batch}/source_fx_quotes.parquet": fx,
        f"{batch}/manifest.json": manifest,
    }
    inventory = []
    for path, body in files.items():
        s3.objects[f"{source}/files/{path}"] = body
        inventory.append({"path": path, "sha256": hashlib.sha256(body).hexdigest()})
    commit = {
        "schema": "alpha_edge_s3_backfill_commit_v2_1",
        "plan_manifest_sha256": "a" * 64,
        "end_exclusive": "2026-11-10",
        "prefix": source, "files": inventory,
    }
    s3.objects[f"{source}/_committed.json"] = json.dumps(commit).encode()
    proposed = publish_one(s3, bucket="test", plan_prefix=plan,
                           end="2026-11-10", cohort=1, execute=False)
    assert proposed["state"] == "ready_for_production_bronze"
    applied = publish_one(s3, bucket="test", plan_prefix=plan,
                          end="2026-11-10", cohort=1, execute=True)
    assert applied["state"] == "production_bronze_committed"
    assert publish_one(s3, bucket="test", plan_prefix=plan,
                       end="2026-11-10", cohort=1, execute=True)["state"] == "production_bronze_verified_existing"
    receipt_key = applied["destination"] + "/_bronze_committed.json"
    receipt = json.loads(s3.objects[receipt_key])
    assert receipt["silver_gold_approved"] is False
    assert receipt["consumer_cutover"] is False
    assert not any("current.json" in key or "/v1/" in key for key in s3.objects)
    summary = run(s3, bucket="test", plan_prefix=plan, end="2026-11-10",
                  start_cohort=1, max_cohorts=2, execute=False)
    assert summary["missing_sources"] == 1
