"""Initialization only publishes candidate when all historical cohorts exist."""

import io

from botocore.exceptions import ClientError

from alpha_edge.market.prepare_initial_gold_v2 import build
from alpha_edge.market.production_gold_release_v2 import encoded, sha


class S3:
    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[Key] = Body


def test_incomplete_historical_cohorts_have_no_production_side_effects():
    s3 = S3()
    plan_root = "dev/market/lake_v2/archives/old-outputs-20261110/"
    fx = b"currency,provider_symbol,direction\n"
    cohorts = {
        "cohort_0001.csv": b"asset_id,yahoo_ticker,currency,calendar_id\nA,GLD,USD,NYSE\n",
        "fx_config.csv": fx,
    }
    plan = {
        "schema": "alpha_edge_lake_v2_universe_preflight_1",
        "cohorts": 1,
        "assets_included": 1,
        "files_sha256": {name: sha(data) for name, data in cohorts.items()},
    }
    cohorts["manifest.json"] = encoded(plan)
    items = []
    for name, payload in cohorts.items():
        path = f"full_universe_preflight_v2/first_review/{name}"
        s3.objects[plan_root + "files/" + path] = payload
        items.append({"path": path, "sha256": sha(payload)})
    s3.objects[plan_root + "_verified_inventory.json"] = encoded(
        {
            "schema": "alpha_edge_local_output_archive_v2_1",
            "files": items,
        }
    )
    before = set(s3.objects)
    result = build(
        s3=s3,
        bucket="test",
        archive_id="old-outputs-20261110",
        plan_hash_prefix=sha(cohorts["manifest.json"])[:24],
        end="2026-11-10",
        run_id="initial_01",
        execute=True,
    )
    assert result["status"] == "blocked_missing_cohorts"
    assert result["missing_cohorts"] == [1]
    assert set(s3.objects) == before
