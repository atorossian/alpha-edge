"""AEA-87 automatic resume: mock-S3 regression tests.

No production AWS calls are made.
"""

import hashlib
import io
import json
from types import SimpleNamespace

import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.market.aea87_rebuild import (
    SOURCE_ROOT,
    ROOT,
    find_reusable_source,
    find_reusable_plan,
)


class MemoryS3:
    def __init__(self, objects):
        self.objects = dict(objects)

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        store = self

        class Paginator:
            def paginate(self, Bucket, Prefix, Delimiter=None):
                prefixes = sorted(
                    set(
                        Prefix + key[len(Prefix) :].split(Delimiter)[0] + Delimiter
                        for key in store.objects
                        if key.startswith(Prefix) and Delimiter in key[len(Prefix) :]
                    )
                )
                return [{"CommonPrefixes": [{"Prefix": x} for x in prefixes]}]

        return Paginator()


def _setup(tmp_path):
    run = "aea87_rebuild_20261103T122403Z_5116397d_source_0001"
    base = f"{SOURCE_ROOT}/{run}/"
    aid = "CRYPTO:AAVE-USD"
    prefix = base + "assets/CRYPTO_AAVE_USD/"
    payloads = {
        prefix + "raw_yahoo.parquet": b"raw",
        prefix + "candidate_ohlcv_usd.parquet": b"price",
        prefix + "candidate_returns_usd.parquet": b"returns",
        prefix + "provider_metadata.json": b'{"currency":"USD"}',
    }
    summary = {
        "run_id": run,
        "status": "STAGING_ONLY",
        "schema": "aea87_full_history_rebuild_v2_local_session_dates",
        "active_histories_modified": False,
        "start": "2010-01-01",
        "end_exclusive": "2026-11-03",
        "assets": {aid: {"status": "STAGED_FOR_REVIEW", "yahoo_ticker": "AAVE-USD"}},
    }
    payloads[base + "summary.json"] = json.dumps(summary).encode()
    payloads[base + "manifest.json"] = json.dumps(
        {"run_id": run, "objects": [{"key": k, "sha256": hashlib.sha256(v).hexdigest()} for k, v in payloads.items()]}
    ).encode()
    universe = tmp_path / "universe.csv"
    pd.DataFrame({"asset_id": [aid], "yahoo_ticker": ["AAVE-USD"]}).to_csv(universe, index=False)
    args = SimpleNamespace(universe=str(universe), start="2010-01-01", end_exclusive="2026-11-03")
    return MemoryS3(payloads), run, aid, args


def test_reuse_valid_complete_source(tmp_path):
    s3, run, aid, args = _setup(tmp_path)
    result = find_reusable_source(s3, "bucket", [aid], args)
    assert result["run_id"] == run


def test_reject_candidate_with_bad_checksum(tmp_path):
    s3, run, aid, args = _setup(tmp_path)
    s3.objects[f"{SOURCE_ROOT}/{run}/assets/CRYPTO_AAVE_USD/candidate_ohlcv_usd.parquet"] = b"tampered"
    assert find_reusable_source(s3, "bucket", [aid], args) is None


def test_reject_pre_fix_utc_shifted_source_history(tmp_path):
    s3, run, aid, args = _setup(tmp_path)
    key = f"{SOURCE_ROOT}/{run}/summary.json"
    summary = json.loads(s3.objects[key])
    summary["schema"] = "aea87_full_history_rebuild_v1"
    s3.objects[key] = json.dumps(summary).encode()
    assert find_reusable_source(s3, "bucket", [aid], args) is None


def test_reject_outdated_source_window(tmp_path):
    s3, run, aid, args = _setup(tmp_path)
    args.end_exclusive = "2026-11-04"
    assert find_reusable_source(s3, "bucket", [aid], args) is None


def test_reject_universe_identity_change(tmp_path):
    s3, run, aid, args = _setup(tmp_path)
    pd.DataFrame({"asset_id": [aid], "yahoo_ticker": ["OTHER-USD"]}).to_csv(args.universe, index=False)
    assert find_reusable_source(s3, "bucket", [aid], args) is None


def test_reject_partial_batch_subset(tmp_path):
    s3, run, aid, args = _setup(tmp_path)
    assert find_reusable_source(s3, "bucket", [aid, "CRYPTO:BTC-USD"], args) is None


def test_prepared_plan_reuse_requires_current_active_manifest(tmp_path):
    s3, run, aid, args = _setup(tmp_path)
    release = run.split("_source_", 1)[0]
    plan_key = f"{ROOT}/{release}_CRYPTO_AAVE_USD/plan.json"
    prev = b'{"parts":["old"]}'
    new = '{"parts":["new"]}'
    manifest_key = "market/manifests/v1/ohlcv_usd/asset_id=CRYPTO:AAVE-USD/year=2026/manifest.json"
    ops = []
    for table in ("ohlcv_usd", "returns_usd"):
        key = manifest_key.replace("ohlcv_usd", table)
        part = f"prepared/{table}.parquet"
        backup = f"backups/{table}.json"
        s3.objects[key] = prev
        s3.objects[backup] = prev
        s3.objects[part] = b"prepared"
        ops.append(
            {
                "asset_id": aid,
                "table": table,
                "manifest_key": key,
                "before_missing": False,
                "before_sha": hashlib.sha256(prev).hexdigest(),
                "backup_key": backup,
                "part_key": part,
                "part_sha": hashlib.sha256(b"prepared").hexdigest(),
                "after": new,
                "after_sha": hashlib.sha256(new.encode()).hexdigest(),
            }
        )
    plan = {
        "asset_id": aid,
        "source_run": run,
        "status": "PREPARED_NOT_PUBLISHED",
        "active_histories_modified": False,
        "operations": ops,
    }
    s3.objects[plan_key] = json.dumps(plan).encode()
    assert find_reusable_plan(s3, "bucket", run, aid)["asset_id"] == aid
    s3.objects[manifest_key] = b"later-third-party-change"
    assert find_reusable_plan(s3, "bucket", run, aid) is None
