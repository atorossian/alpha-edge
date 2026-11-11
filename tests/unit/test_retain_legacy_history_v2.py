"""Historical delisted assets are retained as raw, unapproved evidence."""
from __future__ import annotations

import io
import json

import pytest

from alpha_edge.market.retain_legacy_history_v2 import collect_one


class _Paginator:
    def __init__(self, files):
        self.files = files

    def paginate(self, Bucket, Prefix):
        yield {"Contents": [{"Key": key} for key in self.files if key.startswith(Prefix)]}


class _S3:
    def __init__(self, asset_id):
        self.files = {}
        for table in ("ohlcv_usd", "returns_usd"):
            prefix = f"market/{table}/v1/asset_id={asset_id}/year=2023"
            part = f"{prefix}/part-original.parquet"
            manifest_key = (
                f"market/manifests/v1/{table}/asset_id={asset_id}/year=2023/manifest.json"
            )
            self.files[manifest_key] = json.dumps({
                "asset_id": asset_id, "year": 2023, "parts": [part],
                "dates": ["2023-05-04"],
            }).encode()
            self.files[part] = b"original-legacy-parquet-bytes-" + table.encode()

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self.files)

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.files[Key])}


def test_retention_copies_original_bytes_without_approval(tmp_path):
    aid = "EQH9b41827cef0e0979"
    s3 = _S3(aid)
    contract = collect_one(s3, "bucket", aid, tmp_path)
    assert contract["production_eligible"] is False
    assert contract["historical_prices_approved"] is False
    assert contract["symbol"] == "KVUE"
    assert len(contract["files"]) == 4
    for entry in contract["files"]:
        assert (tmp_path / entry["copy_path"]).read_bytes() == s3.files[entry["source_key"]]


def test_retention_refuses_unreviewed_asset(tmp_path):
    with pytest.raises(ValueError, match="not approved"):
        collect_one(_S3("OTHER"), "bucket", "OTHER", tmp_path)


def test_retention_refuses_manifest_part_escape(tmp_path):
    aid = "EQH9b41827cef0e0979"
    s3 = _S3(aid)
    key = f"market/manifests/v1/ohlcv_usd/asset_id={aid}/year=2023/manifest.json"
    payload = json.loads(s3.files[key])
    payload["parts"] = ["market/returns_usd/v1/asset_id=OTHER/year=2023/part.parquet"]
    s3.files[key] = json.dumps(payload).encode()
    with pytest.raises(ValueError, match="unexpected legacy part"):
        collect_one(s3, "bucket", aid, tmp_path)
