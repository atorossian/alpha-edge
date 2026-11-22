"""Candidate consolidation must preserve coverage and reject ambiguity."""

import json

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market import consolidate_silver_sessions_v3 as c
from alpha_edge.market.lakehouse_v3.storage import parquet_bytes


def test_consolidation(monkeypatch):
    bucket = "alpha-edge-algo"
    assets = [{"asset_id": aid, "yahoo_ticker": aid} for aid in ("A", "B")]
    monkeypatch.setattr(c, "load_universe", lambda *args: (assets, {}))

    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        for run, ids, aid in (("full", ["A", "B"], "A"), ("repair", ["B"], "B")):
            root = f"{c.SILVER_ROOT}/runs/{run}"
            cfg = {"silver_run_id": run, "status": "candidate_only_not_certified", "source_fingerprint": "a" * 64,
                   "selected_asset_ids": ids}
            c._put(s3, bucket, f"{root}/run.json", c._json_bytes(cfg), "application/json")
            prefix = f"{root}/assets/{c._digest(aid.encode())}"
            blob = parquet_bytes(pd.DataFrame([{"asset_id": aid, "issue": "missing_endpoint_session",
                                               "session_date": "2026-11-17", "calendar_id": "NYSE"}]))
            key = f"{prefix}/objects/findings-{c._digest(blob)}.parquet"
            c._put(s3, bucket, key, blob, "application/vnd.apache.parquet")
            manifest = {"schema": c.SILVER_SCHEMA, "asset_id": aid, "silver_run_id": run,
                        "provider_symbol": aid, "production_eligible": False, "calendar_id": "NYSE",
                        "bronze_run_id": "bronze", "quality": {"rows": 10, "issue_rows": 1},
                        "outputs": {"candidate": {"key": f"{prefix}/objects/candidate-test.parquet"},
                                    "findings": {"key": key, "sha256": c._digest(blob), "bytes": len(blob)}}}
            c._put(s3, bucket, f"{prefix}/manifest.json", c._json_bytes(manifest), "application/json")
        with pytest.raises(c.SilverSessionError, match="Incomplete consolidated coverage"):
            c.consolidate(s3, bucket, ["full"])
        result = c.consolidate(s3, bucket, ["full", "repair"], execute=True)
        assert result["indexed"] == 2
        assert result["by_category"] == {"coverage_endpoint_review": 2}
        assert result["top_date_clusters"][0]["affected_assets"] == 2
        key = result["index_s3_uri"].split(f"s3://{bucket}/", 1)[1]
        index = json.loads(c._read(s3, bucket, key))
        assert set(index["assets"]) == {"A", "B"}
        assert index["production_eligible"] is False
        assert c.consolidate(s3, bucket, ["full", "repair"], execute=True) == result

        # A new registry can retire an asset and replace a changed candidate,
        # without allowing unreviewed mapping changes to reuse old artifacts.
        old = {"assets": [{"asset_id": aid, "calendar_id": "NYSE"} for aid in ("A", "B")]}
        new = {"assets": [{"asset_id": "B", "calendar_id": "NYSE", "mapping_source": "revised"}]}
        monkeypatch.setattr(c, "load_universe", lambda s, b, fp: (assets if fp == "a" * 64 else assets[1:], {}))
        monkeypatch.setattr(c, "load_registry", lambda s, b, v, fp: old if fp == "a" * 64 else new)
        # Synthetic registry versions are absent in the original fixture.
        # Install a separate old/new run pair with pinned registry identities.
        for run, fp, registry in (("old-run", "a" * 64, "old"), ("new-run", "b" * 64, "new")):
            root = f"{c.SILVER_ROOT}/runs/{run}"
            cfg = {"silver_run_id": run, "status": "candidate_only_not_certified", "source_fingerprint": fp,
                   "selected_asset_ids": ["A", "B"] if run == "old-run" else ["B"], "registry_version": registry}
            c._put(s3, bucket, f"{root}/run.json", c._json_bytes(cfg), "application/json")
            for aid in cfg["selected_asset_ids"]:
                prefix = f"{root}/assets/{c._digest(aid.encode())}"
                blob = parquet_bytes(pd.DataFrame([{"asset_id": aid, "issue": "missing_endpoint_session",
                                                   "session_date": "2026-11-17", "calendar_id": "NYSE"}]))
                key = f"{prefix}/objects/findings-{c._digest(blob)}.parquet"
                c._put(s3, bucket, key, blob, "application/vnd.apache.parquet")
                manifest = {"schema": c.SILVER_SCHEMA, "asset_id": aid, "silver_run_id": run,
                            "provider_symbol": aid, "production_eligible": False, "calendar_id": "NYSE",
                            "registry_version": registry, "bronze_run_id": "bronze", "quality": {"rows": 10, "issue_rows": 1},
                            "outputs": {"candidate": {"key": f"{prefix}/objects/candidate-test.parquet"},
                                        "findings": {"key": key, "sha256": c._digest(blob), "bytes": len(blob)}}}
                c._put(s3, bucket, f"{prefix}/manifest.json", c._json_bytes(manifest), "application/json")
        revised = c.consolidate(s3, bucket, ["old-run", "new-run"], revision_run="new-run", execute=True)
        assert revised["indexed"] == 1
        index = json.loads(c._read(s3, bucket, revised["index_s3_uri"].split(f"s3://{bucket}/", 1)[1]))
        assert index["assets"]["B"]["silver_run_id"] == "new-run"
        assert index["revision"]["retired_asset_ids"] == ["A"]
