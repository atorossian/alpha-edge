"""Active lake reader must reject missing pointers and verify exact S3 bytes."""

import io

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from alpha_edge.market.active_lake_v2 import ActiveLakeV2, ActiveMarketDataError
from alpha_edge.market.production_gold_release_v2 import CURRENT, encoded, sha
from alpha_edge.market.ingest_bronze_yahoo_v2 import _parquet_bytes


class S3:
    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}


def _source():
    s3 = S3()
    gold = pd.DataFrame(
        {
            "asset_id": ["A", "B"],
            "session_date": ["2026-10-05", "2026-10-05"],
            "close_raw_usd": [100.0, 200.0],
            "close_adjusted_usd": [99.0, 201.0],
            "quality_status": ["accepted", "accepted"],
            "ret_simple_close_adjusted_usd": [0.01, 0.02],
            "ret_log_close_adjusted_usd": [0.0099503, 0.0198026],
            "return_quality_status": ["accepted", "accepted"],
        }
    )
    key = "market/lake_v2/releases/data/r01/cohort-0001/gold.parquet"
    blob = _parquet_bytes(gold)
    s3.objects[key] = blob
    version_key = "market/lake_v2/releases/candidates/r01.json"
    version = {
        "schema": "alpha_edge_v2_release_candidate_1",
        "release_id": "r01",
        "as_of": "2026-10-05",
        "expected_cohorts": 1,
        "expected_assets": 2,
        "cohorts": [{"cohort": 1, "assets": ["A", "B"], "gold": {"key": key, "sha256": sha(blob)}}],
    }
    raw = encoded(version)
    s3.objects[version_key] = raw
    s3.objects[CURRENT] = encoded(
        {
            "schema": "alpha_edge_v2_release_pointer_1",
            "release_id": "r01",
            "release_key": version_key,
            "release_sha256": sha(raw),
        }
    )
    return s3, key


def test_active_reader_filters_asset_and_retains_quality_flags():
    s3, _ = _source()
    reader = ActiveLakeV2.open(s3, "test")
    px = reader.prices("A")
    assert px["close_adjusted_usd"].tolist() == [99.0]
    rr = reader.returns("B")
    assert rr["ret_simple_close_adjusted_usd"].tolist() == [0.02]
    assert rr["return_quality_status"].tolist() == ["accepted"]
    with pytest.raises(ActiveMarketDataError, match="missing or ambiguous"):
        reader.prices("C")


def test_active_reader_refuses_tampered_gold_and_missing_release():
    s3, gold_key = _source()
    reader = ActiveLakeV2.open(s3, "test")
    s3.objects[gold_key] = b"corrupted"
    with pytest.raises(ActiveMarketDataError, match="corrupt"):
        reader.prices("A")
    del s3.objects[CURRENT]
    with pytest.raises(ActiveMarketDataError, match="No active"):
        ActiveLakeV2.open(s3, "test")
