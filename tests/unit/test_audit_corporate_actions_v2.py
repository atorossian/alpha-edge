"""Corporate action v2 audit: verified reference evidence, no S3 writes."""
import io
import json

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from alpha_edge.market.audit_corporate_actions_v2 import (
    CorporateActionAuditError, inspect_adjustment_changes, load_existing_actions,
)


class ReadOnlyS3:
    def __init__(self, parts=None):
        self.parts = parts or {}
    def get_object(self, *, Bucket, Key):
        if Key not in self.parts:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.parts[Key])}
    def put_object(self, **kwargs):
        raise AssertionError("No S3 writes are allowed")


def as_parquet(df):
    bio = io.BytesIO()
    df.to_parquet(bio, index=False)
    return bio.getvalue()


def test_missing_partition_is_not_misreported_as_zero_actions():
    actions, sources = load_existing_actions(
        ReadOnlyS3(), bucket="test-bucket", asset_ids=["A"]
    )
    assert actions.empty
    assert sources.iloc[0]["source_status"] == "missing"


def test_known_split_is_read_without_writing():
    key = "market/reference/v1/corporate_actions/asset_id=A/part-00000.parquet"
    original = pd.DataFrame([{
        "asset_id": "A", "effective_date": "2026-10-29",
        "action_type": "SPLIT", "split_factor": 2.0, "source": "yfinance",
    }])
    actions, sources = load_existing_actions(
        ReadOnlyS3({key: as_parquet(original)}), bucket="test-bucket",
        asset_ids=["A", "B"]
    )
    assert len(actions) == 1
    assert sources.set_index("asset_id").loc["A", "source_status"] == "present_with_rows"
    assert sources.set_index("asset_id").loc["B", "source_status"] == "missing"


def test_corrupt_partition_is_not_silently_ignored():
    key = "market/reference/v1/corporate_actions/asset_id=A/part-00000.parquet"
    with pytest.raises(CorporateActionAuditError, match="Corrupt"):
        load_existing_actions(ReadOnlyS3({key: b"invalid"}),
                              bucket="test-bucket", asset_ids=["A"])


def test_ratio_change_and_nearby_action_evidence():
    gold = pd.DataFrame([
        {"asset_id": "A", "session_date": "2026-10-26", "close": 100, "adj_close": 90},
        {"asset_id": "A", "session_date": "2026-10-27", "close": 100, "adj_close": 90},
        {"asset_id": "A", "session_date": "2026-10-28", "close": 100, "adj_close": 99},
    ])
    actions = pd.DataFrame([{
        "asset_id": "A", "effective_date": pd.Timestamp("2026-10-29"),
        "action_type": "SPLIT", "split_factor": 2,
    }])
    flags = inspect_adjustment_changes(gold, actions)
    assert len(flags) == 1
    assert flags.iloc[0]["nearest_action_date"] == "2026-10-29"
    assert flags.iloc[0]["nearest_action_distance_days"] == 1


def test_constant_rescale_of_close_and_adjusted_close_is_not_claimed_detected():
    gold = pd.DataFrame([
        {"asset_id": "A", "session_date": "2026-10-27", "close": 25.18, "adj_close": 25.18},
        {"asset_id": "A", "session_date": "2026-10-28", "close": 24.308, "adj_close": 24.308},
    ])
    empty = pd.DataFrame(columns=["asset_id", "effective_date", "action_type", "split_factor"])
    assert inspect_adjustment_changes(gold, empty).empty
