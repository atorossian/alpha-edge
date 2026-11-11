import io
import json
import numpy as np
import pytest
import pandas as pd
from botocore.exceptions import ClientError
from alpha_edge.market.aea87_release_finalize import refresh_repaired_returns_cache


def payload(df):
    stream = io.BytesIO()
    df.to_parquet(stream)
    return stream.getvalue()


class FakeS3:
    def __init__(self, objects):
        self.objects = dict(objects)

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[Key] = Body


def test_repaired_asset_cache_replaces_full_series_without_disturbing_others():
    from hashlib import sha256

    dates = pd.date_range("2019-01-01", periods=1300, freq="2D")
    old_index = pd.DatetimeIndex(["2020-01-01", "2020-01-03"])
    old = pd.DataFrame({"A": [0.99, 0.99], "B": [0.02, 0.04]}, index=old_index)
    old.index.name = "date"
    source = pd.DataFrame(
        {
            "asset_id": ["A"] * len(dates),
            "date": dates,
            "ret_log_close_adjusted_usd": [np.nan] + [np.log1p(0.01)] * (len(dates) - 1),
            "ret_close_adjusted_usd": [np.nan] + [0.01] * (len(dates) - 1),
        }
    )
    part = payload(source)
    raw = {
        "market/cache/v1/returns_wide_min5y.parquet": payload(old),
        "market/cache/v1/returns_wide_min5y.meta.json": json.dumps({"n_assets": 2}).encode(),
        "new/part.parquet": part,
    }
    s3 = FakeS3(raw)
    ops = [
        {
            "asset_id": "A",
            "manifest_key": "market/manifests/v1/returns_usd/asset_id=A/year=2019/manifest.json",
            "part_key": "new/part.parquet",
            "staged_sha": sha256(part).hexdigest(),
        }
    ]
    result = refresh_repaired_returns_cache(s3, "test", "release", ops, ["A"])
    assert result["refreshed"] == 1
    actual = pd.read_parquet(io.BytesIO(s3.objects["market/cache/v1/returns_wide_min5y.parquet"]))
    assert actual.loc[pd.Timestamp("2020-01-01"), "B"] == 0.02
    assert actual.loc[pd.Timestamp("2020-01-03"), "B"] == 0.04
    assert actual.loc[pd.Timestamp("2020-01-01"), "A"] != 0.99
    assert actual.loc[dates[10], "A"] == pytest.approx(0.01)


def test_short_history_is_removed_instead_of_leaving_stale_cache_values():
    from hashlib import sha256

    old = pd.DataFrame({"A": [0.99], "B": [0.02]}, index=pd.DatetimeIndex(["2025-01-02"]))
    old.index.name = "date"
    source = pd.DataFrame(
        {
            "asset_id": ["A", "A"],
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "ret_log_close_adjusted_usd": [np.nan, np.log1p(0.03)],
        }
    )
    part = payload(source)
    s3 = FakeS3({"market/cache/v1/returns_wide_min5y.parquet": payload(old), "new/part.parquet": part})
    ops = [
        {
            "asset_id": "A",
            "manifest_key": "market/manifests/v1/returns_usd/asset_id=A/year=2025/manifest.json",
            "part_key": "new/part.parquet",
            "staged_sha": sha256(part).hexdigest(),
        }
    ]
    result = refresh_repaired_returns_cache(s3, "test", "release", ops, ["A"])
    actual = pd.read_parquet(io.BytesIO(s3.objects["market/cache/v1/returns_wide_min5y.parquet"]))
    assert "A" not in actual
    assert actual.loc[pd.Timestamp("2025-01-02"), "B"] == 0.02
    assert result["excluded_below_5y"] == ["A"]
