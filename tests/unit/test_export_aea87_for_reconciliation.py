"""Strict active-manifest AEA-87 exporter tests; no AWS connection."""
import io
import json

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from alpha_edge.market.export_aea87_for_reconciliation import export, fetch_table
from alpha_edge.market.reconcile_aea87_lake_v2 import ReconciliationError


class S3Fake:
    def __init__(self):
        self.data = {}
        self.requests = []
    def get_object(self, *, Bucket, Key):
        self.requests.append(("get", Key))
        if Key not in self.data:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.data[Key])}
    def put_object(self, **kwargs):
        raise AssertionError("Exporter must never call put_object")


def populate(s3, aid="A_US", year=2025, table="ohlcv_usd", dates=None):
    dates = dates or ["2025-01-02", "2025-01-03"]
    field = "close_adjusted_usd" if table == "ohlcv_usd" else "ret_close_adjusted_usd"
    rows = pd.DataFrame([{"asset_id": aid, "date": day, field: i / 100 + 100.0}
                         for i, day in enumerate(dates)])
    buf = io.BytesIO()
    rows.to_parquet(buf, index=False)
    part = f"market/{table}/v1/asset_id={aid}/year={year}/part-test.parquet"
    man = f"market/manifests/v1/{table}/asset_id={aid}/year={year}/manifest.json"
    s3.data[part] = buf.getvalue()
    s3.data[man] = json.dumps({"asset_id": aid, "year": year, "parts": [part]}).encode()
    return part, man


def test_manifest_selected_partitions_are_read_only():
    s3 = S3Fake()
    populate(s3)
    result, journal = fetch_table(
        s3, bucket="example", root="market", version="v1",
        table="ohlcv_usd", asset_ids=["A_US"], years=[2025],
        start="2025-01-01", end_exclusive="2025-01-04",
    )
    assert result["close_adjusted_usd"].tolist() == [100.0, 100.01]
    assert journal[0]["parts"][0]["sha256"]
    assert {verb for verb, _ in s3.requests} == {"get"}


def test_missing_manifest_fails_closed():
    with pytest.raises(ReconciliationError, match="Missing/unreadable"):
        fetch_table(
            S3Fake(), bucket="example", root="market", version="v1",
            table="returns_usd", asset_ids=["A_US"], years=[2025],
            start="2025-01-01", end_exclusive="2025-01-04",
        )


def test_duplicate_active_sessions_fail_closed():
    s3 = S3Fake()
    part, manifest = populate(s3)
    payload = json.loads(s3.data[manifest])
    other = part.replace("part-test", "part-duplicate")
    s3.data[other] = s3.data[part]
    payload["parts"].append(other)
    s3.data[manifest] = json.dumps(payload).encode()
    with pytest.raises(ReconciliationError, match="Duplicate active"):
        fetch_table(
            s3, bucket="example", root="market", version="v1",
            table="ohlcv_usd", asset_ids=["A_US"], years=[2025],
            start="2025-01-01", end_exclusive="2025-01-04",
        )


def test_export_six_assets_requires_readonly_and_writes_only_local(tmp_path):
    s3 = S3Fake()
    universe = tmp_path / "assets.csv"
    ids = [f"A{i}" for i in range(6)]
    pd.DataFrame({"asset_id": ids, "include": [1] * 6}).to_csv(universe, index=False)
    for aid in ids:
        populate(s3, aid=aid)
        populate(s3, aid=aid, table="returns_usd")
    out = tmp_path / "export"
    report = export(
        s3=s3, bucket="example", universe_csv=universe, start="2025-01-01",
        end_exclusive="2025-01-04", output=out,
    )
    assert report["s3_writes"] == 0
    assert report["return_semantics"].startswith("unknown")
    assert (out / "aea87_prices.parquet").exists()
    assert (out / "aea87_returns.parquet").exists()
    with pytest.raises(FileExistsError):
        export(
            s3=s3, bucket="example", universe_csv=universe, start="2025-01-01",
            end_exclusive="2025-01-04", output=out,
        )
