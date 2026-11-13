"""Offline contract and regression tests. No market data files are created."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import boto3
import numpy as np
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market.lakehouse_v3 import ingest, storage, transform, warehouse

BUCKET = "alpha-edge-algo"
RUN = "sample-20261113"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="eu-west-1")
        client.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
        yield client


def _registry():
    return {
        "schema": "alpha_edge_registry_v3_1",
        "assets": [
            {"asset_id": "CRYPTO:BTC-USD", "yahoo_ticker": "BTC-USD", "currency": "USD", "calendar_id": "24/7"}
        ],
        "fx_pairs": [],
    }


def _bronze_frame(n=35):
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "asset_id": ["CRYPTO:BTC-USD"] * n,
        "provider_symbol": ["BTC-USD"] * n,
        "currency_original": ["USD"] * n,
        "session_date": dates.strftime("%Y-%m-%d"),
        "source_timestamp": dates.strftime("%Y-%m-%dT00:00:00"),
        "open": np.arange(100, 100 + n).astype(float),
        "high": np.arange(102, 102 + n).astype(float),
        "low": np.arange(99, 99 + n).astype(float),
        "close": np.arange(101, 101 + n).astype(float),
        "adj_close": np.arange(101, 101 + n).astype(float),
        "volume": np.ones(n) * 100,
        "dividends": np.zeros(n),
        "stock_splits": np.zeros(n),
    })


def _empty_actions():
    return pd.DataFrame(
        columns=["asset_id", "provider_symbol", "effective_date", "value", "action_type", "currency"]
    )


def _empty_fx():
    return pd.DataFrame(columns=["provider_symbol", "session_date", "quote_close"])


def test_storage_immutable_idempotent_and_corruption_detection(s3):
    key = f"{storage.ROOT}/bronze/runs/{RUN}/test.json"
    reference = storage.put_json(s3, BUCKET, key, {"test": True})
    assert json.loads(storage.verified_bytes(s3, BUCKET, reference)) == {"test": True}
    assert storage.put_json(s3, BUCKET, key, {"test": True}) == reference
    with pytest.raises(storage.LakehouseError, match="collision"):
        storage.put_json(s3, BUCKET, key, {"test": False})
    s3.put_object(Bucket=BUCKET, Key=key, Body=b"corrupt")
    with pytest.raises(storage.LakehouseError, match="Corrupt"):
        storage.verified_bytes(s3, BUCKET, reference)
    with pytest.raises(storage.LakehouseError, match="unsafe"):
        storage.put_json(s3, BUCKET, "market/returns_usd/v1/unsafe.json", {})


def test_registry_from_s3_requires_identity_and_consistent_source(s3):
    key = f"{storage.ROOT}/config/test.json"
    storage.put_json(s3, BUCKET, key, _registry())
    assets, fx, checksum = ingest.registry_from_s3(s3, BUCKET, key)
    assert len(assets) == 1 and fx.empty and len(checksum) == 64
    with pytest.raises(storage.LakehouseError):
        ingest.registry_from_s3(s3, BUCKET, "data/universe/universe.csv")


def test_bronze_acquisition_never_creates_local_data(s3, monkeypatch):
    key = f"{storage.ROOT}/config/fixture.json"
    storage.put_json(s3, BUCKET, key, _registry())
    bars = _bronze_frame()
    meta = {"provider": "yahoo_finance", "start": "2026-01-01", "end_exclusive": "2026-02-05"}
    monkeypatch.setattr(ingest, "acquire_yahoo_bars", lambda *a, **kw: (bars.copy(), meta))
    dry = ingest.acquire(
        s3=s3, bucket=BUCKET, registry_key=key, start="2026-01-01",
        end="2026-02-05", run_id=RUN, execute=False,
    )
    assert dry["writes"] == 0
    done = ingest.acquire(
        s3=s3, bucket=BUCKET, registry_key=key, start="2026-01-01",
        end="2026-02-05", run_id=RUN, execute=True,
    )
    assert done["rows"] == {"bars": 35, "actions": 0, "fx": 0}
    manifest = storage.read_json(s3, BUCKET, f"{storage.ROOT}/bronze/runs/{RUN}/manifest.json")
    assert manifest["status"] == "bronze_committed"
    assert len(storage.verified_frame(s3, BUCKET, manifest["artifacts"]["bars"])) == 35


def test_silver_gold_reuse_original_calculators_and_return_contract():
    bars = _bronze_frame()
    assets = pd.DataFrame(_registry()["assets"])
    silver, gold, gaps, report = transform.build_tables(
        bars=bars, fx=_empty_fx(), actions=_empty_actions(),
        assets=assets, fx_registry=pd.DataFrame(columns=["currency", "provider_symbol", "direction"]),
    )
    assert len(silver) == len(gold) == 35
    assert gaps.empty
    assert report["price_quality"] == {"accepted": 35}
    assert pd.isna(gold.iloc[0]["ret_total_usd"])
    assert gold.iloc[1]["ret_total_usd"] == pytest.approx(102 / 101 - 1)
    assert gold.iloc[1]["ret_price_usd"] == pytest.approx(102 / 101 - 1)
    assert gold["corporate_action_independently_verified"].eq(False).all()
    assert gold["sma_20"].notna().any()
    assert not gold.duplicated(["asset_id", "session_date"]).any()


def test_source_actions_checked_against_exact_provider_values():
    bars = _bronze_frame(4)
    bars.loc[2, "dividends"] = 0.75
    actions = pd.DataFrame([{
        "asset_id": "CRYPTO:BTC-USD", "provider_symbol": "BTC-USD",
        "effective_date": "2026-01-03", "value": 0.75,
        "action_type": "DIVIDEND", "currency": "USD",
    }])
    assert transform.validate_events(bars, actions)["dividends"] == 1
    bad = actions.assign(value=0.9)
    with pytest.raises(storage.LakehouseError, match="differs"):
        transform.validate_events(bars, bad)


def test_end_to_end_s3_bronze_to_silver_gold_candidate(s3, monkeypatch):
    registry_key = f"{storage.ROOT}/config/fixture.json"
    storage.put_json(s3, BUCKET, registry_key, _registry())
    monkeypatch.setattr(
        ingest, "acquire_yahoo_bars",
        lambda *a, **kw: (_bronze_frame(), {"provider": "test_yahoo"}),
    )
    ingest.acquire(
        s3=s3, bucket=BUCKET, registry_key=registry_key,
        start="2026-01-01", end="2026-02-05", run_id=RUN, execute=True,
    )
    result = transform.process(s3, BUCKET, bronze_run_id=RUN, execute=True)
    assert result["state"] == "review_required"
    manifest = storage.read_json(s3, BUCKET, f"{storage.ROOT}/candidates/{RUN}/manifest.json")
    assert manifest["approved"] is False
    assert manifest["quality"]["gold_rows"] == 35
    assert storage.verified_frame(s3, BUCKET, manifest["artifacts"]["gold"]).shape[0] == 35
    with pytest.raises(storage.LakehouseError):
        warehouse.approved_candidate(
            s3, BUCKET, RUN, f"{storage.ROOT}/approvals/{RUN}/approval.json"
        )


def test_warehouse_ddl_reads_only_gold_data_and_publishes_one_view():
    sql = warehouse.warehouse_sql(BUCKET, RUN)
    assert f"candidates/{RUN}/gold/" in sql["stage"]
    assert "table_type = 'ICEBERG'" in sql["iceberg"]
    assert "CREATE OR REPLACE VIEW alpha_edge_market.market_asset_daily" in sql["view"]
    assert "ret_total_usd" in sql["view"]
    assert "market/returns_usd/v1" not in str(sql)
    with pytest.raises(storage.LakehouseError):
        warehouse.warehouse_sql(BUCKET, "bad';drop table")


def test_invalid_approval_does_not_start_athena_queries(s3):
    athena = MagicMock()
    with pytest.raises(storage.LakehouseError):
        warehouse.publish(
            s3, athena, bucket=BUCKET, run_id=RUN,
            approval_key=f"{storage.ROOT}/approvals/{RUN}/approval.json",
            workgroup="primary", execute=True,
        )
    athena.start_query_execution.assert_not_called()
