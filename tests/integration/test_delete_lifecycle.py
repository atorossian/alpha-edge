from __future__ import annotations

import json

import pytest
from alpha_edge.core.runtime import runtime_dt_key, runtime_engine_key
from alpha_edge.operations.operation_lifecycle import delete_record_with_audit


def test_delete_removes_active_trade_and_preserves_audit(runtime_cfg, s3_bucket) -> None:
    trade_id = "trade-bad-pnl"
    as_of = "2026-04-02"
    trade_key = runtime_dt_key(runtime_cfg, "trades", as_of, f"trade_{trade_id}.json")
    index_key = runtime_engine_key(runtime_cfg, "trades", "index.json")
    latest_key = runtime_engine_key(runtime_cfg, "trades", "latest.json")
    trade = {
        "trade_id": trade_id,
        "as_of": as_of,
        "ticker": "SOL-USD",
        "reported_pnl": -1477.2,
    }

    for key, payload in (
        (trade_key, trade),
        (latest_key, trade),
        (index_key, {trade_id: {"key": trade_key, "as_of": as_of}}),
    ):
        s3_bucket.put_object(Bucket=runtime_cfg.bucket, Key=key, Body=json.dumps(payload).encode("utf-8"))

    delete_record_with_audit(
        cfg=runtime_cfg,
        table="trades",
        entity_type="trade",
        entity_id=trade_id,
        id_field="trade_id",
        file_prefix="trade",
        index_key=index_key,
        source_script="record_trade.py",
        reason="Incorrect short-close PnL sign",
    )

    with pytest.raises(s3_bucket.exceptions.NoSuchKey):
        s3_bucket.get_object(Bucket=runtime_cfg.bucket, Key=trade_key)

    index = json.loads(s3_bucket.get_object(Bucket=runtime_cfg.bucket, Key=index_key)["Body"].read())
    latest = json.loads(s3_bucket.get_object(Bucket=runtime_cfg.bucket, Key=latest_key)["Body"].read())

    assert trade_id not in index
    assert latest["_stale_latest_marker"] is True

    audit_objects = s3_bucket.list_objects_v2(
        Bucket=runtime_cfg.bucket,
        Prefix=runtime_engine_key(runtime_cfg, "audit_trail") + "/",
    ).get("Contents", [])
    backup_objects = s3_bucket.list_objects_v2(
        Bucket=runtime_cfg.bucket,
        Prefix=runtime_engine_key(runtime_cfg, "audit_backups", "trade") + "/",
    ).get("Contents", [])

    assert len(audit_objects) == 1
    assert len(backup_objects) == 1


def test_real_delete_requires_reason(runtime_cfg, s3_bucket) -> None:
    trade_id = "trade-1"
    as_of = "2026-04-02"
    trade_key = runtime_dt_key(runtime_cfg, "trades", as_of, f"trade_{trade_id}.json")
    s3_bucket.put_object(Bucket=runtime_cfg.bucket, Key=trade_key, Body=b"{}")

    with pytest.raises(ValueError, match="reason is required"):
        delete_record_with_audit(
            cfg=runtime_cfg,
            table="trades",
            entity_type="trade",
            entity_id=trade_id,
            id_field="trade_id",
            file_prefix="trade",
            as_of=as_of,
            source_script="record_trade.py",
            reason=None,
        )
