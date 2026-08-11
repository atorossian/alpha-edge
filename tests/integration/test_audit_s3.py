from __future__ import annotations

import json

from alpha_edge.core.audit import build_audit_event, write_audit_backup, write_audit_event


def test_writes_audit_event_and_backup_to_mocked_s3(runtime_cfg, s3_bucket) -> None:
    payload = {"trade_id": "trade-1", "value": 100.0}
    event = build_audit_event(
        cfg=runtime_cfg,
        event_type="delete",
        entity_type="trade",
        entity_id="trade-1",
        as_of="2026-06-14",
        source_script="record_trade.py",
        source_mode="delete",
        status="success",
        before_payload=payload,
    )

    backup_key = write_audit_backup(
        cfg=runtime_cfg,
        entity_type="trade",
        entity_id="trade-1",
        audit_id=event.audit_id,
        payload=payload,
        as_of="2026-06-14",
    )
    event_key = write_audit_event(cfg=runtime_cfg, event=event)

    backup = json.loads(s3_bucket.get_object(Bucket=runtime_cfg.bucket, Key=backup_key)["Body"].read().decode("utf-8"))
    stored_event = json.loads(
        s3_bucket.get_object(Bucket=runtime_cfg.bucket, Key=event_key)["Body"].read().decode("utf-8")
    )

    assert backup["payload"] == payload
    assert stored_event["audit_id"] == event.audit_id
    assert stored_event["event_type"] == "delete"
