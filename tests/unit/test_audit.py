from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, payload_sha256


def test_payload_hash_is_stable_across_key_order(runtime_cfg) -> None:
    assert payload_sha256({"a": 1, "b": 2}) == payload_sha256({"b": 2, "a": 1})


def test_build_audit_event_captures_before_after_hashes(runtime_cfg) -> None:
    before = {"trade_id": "t1", "value": 100.0}
    after = {"trade_id": "t1", "value": 120.0}

    event = build_audit_event(
        cfg=runtime_cfg,
        event_type="modify",
        entity_type="trade",
        entity_id="t1",
        as_of="2026-06-14",
        source_script="record_trade.py",
        source_mode="edit",
        status="success",
        before_payload=before,
        after_payload=after,
    )

    assert event.env == "dev"
    assert event.entity_id == "t1"
    assert event.metadata["before_payload_sha256"] == payload_sha256(before)
    assert event.metadata["after_payload_sha256"] == payload_sha256(after)
