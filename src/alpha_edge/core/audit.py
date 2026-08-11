from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional

import boto3
import pandas as pd

from alpha_edge.core.runtime import runtime_engine_key
from alpha_edge.core.schemas import AuditEvent, RuntimeConfig


def audit_s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


def utc_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def utc_today() -> str:
    return pd.Timestamp.utcnow().date().strftime("%Y-%m-%d")


def make_run_id(prefix: str = "run") -> str:
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def make_audit_id(prefix: str = "audit") -> str:
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:10]}"


def audit_event_key(cfg: RuntimeConfig, *, audit_id: str, dt: Optional[str] = None) -> str:
    dt_norm = dt or utc_today()
    return runtime_engine_key(cfg, "audit_trail", f"dt={dt_norm}", f"{audit_id}.json")


def audit_backup_key(
    cfg: RuntimeConfig,
    *,
    entity_type: str,
    entity_id: str,
    audit_id: str,
    dt: Optional[str] = None,
) -> str:
    dt_norm = dt or utc_today()
    safe_entity_id = str(entity_id).replace("/", "_").replace(" ", "_")
    return runtime_engine_key(
        cfg,
        "audit_backups",
        entity_type,
        f"dt={dt_norm}",
        f"{safe_entity_id}__{audit_id}.json",
    )


def json_dumps_stable(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def payload_sha256(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if payload is None:
        return None
    return hashlib.sha256(json_dumps_stable(payload).encode("utf-8")).hexdigest()


def _put_json(s3, *, bucket: str, key: str, payload: Dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )


def build_audit_event(
    *,
    cfg: RuntimeConfig,
    event_type: str,
    entity_type: str,
    entity_id: Optional[str],
    as_of: Optional[str],
    source_script: str,
    source_mode: Optional[str],
    status: str,
    run_id: Optional[str] = None,
    audit_id: Optional[str] = None,
    reason: Optional[str] = None,
    input_args: Optional[Dict[str, Any]] = None,
    output_keys: Optional[Iterable[str]] = None,
    backup_keys: Optional[Iterable[str]] = None,
    deleted_keys: Optional[Iterable[str]] = None,
    before_payload: Optional[Dict[str, Any]] = None,
    after_payload: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> AuditEvent:
    meta = dict(metadata or {})
    meta.setdefault("before_payload_sha256", payload_sha256(before_payload))
    meta.setdefault("after_payload_sha256", payload_sha256(after_payload))

    return AuditEvent(
        audit_id=audit_id or make_audit_id(),
        run_id=run_id or make_run_id(),
        event_type=str(event_type),
        entity_type=str(entity_type),
        entity_id=(str(entity_id) if entity_id is not None else None),
        as_of=(str(as_of) if as_of is not None else None),
        env=str(cfg.env),
        bucket=str(cfg.bucket),
        root=str(cfg.engine_root),
        source_script=str(source_script),
        source_mode=(str(source_mode) if source_mode is not None else None),
        created_at_utc=utc_now_iso(),
        status=str(status),
        reason=(str(reason) if reason else None),
        input_args=dict(input_args or {}),
        output_keys=[str(k) for k in (output_keys or [])],
        backup_keys=[str(k) for k in (backup_keys or [])],
        deleted_keys=[str(k) for k in (deleted_keys or [])],
        before_payload=before_payload,
        after_payload=after_payload,
        metadata=meta,
        error=(str(error) if error else None),
    )


def write_audit_event(
    *,
    cfg: RuntimeConfig,
    event: AuditEvent,
    dry_run: bool = False,
) -> str:
    key = audit_event_key(cfg, audit_id=event.audit_id)

    if dry_run:
        print(f"[DRY RUN] Would write audit event: s3://{cfg.bucket}/{key}")
        return key

    s3 = audit_s3_client(cfg)
    _put_json(s3, bucket=cfg.bucket, key=key, payload=asdict(event))
    print(f"[OK] Wrote audit event: s3://{cfg.bucket}/{key}")
    return key


def write_audit_backup(
    *,
    cfg: RuntimeConfig,
    entity_type: str,
    entity_id: str,
    audit_id: str,
    payload: Dict[str, Any],
    as_of: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    key = audit_backup_key(
        cfg,
        entity_type=entity_type,
        entity_id=entity_id,
        audit_id=audit_id,
        dt=as_of,
    )

    backup_payload = {
        "audit_id": audit_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "as_of": as_of,
        "created_at_utc": utc_now_iso(),
        "payload": payload,
    }

    if dry_run:
        print(f"[DRY RUN] Would write audit backup: s3://{cfg.bucket}/{key}")
        return key

    s3 = audit_s3_client(cfg)
    _put_json(s3, bucket=cfg.bucket, key=key, payload=backup_payload)
    print(f"[OK] Wrote audit backup: s3://{cfg.bucket}/{key}")
    return key
