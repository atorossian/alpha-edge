from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import boto3
import pandas as pd

from alpha_edge.core.audit import build_audit_event, make_audit_id, write_audit_backup, write_audit_event
from alpha_edge.core.runtime import runtime_dt_key, runtime_engine_key
from alpha_edge.core.schemas import RuntimeConfig


def s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


def s3_get_bytes(s3, *, bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )


def s3_delete(s3, *, bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)


def s3_exists(s3, *, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def s3_get_json_optional(s3, *, bucket: str, key: str) -> Optional[dict]:
    try:
        raw = s3_get_bytes(s3, bucket=bucket, key=key)
        obj = json.loads(raw.decode("utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _parse_date(s: str) -> str:
    return pd.Timestamp(s).date().strftime("%Y-%m-%d")


def _iso_utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _extract_dt_from_key(key: str) -> Optional[str]:
    for part in key.split("/"):
        if part.startswith("dt="):
            return part[len("dt=") :]
    return None


def load_index(s3, *, cfg: RuntimeConfig, index_key: str) -> dict:
    idx = s3_get_json_optional(s3, bucket=cfg.bucket, key=index_key)
    return idx if isinstance(idx, dict) else {}


def save_index(s3, *, cfg: RuntimeConfig, index_key: str, idx: dict) -> None:
    s3_put_json(s3, bucket=cfg.bucket, key=index_key, payload=idx)


def set_index_record(
    s3,
    *,
    cfg: RuntimeConfig,
    index_key: str,
    entity_id: str,
    key: str,
    as_of: str,
) -> None:
    idx = load_index(s3, cfg=cfg, index_key=index_key)
    idx[str(entity_id)] = {"key": str(key), "as_of": str(as_of)}
    save_index(s3, cfg=cfg, index_key=index_key, idx=idx)


def remove_index_record(
    s3,
    *,
    cfg: RuntimeConfig,
    index_key: Optional[str],
    entity_id: str,
) -> bool:
    if not index_key:
        return False

    idx = load_index(s3, cfg=cfg, index_key=index_key)
    if str(entity_id) not in idx:
        return False

    del idx[str(entity_id)]
    save_index(s3, cfg=cfg, index_key=index_key, idx=idx)
    return True


def resolve_record_key(
    s3,
    *,
    cfg: RuntimeConfig,
    table: str,
    file_prefix: str,
    entity_id: str,
    as_of: Optional[str] = None,
    index_key: Optional[str] = None,
) -> Tuple[str, Optional[str], str]:
    """
    Resolve the active datalake object key.

    Resolution order:
      1. explicit as_of partition
      2. index.json when provided
      3. scan under table prefix, matching <file_prefix>_<entity_id>.json
    """
    if as_of:
        dt = _parse_date(as_of)
        key = runtime_dt_key(cfg, table, dt, f"{file_prefix}_{entity_id}.json")
        return key, dt, "as_of"

    if index_key:
        idx = load_index(s3, cfg=cfg, index_key=index_key)
        meta = idx.get(str(entity_id))
        if isinstance(meta, dict) and meta.get("key"):
            key = str(meta["key"])
            dt = str(meta.get("as_of") or "") or _extract_dt_from_key(key)
            return key, dt, "index"

    filename = f"{file_prefix}_{entity_id}.json"
    prefix = runtime_engine_key(cfg, table) + "/"
    matches: list[str] = []
    token = None

    while True:
        kwargs: Dict[str, Any] = {"Bucket": cfg.bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)

        for it in resp.get("Contents") or []:
            key = str(it.get("Key") or "")
            if key.endswith("/" + filename):
                matches.append(key)

        if not resp.get("IsTruncated"):
            break

        token = resp.get("NextContinuationToken")

    if not matches:
        raise RuntimeError(
            f"No active {table} record found for id={entity_id!r} under "
            f"s3://{cfg.bucket}/{prefix}. Provide --old-as-of if needed."
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous {table} record id={entity_id!r}; found multiple matches={matches}. "
            "Provide --old-as-of to disambiguate."
        )

    key = matches[0]
    return key, _extract_dt_from_key(key), "scan"


def mark_latest_stale_if_points_to_deleted(
    s3,
    *,
    cfg: RuntimeConfig,
    table: str,
    id_field: str,
    entity_id: str,
    dry_run: bool,
) -> tuple[bool, Optional[str]]:
    latest_key = runtime_engine_key(cfg, table, "latest.json")
    latest = s3_get_json_optional(s3, bucket=cfg.bucket, key=latest_key)

    if not isinstance(latest, dict):
        return False, latest_key

    if str(latest.get(id_field)) != str(entity_id):
        return False, latest_key

    stale_payload = dict(latest)
    stale_payload["_stale_latest_marker"] = True
    stale_payload["_stale_reason"] = f"latest.json pointed to deleted {table} id={entity_id}"
    stale_payload["_stale_at_utc"] = _iso_utc_now()

    if dry_run:
        print(f"[DRY RUN] Would mark latest stale: s3://{cfg.bucket}/{latest_key}")
    else:
        s3_put_json(s3, bucket=cfg.bucket, key=latest_key, payload=stale_payload)

    return True, latest_key


def delete_record_with_audit(
    *,
    cfg: RuntimeConfig,
    table: str,
    entity_type: str,
    entity_id: str,
    id_field: str,
    file_prefix: str,
    as_of: Optional[str] = None,
    index_key: Optional[str] = None,
    source_script: str,
    source_mode: str = "delete",
    reason: Optional[str] = None,
    dry_run: bool = False,
    run_id: Optional[str] = None,
    input_args: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Hard-delete an active operation record while preserving traceability.

    This intentionally removes the object from the active table so ledger rebuilds do not
    accidentally consume soft-deleted operations. The full before payload is preserved in
    audit_backups and referenced by the audit_trail event.
    """
    if not dry_run and not reason:
        raise ValueError("--reason is required for real delete operations")

    s3 = s3_client(cfg)
    key, resolved_as_of, resolved_by = resolve_record_key(
        s3,
        cfg=cfg,
        table=table,
        file_prefix=file_prefix,
        entity_id=entity_id,
        as_of=as_of,
        index_key=index_key,
    )

    if not s3_exists(s3, bucket=cfg.bucket, key=key):
        raise RuntimeError(f"Record not found: s3://{cfg.bucket}/{key}")

    obj = s3_get_json_optional(s3, bucket=cfg.bucket, key=key)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Record is not a JSON object: s3://{cfg.bucket}/{key}")

    audit_id = make_audit_id()
    backup_key = write_audit_backup(
        cfg=cfg,
        entity_type=entity_type,
        entity_id=entity_id,
        audit_id=audit_id,
        payload=obj,
        as_of=resolved_as_of,
        dry_run=dry_run,
    )

    latest_key = runtime_engine_key(cfg, table, "latest.json")
    output_keys: list[str] = []
    if index_key:
        output_keys.append(index_key)

    print(f"\n=== DELETE {entity_type.upper()} ===")
    print(f"env:          {cfg.env}")
    print(f"bucket:       {cfg.bucket}")
    print(f"root:         {cfg.engine_root}")
    print(f"id:           {entity_id}")
    print(f"as_of:        {resolved_as_of}")
    print(f"resolved_by:  {resolved_by}")
    print(f"source:       s3://{cfg.bucket}/{key}")
    print(f"backup:       s3://{cfg.bucket}/{backup_key}")
    print(f"reason:       {reason}")
    print("")

    if dry_run:
        print("[DRY RUN] Would delete active record.")
        if index_key:
            print(f"[DRY RUN] Would remove id from index: s3://{cfg.bucket}/{index_key}")
        mark_latest_stale_if_points_to_deleted(
            s3,
            cfg=cfg,
            table=table,
            id_field=id_field,
            entity_id=entity_id,
            dry_run=True,
        )
        audit = build_audit_event(
            cfg=cfg,
            run_id=run_id,
            audit_id=audit_id,
            event_type="delete",
            entity_type=entity_type,
            entity_id=str(entity_id),
            as_of=resolved_as_of,
            source_script=source_script,
            source_mode=source_mode,
            status="dry_run",
            reason=reason,
            input_args=input_args,
            output_keys=output_keys,
            backup_keys=[backup_key],
            deleted_keys=[key],
            before_payload=obj,
            after_payload=None,
            metadata={"source_key": key, "resolved_by": resolved_by, "latest_key": latest_key},
        )
        write_audit_event(cfg=cfg, event=audit, dry_run=True)
        return

    s3_delete(s3, bucket=cfg.bucket, key=key)
    index_removed = remove_index_record(s3, cfg=cfg, index_key=index_key, entity_id=entity_id)
    latest_marked, latest_key = mark_latest_stale_if_points_to_deleted(
        s3,
        cfg=cfg,
        table=table,
        id_field=id_field,
        entity_id=entity_id,
        dry_run=False,
    )

    if latest_marked and latest_key:
        output_keys.append(latest_key)

    audit = build_audit_event(
        cfg=cfg,
        run_id=run_id,
        audit_id=audit_id,
        event_type="delete",
        entity_type=entity_type,
        entity_id=str(entity_id),
        as_of=resolved_as_of,
        source_script=source_script,
        source_mode=source_mode,
        status="success",
        reason=reason,
        input_args=input_args,
        output_keys=output_keys,
        backup_keys=[backup_key],
        deleted_keys=[key],
        before_payload=obj,
        after_payload=None,
        metadata={
            "source_key": key,
            "resolved_by": resolved_by,
            "index_removed": index_removed,
            "latest_marked_stale": latest_marked,
            "latest_key": latest_key,
        },
    )
    write_audit_event(cfg=cfg, event=audit, dry_run=False)

    print("[OK] Deleted active record:")
    print(f"  s3://{cfg.bucket}/{key}")
    print("[OK] Preserved audit backup:")
    print(f"  s3://{cfg.bucket}/{backup_key}")
    if index_removed:
        print("[OK] Removed from index:")
        print(f"  s3://{cfg.bucket}/{index_key}")
    if latest_marked:
        print("[WARN] latest.json pointed to deleted record and was marked stale:")
        print(f"  s3://{cfg.bucket}/{latest_key}")
    print("")
