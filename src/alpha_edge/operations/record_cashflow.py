from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from typing import Any, Dict, Literal, Optional, Tuple

import boto3
import pandas as pd

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import (
    load_runtime_config,
    require_prod_confirmation,
    runtime_dt_key,
    runtime_engine_key,
)
from alpha_edge.core.schemas import Cashflow, RuntimeConfig
from alpha_edge.operations.operation_lifecycle import (
    delete_record_with_audit,
    load_index,
    resolve_record_key,
    set_index_record,
    s3_get_json_optional,
    save_index,
)


CASHFLOWS_TABLE = "cashflows"


def s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


def cashflows_index_key(cfg: RuntimeConfig) -> str:
    return runtime_engine_key(cfg, CASHFLOWS_TABLE, "index.json")


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )


def s3_get_bytes(s3, *, bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def s3_exists(s3, *, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def s3_copy(s3, *, bucket: str, src_key: str, dst_key: str) -> None:
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        ContentType="application/json",
        MetadataDirective="COPY",
    )


def s3_delete(s3, *, bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)


def _parse_date(s: str) -> str:
    return pd.Timestamp(s).date().strftime("%Y-%m-%d")


def _iso_utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _validate_positive(name: str, x: float) -> float:
    x = float(x)
    if not (x > 0.0):
        raise ValueError(f"{name} must be > 0")
    return x


def _normalize_type(x: str) -> Literal["DEPOSIT", "WITHDRAWAL"]:
    s = str(x).upper().strip()
    if s not in {"DEPOSIT", "WITHDRAWAL"}:
        raise ValueError("type must be DEPOSIT or WITHDRAWAL")
    return s  # type: ignore[return-value]


def _extract_dt_from_key(key: str) -> Optional[str]:
    for part in key.split("/"):
        if part.startswith("dt="):
            return part[len("dt=") :]
    return None


def _list_cashflow_keys(s3, *, cfg: RuntimeConfig) -> list[str]:
    prefix = runtime_engine_key(cfg, CASHFLOWS_TABLE) + "/"
    keys: list[str] = []
    token = None

    while True:
        kwargs: Dict[str, Any] = {"Bucket": cfg.bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents") or []:
            k = str(it.get("Key") or "")
            if k.endswith(".json") and "/cashflow_" in k:
                keys.append(k)

        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

    return keys


def rebuild_cashflows_index(s3, *, cfg: RuntimeConfig) -> Tuple[int, int, dict]:
    keys = _list_cashflow_keys(s3, cfg=cfg)
    idx: Dict[str, Dict[str, str]] = {}
    scanned = 0
    indexed = 0

    for k in keys:
        scanned += 1
        name = k.split("/")[-1]
        if not (name.startswith("cashflow_") and name.endswith(".json")):
            continue

        cashflow_id = name[len("cashflow_") : -len(".json")]
        as_of = _extract_dt_from_key(k) or ""
        idx[str(cashflow_id)] = {"key": str(k), "as_of": as_of}
        indexed += 1

    save_index(s3, cfg=cfg, index_key=cashflows_index_key(cfg), idx=idx)
    return scanned, indexed, idx


def migrate_cashflows_index(
    *,
    cfg: RuntimeConfig,
    dry_run: bool = False,
    run_id: Optional[str] = None,
    input_args: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
) -> None:
    s3 = s3_client(cfg)
    keys = _list_cashflow_keys(s3, cfg=cfg)

    idx: Dict[str, Dict[str, str]] = {}
    scanned = 0
    indexed = 0

    for k in keys:
        scanned += 1
        name = k.split("/")[-1]
        if not (name.startswith("cashflow_") and name.endswith(".json")):
            continue
        cashflow_id = name[len("cashflow_") : -len(".json")]
        idx[str(cashflow_id)] = {"key": str(k), "as_of": _extract_dt_from_key(k) or ""}
        indexed += 1

    print("\n=== MIGRATE CASHFLOWS INDEX ===")
    print(f"env:      {cfg.env}")
    print(f"bucket:   {cfg.bucket}")
    print(f"root:     {cfg.engine_root}")
    print(f"scanned:  {scanned}")
    print(f"indexed:  {indexed}")
    print(f"index:    s3://{cfg.bucket}/{cashflows_index_key(cfg)}")

    audit = build_audit_event(
        cfg=cfg,
        run_id=run_id,
        event_type="migrate",
        entity_type="cashflows_index",
        entity_id="index.json",
        as_of=None,
        source_script="record_cashflow.py",
        source_mode="migrate",
        status=("dry_run" if dry_run else "success"),
        reason=reason,
        input_args=input_args,
        output_keys=[cashflows_index_key(cfg)],
        metadata={"objects_scanned": scanned, "objects_indexed": indexed},
    )

    if dry_run:
        print("[DRY RUN] no write performed.")
        write_audit_event(cfg=cfg, event=audit, dry_run=True)
        print("")
        return

    save_index(s3, cfg=cfg, index_key=cashflows_index_key(cfg), idx=idx)
    write_audit_event(cfg=cfg, event=audit, dry_run=False)
    print("[OK] index.json written/overwritten.")
    print("")


def record_cashflow(
    *,
    cfg: RuntimeConfig,
    as_of: str,
    type: str,
    amount: float,
    currency: str = "USD",
    account_id: str = "main",
    ts_utc: Optional[str] = None,
    cashflow_id: Optional[str] = None,
    note: Optional[str] = None,
    dry_run: bool = False,
    run_id: Optional[str] = None,
    input_args: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
) -> None:
    s3 = s3_client(cfg)

    as_of_norm = _parse_date(as_of)
    type_norm = _normalize_type(type)
    amt = _validate_positive("amount", amount)
    ccy = str(currency).upper().strip() or "USD"
    acct = str(account_id).strip() or "main"

    if ts_utc is None:
        ts_utc = _iso_utc_now()

    if cashflow_id is None:
        cashflow_id = f"{as_of_norm.replace('-', '')}-{uuid.uuid4().hex[:10]}"

    cf = Cashflow(
        cashflow_id=str(cashflow_id),
        as_of=as_of_norm,
        ts_utc=str(ts_utc),
        account_id=acct,
        type=type_norm,
        amount=float(amt),
        currency=ccy,
        note=(str(note) if note else None),
    )

    payload = asdict(cf)

    key = runtime_dt_key(cfg, CASHFLOWS_TABLE, as_of_norm, f"cashflow_{cf.cashflow_id}.json")
    latest_key = runtime_engine_key(cfg, CASHFLOWS_TABLE, "latest.json")
    index_key = cashflows_index_key(cfg)

    print("\n=== RECORD CASHFLOW ===")
    print(f"env:          {cfg.env}")
    print(f"bucket:       {cfg.bucket}")
    print(f"root:         {cfg.engine_root}")
    print(f"as_of:        {cf.as_of}")
    print(f"cashflow_id:  {cf.cashflow_id}")
    print(f"ts_utc:       {cf.ts_utc}")
    print(f"account_id:   {cf.account_id}")
    print(f"type:         {cf.type}")
    print(f"amount:       {cf.amount} {cf.currency}")

    if cf.note:
        print(f"note:         {cf.note}")

    print("")

    output_keys = [key, latest_key, index_key]
    audit = build_audit_event(
        cfg=cfg,
        run_id=run_id,
        event_type="create",
        entity_type="cashflow",
        entity_id=str(cf.cashflow_id),
        as_of=as_of_norm,
        source_script="record_cashflow.py",
        source_mode="record",
        status=("dry_run" if dry_run else "success"),
        reason=reason,
        input_args=input_args,
        output_keys=output_keys,
        after_payload=payload,
        metadata={"cashflow_type": cf.type, "amount": cf.amount, "currency": cf.currency},
    )

    if dry_run:
        print("[DRY RUN] Would write:")
        print(f"  s3://{cfg.bucket}/{key}")
        print(f"  s3://{cfg.bucket}/{latest_key}")
        print(f"  s3://{cfg.bucket}/{index_key} (update cashflow_id mapping)")
        write_audit_event(cfg=cfg, event=audit, dry_run=True)
        return

    s3_put_json(s3, bucket=cfg.bucket, key=key, payload=payload)
    s3_put_json(s3, bucket=cfg.bucket, key=latest_key, payload=payload)
    set_index_record(s3, cfg=cfg, index_key=index_key, entity_id=str(cf.cashflow_id), key=key, as_of=as_of_norm)
    write_audit_event(cfg=cfg, event=audit, dry_run=False)

    print("[OK] Wrote cashflow:")
    print(f"  s3://{cfg.bucket}/{key}")
    print("[OK] Updated latest:")
    print(f"  s3://{cfg.bucket}/{latest_key}")
    print("[OK] Updated index:")
    print(f"  s3://{cfg.bucket}/{index_key}")
    print("")


def edit_cashflow(
    *,
    cfg: RuntimeConfig,
    cashflow_id: str,
    old_as_of: Optional[str],
    patch: Dict[str, Any],
    new_as_of: Optional[str] = None,
    dry_run: bool = False,
    run_id: Optional[str] = None,
    input_args: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
) -> None:
    s3 = s3_client(cfg)
    index_key = cashflows_index_key(cfg)

    old_key, old_dt, resolved_by = resolve_record_key(
        s3,
        cfg=cfg,
        table=CASHFLOWS_TABLE,
        file_prefix="cashflow",
        entity_id=str(cashflow_id),
        as_of=old_as_of,
        index_key=index_key,
    )

    if not s3_exists(s3, bucket=cfg.bucket, key=old_key):
        raise RuntimeError(f"Cashflow not found: s3://{cfg.bucket}/{old_key}")

    obj = s3_get_json_optional(s3, bucket=cfg.bucket, key=old_key)
    if not isinstance(obj, dict):
        raise RuntimeError("Cashflow JSON is not an object")

    before_payload = dict(obj)

    for k, v in patch.items():
        obj[k] = v

    if new_as_of:
        obj["as_of"] = _parse_date(new_as_of)

    obj["type"] = _normalize_type(obj.get("type"))
    obj["amount"] = _validate_positive("amount", obj.get("amount"))
    obj["currency"] = str(obj.get("currency") or "USD").upper().strip() or "USD"
    obj["account_id"] = str(obj.get("account_id") or "main").strip() or "main"

    dst_dt = _parse_date(str(obj.get("as_of") or old_dt or ""))
    dst_key = runtime_dt_key(cfg, CASHFLOWS_TABLE, dst_dt, f"cashflow_{cashflow_id}.json")
    latest_key = runtime_engine_key(cfg, CASHFLOWS_TABLE, "latest.json")

    print("\n=== EDIT CASHFLOW ===")
    print(f"env:          {cfg.env}")
    print(f"bucket:       {cfg.bucket}")
    print(f"root:         {cfg.engine_root}")
    print(f"cashflow_id:  {cashflow_id}")
    print(f"resolved_by:  {resolved_by}")
    print(f"from:         s3://{cfg.bucket}/{old_key}")
    print(f"to:           s3://{cfg.bucket}/{dst_key}")
    print(f"patch_keys:   {sorted(list(patch.keys()))}")
    print("")

    output_keys = [dst_key, latest_key, index_key]
    deleted_keys = [old_key] if dst_key != old_key else []
    audit = build_audit_event(
        cfg=cfg,
        run_id=run_id,
        event_type="modify",
        entity_type="cashflow",
        entity_id=str(cashflow_id),
        as_of=dst_dt,
        source_script="record_cashflow.py",
        source_mode="edit",
        status=("dry_run" if dry_run else "success"),
        reason=reason,
        input_args=input_args,
        output_keys=output_keys,
        deleted_keys=deleted_keys,
        before_payload=before_payload,
        after_payload=obj,
        metadata={"patch_keys": sorted(list(patch.keys())), "old_key": old_key, "new_key": dst_key},
    )

    if dry_run:
        print("[DRY RUN] Would update JSON, latest, and index.")
        write_audit_event(cfg=cfg, event=audit, dry_run=True)
        return

    if dst_key != old_key:
        s3_copy(s3, bucket=cfg.bucket, src_key=old_key, dst_key=dst_key)
        s3_delete(s3, bucket=cfg.bucket, key=old_key)

    s3_put_json(s3, bucket=cfg.bucket, key=dst_key, payload=obj)
    s3_put_json(s3, bucket=cfg.bucket, key=latest_key, payload=obj)
    set_index_record(s3, cfg=cfg, index_key=index_key, entity_id=str(cashflow_id), key=dst_key, as_of=dst_dt)
    write_audit_event(cfg=cfg, event=audit, dry_run=False)

    print("[OK] Updated cashflow:")
    print(f"  s3://{cfg.bucket}/{dst_key}")
    print("[OK] Updated latest:")
    print(f"  s3://{cfg.bucket}/{latest_key}")
    print("[OK] Updated index:")
    print(f"  s3://{cfg.bucket}/{index_key}")
    print("")


def delete_cashflow(
    *,
    cfg: RuntimeConfig,
    cashflow_id: str,
    old_as_of: Optional[str] = None,
    reason: Optional[str] = None,
    dry_run: bool = False,
    run_id: Optional[str] = None,
    input_args: Optional[Dict[str, Any]] = None,
) -> None:
    delete_record_with_audit(
        cfg=cfg,
        table=CASHFLOWS_TABLE,
        entity_type="cashflow",
        entity_id=str(cashflow_id),
        id_field="cashflow_id",
        file_prefix="cashflow",
        as_of=old_as_of,
        index_key=cashflows_index_key(cfg),
        source_script="record_cashflow.py",
        source_mode="delete",
        reason=reason,
        dry_run=dry_run,
        run_id=run_id,
        input_args=input_args,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Record/edit/delete/migrate cashflow records in S3.")

    ap.add_argument("--mode", choices=["record", "edit", "delete", "migrate"], default="record")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument("--as-of", default=None, help="Date YYYY-MM-DD. Required for record mode.")
    ap.add_argument("--type", default=None, choices=["DEPOSIT", "WITHDRAWAL", "deposit", "withdrawal"])
    ap.add_argument("--amount", default=None, type=float)
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--account-id", default="main")

    ap.add_argument("--ts-utc", default=None)
    ap.add_argument("--cashflow-id", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--old-as-of", default=None, help="Existing record date. Used for edit/delete if index is unavailable/stale.")
    ap.add_argument("--new-as-of", default=None, help="Move cashflow to a new dt partition in edit mode.")
    ap.add_argument("--reason", default=None, help="Business reason. Required for real delete operations.")
    ap.add_argument("--dry-run", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)

    if not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    input_args = vars(args)

    with capture_script_run(
        cfg=cfg,
        script_name="record_cashflow.py",
        input_args=input_args,
        dry_run=bool(args.dry_run),
    ) as run_id:
        if args.mode == "migrate":
            migrate_cashflows_index(
                cfg=cfg,
                dry_run=bool(args.dry_run),
                run_id=run_id,
                input_args=input_args,
                reason=args.reason,
            )
            return

        if args.mode == "delete":
            if not args.cashflow_id:
                raise ValueError("--cashflow-id is required for --mode delete")

            delete_cashflow(
                cfg=cfg,
                cashflow_id=str(args.cashflow_id),
                old_as_of=(str(args.old_as_of) if args.old_as_of else None),
                reason=args.reason,
                dry_run=bool(args.dry_run),
                run_id=run_id,
                input_args=input_args,
            )
            return

        if args.mode == "edit":
            if not args.cashflow_id:
                raise ValueError("--cashflow-id is required for --mode edit")

            patch: Dict[str, Any] = {}
            if args.as_of is not None:
                patch["as_of"] = _parse_date(args.as_of)
            if args.ts_utc is not None:
                patch["ts_utc"] = str(args.ts_utc)
            if args.type is not None:
                patch["type"] = _normalize_type(args.type)
            if args.amount is not None:
                patch["amount"] = _validate_positive("amount", args.amount)
            if args.currency is not None:
                patch["currency"] = str(args.currency).upper().strip() or "USD"
            if args.account_id is not None:
                patch["account_id"] = str(args.account_id).strip() or "main"
            if args.note is not None:
                patch["note"] = (str(args.note) if args.note else None)

            if not patch and not args.new_as_of:
                raise ValueError("Nothing to edit: provide at least one patch field or --new-as-of.")

            edit_cashflow(
                cfg=cfg,
                cashflow_id=str(args.cashflow_id),
                old_as_of=(str(args.old_as_of) if args.old_as_of else None),
                new_as_of=(str(args.new_as_of) if args.new_as_of else None),
                patch=patch,
                dry_run=bool(args.dry_run),
                run_id=run_id,
                input_args=input_args,
                reason=args.reason,
            )
            return

        if not args.as_of:
            raise ValueError("--as-of is required for --mode record")
        if args.type is None:
            raise ValueError("--type is required for --mode record")
        if args.amount is None:
            raise ValueError("--amount is required for --mode record")

        record_cashflow(
            cfg=cfg,
            as_of=str(args.as_of),
            type=str(args.type),
            amount=float(args.amount),
            currency=str(args.currency),
            account_id=str(args.account_id),
            ts_utc=args.ts_utc,
            cashflow_id=args.cashflow_id,
            note=args.note,
            dry_run=bool(args.dry_run),
            run_id=run_id,
            input_args=input_args,
            reason=args.reason,
        )


if __name__ == "__main__":
    main()
