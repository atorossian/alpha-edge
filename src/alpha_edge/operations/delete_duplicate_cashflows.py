from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import boto3

from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation


def s3_get_json(s3, *, bucket: str, key: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Delete duplicated broker-cash cashflows after backing them up.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--env", default="prod", choices=["dev", "staging", "prod"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-prod-write", action="store_true")
    ap.add_argument("--backup-prefix", default="engine/v1/cashflows_audit/duplicate_cleanup_20260809")
    args = ap.parse_args()

    cfg = load_runtime_config(args.env)
    s3 = boto3.client("s3", region_name=cfg.region)

    if not args.dry_run:
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("=== CASHFLOW DUPLICATE CLEANUP ===")
    print("env:", cfg.env)
    print("bucket:", cfg.bucket)
    print("rows:", len(rows))
    print("dry_run:", bool(args.dry_run))
    print("")

    deleted = 0
    skipped = 0

    for i, row in enumerate(rows, start=1):
        key = str(row["delete_s3_key"]).strip()
        expected_id = str(row["delete_cashflow_id"]).strip()
        keep_id = str(row.get("keep_cashflow_id", "")).strip()

        if not key:
            print(f"[SKIP] row={i}: missing key")
            skipped += 1
            continue

        try:
            payload = s3_get_json(s3, bucket=cfg.bucket, key=key)
        except Exception as e:
            print(f"[SKIP] row={i}: cannot read {key}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        got_id = str(payload.get("cashflow_id", "")).strip()
        if got_id != expected_id:
            print(f"[SKIP] row={i}: cashflow_id mismatch for {key}: got={got_id!r}, expected={expected_id!r}")
            skipped += 1
            continue

        backup_key = (
            args.backup_prefix.strip("/") +
            f"/dt={row.get('as_of', 'unknown')}/cashflow_{expected_id}.json"
        )

        print(f"[DELETE] duplicate={expected_id} keep={keep_id}")
        print(f"         key=s3://{cfg.bucket}/{key}")
        print(f"         backup=s3://{cfg.bucket}/{backup_key}")

        if args.dry_run:
            continue

        s3.copy_object(
            Bucket=cfg.bucket,
            CopySource={"Bucket": cfg.bucket, "Key": key},
            Key=backup_key,
            MetadataDirective="COPY",
        )
        s3.delete_object(Bucket=cfg.bucket, Key=key)
        deleted += 1

    print("")
    print("deleted:", deleted)
    print("skipped:", skipped)
    if args.dry_run:
        print("[DRY RUN] no S3 objects were modified.")


if __name__ == "__main__":
    main()
