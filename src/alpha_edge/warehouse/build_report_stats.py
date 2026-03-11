from __future__ import annotations

import argparse
import io
import json
from typing import Any, Optional

import boto3
import pandas as pd
import pyarrow as pa

# Reuse the existing warehouse builder utilities (single source of truth)
from alpha_edge.warehouse.build_warehouse import (
    DEFAULT_BUCKET,
    DEFAULT_ENGINE_ROOT,
    DEFAULT_REGION,
    lake_key,
    now_ts_utc_ms,
    parse_date,
    s3_get_bytes,
    s3_put_parquet_table,
    s3_client,
    wh_key,
    build_fct_daily_report_stats_for_dt,
)

WAREHOUSE_VERSION = "v=1"  # already used inside wh_key()

def s3_key_exists(s3, *, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build ONLY fct_daily_report_stats warehouse partition for a single dt."
    )
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--engine-root", default=DEFAULT_ENGINE_ROOT)

    ap.add_argument("--dt", required=True, help="Partition date YYYY-MM-DD")
    ap.add_argument("--account-id", default="main")

    ap.add_argument(
        "--report-key",
        default=None,
        help="S3 key to report.json for this dt. If omitted, defaults to engine_root/reports/dt=DT/report.json",
    )

    ap.add_argument("--force", action="store_true", help="Rewrite partition even if it exists.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    bucket = str(args.bucket)
    region = str(args.region)
    engine_root = str(args.engine_root).strip("/")
    account_id = str(args.account_id)
    dt_str = parse_date(args.dt)

    s3 = s3_client(region)
    load_ts = now_ts_utc_ms()

    # default report key
    report_key = args.report_key
    if report_key is None:
        report_key = lake_key(engine_root, "reports", f"dt={dt_str}", "report.json")

    out_key = wh_key(engine_root, "fct_daily_report_stats", f"dt={dt_str}", "part-00000.parquet")

    if (not args.force) and s3_key_exists(s3, bucket=bucket, key=out_key):
        print(f"[OK] already exists -> skipped: s3://{bucket}/{out_key}")
        return

    table = build_fct_daily_report_stats_for_dt(
        s3,
        bucket=bucket,
        report_key=report_key,
        dt=dt_str,
        account_id=account_id,
        load_ts=load_ts,
    )

    if table is None:
        print(f"[WARN] report missing -> skipped (expected key: s3://{bucket}/{report_key})")
        return

    print(f"[fct_daily_report_stats] rows={table.num_rows} -> s3://{bucket}/{out_key}")
    if args.dry_run:
        print("[DRY RUN] no write performed.")
        return

    s3_put_parquet_table(s3, bucket=bucket, key=out_key, table=table)
    print("[OK] done.")

if __name__ == "__main__":
    main()