from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run

import argparse

import boto3

from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import RuntimeConfig
from alpha_edge.warehouse.build_warehouse import (
    build_fct_daily_report_stats_for_dt,
    lake_key,
    now_ts_utc_ms,
    parse_date,
    s3_put_parquet_table,
    wh_key,
)


def s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


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

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument("--dt", required=True, help="Partition date YYYY-MM-DD")
    ap.add_argument("--account-id", default="main")

    ap.add_argument(
        "--report-key",
        default=None,
        help="S3 key to report.json. If omitted, defaults to <env-root>/daily_reports/dt=DT/report.json",
    )

    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    return ap.parse_args()


def _main_impl() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)

    if not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    account_id = str(args.account_id)
    dt_str = parse_date(args.dt)

    s3 = s3_client(cfg)
    load_ts = now_ts_utc_ms()

    report_key = args.report_key
    if report_key is None:
        report_key = lake_key(cfg, "daily_reports", f"dt={dt_str}", "report.json")

    out_key = wh_key(cfg, "fct_daily_report_stats", f"dt={dt_str}", "part-00000.parquet")

    print("\n=== BUILD DAILY REPORT STATS ONLY ===")
    print(f"env:        {cfg.env}")
    print(f"bucket:     {cfg.bucket}")
    print(f"region:     {cfg.region}")
    print(f"root:       {cfg.engine_root}")
    print(f"dt:         {dt_str}")
    print(f"account_id: {account_id}")
    print(f"report_key: s3://{cfg.bucket}/{report_key}")
    print(f"out_key:    s3://{cfg.bucket}/{out_key}")
    print(f"force:      {bool(args.force)}")
    print(f"dry_run:    {bool(args.dry_run)}")
    print("")

    if (not args.force) and s3_key_exists(s3, bucket=cfg.bucket, key=out_key):
        print(f"[OK] already exists -> skipped: s3://{cfg.bucket}/{out_key}")
        return

    table = build_fct_daily_report_stats_for_dt(
        s3,
        cfg=cfg,
        report_key=report_key,
        dt=dt_str,
        account_id=account_id,
        load_ts=load_ts,
    )

    if table is None:
        print(f"[WARN] report missing -> skipped expected=s3://{cfg.bucket}/{report_key}")
        return

    print(f"[fct_daily_report_stats] rows={table.num_rows} -> s3://{cfg.bucket}/{out_key}")

    if args.dry_run:
        print("[DRY RUN] no write performed.")
        return

    s3_put_parquet_table(s3, bucket=cfg.bucket, key=out_key, table=table)

    print("[OK] done.")


# ----------------------------
# Audit/logging entrypoint wrapper
# ----------------------------
def _tier1_audit_is_dry_run(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "dry_run", False) or getattr(args, "no_write", False))


def main_with_audit() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = _tier1_audit_is_dry_run(args)

    with capture_script_run(
        cfg=cfg,
        script_name="build_report_stats.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            _main_impl()

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="daily_report_stats",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="build_report_stats.py",
                source_mode="daily_report_stats",
                status=("dry_run" if is_dry_run else "success"),
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                    "note": "Tier 1 audit event is entrypoint-level. Detailed output keys/row counts are available in the script log stdout and script-specific metadata where emitted by the script.",
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
        except Exception as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="daily_report_stats",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="build_report_stats.py",
                source_mode="daily_report_stats",
                status="failed",
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise


def main() -> None:
    main_with_audit()


if __name__ == "__main__":
    main()
