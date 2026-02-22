from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional

import boto3


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT = "engine/v1"

TRADES_TABLE = "trades"
LEDGER_TABLE = "ledger"

WAREHOUSE_ROOT = "warehouse"
WAREHOUSE_VERSION = "v=1"


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(region: str):
    return boto3.client("s3", region_name=region)


def join_key(*parts: str) -> str:
    return "/".join([p.strip("/") for p in parts if p is not None and str(p).strip("/") != ""])


def lake_key(engine_root: str, *parts: str) -> str:
    return join_key(engine_root, *parts)


def wh_key(engine_root: str, table: str, *parts: str) -> str:
    return join_key(engine_root, WAREHOUSE_ROOT, table, WAREHOUSE_VERSION, *parts)


def s3_list_objects(s3, *, bucket: str, prefix: str) -> list[dict]:
    out: list[dict] = []
    token = None
    while True:
        kwargs: dict[str, Any] = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        out.extend(resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


def s3_key_exists(s3, *, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


# ----------------------------
# Date helpers
# ----------------------------
_DT_PART_RE = re.compile(r"/dt=(\d{4}-\d{2}-\d{2})/")


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(str(s).strip())


def fmt_date(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def daterange(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        return []
    out: list[dt.date] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = cur + dt.timedelta(days=1)
    return out


def discover_first_trade_date(s3, *, bucket: str, engine_root: str) -> Optional[dt.date]:
    """
    Earliest dt=... folder under engine_root/trades/.
    """
    prefix = lake_key(engine_root, TRADES_TABLE, "dt=")
    objs = s3_list_objects(s3, bucket=bucket, prefix=prefix)
    if not objs:
        return None

    dates: list[dt.date] = []
    for o in objs:
        k = o.get("Key")
        if not isinstance(k, str):
            continue
        m = _DT_PART_RE.search(k.replace("\\", "/"))
        if not m:
            continue
        try:
            dates.append(parse_date(m.group(1)))
        except Exception:
            continue

    return min(dates) if dates else None


def discover_latest_dt_under_prefix(s3, *, bucket: str, prefix: str) -> Optional[dt.date]:
    objs = s3_list_objects(s3, bucket=bucket, prefix=prefix)
    if not objs:
        return None

    latest: Optional[dt.date] = None
    for o in objs:
        k = o.get("Key")
        if not isinstance(k, str):
            continue
        m = _DT_PART_RE.search(k.replace("\\", "/"))
        if not m:
            continue
        try:
            d = parse_date(m.group(1))
        except Exception:
            continue
        if latest is None or d > latest:
            latest = d
    return latest


def discover_latest_warehouse_dt(s3, *, bucket: str, engine_root: str) -> Optional[dt.date]:
    """
    Resume point is the most recent dt among the main fact tables.
    """
    candidates = [
        wh_key(engine_root, "fct_account_pnl_daily", ""),
        wh_key(engine_root, "fct_positions_daily", ""),
        wh_key(engine_root, "fct_trades", ""),
    ]
    for p in candidates:
        latest = discover_latest_dt_under_prefix(s3, bucket=bucket, prefix=p)
        if latest is not None:
            return latest
    return None


# ----------------------------
# Runner logic
# ----------------------------
@dataclass
class DayPlan:
    dt: str
    need_ledger: bool
    need_wh_trades: bool
    need_wh_positions: bool
    need_wh_pnl: bool
    need_wh_report: bool


def plan_for_day(
    s3,
    *,
    bucket: str,
    engine_root: str,
    dt_str: str,
    force_ledger: bool,
    force_warehouse: bool,
) -> DayPlan:
    ledger_positions_key = lake_key(engine_root, LEDGER_TABLE, f"dt={dt_str}", "positions.json")
    ledger_pnl_key = lake_key(engine_root, LEDGER_TABLE, f"dt={dt_str}", "pnl.json")

    ledger_exists = s3_key_exists(s3, bucket=bucket, key=ledger_positions_key) and s3_key_exists(
        s3, bucket=bucket, key=ledger_pnl_key
    )
    need_ledger = (not ledger_exists) or force_ledger

    wh_trades_key = wh_key(engine_root, "fct_trades", f"dt={dt_str}", "part-00000.parquet")
    wh_positions_key = wh_key(engine_root, "fct_positions_daily", f"dt={dt_str}", "part-00000.parquet")
    wh_pnl_key = wh_key(engine_root, "fct_account_pnl_daily", f"dt={dt_str}", "part-00000.parquet")
    wh_report_key = wh_key(engine_root, "fct_daily_report_stats", f"dt={dt_str}", "part-00000.parquet")

    wh_trades_exists = s3_key_exists(s3, bucket=bucket, key=wh_trades_key)
    wh_positions_exists = s3_key_exists(s3, bucket=bucket, key=wh_positions_key)
    wh_pnl_exists = s3_key_exists(s3, bucket=bucket, key=wh_pnl_key)
    wh_report_exists = s3_key_exists(s3, bucket=bucket, key=wh_report_key)

    need_wh_trades = (not wh_trades_exists) or force_warehouse
    need_wh_positions = (not wh_positions_exists) or force_warehouse
    need_wh_pnl = (not wh_pnl_exists) or force_warehouse
    need_wh_report = (not wh_report_exists) or force_warehouse

    return DayPlan(
        dt=dt_str,
        need_ledger=need_ledger,
        need_wh_trades=need_wh_trades,
        need_wh_positions=need_wh_positions,
        need_wh_pnl=need_wh_pnl,
        need_wh_report=need_wh_report,
    )


def run_subprocess(cmd: list[str]) -> None:
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Daily warehouse update: rebuild_ledger (Option A) + build_warehouse for calendar days."
    )

    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--engine-root", default=DEFAULT_ENGINE_ROOT)

    ap.add_argument(
        "--start",
        default="auto",
        help="Start date YYYY-MM-DD, or 'auto' to resume from latest warehouse dt + 1 (default: auto).",
    )
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD. Default: today (local machine date).")

    ap.add_argument("--account-id", default="main")

    ap.add_argument("--build-dim-assets", action="store_true", help="Build dim_assets snapshot once at the start.")
    ap.add_argument("--universe-path", default=None, help="Local universe.csv path (required if --build-dim-assets).")

    ap.add_argument("--force-ledger", action="store_true", help="Rebuild ledger even if it exists for dt.")
    ap.add_argument("--force-warehouse", action="store_true", help="Rewrite warehouse partitions even if they exist.")
    ap.add_argument("--dry-run", action="store_true", help="Print what would run but do not execute.")
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately on first failing day. Default: continue and print errors.",
    )

    ap.add_argument(
        "--ledger-prices-mode",
        choices=["asof", "latest"],
        default="asof",
        help="Pricing mode to pass to rebuild_ledger.py. Use 'asof' for historical daily MTM.",
    )

    ap.add_argument(
        "--rebuild-ledger-module",
        default="alpha_edge.operations.rebuild_ledger",
        help="Python module to execute for ledger rebuild (must support --as-of, --start, --end, --prices-mode).",
    )
    ap.add_argument(
        "--build-warehouse-module",
        default="alpha_edge.warehouse.build_warehouse",
        help="Python module to execute for warehouse build (must support --dt).",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    bucket = str(args.bucket)
    region = str(args.region)
    engine_root = str(args.engine_root).strip("/")
    account_id = str(args.account_id)

    s3 = s3_client(region)

    # This is the ONLY correct start for rebuild_ledger: we must include all prior trades to carry positions forward.
    first_trade_d = discover_first_trade_date(s3, bucket=bucket, engine_root=engine_root)
    if first_trade_d is None:
        raise SystemExit("No trades found under engine_root/trades/dt=...")

    end_d = parse_date(args.end) if args.end else dt.date.today()

    # Warehouse start (resume)
    start_arg = str(args.start).strip().lower()
    if start_arg == "auto":
        latest_wh = discover_latest_warehouse_dt(s3, bucket=bucket, engine_root=engine_root)
        if latest_wh is not None:
            wh_start_d = latest_wh + dt.timedelta(days=1)
        else:
            wh_start_d = first_trade_d
    else:
        wh_start_d = parse_date(args.start)

    if wh_start_d > end_d:
        print(f"[OK] nothing to do: start={fmt_date(wh_start_d)} is after end={fmt_date(end_d)}")
        return

    days = daterange(wh_start_d, end_d)
    if not days:
        print("[OK] nothing to do (empty date range).")
        return

    print("\n=== WAREHOUSE DAILY UPDATE (OPTION A) ===")
    print(f"bucket:        {bucket}")
    print(f"engine_root:   {engine_root}")
    print(f"region:        {region}")
    print(f"account_id:    {account_id}")
    print(f"ledger_start:  {fmt_date(first_trade_d)} (first trade dt)")
    print(f"warehouse_rng: {fmt_date(wh_start_d)} -> {fmt_date(end_d)} ({len(days)} calendar days)")
    print(f"dry_run:       {bool(args.dry_run)}")
    print(f"prices_mode:   {str(args.ledger_prices_mode)}")
    print("")

    # Optional dim_assets (once)
    if args.build_dim_assets:
        if not args.universe_path:
            raise SystemExit("--build-dim-assets requires --universe-path")
        cmd = [
            sys.executable,
            "-m",
            str(args.build_warehouse_module),
            "--bucket",
            bucket,
            "--region",
            region,
            "--engine-root",
            engine_root,
            "--dt",
            fmt_date(wh_start_d),
            "--account-id",
            account_id,
            "--build-dim-assets",
            "--dim-assets-only",
            "--universe-path",
            str(args.universe_path),
        ]
        if args.dry_run:
            cmd.append("--dry-run")

        print("[plan] build dim_assets snapshot once (dim-assets-only)")
        if args.dry_run:
            print("[dry] would run:")
            print(" ".join(cmd))
        else:
            run_subprocess(cmd)
        print("")

    errors: list[str] = []
    for d in days:
        dt_str = fmt_date(d)

        plan = plan_for_day(
            s3,
            bucket=bucket,
            engine_root=engine_root,
            dt_str=dt_str,
            force_ledger=bool(args.force_ledger),
            force_warehouse=bool(args.force_warehouse),
        )

        if (
            (not plan.need_ledger)
            and (not plan.need_wh_trades)
            and (not plan.need_wh_positions)
            and (not plan.need_wh_pnl)
            and (not plan.need_wh_report)
        ):
            continue

        print(f"\n--- dt={dt_str} ---")
        print(
            f"[plan] ledger={'YES' if plan.need_ledger else 'no'} | "
            f"wh(trades={'YES' if plan.need_wh_trades else 'no'}, "
            f"positions={'YES' if plan.need_wh_positions else 'no'}, "
            f"pnl={'YES' if plan.need_wh_pnl else 'no'}, "
            f"report={'YES' if plan.need_wh_report else 'no'})"
        )

        try:
            if plan.need_ledger:
                cmd = [
                    sys.executable,
                    "-m",
                    str(args.rebuild_ledger_module),
                    "--start",
                    fmt_date(first_trade_d),  # IMPORTANT FIX
                    "--end",
                    dt_str,
                    "--as-of",
                    dt_str,
                    "--prices-mode",
                    str(args.ledger_prices_mode),
                ]
                if args.dry_run:
                    cmd.append("--dry-run")

                if args.dry_run:
                    print("[dry] would run ledger rebuild:")
                    print(" ".join(cmd))
                else:
                    run_subprocess(cmd)

            cmd = [
                sys.executable,
                "-m",
                str(args.build_warehouse_module),
                "--bucket",
                bucket,
                "--region",
                region,
                "--engine-root",
                engine_root,
                "--dt",
                dt_str,
                "--account-id",
                account_id,
            ]
            if args.dry_run:
                cmd.append("--dry-run")

            if args.dry_run:
                print("[dry] would run warehouse build:")
                print(" ".join(cmd))
            else:
                run_subprocess(cmd)

        except Exception as e:
            msg = f"dt={dt_str} :: {type(e).__name__}: {e}"
            print(f"[ERROR] {msg}")
            errors.append(msg)
            if args.stop_on_error:
                break

    print("\n=== DONE ===")
    if errors:
        print(f"[WARN] failures={len(errors)}")
        for m in errors[:30]:
            print(f" - {m}")
        if len(errors) > 30:
            print(f" - ... ({len(errors) - 30} more)")
        raise SystemExit(2)

    print("[OK] update completed with no errors.")


if __name__ == "__main__":
    main()