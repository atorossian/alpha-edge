from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from typing import Optional, Any

import boto3
import pandas as pd


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT = "engine/v1"

TRADES_TABLE = "trades"


# ------------------------------------------------
# S3 helpers
# ------------------------------------------------
def s3_client(region: str):
    return boto3.client("s3", region_name=region)


def s3_list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None

    while True:
        kwargs: dict[str, Any] = dict(Bucket=bucket, Prefix=prefix)

        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)

        for it in resp.get("Contents", []):
            k = it.get("Key")
            if isinstance(k, str):
                keys.append(k)

        if not resp.get("IsTruncated"):
            break

        token = resp.get("NextContinuationToken")

    return keys


def s3_get_json(s3, *, bucket: str, key: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def parse_date(x: str) -> dt.date:
    return dt.date.fromisoformat(str(x).strip())


def fmt_date(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + dt.timedelta(days=1)


def is_month_end(d: dt.date) -> bool:
    return (d + dt.timedelta(days=1)).month != d.month


# ------------------------------------------------
# Discover dates
# ------------------------------------------------
def discover_first_activity_date(
    s3,
    *,
    bucket: str,
    engine_root: str,
) -> dt.date:
    prefix = f"{engine_root}/{TRADES_TABLE}/dt="
    keys = s3_list_keys(s3, bucket=bucket, prefix=prefix)

    dates: list[dt.date] = []

    for k in keys:
        parts = k.split("/")
        for p in parts:
            if p.startswith("dt="):
                try:
                    dates.append(parse_date(p.replace("dt=", "")))
                except Exception:
                    pass

    if not dates:
        raise RuntimeError("No trades found in S3.")

    return min(dates)


def discover_first_trade_for_asset(
    s3,
    *,
    bucket: str,
    engine_root: str,
    asset_id: Optional[str],
    ticker: Optional[str],
) -> dt.date:
    prefix = f"{engine_root}/{TRADES_TABLE}/dt="
    keys = s3_list_keys(s3, bucket=bucket, prefix=prefix)
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]

    first: Optional[dt.date] = None

    for k in keys:
        trade = s3_get_json(s3, bucket=bucket, key=k)

        aid = str(trade.get("asset_id", "")).strip()
        tkr = str(trade.get("ticker", "")).upper().strip()

        if asset_id and aid != str(asset_id).strip():
            continue

        if ticker and tkr != str(ticker).upper().strip():
            continue

        as_of = parse_date(trade["as_of"])

        if first is None or as_of < first:
            first = as_of

    if first is None:
        raise RuntimeError("Asset not found in trades.")

    return first


# ------------------------------------------------
# Runner
# ------------------------------------------------
def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}")


# ------------------------------------------------
# CLI
# ------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Repair ledger + warehouse history for one asset over an affected date range.")

    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--engine-root", default=DEFAULT_ENGINE_ROOT)

    ap.add_argument("--asset-id", default=None)
    ap.add_argument("--ticker", default=None)

    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)

    ap.add_argument("--account-id", default="main")

    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")

    ap.add_argument("--use-checkpoints", action="store_true", help="Pass --use-checkpoints to rebuild_ledger.")
    ap.add_argument(
        "--write-checkpoints",
        action="store_true",
        help="Pass --write-checkpoints to rebuild_ledger for checkpoint emission during repair.",
    )
    ap.add_argument(
        "--checkpoint-policy",
        choices=["month_end", "always"],
        default="month_end",
        help="Checkpoint emission policy passed to rebuild_ledger.",
    )

    ap.add_argument(
        "--ledger-module",
        default="alpha_edge.operations.rebuild_ledger",
    )
    ap.add_argument(
        "--warehouse-module",
        default="alpha_edge.warehouse.build_warehouse",
    )

    return ap.parse_args()


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    args = parse_args()

    s3 = s3_client(args.region)

    first_account_activity = discover_first_activity_date(
        s3,
        bucket=args.bucket,
        engine_root=args.engine_root,
    )

    if args.start:
        repair_start = parse_date(args.start)
    else:
        repair_start = discover_first_trade_for_asset(
            s3,
            bucket=args.bucket,
            engine_root=args.engine_root,
            asset_id=args.asset_id,
            ticker=args.ticker,
        )

    repair_end = parse_date(args.end) if args.end else dt.date.today()

    print()
    print("=== REPAIR ASSET HISTORY ===")
    print(f"asset_id:           {args.asset_id}")
    print(f"ticker:             {args.ticker}")
    print(f"ledger_start:       {fmt_date(first_account_activity)}")
    print(f"repair_range:       {fmt_date(repair_start)} -> {fmt_date(repair_end)}")
    print(f"use_checkpoints:    {bool(args.use_checkpoints)}")
    print(f"write_checkpoints:  {bool(args.write_checkpoints)}")
    print(f"checkpoint_policy:  {args.checkpoint_policy}")
    print()

    errors: list[str] = []

    for d in daterange(repair_start, repair_end):
        dt_str = fmt_date(d)

        print()
        print(f"--- dt={dt_str} ---")

        try:
            ledger_cmd = [
                sys.executable,
                "-m",
                args.ledger_module,
                "--account-id",
                args.account_id,
                "--start",
                fmt_date(first_account_activity),
                "--end",
                dt_str,
                "--as-of",
                dt_str,
                "--prices-mode",
                "asof",
            ]

            if args.use_checkpoints:
                ledger_cmd.append("--use-checkpoints")

            if args.write_checkpoints:
                ledger_cmd.extend(
                    [
                        "--write-checkpoints",
                        "--checkpoint-policy",
                        args.checkpoint_policy,
                    ]
                )

            if args.dry_run:
                print("[dry]", " ".join(ledger_cmd))
            else:
                run_cmd(ledger_cmd)

            wh_cmd = [
                sys.executable,
                "-m",
                args.warehouse_module,
                "--bucket",
                args.bucket,
                "--region",
                args.region,
                "--engine-root",
                args.engine_root,
                "--dt",
                dt_str,
                "--account-id",
                args.account_id,
            ]

            if args.dry_run:
                print("[dry]", " ".join(wh_cmd))
            else:
                run_cmd(wh_cmd)

        except Exception as e:
            msg = f"{dt_str} :: {e}"
            print("[ERROR]", msg)
            errors.append(msg)

            if args.stop_on_error:
                break

    print()
    print("=== DONE ===")

    if errors:
        print(f"failures={len(errors)}")
        for m in errors:
            print(" -", m)
        raise SystemExit(2)

    print("repair completed.")


if __name__ == "__main__":
    main()