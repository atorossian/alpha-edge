from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from typing import Any, Optional

import boto3

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import RuntimeConfig


TRADES_TABLE = "trades"


# ------------------------------------------------
# S3 helpers
# ------------------------------------------------
def s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


def engine_key(cfg: RuntimeConfig, *parts: str) -> str:
    return "/".join([cfg.engine_root.strip("/")] + [p.strip("/") for p in parts])


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


# ------------------------------------------------
# Discover dates
# ------------------------------------------------
def discover_first_activity_date(
    s3,
    *,
    cfg: RuntimeConfig,
) -> dt.date:
    prefix = engine_key(cfg, TRADES_TABLE, "dt=")
    keys = s3_list_keys(s3, bucket=cfg.bucket, prefix=prefix)

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
        raise RuntimeError(f"No trades found in s3://{cfg.bucket}/{prefix}")

    return min(dates)


def discover_first_trade_for_asset(
    s3,
    *,
    cfg: RuntimeConfig,
    asset_id: Optional[str],
    ticker: Optional[str],
) -> dt.date:
    prefix = engine_key(cfg, TRADES_TABLE, "dt=")
    keys = s3_list_keys(s3, bucket=cfg.bucket, prefix=prefix)
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]

    first: Optional[dt.date] = None

    for k in keys:
        trade = s3_get_json(s3, bucket=cfg.bucket, key=k)

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
        raise RuntimeError(
            f"Asset not found in trades under s3://{cfg.bucket}/{prefix} "
            f"(asset_id={asset_id!r}, ticker={ticker!r})"
        )

    return first


# ------------------------------------------------
# Runner
# ------------------------------------------------
def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {' '.join(cmd)}")


# ------------------------------------------------
# CLI
# ------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Repair ledger + warehouse history for one asset over an affected date range."
    )

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

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
        help="Pass --write-checkpoints to rebuild_ledger during repair.",
    )
    ap.add_argument(
        "--checkpoint-policy",
        choices=["month_end", "always"],
        default="month_end",
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

    cfg = load_runtime_config(args.env)

    if not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    with capture_script_run(
        cfg=cfg,
        script_name="repair_asset_history.py",
        input_args=vars(args),
        dry_run=bool(args.dry_run),
    ) as run_id:
        if not args.asset_id and not args.ticker:
            raise ValueError("Provide at least --asset-id or --ticker")

        s3 = s3_client(cfg)

        first_account_activity = discover_first_activity_date(
            s3,
            cfg=cfg,
        )

        if args.start:
            repair_start = parse_date(args.start)
        else:
            repair_start = discover_first_trade_for_asset(
                s3,
                cfg=cfg,
                asset_id=args.asset_id,
                ticker=args.ticker,
            )

        repair_end = parse_date(args.end) if args.end else dt.date.today()

        print()
        print("=== REPAIR ASSET HISTORY ===")
        print(f"env:                {cfg.env}")
        print(f"bucket:             {cfg.bucket}")
        print(f"region:             {cfg.region}")
        print(f"root:               {cfg.engine_root}")
        print(f"asset_id:           {args.asset_id}")
        print(f"ticker:             {args.ticker}")
        print(f"ledger_start:       {fmt_date(first_account_activity)}")
        print(f"repair_range:       {fmt_date(repair_start)} -> {fmt_date(repair_end)}")
        print(f"use_checkpoints:    {bool(args.use_checkpoints)}")
        print(f"write_checkpoints:  {bool(args.write_checkpoints)}")
        print(f"checkpoint_policy:  {args.checkpoint_policy}")
        print(f"dry_run:            {bool(args.dry_run)}")
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
                    "--env",
                    cfg.env,
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

                if cfg.is_prod:
                    ledger_cmd.append("--confirm-prod-write")

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
                    ledger_cmd.append("--dry-run")
                    print("[dry]", " ".join(ledger_cmd))
                else:
                    run_cmd(ledger_cmd)

                wh_cmd = [
                    sys.executable,
                    "-m",
                    args.warehouse_module,
                    "--env",
                    cfg.env,
                    "--dt",
                    dt_str,
                    "--account-id",
                    args.account_id,
                ]

                if cfg.is_prod:
                    wh_cmd.append("--confirm-prod-write")

                if args.dry_run:
                    wh_cmd.append("--dry-run")
                    print("[dry]", " ".join(wh_cmd))
                else:
                    run_cmd(wh_cmd)

            except Exception as e:
                msg = f"{dt_str} :: {type(e).__name__}: {e}"
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

        repair_dates = [fmt_date(d) for d in daterange(repair_start, repair_end)]
        audit_event = build_audit_event(
            cfg=cfg,
            run_id=run_id,
            event_type="repair",
            entity_type="asset_history",
            entity_id=(args.asset_id or args.ticker),
            as_of=fmt_date(repair_end),
            source_script="repair_asset_history.py",
            source_mode="repair",
            status=("dry_run" if args.dry_run else "success"),
            reason=None,
            input_args=vars(args),
            metadata={
                "asset_id": args.asset_id,
                "ticker": args.ticker,
                "ledger_start": fmt_date(first_account_activity),
                "repair_start": fmt_date(repair_start),
                "repair_end": fmt_date(repair_end),
                "repair_dates_n": len(repair_dates),
                "repair_dates": repair_dates,
                "ledger_module": args.ledger_module,
                "warehouse_module": args.warehouse_module,
                "note": "Child ledger/warehouse commands write their own audit events when patched.",
            },
        )
        write_audit_event(cfg=cfg, event=audit_event, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()