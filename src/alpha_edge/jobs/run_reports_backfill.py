# run_reports_backfill.py
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from alpha_edge.core.data_loader import (
    s3_init,
    s3_list_keys,
    s3_load_latest_json_asof,
)
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation
from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.jobs.run_daily_report import run_daily_cycle_asof


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT = "engine/v1"


# ----------------------------
# Runtime helpers
# ----------------------------
def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET))


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION))


def cfg_engine_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "engine_root", DEFAULT_ENGINE_ROOT)).strip("/")


def cfg_env(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "env", "dev"))


# ----------------------------
# Helpers
# ----------------------------
def parse_date(s: str) -> str:
    return pd.Timestamp(s).strftime("%Y-%m-%d")


def key_exists(s3, *, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def list_trade_dts(s3, *, bucket: str, engine_root: str) -> List[str]:
    prefix = f"{engine_root.strip('/')}/trades/"
    keys = s3_list_keys(s3, bucket=bucket, prefix=prefix)
    dts: set[str] = set()

    for k in keys:
        parts = str(k).split("/")
        for p in parts:
            if p.startswith("dt=") and len(p) == len("dt=YYYY-MM-DD"):
                dts.add(p[len("dt="):])
                break

    return sorted(dts)


@dataclass(frozen=True)
class GoalLadderCfg:
    mults: Tuple[float, float, float] = (1.20, 1.40, 1.60)
    round_to: float = 50.0
    min_main_goal_usd: float = 500.0
    min_goal_gap_usd: float = 100.0


def _round_to_step(x: float, step: float) -> float:
    if step is None or step <= 0:
        return float(x)
    return float(step) * round(float(x) / float(step))


def build_goals_from_equity(equity: float, cfg: GoalLadderCfg) -> Tuple[List[float], float]:
    e = float(equity)
    if not math.isfinite(e) or e <= 0:
        goals = [7500.0, 10000.0, 12500.0]
        return goals, 10000.0

    raw = [e * cfg.mults[0], e * cfg.mults[1], e * cfg.mults[2]]

    raw[1] = max(raw[1], float(cfg.min_main_goal_usd))
    raw[0] = min(raw[0], raw[1] - cfg.min_goal_gap_usd)
    raw[2] = max(raw[2], raw[1] + cfg.min_goal_gap_usd)

    goals = [_round_to_step(x, cfg.round_to) for x in raw]
    goals = sorted([float(g) for g in goals])

    return goals, float(goals[1])


def maybe_rebuild_ledger_for_dt(
    *,
    dt_str: str,
    start: str,
    prices_mode: str,
    account_id: str,
    env: Optional[str],
    confirm_prod_write: bool,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "alpha_edge.operations.rebuild_ledger",
        "--account-id",
        account_id,
        "--start",
        start,
        "--end",
        dt_str,
        "--as-of",
        dt_str,
        "--prices-mode",
        prices_mode,
    ]

    if env:
        cmd.extend(["--env", env])
    if confirm_prod_write:
        cmd.append("--confirm-prod-write")

    print("[ledger] " + " ".join(cmd))
    subprocess.check_call(cmd)


def load_equity_asof(s3, *, bucket: str, account_root: str, dt_str: str) -> Optional[float]:
    payload = s3_load_latest_json_asof(
        s3,
        bucket=bucket,
        root_prefix=account_root,
        table="ledger/pnl",
        as_of=dt_str,
    ) or {}

    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        summary = payload["summary"]
    else:
        summary = payload if isinstance(payload, dict) else {}

    for k in ("equity_usd", "equity", "total_equity_usd", "total_equity"):
        v = summary.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass

    return None


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backfill daily_reports for historical dates.")

    ap.add_argument("--bucket", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--engine-root", default=None)
    ap.add_argument("--account-id", default="main")

    ap.add_argument("--start", default="auto", help="YYYY-MM-DD or 'auto' for first trade dt")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")

    ap.add_argument("--ledger-prices-mode", default="asof", choices=["asof", "latest"])
    ap.add_argument("--rebuild-ledger", action="store_true")

    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    return ap.parse_args()


# ----------------------------
# Main
# ----------------------------
def _main_impl(args: argparse.Namespace) -> None:

    cfg = load_runtime_config(args.env)

    bucket = str(args.bucket or cfg_bucket(cfg))
    region = str(args.region or cfg_region(cfg))
    engine_root = str(args.engine_root or cfg_engine_root(cfg)).strip("/")
    account_id = str(args.account_id)

    if not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    account_root = engine_root
    s3 = s3_init(region)

    end_dt = parse_date(args.end)

    if str(args.start).lower() == "auto":
        trade_dts = list_trade_dts(s3, bucket=bucket, engine_root=engine_root)
        if not trade_dts:
            raise SystemExit("No trades found; cannot auto-start. Provide --start YYYY-MM-DD.")
        start_dt = trade_dts[0]
    else:
        start_dt = parse_date(args.start)

    rng = pd.date_range(start_dt, end_dt, freq="D")

    print("\n=== REPORTS BACKFILL ===")
    print(f"env:            {cfg_env(cfg)}")
    print(f"bucket:         {bucket}")
    print(f"region:         {region}")
    print(f"engine_root:    {engine_root}")
    print(f"account_id:     {account_id}")
    print(f"range:          {start_dt} -> {end_dt} ({len(rng)} days)")
    print(f"rebuild_ledger: {bool(args.rebuild_ledger)} prices_mode={args.ledger_prices_mode}")
    print(f"skip_existing:  {bool(args.skip_existing)}")
    print(f"dry_run:        {bool(args.dry_run)}")
    print("")

    failures = 0

    for day in rng:
        dt_str = day.strftime("%Y-%m-%d")
        report_key = f"{engine_root}/daily_reports/dt={dt_str}/report.json"

        if args.skip_existing and key_exists(s3, bucket=bucket, key=report_key):
            print(f"[skip] dt={dt_str} report exists -> s3://{bucket}/{report_key}")
            continue

        print(f"\n--- dt={dt_str} ---")

        if args.rebuild_ledger:
            if args.dry_run:
                print(
                    f"[DRY RUN] would rebuild ledger for dt={dt_str} "
                    f"(start={start_dt}, prices_mode={args.ledger_prices_mode})"
                )
            else:
                try:
                    maybe_rebuild_ledger_for_dt(
                        dt_str=dt_str,
                        start=start_dt,
                        prices_mode=str(args.ledger_prices_mode),
                        account_id=account_id,
                        env=args.env,
                        confirm_prod_write=bool(args.confirm_prod_write),
                    )
                except Exception as e:
                    failures += 1
                    print(f"[ERROR] ledger rebuild failed dt={dt_str}: {type(e).__name__}: {e}")
                    if args.stop_on_error:
                        raise
                    continue

        eq = load_equity_asof(s3, bucket=bucket, account_root=account_root, dt_str=dt_str)
        if eq is None:
            failures += 1
            print(f"[ERROR] cannot load equity as-of dt={dt_str} from ledger/pnl; skipping")
            if args.stop_on_error:
                raise RuntimeError(f"Missing equity for dt={dt_str}")
            continue

        goals, main_goal = build_goals_from_equity(
            eq,
            GoalLadderCfg(mults=(1.20, 1.40, 1.60), round_to=50.0),
        )

        if args.dry_run:
            print(
                f"[DRY RUN] would run_daily_cycle_asof("
                f"as_of={dt_str}, backtest_run_id='backfill', equity_override={eq:.2f})"
            )
            print(f"[DRY RUN] goals={goals} main_goal={main_goal}")
            print(f"[DRY RUN] expected output -> s3://{bucket}/{report_key}")
            continue

        try:
            run_daily_cycle_asof(
                as_of=dt_str,
                backtest_run_id="backfill",
                write_outputs=True,
                update_latest=False,
                equity_override=float(eq),
                goals_override=list(goals),
                main_goal_override=float(main_goal),
            )
        except Exception as e:
            failures += 1
            print(f"[ERROR] report build failed dt={dt_str}: {type(e).__name__}: {e}")
            if args.stop_on_error:
                raise
            continue

        print(f"[OK] report -> s3://{bucket}/{report_key}")

    print("\n=== DONE ===")
    if failures:
        print(f"[WARN] failures={failures}")
        raise SystemExit(2)

    print("[OK] all done")


if __name__ == "__main__":
    main()