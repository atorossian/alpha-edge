# run_reports_backfill.py
from __future__ import annotations

import argparse
import subprocess
from typing import Optional, List

import pandas as pd
import math
from dataclasses import dataclass
from typing import Tuple, List

from alpha_edge.core.data_loader import (
    s3_init,
    s3_list_keys,
    s3_load_latest_json_asof,
)
from alpha_edge.jobs.run_daily_report import run_daily_cycle_asof  # adjust if your path differs

BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"


# ----------------------------
# Helpers
# ----------------------------
def parse_date(s: str) -> str:
    return pd.Timestamp(s).strftime("%Y-%m-%d")


def engine_prefix(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def key_exists(s3, *, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def list_trade_dts(s3, *, bucket: str, engine_root: str) -> List[str]:
    """
    Reads dt=YYYY-MM-DD from trade files under:
      <engine_root>/trades/dt=.../trade_*.json
    """
    prefix = f"{engine_root.strip('/')}/trades/"
    keys = s3_list_keys(s3, bucket=bucket, prefix=prefix)
    dts = set()

    for k in keys:
        parts = str(k).split("/")
        for p in parts:
            # cheap parse "dt=YYYY-MM-DD"
            if p.startswith("dt=") and len(p) == len("dt=YYYY-MM-DD"):
                dts.add(p[len("dt="):])
                break

    return sorted(dts)

@dataclass(frozen=True)
class GoalLadderCfg:
    # equity multipliers for the 3 goals
    mults: Tuple[float, float, float] = (1.20, 1.40, 1.60)

    # rounding step to make goals "pretty"
    round_to: float = 50.0  # set to 1.0 to disable

    # hard floors to prevent tiny-goal weirdness
    min_main_goal_usd: float = 500.0
    min_goal_gap_usd: float = 100.0  # ensure goal2-goal1 and goal3-goal2 not too small

def _round_to_step(x: float, step: float) -> float:
    if step is None or step <= 0:
        return float(x)
    return float(step) * round(float(x) / float(step))

def build_goals_from_equity(equity: float, cfg: GoalLadderCfg) -> Tuple[List[float], float]:
    e = float(equity)
    if not math.isfinite(e) or e <= 0:
        # fallback: keep something safe; but ideally equity is always valid
        goals = [7500.0, 10000.0, 12500.0]
        return goals, 10000.0

    raw = [e * cfg.mults[0], e * cfg.mults[1], e * cfg.mults[2]]

    # main goal floor
    raw[1] = max(raw[1], float(cfg.min_main_goal_usd))

    # enforce spacing so the ladder isn't degenerate for small equity
    raw[0] = min(raw[0], raw[1] - cfg.min_goal_gap_usd)
    raw[2] = max(raw[2], raw[1] + cfg.min_goal_gap_usd)

    # rounding
    goals = [_round_to_step(x, cfg.round_to) for x in raw]
    goals = sorted([float(g) for g in goals])

    main_goal = float(goals[1])
    return goals, main_goal

def maybe_rebuild_ledger_for_dt(
    *,
    dt_str: str,
    start: str,
    prices_mode: str,
) -> None:
    """
    Calls the ledger rebuild via CLI for consistency.
    IMPORTANT: This assumes there is a module entrypoint:
      python -m alpha_edge.operations.rebuild_ledger
    Adjust module path if yours is different.
    """
    cmd = [
        "python",
        "-m",
        "alpha_edge.operations.rebuild_ledger",
        "--start",
        start,
        "--end",
        dt_str,
        "--as-of",
        dt_str,
        "--prices-mode",
        prices_mode,
    ]

    print("[ledger] " + " ".join(cmd))
    subprocess.check_call(cmd)


def load_equity_asof(s3, *, bucket: str, account_root: str, dt_str: str) -> Optional[float]:
    """
    Reads equity from ledger/pnl as-of dt_str.

    Your rebuild_ledger.py writes:
      {
        "as_of": "...",
        "method": "...",
        "summary": {
           "equity_usd": ...,
           ...
        }
      }

    So equity is under payload["summary"] (not top-level).
    """
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
        # fallback for older shapes
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

    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--engine-root", default=ENGINE_ROOT)
    ap.add_argument("--account-id", default="main")

    ap.add_argument("--start", default="auto", help="YYYY-MM-DD or 'auto' (first trade dt)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")

    ap.add_argument("--ledger-prices-mode", default="asof", choices=["asof", "latest"])
    ap.add_argument("--rebuild-ledger", action="store_true", help="Rebuild ledger for each dt before reporting.")

    ap.add_argument("--skip-existing", action="store_true", help="Skip dt if daily report already exists.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop immediately on first failure.")

    return ap.parse_args()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()

    bucket = str(args.bucket)
    region = str(args.region)
    engine_root = str(args.engine_root).strip("/")
    account_id = str(args.account_id)

    # NOTE: you currently use root_prefix=engine_root in your run_daily_report backtest mode.
    # If you later move to per-account roots, change this here.
    account_root = engine_root

    s3 = s3_init(region)

    end_dt = parse_date(args.end)

    # Determine start dt
    if str(args.start).lower() == "auto":
        trade_dts = list_trade_dts(s3, bucket=bucket, engine_root=engine_root)
        if not trade_dts:
            raise SystemExit("No trades found; cannot auto-start. Provide --start YYYY-MM-DD.")
        start_dt = trade_dts[0]
    else:
        start_dt = parse_date(args.start)

    # Calendar daily range (inclusive)
    rng = pd.date_range(start_dt, end_dt, freq="D")

    print("=== REPORTS BACKFILL ===")
    print(f"bucket:        {bucket}")
    print(f"engine_root:   {engine_root}")
    print(f"account_id:    {account_id}")
    print(f"range:         {start_dt} -> {end_dt} ({len(rng)} days)")
    print(f"rebuild_ledger:{bool(args.rebuild_ledger)} prices_mode={args.ledger_prices_mode}")
    print(f"skip_existing: {bool(args.skip_existing)}")
    print(f"dry_run:       {bool(args.dry_run)}")
    print("")

    failures = 0

    for dt in rng:
        dt_str = dt.strftime("%Y-%m-%d")

        report_key = f"{engine_root}/daily_reports/dt={dt_str}/report.json"
        if args.skip_existing and key_exists(s3, bucket=bucket, key=report_key):
            print(f"[skip] dt={dt_str} report exists -> s3://{bucket}/{report_key}")
            continue

        print(f"\n--- dt={dt_str} ---")

        # Optionally rebuild ledger up to this dt
        if args.rebuild_ledger:
            if args.dry_run:
                print(f"[DRY RUN] would rebuild ledger for dt={dt_str} (start={start_dt}, prices_mode={args.ledger_prices_mode})")
            else:
                try:
                    maybe_rebuild_ledger_for_dt(
                        dt_str=dt_str,
                        start=start_dt,
                        prices_mode=str(args.ledger_prices_mode),
                    )
                except Exception as e:
                    failures += 1
                    print(f"[ERROR] ledger rebuild failed dt={dt_str}: {type(e).__name__}: {e}")
                    if args.stop_on_error:
                        raise
                    continue

        # Load equity as-of dt
        eq = load_equity_asof(s3, bucket=bucket, account_root=account_root, dt_str=dt_str)
        if eq is None:
            failures += 1
            print(f"[ERROR] cannot load equity as-of dt={dt_str} from ledger/pnl; skipping")
            if args.stop_on_error:
                raise RuntimeError(f"Missing equity for dt={dt_str}")
            continue
        
        goals, main_goal = build_goals_from_equity(eq, GoalLadderCfg(mults=(1.20, 1.40, 1.60), round_to=50.0))

        if args.dry_run:
            print(f"[DRY RUN] would run_daily_cycle_asof(as_of={dt_str}, backtest_run_id='backfill', equity_override={eq:.2f})")
            print(f"[DRY RUN] expected output -> s3://{bucket}/{report_key}")
            continue

        # Build report (backtest mode, no latest pointer updates)
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
    else:
        print("[OK] all done")


if __name__ == "__main__":
    main()