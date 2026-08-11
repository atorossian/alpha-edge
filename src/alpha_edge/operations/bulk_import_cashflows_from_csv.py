from __future__ import annotations

import argparse
import pandas as pd

from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.operations.record_cashflow import record_cashflow


def _is_blank(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x)) or str(x).strip() == ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk import cashflows from a CSV using record_cashflow().")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--env", default="prod", choices=["dev", "staging", "prod"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-prod-write", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_runtime_config(args.env)
    if not args.dry_run:
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    df = pd.read_csv(args.csv)
    required = {"as_of", "type", "amount", "currency", "ts_utc", "cashflow_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if args.limit is not None:
        df = df.head(int(args.limit))

    ok = 0
    failed = 0

    for i, row in df.iterrows():
        try:
            print(f"\n--- row {i + 1}/{len(df)} cashflow_id={row['cashflow_id']} ---")
            record_cashflow(
                cfg=cfg,
                as_of=str(row["as_of"]),
                type=str(row["type"]),
                amount=float(row["amount"]),
                currency=str(row["currency"]),
                account_id=str(row.get("account_id") or "main"),
                ts_utc=str(row["ts_utc"]),
                cashflow_id=str(row["cashflow_id"]),
                note=(None if _is_blank(row.get("note")) else str(row.get("note"))),
                dry_run=bool(args.dry_run),
                input_args={"source_csv": args.csv, "row": int(i)},
                reason="broker_cash_statement_20260809_full_cashflow_backfill",
            )
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAILED] row={i + 1} cashflow_id={row.get('cashflow_id')} error={type(e).__name__}: {e}")

    print("\n=== BULK CASHFLOW IMPORT SUMMARY ===")
    print(f"rows:   {len(df)}")
    print(f"ok:     {ok}")
    print(f"failed: {failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
