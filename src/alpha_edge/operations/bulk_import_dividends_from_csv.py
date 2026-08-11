from __future__ import annotations

import argparse
import math
from typing import Any

import pandas as pd

from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.operations.record_dividend import record_dividend


def _is_blank(x: Any) -> bool:
    if x is None:
        return True
    try:
        if isinstance(x, float) and math.isnan(x):
            return True
    except Exception:
        pass
    s = str(x).strip()
    return s == "" or s.lower() in {"none", "nan", "nat"}


def _maybe_float(x: Any) -> float | None:
    if _is_blank(x):
        return None
    return float(x)


def _maybe_str(x: Any) -> str | None:
    if _is_blank(x):
        return None
    return str(x).strip()


def _parse_date(x: Any) -> str:
    return pd.to_datetime(str(x)).date().isoformat()


def _parse_ts_utc(x: Any) -> str:
    # Compatible with older pandas versions where pd.Timestamp(..., utc=True)
    # is not supported.
    return pd.to_datetime(str(x), utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk import dividend records from CSV through record_dividend().")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--env", default="prod", choices=["dev", "staging", "prod"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-prod-write", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--universe-path", default=None)
    ap.add_argument("--universe-key", default=None)
    ap.add_argument("--strict-universe", action="store_true")
    ap.add_argument("--strict-math", action="store_true")
    ap.add_argument("--math-tol", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_runtime_config(args.env)
    if not args.dry_run:
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    df = pd.read_csv(args.csv)

    required = {"dividend_id", "as_of", "ts_utc", "ticker", "amount", "currency"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    n = len(df) if args.limit is None else min(int(args.limit), len(df))

    print("=== BULK IMPORT DIVIDENDS ===")
    print("env:", cfg.env)
    print("bucket:", cfg.bucket)
    print("engine_root:", cfg.engine_root)
    print("csv:", args.csv)
    print("rows_total:", len(df))
    print("rows_to_process:", n)
    print("dry_run:", bool(args.dry_run))
    print("universe_path:", args.universe_path)
    print("strict_universe:", bool(args.strict_universe))
    print("")

    ok = 0
    failed = 0

    for i in range(n):
        row = df.iloc[i]
        try:
            dividend_id = str(row["dividend_id"]).strip()
            as_of = _parse_date(row["as_of"])
            ts_utc = _parse_ts_utc(row["ts_utc"])
            ticker = str(row["ticker"]).upper().strip()
            currency = str(row["currency"]).upper().strip()
            amount = float(row["amount"])

            source_section = _maybe_str(row.get("source_section", None))
            dividend_type = _maybe_str(row.get("dividend_type", None))
            position_type = _maybe_str(row.get("position_type", None))
            raw_line = _maybe_str(row.get("raw_line", None))
            base_note = _maybe_str(row.get("note", None)) or "broker_statement_20260809_dividend_full_reset"

            note_parts = [base_note]
            if source_section:
                note_parts.append(f"source_section={source_section}")
            if dividend_type:
                note_parts.append(f"dividend_type={dividend_type}")
            if position_type:
                note_parts.append(f"position_type={position_type}")
            if raw_line:
                note_parts.append(f"raw_line={raw_line}")
            note = " | ".join(note_parts)

            print(f"[ROW {i}] {dividend_id} {as_of} {ticker} {amount} {currency}")

            record_dividend(
                cfg=cfg,
                as_of=as_of,
                ticker=ticker,
                asset_id=_maybe_str(row.get("asset_id", None)),
                amount=amount,
                currency=currency,
                account_id="main",
                ts_utc=ts_utc,
                dividend_id=dividend_id,
                note=note,
                shares_held=_maybe_float(row.get("quantity", None)),
                dividend_per_share=_maybe_float(row.get("dividend_per_share", None)),
                source="broker_statement_20260809",
                strict_math=bool(args.strict_math),
                math_tol=float(args.math_tol),
                universe_path=args.universe_path,
                universe_key=args.universe_key,
                strict_universe=bool(args.strict_universe),
                dry_run=bool(args.dry_run),
                reason="dividend_full_reset_20260809",
                input_args=vars(args),
            )
            ok += 1

        except Exception as e:
            failed += 1
            print("[FAILED]", i, type(e).__name__, str(e))

    print("")
    print("ok:", ok)
    print("failed:", failed)
    if args.dry_run:
        print("[DRY RUN] no S3 objects were modified.")


if __name__ == "__main__":
    main()
