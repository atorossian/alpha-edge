# bulk_import_trades.py
from __future__ import annotations

import argparse
from typing import Optional
import csv
from pathlib import Path
import time
import hashlib

import pandas as pd

from alpha_edge.operations.record_trade import record_trade


def _is_blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in ("none", "nan")


def _maybe_str(x) -> Optional[str]:
    return None if _is_blank(x) else str(x)


def normalize_ticker_like_yahoo(raw: str) -> str:
    """
    Normalizes Quantfury crypto pair tickers to a Yahoo-like format using hyphen:
      "BTC/USDT" -> "BTC-USDT"
      "ETH/USD"  -> "ETH-USD"

    Leaves equities as-is:
      "NVDA" -> "NVDA"
      "AI.PA" -> "AI.PA"
    """
    s = str(raw).strip().upper()

    if "/" in s:
        base, quote = [x.strip() for x in s.split("/", 1)]
        if base and quote:
            return f"{base}-{quote}"

    return s


def _normalize_side(x: str) -> str:
    s = str(x).upper().strip()
    if s not in ("BUY", "SELL"):
        raise ValueError(f"Invalid side={x!r}")
    return s


def _normalize_as_of(x: str) -> str:
    return pd.Timestamp(x).date().strftime("%Y-%m-%d")


def make_trade_id_deterministic(
    *,
    as_of: str,
    ts_utc: Optional[str],
    ticker: str,
    side: str,
    quantity: float,
    price: float,
    currency: str,
) -> str:
    """
    Deterministic trade_id so bulk re-runs are idempotent even if CSV doesn't include trade_id.

    Important: Use stable fields. If your CSV has an exchange-provided trade id,
    you should prefer that instead.
    """
    # ts_utc may be missing; include it if present, else empty string
    ts = "" if ts_utc is None else str(ts_utc).strip()

    raw = (
        f"{as_of}|{ts}|{ticker}|{side}|"
        f"{quantity:.10f}|{price:.10f}|{currency}"
    )
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{as_of.replace('-', '')}-{h}"


def import_csv(
    *,
    csv_path: str,
    dry_run: bool = False,
    limit: int | None = None,
    out_dir: str = "bulk_logs",
    print_every: int = 50,
    max_retries: int = 3,
    retry_sleep_sec: float = 1.0,
) -> None:
    df = pd.read_csv(csv_path)

    required = {"as_of", "ticker", "side", "quantity", "price"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # --- Ensure optional columns exist ---
    if "currency" not in df.columns:
        df["currency"] = "USD"
    if "ts_utc" not in df.columns:
        df["ts_utc"] = None
    if "note" not in df.columns:
        df["note"] = None
    if "choice_id" not in df.columns:
        df["choice_id"] = None
    if "portfolio_run_id" not in df.columns:
        df["portfolio_run_id"] = None
    if "action_tag" not in df.columns:
        df["action_tag"] = None
    if "quantity_unit" not in df.columns:
        df["quantity_unit"] = None
    if "value" not in df.columns:
        df["value"] = None
    if "reported_pnl" not in df.columns:
        df["reported_pnl"] = None


    # IMPORTANT: idempotency
    # If CSV has trade_id -> use it
    # Else -> deterministically generate it
    if "trade_id" not in df.columns:
        df["trade_id"] = None

    # --- Normalize basics ---
    df["as_of"] = df["as_of"].apply(_normalize_as_of)
    df["ticker"] = df["ticker"].astype(str).apply(normalize_ticker_like_yahoo)
    df["side"] = df["side"].apply(_normalize_side)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["currency"] = df["currency"].astype(str).str.upper().str.strip().replace("", "USD")
    df["action_tag"] = df["action_tag"].apply(_maybe_str)
    df["quantity_unit"] = df["quantity_unit"].apply(_maybe_str)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["reported_pnl"] = pd.to_numeric(df["reported_pnl"], errors="coerce")


    # normalize ts_utc and optional strings
    df["ts_utc"] = df["ts_utc"].apply(_maybe_str)
    df["note"] = df["note"].apply(_maybe_str)
    df["choice_id"] = df["choice_id"].apply(_maybe_str)
    df["portfolio_run_id"] = df["portfolio_run_id"].apply(_maybe_str)

    # --- Fill trade_id deterministically if missing/blank ---
    def _fill_trade_id(row) -> str:
        if not _is_blank(row.get("trade_id")):
            return str(row["trade_id"])
        return make_trade_id_deterministic(
            as_of=str(row["as_of"]),
            ts_utc=row["ts_utc"],
            ticker=str(row["ticker"]),
            side=str(row["side"]),
            quantity=float(row["quantity"]),
            price=float(row["price"]),
            currency=str(row["currency"]) or "USD",
        )

    df["trade_id"] = df.apply(_fill_trade_id, axis=1)

    # --- sanity: duplicates inside CSV ---
    dup = df["trade_id"].duplicated().sum()
    if dup > 0:
        # Don't proceed silently; duplicated trade_id means overwrites inside import.
        raise ValueError(
            f"CSV contains {dup} duplicate trade_id values. "
            "Fix upstream ID generation or include a stable trade id from broker/export."
        )

    n = len(df) if limit is None else min(len(df), int(limit))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    errors_path = Path(out_dir) / "failed_rows.csv"
    ok_path = Path(out_dir) / "uploaded_rows.csv"

    errors_f = open(errors_path, "w", newline="", encoding="utf-8")
    ok_f = open(ok_path, "w", newline="", encoding="utf-8")

    err_fields = [
        "row_idx", "trade_id", "as_of", "ticker", "side", "quantity", "price",
        "currency", "ts_utc",
        "action_tag", "quantity_unit", "value", "reported_pnl",
        "choice_id", "portfolio_run_id", "note", "error",
    ]
    ok_fields = [
        "row_idx", "trade_id", "as_of", "ticker", "side", "quantity", "price",
        "currency", "ts_utc",
        "action_tag", "quantity_unit", "value", "reported_pnl",
        "choice_id", "portfolio_run_id", "note",
    ]


    err_writer = csv.DictWriter(errors_f, fieldnames=err_fields)
    ok_writer = csv.DictWriter(ok_f, fieldnames=ok_fields)
    err_writer.writeheader()
    ok_writer.writeheader()

    ok = 0
    failed = 0

    print(f"[bulk] csv={csv_path} rows={len(df)} importing={n} dry_run={dry_run}")
    print(f"[bulk] writing logs to: {errors_path} and {ok_path}")
    print("[bulk] trade_id is guaranteed (CSV-provided or deterministic) -> import is idempotent (safe to re-run).")

    try:
        for i in range(n):
            row = df.iloc[i]

            payload = dict(
                row_idx=i,
                trade_id=str(row["trade_id"]),
                as_of=str(row["as_of"]),
                ticker=str(row["ticker"]),
                side=str(row["side"]),
                quantity=float(row["quantity"]) if pd.notna(row["quantity"]) else None,
                price=float(row["price"]) if pd.notna(row["price"]) else None,
                currency=str(row["currency"]).upper().strip() if pd.notna(row["currency"]) else "USD",
                ts_utc=row["ts_utc"],
                choice_id=row["choice_id"],
                portfolio_run_id=row["portfolio_run_id"],
                note=row["note"],
                action_tag=row["action_tag"],
                quantity_unit=row["quantity_unit"],
                value=float(row["value"]) if pd.notna(row["value"]) else None,
                reported_pnl=float(row["reported_pnl"]) if pd.notna(row["reported_pnl"]) else None,
            )

            try:
                # basic validation
                if payload["side"] not in ("BUY", "SELL"):
                    raise ValueError(f"Invalid side={payload['side']}")
                if payload["quantity"] is None or payload["quantity"] <= 0:
                    raise ValueError("quantity must be > 0")
                if payload["price"] is None or payload["price"] <= 0:
                    raise ValueError("price must be > 0")

                # retry loop for transient issues
                last_exc: Exception | None = None
                for attempt in range(max_retries + 1):
                    try:
                        record_trade(
                            as_of=payload["as_of"],
                            ticker=payload["ticker"],
                            side=payload["side"],
                            quantity=payload["quantity"],
                            price=payload["price"],
                            currency=payload["currency"],
                            trade_id=payload["trade_id"],
                            ts_utc=payload["ts_utc"],
                            action_tag=payload["action_tag"],
                            quantity_unit=payload["quantity_unit"],
                            value=payload["value"],
                            reported_pnl=payload["reported_pnl"],
                            choice_id=payload["choice_id"],
                            portfolio_run_id=payload["portfolio_run_id"],
                            note=payload["note"],
                            dry_run=dry_run,
                        )

                        last_exc = None
                        break
                    except Exception as e:
                        last_exc = e
                        if attempt < max_retries:
                            time.sleep(retry_sleep_sec * (attempt + 1))

                if last_exc is not None:
                    raise last_exc

                ok_writer.writerow({k: payload.get(k) for k in ok_fields})
                ok_f.flush()
                ok += 1

            except Exception as e:
                failed += 1
                payload["error"] = f"{type(e).__name__}: {e}"
                err_writer.writerow({k: payload.get(k) for k in err_fields})
                errors_f.flush()

            if (i + 1) % print_every == 0:
                print(f"[bulk] progress {i+1}/{n} ok={ok} failed={failed}")

    finally:
        errors_f.close()
        ok_f.close()

    print(f"[bulk] done ok={ok} failed={failed}")
    print(f"[bulk] failures saved to {errors_path}")


def main():
    ap = argparse.ArgumentParser(description="Bulk import trades from CSV using record_trade().")
    ap.add_argument("--csv", required=True, help="Path to CSV file.")
    ap.add_argument("--dry-run", action="store_true", help="Validate and print S3 keys but do not write.")
    ap.add_argument("--limit", type=int, default=None, help="Import only first N rows.")
    ap.add_argument("--out-dir", default="bulk_logs", help="Output directory for logs.")
    ap.add_argument("--print-every", type=int, default=50, help="Progress print frequency.")
    ap.add_argument("--max-retries", type=int, default=3, help="Retries per row on failure.")
    ap.add_argument("--retry-sleep-sec", type=float, default=1.0, help="Base sleep seconds between retries.")
    args = ap.parse_args()

    import_csv(
        csv_path=args.csv,
        dry_run=bool(args.dry_run),
        limit=args.limit,
        out_dir=str(args.out_dir),
        print_every=int(args.print_every),
        max_retries=int(args.max_retries),
        retry_sleep_sec=float(args.retry_sleep_sec),
    )


if __name__ == "__main__":
    main()
