from __future__ import annotations

import argparse
import datetime as dt
import json
import uuid
from dataclasses import asdict
from typing import Optional, Literal

import boto3
import pandas as pd

from alpha_edge.core.schemas import Trade  # make sure this import path matches your project


BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"
TRADES_TABLE = "trades"


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(region: str = REGION):
    return boto3.client("s3", region_name=region)


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def dt_key(table: str, dt_str: str, filename: str) -> str:
    return engine_key(table, f"dt={dt_str}", filename)


# ----------------------------
# Validation
# ----------------------------
def _parse_date(s: str) -> str:
    # normalize to YYYY-MM-DD
    d = pd.Timestamp(s).date()
    return d.strftime("%Y-%m-%d")


def _iso_utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _validate_side(s: str) -> Literal["BUY", "SELL"]:
    s = str(s).upper().strip()
    if s not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    return s  # type: ignore[return-value]


def _validate_positive(name: str, x: float) -> float:
    x = float(x)
    if not (x > 0.0):
        raise ValueError(f"{name} must be > 0")
    return x

def _normalize_action_tag(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s == "":
        return None
    allowed = {"open", "close", "add", "reduce"}
    if s not in allowed:
        raise ValueError(f"action_tag must be one of {sorted(allowed)} (got {x!r})")
    return s

def _normalize_quantity_unit(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s == "":
        return None
    # normalize singular/plural to ONE canonical form
    unit_map = {
        "share": "shares",
        "shares": "shares",
        "contract": "contracts",
        "contracts": "contracts",
        "coin": "coins",
        "coins": "coins",
    }
    return unit_map.get(s, s)


# ----------------------------
# Core
# ----------------------------
def record_trade(
    *,
    as_of: str,
    ticker: str,
    side: str,
    quantity: float,
    price: float,
    currency: str = "USD",
    trade_id: Optional[str] = None,
    ts_utc: Optional[str] = None,

    # NEW
    action_tag: Optional[str] = None,
    quantity_unit: Optional[str] = None,
    value: Optional[float] = None,
    reported_pnl: Optional[float] = None,

    choice_id: Optional[str] = None,
    portfolio_run_id: Optional[str] = None,
    note: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    s3 = s3_client(REGION)

    as_of_norm = _parse_date(as_of)
    side_norm = _validate_side(side)
    ticker = str(ticker).upper().strip()
    currency = str(currency).upper().strip() or "USD"

    qty = _validate_positive("quantity", quantity)
    px = _validate_positive("price", price)

    action_tag_norm = _normalize_action_tag(action_tag)
    unit_norm = _normalize_quantity_unit(quantity_unit)

    value_norm = None
    if value is not None:
        value_norm = float(value)

    reported_pnl_norm = None
    if reported_pnl is not None:
        reported_pnl_norm = float(reported_pnl)

    # HARD RULES for derivatives:
    if unit_norm == "contracts":
        if action_tag_norm is None:
            raise ValueError("contracts trades require action_tag (open/close/add/reduce)")
        if value_norm is None:
            raise ValueError("contracts trades require value (notional)")

    if ts_utc is None:
        ts_utc = _iso_utc_now()

    if trade_id is None:
        # deterministic-ish id: date + short uuid
        trade_id = f"{as_of_norm.replace('-', '')}-{uuid.uuid4().hex[:10]}"

    trade = Trade(
        trade_id=str(trade_id),
        as_of=as_of_norm,
        ts_utc=str(ts_utc),
        ticker=ticker,
        side=side_norm,
        quantity=float(qty),
        price=float(px),
        currency=currency,
        choice_id=(str(choice_id) if choice_id else None),
        portfolio_run_id=(str(portfolio_run_id) if portfolio_run_id else None),
        note=(str(note) if note else None),
        action_tag=action_tag_norm,
        quantity_unit=unit_norm,
        value=value_norm,
        reported_pnl=reported_pnl_norm,

    )

    payload = asdict(trade)

    trade_key = dt_key(TRADES_TABLE, as_of_norm, f"trade_{trade.trade_id}.json")
    latest_key = engine_key(TRADES_TABLE, "latest.json")

    print("\n=== RECORD TRADE ===")
    print(f"as_of:     {trade.as_of}")
    print(f"trade_id:  {trade.trade_id}")
    print(f"ts_utc:    {trade.ts_utc}")
    print(f"{trade.side} {trade.ticker} qty={trade.quantity} px={trade.price} {trade.currency}")
    if trade.choice_id:
        print(f"choice_id: {trade.choice_id}")
    if trade.portfolio_run_id:
        print(f"run_id:    {trade.portfolio_run_id}")
    if trade.note:
        print(f"note:      {trade.note}")
    print("")

    if dry_run:
        print("[DRY RUN] Would write:")
        print(f"  s3://{BUCKET}/{trade_key}")
        print(f"  s3://{BUCKET}/{latest_key}")
        return

    s3_put_json(s3, bucket=BUCKET, key=trade_key, payload=payload)
    s3_put_json(s3, bucket=BUCKET, key=latest_key, payload=payload)

    print("[OK] Wrote trade:")
    print(f"  s3://{BUCKET}/{trade_key}")
    print("[OK] Updated latest:")
    print(f"  s3://{BUCKET}/{latest_key}")
    print("")


def parse_args():
    ap = argparse.ArgumentParser(description="Append exactly one trade to S3 (no inference, no portfolio logic).")

    ap.add_argument("--as-of", required=True, help="Trade date (YYYY-MM-DD).")
    ap.add_argument("--ticker", required=True, help="Ticker, e.g. NVDA.")
    ap.add_argument("--side", required=True, choices=["BUY", "SELL", "buy", "sell"], help="BUY or SELL.")
    ap.add_argument("--quantity", required=True, type=float, help="Quantity (shares). Must be > 0.")
    ap.add_argument("--price", required=True, type=float, help="Trade price. Must be > 0.")
    ap.add_argument("--currency", default="USD", help="Currency code, default USD.")

    # optional metadata
    ap.add_argument("--trade-id", default=None, help="If omitted, auto-generated.")
    ap.add_argument("--ts-utc", default=None, help="If omitted, uses current UTC ISO timestamp.")
    ap.add_argument("--choice-id", default=None, help="Link to portfolio choice if relevant.")
    ap.add_argument("--portfolio-run-id", default=None, help="Link to portfolio search run if relevant.")
    ap.add_argument("--note", default=None, help="Free text note.")
    ap.add_argument("--dry-run", action="store_true", help="Show S3 keys but do not write.")
    ap.add_argument("--action-tag", default=None, help="open/close/add/reduce (needed for contracts).")
    ap.add_argument("--quantity-unit", default=None, help="shares/contracts/coins (normalized).")
    ap.add_argument("--value", default=None, type=float, help="Notional value (required for contracts).")
    ap.add_argument("--reported-pnl", default=None, type=float, help="Broker reported PnL (optional).")


    return ap.parse_args()


def main():
    args = parse_args()
    record_trade(
        as_of=args.as_of,
        ticker=args.ticker,
        side=args.side,
        quantity=args.quantity,
        price=args.price,
        currency=args.currency,
        trade_id=args.trade_id,
        ts_utc=args.ts_utc,
        choice_id=args.choice_id,
        portfolio_run_id=args.portfolio_run_id,
        note=args.note,
        dry_run=bool(args.dry_run),
        action_tag=args.action_tag,
        quantity_unit=args.quantity_unit,
        value=args.value,
        reported_pnl=args.reported_pnl,

    )


if __name__ == "__main__":
    main()
