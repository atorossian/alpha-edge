from __future__ import annotations

import argparse
import json
import uuid
import io
from dataclasses import asdict
from typing import Optional, Literal, Tuple

import boto3
import pandas as pd

from alpha_edge.core.schemas import Trade  # make sure this import path matches your project


BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"
TRADES_TABLE = "trades"

DEFAULT_UNIVERSE_KEY = f"{ENGINE_ROOT}/universe/universe.csv"  # only used if --universe-key is provided


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


def s3_get_bytes(s3, *, bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


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
    unit_map = {
        "share": "shares",
        "shares": "shares",
        "contract": "contracts",
        "contracts": "contracts",
        "coin": "coins",
        "coins": "coins",
        "ounce": "ounces",
        "ounces": "ounces",
    }
    return unit_map.get(s, s)


# ----------------------------
# Universe lookup (asset_id resolver)
# ----------------------------
def _load_universe_df(
    *,
    universe_path: Optional[str],
    universe_key: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    """
    Returns: (df, ref_string)
    """
    if universe_path:
        df = pd.read_csv(universe_path)
        return df, f"file://{universe_path}"

    if universe_key:
        s3 = s3_client(REGION)
        raw = s3_get_bytes(s3, bucket=BUCKET, key=universe_key)
        df = pd.read_csv(io.BytesIO(raw))
        return df, f"s3://{BUCKET}/{universe_key}"

    raise RuntimeError("Universe not provided (need --universe-path or --universe-key) to resolve asset_id.")


def _resolve_asset_id_from_universe(
    *,
    broker_ticker: str,
    universe_path: Optional[str],
    universe_key: Optional[str],
) -> Tuple[str, str]:
    """
    Resolve asset_id by broker_ticker (preferred) or ticker (fallback).

    Returns: (asset_id, universe_ref)
    Raises:
      - RuntimeError if not found or ambiguous
    """
    df, ref = _load_universe_df(universe_path=universe_path, universe_key=universe_key)

    required = {"asset_id", "broker_ticker"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Universe CSV missing required columns {missing} (ref={ref})")

    bt = str(broker_ticker).upper().strip()

    # normalize relevant cols
    df = df.copy()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["broker_ticker"] = df["broker_ticker"].astype(str).str.upper().str.strip()
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # 1) match by broker_ticker
    m = df[df["broker_ticker"] == bt]
    if len(m) == 1:
        aid = str(m.iloc[0]["asset_id"]).strip()
        if not aid:
            raise RuntimeError(f"Universe match found but asset_id empty (broker_ticker={bt}, ref={ref})")
        return aid, ref

    # 2) fallback to ticker if present (some universes keep both identical; some not)
    if len(m) == 0 and "ticker" in df.columns:
        m2 = df[df["ticker"] == bt]
        if len(m2) == 1:
            aid = str(m2.iloc[0]["asset_id"]).strip()
            if not aid:
                raise RuntimeError(f"Universe match found but asset_id empty (ticker={bt}, ref={ref})")
            return aid, ref
        if len(m2) > 1:
            sample = m2[["asset_id", "broker_ticker", "ticker"]].head(10).to_dict("records")
            raise RuntimeError(
                f"Ambiguous universe mapping for ticker={bt}: {len(m2)} rows match (ref={ref}). "
                f"Sample={sample}"
            )

    if len(m) > 1:
        sample_cols = [c for c in ["asset_id", "broker_ticker", "ticker", "yahoo_ticker", "name"] if c in df.columns]
        sample = m[sample_cols].head(10).to_dict("records")
        raise RuntimeError(
            f"Ambiguous universe mapping for broker_ticker={bt}: {len(m)} rows match (ref={ref}). "
            f"Sample={sample}"
        )

    raise RuntimeError(f"No universe mapping for broker_ticker/ticker={bt} (ref={ref}).")


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
    asset_id: Optional[str] = None,

    # universe resolver inputs (optional; used if asset_id not provided)
    universe_path: Optional[str] = None,
    universe_key: Optional[str] = None,
    strict_universe: bool = False,

    # metadata
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
    ticker_norm = str(ticker).upper().strip()
    currency_norm = str(currency).upper().strip() or "USD"

    qty = _validate_positive("quantity", quantity)
    px = _validate_positive("price", price)

    action_tag_norm = _normalize_action_tag(action_tag)
    unit_norm = _normalize_quantity_unit(quantity_unit)

    value_norm = None if value is None else float(value)
    reported_pnl_norm = None if reported_pnl is None else float(reported_pnl)

    # HARD RULES for derivatives:
    if unit_norm == "contracts":
        if action_tag_norm is None:
            raise ValueError("contracts trades require action_tag (open/close/add/reduce)")
        if value_norm is None:
            raise ValueError("contracts trades require value (notional)")

    # asset_id resolution
    universe_ref = None
    asset_id_norm = None if asset_id is None else str(asset_id).strip()
    if not asset_id_norm:
        if universe_path or universe_key:
            asset_id_norm, universe_ref = _resolve_asset_id_from_universe(
                broker_ticker=ticker_norm,
                universe_path=universe_path,
                universe_key=universe_key,
            )
        elif strict_universe:
            raise ValueError("asset_id not provided and --strict-universe requested but no universe provided.")
        else:
            asset_id_norm = None

    if strict_universe and not asset_id_norm:
        raise ValueError(
            "asset_id could not be resolved (strict mode). Provide --asset-id or --universe-path/--universe-key."
        )

    if ts_utc is None:
        ts_utc = _iso_utc_now()

    if trade_id is None:
        trade_id = f"{as_of_norm.replace('-', '')}-{uuid.uuid4().hex[:10]}"

    trade = Trade(
        trade_id=str(trade_id),
        as_of=as_of_norm,
        ts_utc=str(ts_utc),
        asset_id=asset_id_norm,  # IMPORTANT
        ticker=ticker_norm,
        side=side_norm,
        quantity=float(qty),
        price=float(px),
        currency=currency_norm,
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
    print(f"asset_id:  {trade.asset_id}")
    print(f"{trade.side} {trade.ticker} qty={trade.quantity} px={trade.price} {trade.currency}")
    if universe_ref:
        print(f"universe:  {universe_ref}")
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
    ap.add_argument("--ticker", required=True, help="Broker ticker, e.g. NVDA or BTC-USD or EUR-USD.")
    ap.add_argument("--side", required=True, choices=["BUY", "SELL", "buy", "sell"], help="BUY or SELL.")
    ap.add_argument("--quantity", required=True, type=float, help="Quantity (>0).")
    ap.add_argument("--price", required=True, type=float, help="Trade price (>0).")
    ap.add_argument("--currency", default="USD", help="Currency code, default USD.")

    # asset id
    ap.add_argument("--asset-id", default=None, help="Asset ID (recommended). If omitted, may resolve via universe.")

    # universe resolver (local preferred)
    ap.add_argument("--universe-path", default=None, help="Local universe.csv path (to resolve asset_id).")
    ap.add_argument(
        "--universe-key",
        default=None,
        help=f"S3 key for universe.csv (optional). Example: {DEFAULT_UNIVERSE_KEY}",
    )
    ap.add_argument(
        "--strict-universe",
        action="store_true",
        help="Fail if asset_id cannot be resolved (requires --asset-id or universe lookup).",
    )

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
        asset_id=args.asset_id,
        universe_path=args.universe_path,
        universe_key=args.universe_key,
        strict_universe=bool(args.strict_universe),
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
