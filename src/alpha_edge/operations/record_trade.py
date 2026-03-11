# record_trade.py  (record + edit + migrate index)
from __future__ import annotations

import argparse
import json
import uuid
import io
from dataclasses import asdict
from typing import Optional, Literal, Tuple, Any, Dict

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


def s3_exists(s3, *, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def s3_copy(s3, *, bucket: str, src_key: str, dst_key: str) -> None:
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        ContentType="application/json",
        MetadataDirective="COPY",
    )


def s3_delete(s3, *, bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)


def s3_get_json_optional(s3, *, bucket: str, key: str) -> Optional[dict]:
    try:
        raw = s3_get_bytes(s3, bucket=bucket, key=key)
        obj = json.loads(raw.decode("utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ----------------------------
# Validation
# ----------------------------
def _parse_date(s: str) -> str:
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

    df = df.copy()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["broker_ticker"] = df["broker_ticker"].astype(str).str.upper().str.strip()
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    m = df[df["broker_ticker"] == bt]
    if len(m) == 1:
        aid = str(m.iloc[0]["asset_id"]).strip()
        if not aid:
            raise RuntimeError(f"Universe match found but asset_id empty (broker_ticker={bt}, ref={ref})")
        return aid, ref

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
# Trades index (trade_id -> S3 key)
# ----------------------------
TRADES_INDEX_KEY = engine_key(TRADES_TABLE, "index.json")
TRADES_AUDIT_PREFIX = engine_key("trades_audit")


def _load_trades_index(s3) -> dict:
    idx = s3_get_json_optional(s3, bucket=BUCKET, key=TRADES_INDEX_KEY)
    return idx if isinstance(idx, dict) else {}


def _save_trades_index(s3, idx: dict) -> None:
    s3_put_json(s3, bucket=BUCKET, key=TRADES_INDEX_KEY, payload=idx)


def _index_set_trade(s3, *, trade_id: str, key: str, as_of: str) -> None:
    idx = _load_trades_index(s3)
    idx[str(trade_id)] = {"key": str(key), "as_of": str(as_of)}
    _save_trades_index(s3, idx)


def _audit_backup_key(src_key: str) -> str:
    # put backups under engine/v1/trades_audit/ with safe filename
    safe = src_key.replace("/", "__")
    return f"{TRADES_AUDIT_PREFIX}/{safe}"


def _rebuild_trades_index(s3) -> Tuple[int, int, Dict[str, Any]]:
    """
    Scan S3 under engine/v1/trades/dt=... and rebuild index.json (in-memory).
    Returns (scanned_keys, indexed, index_dict)
    """
    prefix = engine_key(TRADES_TABLE) + "/"
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in (resp.get("Contents") or []):
            k = it.get("Key", "")
            if not k:
                continue
            if not k.endswith(".json"):
                continue
            if "/trade_" not in k:
                continue
            # defensive excludes
            if k.endswith("/index.json") or k.endswith("/latest.json"):
                continue
            keys.append(k)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

    idx: Dict[str, Any] = {}
    scanned = 0
    indexed = 0

    for k in keys:
        scanned += 1
        name = k.split("/")[-1]
        if not (name.startswith("trade_") and name.endswith(".json")):
            continue
        trade_id = name[len("trade_") : -len(".json")]

        as_of = ""
        for part in k.split("/"):
            if part.startswith("dt="):
                as_of = part[len("dt=") :]
                break

        idx[str(trade_id)] = {"key": str(k), "as_of": str(as_of)}
        indexed += 1

    return scanned, indexed, idx


def migrate_trades_index(*, dry_run: bool = False) -> None:
    s3 = s3_client(REGION)
    scanned, indexed, idx = _rebuild_trades_index(s3)

    print("\n=== MIGRATE TRADES INDEX ===")
    print(f"scanned:  {scanned}")
    print(f"indexed:  {indexed}")
    print(f"index:    s3://{BUCKET}/{TRADES_INDEX_KEY}")
    if dry_run:
        print("[DRY RUN] no write performed.")
        print("")
        return

    _save_trades_index(s3, idx)
    print("[OK] index.json written/overwritten.")
    print("")


# ----------------------------
# Core: record
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
    universe_path: Optional[str] = None,
    universe_key: Optional[str] = None,
    strict_universe: bool = False,
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
        asset_id=asset_id_norm,
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
    if trade.action_tag:
        print(f"action_tag:{trade.action_tag}")
    if trade.quantity_unit:
        print(f"unit:      {trade.quantity_unit}")
    if trade.value is not None:
        print(f"value:     {trade.value}")
    if trade.reported_pnl is not None:
        print(f"reported:  {trade.reported_pnl}")
    print("")

    if dry_run:
        print("[DRY RUN] Would write:")
        print(f"  s3://{BUCKET}/{trade_key}")
        print(f"  s3://{BUCKET}/{latest_key}")
        print(f"  s3://{BUCKET}/{TRADES_INDEX_KEY} (update trade_id mapping)")
        return

    s3_put_json(s3, bucket=BUCKET, key=trade_key, payload=payload)
    s3_put_json(s3, bucket=BUCKET, key=latest_key, payload=payload)
    _index_set_trade(s3, trade_id=str(trade.trade_id), key=trade_key, as_of=as_of_norm)

    print("[OK] Wrote trade:")
    print(f"  s3://{BUCKET}/{trade_key}")
    print("[OK] Updated latest:")
    print(f"  s3://{BUCKET}/{latest_key}")
    print("[OK] Updated index:")
    print(f"  s3://{BUCKET}/{TRADES_INDEX_KEY}")
    print("")


# ----------------------------
# Core: edit
# ----------------------------
def edit_trade(
    *,
    trade_id: str,
    old_as_of: Optional[str],
    patch: dict,
    new_as_of: Optional[str] = None,
    dry_run: bool = False,
    write_backup: bool = True,
) -> None:
    s3 = s3_client(REGION)

    # Resolve existing key (fast path by dt, else via index)
    old_key = None
    old_dt = None

    if old_as_of:
        old_dt = _parse_date(old_as_of)
        old_key = dt_key(TRADES_TABLE, old_dt, f"trade_{trade_id}.json")
    else:
        idx = _load_trades_index(s3)
        meta = idx.get(str(trade_id))
        if isinstance(meta, dict) and meta.get("key"):
            old_key = str(meta["key"])
            old_dt = str(meta.get("as_of") or "") or None

    if not old_key:
        raise ValueError(
            "Cannot resolve trade location. Provide --old-as-of once, or run --mode migrate to build index."
        )

    if not s3_exists(s3, bucket=BUCKET, key=old_key):
        raise RuntimeError(f"Trade not found: s3://{BUCKET}/{old_key} (index may be stale; run --mode migrate)")

    obj = json.loads(s3_get_bytes(s3, bucket=BUCKET, key=old_key).decode("utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("Trade JSON is not an object")

    # Apply patch
    for k, v in patch.items():
        obj[k] = v

    # If moving partitions, keep payload consistent
    if new_as_of:
        obj["as_of"] = _parse_date(new_as_of)

    # Validate action_tag if present
    if "action_tag" in obj:
        at = _normalize_action_tag(obj.get("action_tag"))
        if at is None:
            raise ValueError("action_tag cannot be null/empty after edit (must be open/close/add/reduce).")
        obj["action_tag"] = at

    # Validate side if present
    if "side" in obj:
        obj["side"] = _validate_side(obj.get("side"))

    if "quantity" in obj:
        obj["quantity"] = _validate_positive("quantity", obj.get("quantity"))
    if "price" in obj:
        obj["price"] = _validate_positive("price", obj.get("price"))
    if "quantity_unit" in obj and obj.get("quantity_unit") is not None:
        obj["quantity_unit"] = _normalize_quantity_unit(obj.get("quantity_unit"))

    # Decide destination dt
    if new_as_of:
        dst_dt = _parse_date(new_as_of)
    else:
        dst_dt = old_dt or _parse_date(str(obj.get("as_of") or ""))

    if not dst_dt:
        raise ValueError("Could not resolve destination dt (missing old dt and payload as_of).")

    dst_key = dt_key(TRADES_TABLE, dst_dt, f"trade_{trade_id}.json")

    print("\n=== EDIT TRADE ===")
    print(f"trade_id:  {trade_id}")
    print(f"from:      s3://{BUCKET}/{old_key}")
    print(f"to:        s3://{BUCKET}/{dst_key}")
    print(f"patch:     {sorted(list(patch.keys()))}")
    if write_backup:
        print(f"backup:    s3://{BUCKET}/{_audit_backup_key(old_key)}")
    print("")

    if dry_run:
        print("[DRY RUN] no writes performed.")
        return

    # Backup original
    if write_backup:
        backup_key = _audit_backup_key(old_key)
        s3_copy(s3, bucket=BUCKET, src_key=old_key, dst_key=backup_key)

    # Move if dt changes
    if dst_key != old_key:
        s3_copy(s3, bucket=BUCKET, src_key=old_key, dst_key=dst_key)
        s3_delete(s3, bucket=BUCKET, key=old_key)

    # Write patched payload (overwrite destination)
    s3_put_json(s3, bucket=BUCKET, key=dst_key, payload=obj)

    # Keep latest.json semantics
    latest_key = engine_key(TRADES_TABLE, "latest.json")
    s3_put_json(s3, bucket=BUCKET, key=latest_key, payload=obj)

    # Update index
    _index_set_trade(s3, trade_id=trade_id, key=dst_key, as_of=dst_dt)

    print("[OK] Updated:")
    print(f"  s3://{BUCKET}/{dst_key}")
    print("[OK] Updated index + latest.json")
    print("")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Record/edit trades in S3 (idempotent per trade_id).")

    ap.add_argument("--mode", choices=["record", "edit", "migrate"], default="record")

    # record args (validated in main() when mode=record)
    ap.add_argument("--as-of", required=False, help="Trade date (YYYY-MM-DD).")
    ap.add_argument("--ticker", required=False, help="Broker ticker, e.g. NVDA or BTC-USD or EUR-USD.")
    ap.add_argument("--side", required=False, choices=["BUY", "SELL", "buy", "sell"], help="BUY or SELL.")
    ap.add_argument("--quantity", required=False, type=float, help="Quantity (>0).")
    ap.add_argument("--price", required=False, type=float, help="Trade price (>0).")
    ap.add_argument("--currency", default="USD", help="Currency code, default USD.")

    ap.add_argument("--trade-id", default=None, help="If omitted, auto-generated (record) or required (edit).")
    ap.add_argument("--ts-utc", default=None, help="If omitted, uses current UTC ISO timestamp (record).")

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

    # metadata
    ap.add_argument("--choice-id", default=None, help="Link to portfolio choice if relevant.")
    ap.add_argument("--portfolio-run-id", default=None, help="Link to portfolio search run if relevant.")
    ap.add_argument("--note", default=None, help="Free text note.")
    ap.add_argument("--action-tag", default=None, help="open/close/add/reduce.")
    ap.add_argument("--quantity-unit", default=None, help="shares/contracts/coins (normalized).")
    ap.add_argument("--value", default=None, type=float, help="Notional value (required for contracts).")
    ap.add_argument("--reported-pnl", default=None, type=float, help="Broker reported PnL (optional).")

    # edit-only
    ap.add_argument("--old-as-of", default=None, help="Existing dt for the trade (optional if index exists).")
    ap.add_argument("--new-as-of", default=None, help="Move trade to new dt partition (rare).")
    ap.add_argument("--no-backup", action="store_true", help="Disable backup copy on edit (not recommended).")

    ap.add_argument("--dry-run", action="store_true", help="Show actions but do not write.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "migrate":
        migrate_trades_index(dry_run=bool(args.dry_run))
        return

    if args.mode == "edit":
        if not args.trade_id:
            raise ValueError("--trade-id is required for --mode edit")

        patch: Dict[str, Any] = {}

        # patch only fields provided
        if args.action_tag is not None:
            patch["action_tag"] = _normalize_action_tag(args.action_tag)
        if args.quantity_unit is not None:
            patch["quantity_unit"] = _normalize_quantity_unit(args.quantity_unit)
        if args.value is not None:
            patch["value"] = float(args.value)
        if args.reported_pnl is not None:
            patch["reported_pnl"] = float(args.reported_pnl)
        if args.note is not None:
            patch["note"] = (str(args.note) if args.note else None)
        if args.side is not None:
            patch["side"] = _validate_side(args.side)
        if args.quantity is not None:
            patch["quantity"] = _validate_positive("quantity", args.quantity)
        if args.price is not None:
            patch["price"] = _validate_positive("price", args.price)

        if not patch and not args.new_as_of:
            raise ValueError("Nothing to edit: provide at least one patch field (e.g. --action-tag) or --new-as-of.")

        edit_trade(
            trade_id=str(args.trade_id),
            old_as_of=(str(args.old_as_of) if args.old_as_of else None),
            new_as_of=(str(args.new_as_of) if args.new_as_of else None),
            patch=patch,
            dry_run=bool(args.dry_run),
            write_backup=(not bool(args.no_backup)),
        )
        return

    # record
    # enforce required inputs only for record mode
    for name in ["as_of", "ticker", "side", "quantity", "price"]:
        if getattr(args, name) in (None, ""):
            raise ValueError(f"--{name.replace('_','-')} is required for --mode record")

    record_trade(
        as_of=str(args.as_of),
        ticker=str(args.ticker),
        side=str(args.side),
        quantity=float(args.quantity),
        price=float(args.price),
        currency=str(args.currency),
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