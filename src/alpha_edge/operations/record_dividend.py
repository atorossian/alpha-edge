from __future__ import annotations

import argparse
import io
import json
import math
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List

import boto3
import pandas as pd


BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"

DIVIDENDS_TABLE = "dividends"


# -----------------------------
# S3 + Key helpers
# -----------------------------
def s3_client(region: str = REGION):
    return boto3.client("s3", region_name=region)


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def dt_key(table: str, dt_str: str, filename: str) -> str:
    return engine_key(table, f"dt={dt_str}", filename)


DIVIDENDS_INDEX_KEY = engine_key(DIVIDENDS_TABLE, "index.json")


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


def s3_get_json_optional(s3, *, bucket: str, key: str) -> Optional[dict]:
    try:
        raw = s3_get_bytes(s3, bucket=bucket, key=key)
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def s3_exists(s3, *, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def s3_copy(s3, *, bucket: str, src_key: str, dst_key: str) -> None:
    s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": src_key}, Key=dst_key)


def s3_delete(s3, *, bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)


def s3_list_keys(s3, *, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in (resp.get("Contents") or []):
            k = it["Key"]
            # Only dividend records (not latest.json)
            if k.endswith(".json") and "/dividend_" in k:
                keys.append(k)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


# -----------------------------
# Index helpers: dividend_id -> key (+as_of)
# -----------------------------
def _load_dividends_index(s3) -> dict:
    idx = s3_get_json_optional(s3, bucket=BUCKET, key=DIVIDENDS_INDEX_KEY)
    return idx if isinstance(idx, dict) else {}


def _save_dividends_index(s3, idx: dict) -> None:
    s3_put_json(s3, bucket=BUCKET, key=DIVIDENDS_INDEX_KEY, payload=idx)


def _index_set(s3, *, dividend_id: str, key: str, as_of: str) -> None:
    idx = _load_dividends_index(s3)
    idx[str(dividend_id)] = {"key": str(key), "as_of": str(as_of)}
    _save_dividends_index(s3, idx)


def _rebuild_dividends_index(s3) -> Tuple[int, int]:
    """
    Rebuild from S3 keys. Returns (scanned, indexed).
    """
    prefix = engine_key(DIVIDENDS_TABLE) + "/"
    keys = s3_list_keys(s3, bucket=BUCKET, prefix=prefix)

    idx: Dict[str, Dict[str, str]] = {}
    scanned = 0
    indexed = 0

    for k in keys:
        scanned += 1
        name = k.split("/")[-1]
        if not (name.startswith("dividend_") and name.endswith(".json")):
            continue

        dividend_id = name[len("dividend_") : -len(".json")]

        # extract as_of from dt=YYYY-MM-DD in key
        as_of = ""
        for part in k.split("/"):
            if part.startswith("dt="):
                as_of = part[len("dt=") :]
                break

        # fallback: read JSON if dt= missing (rare)
        if not as_of:
            try:
                raw = s3_get_bytes(s3, bucket=BUCKET, key=k)
                obj = json.loads(raw.decode("utf-8"))
                as_of = str(obj.get("as_of") or "")
            except Exception:
                as_of = ""

        idx[str(dividend_id)] = {"key": str(k), "as_of": as_of}
        indexed += 1

    _save_dividends_index(s3, idx)
    return scanned, indexed


# -----------------------------
# Parsing / validation helpers
# -----------------------------
def _parse_date(s: str) -> str:
    return pd.Timestamp(s).date().strftime("%Y-%m-%d")


def _iso_utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _validate_signed_nonzero(name: str, x: float) -> float:
    x = float(x)
    if x == 0.0:
        raise ValueError(f"{name} must be non-zero (positive to receive, negative to pay)")
    return x


def _validate_nonneg(name: str, x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    if x < 0.0:
        raise ValueError(f"{name} must be >= 0 (or omitted)")
    return x


def _validate_positive_optional(name: str, x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    if not (x > 0.0):
        raise ValueError(f"{name} must be > 0 (or omitted)")
    return x


def _validate_date_optional(name: str, s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = str(s).strip()
    if not s2:
        return None
    return _parse_date(s2)


def _extract_dt_from_key(key: str) -> Optional[str]:
    for part in key.split("/"):
        if part.startswith("dt="):
            return part[len("dt=") :]
    return None


# ---- optional universe resolver for asset_id (same spirit as record_trade) ----
DEFAULT_UNIVERSE_KEY = f"{ENGINE_ROOT}/universe/universe.csv"


def _load_universe_df(*, universe_path: Optional[str], universe_key: Optional[str]) -> Tuple[pd.DataFrame, str]:
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
            raise RuntimeError(f"Ambiguous universe mapping for ticker={bt}: {len(m2)} rows match (ref={ref}).")

    if len(m) > 1:
        raise RuntimeError(f"Ambiguous universe mapping for broker_ticker={bt}: {len(m)} rows match (ref={ref}).")

    raise RuntimeError(f"No universe mapping for broker_ticker/ticker={bt} (ref={ref}).")


@dataclass
class Dividend:
    dividend_id: str
    as_of: str
    ts_utc: str
    account_id: str
    asset_id: str
    ticker: Optional[str]
    amount: float
    currency: str
    withholding_tax: Optional[float] = None
    note: Optional[str] = None

    # Quantfury-friendly
    shares_held: Optional[float] = None
    dividend_per_share: Optional[float] = None
    ex_date: Optional[str] = None
    record_date: Optional[str] = None
    pay_date: Optional[str] = None
    gross_amount: Optional[float] = None
    source: Optional[str] = None


_NEW_SCHEMA_KEYS = [
    "shares_held",
    "dividend_per_share",
    "ex_date",
    "record_date",
    "pay_date",
    "gross_amount",
    "source",
]


def record_dividend(
    *,
    as_of: str,
    ticker: Optional[str],
    asset_id: Optional[str],
    amount: float,
    currency: str = "USD",
    withholding_tax: Optional[float] = None,
    account_id: str = "main",
    ts_utc: Optional[str] = None,
    dividend_id: Optional[str] = None,
    note: Optional[str] = None,
    shares_held: Optional[float] = None,
    dividend_per_share: Optional[float] = None,
    ex_date: Optional[str] = None,
    record_date: Optional[str] = None,
    pay_date: Optional[str] = None,
    source: Optional[str] = "manual",
    strict_math: bool = False,
    math_tol: float = 0.05,
    universe_path: Optional[str] = None,
    universe_key: Optional[str] = None,
    strict_universe: bool = False,
    dry_run: bool = False,
) -> None:
    s3 = s3_client(REGION)

    as_of_norm = _parse_date(as_of)
    amt = _validate_signed_nonzero("amount", amount)
    ccy = str(currency).upper().strip() or "USD"
    acct = str(account_id).strip() or "main"
    tkr = None if ticker is None else str(ticker).upper().strip()

    tax = _validate_nonneg("withholding_tax", withholding_tax)

    sh = _validate_positive_optional("shares_held", shares_held)
    dps = _validate_positive_optional("dividend_per_share", dividend_per_share)
    exd = _validate_date_optional("ex_date", ex_date)
    rcd = _validate_date_optional("record_date", record_date)
    payd = _validate_date_optional("pay_date", pay_date)

    aid = None if asset_id is None else str(asset_id).strip()
    universe_ref = None

    if not aid:
        if tkr and (universe_path or universe_key):
            aid, universe_ref = _resolve_asset_id_from_universe(
                broker_ticker=tkr,
                universe_path=universe_path,
                universe_key=universe_key,
            )
        elif strict_universe:
            raise ValueError(
                "asset_id not provided and --strict-universe requested but cannot resolve (need ticker+universe)."
            )
        else:
            raise ValueError("asset_id is required for dividends (provide --asset-id, or --ticker + universe to resolve).")

    if ts_utc is None:
        ts_utc = _iso_utc_now()

    if dividend_id is None:
        dividend_id = f"{as_of_norm.replace('-', '')}-{uuid.uuid4().hex[:10]}"

    gross = None
    if sh is not None and dps is not None:
        gross = float(sh) * float(dps)
        if strict_math:
            tax_amt = float(tax) if tax is not None else 0.0
            implied = (float(amt) + tax_amt) if float(amt) > 0 else (math.fabs(float(amt)) + tax_amt)
            if math.fabs(implied - gross) > float(math_tol):
                raise ValueError(
                    f"Dividend math mismatch: gross(shares*per_share)={gross:.6f} "
                    f"but implied={implied:.6f} (from amount={amt:.6f}, tax={tax_amt:.6f}) "
                    f"(tol={math_tol})."
                )

    div = Dividend(
        dividend_id=str(dividend_id),
        as_of=as_of_norm,
        ts_utc=str(ts_utc),
        account_id=acct,
        asset_id=str(aid),
        ticker=tkr,
        amount=float(amt),
        currency=ccy,
        withholding_tax=(float(tax) if tax is not None else None),
        note=(str(note) if note else None),
        shares_held=(float(sh) if sh is not None else None),
        dividend_per_share=(float(dps) if dps is not None else None),
        ex_date=exd,
        record_date=rcd,
        pay_date=payd,
        gross_amount=(float(gross) if gross is not None else None),
        source=(str(source).strip() if source else None),
    )

    payload = asdict(div)

    key = dt_key(DIVIDENDS_TABLE, as_of_norm, f"dividend_{div.dividend_id}.json")
    latest_key = engine_key(DIVIDENDS_TABLE, "latest.json")

    print("\n=== RECORD DIVIDEND ===")
    print(f"as_of:        {div.as_of}")
    print(f"dividend_id:  {div.dividend_id}")
    print(f"ts_utc:       {div.ts_utc}")
    print(f"account_id:   {div.account_id}")
    print(f"asset_id:     {div.asset_id}")
    if div.ticker:
        print(f"ticker:       {div.ticker}")
    print(f"amount:       {div.amount} {div.currency}")
    if div.withholding_tax is not None:
        print(f"withholding:  {div.withholding_tax} {div.currency}")
    if div.shares_held is not None:
        print(f"shares_held:  {div.shares_held}")
    if div.dividend_per_share is not None:
        print(f"per_share:    {div.dividend_per_share} {div.currency}")
    if div.gross_amount is not None:
        print(f"gross:        {div.gross_amount} {div.currency}")
    if div.ex_date:
        print(f"ex_date:      {div.ex_date}")
    if div.record_date:
        print(f"record_date:  {div.record_date}")
    if div.pay_date:
        print(f"pay_date:     {div.pay_date}")
    if div.source:
        print(f"source:       {div.source}")
    if universe_ref:
        print(f"universe:     {universe_ref}")
    if div.note:
        print(f"note:         {div.note}")
    print("")

    if dry_run:
        print("[DRY RUN] Would write:")
        print(f"  s3://{BUCKET}/{key}")
        print(f"  s3://{BUCKET}/{latest_key}")
        print(f"  s3://{BUCKET}/{DIVIDENDS_INDEX_KEY} (index update)")
        return

    s3_put_json(s3, bucket=BUCKET, key=key, payload=payload)
    s3_put_json(s3, bucket=BUCKET, key=latest_key, payload=payload)

    # Update index AFTER successful write
    _index_set(s3, dividend_id=div.dividend_id, key=key, as_of=div.as_of)

    print("[OK] Wrote dividend:")
    print(f"  s3://{BUCKET}/{key}")
    print("[OK] Updated latest:")
    print(f"  s3://{BUCKET}/{latest_key}")
    print(f"[OK] Updated index:")
    print(f"  s3://{BUCKET}/{DIVIDENDS_INDEX_KEY}")
    print("")


def edit_dividend(
    *,
    dividend_id: str,
    old_as_of: Optional[str],
    patch: Dict[str, Any],
    new_as_of: Optional[str] = None,
    strict_math: bool = False,
    math_tol: float = 0.05,
    dry_run: bool = False,
) -> None:
    s3 = s3_client(REGION)

    # Resolve existing key:
    old_key: Optional[str] = None
    old_dt: Optional[str] = None

    if old_as_of:
        old_dt = _parse_date(old_as_of)
        old_key = dt_key(DIVIDENDS_TABLE, old_dt, f"dividend_{dividend_id}.json")
    else:
        idx = _load_dividends_index(s3)
        meta = idx.get(str(dividend_id))
        if isinstance(meta, dict) and meta.get("key"):
            old_key = str(meta["key"])
            old_dt = _extract_dt_from_key(old_key)

    if not old_key:
        raise ValueError(
            "Cannot resolve dividend location. Provide --old-as-of once, or run --mode migrate to rebuild the index."
        )

    if not s3_exists(s3, bucket=BUCKET, key=old_key):
        raise RuntimeError(f"Dividend not found: s3://{BUCKET}/{old_key} (index may be stale; run --mode migrate)")

    raw = s3_get_bytes(s3, bucket=BUCKET, key=old_key)
    obj = json.loads(raw.decode("utf-8"))

    # Apply patch keys (omitted keys are unchanged)
    for k, v in patch.items():
        obj[k] = v

    # Ensure schema keys exist
    for k in _NEW_SCHEMA_KEYS:
        obj.setdefault(k, None)

    # Recompute gross if shares+per_share present and gross not explicitly set in patch
    if "gross_amount" not in patch:
        sh = obj.get("shares_held")
        dps = obj.get("dividend_per_share")
        if sh is not None and dps is not None:
            try:
                obj["gross_amount"] = float(sh) * float(dps)
            except Exception:
                pass

    # Strict math validation (works for both long and short)
    if strict_math:
        gross = obj.get("gross_amount")
        amt = obj.get("amount")
        tax = obj.get("withholding_tax")
        if gross is not None and amt is not None:
            tax_amt = float(tax) if tax is not None else 0.0
            gross_f = float(gross)
            amt_f = float(amt)
            implied = (amt_f + tax_amt) if amt_f > 0 else (math.fabs(amt_f) + tax_amt)
            if math.fabs(implied - gross_f) > float(math_tol):
                raise ValueError(
                    f"Dividend math mismatch (edit): gross={gross_f:.6f} "
                    f"but implied={implied:.6f} (from amount={amt_f:.6f}, tax={tax_amt:.6f}) "
                    f"(tol={math_tol})."
                )

    # Decide destination dt partition
    if new_as_of:
        dst_dt = _parse_date(new_as_of)
    else:
        # Prefer obj["as_of"], else fall back to old_dt extracted from key
        obj_as_of = obj.get("as_of")
        if isinstance(obj_as_of, str) and obj_as_of.strip():
            dst_dt = _parse_date(obj_as_of.strip())
        elif old_dt:
            dst_dt = old_dt
        else:
            raise ValueError("Cannot determine destination dt partition. Provide --new-as-of or ensure record has as_of.")

    dst_key = dt_key(DIVIDENDS_TABLE, dst_dt, f"dividend_{dividend_id}.json")

    print("\n=== EDIT DIVIDEND ===")
    print(f"dividend_id:  {dividend_id}")
    print(f"from:         s3://{BUCKET}/{old_key}")
    print(f"to:           s3://{BUCKET}/{dst_key}")
    print(f"patch_keys:   {sorted(list(patch.keys()))}")
    print("")

    if dry_run:
        print("[DRY RUN] Would update JSON (and move if dt changed).")
        print(f"[DRY RUN] Would update index: s3://{BUCKET}/{DIVIDENDS_INDEX_KEY}")
        return

    # If dt changed, move object (copy+delete)
    if dst_key != old_key:
        s3_copy(s3, bucket=BUCKET, src_key=old_key, dst_key=dst_key)
        s3_delete(s3, bucket=BUCKET, key=old_key)

    # Write updated object
    s3_put_json(s3, bucket=BUCKET, key=dst_key, payload=obj)

    # Keep your current semantics: latest.json is "last write"
    latest_key = engine_key(DIVIDENDS_TABLE, "latest.json")
    s3_put_json(s3, bucket=BUCKET, key=latest_key, payload=obj)

    # Update index AFTER successful write/move
    as_of_for_index = str(obj.get("as_of") or dst_dt)
    _index_set(s3, dividend_id=str(dividend_id), key=dst_key, as_of=as_of_for_index)

    print("[OK] Updated:")
    print(f"  s3://{BUCKET}/{dst_key}")
    print("[OK] Updated latest:")
    print(f"  s3://{BUCKET}/{latest_key}")
    print("[OK] Updated index:")
    print(f"  s3://{BUCKET}/{DIVIDENDS_INDEX_KEY}")
    print("")


def migrate_dividends_schema(*, dry_run: bool = False) -> None:
    s3 = s3_client(REGION)
    prefix = engine_key(DIVIDENDS_TABLE) + "/"
    keys = s3_list_keys(s3, bucket=BUCKET, prefix=prefix)

    scanned = 0
    updated = 0

    for key in keys:
        scanned += 1
        raw = s3_get_bytes(s3, bucket=BUCKET, key=key)
        obj = json.loads(raw.decode("utf-8"))

        changed = False
        for k in _NEW_SCHEMA_KEYS:
            if k not in obj:
                obj[k] = None
                changed = True

        if changed:
            updated += 1
            if not dry_run:
                s3_put_json(s3, bucket=BUCKET, key=key, payload=obj)

    print("\n=== MIGRATE DIVIDENDS SCHEMA ===")
    print(f"prefix:         s3://{BUCKET}/{prefix}")
    print(f"objects_scanned:{scanned}")
    print(f"objects_updated:{updated}")

    if dry_run:
        print("[DRY RUN] No writes performed.")
        print("[DRY RUN] Index NOT rebuilt.")
    else:
        # Rebuild index so edit can work without --old-as-of
        scanned2, indexed2 = _rebuild_dividends_index(s3)
        print("[OK] Migration completed.")
        print(f"[OK] Index rebuilt: scanned={scanned2} indexed={indexed2}")
        print(f"  s3://{BUCKET}/{DIVIDENDS_INDEX_KEY}")
    print("")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Record/edit/migrate dividend records in S3.")

    ap.add_argument("--mode", choices=["record", "edit", "migrate"], default="record")

    ap.add_argument("--as-of", default=None, help="Date (YYYY-MM-DD). (record-mode)")
    ap.add_argument("--amount", default=None, type=float, help="Net amount (record-mode required).")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--withholding-tax", default=None, type=float)
    ap.add_argument("--tax", dest="withholding_tax", default=None, help="Alias for --withholding-tax (backwards compatible).")
    ap.add_argument("--account-id", default="main")

    ap.add_argument("--asset-id", default=None)
    ap.add_argument("--ticker", default=None)

    ap.add_argument("--universe-path", default=None)
    ap.add_argument("--universe-key", default=None, help=f"Example: {DEFAULT_UNIVERSE_KEY}")
    ap.add_argument("--strict-universe", action="store_true")

    ap.add_argument("--ts-utc", default=None)
    ap.add_argument("--dividend-id", default=None)
    ap.add_argument("--note", default=None)

    ap.add_argument("--shares-held", default=None, type=float)
    ap.add_argument("--dividend-per-share", default=None, type=float)
    ap.add_argument("--ex-date", default=None)
    ap.add_argument("--record-date", default=None)
    ap.add_argument("--pay-date", default=None)
    ap.add_argument("--source", default="manual")

    ap.add_argument("--strict-math", action="store_true")
    ap.add_argument("--math-tol", default=0.05, type=float)

    ap.add_argument("--old-as-of", default=None, help="(edit-mode optional if index exists) Existing dt partition for the record.")
    ap.add_argument("--new-as-of", default=None, help="(edit-mode optional) Move record to new dt partition.")

    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "migrate":
        migrate_dividends_schema(dry_run=bool(args.dry_run))
        return

    if args.mode == "edit":
        if not args.dividend_id:
            raise ValueError("--dividend-id is required for --mode edit")

        patch: Dict[str, Any] = {}

        if args.as_of is not None:
            patch["as_of"] = _parse_date(args.as_of)
        if args.ts_utc is not None:
            patch["ts_utc"] = str(args.ts_utc)
        if args.account_id is not None:
            patch["account_id"] = str(args.account_id).strip() or "main"
        if args.asset_id is not None:
            patch["asset_id"] = str(args.asset_id).strip()
        if args.ticker is not None:
            patch["ticker"] = str(args.ticker).upper().strip()
        if args.amount is not None:
            patch["amount"] = float(_validate_signed_nonzero("amount", args.amount))
        if args.currency is not None:
            patch["currency"] = str(args.currency).upper().strip() or "USD"
        if args.withholding_tax is not None:
            patch["withholding_tax"] = float(_validate_nonneg("withholding_tax", args.withholding_tax))
        if args.note is not None:
            patch["note"] = (str(args.note) if args.note else None)

        if args.shares_held is not None:
            patch["shares_held"] = float(_validate_positive_optional("shares_held", args.shares_held))
        if args.dividend_per_share is not None:
            patch["dividend_per_share"] = float(
                _validate_positive_optional("dividend_per_share", args.dividend_per_share)
            )
        if args.ex_date is not None:
            patch["ex_date"] = _validate_date_optional("ex_date", args.ex_date)
        if args.record_date is not None:
            patch["record_date"] = _validate_date_optional("record_date", args.record_date)
        if args.pay_date is not None:
            patch["pay_date"] = _validate_date_optional("pay_date", args.pay_date)
        if args.source is not None:
            patch["source"] = (str(args.source).strip() if args.source else None)

        edit_dividend(
            dividend_id=str(args.dividend_id),
            old_as_of=args.old_as_of,  # IMPORTANT: don't cast None -> "None"
            new_as_of=(str(args.new_as_of) if args.new_as_of else None),
            patch=patch,
            strict_math=bool(args.strict_math),
            math_tol=float(args.math_tol),
            dry_run=bool(args.dry_run),
        )
        return

    # record
    if not args.as_of:
        raise ValueError("--as-of is required for --mode record")
    if args.amount is None:
        raise ValueError("--amount is required for --mode record")

    record_dividend(
        as_of=args.as_of,
        ticker=args.ticker,
        asset_id=args.asset_id,
        amount=float(args.amount),
        currency=args.currency,
        withholding_tax=args.withholding_tax,
        account_id=args.account_id,
        ts_utc=args.ts_utc,
        dividend_id=args.dividend_id,
        note=args.note,
        shares_held=args.shares_held,
        dividend_per_share=args.dividend_per_share,
        ex_date=args.ex_date,
        record_date=args.record_date,
        pay_date=args.pay_date,
        source=args.source,
        strict_math=bool(args.strict_math),
        math_tol=float(args.math_tol),
        universe_path=args.universe_path,
        universe_key=args.universe_key,
        strict_universe=bool(args.strict_universe),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()