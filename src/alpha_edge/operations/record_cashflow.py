from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Literal

import boto3
import pandas as pd


BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"

CASHFLOWS_TABLE = "cashflows"


def s3_client(region: str = REGION):
    return boto3.client("s3", region_name=region)


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def dt_key(table: str, dt_str: str, filename: str) -> str:
    return engine_key(table, f"dt={dt_str}", filename)


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _parse_date(s: str) -> str:
    return pd.Timestamp(s).date().strftime("%Y-%m-%d")


def _iso_utc_now() -> str:
    # store as ISO; consumers can parse
    return pd.Timestamp.utcnow().isoformat()


def _validate_positive(name: str, x: float) -> float:
    x = float(x)
    if not (x > 0.0):
        raise ValueError(f"{name} must be > 0")
    return x


def _normalize_type(x: str) -> Literal["DEPOSIT", "WITHDRAWAL"]:
    s = str(x).upper().strip()
    if s not in {"DEPOSIT", "WITHDRAWAL"}:
        raise ValueError("type must be DEPOSIT or WITHDRAWAL")
    return s  # type: ignore[return-value]


@dataclass
class Cashflow:
    cashflow_id: str
    as_of: str
    ts_utc: str
    account_id: str
    type: str  # DEPOSIT / WITHDRAWAL
    amount: float
    currency: str
    note: Optional[str] = None


def record_cashflow(
    *,
    as_of: str,
    type: str,
    amount: float,
    currency: str = "USD",
    account_id: str = "main",
    ts_utc: Optional[str] = None,
    cashflow_id: Optional[str] = None,
    note: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    s3 = s3_client(REGION)

    as_of_norm = _parse_date(as_of)
    type_norm = _normalize_type(type)
    amt = _validate_positive("amount", amount)
    ccy = str(currency).upper().strip() or "USD"
    acct = str(account_id).strip() or "main"

    if ts_utc is None:
        ts_utc = _iso_utc_now()

    if cashflow_id is None:
        cashflow_id = f"{as_of_norm.replace('-', '')}-{uuid.uuid4().hex[:10]}"

    cf = Cashflow(
        cashflow_id=str(cashflow_id),
        as_of=as_of_norm,
        ts_utc=str(ts_utc),
        account_id=acct,
        type=type_norm,
        amount=float(amt),
        currency=ccy,
        note=(str(note) if note else None),
    )

    payload = asdict(cf)

    key = dt_key(CASHFLOWS_TABLE, as_of_norm, f"cashflow_{cf.cashflow_id}.json")
    latest_key = engine_key(CASHFLOWS_TABLE, "latest.json")

    print("\n=== RECORD CASHFLOW ===")
    print(f"as_of:        {cf.as_of}")
    print(f"cashflow_id:  {cf.cashflow_id}")
    print(f"ts_utc:       {cf.ts_utc}")
    print(f"account_id:   {cf.account_id}")
    print(f"type:         {cf.type}")
    print(f"amount:       {cf.amount} {cf.currency}")
    if cf.note:
        print(f"note:         {cf.note}")
    print("")

    if dry_run:
        print("[DRY RUN] Would write:")
        print(f"  s3://{BUCKET}/{key}")
        print(f"  s3://{BUCKET}/{latest_key}")
        return

    s3_put_json(s3, bucket=BUCKET, key=key, payload=payload)
    s3_put_json(s3, bucket=BUCKET, key=latest_key, payload=payload)

    print("[OK] Wrote cashflow:")
    print(f"  s3://{BUCKET}/{key}")
    print("[OK] Updated latest:")
    print(f"  s3://{BUCKET}/{latest_key}")
    print("")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Append exactly one cashflow record to S3.")

    ap.add_argument("--as-of", required=True, help="Date (YYYY-MM-DD).")
    ap.add_argument("--type", required=True, choices=["DEPOSIT", "WITHDRAWAL", "deposit", "withdrawal"])
    ap.add_argument("--amount", required=True, type=float, help="Positive amount.")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--account-id", default="main")

    ap.add_argument("--ts-utc", default=None)
    ap.add_argument("--cashflow-id", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    record_cashflow(
        as_of=args.as_of,
        type=args.type,
        amount=args.amount,
        currency=args.currency,
        account_id=args.account_id,
        ts_utc=args.ts_utc,
        cashflow_id=args.cashflow_id,
        note=args.note,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()