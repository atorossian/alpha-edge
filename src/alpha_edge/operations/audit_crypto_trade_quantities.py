from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd


BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"
TRADES_TABLE = "trades"

QTY_EPS = 1e-9


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(region: str = REGION):
    return boto3.client("s3", region_name=region)


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def s3_list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs: dict[str, Any] = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            keys.append(it["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def s3_get_json(s3, *, bucket: str, key: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return json.loads(body.decode("utf-8"))


# ----------------------------
# Trade loading / parsing
# ----------------------------
def _parse_ts(ts_utc: str) -> pd.Timestamp:
    t = pd.to_datetime(ts_utc, errors="coerce", utc=True)
    if pd.isna(t):
        raise ValueError(f"Invalid ts_utc: {ts_utc}")
    return t


def _normalize_unit(u: Optional[str]) -> Optional[str]:
    if u is None:
        return None
    s = str(u).strip().lower()
    if s in {"share", "shares"}:
        return "shares"
    if s in {"contract", "contracts"}:
        return "contracts"
    if s in {"coin", "coins"}:
        return "coins"
    if s in {"ounce", "ounces"}:
        return "ounces"
    # crypto symbol units that should really behave as coins
    if s in {
        "btc", "eth", "sol", "ada", "xrp", "dot", "ltc", "bnb",
        "avax", "link", "matic", "atom", "near", "uni", "aave",
        "trx", "etc", "doge", "hbar", "sui", "dash", "bch"
    }:
        return "coins"
    return s


_FX_PAIR_RE = r"^[A-Z]{3}[-/][A-Z]{3}$"
_CRYPTO_BASES = {
    "BTC", "ETH", "ADA", "XRP", "DOT", "BCH", "LTC", "SOL", "DOGE", "SUI",
    "HBAR", "DASH", "BNB", "AVAX", "LINK", "MATIC", "ATOM", "NEAR", "UNI",
    "AAVE", "TRX", "ETC"
}
_CRYPTO_QUOTES = {"USD", "USDT", "USDC", "EUR"}


def _is_crypto_pair(ticker: str) -> bool:
    t = str(ticker).upper().strip()
    if "-" not in t:
        return False
    base, quote = t.split("-", 1)
    return base in _CRYPTO_BASES and quote in _CRYPTO_QUOTES


def _load_trades(s3, *, start: Optional[str] = None, end: Optional[str] = None) -> List[dict]:
    prefix = engine_key(TRADES_TABLE, "dt=")
    keys = s3_list_keys(s3, bucket=BUCKET, prefix=prefix)
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]

    start_d = pd.Timestamp(start).date() if start else None
    end_d = pd.Timestamp(end).date() if end else None

    out: List[dict] = []
    for k in keys:
        parts = k.split("/")
        dt_part = next((p for p in parts if p.startswith("dt=")), None)
        if not dt_part:
            continue
        d = pd.Timestamp(dt_part.replace("dt=", "")).date()
        if start_d and d < start_d:
            continue
        if end_d and d > end_d:
            continue

        payload = s3_get_json(s3, bucket=BUCKET, key=k)
        if isinstance(payload, dict):
            payload["_s3_key"] = k
            out.append(payload)

    return out


def _normalize_trade(t: dict) -> dict:
    trade_id = str(t.get("trade_id"))
    as_of = str(t.get("as_of"))
    ts_utc = str(t.get("ts_utc"))
    ticker = str(t.get("ticker")).upper().strip()
    side = str(t.get("side")).upper().strip()

    qty = t.get("quantity")
    qty = None if qty is None else float(qty)

    price = t.get("price")
    price = None if price is None else float(price)

    ccy = str(t.get("currency") or "USD").upper().strip()
    asset_id = t.get("asset_id", None)
    asset_id = None if asset_id is None else str(asset_id).strip()

    action_tag = t.get("action_tag", None)
    action_tag = None if action_tag is None else str(action_tag).strip().lower()

    quantity_unit = _normalize_unit(t.get("quantity_unit"))

    value = t.get("value", None)
    value = None if value is None else float(value)

    reported_pnl = t.get("reported_pnl", None)
    reported_pnl = None if reported_pnl is None else float(reported_pnl)

    _ = pd.Timestamp(as_of).date()
    _ = _parse_ts(ts_utc)

    return {
        "trade_id": trade_id,
        "as_of": as_of,
        "ts_utc": ts_utc,
        "ticker": ticker,
        "asset_id": asset_id,
        "side": side,
        "quantity": qty,
        "price": price,
        "currency": ccy,
        "action_tag": action_tag,
        "quantity_unit": quantity_unit,
        "value": value,
        "reported_pnl": reported_pnl,
        "note": t.get("note"),
        "_s3_key": t.get("_s3_key"),
    }


# ----------------------------
# Audit
# ----------------------------
@dataclass
class AuditRow:
    asset_id: str
    ticker: str
    trade_id: str
    as_of: str
    ts_utc: str
    side: str
    action_tag: str
    quantity: Optional[float]
    price: Optional[float]
    value: Optional[float]
    reported_pnl: Optional[float]
    quantity_unit: Optional[str]
    qty_before: float
    qty_delta: float
    qty_after: float
    status: str
    issue_code: str
    issue_detail: str
    s3_key: Optional[str]


def _action_pri(tag: str | None) -> int:
    return 0 if tag in {"open", "add"} else 1


def _side_pri(side: str) -> int:
    return 0 if side == "SELL" else 1


def _snap(x: float, eps: float = QTY_EPS) -> float:
    return 0.0 if abs(float(x)) < eps else float(x)


def audit_crypto_quantity_consistency(trades: list[dict]) -> tuple[list[AuditRow], list[dict]]:
    norm = [_normalize_trade(t) for t in trades]
    norm = [t for t in norm if _is_crypto_pair(t["ticker"])]

    # Keep only trades with asset_id for proper grouping
    grouped: dict[str, list[dict]] = {}
    missing_asset_rows: list[AuditRow] = []

    for t in norm:
        asset_id = t.get("asset_id")
        if not asset_id:
            missing_asset_rows.append(
                AuditRow(
                    asset_id="",
                    ticker=t["ticker"],
                    trade_id=t["trade_id"],
                    as_of=t["as_of"],
                    ts_utc=t["ts_utc"],
                    side=t["side"],
                    action_tag=str(t.get("action_tag")),
                    quantity=t.get("quantity"),
                    price=t.get("price"),
                    value=t.get("value"),
                    reported_pnl=t.get("reported_pnl"),
                    quantity_unit=t.get("quantity_unit"),
                    qty_before=0.0,
                    qty_delta=0.0,
                    qty_after=0.0,
                    status="REFactor",
                    issue_code="MISSING_ASSET_ID",
                    issue_detail="Trade missing asset_id; cannot audit position chain safely.",
                    s3_key=t.get("_s3_key"),
                )
            )
            continue
        grouped.setdefault(asset_id, []).append(t)

    audit_rows: list[AuditRow] = []
    asset_summary: list[dict] = []

    for asset_id, rows in grouped.items():
        rows.sort(
            key=lambda t: (
                _parse_ts(t["ts_utc"]),
                _action_pri(t.get("action_tag")),
                _side_pri(t["side"]),
                t["trade_id"],
            )
        )

        ticker_set = {str(t["ticker"]).upper().strip() for t in rows}
        ticker = sorted(ticker_set)[0] if ticker_set else ""
        qty_open = 0.0
        asset_issues: list[str] = []

        for t in rows:
            trade_id = t["trade_id"]
            side = t["side"]
            action_tag = t.get("action_tag")
            qty = t.get("quantity")
            px = t.get("price")
            val = t.get("value")

            qty_before = qty_open
            status = "OK"
            issue_code = ""
            issue_detail = ""
            qty_delta = 0.0

            if action_tag not in {"open", "add", "close", "reduce"}:
                status = "REFactor"
                issue_code = "BAD_ACTION_TAG"
                issue_detail = f"Invalid action_tag={action_tag!r}"
            elif qty is None or qty <= 0:
                status = "REFactor"
                issue_code = "BAD_QUANTITY"
                issue_detail = f"Invalid quantity={qty!r}"
            elif px is None or px <= 0:
                status = "REFactor"
                issue_code = "BAD_PRICE"
                issue_detail = f"Invalid price={px!r}"
            else:
                # SPOT-style signed quantity logic
                if action_tag in {"open", "add"}:
                    qty_delta = qty if side == "BUY" else -qty
                    qty_open = _snap(qty_open + qty_delta)

                    # optional warning: adding opposite direction while still open
                    if qty_before > QTY_EPS and side == "SELL":
                        status = "CHECK"
                        issue_code = "OPEN_ADD_OPPOSITE_TO_LONG"
                        issue_detail = (
                            "open/add SELL while existing long quantity was open. "
                            "Could be valid if this asset was actually intended as short/CFD, "
                            "but for crypto SPOT this is suspicious."
                        )
                    elif qty_before < -QTY_EPS and side == "BUY":
                        status = "CHECK"
                        issue_code = "OPEN_ADD_OPPOSITE_TO_SHORT"
                        issue_detail = (
                            "open/add BUY while existing short quantity was open. "
                            "Could be valid in margin/CFD logic, suspicious for crypto SPOT."
                        )

                else:
                    # close / reduce
                    if side == "SELL":
                        # closing long
                        if qty_before <= QTY_EPS:
                            status = "REFactor"
                            issue_code = "SELL_CLOSE_WITHOUT_LONG"
                            issue_detail = (
                                f"SELL {action_tag} but quantity before was {qty_before:.12f}; "
                                "no long exposure available to close."
                            )
                        elif qty - qty_before > QTY_EPS:
                            status = "REFactor"
                            issue_code = "SELL_CLOSE_EXCEEDS_LONG"
                            issue_detail = (
                                f"SELL {action_tag} quantity={qty:.12f} exceeds long open quantity="
                                f"{qty_before:.12f}."
                            )
                        else:
                            qty_delta = -qty
                            qty_open = _snap(qty_open + qty_delta)

                    elif side == "BUY":
                        # closing short
                        if qty_before >= -QTY_EPS:
                            status = "REFactor"
                            issue_code = "BUY_CLOSE_WITHOUT_SHORT"
                            issue_detail = (
                                f"BUY {action_tag} but quantity before was {qty_before:.12f}; "
                                "no short exposure available to close."
                            )
                        elif qty - abs(qty_before) > QTY_EPS:
                            status = "REFactor"
                            issue_code = "BUY_CLOSE_EXCEEDS_SHORT"
                            issue_detail = (
                                f"BUY {action_tag} quantity={qty:.12f} exceeds short open quantity="
                                f"{abs(qty_before):.12f}."
                            )
                        else:
                            qty_delta = qty
                            qty_open = _snap(qty_open + qty_delta)
                    else:
                        status = "REFactor"
                        issue_code = "BAD_SIDE"
                        issue_detail = f"Invalid side={side!r}"

            qty_after = qty_open if status != "REFactor" or qty_delta != 0.0 else qty_before

            audit_rows.append(
                AuditRow(
                    asset_id=asset_id,
                    ticker=ticker,
                    trade_id=trade_id,
                    as_of=t["as_of"],
                    ts_utc=t["ts_utc"],
                    side=side,
                    action_tag=str(action_tag),
                    quantity=qty,
                    price=px,
                    value=val,
                    reported_pnl=t.get("reported_pnl"),
                    quantity_unit=t.get("quantity_unit"),
                    qty_before=float(qty_before),
                    qty_delta=float(qty_delta),
                    qty_after=float(qty_after),
                    status=status,
                    issue_code=issue_code,
                    issue_detail=issue_detail,
                    s3_key=t.get("_s3_key"),
                )
            )

            if status in {"REFactor", "CHECK"}:
                asset_issues.append(issue_code or "UNKNOWN")

        asset_summary.append(
            {
                "asset_id": asset_id,
                "ticker": ticker,
                "trade_count": len(rows),
                "final_quantity": float(qty_open),
                "n_ok": sum(1 for r in audit_rows if r.asset_id == asset_id and r.status == "OK"),
                "n_check": sum(1 for r in audit_rows if r.asset_id == asset_id and r.status == "CHECK"),
                "n_refactor": sum(1 for r in audit_rows if r.asset_id == asset_id and r.status == "REFactor"),
                "issue_codes": "|".join(sorted(set(asset_issues))) if asset_issues else "",
            }
        )

    audit_rows = missing_asset_rows + audit_rows
    return audit_rows, asset_summary


# ----------------------------
# Output
# ----------------------------
def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit crypto trade quantity consistency for SPOT migration.")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out-dir", default="./data/audit_crypto_quantity")
    args = ap.parse_args()

    s3 = s3_client(REGION)
    trades = _load_trades(s3, start=args.start, end=args.end)

    audit_rows, asset_summary = audit_crypto_quantity_consistency(trades)

    out_dir = args.out_dir
    pd.DataFrame([asdict(r) for r in audit_rows]).to_csv(f"{out_dir}_rows.csv", index=False)
    pd.DataFrame(asset_summary).to_csv(f"{out_dir}_assets.csv", index=False)

    print("\n=== CRYPTO QUANTITY AUDIT ===")
    print(f"trades_loaded:    {len(trades)}")
    print(f"audit_rows:       {len(audit_rows)}")
    print(f"assets_audited:   {len(asset_summary)}")
    print(f"rows_csv:         {out_dir}_rows.csv")
    print(f"assets_csv:       {out_dir}_assets.csv")
    print("")

    if asset_summary:
        df = pd.DataFrame(asset_summary)
        bad = df[(df["n_refactor"] > 0) | (df["n_check"] > 0)].copy()
        if not bad.empty:
            print("Assets needing review:")
            print(bad.to_string(index=False))
            print("")
        else:
            print("No quantity inconsistencies detected in audited crypto assets.\n")


if __name__ == "__main__":
    main()