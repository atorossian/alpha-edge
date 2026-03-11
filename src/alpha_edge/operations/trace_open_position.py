from __future__ import annotations

import argparse
import json
import datetime as dt
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, List, Optional

import boto3
import pandas as pd


BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"
TRADES_TABLE = "trades"

QTY_EPS = 1e-9
CLOSE_QTY_EPS = 5e-8
LOT_EPS = 1e-9

_FX_BASES = {
    "EUR", "USD", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD",
    "SEK", "NOK", "DKK", "CNH", "HKD", "SGD", "MXN", "ZAR",
}
_CRYPTO_BASES = {
    "BTC", "ETH", "ADA", "XRP", "DOT", "BCH", "LTC", "SOL",
    "DOGE", "SUI", "HBAR", "DASH", "BNB", "AVAX", "LINK",
    "MATIC", "ATOM", "NEAR", "UNI", "AAVE", "TRX", "ETC",
    "QTUM",
}
_CRYPTO_QUOTES = {"USD", "USDT", "USDC", "EUR"}


# ----------------------------
# Helpers
# ----------------------------
def _snap(x: float, eps: float = QTY_EPS) -> float:
    return 0.0 if abs(float(x)) < eps else float(x)


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
    if s in {
        "btc", "eth", "sol", "ada", "xrp", "dot", "ltc", "bnb",
        "avax", "link", "matic", "atom", "near", "uni", "aave",
        "trx", "etc", "doge", "hbar", "sui", "dash", "bch", "qtum",
    }:
        return "coins"
    if s in {"derivative", "derivatives"}:
        return "derivative"
    return s


def _is_fx_pair(ticker: str) -> bool:
    t = str(ticker).upper().strip().replace("/", "-")
    if "-" not in t:
        return False
    base, quote = t.split("-", 1)
    return base in _FX_BASES and quote in _FX_BASES


def _is_crypto_pair(ticker: str) -> bool:
    t = str(ticker).upper().strip().replace("/", "-")
    if "-" not in t:
        return False
    base, quote = t.split("-", 1)
    return base in _CRYPTO_BASES and quote in _CRYPTO_QUOTES


def _route_asset_side(*, ticker: str, quantity_unit: Optional[str]) -> str:
    t = str(ticker).upper().strip()
    if _is_fx_pair(t):
        return "NOTIONAL"
    return "SPOT"


def _action_pri(tag: str | None) -> int:
    return 0 if tag in {"open", "add"} else 1


def _side_pri(side: str) -> int:
    return 0 if side == "SELL" else 1


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
# Data models
# ----------------------------
@dataclass
class TradeRow:
    trade_id: str
    as_of: str
    ts_utc: str
    asset_id: str
    ticker: str
    side: str
    action_tag: str
    quantity: float
    price: float
    currency: str
    quantity_unit: Optional[str]
    value: Optional[float]
    reported_pnl: Optional[float]
    s3_key: str


@dataclass
class OpenQtyLot:
    trade_id: str
    as_of: str
    ts_utc: str
    asset_id: str
    ticker: str
    side: str
    action_tag: str
    quantity_open: float
    quantity_remaining: float
    price: float
    value: Optional[float]
    quantity_unit: Optional[str]


@dataclass
class OpenValueLot:
    trade_id: str
    as_of: str
    ts_utc: str
    asset_id: str
    ticker: str
    side: str
    action_tag: str
    value_open: float
    value_remaining: float
    price: float
    quantity: Optional[float]
    quantity_unit: Optional[str]


# ----------------------------
# Loading / normalization
# ----------------------------
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


def _normalize_trade(t: dict) -> TradeRow:
    trade_id = str(t.get("trade_id"))
    as_of = str(t.get("as_of"))
    ts_utc = str(t.get("ts_utc"))
    ticker = str(t.get("ticker")).upper().strip()
    side = str(t.get("side")).upper().strip()
    qty = float(t.get("quantity"))
    price = float(t.get("price"))
    ccy = str(t.get("currency") or "USD").upper().strip()

    asset_id = t.get("asset_id", None)
    asset_id = None if asset_id is None else str(asset_id).strip()
    if not asset_id:
        raise ValueError(f"Trade {trade_id} missing asset_id")

    if side not in ("BUY", "SELL"):
        raise ValueError(f"Trade {trade_id}: invalid side={side}")
    if qty <= 0 or price <= 0:
        raise ValueError(f"Trade {trade_id}: qty and price must be > 0")

    _ = pd.Timestamp(as_of).date()
    _ = _parse_ts(ts_utc)

    action_tag = (t.get("action_tag") or None)
    action_tag = None if action_tag is None else str(action_tag).strip().lower()
    if action_tag not in {"open", "close", "add", "reduce"}:
        raise ValueError(f"Trade {trade_id}: invalid action_tag={action_tag}")

    quantity_unit = _normalize_unit(t.get("quantity_unit") or None)

    value = t.get("value", None)
    reported_pnl = t.get("reported_pnl", None)

    try:
        value = None if value is None else float(value)
    except Exception:
        value = None

    try:
        reported_pnl = None if reported_pnl is None else float(reported_pnl)
    except Exception:
        reported_pnl = None

    return TradeRow(
        trade_id=trade_id,
        as_of=as_of,
        ts_utc=ts_utc,
        asset_id=asset_id,
        ticker=ticker,
        side=side,
        action_tag=action_tag,
        quantity=qty,
        price=price,
        currency=ccy,
        quantity_unit=quantity_unit,
        value=value,
        reported_pnl=reported_pnl,
        s3_key=str(t.get("_s3_key") or ""),
    )


# ----------------------------
# Trace engine
# ----------------------------
def trace_open_position(
    trades: List[TradeRow],
    *,
    debug: bool = True,
) -> dict[str, Any]:
    spot_long_lots: Dict[str, Deque[OpenQtyLot]] = {}
    spot_short_lots: Dict[str, Deque[OpenQtyLot]] = {}
    notional_long_lots: Dict[str, Deque[OpenValueLot]] = {}
    notional_short_lots: Dict[str, Deque[OpenValueLot]] = {}

    events: list[dict[str, Any]] = []

    def get_spot_long(asset_id: str) -> Deque[OpenQtyLot]:
        if asset_id not in spot_long_lots:
            spot_long_lots[asset_id] = deque()
        return spot_long_lots[asset_id]

    def get_spot_short(asset_id: str) -> Deque[OpenQtyLot]:
        if asset_id not in spot_short_lots:
            spot_short_lots[asset_id] = deque()
        return spot_short_lots[asset_id]

    def get_notional_long(asset_id: str) -> Deque[OpenValueLot]:
        if asset_id not in notional_long_lots:
            notional_long_lots[asset_id] = deque()
        return notional_long_lots[asset_id]

    def get_notional_short(asset_id: str) -> Deque[OpenValueLot]:
        if asset_id not in notional_short_lots:
            notional_short_lots[asset_id] = deque()
        return notional_short_lots[asset_id]

    norm = sorted(
        trades,
        key=lambda t: (
            _parse_ts(t.ts_utc),
            _action_pri(t.action_tag),
            _side_pri(t.side),
            t.trade_id,
        ),
    )

    for t in norm:
        route = _route_asset_side(ticker=t.ticker, quantity_unit=t.quantity_unit)

        event: dict[str, Any] = {
            "trade_id": t.trade_id,
            "as_of": t.as_of,
            "ts_utc": t.ts_utc,
            "asset_id": t.asset_id,
            "ticker": t.ticker,
            "side": t.side,
            "action_tag": t.action_tag,
            "quantity": t.quantity,
            "price": t.price,
            "value": t.value,
            "quantity_unit": t.quantity_unit,
            "route": route,
            "consumed_from": [],
            "status": "OK",
        }

        if route == "NOTIONAL":
            if t.value is None:
                event["status"] = "ERROR"
                event["error"] = "NOTIONAL trade missing value"
                events.append(event)
                continue

            if t.side == "BUY" and t.action_tag in {"open", "add"}:
                lots = get_notional_long(t.asset_id)
                lots.append(
                    OpenValueLot(
                        trade_id=t.trade_id,
                        as_of=t.as_of,
                        ts_utc=t.ts_utc,
                        asset_id=t.asset_id,
                        ticker=t.ticker,
                        side=t.side,
                        action_tag=t.action_tag,
                        value_open=float(t.value),
                        value_remaining=float(t.value),
                        price=t.price,
                        quantity=t.quantity,
                        quantity_unit=t.quantity_unit,
                    )
                )
                events.append(event)
                continue

            if t.side == "SELL" and t.action_tag in {"close", "reduce"}:
                lots = get_notional_long(t.asset_id)
                remaining = float(t.value)

                while remaining > LOT_EPS and lots:
                    lot = lots[0]
                    close_value = min(remaining, lot.value_remaining)
                    lot.value_remaining = _snap(lot.value_remaining - close_value, LOT_EPS)
                    remaining = _snap(remaining - close_value, LOT_EPS)

                    event["consumed_from"].append(
                        {
                            "open_trade_id": lot.trade_id,
                            "open_as_of": lot.as_of,
                            "open_price": lot.price,
                            "value_consumed": close_value,
                            "value_left_in_lot": lot.value_remaining,
                        }
                    )

                    if lot.value_remaining <= LOT_EPS:
                        lots.popleft()

                if remaining > LOT_EPS:
                    event["status"] = "ERROR"
                    event["error"] = f"SELL close/reduce exceeds long exposure; remaining_notional={remaining:.6f}"

                events.append(event)
                continue

            if t.side == "SELL" and t.action_tag in {"open", "add"}:
                lots = get_notional_short(t.asset_id)
                lots.append(
                    OpenValueLot(
                        trade_id=t.trade_id,
                        as_of=t.as_of,
                        ts_utc=t.ts_utc,
                        asset_id=t.asset_id,
                        ticker=t.ticker,
                        side=t.side,
                        action_tag=t.action_tag,
                        value_open=float(t.value),
                        value_remaining=float(t.value),
                        price=t.price,
                        quantity=t.quantity,
                        quantity_unit=t.quantity_unit,
                    )
                )
                events.append(event)
                continue

            if t.side == "BUY" and t.action_tag in {"close", "reduce"}:
                lots = get_notional_short(t.asset_id)
                remaining = float(t.value)

                while remaining > LOT_EPS and lots:
                    lot = lots[0]
                    close_value = min(remaining, lot.value_remaining)
                    lot.value_remaining = _snap(lot.value_remaining - close_value, LOT_EPS)
                    remaining = _snap(remaining - close_value, LOT_EPS)

                    event["consumed_from"].append(
                        {
                            "open_trade_id": lot.trade_id,
                            "open_as_of": lot.as_of,
                            "open_price": lot.price,
                            "value_consumed": close_value,
                            "value_left_in_lot": lot.value_remaining,
                        }
                    )

                    if lot.value_remaining <= LOT_EPS:
                        lots.popleft()

                if remaining > LOT_EPS:
                    event["status"] = "ERROR"
                    event["error"] = f"BUY close/reduce exceeds short exposure; remaining_notional={remaining:.6f}"

                events.append(event)
                continue

            event["status"] = "ERROR"
            event["error"] = "Unsupported NOTIONAL combination"
            events.append(event)
            continue

        # SPOT branch
        if t.side == "BUY" and t.action_tag in {"open", "add"}:
            lots = get_spot_long(t.asset_id)
            lots.append(
                OpenQtyLot(
                    trade_id=t.trade_id,
                    as_of=t.as_of,
                    ts_utc=t.ts_utc,
                    asset_id=t.asset_id,
                    ticker=t.ticker,
                    side=t.side,
                    action_tag=t.action_tag,
                    quantity_open=float(t.quantity),
                    quantity_remaining=float(t.quantity),
                    price=t.price,
                    value=t.value,
                    quantity_unit=t.quantity_unit,
                )
            )
            events.append(event)
            continue

        if t.side == "SELL" and t.action_tag in {"close", "reduce"}:
            lots = get_spot_long(t.asset_id)
            remaining = float(t.quantity)

            while remaining > QTY_EPS and lots:
                lot = lots[0]
                close_qty = min(remaining, lot.quantity_remaining)
                lot.quantity_remaining = _snap(lot.quantity_remaining - close_qty, QTY_EPS)
                remaining = _snap(remaining - close_qty, QTY_EPS)

                event["consumed_from"].append(
                    {
                        "open_trade_id": lot.trade_id,
                        "open_as_of": lot.as_of,
                        "open_price": lot.price,
                        "qty_consumed": close_qty,
                        "qty_left_in_lot": lot.quantity_remaining,
                    }
                )

                if lot.quantity_remaining <= QTY_EPS:
                    lots.popleft()

            remaining = _snap(remaining, CLOSE_QTY_EPS)
            if remaining > CLOSE_QTY_EPS:
                event["status"] = "ERROR"
                event["error"] = f"SELL close/reduce exceeds long exposure; remaining_qty={remaining:.12f}"

            events.append(event)
            continue

        if t.side == "SELL" and t.action_tag in {"open", "add"}:
            lots = get_spot_short(t.asset_id)
            lots.append(
                OpenQtyLot(
                    trade_id=t.trade_id,
                    as_of=t.as_of,
                    ts_utc=t.ts_utc,
                    asset_id=t.asset_id,
                    ticker=t.ticker,
                    side=t.side,
                    action_tag=t.action_tag,
                    quantity_open=float(t.quantity),
                    quantity_remaining=float(t.quantity),
                    price=t.price,
                    value=t.value,
                    quantity_unit=t.quantity_unit,
                )
            )
            events.append(event)
            continue

        if t.side == "BUY" and t.action_tag in {"close", "reduce"}:
            lots = get_spot_short(t.asset_id)
            remaining = float(t.quantity)

            while remaining > QTY_EPS and lots:
                lot = lots[0]
                close_qty = min(remaining, lot.quantity_remaining)
                lot.quantity_remaining = _snap(lot.quantity_remaining - close_qty, QTY_EPS)
                remaining = _snap(remaining - close_qty, QTY_EPS)

                event["consumed_from"].append(
                    {
                        "open_trade_id": lot.trade_id,
                        "open_as_of": lot.as_of,
                        "open_price": lot.price,
                        "qty_consumed": close_qty,
                        "qty_left_in_lot": lot.quantity_remaining,
                    }
                )

                if lot.quantity_remaining <= QTY_EPS:
                    lots.popleft()

            remaining = _snap(remaining, CLOSE_QTY_EPS)
            if remaining > CLOSE_QTY_EPS:
                event["status"] = "ERROR"
                event["error"] = f"BUY close/reduce exceeds short exposure; remaining_qty={remaining:.12f}"

            events.append(event)
            continue

        event["status"] = "ERROR"
        event["error"] = "Unsupported SPOT combination"
        events.append(event)

    remaining_spot_long = []
    for asset_id, lots in spot_long_lots.items():
        for lot in lots:
            remaining_spot_long.append(asdict(lot))

    remaining_spot_short = []
    for asset_id, lots in spot_short_lots.items():
        for lot in lots:
            remaining_spot_short.append(asdict(lot))

    remaining_notional_long = []
    for asset_id, lots in notional_long_lots.items():
        for lot in lots:
            remaining_notional_long.append(asdict(lot))

    remaining_notional_short = []
    for asset_id, lots in notional_short_lots.items():
        for lot in lots:
            remaining_notional_short.append(asdict(lot))

    return {
        "events": events,
        "remaining_spot_long": remaining_spot_long,
        "remaining_spot_short": remaining_spot_short,
        "remaining_notional_long": remaining_notional_long,
        "remaining_notional_short": remaining_notional_short,
    }


# ----------------------------
# CLI / rendering
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Trace which trade(s) still leave an open position.")
    ap.add_argument("--ticker", default=None, help="Filter by ticker, e.g. SOL-USD or VXQ4")
    ap.add_argument("--asset-id", default=None, help="Filter by asset_id")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--json-out", default=None, help="Optional path to save full trace as JSON")
    args = ap.parse_args()

    if not args.ticker and not args.asset_id:
        raise ValueError("Provide at least --ticker or --asset-id")

    s3 = s3_client(REGION)
    raw = _load_trades(s3, start=args.start, end=args.end)
    trades: list[TradeRow] = []

    for r in raw:
        try:
            t = _normalize_trade(r)
        except Exception:
            continue

        if args.ticker and t.ticker != str(args.ticker).upper().strip():
            continue
        if args.asset_id and t.asset_id != str(args.asset_id).strip():
            continue
        trades.append(t)

    if not trades:
        print("No matching trades found.")
        return

    trace = trace_open_position(trades)

    print("\n=== MATCHED TRADES ===")
    for t in sorted(trades, key=lambda x: (_parse_ts(x.ts_utc), _action_pri(x.action_tag), _side_pri(x.side), x.trade_id)):
        print(
            f"{t.as_of} {t.ts_utc} | {t.trade_id} | {t.ticker} | "
            f"{t.side} {t.action_tag} | qty={t.quantity} | px={t.price} | val={t.value} | unit={t.quantity_unit}"
        )

    print("\n=== TRACE EVENTS ===")
    for e in trace["events"]:
        print(
            f"{e['trade_id']} | route={e['route']} | {e['side']} {e['action_tag']} | "
            f"qty={e['quantity']} | px={e['price']} | val={e['value']} | status={e['status']}"
        )
        if e["consumed_from"]:
            for c in e["consumed_from"]:
                if "qty_consumed" in c:
                    print(
                        f"    consumes open_trade={c['open_trade_id']} | "
                        f"qty_consumed={c['qty_consumed']} | qty_left_in_lot={c['qty_left_in_lot']}"
                    )
                else:
                    print(
                        f"    consumes open_trade={c['open_trade_id']} | "
                        f"value_consumed={c['value_consumed']} | value_left_in_lot={c['value_left_in_lot']}"
                    )
        if e["status"] != "OK":
            print(f"    ERROR: {e['error']}")

    print("\n=== REMAINING OPEN LOTS ===")
    for section in [
        "remaining_spot_long",
        "remaining_spot_short",
        "remaining_notional_long",
        "remaining_notional_short",
    ]:
        rows = trace[section]
        print(f"\n{section}: {len(rows)}")
        for r in rows:
            if "quantity_remaining" in r:
                print(
                    f"  open_trade={r['trade_id']} | {r['ticker']} | "
                    f"qty_open={r['quantity_open']} | qty_remaining={r['quantity_remaining']} | "
                    f"px={r['price']} | side={r['side']} {r['action_tag']}"
                )
            else:
                print(
                    f"  open_trade={r['trade_id']} | {r['ticker']} | "
                    f"value_open={r['value_open']} | value_remaining={r['value_remaining']} | "
                    f"px={r['price']} | side={r['side']} {r['action_tag']}"
                )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)
        print(f"\n[OK] Wrote JSON trace to {args.json_out}")


if __name__ == "__main__":
    main()