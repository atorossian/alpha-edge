from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd


BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"
TRADES_TABLE = "trades"

QTY_DECIMALS = 8
VALUE_DECIMALS = 2
QTY_TOL = 1e-8
VALUE_TOL = 0.01
INTEGER_QTY_TOL = 1e-6

# Explicit exceptions for equities that may legitimately be fractional
FRACTIONAL_EQUITY_TICKERS = {
    "AI.PA",
}
FRACTIONAL_EQUITY_ASSET_IDS = {
    # "EQHxxxxxxxxxxxxxxxxxxx",
}

FX_BASES = {
    "EUR", "USD", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD",
    "SEK", "NOK", "DKK", "CNH", "HKD", "SGD", "MXN", "ZAR",
}
CRYPTO_BASES = {
    "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "LTC", "BNB",
    "AVAX", "LINK", "MATIC", "ATOM", "NEAR", "UNI", "AAVE",
    "TRX", "ETC", "DOGE", "HBAR", "SUI", "DASH", "BCH", "QTUM",
    "APT", "ARB", "INJ", "MANA", "NEO", "RENDER",
}
CRYPTO_QUOTES = {"USD", "USDT", "USDC", "EUR"}


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(region: str = REGION):
    return boto3.client("s3", region_name=region)


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def dt_key(table: str, dt_str: str, filename: str) -> str:
    return engine_key(table, f"dt={dt_str}", filename)


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


def s3_get_json_optional(s3, *, bucket: str, key: str) -> Optional[dict]:
    try:
        return s3_get_json(s3, bucket=bucket, key=key)
    except Exception:
        return None


# ----------------------------
# Helpers
# ----------------------------
def _round_qty(x: float, decimals: int = QTY_DECIMALS) -> float:
    return round(float(x), decimals)


def _round_value(x: float, decimals: int = VALUE_DECIMALS) -> float:
    return round(float(x), decimals)


def _is_finite_positive(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v) and v > 0.0
    except Exception:
        return False


def _is_integer_like(x: Any, tol: float = INTEGER_QTY_TOL) -> bool:
    try:
        v = float(x)
        return math.isfinite(v) and v > 0.0 and abs(v - round(v)) <= tol
    except Exception:
        return False


def _parse_ts(ts_utc: str) -> pd.Timestamp:
    t = pd.to_datetime(ts_utc, errors="coerce", utc=True)
    if pd.isna(t):
        raise ValueError(f"Invalid ts_utc: {ts_utc}")
    return t


def _normalize_quantity_unit(x: Optional[str], ticker: str) -> Optional[str]:
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
        "btc": "coins",
        "eth": "coins",
        "sol": "coins",
        "ada": "coins",
        "xrp": "coins",
        "dot": "coins",
        "ltc": "coins",
        "bnb": "coins",
        "avax": "coins",
        "link": "coins",
        "matic": "coins",
        "atom": "coins",
        "near": "coins",
        "uni": "coins",
        "aave": "coins",
        "trx": "coins",
        "etc": "coins",
        "doge": "coins",
        "hbar": "coins",
        "sui": "coins",
        "dash": "coins",
        "bch": "coins",
        "qtum": "coins",
        "apt": "coins",
        "arb": "coins",
        "inj": "coins",
        "mana": "coins",
        "neo": "coins",
        "render": "coins",
        "derivative": "derivative",
        "derivatives": "derivative",
    }
    out = unit_map.get(s, s)

    t = str(ticker).upper().strip().replace("/", "-")
    if "-" in t:
        base = t.split("-", 1)[0]
        if base in CRYPTO_BASES:
            return "coins"

    return out


def _infer_quantity_from_value_price(value: float, price: float) -> float:
    if not _is_finite_positive(value):
        raise ValueError(f"Cannot infer quantity: invalid value={value!r}")
    if not _is_finite_positive(price):
        raise ValueError(f"Cannot infer quantity: invalid price={price!r}")
    return float(value) / float(price)


def _is_crypto_pair(ticker: str) -> bool:
    t = str(ticker).upper().strip().replace("/", "-")
    if "-" not in t:
        return False
    base, quote = t.split("-", 1)
    return base in CRYPTO_BASES and quote in CRYPTO_QUOTES


def _is_fx_pair(ticker: str) -> bool:
    t = str(ticker).upper().strip().replace("/", "-")
    if "-" not in t:
        return False
    base, quote = t.split("-", 1)
    return base in FX_BASES and quote in FX_BASES


def _quantity_policy(*, ticker: str, asset_id: str, quantity_unit: Optional[str]) -> str:
    """
    Returns:
      - 'fractional' for crypto / FX / derivatives / explicit exceptions
      - 'integer' for normal equities / ETFs
    """
    t = str(ticker).upper().strip()
    aid = str(asset_id or "").strip()
    unit = str(quantity_unit or "").strip().lower()

    if aid in FRACTIONAL_EQUITY_ASSET_IDS or t in FRACTIONAL_EQUITY_TICKERS:
        return "fractional"

    if _is_crypto_pair(t):
        return "fractional"

    if _is_fx_pair(t):
        return "fractional"

    if unit in {"coins", "contracts", "derivative", "derivatives"}:
        return "fractional"

    return "integer"


def _normalize_repaired_quantity(
    *,
    raw_qty: float,
    policy: str,
) -> tuple[float, bool, str]:
    if not math.isfinite(raw_qty) or raw_qty <= 0.0:
        return raw_qty, False, "raw quantity is invalid"

    if policy == "fractional":
        return _round_qty(raw_qty), True, "fractional quantity allowed"

    nearest = round(raw_qty)
    if abs(raw_qty - nearest) <= INTEGER_QTY_TOL:
        return float(nearest), True, "integer quantity snapped"

    return _round_qty(raw_qty), False, "materially fractional quantity for integer-only instrument"


def _action_pri(tag: str | None) -> int:
    return 0 if tag in {"open", "add"} else 1


def _side_pri(side: str) -> int:
    return 0 if side == "SELL" else 1


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
    quantity: Optional[float]
    price: Optional[float]
    value: Optional[float]
    reported_pnl: Optional[float]
    quantity_unit: Optional[str]
    s3_key: str


@dataclass
class RepairRow:
    trade_id: str
    as_of: str
    ts_utc: str
    asset_id: str
    ticker: str
    side: str
    action_tag: str
    old_quantity: Optional[float]
    new_quantity: Optional[float]
    old_price: Optional[float]
    new_price: Optional[float]
    old_value: Optional[float]
    new_value: Optional[float]
    old_quantity_unit: Optional[str]
    new_quantity_unit: Optional[str]
    old_reported_pnl: Optional[float]
    new_reported_pnl: Optional[float]
    quantity_policy: Optional[str]
    repair_action: str
    repair_reason: str
    command: str
    s3_key: str


# ----------------------------
# Trade loading
# ----------------------------
def _load_all_trades(s3, *, bucket: str) -> List[dict]:
    prefix = engine_key(TRADES_TABLE, "dt=")
    keys = s3_list_keys(s3, bucket=bucket, prefix=prefix)
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]

    out: List[dict] = []
    for k in keys:
        payload = s3_get_json_optional(s3, bucket=bucket, key=k)
        if isinstance(payload, dict):
            payload["_s3_key"] = k
            out.append(payload)
    return out


def _normalize_trade(t: dict) -> TradeRow:
    trade_id = str(t.get("trade_id") or "").strip()
    as_of = str(t.get("as_of") or "").strip()
    ts_utc = str(t.get("ts_utc") or "").strip()
    asset_id = str(t.get("asset_id") or "").strip()
    ticker = str(t.get("ticker") or "").upper().strip()
    side = str(t.get("side") or "").upper().strip()
    action_tag = str(t.get("action_tag") or "").strip().lower()
    quantity = None if t.get("quantity") is None else float(t.get("quantity"))
    price = None if t.get("price") is None else float(t.get("price"))
    value = None if t.get("value") is None else float(t.get("value"))
    reported_pnl = None if t.get("reported_pnl") is None else float(t.get("reported_pnl"))
    quantity_unit = _normalize_quantity_unit(t.get("quantity_unit"), ticker=ticker)
    s3_key = str(t.get("_s3_key") or "").strip()

    if not trade_id or not as_of or not ts_utc or not asset_id or not ticker or not side:
        raise ValueError(f"Bad trade payload: missing core fields trade_id={trade_id!r}")

    return TradeRow(
        trade_id=trade_id,
        as_of=as_of,
        ts_utc=ts_utc,
        asset_id=asset_id,
        ticker=ticker,
        side=side,
        action_tag=action_tag,
        quantity=quantity,
        price=price,
        value=value,
        reported_pnl=reported_pnl,
        quantity_unit=quantity_unit,
        s3_key=s3_key,
    )


# ----------------------------
# Command building
# ----------------------------
def _build_edit_command_args(row: RepairRow, python_entrypoint: str, dry_run: bool) -> List[str]:
    parts = [
        "poetry",
        "run",
        "python",
        python_entrypoint,
        "--mode", "edit",
        "--trade-id", str(row.trade_id),
        "--old-as-of", str(row.as_of),
    ]

    if row.new_quantity is not None and (
        row.old_quantity is None or abs(float(row.new_quantity) - float(row.old_quantity)) > QTY_TOL
    ):
        parts.extend(["--quantity", str(row.new_quantity)])

    if row.new_price is not None and row.old_price != row.new_price:
        parts.extend(["--price", str(row.new_price)])

    if row.new_value is not None and (
        row.old_value is None or abs(float(row.new_value) - float(row.old_value)) > VALUE_TOL
    ):
        parts.extend(["--value", str(row.new_value)])

    if row.new_quantity_unit is not None and row.old_quantity_unit != row.new_quantity_unit:
        parts.extend(["--quantity-unit", str(row.new_quantity_unit)])

    if row.new_reported_pnl is not None and row.old_reported_pnl != row.new_reported_pnl:
        parts.extend(["--reported-pnl", str(row.new_reported_pnl)])

    if dry_run:
        parts.append("--dry-run")

    return parts


def _command_args_to_text(args: List[str]) -> str:
    def q(x: str) -> str:
        if any(ch in x for ch in [' ', '"', "'"]):
            return '"' + x.replace('"', '\\"') + '"'
        return x
    return " ".join(q(x) for x in args)


# ----------------------------
# Manual row helper
# ----------------------------
def _manual_row(
    t: TradeRow,
    reason: str,
    *,
    new_qty: Optional[float] = None,
    new_val: Optional[float] = None,
    quantity_policy: Optional[str] = None,
) -> RepairRow:
    return RepairRow(
        trade_id=t.trade_id,
        as_of=t.as_of,
        ts_utc=t.ts_utc,
        asset_id=t.asset_id,
        ticker=t.ticker,
        side=t.side,
        action_tag=t.action_tag,
        old_quantity=t.quantity,
        new_quantity=new_qty,
        old_price=t.price,
        new_price=t.price,
        old_value=t.value,
        new_value=new_val if new_val is not None else t.value,
        old_quantity_unit=t.quantity_unit,
        new_quantity_unit=_normalize_quantity_unit(t.quantity_unit, ticker=t.ticker),
        old_reported_pnl=t.reported_pnl,
        new_reported_pnl=t.reported_pnl,
        quantity_policy=quantity_policy,
        repair_action="MANUAL_REVIEW",
        repair_reason=reason,
        command="",
        s3_key=t.s3_key,
    )


# ----------------------------
# Generic chain-aware repair
# ----------------------------
def _process_asset_chain(trades: List[TradeRow]) -> tuple[list[RepairRow], list[RepairRow]]:
    """
    Universal quantity-based repair logic.

    LONG:
      BUY open/add    -> increases long exposure
      SELL reduce     -> reduces long partially
      SELL close      -> closes remaining long

    SHORT:
      SELL open/add   -> increases short exposure
      BUY reduce      -> reduces short partially
      BUY close       -> closes remaining short
    """
    auto_rows: list[RepairRow] = []
    manual_rows: list[RepairRow] = []

    trades = sorted(
        trades,
        key=lambda t: (
            _parse_ts(t.ts_utc),
            _action_pri(t.action_tag),
            _side_pri(t.side),
            t.trade_id,
        ),
    )

    open_long_qty = 0.0
    open_short_qty = 0.0

    for t in trades:
        old_qty = t.quantity
        old_px = t.price
        old_val = t.value
        old_unit = t.quantity_unit
        old_rpnl = t.reported_pnl

        new_qty = old_qty
        new_px = old_px
        new_val = old_val
        new_unit = _normalize_quantity_unit(old_unit, ticker=t.ticker)
        new_rpnl = old_rpnl
        reason_parts: list[str] = []

        quantity_policy = _quantity_policy(
            ticker=t.ticker,
            asset_id=t.asset_id,
            quantity_unit=new_unit,
        )

        if not _is_finite_positive(old_px):
            manual_rows.append(_manual_row(t, "Missing/invalid price", quantity_policy=quantity_policy))
            continue

        # ----------------------------
        # LONG OPEN / ADD
        # ----------------------------
        if t.side == "BUY" and t.action_tag in {"open", "add"}:
            if open_short_qty > QTY_TOL:
                manual_rows.append(
                    _manual_row(
                        t,
                        "BUY open/add encountered while short chain is still open",
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            # Integer instruments: trust stored integer quantity first
            if quantity_policy == "integer" and _is_integer_like(old_qty):
                new_qty = float(round(float(old_qty)))
                new_val = _round_value(float(new_qty) * float(old_px))
                open_long_qty = _round_qty(open_long_qty + float(new_qty))
                reason_parts.append("BUY open/add => trusted stored integer quantity")
                reason_parts.append("value := quantity * price")
            else:
                if not _is_finite_positive(old_val):
                    manual_rows.append(
                        _manual_row(
                            t,
                            "BUY open/add missing/invalid value",
                            quantity_policy=quantity_policy,
                        )
                    )
                    continue

                raw_qty = _infer_quantity_from_value_price(float(old_val), float(old_px))
                norm_qty, ok, qty_detail = _normalize_repaired_quantity(
                    raw_qty=raw_qty,
                    policy=quantity_policy,
                )
                if not ok:
                    manual_rows.append(
                        _manual_row(
                            t,
                            qty_detail,
                            new_qty=norm_qty,
                            new_val=_round_value(float(norm_qty) * float(old_px)),
                            quantity_policy=quantity_policy,
                        )
                    )
                    continue

                new_qty = norm_qty
                new_val = _round_value(float(new_qty) * float(old_px))
                open_long_qty = _round_qty(open_long_qty + float(new_qty))
                reason_parts.append("BUY open/add => quantity := value / price")
                reason_parts.append("value := quantity * price")

        # ----------------------------
        # LONG REDUCE / CLOSE
        # ----------------------------
        elif t.side == "SELL" and t.action_tag == "close":
            if open_short_qty > QTY_TOL:
                manual_rows.append(
                    _manual_row(
                        t,
                        "SELL close encountered while short chain is still open",
                        quantity_policy=quantity_policy,
                    )
                )
                continue
            if open_long_qty <= QTY_TOL:
                manual_rows.append(
                    _manual_row(
                        t,
                        "SELL close with no remaining open long quantity",
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            raw_qty = float(open_long_qty)
            norm_qty, ok, qty_detail = _normalize_repaired_quantity(
                raw_qty=raw_qty,
                policy=quantity_policy,
            )
            if not ok:
                manual_rows.append(
                    _manual_row(
                        t,
                        f"Close quantity invalid under policy: {qty_detail}",
                        new_qty=norm_qty,
                        new_val=_round_value(float(norm_qty) * float(old_px)),
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            new_qty = norm_qty
            new_val = _round_value(float(new_qty) * float(old_px))
            open_long_qty = 0.0
            reason_parts.append("SELL close => quantity := remaining open long qty")
            reason_parts.append("value := quantity * price")

        elif t.side == "SELL" and t.action_tag == "reduce":
            if open_short_qty > QTY_TOL:
                manual_rows.append(
                    _manual_row(
                        t,
                        "SELL reduce encountered while short chain is still open",
                        quantity_policy=quantity_policy,
                    )
                )
                continue
            if open_long_qty <= QTY_TOL:
                manual_rows.append(
                    _manual_row(
                        t,
                        "SELL reduce with no remaining open long quantity",
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            inferred_qty = None
            if _is_finite_positive(old_val):
                inferred_qty = _infer_quantity_from_value_price(float(old_val), float(old_px))

            candidate_qty = None
            if quantity_policy == "integer" and _is_integer_like(old_qty):
                candidate_qty = float(round(float(old_qty)))
            elif old_qty is not None and old_qty > 0.0:
                candidate_qty = float(old_qty)
            elif inferred_qty is not None:
                candidate_qty = float(inferred_qty)

            if candidate_qty is None:
                manual_rows.append(
                    _manual_row(
                        t,
                        "SELL reduce missing both usable quantity and value",
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            raw_qty = min(float(candidate_qty), float(open_long_qty))
            norm_qty, ok, qty_detail = _normalize_repaired_quantity(
                raw_qty=raw_qty,
                policy=quantity_policy,
            )
            if not ok:
                manual_rows.append(
                    _manual_row(
                        t,
                        qty_detail,
                        new_qty=norm_qty,
                        new_val=_round_value(float(norm_qty) * float(old_px)),
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            new_qty = norm_qty
            new_val = _round_value(float(new_qty) * float(old_px))
            open_long_qty = _round_qty(open_long_qty - float(new_qty))
            reason_parts.append("SELL reduce => quantity := min(candidate qty, remaining open long qty)")
            reason_parts.append("value := quantity * price")

        # ----------------------------
        # SHORT OPEN / ADD
        # ----------------------------
        elif t.side == "SELL" and t.action_tag in {"open", "add"}:
            if open_long_qty > QTY_TOL:
                manual_rows.append(
                    _manual_row(
                        t,
                        "SELL open/add encountered while long chain is still open",
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            # Integer instruments: trust stored integer quantity first
            if quantity_policy == "integer" and _is_integer_like(old_qty):
                new_qty = float(round(float(old_qty)))
                new_val = _round_value(float(new_qty) * float(old_px))
                open_short_qty = _round_qty(open_short_qty + float(new_qty))
                reason_parts.append("SELL open/add => trusted stored integer quantity")
                reason_parts.append("value := quantity * price")
            else:
                if not _is_finite_positive(old_val):
                    manual_rows.append(
                        _manual_row(
                            t,
                            "SELL open/add missing/invalid value",
                            quantity_policy=quantity_policy,
                        )
                    )
                    continue

                raw_qty = _infer_quantity_from_value_price(float(old_val), float(old_px))
                norm_qty, ok, qty_detail = _normalize_repaired_quantity(
                    raw_qty=raw_qty,
                    policy=quantity_policy,
                )
                if not ok:
                    manual_rows.append(
                        _manual_row(
                            t,
                            qty_detail,
                            new_qty=norm_qty,
                            new_val=_round_value(float(norm_qty) * float(old_px)),
                            quantity_policy=quantity_policy,
                        )
                    )
                    continue

                new_qty = norm_qty
                new_val = _round_value(float(new_qty) * float(old_px))
                open_short_qty = _round_qty(open_short_qty + float(new_qty))
                reason_parts.append("SELL open/add => quantity := value / price")
                reason_parts.append("value := quantity * price")

        # ----------------------------
        # SHORT REDUCE / CLOSE
        # ----------------------------
        elif t.side == "BUY" and t.action_tag == "close":
            if open_long_qty > QTY_TOL:
                manual_rows.append(
                    _manual_row(
                        t,
                        "BUY close encountered while long chain is still open",
                        quantity_policy=quantity_policy,
                    )
                )
                continue
            if open_short_qty <= QTY_TOL:
                manual_rows.append(
                    _manual_row(
                        t,
                        "BUY close with no remaining open short quantity",
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            raw_qty = float(open_short_qty)
            norm_qty, ok, qty_detail = _normalize_repaired_quantity(
                raw_qty=raw_qty,
                policy=quantity_policy,
            )
            if not ok:
                manual_rows.append(
                    _manual_row(
                        t,
                        f"Close quantity invalid under policy: {qty_detail}",
                        new_qty=norm_qty,
                        new_val=_round_value(float(norm_qty) * float(old_px)),
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            new_qty = norm_qty
            new_val = _round_value(float(new_qty) * float(old_px))
            open_short_qty = 0.0
            reason_parts.append("BUY close => quantity := remaining open short qty")
            reason_parts.append("value := quantity * price")

        elif t.side == "BUY" and t.action_tag == "reduce":
            if open_long_qty > QTY_TOL:
                manual_rows.append(
                    _manual_row(
                        t,
                        "BUY reduce encountered while long chain is still open",
                        quantity_policy=quantity_policy,
                    )
                )
                continue
            if open_short_qty <= QTY_TOL:
                manual_rows.append(
                    _manual_row(
                        t,
                        "BUY reduce with no remaining open short quantity",
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            inferred_qty = None
            if _is_finite_positive(old_val):
                inferred_qty = _infer_quantity_from_value_price(float(old_val), float(old_px))

            candidate_qty = None
            if quantity_policy == "integer" and _is_integer_like(old_qty):
                candidate_qty = float(round(float(old_qty)))
            elif old_qty is not None and old_qty > 0.0:
                candidate_qty = float(old_qty)
            elif inferred_qty is not None:
                candidate_qty = float(inferred_qty)

            if candidate_qty is None:
                manual_rows.append(
                    _manual_row(
                        t,
                        "BUY reduce missing both usable quantity and value",
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            raw_qty = min(float(candidate_qty), float(open_short_qty))
            norm_qty, ok, qty_detail = _normalize_repaired_quantity(
                raw_qty=raw_qty,
                policy=quantity_policy,
            )
            if not ok:
                manual_rows.append(
                    _manual_row(
                        t,
                        qty_detail,
                        new_qty=norm_qty,
                        new_val=_round_value(float(norm_qty) * float(old_px)),
                        quantity_policy=quantity_policy,
                    )
                )
                continue

            new_qty = norm_qty
            new_val = _round_value(float(new_qty) * float(old_px))
            open_short_qty = _round_qty(open_short_qty - float(new_qty))
            reason_parts.append("BUY reduce => quantity := min(candidate qty, remaining open short qty)")
            reason_parts.append("value := quantity * price")

        else:
            manual_rows.append(
                _manual_row(
                    t,
                    "Unsupported chain semantic in automatic repair",
                    quantity_policy=quantity_policy,
                )
            )
            continue

        changed = False
        if old_qty is None or abs(float(new_qty) - float(old_qty)) > QTY_TOL:
            changed = True
        if new_unit != old_unit:
            changed = True
        if old_val is None or abs(float(new_val) - float(old_val)) > VALUE_TOL:
            changed = True

        auto_rows.append(
            RepairRow(
                trade_id=t.trade_id,
                as_of=t.as_of,
                ts_utc=t.ts_utc,
                asset_id=t.asset_id,
                ticker=t.ticker,
                side=t.side,
                action_tag=t.action_tag,
                old_quantity=old_qty,
                new_quantity=new_qty,
                old_price=old_px,
                new_price=new_px,
                old_value=old_val,
                new_value=new_val,
                old_quantity_unit=old_unit,
                new_quantity_unit=new_unit,
                old_reported_pnl=old_rpnl,
                new_reported_pnl=new_rpnl,
                quantity_policy=quantity_policy,
                repair_action="AUTO_PATCH" if changed else "NO_CHANGE",
                repair_reason="; ".join(reason_parts) if reason_parts else "no effective patch required",
                command="",
                s3_key=t.s3_key,
            )
        )

    return auto_rows, manual_rows


# ----------------------------
# Main orchestration
# ----------------------------
def repair_all_trade_chains(
    *,
    out_auto_csv: str,
    out_manual_csv: str,
    python_entrypoint: str,
    execute: bool = False,
    dry_run: bool = False,
    stop_on_error: bool = False,
    asset_ids: Optional[set[str]] = None,
    tickers: Optional[set[str]] = None,
) -> None:
    s3 = s3_client(REGION)
    raw_trades = _load_all_trades(s3, bucket=BUCKET)

    all_trades: list[TradeRow] = []
    for raw in raw_trades:
        try:
            t = _normalize_trade(raw)
            if asset_ids and t.asset_id not in asset_ids:
                continue
            if tickers and t.ticker not in tickers:
                continue
            all_trades.append(t)
        except Exception:
            continue

    grouped: dict[str, list[TradeRow]] = {}
    for t in all_trades:
        grouped.setdefault(t.asset_id, []).append(t)

    auto_rows: list[RepairRow] = []
    manual_rows: list[RepairRow] = []

    exec_ok = 0
    exec_fail = 0

    for asset_id, rows in sorted(grouped.items(), key=lambda kv: kv[0]):
        asset_auto, asset_manual = _process_asset_chain(rows)

        for row in asset_auto:
            if row.repair_action != "NO_CHANGE":
                cmd_args = _build_edit_command_args(row, python_entrypoint=python_entrypoint, dry_run=dry_run)
                row.command = _command_args_to_text(cmd_args)

                if execute:
                    print("\n=== EXECUTE TRADE CHAIN REPAIR ===")
                    print(row.command)
                    try:
                        subprocess.run(cmd_args, check=True)
                        exec_ok += 1
                    except subprocess.CalledProcessError as e:
                        exec_fail += 1
                        row.repair_action = "EXEC_FAILED"
                        row.repair_reason = f"{row.repair_reason}; execution failed rc={e.returncode}"
                        if stop_on_error:
                            auto_rows.extend(asset_auto)
                            manual_rows.extend(asset_manual)
                            pd.DataFrame([asdict(r) for r in auto_rows]).to_csv(out_auto_csv, index=False)
                            pd.DataFrame([asdict(r) for r in manual_rows]).to_csv(out_manual_csv, index=False)
                            raise

        auto_rows.extend(asset_auto)
        manual_rows.extend(asset_manual)

    pd.DataFrame([asdict(r) for r in auto_rows]).to_csv(out_auto_csv, index=False)
    pd.DataFrame([asdict(r) for r in manual_rows]).to_csv(out_manual_csv, index=False)

    print("\n=== ALL TRADE CHAIN REPAIR ===")
    print(f"assets:              {len(grouped)}")
    print(f"trades:              {len(all_trades)}")
    print(f"auto_rows:           {len(auto_rows)}")
    print(f"manual_rows:         {len(manual_rows)}")
    print(f"auto_csv:            {out_auto_csv}")
    print(f"manual_csv:          {out_manual_csv}")
    print(f"execute:             {execute}")
    print(f"dry_run:             {dry_run}")
    if execute:
        print(f"exec_ok:             {exec_ok}")
        print(f"exec_fail:           {exec_fail}")
    print("")


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Chain-aware quantity repair for all trade types."
    )
    ap.add_argument("--out-auto-csv", default="./data/repair_all_trade_chains_auto.csv")
    ap.add_argument("--out-manual-csv", default="./data/repair_all_trade_chains_manual.csv")
    ap.add_argument(
        "--python-entrypoint",
        default="src/alpha_edge/operations/record_trade.py",
        help="Path used in generated poetry edit commands.",
    )
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument(
        "--asset-ids",
        default=None,
        help='Optional comma-separated asset_id filter, e.g. "CRYPTO:BTC-USD,EQHxxxx"',
    )
    ap.add_argument(
        "--tickers",
        default=None,
        help='Optional comma-separated ticker filter, e.g. "BTC-USD,VXH6"',
    )
    args = ap.parse_args()

    asset_ids = None
    if args.asset_ids:
        asset_ids = {x.strip() for x in str(args.asset_ids).split(",") if x.strip()}

    tickers = None
    if args.tickers:
        tickers = {x.strip().upper() for x in str(args.tickers).split(",") if x.strip()}

    repair_all_trade_chains(
        out_auto_csv=str(args.out_auto_csv),
        out_manual_csv=str(args.out_manual_csv),
        python_entrypoint=str(args.python_entrypoint),
        execute=bool(args.execute),
        dry_run=bool(args.dry_run),
        stop_on_error=bool(args.stop_on_error),
        asset_ids=asset_ids,
        tickers=tickers,
    )


if __name__ == "__main__":
    main()