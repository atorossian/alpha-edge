from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import boto3
import pandas as pd
import yfinance as yf

from alpha_edge.core.market_store import MarketStore

BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"

TRADES_TABLE = "trades"
LEDGER_TABLE = "ledger"

# ----------------------------
# Numeric stability helpers
# ----------------------------
QTY_EPS = 1e-9
LOT_EPS = 1e-9
# NEW: position-level dust filters (presentation / hygiene)
POSITION_QTY_EPS = 1e-6        # shares/coins below this are dust
POSITION_NOTIONAL_EPS = 1e-4   # $0.0001 notional is dust

def _snap(x: float, eps: float = QTY_EPS) -> float:
    return 0.0 if abs(float(x)) < eps else float(x)

# Spot lot: (qty_signed, entry_price_usd)
Lot = Tuple[float, float]

# Notional lot: (notional_signed_usd, entry_price_usd)
ContractLot = Tuple[float, float]

def _clean_lots(lots: Deque[Lot], eps: float = LOT_EPS) -> Deque[Lot]:
    out: Deque[Lot] = deque()
    for q, px in lots:
        q = _snap(q, eps)
        if q == 0.0:
            continue
        px = float(px)

        if out:
            q0, px0 = out[-1]
            # merge only if exactly same price and same sign
            if (q0 > 0) == (q > 0) and abs(px0 - px) <= 0.0:
                out[-1] = (q0 + q, px0)
            else:
                out.append((q, px))
        else:
            out.append((q, px))
    return out

# ----------------------------
# S3 helpers
# ----------------------------
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

def s3_list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix)
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

def s3_get_parquet_df(s3, *, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return pd.read_parquet(io.BytesIO(data), engine="pyarrow")

# ----------------------------
# FX helpers
# ----------------------------
def _download_daily_fx_usd_per_ccy(ccy: str, start: str, end: str) -> pd.Series:
    ccy = ccy.upper().strip()
    if ccy in {"USD", "USDT", "USDC"}:
        return pd.Series(dtype=float)

    tkr = f"{ccy}USD=X"
    df = yf.download(
        tkr,
        start=start,
        end=end,
        interval="1d",
        progress=False,
        threads=False,
        auto_adjust=False,
        group_by="column",
    )
    if df is None or df.empty:
        raise RuntimeError(f"No FX data returned for {tkr}")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", tkr) in df.columns:
            s = df[("Adj Close", tkr)]
        elif ("Close", tkr) in df.columns:
            s = df[("Close", tkr)]
        else:
            lvl0 = df.columns.get_level_values(0)
            if "Adj Close" in lvl0:
                s = df.loc[:, lvl0 == "Adj Close"].iloc[:, 0]
            elif "Close" in lvl0:
                s = df.loc[:, lvl0 == "Close"].iloc[:, 0]
            else:
                raise KeyError(f"{tkr}: neither Adj Close nor Close found")
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            raise KeyError(f"{tkr}: neither Adj Close nor Close found")

    s = s.dropna().astype("float64")
    s.index = pd.to_datetime(s.index).date
    s.name = ccy
    return s

def _fx_to_usd(ccy: str, d: dt.date, fx_map: dict[str, pd.Series]) -> float:
    ccy = ccy.upper().strip()
    if ccy in {"USD", "USDT", "USDC"}:
        return 1.0
    s = fx_map[ccy]
    while d not in s.index:
        d = d - dt.timedelta(days=1)
    return float(s.loc[d])

# ----------------------------
# Ledger data structures
# ----------------------------
@dataclass
class PositionLotAvg:
    ticker: str
    quantity: float
    avg_cost: float
    currency: str = "USD"

@dataclass
class PositionView:
    ticker: str
    quantity: float
    avg_cost: float
    last_price: float | None
    market_value: float | None
    cost_value: float
    unrealized_pnl: float | None
    currency: str = "USD"

@dataclass
class DerivativePositionView:
    ticker: str
    side: str               # LONG/SHORT
    open_notional_usd: float
    avg_entry_price: float
    currency: str = "USD"

@dataclass
class PnLSummary:
    as_of: str
    trade_count: int
    tickers_spot: int
    tickers_derivatives: int
    realized_pnl: float
    unrealized_pnl_spot: float
    total_pnl: float

# ----------------------------
# Trade loading + normalization
# ----------------------------
def _parse_ts(ts_utc: str) -> pd.Timestamp:
    t = pd.to_datetime(ts_utc, errors="coerce", utc=True)
    if pd.isna(t):
        raise ValueError(f"Invalid ts_utc: {ts_utc}")
    return t

def _load_trades(
    s3,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[dict]:
    prefix = engine_key(TRADES_TABLE, "dt=")
    keys = s3_list_keys(s3, bucket=BUCKET, prefix=prefix)
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]
    if not keys:
        return []

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

    out.sort(key=lambda x: (_parse_ts(str(x.get("ts_utc", ""))), str(x.get("trade_id", ""))))
    return out

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
    return s

def _normalize_trade(t: dict) -> dict:
    trade_id = str(t.get("trade_id"))
    as_of = str(t.get("as_of"))
    ts_utc = str(t.get("ts_utc"))
    ticker = str(t.get("ticker")).upper().strip()
    side = str(t.get("side")).upper().strip()
    qty = float(t.get("quantity"))
    price = float(t.get("price"))
    ccy = str(t.get("currency") or "USD").upper().strip()

    if side not in ("BUY", "SELL"):
        raise ValueError(f"Trade {trade_id}: invalid side={side}")
    if qty <= 0 or price <= 0:
        raise ValueError(f"Trade {trade_id}: qty and price must be > 0")

    _ = pd.Timestamp(as_of).date()
    _ = _parse_ts(ts_utc)

    action_tag = (t.get("action_tag") or None)
    action_tag = None if action_tag is None else str(action_tag).strip().lower()

    quantity_unit = _normalize_unit(t.get("quantity_unit") or None)

    value = t.get("value", None)
    reported_pnl = t.get("reported_pnl", None)

    try:
        value = None if value is None or (isinstance(value, float) and pd.isna(value)) else float(value)
    except Exception:
        value = None
    try:
        reported_pnl = None if reported_pnl is None or (isinstance(reported_pnl, float) and pd.isna(reported_pnl)) else float(reported_pnl)
    except Exception:
        reported_pnl = None

    return {
        "trade_id": trade_id,
        "as_of": as_of,
        "ts_utc": ts_utc,
        "ticker": ticker,
        "side": side,
        "quantity": qty,
        "price": price,
        "currency": ccy,
        "note": t.get("note"),
        "_s3_key": t.get("_s3_key"),
        "action_tag": action_tag,
        "quantity_unit": quantity_unit,
        "value": value,
        "reported_pnl": reported_pnl,
    }

# ----------------------------
# Asset-type routing
# ----------------------------
_FX_PAIR_RE = re.compile(r"^[A-Z]{3}[-/][A-Z]{3}$")
_FUT_CODE_RE = re.compile(r"^[A-Z]{1,3}[FGHJKMNQUVXZ]\d{1,2}$")  # e.g. VXQ4, VXF5, ESU4

_CRYPTO_BASES = {
    # common Quantfury crypto bases seen in your logs / typical universe
    "BTC", "ETH", "ADA", "XRP", "DOT", "BCH", "LTC", "SOL", "DOGE", "SUI", "HBAR", "DASH",
    "BNB", "AVAX", "LINK", "MATIC", "ATOM", "NEAR", "UNI", "AAVE", "TRX", "ETC",
}
_CRYPTO_QUOTES = {"USD", "USDT", "USDC", "EUR"}

def _is_fx_pair(ticker: str) -> bool:
    t = str(ticker).upper().strip()
    return bool(_FX_PAIR_RE.match(t))

def _is_crypto_pair(ticker: str) -> bool:
    t = str(ticker).upper().strip()
    if "-" not in t:
        return False
    base, quote = t.split("-", 1)
    base = base.strip()
    quote = quote.strip()
    if not base or not quote:
        return False
    if quote not in _CRYPTO_QUOTES:
        return False
    # if base is known crypto base -> treat as crypto pair
    return base in _CRYPTO_BASES

def _route_asset_side(*, ticker: str, quantity_unit: Optional[str]) -> str:
    """
    Returns:
      - NOTIONAL for FX pairs and crypto pairs (always)
      - NOTIONAL for anything explicitly marked as contracts
      - SPOT otherwise
    """
    t = str(ticker).upper().strip()
    unit = (quantity_unit or "").lower().strip()

    # Always NOTIONAL for FX tickers like USD-CNY / EUR-USD, regardless of unit label
    if _is_fx_pair(t):
        return "NOTIONAL"

    # Always NOTIONAL for crypto pairs like BTC-USD / ADA-USDT, regardless of unit label
    if _is_crypto_pair(t):
        return "NOTIONAL"

    # Explicit contracts => NOTIONAL (futures/CFDs)
    if unit == "contracts":
        return "NOTIONAL"

    # Everything else is SPOT (equities/ETFs/etc)
    return "SPOT"

# ----------------------------
# FIFO accounting (spot + notional using action_tag)
# ----------------------------
def rebuild_positions_and_pnl_fifo(
    trades: List[dict],
    *,
    fx_map: dict[str, pd.Series] | None = None,
    debug: bool = False,
    recon_path: str | None = None,
) -> Tuple[Dict[str, PositionLotAvg], Dict[str, DerivativePositionView], float]:
    """
    SPOT side:
        - Quantity FIFO
        - Shorts allowed
        - action_tag is informational only (NOT enforced)
        - BUY/SELL determines direction
        - NO implicit flip inside a single trade

    NOTIONAL side (futures/crypto/FX):
        - Notional FIFO by `value` (USD notional)
        - action_tag enforced
        - `value` required
        - NO flipping allowed (raise)
    """
    spot_lots: Dict[str, Deque[Lot]] = {}
    notional_lots: Dict[str, Deque[ContractLot]] = {}

    ccy_by_ticker: Dict[str, str] = {}
    realized = 0.0
    recon_rows: list[dict] = []

    def _fx(ccy: str, as_of: str) -> float:
        d = pd.Timestamp(as_of).date()
        if fx_map is None:
            return 1.0
        return _fx_to_usd(ccy, d, fx_map) if ccy != "USD" else 1.0

    def _get_spot(ticker: str) -> Deque[Lot]:
        if ticker not in spot_lots:
            spot_lots[ticker] = deque()
        return spot_lots[ticker]

    def _get_notional(ticker: str) -> Deque[ContractLot]:
        if ticker not in notional_lots:
            notional_lots[ticker] = deque()
        return notional_lots[ticker]

    def _weighted_avg_cost_spot(lots: Deque[Lot]) -> float:
        num = 0.0
        den = 0.0
        for q, px in lots:
            aq = abs(float(q))
            num += aq * float(px)
            den += aq
        return float(num / den) if den > QTY_EPS else 0.0

    def _weighted_avg_entry_notional(lots: Deque[ContractLot]) -> Tuple[float, float]:
        # returns (avg_entry_price, total_abs_notional)
        num = 0.0
        den = 0.0
        for v, px in lots:
            av = abs(float(v))
            num += av * float(px)
            den += av
        if den <= LOT_EPS:
            return 0.0, 0.0
        return float(num / den), float(den)

    # Normalize first
    norm = [_normalize_trade(t) for t in trades]

    def _action_pri(tag: str | None) -> int:
        # open/add must happen before close/reduce if same timestamp
        return 0 if tag in {"open", "add"} else 1

    def _side_pri(side: str) -> int:
        # optional: SELL before BUY
        return 0 if side == "SELL" else 1

    norm.sort(
        key=lambda t: (
            _parse_ts(str(t.get("ts_utc", ""))),
            _action_pri(t.get("action_tag")),
            _side_pri(str(t.get("side", "")).upper().strip()),
            str(t.get("trade_id", "")),
        )
    )

    for t in norm:

        ticker = t["ticker"]
        side = t["side"]
        qty = float(t["quantity"])
        price = float(t["price"])
        ccy = t["currency"]
        trade_id = t["trade_id"]

        action_tag = t.get("action_tag")  # open/close/add/reduce or None
        unit = t.get("quantity_unit")
        value = t.get("value")

        if ticker in ccy_by_ticker and ccy_by_ticker[ticker] != ccy:
            raise ValueError(f"Currency mismatch for {ticker}: {ccy_by_ticker[ticker]} vs {ccy}")
        ccy_by_ticker[ticker] = ccy

        fx = _fx(ccy, t["as_of"])
        price_usd = float(price) * float(fx)

        trade_realized_delta = 0.0

        # enforce action_tag always for deterministic reconciliation
        if action_tag not in {"open", "close", "add", "reduce"}:
            raise ValueError(
                f"Trade {trade_id} ({ticker}) requires action_tag in "
                f"{{open, close, add, reduce}} (got {action_tag!r})."
            )

        route = _route_asset_side(ticker=ticker, quantity_unit=unit)

        # -----------------------------
        # NOTIONAL SIDE (futures/crypto/FX)
        # -----------------------------
        if route == "NOTIONAL":
            if value is None:
                raise ValueError(
                    f"Trade {trade_id} ({ticker}) is NOTIONAL-side (futures/crypto/FX) but value is missing. "
                    f"Need Quantfury 'value' to compute notional PnL."
                )

            value_usd = float(value) * float(fx)
            lots = _get_notional(ticker)  # Deque[(signed_notional_usd, entry_price_usd)]

            # OPEN/ADD => increase exposure, NO realized PnL
            if action_tag in {"open", "add"}:
                if side == "BUY":
                    lots.append((+value_usd, float(price_usd)))  # long
                else:
                    lots.append((-value_usd, float(price_usd)))  # short

            # CLOSE/REDUCE => must decrease exposure, realize PnL against existing lots
            else:  # {"close","reduce"}
                remaining_value = float(value_usd)

                if side == "SELL":
                    # SELL close/reduce closes LONG lots (positive)
                    while remaining_value > LOT_EPS and lots and lots[0][0] > 0:
                        lot_v, lot_px = lots[0]
                        lot_size = abs(float(lot_v))
                        close_value = min(remaining_value, lot_size)

                        delta = close_value * ((float(price_usd) - float(lot_px)) / float(lot_px))
                        realized += delta
                        trade_realized_delta += delta

                        left = lot_size - close_value
                        remaining_value -= close_value
                        lots.popleft()

                        left = _snap(left, LOT_EPS)
                        remaining_value = _snap(remaining_value, LOT_EPS)
                        if left != 0.0:
                            lots.appendleft((+left, float(lot_px)))

                    if remaining_value > LOT_EPS:
                        raise ValueError(
                            f"Trade {trade_id} ({ticker}) SELL close/reduce exceeds long exposure (NO FLIP). "
                            f"Remaining notional={remaining_value:.6f} USD. Lots={list(lots)[:5]} ..."
                        )

                else:  # BUY
                    # BUY close/reduce closes SHORT lots (negative)
                    while remaining_value > LOT_EPS and lots and lots[0][0] < 0:
                        lot_v, lot_px = lots[0]
                        lot_size = abs(float(lot_v))
                        close_value = min(remaining_value, lot_size)

                        delta = close_value * ((float(lot_px) - float(price_usd)) / float(lot_px))
                        realized += delta
                        trade_realized_delta += delta

                        left = lot_size - close_value
                        remaining_value -= close_value
                        lots.popleft()

                        left = _snap(left, LOT_EPS)
                        remaining_value = _snap(remaining_value, LOT_EPS)
                        if left != 0.0:
                            lots.appendleft((-left, float(lot_px)))

                    if remaining_value > LOT_EPS:
                        raise ValueError(
                            f"Trade {trade_id} ({ticker}) BUY close/reduce exceeds short exposure (NO FLIP). "
                            f"Remaining notional={remaining_value:.6f} USD. Lots={list(lots)[:5]} ..."
                        )

            lots = _clean_lots(lots, LOT_EPS)
            if lots:
                notional_lots[ticker] = lots
            else:
                notional_lots.pop(ticker, None)

            reported_pnl = t.get("reported_pnl")
            reported_pnl_usd = None if reported_pnl is None else float(reported_pnl) * float(fx)
            diff = None if reported_pnl_usd is None else float(trade_realized_delta - reported_pnl_usd)

            recon_rows.append({
                "trade_id": trade_id,
                "ts_utc": t.get("ts_utc"),
                "as_of": t.get("as_of"),
                "ticker": ticker,
                "side": side,
                "action_tag": action_tag,
                "quantity": qty,
                "price": price,
                "currency": ccy,
                "quantity_unit": (unit or "notional"),
                "route": "NOTIONAL",
                "value_usd": float(value_usd),
                "fx_to_usd": float(fx),
                "price_usd": float(price_usd),
                "engine_realized_delta": float(trade_realized_delta),
                "reported_pnl": reported_pnl,
                "reported_pnl_usd": reported_pnl_usd,
                "diff_engine_minus_reported_usd": diff,
                "note": t.get("note"),
                "_s3_key": t.get("_s3_key"),
            })
            continue

        # -----------------------------
        # SPOT SIDE (ETFs/shares/rest): quantity FIFO, action_tag aware, NO FLIP
        # -----------------------------
        lots = _get_spot(ticker)

        # OPEN/ADD => increase exposure in direction of side, no realized PnL
        if action_tag in {"open", "add"}:
            if side == "BUY":
                lots.append((+qty, float(price_usd)))   # add to long
            else:
                lots.append((-qty, float(price_usd)))   # add to short

        # REDUCE/CLOSE => must decrease exposure, realize PnL against opposite lots
        else:  # {"reduce","close"}
            remaining = float(qty)

            if side == "SELL":
                # SELL reduce/close closes LONG lots (positive qty lots)
                while remaining > QTY_EPS and lots and lots[0][0] > 0:
                    lot_q, lot_px = lots[0]
                    lot_size = abs(float(lot_q))
                    close_qty = min(remaining, lot_size)

                    # long close pnl = (exit - entry) * qty
                    delta = close_qty * (float(price_usd) - float(lot_px))
                    realized += delta
                    trade_realized_delta += delta

                    left = lot_size - close_qty
                    remaining -= close_qty
                    lots.popleft()

                    left = _snap(left, QTY_EPS)
                    remaining = _snap(remaining, QTY_EPS)
                    if left != 0.0:
                        lots.appendleft((+left, float(lot_px)))

                # NO FLIP: cannot sell-reduce more long than you have
                if remaining > QTY_EPS:
                    # allow tiny dust from rounding
                    if remaining <= 1e-8:
                        remaining = 0.0
                    else:
                        raise ValueError(
                            f"Trade {trade_id} ({ticker}) SELL {action_tag} exceeds LONG exposure (NO FLIP). "
                            f"Remaining qty={remaining:.12f}. Lots_head={list(lots)[:3]}"
                        )

            else:  # side == "BUY"
                # BUY reduce/close closes SHORT lots (negative qty lots)
                while remaining > QTY_EPS and lots and lots[0][0] < 0:
                    lot_q, lot_px = lots[0]
                    lot_size = abs(float(lot_q))
                    close_qty = min(remaining, lot_size)

                    # short close pnl = (entry - exit) * qty
                    delta = close_qty * (float(lot_px) - float(price_usd))
                    realized += delta
                    trade_realized_delta += delta

                    left = lot_size - close_qty
                    remaining -= close_qty
                    lots.popleft()

                    left = _snap(left, QTY_EPS)
                    remaining = _snap(remaining, QTY_EPS)
                    if left != 0.0:
                        lots.appendleft((-left, float(lot_px)))

                # NO FLIP: cannot buy-reduce more short than you have
                if remaining > QTY_EPS:
                    if remaining <= 1e-8:
                        remaining = 0.0
                    else:
                        raise ValueError(
                            f"Trade {trade_id} ({ticker}) BUY {action_tag} exceeds SHORT exposure (NO FLIP). "
                            f"Remaining qty={remaining:.12f}. Lots_head={list(lots)[:3]}"
                        )

        lots = _clean_lots(lots, LOT_EPS)
        if lots:
            spot_lots[ticker] = lots
        else:
            spot_lots.pop(ticker, None)


        reported_pnl = t.get("reported_pnl")
        diff = None if reported_pnl is None else float(trade_realized_delta - float(reported_pnl))

        recon_rows.append({
            "trade_id": trade_id,
            "ts_utc": t.get("ts_utc"),
            "as_of": t.get("as_of"),
            "ticker": ticker,
            "side": side,
            "action_tag": action_tag,
            "quantity": qty,
            "price": price,
            "currency": ccy,
            "quantity_unit": (unit or "spot"),
            "route": "SPOT",
            "value_usd": None,
            "fx_to_usd": float(fx),
            "price_usd": float(price_usd),
            "engine_realized_delta": float(trade_realized_delta),
            "reported_pnl": (None if reported_pnl is None else float(reported_pnl)),
            "diff_engine_minus_reported": diff,
            "note": t.get("note"),
            "_s3_key": t.get("_s3_key"),
        })

    # Build SPOT positions
    positions_spot: Dict[str, PositionLotAvg] = {}
    for ticker, lots in spot_lots.items():
        q = sum(float(qty_signed) for qty_signed, _ in lots)
        q = _snap(q, QTY_EPS)
        if q == 0.0:
            continue
        avg_cost = _weighted_avg_cost_spot(lots)
        positions_spot[ticker] = PositionLotAvg(ticker=ticker, quantity=float(q), avg_cost=float(avg_cost), currency="USD")

    # Build NOTIONAL positions (notional + avg entry)
    positions_deriv: Dict[str, DerivativePositionView] = {}
    for ticker, lots in notional_lots.items():
        net = sum(float(v) for v, _ in lots)
        net = _snap(net, LOT_EPS)
        if net == 0.0:
            continue
        avg_px, _ = _weighted_avg_entry_notional(lots)
        positions_deriv[ticker] = DerivativePositionView(
            ticker=ticker,
            side=("LONG" if net > 0 else "SHORT"),
            open_notional_usd=float(abs(net)),
            avg_entry_price=float(avg_px),
            currency="USD",
        )

    for tkr in list(positions_spot.keys()):
        if abs(float(positions_spot[tkr].quantity)) < POSITION_QTY_EPS:
            positions_spot.pop(tkr, None)

    for tkr in list(positions_deriv.keys()):
        if abs(float(positions_deriv[tkr].open_notional_usd)) < POSITION_NOTIONAL_EPS:
            positions_deriv.pop(tkr, None)
            
    # Write reconciliation CSV
    if recon_path is not None:
        df = pd.DataFrame(recon_rows)
        for c in ["reported_pnl", "engine_realized_delta", "diff_engine_minus_reported"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["engine_realized_delta"] = df["engine_realized_delta"].fillna(0.0)
        df_focus = df[df["reported_pnl"].notna() | (df["engine_realized_delta"].abs() > 1e-12)].copy()
        df_focus["abs_diff"] = df_focus["diff_engine_minus_reported"].abs()
        df_focus = df_focus.sort_values(["abs_diff", "ts_utc"], ascending=[False, True])
        df_focus.to_csv(recon_path, index=False)

        if debug:
            sub = df_focus[df_focus["reported_pnl"].notna()].copy()
            rep_sum = float(sub["reported_pnl"].sum()) if not sub.empty else 0.0
            eng_sum = float(sub["engine_realized_delta"].sum()) if not sub.empty else 0.0
            print("\n=== TOTALS (ONLY TRADES WITH REPORTED_PNL) ===")
            print(f"reported_sum: {rep_sum:,.2f}")
            print(f"engine_sum:   {eng_sum:,.2f}")
            print(f"diff:         {(eng_sum - rep_sum):,.2f}")
            print(f"count:        {int(len(sub))}")
            print(f"\n[debug] wrote reconciliation CSV: {recon_path}")

    return positions_spot, positions_deriv, float(realized)

# ----------------------------
# Splits
# ----------------------------
@dataclass(frozen=True)
class SplitEvent:
    ticker: str
    effective_date: str
    factor: float

def apply_split_events_to_trades(trades: list[dict], events: Iterable[SplitEvent]) -> list[dict]:
    evs = sorted(
        (
            SplitEvent(
                ticker=str(e.ticker).upper().strip(),
                effective_date=str(e.effective_date),
                factor=float(e.factor),
            )
            for e in events
        ),
        key=lambda e: (e.ticker, pd.Timestamp(e.effective_date).date()),
    )

    out: list[dict] = []
    for raw in trades:
        t = dict(raw)

        ticker = str(t.get("ticker", "")).upper().strip()
        as_of = pd.Timestamp(str(t.get("as_of"))).date()

        qty = float(t.get("quantity"))
        px = float(t.get("price"))

        for e in evs:
            if e.ticker != ticker:
                continue
            d_eff = pd.Timestamp(e.effective_date).date()
            if as_of < d_eff:
                qty *= e.factor
                px /= e.factor

        t["ticker"] = ticker
        t["quantity"] = float(qty)
        t["price"] = float(px)
        out.append(t)

    return out

# ----------------------------
# Spot views
# ----------------------------
def build_position_views(*, positions: Dict[str, PositionLotAvg], px_map: Dict[str, float]) -> List[PositionView]:
    out: List[PositionView] = []
    for t, p in sorted(positions.items(), key=lambda kv: kv[0]):
        qty = float(p.quantity)
        avg_cost = float(p.avg_cost)
        last = px_map.get(t)

        cost_value = abs(qty) * avg_cost

        if last is None or not pd.notna(last):
            out.append(
                PositionView(
                    ticker=t,
                    quantity=qty,
                    avg_cost=avg_cost,
                    last_price=None,
                    market_value=None,
                    cost_value=cost_value,
                    unrealized_pnl=None,
                    currency=p.currency,
                )
            )
            continue

        last = float(last)
        market_value = qty * last

        if qty > 0:
            unrealized_pnl = (last - avg_cost) * qty
        else:
            unrealized_pnl = (avg_cost - last) * abs(qty)

        out.append(
            PositionView(
                ticker=t,
                quantity=qty,
                avg_cost=avg_cost,
                last_price=last,
                market_value=float(market_value),
                cost_value=float(cost_value),
                unrealized_pnl=float(unrealized_pnl),
                currency=p.currency,
            )
        )
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Rebuild positions + PnL from trade ledger (fifo_with_splits).")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quantfury-csv", default=None, help="Optional CSV attach value/unit/tag/reported_pnl by trade_id (backfill only).")
    args = ap.parse_args()

    s3 = s3_client(REGION)
    _ = MarketStore(bucket=BUCKET)

    trades = _load_trades(s3, start=args.start, end=args.end)
    if not trades:
        raise RuntimeError("No trades found under engine/v1/trades/")

    # Optional attach Quantfury CSV fields (recon/backfill)
    qcsv = args.quantfury_csv or "./data/quantfury_trades.csv"
    try:
        qdf = pd.read_csv(qcsv)
        qdf["trade_id"] = qdf["trade_id"].astype(str)

        for c in ["value", "reported_pnl"]:
            if c in qdf.columns:
                qdf[c] = pd.to_numeric(qdf[c], errors="coerce")
        for c in ["quantity_unit", "action_tag"]:
            if c in qdf.columns:
                qdf[c] = qdf[c].astype(str)

        cols = [c for c in ["reported_pnl", "value", "quantity_unit", "action_tag"] if c in qdf.columns]
        qmap = qdf.set_index("trade_id")[cols].to_dict("index")

        attached = 0
        for tr in trades:
            tid = str(tr.get("trade_id") or "")
            if tid in qmap:
                for k, v in qmap[tid].items():
                    if v is None:
                        continue
                    if isinstance(v, float) and pd.isna(v):
                        continue
                    tr[k] = v
                attached += 1

        print(f"[debug] attached reported_pnl/value/unit/tag to {attached}/{len(trades)} trades from {qcsv}")
    except Exception as e:
        print(f"[debug] could not attach quantfury CSV fields ({qcsv}): {type(e).__name__}: {e}")

    # FX map
    trade_ccys = {str(t.get("currency") or "USD").upper().strip() for t in trades}
    trade_ccys.discard("USD")

    fx_map = {}
    if trade_ccys:
        start = str(min(pd.Timestamp(t["as_of"]).date() for t in trades) - dt.timedelta(days=10))
        end = str(max(pd.Timestamp(t["as_of"]).date() for t in trades) + dt.timedelta(days=10))
        for ccy in sorted(trade_ccys):
            fx_map[ccy] = _download_daily_fx_usd_per_ccy(ccy, start=start, end=end)

    # Splits
    events = [
        SplitEvent("WMT", "2024-02-26", 3.0),
        SplitEvent("AI.PA", "2024-06-10", 1.1),
    ]
    trades_adj = apply_split_events_to_trades(trades, events)

    positions_spot, positions_deriv, realized = rebuild_positions_and_pnl_fifo(
        trades_adj,
        fx_map=fx_map,
        debug=True,
        recon_path="./data/recon_trades_fifo_vs_reported.csv",
    )

    # Spot prices for unrealized
    latest_prices_key = "market/snapshots/v1/latest_prices.parquet"
    latest_prices_df = s3_get_parquet_df(s3, bucket=BUCKET, key=latest_prices_key)

    px_map = (
        latest_prices_df.assign(ticker=lambda d: d["ticker"].astype(str))
        .set_index("ticker")["adj_close_usd"]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .to_dict()
    )

    spot_views = build_position_views(positions=positions_spot, px_map=px_map)

    unreal_spot = 0.0
    missing_px_spot = 0
    for v in spot_views:
        if v.unrealized_pnl is None:
            missing_px_spot += 1
        else:
            unreal_spot += float(v.unrealized_pnl)

    as_of = args.as_of or pd.Timestamp(dt.date.today()).strftime("%Y-%m-%d")

    pnl = PnLSummary(
        as_of=as_of,
        trade_count=int(len(trades)),
        tickers_spot=int(len(spot_views)),
        tickers_derivatives=int(len(positions_deriv)),
        realized_pnl=float(realized),
        unrealized_pnl_spot=float(unreal_spot),
        total_pnl=float(realized + unreal_spot),
    )

    positions_payload = {
        "as_of": as_of,
        "method": "fifo_with_splits_spot_vs_notional_by_asset_type_action_tag",
        "spot_positions": [asdict(v) for v in spot_views],
        "derivatives_positions": [asdict(v) for v in positions_deriv.values()],
        "stats": {
            "n_spot_positions": int(len(spot_views)),
            "n_notional_positions": int(len(positions_deriv)),
            "missing_price_spot_n": int(missing_px_spot),
        },
        "sources": {
            "trades_prefix": f"s3://{BUCKET}/{engine_key(TRADES_TABLE)}/",
            "prices_snapshot": "market/snapshots/v1/latest_prices.parquet",
        },
    }

    pnl_payload = {
        "as_of": as_of,
        "method": positions_payload["method"],
        "summary": asdict(pnl),
        "sources": positions_payload["sources"],
    }

    print("\n=== LEDGER REBUILD (FIFO WITH SPLITS) ===")
    print(f"as_of:            {as_of}")
    print(f"trades:           {len(trades)}")
    print(f"open positions:   {len(spot_views)} (spot) + {len(positions_deriv)} (notional)")
    print(f"realized_pnl:     {realized:,.2f}")
    print(f"unrealized_pnl:   {unreal_spot:,.2f} (missing_px_spot={missing_px_spot})")
    print(f"total_pnl:        {realized + unreal_spot:,.2f}")
    print("")

    positions_key = dt_key(LEDGER_TABLE, as_of, "positions.json")
    pnl_key = dt_key(LEDGER_TABLE, as_of, "pnl.json")
    positions_latest_key = engine_key(LEDGER_TABLE, "positions", "latest.json")
    pnl_latest_key = engine_key(LEDGER_TABLE, "pnl", "latest.json")

    if args.dry_run:
        print("[DRY RUN] Would write:")
        print(f"  s3://{BUCKET}/{positions_key}")
        print(f"  s3://{BUCKET}/{pnl_key}")
        print(f"  s3://{BUCKET}/{positions_latest_key}")
        print(f"  s3://{BUCKET}/{pnl_latest_key}")
        return

    s3_put_json(s3, bucket=BUCKET, key=positions_key, payload=positions_payload)
    s3_put_json(s3, bucket=BUCKET, key=pnl_key, payload=pnl_payload)
    s3_put_json(s3, bucket=BUCKET, key=positions_latest_key, payload=positions_payload)
    s3_put_json(s3, bucket=BUCKET, key=pnl_latest_key, payload=pnl_payload)

    print("[OK] Wrote:")
    print(f"  s3://{BUCKET}/{positions_key}")
    print(f"  s3://{BUCKET}/{pnl_key}")
    print("[OK] Updated latest:")
    print(f"  s3://{BUCKET}/{positions_latest_key}")
    print(f"  s3://{BUCKET}/{pnl_latest_key}")
    print("")

if __name__ == "__main__":
    main()
