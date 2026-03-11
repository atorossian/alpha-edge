from __future__ import annotations

import argparse
import datetime as dt
import io
import json
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import boto3
import pandas as pd
import yfinance as yf

from alpha_edge.core.market_store import MarketStore

BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"

TRADES_TABLE = "trades"
LEDGER_TABLE = "ledger"

CASHFLOWS_TABLE = "cashflows"
DIVIDENDS_TABLE = "dividends"

OHLCV_USD_ROOT = "market/ohlcv_usd/v1"

LEDGER_CHECKPOINTS_ROOT = "ledger_checkpoints"
LEDGER_CHECKPOINTS_VERSION = "v=1"

QTY_EPS = 1e-9
CLOSE_QTY_EPS = 5e-8
LOT_EPS = 1e-9
CLOSE_VALUE_EPS = 0.05
POSITION_QTY_EPS = 1e-6
POSITION_NOTIONAL_EPS = 1e-4

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


def _snap(x: float, eps: float = QTY_EPS) -> float:
    return 0.0 if abs(float(x)) < eps else float(x)


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
    value: float | None
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
    quantity: float | None
    quantity_unit: Optional[str]


@dataclass
class PositionLotAvg:
    asset_id: str
    ticker: str
    quantity: float
    avg_cost: float
    currency: str = "USD"


@dataclass
class PositionView:
    asset_id: str
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
    asset_id: str
    ticker: str
    side: str
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
    dividends_pnl_usd: float
    net_cashflow_usd: float
    total_pnl_usd: float
    equity_usd: float


@dataclass
class LedgerState:
    spot_long_lots: Dict[str, Deque[OpenQtyLot]]
    spot_short_lots: Dict[str, Deque[OpenQtyLot]]
    notional_long_lots: Dict[str, Deque[OpenValueLot]]
    notional_short_lots: Dict[str, Deque[OpenValueLot]]
    ticker_by_asset: Dict[str, str]
    ccy_by_asset: Dict[str, str]
    realized_pnl_usd: float
    net_cashflow_usd: float
    dividends_pnl_usd: float
    trade_count: int


@dataclass(frozen=True)
class SplitEvent:
    ticker: str
    effective_date: str
    factor: float


def _clean_abs_lots(lots: Deque[OpenQtyLot], eps: float = LOT_EPS) -> Deque[OpenQtyLot]:
    out: Deque[OpenQtyLot] = deque()
    for lot in lots:
        lot.quantity_remaining = _snap(abs(float(lot.quantity_remaining)), eps)
        if lot.quantity_remaining == 0.0:
            continue
        out.append(lot)
    return out


def _clean_abs_notional_lots(lots: Deque[OpenValueLot], eps: float = LOT_EPS) -> Deque[OpenValueLot]:
    out: Deque[OpenValueLot] = deque()
    for lot in lots:
        lot.value_remaining = _snap(abs(float(lot.value_remaining)), eps)
        if lot.value_remaining == 0.0:
            continue
        out.append(lot)
    return out


def _clone_qty_lot(lot: OpenQtyLot) -> OpenQtyLot:
    return OpenQtyLot(**asdict(lot))


def _clone_value_lot(lot: OpenValueLot) -> OpenValueLot:
    return OpenValueLot(**asdict(lot))


def _clone_qty_lots_dict(d: Dict[str, Deque[OpenQtyLot]]) -> Dict[str, Deque[OpenQtyLot]]:
    out: Dict[str, Deque[OpenQtyLot]] = {}
    for asset_id, lots in d.items():
        out[str(asset_id)] = deque(_clone_qty_lot(lot) for lot in lots)
    return out


def _clone_value_lots_dict(d: Dict[str, Deque[OpenValueLot]]) -> Dict[str, Deque[OpenValueLot]]:
    out: Dict[str, Deque[OpenValueLot]] = {}
    for asset_id, lots in d.items():
        out[str(asset_id)] = deque(_clone_value_lot(lot) for lot in lots)
    return out


def clone_ledger_state(state: LedgerState) -> LedgerState:
    return LedgerState(
        spot_long_lots=_clone_qty_lots_dict(state.spot_long_lots),
        spot_short_lots=_clone_qty_lots_dict(state.spot_short_lots),
        notional_long_lots=_clone_value_lots_dict(state.notional_long_lots),
        notional_short_lots=_clone_value_lots_dict(state.notional_short_lots),
        ticker_by_asset={str(k): str(v) for k, v in state.ticker_by_asset.items()},
        ccy_by_asset={str(k): str(v) for k, v in state.ccy_by_asset.items()},
        realized_pnl_usd=float(state.realized_pnl_usd),
        net_cashflow_usd=float(state.net_cashflow_usd),
        dividends_pnl_usd=float(state.dividends_pnl_usd),
        trade_count=int(state.trade_count),
    )


def build_empty_ledger_state() -> LedgerState:
    return LedgerState(
        spot_long_lots={},
        spot_short_lots={},
        notional_long_lots={},
        notional_short_lots={},
        ticker_by_asset={},
        ccy_by_asset={},
        realized_pnl_usd=0.0,
        net_cashflow_usd=0.0,
        dividends_pnl_usd=0.0,
        trade_count=0,
    )


def s3_client(region: str = REGION):
    return boto3.client("s3", region_name=region)


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def dt_key(table: str, dt_str: str, filename: str) -> str:
    return engine_key(table, f"dt={dt_str}", filename)


def checkpoint_key(*, account_id: str, as_of: str, filename: str = "state.json") -> str:
    return engine_key(
        LEDGER_CHECKPOINTS_ROOT,
        LEDGER_CHECKPOINTS_VERSION,
        f"account_id={str(account_id).strip()}",
        f"as_of={str(as_of).strip()}",
        filename,
    )


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
# Time parsing
# ----------------------------
def _parse_ts(ts_utc: str) -> pd.Timestamp:
    t = pd.to_datetime(ts_utc, errors="coerce", utc=True)
    if pd.isna(t):
        raise ValueError(f"Invalid ts_utc: {ts_utc}")
    return t


# ----------------------------
# Load trades / cashflows / dividends
# ----------------------------
def _load_trades(s3, *, start: Optional[str] = None, end: Optional[str] = None) -> List[dict]:
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


def _load_cashflows(s3, *, start: Optional[str], end: Optional[str]) -> List[dict]:
    prefix = engine_key(CASHFLOWS_TABLE, "dt=")
    keys = s3_list_keys(s3, bucket=BUCKET, prefix=prefix)
    keys = [k for k in keys if k.endswith(".json") and "/cashflow_" in k]
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

    out.sort(key=lambda x: (_parse_ts(str(x.get("ts_utc", ""))), str(x.get("cashflow_id", ""))))
    return out


def _load_dividends(s3, *, start: Optional[str], end: Optional[str]) -> List[dict]:
    prefix = engine_key(DIVIDENDS_TABLE, "dt=")
    keys = s3_list_keys(s3, bucket=BUCKET, prefix=prefix)
    keys = [k for k in keys if k.endswith(".json") and "/dividend_" in k]
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

    out.sort(key=lambda x: (_parse_ts(str(x.get("ts_utc", ""))), str(x.get("dividend_id", ""))))
    return out


def _load_split_events_from_store() -> list[SplitEvent]:
    store = MarketStore(bucket=BUCKET, region=REGION)
    df = store.read_corporate_actions(
        columns=["asset_id", "ticker", "effective_date", "action_type", "split_factor"]
    )

    if df is None or df.empty:
        return []

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action_type"] = df["action_type"].astype(str).str.upper().str.strip()
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce").dt.date
    df["split_factor"] = pd.to_numeric(df["split_factor"], errors="coerce")

    df = df[
        (df["action_type"] == "SPLIT")
        & df["ticker"].notna()
        & df["effective_date"].notna()
        & df["split_factor"].notna()
    ].copy()

    df = df.sort_values(["ticker", "effective_date"], kind="stable")

    out: list[SplitEvent] = []
    for _, r in df.iterrows():
        out.append(
            SplitEvent(
                ticker=str(r["ticker"]).upper().strip(),
                effective_date=pd.Timestamp(r["effective_date"]).date().isoformat(),
                factor=float(r["split_factor"]),
            )
        )
    return out


# ----------------------------
# Checkpoint helpers
# ----------------------------
def _lots_dict_to_rows_qty(d: Dict[str, Deque[OpenQtyLot]]) -> list[dict]:
    rows: list[dict] = []
    for asset_id in sorted(d):
        for lot in d[asset_id]:
            rows.append(asdict(lot))
    return rows


def _lots_dict_to_rows_value(d: Dict[str, Deque[OpenValueLot]]) -> list[dict]:
    rows: list[dict] = []
    for asset_id in sorted(d):
        for lot in d[asset_id]:
            rows.append(asdict(lot))
    return rows


def ledger_state_to_payload(
    *,
    account_id: str,
    as_of: str,
    method: str,
    state: LedgerState,
) -> dict:
    return {
        "account_id": str(account_id).strip(),
        "as_of": str(as_of).strip(),
        "method": str(method),
        "created_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "realized_pnl_usd": float(state.realized_pnl_usd),
        "net_cashflow_usd": float(state.net_cashflow_usd),
        "dividends_pnl_usd": float(state.dividends_pnl_usd),
        "trade_count": int(state.trade_count),
        "spot_long_lots": _lots_dict_to_rows_qty(state.spot_long_lots),
        "spot_short_lots": _lots_dict_to_rows_qty(state.spot_short_lots),
        "notional_long_lots": _lots_dict_to_rows_value(state.notional_long_lots),
        "notional_short_lots": _lots_dict_to_rows_value(state.notional_short_lots),
        "ticker_by_asset": {str(k): str(v) for k, v in state.ticker_by_asset.items()},
        "ccy_by_asset": {str(k): str(v) for k, v in state.ccy_by_asset.items()},
    }


def _rows_to_qty_lots_dict(rows: list[dict]) -> Dict[str, Deque[OpenQtyLot]]:
    out: Dict[str, Deque[OpenQtyLot]] = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        lot = OpenQtyLot(
            trade_id=str(r.get("trade_id", "")),
            as_of=str(r.get("as_of", "")),
            ts_utc=str(r.get("ts_utc", "")),
            asset_id=str(r.get("asset_id", "")),
            ticker=str(r.get("ticker", "")).upper().strip(),
            side=str(r.get("side", "")).upper().strip(),
            action_tag=str(r.get("action_tag", "")).strip().lower(),
            quantity_open=float(r.get("quantity_open", 0.0)),
            quantity_remaining=float(r.get("quantity_remaining", 0.0)),
            price=float(r.get("price", 0.0)),
            value=None if r.get("value") is None else float(r.get("value")),
            quantity_unit=None if r.get("quantity_unit") is None else str(r.get("quantity_unit")),
        )
        asset_id = str(lot.asset_id).strip()
        if asset_id not in out:
            out[asset_id] = deque()
        out[asset_id].append(lot)

    for asset_id in list(out.keys()):
        out[asset_id] = _clean_abs_lots(out[asset_id], LOT_EPS)
        if not out[asset_id]:
            out.pop(asset_id, None)
    return out


def _rows_to_value_lots_dict(rows: list[dict]) -> Dict[str, Deque[OpenValueLot]]:
    out: Dict[str, Deque[OpenValueLot]] = {}
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        lot = OpenValueLot(
            trade_id=str(r.get("trade_id", "")),
            as_of=str(r.get("as_of", "")),
            ts_utc=str(r.get("ts_utc", "")),
            asset_id=str(r.get("asset_id", "")),
            ticker=str(r.get("ticker", "")).upper().strip(),
            side=str(r.get("side", "")).upper().strip(),
            action_tag=str(r.get("action_tag", "")).strip().lower(),
            value_open=float(r.get("value_open", 0.0)),
            value_remaining=float(r.get("value_remaining", 0.0)),
            price=float(r.get("price", 0.0)),
            quantity=None if r.get("quantity") is None else float(r.get("quantity")),
            quantity_unit=None if r.get("quantity_unit") is None else str(r.get("quantity_unit")),
        )
        asset_id = str(lot.asset_id).strip()
        if asset_id not in out:
            out[asset_id] = deque()
        out[asset_id].append(lot)

    for asset_id in list(out.keys()):
        out[asset_id] = _clean_abs_notional_lots(out[asset_id], LOT_EPS)
        if not out[asset_id]:
            out.pop(asset_id, None)
    return out


def ledger_state_from_payload(payload: dict) -> LedgerState:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dict.")

    return LedgerState(
        spot_long_lots=_rows_to_qty_lots_dict(payload.get("spot_long_lots", [])),
        spot_short_lots=_rows_to_qty_lots_dict(payload.get("spot_short_lots", [])),
        notional_long_lots=_rows_to_value_lots_dict(payload.get("notional_long_lots", [])),
        notional_short_lots=_rows_to_value_lots_dict(payload.get("notional_short_lots", [])),
        ticker_by_asset={
            str(k).strip(): str(v).upper().strip()
            for k, v in (payload.get("ticker_by_asset") or {}).items()
        },
        ccy_by_asset={
            str(k).strip(): str(v).upper().strip()
            for k, v in (payload.get("ccy_by_asset") or {}).items()
        },
        realized_pnl_usd=float(payload.get("realized_pnl_usd", 0.0)),
        net_cashflow_usd=float(payload.get("net_cashflow_usd", 0.0)),
        dividends_pnl_usd=float(payload.get("dividends_pnl_usd", 0.0)),
        trade_count=int(payload.get("trade_count", 0)),
    )


def write_ledger_checkpoint(
    s3,
    *,
    bucket: str,
    account_id: str,
    as_of: str,
    method: str,
    state: LedgerState,
) -> str:
    key = checkpoint_key(account_id=account_id, as_of=as_of, filename="state.json")
    payload = ledger_state_to_payload(
        account_id=account_id,
        as_of=as_of,
        method=method,
        state=state,
    )
    s3_put_json(s3, bucket=bucket, key=key, payload=payload)
    return key


def read_ledger_checkpoint(
    s3,
    *,
    bucket: str,
    account_id: str,
    as_of: str,
) -> Optional[LedgerState]:
    key = checkpoint_key(account_id=account_id, as_of=as_of, filename="state.json")
    try:
        payload = s3_get_json(s3, bucket=bucket, key=key)
    except Exception:
        return None
    return ledger_state_from_payload(payload)


def discover_latest_checkpoint_asof(
    s3,
    *,
    bucket: str,
    account_id: str,
    end_as_of: str,
) -> Optional[str]:
    prefix = engine_key(
        LEDGER_CHECKPOINTS_ROOT,
        LEDGER_CHECKPOINTS_VERSION,
        f"account_id={str(account_id).strip()}",
    )
    keys = s3_list_keys(s3, bucket=bucket, prefix=prefix)
    end_d = pd.Timestamp(end_as_of).date()

    best: Optional[dt.date] = None
    for k in keys:
        parts = k.split("/")
        p = next((x for x in parts if x.startswith("as_of=")), None)
        if not p:
            continue
        try:
            d = pd.Timestamp(p.replace("as_of=", "")).date()
        except Exception:
            continue
        if d <= end_d and (best is None or d > best):
            best = d

    return None if best is None else best.isoformat()


def _is_month_end(as_of: str) -> bool:
    d = pd.Timestamp(as_of).date()
    return (d + dt.timedelta(days=1)).month != d.month


# ----------------------------
# Normalize + routing
# ----------------------------
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
        "trx", "etc", "doge", "hbar", "sui", "dash", "bch", "qtum"
    }:
        return "coins"
    if s in {"derivative", "derivatives"}:
        return "derivative"
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

    asset_id = t.get("asset_id", None)
    asset_id = None if asset_id is None else str(asset_id).strip()
    if not asset_id:
        raise ValueError(
            f"Trade {trade_id} missing asset_id. "
            "Re-record trades with record_trade.py after asset_id support, or backfill asset_id."
        )

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
        "asset_id": asset_id,
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
    _ = str(ticker).upper().strip()
    # if _is_fx_pair(t):
    #     return "NOTIONAL"
    return "SPOT"


# ----------------------------
# Split adjustments
# ----------------------------
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
# Stateful FIFO engine
# ----------------------------
def replay_trades_into_state(
    trades: List[dict],
    *,
    state: LedgerState | None = None,
    fx_map: dict[str, pd.Series] | None = None,
    debug: bool = False,
    recon_path: str | None = None,
) -> Tuple[LedgerState, Dict[str, PositionLotAvg], Dict[str, DerivativePositionView], float, dict]:
    _ = recon_path  # reserved for future use

    if state is None:
        st = build_empty_ledger_state()
    else:
        st = clone_ledger_state(state)

    spot_long_lots = st.spot_long_lots
    spot_short_lots = st.spot_short_lots
    notional_long_lots = st.notional_long_lots
    notional_short_lots = st.notional_short_lots
    ticker_by_asset = st.ticker_by_asset
    ccy_by_asset = st.ccy_by_asset
    realized = float(st.realized_pnl_usd)

    def _fx(ccy: str, as_of: str) -> float:
        d = pd.Timestamp(as_of).date()
        if fx_map is None:
            return 1.0
        return _fx_to_usd(ccy, d, fx_map) if ccy != "USD" else 1.0

    def _get_spot_long(asset_id: str) -> Deque[OpenQtyLot]:
        if asset_id not in spot_long_lots:
            spot_long_lots[asset_id] = deque()
        return spot_long_lots[asset_id]

    def _get_spot_short(asset_id: str) -> Deque[OpenQtyLot]:
        if asset_id not in spot_short_lots:
            spot_short_lots[asset_id] = deque()
        return spot_short_lots[asset_id]

    def _get_notional_long(asset_id: str) -> Deque[OpenValueLot]:
        if asset_id not in notional_long_lots:
            notional_long_lots[asset_id] = deque()
        return notional_long_lots[asset_id]

    def _get_notional_short(asset_id: str) -> Deque[OpenValueLot]:
        if asset_id not in notional_short_lots:
            notional_short_lots[asset_id] = deque()
        return notional_short_lots[asset_id]

    norm = [_normalize_trade(t) for t in trades]

    def _action_pri(tag: str | None) -> int:
        return 0 if tag in {"open", "add"} else 1

    def _side_pri(side: str) -> int:
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
        asset_id = str(t["asset_id"])
        ticker = str(t["ticker"]).upper().strip()
        side = t["side"]
        qty = float(t["quantity"])
        price = float(t["price"])
        ccy = t["currency"]
        trade_id = t["trade_id"]
        action_tag = t.get("action_tag")
        unit = t.get("quantity_unit")
        value = t.get("value")

        if action_tag not in {"open", "close", "add", "reduce"}:
            raise ValueError(
                f"Trade {trade_id} ({ticker}/{asset_id}) requires action_tag in "
                f"{{open, close, add, reduce}} (got {action_tag!r})."
            )

        if asset_id in ticker_by_asset and ticker_by_asset[asset_id] != ticker:
            raise ValueError(f"Ticker mismatch for asset_id={asset_id}: {ticker_by_asset[asset_id]} vs {ticker}")
        ticker_by_asset[asset_id] = ticker

        if asset_id in ccy_by_asset and ccy_by_asset[asset_id] != ccy:
            raise ValueError(f"Currency mismatch for asset_id={asset_id}: {ccy_by_asset[asset_id]} vs {ccy}")
        ccy_by_asset[asset_id] = ccy

        fx = _fx(ccy, t["as_of"])
        price_usd = float(price) * float(fx)

        route = _route_asset_side(ticker=ticker, quantity_unit=unit)

        if debug and (_is_crypto_pair(ticker) or unit in {"contracts", "derivative", "derivatives"}):
            print(
                f"[route] trade_id={trade_id} asset_id={asset_id} ticker={ticker} "
                f"side={side} action_tag={action_tag} unit={unit} value={value} qty={qty} route={route}"
            )

        if route == "NOTIONAL":
            if value is None:
                raise ValueError(
                    f"Trade {trade_id} ({ticker}/{asset_id}) is NOTIONAL-side but value is missing. "
                    f"Need broker 'value' to compute notional PnL."
                )

            value_usd = float(value) * float(fx)

            if side == "BUY" and action_tag in {"open", "add"}:
                lots = _get_notional_long(asset_id)
                lots.append(
                    OpenValueLot(
                        trade_id=trade_id,
                        as_of=t["as_of"],
                        ts_utc=t["ts_utc"],
                        asset_id=asset_id,
                        ticker=ticker,
                        side=side,
                        action_tag=action_tag,
                        value_open=abs(value_usd),
                        value_remaining=abs(value_usd),
                        price=float(price_usd),
                        quantity=qty,
                        quantity_unit=unit,
                    )
                )
                notional_long_lots[asset_id] = _clean_abs_notional_lots(lots, LOT_EPS)
                continue

            if side == "SELL" and action_tag in {"close", "reduce"}:
                lots = _get_notional_long(asset_id)
                remaining_value = float(value_usd)

                while remaining_value > LOT_EPS and lots:
                    lot = lots[0]
                    lot_size = abs(float(lot.value_remaining))
                    close_value = min(remaining_value, lot_size)
                    delta = close_value * ((float(price_usd) - float(lot.price)) / float(lot.price))
                    realized += delta

                    lot.value_remaining = _snap(lot.value_remaining - close_value, LOT_EPS)
                    remaining_value = _snap(remaining_value - close_value, LOT_EPS)

                    if lot.value_remaining <= LOT_EPS:
                        lots.popleft()

                remaining_value = _snap(remaining_value, CLOSE_VALUE_EPS)
                if remaining_value > CLOSE_VALUE_EPS:
                    raise ValueError(
                        f"Trade {trade_id} ({ticker}/{asset_id}) SELL close/reduce exceeds long exposure (NO FLIP). "
                        f"Remaining notional={remaining_value:.6f} USD."
                    )

                lots = _clean_abs_notional_lots(lots, LOT_EPS)
                if lots:
                    notional_long_lots[asset_id] = lots
                else:
                    notional_long_lots.pop(asset_id, None)
                continue

            if side == "SELL" and action_tag in {"open", "add"}:
                lots = _get_notional_short(asset_id)
                lots.append(
                    OpenValueLot(
                        trade_id=trade_id,
                        as_of=t["as_of"],
                        ts_utc=t["ts_utc"],
                        asset_id=asset_id,
                        ticker=ticker,
                        side=side,
                        action_tag=action_tag,
                        value_open=abs(value_usd),
                        value_remaining=abs(value_usd),
                        price=float(price_usd),
                        quantity=qty,
                        quantity_unit=unit,
                    )
                )
                notional_short_lots[asset_id] = _clean_abs_notional_lots(lots, LOT_EPS)
                continue

            if side == "BUY" and action_tag in {"close", "reduce"}:
                lots = _get_notional_short(asset_id)
                remaining_value = float(value_usd)

                while remaining_value > LOT_EPS and lots:
                    lot = lots[0]
                    lot_size = abs(float(lot.value_remaining))
                    close_value = min(remaining_value, lot_size)
                    delta = close_value * ((float(lot.price) - float(price_usd)) / float(lot.price))
                    realized += delta

                    lot.value_remaining = _snap(lot.value_remaining - close_value, LOT_EPS)
                    remaining_value = _snap(remaining_value - close_value, LOT_EPS)

                    if lot.value_remaining <= LOT_EPS:
                        lots.popleft()

                remaining_value = _snap(remaining_value, CLOSE_VALUE_EPS)
                if remaining_value > CLOSE_VALUE_EPS:
                    raise ValueError(
                        f"Trade {trade_id} ({ticker}/{asset_id}) BUY close/reduce exceeds short exposure (NO FLIP). "
                        f"Remaining notional={remaining_value:.6f} USD."
                    )

                lots = _clean_abs_notional_lots(lots, LOT_EPS)
                if lots:
                    notional_short_lots[asset_id] = lots
                else:
                    notional_short_lots.pop(asset_id, None)
                continue

            raise ValueError(
                f"Trade {trade_id} ({ticker}/{asset_id}) unsupported NOTIONAL side/action combination: "
                f"side={side}, action_tag={action_tag}"
            )

        if side == "BUY" and action_tag in {"open", "add"}:
            lots = _get_spot_long(asset_id)
            lots.append(
                OpenQtyLot(
                    trade_id=trade_id,
                    as_of=t["as_of"],
                    ts_utc=t["ts_utc"],
                    asset_id=asset_id,
                    ticker=ticker,
                    side=side,
                    action_tag=action_tag,
                    quantity_open=abs(qty),
                    quantity_remaining=abs(qty),
                    price=float(price_usd),
                    value=value,
                    quantity_unit=unit,
                )
            )
            spot_long_lots[asset_id] = _clean_abs_lots(lots, LOT_EPS)
            continue

        if side == "SELL" and action_tag in {"close", "reduce"}:
            lots = _get_spot_long(asset_id)
            remaining = float(qty)

            while remaining > QTY_EPS and lots:
                lot = lots[0]
                lot_size = abs(float(lot.quantity_remaining))
                close_qty = min(remaining, lot_size)

                delta = close_qty * (float(price_usd) - float(lot.price))
                realized += delta

                lot.quantity_remaining = _snap(lot.quantity_remaining - close_qty, QTY_EPS)
                remaining = _snap(remaining - close_qty, QTY_EPS)

                if lot.quantity_remaining <= QTY_EPS:
                    lots.popleft()

            remaining = _snap(remaining, CLOSE_QTY_EPS)
            if remaining > CLOSE_QTY_EPS:
                raise ValueError(
                    f"Trade {trade_id} ({ticker}/{asset_id}) SELL {action_tag} exceeds LONG exposure (NO FLIP). "
                    f"Remaining qty={remaining:.12f}."
                )

            lots = _clean_abs_lots(lots, LOT_EPS)
            if lots:
                spot_long_lots[asset_id] = lots
            else:
                spot_long_lots.pop(asset_id, None)
            continue

        if side == "SELL" and action_tag in {"open", "add"}:
            lots = _get_spot_short(asset_id)
            lots.append(
                OpenQtyLot(
                    trade_id=trade_id,
                    as_of=t["as_of"],
                    ts_utc=t["ts_utc"],
                    asset_id=asset_id,
                    ticker=ticker,
                    side=side,
                    action_tag=action_tag,
                    quantity_open=abs(qty),
                    quantity_remaining=abs(qty),
                    price=float(price_usd),
                    value=value,
                    quantity_unit=unit,
                )
            )
            spot_short_lots[asset_id] = _clean_abs_lots(lots, LOT_EPS)
            continue

        if side == "BUY" and action_tag in {"close", "reduce"}:
            lots = _get_spot_short(asset_id)
            remaining = float(qty)

            while remaining > QTY_EPS and lots:
                lot = lots[0]
                lot_size = abs(float(lot.quantity_remaining))
                close_qty = min(remaining, lot_size)

                delta = close_qty * (float(lot.price) - float(price_usd))
                realized += delta

                lot.quantity_remaining = _snap(lot.quantity_remaining - close_qty, QTY_EPS)
                remaining = _snap(remaining - close_qty, QTY_EPS)

                if lot.quantity_remaining <= QTY_EPS:
                    lots.popleft()

            remaining = _snap(remaining, CLOSE_QTY_EPS)
            if remaining > CLOSE_QTY_EPS:
                raise ValueError(
                    f"Trade {trade_id} ({ticker}/{asset_id}) BUY {action_tag} exceeds SHORT exposure (NO FLIP). "
                    f"Remaining qty={remaining:.12f}."
                )

            lots = _clean_abs_lots(lots, LOT_EPS)
            if lots:
                spot_short_lots[asset_id] = lots
            else:
                spot_short_lots.pop(asset_id, None)
            continue

        raise ValueError(
            f"Trade {trade_id} ({ticker}/{asset_id}) unsupported SPOT side/action combination: "
            f"side={side}, action_tag={action_tag}"
        )

    positions_spot: Dict[str, PositionLotAvg] = {}
    all_spot_assets = set(spot_long_lots.keys()) | set(spot_short_lots.keys())
    for asset_id in all_spot_assets:
        long_lots = spot_long_lots.get(asset_id, deque())
        short_lots = spot_short_lots.get(asset_id, deque())

        long_qty = sum(float(lot.quantity_remaining) for lot in long_lots)
        short_qty = sum(float(lot.quantity_remaining) for lot in short_lots)
        net_qty = _snap(long_qty - short_qty, QTY_EPS)

        if net_qty == 0.0:
            continue

        if net_qty > 0:
            num = sum(abs(float(lot.quantity_remaining)) * float(lot.price) for lot in long_lots)
            den = sum(abs(float(lot.quantity_remaining)) for lot in long_lots)
        else:
            num = sum(abs(float(lot.quantity_remaining)) * float(lot.price) for lot in short_lots)
            den = sum(abs(float(lot.quantity_remaining)) for lot in short_lots)

        avg_cost = float(num / den) if den > QTY_EPS else 0.0

        positions_spot[asset_id] = PositionLotAvg(
            asset_id=str(asset_id),
            ticker=str(ticker_by_asset.get(asset_id, "")),
            quantity=float(net_qty),
            avg_cost=float(avg_cost),
            currency="USD",
        )

    positions_deriv: Dict[str, DerivativePositionView] = {}
    all_notional_assets = set(notional_long_lots.keys()) | set(notional_short_lots.keys())
    for asset_id in all_notional_assets:
        long_lots = notional_long_lots.get(asset_id, deque())
        short_lots = notional_short_lots.get(asset_id, deque())

        long_v = sum(float(lot.value_remaining) for lot in long_lots)
        short_v = sum(float(lot.value_remaining) for lot in short_lots)
        net = _snap(long_v - short_v, LOT_EPS)

        if net == 0.0:
            continue

        if net > 0:
            num = sum(abs(float(lot.value_remaining)) * float(lot.price) for lot in long_lots)
            den = sum(abs(float(lot.value_remaining)) for lot in long_lots)
        else:
            num = sum(abs(float(lot.value_remaining)) * float(lot.price) for lot in short_lots)
            den = sum(abs(float(lot.value_remaining)) for lot in short_lots)

        avg_px = float(num / den) if den > LOT_EPS else 0.0

        positions_deriv[asset_id] = DerivativePositionView(
            asset_id=str(asset_id),
            ticker=str(ticker_by_asset.get(asset_id, "")),
            side=("LONG" if net > 0 else "SHORT"),
            open_notional_usd=float(abs(net)),
            avg_entry_price=float(avg_px),
            currency="USD",
        )

    for aid in list(positions_spot.keys()):
        if abs(float(positions_spot[aid].quantity)) < POSITION_QTY_EPS:
            positions_spot.pop(aid, None)

    for aid in list(positions_deriv.keys()):
        if abs(float(positions_deriv[aid].open_notional_usd)) < POSITION_NOTIONAL_EPS:
            positions_deriv.pop(aid, None)

    open_lot_trace = {
        "spot_long_lots": [
            asdict(lot)
            for asset_id in sorted(spot_long_lots)
            for lot in spot_long_lots[asset_id]
        ],
        "spot_short_lots": [
            asdict(lot)
            for asset_id in sorted(spot_short_lots)
            for lot in spot_short_lots[asset_id]
        ],
        "notional_long_lots": [
            asdict(lot)
            for asset_id in sorted(notional_long_lots)
            for lot in notional_long_lots[asset_id]
        ],
        "notional_short_lots": [
            asdict(lot)
            for asset_id in sorted(notional_short_lots)
            for lot in notional_short_lots[asset_id]
        ],
    }

    out_state = LedgerState(
        spot_long_lots=spot_long_lots,
        spot_short_lots=spot_short_lots,
        notional_long_lots=notional_long_lots,
        notional_short_lots=notional_short_lots,
        ticker_by_asset=ticker_by_asset,
        ccy_by_asset=ccy_by_asset,
        realized_pnl_usd=float(realized),
        net_cashflow_usd=float(st.net_cashflow_usd),
        dividends_pnl_usd=float(st.dividends_pnl_usd),
        trade_count=int(st.trade_count + len(norm)),
    )

    return out_state, positions_spot, positions_deriv, float(realized), open_lot_trace


# ----------------------------
# Market price helpers
# ----------------------------
def _ohlcv_prefix_for_asset_year(asset_id: str, year: int) -> str:
    return f"{OHLCV_USD_ROOT}/asset_id={asset_id}/year={year}/"


def _pick_price_column(df: pd.DataFrame) -> str:
    for c in ["close_raw_usd", "adj_close_usd", "close_usd", "adj_close", "close", "Adj Close", "Close"]:
        if c in df.columns:
            return c
    raise KeyError(f"OHLCV parquet missing expected close column. cols={list(df.columns)[:50]}")


def _normalize_ohlcv_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    else:
        out["date"] = pd.to_datetime(out.index, errors="coerce").dt.date
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out


def _load_asset_close_usd_series_for_year(
    s3,
    *,
    bucket: str,
    asset_id: str,
    year: int,
    cache: dict[tuple[str, int], pd.Series],
) -> pd.Series:
    k = (asset_id, int(year))
    if k in cache:
        return cache[k]

    prefix = _ohlcv_prefix_for_asset_year(asset_id, int(year))
    keys = s3_list_keys(s3, bucket=bucket, prefix=prefix)
    keys = [x for x in keys if x.endswith(".parquet")]
    if not keys:
        s = pd.Series(dtype="float64")
        cache[k] = s
        return s

    dfs: list[pd.DataFrame] = []
    for key in sorted(keys):
        try:
            df = s3_get_parquet_df(s3, bucket=bucket, key=key)
            if df is None or df.empty:
                continue
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        s = pd.Series(dtype="float64")
        cache[k] = s
        return s

    df_all = pd.concat(dfs, ignore_index=False, sort=False)
    df_all = _normalize_ohlcv_dates(df_all)
    col = _pick_price_column(df_all)

    s = pd.to_numeric(df_all[col], errors="coerce")
    s.index = df_all["date"].astype(object)
    s = s.dropna()
    cache[k] = s
    return s


def get_asset_close_usd_asof(
    s3,
    *,
    bucket: str,
    asset_id: str,
    asof_date: dt.date,
    cache: dict[tuple[str, int], pd.Series],
    max_lookback_days: int = 14,
) -> Optional[float]:
    d = asof_date
    for _ in range(max_lookback_days + 1):
        s = _load_asset_close_usd_series_for_year(s3, bucket=bucket, asset_id=asset_id, year=int(d.year), cache=cache)
        if not s.empty and d in s.index:
            try:
                return float(s.loc[d])
            except Exception:
                pass
        d = d - dt.timedelta(days=1)
    return None


def build_position_views(*, positions: Dict[str, PositionLotAvg], px_by_asset_id: Dict[str, float]) -> List[PositionView]:
    out: List[PositionView] = []
    for asset_id, p in sorted(positions.items(), key=lambda kv: kv[0]):
        qty = float(p.quantity)
        avg_cost = float(p.avg_cost)
        tkr = str(p.ticker).upper().strip()
        last = px_by_asset_id.get(str(asset_id))
        cost_value = abs(qty) * avg_cost

        if last is None or not pd.notna(last):
            out.append(
                PositionView(
                    asset_id=str(asset_id),
                    ticker=tkr,
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
                asset_id=str(asset_id),
                ticker=tkr,
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
# Cashflow/dividend accumulators
# ----------------------------
def accumulate_net_cashflow_usd(
    cashflows: List[dict],
    *,
    fx_map: dict[str, pd.Series],
    initial: float = 0.0,
) -> float:
    total = float(initial)
    for cf in cashflows:
        try:
            d0 = pd.Timestamp(str(cf.get("as_of"))).date()
            ccy = str(cf.get("currency") or "USD").upper().strip()
            amt = float(cf.get("amount"))
            kind = str(cf.get("type") or cf.get("direction") or cf.get("cashflow_type") or "").strip().upper()
            sign = +1.0 if kind in {"DEPOSIT", "IN", "CREDIT"} else -1.0 if kind in {"WITHDRAWAL", "OUT", "DEBIT"} else None
            if sign is None:
                signed_amt = cf.get("amount_signed", None)
                if signed_amt is not None:
                    amt_signed = float(signed_amt)
                else:
                    raise ValueError(f"Unknown cashflow type={kind!r}")
            else:
                amt_signed = sign * amt

            if ccy in {"USD", "USDT", "USDC"}:
                fx = 1.0
            else:
                fx = _fx_to_usd(ccy, d0, fx_map)
            total += float(amt_signed) * float(fx)
        except Exception:
            continue
    return float(total)


def accumulate_dividends_pnl_usd(
    dividends: List[dict],
    *,
    fx_map: dict[str, pd.Series],
    initial: float = 0.0,
) -> float:
    total = float(initial)
    for dv in dividends:
        try:
            d0 = pd.Timestamp(str(dv.get("as_of"))).date()
            ccy = str(dv.get("currency") or "USD").upper().strip()
            amt = float(dv.get("amount"))
            tax = dv.get("tax", None)
            tax = 0.0 if tax is None or (isinstance(tax, float) and pd.isna(tax)) else float(tax)

            if ccy in {"USD", "USDT", "USDC"}:
                fx = 1.0
            else:
                fx = _fx_to_usd(ccy, d0, fx_map)

            total += (amt * fx) - (tax * fx)
        except Exception:
            continue
    return float(total)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Rebuild positions + PnL from trade ledger (fifo_with_splits).")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--account-id", default="main")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quantfury-csv", default=None)
    ap.add_argument(
        "--prices-mode",
        choices=["asof", "latest"],
        default="asof",
        help="SPOT pricing source for unrealized PnL.",
    )
    ap.add_argument("--use-checkpoints", action="store_true", help="Resume from latest checkpoint <= end date if available.")
    ap.add_argument("--write-checkpoints", action="store_true", help="Write a checkpoint after successful rebuild.")
    ap.add_argument(
        "--checkpoint-policy",
        choices=["month_end", "always"],
        default="month_end",
        help="When --write-checkpoints is set, choose when to emit a checkpoint.",
    )
    args = ap.parse_args()

    s3 = s3_client(REGION)
    _ = MarketStore(bucket=BUCKET)

    method_name = "fifo_with_splits_spot_vs_notional_by_asset_type_action_tag_asset_id_native_v3_open_lot_trace"
    as_of = args.as_of or pd.Timestamp(dt.date.today()).strftime("%Y-%m-%d")
    checkpoint_loaded_asof: Optional[str] = None
    base_state: LedgerState | None = None

    effective_start = args.start
    if args.use_checkpoints:
        latest_ckpt_asof = discover_latest_checkpoint_asof(
            s3,
            bucket=BUCKET,
            account_id=str(args.account_id),
            end_as_of=str(args.end or as_of),
        )
        if latest_ckpt_asof is not None:
            base_state = read_ledger_checkpoint(
                s3,
                bucket=BUCKET,
                account_id=str(args.account_id),
                as_of=latest_ckpt_asof,
            )
            if base_state is not None:
                checkpoint_loaded_asof = latest_ckpt_asof
                effective_start = (pd.Timestamp(latest_ckpt_asof).date() + dt.timedelta(days=1)).isoformat()

    trades = _load_trades(s3, start=effective_start, end=args.end)
    cashflows = _load_cashflows(s3, start=effective_start, end=args.end)
    dividends = _load_dividends(s3, start=effective_start, end=args.end)

    if (not trades) and (not cashflows) and (not dividends) and (base_state is None):
        raise RuntimeError("No activity found under engine/v1/{trades,cashflows,dividends}/ for the requested window.")

    all_ccys = set()
    for t in trades:
        all_ccys.add(str(t.get("currency") or "USD").upper().strip())
    for c in cashflows:
        all_ccys.add(str(c.get("currency") or "USD").upper().strip())
    for d in dividends:
        all_ccys.add(str(d.get("currency") or "USD").upper().strip())

    if base_state is not None:
        for ccy in base_state.ccy_by_asset.values():
            all_ccys.add(str(ccy).upper().strip())

    all_ccys.discard("USD")
    all_ccys.discard("USDT")
    all_ccys.discard("USDC")

    def _activity_date_span(objs: list[dict]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        if not objs:
            return None
        ds = [pd.Timestamp(str(x.get("as_of"))).normalize() for x in objs if x.get("as_of")]
        ds = [d for d in ds if pd.notna(d)]
        if not ds:
            return None
        return min(ds), max(ds)

    spans = []
    for seq in (trades, cashflows, dividends):
        sp = _activity_date_span(seq)
        if sp is not None:
            spans.append(sp)

    if base_state is not None and checkpoint_loaded_asof is not None:
        ckpt_day = pd.Timestamp(checkpoint_loaded_asof).normalize()
        spans.append((ckpt_day, ckpt_day))

    fx_map: dict[str, pd.Series] = {}
    if all_ccys and spans:
        min_d = min(s[0] for s in spans).date() - dt.timedelta(days=10)
        max_d = max(s[1] for s in spans).date() + dt.timedelta(days=10)
        start_fx = str(min_d)
        end_fx = str(max_d)

        for ccy in sorted(all_ccys):
            fx_map[ccy] = _download_daily_fx_usd_per_ccy(ccy, start=start_fx, end=end_fx)

    events = _load_split_events_from_store()
    trades_adj = apply_split_events_to_trades(trades, events) if trades else []

    final_state, positions_spot, positions_deriv, realized, open_lot_trace = replay_trades_into_state(
        trades_adj,
        state=base_state,
        fx_map=fx_map,
        debug=False,
        recon_path="./data/recon_trades_fifo_vs_reported.csv",
    )

    final_state.net_cashflow_usd = accumulate_net_cashflow_usd(
        cashflows,
        fx_map=fx_map,
        initial=(base_state.net_cashflow_usd if base_state is not None else 0.0),
    )
    final_state.dividends_pnl_usd = accumulate_dividends_pnl_usd(
        dividends,
        fx_map=fx_map,
        initial=(base_state.dividends_pnl_usd if base_state is not None else 0.0),
    )

    asof_date = pd.Timestamp(as_of).date()

    px_by_asset_id: dict[str, float] = {}
    missing_px_spot = 0

    if args.prices_mode == "asof":
        close_cache: dict[tuple[str, int], pd.Series] = {}
        for asset_id in positions_spot.keys():
            px = get_asset_close_usd_asof(
                s3,
                bucket=BUCKET,
                asset_id=str(asset_id),
                asof_date=asof_date,
                cache=close_cache,
                max_lookback_days=14,
            )
            if px is None:
                missing_px_spot += 1
            else:
                px_by_asset_id[str(asset_id)] = float(px)
        prices_source_ref = f"s3://{BUCKET}/{OHLCV_USD_ROOT}/asset_id=<.>/year=<.>/"
    else:
        latest_prices_key = "market/snapshots/v1/latest_prices.parquet"
        latest_prices_df = s3_get_parquet_df(s3, bucket=BUCKET, key=latest_prices_key)
        px_map_ticker = (
            latest_prices_df.assign(ticker=lambda d: d["ticker"].astype(str).str.upper().str.strip())
            .set_index("ticker")["close_raw_usd"]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .to_dict()
        )
        for asset_id, p in positions_spot.items():
            tkr = str(p.ticker).upper().strip()
            px = px_map_ticker.get(tkr)
            if px is None:
                missing_px_spot += 1
            else:
                px_by_asset_id[str(asset_id)] = float(px)
        prices_source_ref = f"s3://{BUCKET}/{latest_prices_key}"

    spot_views = build_position_views(positions=positions_spot, px_by_asset_id=px_by_asset_id)

    unreal_spot = 0.0
    for v in spot_views:
        if v.unrealized_pnl is not None:
            unreal_spot += float(v.unrealized_pnl)

    total_pnl_usd = float(final_state.realized_pnl_usd + unreal_spot + final_state.dividends_pnl_usd)
    equity_usd = float(total_pnl_usd + final_state.net_cashflow_usd)

    pnl = PnLSummary(
        as_of=as_of,
        trade_count=int(final_state.trade_count),
        tickers_spot=int(len(spot_views)),
        tickers_derivatives=int(len(positions_deriv)),
        realized_pnl=float(final_state.realized_pnl_usd),
        unrealized_pnl_spot=float(unreal_spot),
        dividends_pnl_usd=float(final_state.dividends_pnl_usd),
        net_cashflow_usd=float(final_state.net_cashflow_usd),
        total_pnl_usd=float(total_pnl_usd),
        equity_usd=float(equity_usd),
    )

    positions_payload = {
        "as_of": as_of,
        "method": method_name,
        "spot_positions": [asdict(v) for v in spot_views],
        "derivatives_positions": [asdict(v) for v in positions_deriv.values()],
        "stats": {
            "n_spot_positions": int(len(spot_views)),
            "n_notional_positions": int(len(positions_deriv)),
            "missing_price_spot_n": int(missing_px_spot),
            "prices_mode": str(args.prices_mode),
            "checkpoint_loaded_asof": checkpoint_loaded_asof,
            "effective_replay_start": effective_start,
        },
        "sources": {
            "trades_prefix": f"s3://{BUCKET}/{engine_key(TRADES_TABLE)}/",
            "cashflows_prefix": f"s3://{BUCKET}/{engine_key(CASHFLOWS_TABLE)}/",
            "dividends_prefix": f"s3://{BUCKET}/{engine_key(DIVIDENDS_TABLE)}/",
            "prices_source": prices_source_ref,
        },
    }

    pnl_payload = {
        "as_of": as_of,
        "method": method_name,
        "summary": asdict(pnl),
        "sources": positions_payload["sources"],
    }

    open_lot_trace_payload = {
        "as_of": as_of,
        "method": method_name,
        "summary": {
            "spot_long_lots_n": len(open_lot_trace["spot_long_lots"]),
            "spot_short_lots_n": len(open_lot_trace["spot_short_lots"]),
            "notional_long_lots_n": len(open_lot_trace["notional_long_lots"]),
            "notional_short_lots_n": len(open_lot_trace["notional_short_lots"]),
        },
        "open_lots": open_lot_trace,
        "sources": positions_payload["sources"],
    }

    print("\n=== LEDGER REBUILD (FIFO WITH SPLITS) ===")
    print(f"as_of:                {as_of}")
    print(f"checkpoint_loaded:    {checkpoint_loaded_asof}")
    print(f"effective_start:      {effective_start}")
    print(f"trades_loaded:        {len(trades)}")
    print(f"cashflows_loaded:     {len(cashflows)}")
    print(f"dividends_loaded:     {len(dividends)}")
    print(f"trade_count_total:    {final_state.trade_count}")
    print(f"open positions:       {len(spot_views)} (spot) + {len(positions_deriv)} (notional)")
    print(f"open lots trace:      {open_lot_trace_payload['summary']}")
    print(f"realized_pnl:         {final_state.realized_pnl_usd:,.2f}")
    print(f"unrealized_pnl_spot:  {unreal_spot:,.2f} (missing_px_spot={missing_px_spot})")
    print(f"dividends_pnl_usd:    {final_state.dividends_pnl_usd:,.2f}")
    print(f"net_cashflow_usd:     {final_state.net_cashflow_usd:,.2f}")
    print(f"total_pnl_usd:        {total_pnl_usd:,.2f}")
    print(f"equity_usd:           {equity_usd:,.2f}")
    print(f"prices_mode:          {args.prices_mode}")
    print("")

    positions_key = dt_key(LEDGER_TABLE, as_of, "positions.json")
    pnl_key = dt_key(LEDGER_TABLE, as_of, "pnl.json")
    open_lot_trace_key = dt_key(LEDGER_TABLE, as_of, "open_lot_trace.json")

    positions_latest_key = engine_key(LEDGER_TABLE, "positions", "latest.json")
    pnl_latest_key = engine_key(LEDGER_TABLE, "pnl", "latest.json")
    open_lot_trace_latest_key = engine_key(LEDGER_TABLE, "open_lot_trace", "latest.json")

    should_write_checkpoint = bool(args.write_checkpoints) and (
        args.checkpoint_policy == "always" or (args.checkpoint_policy == "month_end" and _is_month_end(as_of))
    )
    checkpoint_out_key = checkpoint_key(account_id=str(args.account_id), as_of=as_of, filename="state.json")

    if args.dry_run:
        print("[DRY RUN] Would write:")
        print(f"  s3://{BUCKET}/{positions_key}")
        print(f"  s3://{BUCKET}/{pnl_key}")
        print(f"  s3://{BUCKET}/{open_lot_trace_key}")
        print(f"  s3://{BUCKET}/{positions_latest_key}")
        print(f"  s3://{BUCKET}/{pnl_latest_key}")
        print(f"  s3://{BUCKET}/{open_lot_trace_latest_key}")
        if should_write_checkpoint:
            print(f"  s3://{BUCKET}/{checkpoint_out_key}")
        return

    s3_put_json(s3, bucket=BUCKET, key=positions_key, payload=positions_payload)
    s3_put_json(s3, bucket=BUCKET, key=pnl_key, payload=pnl_payload)
    s3_put_json(s3, bucket=BUCKET, key=open_lot_trace_key, payload=open_lot_trace_payload)

    s3_put_json(s3, bucket=BUCKET, key=positions_latest_key, payload=positions_payload)
    s3_put_json(s3, bucket=BUCKET, key=pnl_latest_key, payload=pnl_payload)
    s3_put_json(s3, bucket=BUCKET, key=open_lot_trace_latest_key, payload=open_lot_trace_payload)

    if should_write_checkpoint:
        write_ledger_checkpoint(
            s3,
            bucket=BUCKET,
            account_id=str(args.account_id),
            as_of=as_of,
            method=method_name,
            state=final_state,
        )

    print("[OK] Wrote:")
    print(f"  s3://{BUCKET}/{positions_key}")
    print(f"  s3://{BUCKET}/{pnl_key}")
    print(f"  s3://{BUCKET}/{open_lot_trace_key}")
    print("[OK] Updated latest:")
    print(f"  s3://{BUCKET}/{positions_latest_key}")
    print(f"  s3://{BUCKET}/{pnl_latest_key}")
    print(f"  s3://{BUCKET}/{open_lot_trace_latest_key}")
    if should_write_checkpoint:
        print("[OK] Wrote checkpoint:")
        print(f"  s3://{BUCKET}/{checkpoint_out_key}")
    print("")


if __name__ == "__main__":
    main()