# equity_valuation.py
from __future__ import annotations

import io
import json
from dataclasses import dataclass, asdict
from typing import Any, Optional

import boto3
import numpy as np
import pandas as pd

from alpha_edge.core.schemas import RuntimeConfig
from alpha_edge.core.runtime import runtime_engine_key, runtime_market_key
from alpha_edge.core.data_loader import parse_ledger_positions_obj


@dataclass
class EquityValuationResult:
    equity: float
    as_of: str
    valuation_source: str
    price_source: str | None
    valuation_timestamp_utc: str
    ledger_positions_key: str | None = None
    ledger_pnl_key: str | None = None
    manual_override_used: bool = False
    net_cashflow_usd: float | None = None
    realized_pnl_usd: float | None = None
    dividends_pnl_usd: float | None = None
    live_unrealized_pnl_usd: float | None = None
    spot_positions_n: int | None = None
    derivative_positions_n: int | None = None
    missing_price_assets: list[str] | None = None
    stale_price_assets: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_as_of(as_of: str | None) -> str:
    if as_of is None:
        return pd.Timestamp.utcnow().date().strftime("%Y-%m-%d")
    return pd.Timestamp(as_of).date().strftime("%Y-%m-%d")


def s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


def _s3_get_json_optional(s3, *, bucket: str, key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        payload = json.loads(obj["Body"].read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _s3_get_parquet_optional(s3, *, bucket: str, key: str) -> pd.DataFrame:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception:
        return pd.DataFrame()


def ledger_positions_key(cfg: RuntimeConfig, as_of: str) -> str:
    return runtime_engine_key(cfg, "ledger", f"dt={as_of}", "positions.json")


def ledger_pnl_key(cfg: RuntimeConfig, as_of: str) -> str:
    return runtime_engine_key(cfg, "ledger", f"dt={as_of}", "pnl.json")


def ledger_positions_latest_key(cfg: RuntimeConfig) -> str:
    return runtime_engine_key(cfg, "ledger", "positions", "latest.json")


def ledger_pnl_latest_key(cfg: RuntimeConfig) -> str:
    return runtime_engine_key(cfg, "ledger", "pnl", "latest.json")


def latest_prices_snapshot_key(cfg: RuntimeConfig) -> str:
    return runtime_market_key(cfg, "snapshots", "v1", "latest_prices.parquet")


def _load_ledger_payloads(
    *,
    cfg: RuntimeConfig,
    as_of: str,
    allow_latest_fallback: bool = True,
) -> tuple[dict, dict, str, str]:
    s3 = s3_client(cfg)

    pos_key = ledger_positions_key(cfg, as_of)
    pnl_key = ledger_pnl_key(cfg, as_of)

    positions = _s3_get_json_optional(s3, bucket=cfg.bucket, key=pos_key)
    pnl = _s3_get_json_optional(s3, bucket=cfg.bucket, key=pnl_key)

    if (positions is None or pnl is None) and allow_latest_fallback:
        latest_pos_key = ledger_positions_latest_key(cfg)
        latest_pnl_key = ledger_pnl_latest_key(cfg)
        positions_latest = _s3_get_json_optional(s3, bucket=cfg.bucket, key=latest_pos_key)
        pnl_latest = _s3_get_json_optional(s3, bucket=cfg.bucket, key=latest_pnl_key)
        if positions is None and positions_latest is not None:
            positions = positions_latest
            pos_key = latest_pos_key
        if pnl is None and pnl_latest is not None:
            pnl = pnl_latest
            pnl_key = latest_pnl_key

    if positions is None:
        raise RuntimeError(
            f"Missing ledger positions for as_of={as_of}. Tried s3://{cfg.bucket}/{ledger_positions_key(cfg, as_of)}"
        )
    if pnl is None:
        raise RuntimeError(
            f"Missing ledger PnL for as_of={as_of}. Tried s3://{cfg.bucket}/{ledger_pnl_key(cfg, as_of)}"
        )

    return positions, pnl, pos_key, pnl_key


def _load_latest_price_map(
    *,
    cfg: RuntimeConfig,
    price_column: str | None = None,
) -> tuple[dict[str, float], dict[str, str], str, dict[str, Any]]:
    key = latest_prices_snapshot_key(cfg)
    df = _s3_get_parquet_optional(s3_client(cfg), bucket=cfg.bucket, key=key)
    if df is None or df.empty:
        raise RuntimeError(f"Missing or empty latest prices snapshot: s3://{cfg.bucket}/{key}")

    df = df.copy()
    if "asset_id" not in df.columns and "ticker" not in df.columns:
        raise RuntimeError(f"Latest prices snapshot has neither asset_id nor ticker column. columns={list(df.columns)}")

    candidate_cols = [price_column] if price_column else []
    candidate_cols += [
        "close_raw_usd",
        "close_adjusted_usd",
        "close_usd",
        "adj_close_usd",
        "price_usd",
        "last_price_usd",
        "last_price",
        "close",
    ]
    candidate_cols = [c for c in candidate_cols if c and c in df.columns]
    if not candidate_cols:
        raise RuntimeError(f"No usable price column in latest prices snapshot. columns={list(df.columns)}")

    px_col = candidate_cols[0]
    df[px_col] = pd.to_numeric(df[px_col], errors="coerce")
    df = df[df[px_col].notna() & np.isfinite(df[px_col]) & (df[px_col] > 0)].copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")

    asset_px: dict[str, float] = {}
    ticker_px: dict[str, float] = {}
    price_dates: dict[str, str] = {}

    if "asset_id" in df.columns:
        tmp = df.copy()
        tmp["asset_id"] = tmp["asset_id"].astype(str).str.strip()
        tmp = tmp[tmp["asset_id"] != ""]
        tmp = tmp.drop_duplicates(subset=["asset_id"], keep="last")
        asset_px = dict(zip(tmp["asset_id"], tmp[px_col].astype(float)))
        if "date" in tmp.columns:
            price_dates.update(dict(zip(tmp["asset_id"], tmp["date"].astype(str))))

    if "ticker" in df.columns:
        tmp = df.copy()
        tmp["ticker"] = tmp["ticker"].astype(str).str.upper().str.strip()
        tmp = tmp[tmp["ticker"] != ""]
        tmp = tmp.drop_duplicates(subset=["ticker"], keep="last")
        ticker_px = dict(zip(tmp["ticker"], tmp[px_col].astype(float)))
        if "date" in tmp.columns:
            price_dates.update({str(k): str(v) for k, v in zip(tmp["ticker"], tmp["date"].astype(str))})

    meta = {
        "price_column": px_col,
        "rows": int(len(df)),
        "price_dates": price_dates,
    }
    return asset_px, ticker_px, f"s3://{cfg.bucket}/{key}", meta


def compute_live_equity_from_ledger_and_prices(
    *,
    pnl_summary: dict,
    spot_rows: list[dict],
    prices_for_valuation: pd.Series | dict,
) -> float:
    """Canonical current-equity calculation used by daily report and standalone valuation.

    Equity is calculated as:
      net cashflow + realized PnL + dividends + live-marked unrealized PnL.

    For long spot positions, unrealized PnL = (price - avg_cost) * quantity.
    For short spot positions, unrealized PnL = (avg_cost - price) * abs(quantity).
    """
    net_cashflow = float(pnl_summary.get("net_cashflow_usd", 0.0) or 0.0)
    realized = float(pnl_summary.get("realized_pnl", pnl_summary.get("realized_pnl_usd", 0.0)) or 0.0)
    dividends = float(pnl_summary.get("dividends_pnl_usd", 0.0) or 0.0)

    if not isinstance(prices_for_valuation, pd.Series):
        prices_for_valuation = pd.Series(dict(prices_for_valuation or {}), dtype="float64")
    prices_for_valuation.index = prices_for_valuation.index.astype(str)

    unrealized = 0.0
    missing: list[str] = []

    for r in spot_rows or []:
        ticker = str(r.get("ticker") or "").upper().strip()
        asset_id = str(r.get("asset_id") or "").strip()
        lookup_keys = [k for k in [asset_id, ticker] if k]
        if not lookup_keys:
            continue

        qty = float(r.get("quantity") or 0.0)
        if abs(qty) <= 0.0:
            continue

        avg_cost = r.get("avg_cost")
        if avg_cost is None:
            raise RuntimeError(f"Cannot compute live equity: missing avg_cost for {ticker or asset_id}")

        px = np.nan
        for key in lookup_keys:
            candidate = prices_for_valuation.get(key, np.nan)
            if np.isfinite(float(candidate)):
                px = float(candidate)
                break

        if not np.isfinite(float(px)):
            missing.append(ticker or asset_id)
            continue

        if qty > 0:
            unrealized += (float(px) - float(avg_cost)) * qty
        else:
            unrealized += (float(avg_cost) - float(px)) * abs(qty)

    if missing:
        raise RuntimeError("Cannot compute live equity: missing valuation prices for " + ", ".join(sorted(set(missing))))

    return float(net_cashflow + realized + dividends + unrealized)


def resolve_current_equity(
    *,
    cfg: RuntimeConfig,
    as_of: str | None = None,
    equity_override: float | None = None,
    allow_latest_fallback: bool = True,
    price_column: str | None = None,
) -> EquityValuationResult:
    """Resolve current equity for scripts that require equity/equity0.

    If equity_override is supplied it is returned as an explicit manual bypass.
    Otherwise, the function loads ledger positions + PnL and marks open spot
    positions with the latest market prices snapshot.
    """
    as_of_norm = _parse_as_of(as_of)
    valuation_ts = _utc_now()

    if equity_override is not None:
        return EquityValuationResult(
            equity=float(equity_override),
            as_of=as_of_norm,
            valuation_source="manual_override",
            price_source=None,
            valuation_timestamp_utc=valuation_ts,
            manual_override_used=True,
            metadata={"note": "Explicit equity override supplied by caller."},
        )

    positions_obj, pnl_obj, positions_key, pnl_key = _load_ledger_payloads(
        cfg=cfg,
        as_of=as_of_norm,
        allow_latest_fallback=allow_latest_fallback,
    )
    spot_rows, deriv_rows = parse_ledger_positions_obj(positions_obj)
    pnl_summary = (pnl_obj.get("summary") or {}) if isinstance(pnl_obj, dict) else {}

    asset_px, ticker_px, price_source, price_meta = _load_latest_price_map(cfg=cfg, price_column=price_column)
    prices = pd.Series({**ticker_px, **asset_px}, dtype="float64")

    equity = compute_live_equity_from_ledger_and_prices(
        pnl_summary=pnl_summary,
        spot_rows=spot_rows,
        prices_for_valuation=prices,
    )

    net_cashflow = float(pnl_summary.get("net_cashflow_usd", 0.0) or 0.0)
    realized = float(pnl_summary.get("realized_pnl", pnl_summary.get("realized_pnl_usd", 0.0)) or 0.0)
    dividends = float(pnl_summary.get("dividends_pnl_usd", 0.0) or 0.0)
    live_unrealized = float(equity - net_cashflow - realized - dividends)

    return EquityValuationResult(
        equity=float(equity),
        as_of=as_of_norm,
        valuation_source="ledger_cashflows_realized_dividends_plus_latest_price_marks",
        price_source=price_source,
        valuation_timestamp_utc=valuation_ts,
        ledger_positions_key=f"s3://{cfg.bucket}/{positions_key}",
        ledger_pnl_key=f"s3://{cfg.bucket}/{pnl_key}",
        manual_override_used=False,
        net_cashflow_usd=float(net_cashflow),
        realized_pnl_usd=float(realized),
        dividends_pnl_usd=float(dividends),
        live_unrealized_pnl_usd=float(live_unrealized),
        spot_positions_n=int(len(spot_rows)),
        derivative_positions_n=int(len(deriv_rows)),
        missing_price_assets=[],
        stale_price_assets=[],
        metadata={"price_snapshot": price_meta},
    )


def print_equity_valuation_result(result: EquityValuationResult) -> None:
    print("\n=== CURRENT EQUITY VALUATION ===")
    print(f"as_of:                {result.as_of}")
    print(f"equity:               {result.equity:,.2f} USD")
    print(f"valuation_source:     {result.valuation_source}")
    print(f"manual_override:      {result.manual_override_used}")
    print(f"price_source:         {result.price_source}")
    print(f"ledger_positions:     {result.ledger_positions_key}")
    print(f"ledger_pnl:           {result.ledger_pnl_key}")
    if result.net_cashflow_usd is not None:
        print(f"net_cashflow_usd:     {result.net_cashflow_usd:,.2f}")
    if result.realized_pnl_usd is not None:
        print(f"realized_pnl_usd:     {result.realized_pnl_usd:,.2f}")
    if result.dividends_pnl_usd is not None:
        print(f"dividends_pnl_usd:    {result.dividends_pnl_usd:,.2f}")
    if result.live_unrealized_pnl_usd is not None:
        print(f"live_unrealized_usd:  {result.live_unrealized_pnl_usd:,.2f}")
    print(f"valuation_ts_utc:     {result.valuation_timestamp_utc}")
    print("")
