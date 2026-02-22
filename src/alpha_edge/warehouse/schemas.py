from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa


# ----------------------------
# Canonical Warehouse Schemas (v=1)
# ----------------------------
# Notes:
# - Use pa.date32() for dates (Athena friendly).
# - Use pa.timestamp("ms") for timestamps (no timezone).
# - Keep names stable; evolve via v=2 folder when needed.
# ----------------------------

DIM_ASSETS_SCHEMA = pa.schema(
    [
        pa.field("asset_id", pa.string(), nullable=False),
        pa.field("row_id", pa.string(), nullable=True),
        pa.field("broker_ticker", pa.string(), nullable=True),
        pa.field("ticker", pa.string(), nullable=True),
        pa.field("yahoo_ticker", pa.string(), nullable=True),
        pa.field("name", pa.string(), nullable=True),
        pa.field("asset_class", pa.string(), nullable=True),
        pa.field("role", pa.string(), nullable=True),
        pa.field("currency", pa.string(), nullable=True),
        pa.field("exchange", pa.string(), nullable=True),
        pa.field("country", pa.string(), nullable=True),
        pa.field("market", pa.string(), nullable=True),
        pa.field("include", pa.int32(), nullable=True),
        pa.field("is_tradable", pa.bool_(), nullable=True),
        pa.field("lock_yahoo_ticker", pa.int32(), nullable=True),
        pa.field("yahoo_ok", pa.bool_(), nullable=True),
        pa.field("yahoo_symbol_used", pa.string(), nullable=True),
        pa.field("resolver_debug", pa.string(), nullable=True),
        pa.field("valid_from", pa.date32(), nullable=True),
        pa.field("valid_to", pa.date32(), nullable=True),
        pa.field("load_ts_utc", pa.timestamp("ms"), nullable=False),
        pa.field("source_ref", pa.string(), nullable=True),
    ]
)

FCT_TRADES_SCHEMA = pa.schema(
    [
        pa.field("trade_id", pa.string(), nullable=False),
        pa.field("as_of_date", pa.date32(), nullable=False),
        pa.field("ts_utc", pa.timestamp("ms"), nullable=True),
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("asset_id", pa.string(), nullable=True),
        pa.field("broker_ticker", pa.string(), nullable=True),
        pa.field("side", pa.string(), nullable=True),
        pa.field("quantity", pa.float64(), nullable=True),
        pa.field("price", pa.float64(), nullable=True),
        pa.field("currency", pa.string(), nullable=True),
        pa.field("action_tag", pa.string(), nullable=True),
        pa.field("quantity_unit", pa.string(), nullable=True),
        pa.field("value", pa.float64(), nullable=True),
        pa.field("reported_pnl", pa.float64(), nullable=True),
        pa.field("choice_id", pa.string(), nullable=True),
        pa.field("portfolio_run_id", pa.string(), nullable=True),
        pa.field("note", pa.string(), nullable=True),
        pa.field("source_key", pa.string(), nullable=False),
        pa.field("load_ts_utc", pa.timestamp("ms"), nullable=False),
    ]
)

FCT_POSITIONS_DAILY_SCHEMA = pa.schema(
    [
        pa.field("as_of_date", pa.date32(), nullable=False),
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("asset_id", pa.string(), nullable=True),
        pa.field("broker_ticker", pa.string(), nullable=True),
        pa.field("position_type", pa.string(), nullable=False),  # SPOT / NOTIONAL
        pa.field("side", pa.string(), nullable=True),           # NOTIONAL: LONG/SHORT
        pa.field("quantity", pa.float64(), nullable=True),      # SPOT
        pa.field("avg_cost_usd", pa.float64(), nullable=True),  # SPOT
        pa.field("last_price_usd", pa.float64(), nullable=True),# SPOT
        pa.field("market_value_usd", pa.float64(), nullable=True),# SPOT
        pa.field("cost_value_usd", pa.float64(), nullable=True),# SPOT
        pa.field("unrealized_pnl_usd", pa.float64(), nullable=True),# SPOT
        pa.field("open_notional_usd", pa.float64(), nullable=True), # NOTIONAL
        pa.field("avg_entry_price_usd", pa.float64(), nullable=True),# NOTIONAL
        pa.field("currency", pa.string(), nullable=True),
        pa.field("missing_price_flag", pa.bool_(), nullable=True),
        pa.field("source_key", pa.string(), nullable=False),
        pa.field("load_ts_utc", pa.timestamp("ms"), nullable=False),
    ]
)

FCT_ACCOUNT_PNL_DAILY_SCHEMA = pa.schema(
    [
        pa.field("as_of_date", pa.date32(), nullable=False),
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("realized_pnl_usd", pa.float64(), nullable=True),
        pa.field("unrealized_pnl_usd", pa.float64(), nullable=True),
        pa.field("total_pnl_usd", pa.float64(), nullable=True),
        pa.field("equity_usd", pa.float64(), nullable=True),
        pa.field("trade_count", pa.int64(), nullable=True),
        pa.field("tickers_spot", pa.int64(), nullable=True),
        pa.field("tickers_derivatives", pa.int64(), nullable=True),
        pa.field("method", pa.string(), nullable=True),
        pa.field("source_key", pa.string(), nullable=False),
        pa.field("load_ts_utc", pa.timestamp("ms"), nullable=False),
    ]
)

FCT_DAILY_REPORT_STATS_SCHEMA = pa.schema(
    [
        pa.field("as_of_date", pa.date32(), nullable=False),
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("total_notional_usd", pa.float64(), nullable=True),
        pa.field("equity_usd", pa.float64(), nullable=True),
        pa.field("leverage", pa.float64(), nullable=True),
        pa.field("ann_return", pa.float64(), nullable=True),
        pa.field("ann_vol", pa.float64(), nullable=True),
        pa.field("sharpe", pa.float64(), nullable=True),
        pa.field("max_drawdown", pa.float64(), nullable=True),
        pa.field("ruin_prob", pa.float64(), nullable=True),
        pa.field("score", pa.float64(), nullable=True),
        pa.field("alpha_vs_bench", pa.float64(), nullable=True),
        pa.field("source_key", pa.string(), nullable=False),
        pa.field("load_ts_utc", pa.timestamp("ms"), nullable=False),
    ]
)


# ----------------------------
# Schema enforcement
# ----------------------------
@dataclass(frozen=True)
class EnforceResult:
    table: pa.Table
    missing_cols_added: List[str]
    extra_cols_dropped: List[str]


def enforce_schema(df: pd.DataFrame, schema: pa.Schema) -> EnforceResult:
    """
    Enforce:
      - exact column set (schema-driven)
      - exact column order
      - arrow types on write

    Strategy:
      - add missing columns as null
      - drop extra columns
      - build Arrow table using schema (safe=False to allow casts)
    """
    df = df.copy()

    schema_cols = [f.name for f in schema]
    df_cols = list(df.columns)

    missing = [c for c in schema_cols if c not in df_cols]
    extra = [c for c in df_cols if c not in schema_cols]

    for c in missing:
        df[c] = None

    if extra:
        df = df.drop(columns=extra)

    # reorder
    df = df[schema_cols]

    # normalize a few common types for stability (dates + timestamps)
    for field in schema:
        name = field.name
        t = field.type

        if pa.types.is_date32(t):
            # accept strings / timestamps and normalize to date
            df[name] = pd.to_datetime(df[name], errors="coerce").dt.date

        elif pa.types.is_timestamp(t):
            # keep as pandas datetime64[ns] (Arrow will convert)
            df[name] = pd.to_datetime(df[name], errors="coerce", utc=False)

        elif pa.types.is_boolean(t):
            # allow 0/1, "true"/"false"
            if df[name].dtype != "bool":
                df[name] = df[name].map(
                    lambda x: None
                    if pd.isna(x)
                    else (bool(int(x)) if str(x).isdigit() else str(x).strip().lower() in {"true", "1", "yes", "y"})
                )

        elif pa.types.is_integer(t):
            # pandas nullable integer
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")

        elif pa.types.is_floating(t):
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("float64")

        else:
            # strings / other: keep as object; Arrow will handle
            pass

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False, safe=False)
    return EnforceResult(table=table, missing_cols_added=missing, extra_cols_dropped=extra)


SCHEMAS_V1: Dict[str, pa.Schema] = {
    "dim_assets": DIM_ASSETS_SCHEMA,
    "fct_trades": FCT_TRADES_SCHEMA,
    "fct_positions_daily": FCT_POSITIONS_DAILY_SCHEMA,
    "fct_account_pnl_daily": FCT_ACCOUNT_PNL_DAILY_SCHEMA,
    "fct_daily_report_stats": FCT_DAILY_REPORT_STATS_SCHEMA,
}
