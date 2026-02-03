# backtest/data_view.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable

from alpha_edge.core.market_store import MarketStore

def slice_returns_wide_asof(
    wide: pd.DataFrame,
    *,
    end_date: str,
    lookback_days: int,
) -> pd.DataFrame:
    idx = pd.to_datetime(wide.index, errors="coerce")
    wide = wide.copy()
    wide.index = pd.DatetimeIndex(idx).tz_localize(None).normalize()

    end_ts = pd.Timestamp(end_date).normalize()
    # take last lookback_days rows ending at end_date
    sub = wide.loc[wide.index <= end_ts]
    if sub.empty:
        raise RuntimeError("No returns data at/before end_date.")
    if sub.shape[0] > lookback_days:
        sub = sub.iloc[-lookback_days:]
    return sub

def get_close_prices_asof(
    *,
    store: MarketStore,
    asset_ids: Iterable[str],
    as_of: str,
    lookback_days: int = 10,
) -> Dict[str, float]:
    """
    Returns adj_close_usd at last available date <= as_of for each asset_id.
    Uses a small lookback window to tolerate missing market days.
    """
    as_of_ts = pd.Timestamp(as_of).normalize()
    start = (as_of_ts - pd.Timedelta(days=int(lookback_days) * 2)).strftime("%Y-%m-%d")
    end = as_of_ts.strftime("%Y-%m-%d")

    df = store.read_ohlcv_usd(
        asset_ids=list(asset_ids),
        start=start,
        end=end,
        columns=["date", "asset_id", "adj_close_usd"],
    )
    if df is None or df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df["adj_close_usd"] = pd.to_numeric(df["adj_close_usd"], errors="coerce")
    df = df.dropna(subset=["date", "asset_id", "adj_close_usd"])
    df = df[df["date"] <= as_of_ts]
    if df.empty:
        return {}

    # last close per asset_id
    df = df.sort_values(["asset_id", "date"])
    last = df.groupby("asset_id", as_index=False)["adj_close_usd"].last()
    out = dict(zip(last["asset_id"].astype(str), last["adj_close_usd"].astype(float)))
    return {k: float(v) for k, v in out.items() if np.isfinite(v)}
