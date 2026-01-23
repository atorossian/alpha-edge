from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from alpha_edge.core.market_store import MarketStore


@dataclass
class MarketData:
    ohlcv: pd.DataFrame
    returns: pd.DataFrame
    latest_prices: pd.DataFrame
    latest_returns: pd.DataFrame


def load_latest(store: MarketStore) -> tuple[pd.DataFrame, pd.DataFrame]:
    lp = store.read_latest_prices_snapshot()
    lr = store.read_latest_returns_snapshot()
    return lp, lr


def load_ohlcv_usd_long(
    store: MarketStore,
    tickers: Iterable[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    cols = [
        "date", "ticker", "open", "high", "low", "close", "adj_close", "volume",
        "currency", "fx_to_usd", "close_usd", "adj_close_usd"
    ]
    return store.read_ohlcv_usd(tickers=tickers, start=start, end=end, columns=cols)


def load_returns_usd_long(
    store: MarketStore,
    tickers: Iterable[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    cols = ["date", "ticker", "ret_adj_close_usd"]
    return store.read_returns_usd(tickers=tickers, start=start, end=end, columns=cols)


def returns_matrix(
    returns_long: pd.DataFrame,
    *,
    value_col: str = "ret_adj_close_usd",
) -> pd.DataFrame:
    """
    Convert long returns -> wide matrix (Date index, columns=tickers).
    """
    if returns_long.empty:
        return pd.DataFrame()
    df = returns_long.copy()
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="ticker", values=value_col, aggfunc="last")
    wide = wide.sort_index()
    return wide


def load_market_data_for_engine(
    *,
    bucket: str,
    tickers: Iterable[str],
    start: str,
    end: Optional[str] = None,
) -> MarketData:
    store = MarketStore(bucket=bucket)
    latest_prices, latest_returns = load_latest(store)
    ohlcv = load_ohlcv_usd_long(store, tickers, start=start, end=end)
    rets = load_returns_usd_long(store, tickers, start=start, end=end)
    return MarketData(
        ohlcv=ohlcv,
        returns=rets,
        latest_prices=latest_prices,
        latest_returns=latest_returns,
    )
