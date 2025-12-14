# price_update.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import datetime as dt

import pandas as pd
import yfinance as yf


def fetch_prices_for_universe(
    tickers: Iterable[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Fetch OHLCV for multiple tickers between start and end (YYYY-MM-DD).
    Returns a long DataFrame: date, ticker, open, high, low, close, adj_close, volume.
    """
    start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
    end_str = pd.to_datetime(end).strftime("%Y-%m-%d")

    frames = []
    for t in tickers:
        # yfinance can handle multi-day ranges; we'll stack them
        df = yf.download(t, start=start_str, end=end_str, auto_adjust=False)
        if df.empty:
            continue
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        df = df.reset_index().rename(columns={"Date": "date"})
        df["ticker"] = t
        # optional: add a currency column if you know it; otherwise assume USD for now
        df["currency"] = "USD"
        frames.append(df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "currency"]])

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "currency"])

    all_data = pd.concat(frames, ignore_index=True)
    all_data["date"] = pd.to_datetime(all_data["date"])
    return all_data


def write_partitions_by_date(
    df: pd.DataFrame,
    dest_root: str | Path,
):
    """
    Takes a long DataFrame with a 'date' column and writes one Parquet per date:

    dest_root/dt=YYYY-MM-DD.parquet
    """
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return

    df["date"] = pd.to_datetime(df["date"])
    for dt_val, group in df.groupby(df["date"].dt.date):
        dt_str = pd.to_datetime(dt_val).strftime("%Y-%m-%d")
        path = dest_root / f"dt={dt_str}.parquet"
        # Overwrite or append; for daily runs this is usually overwrite
        group.to_parquet(path, index=False)


def update_daily_partition(
    tickers: Iterable[str],
    dest_root: str | Path,
    date: dt.date | None = None,
):
    """
    Fetch prices for a single date and write dt=YYYY-MM-DD.parquet.
    yfinance uses [start, end) so we set end = date + 1 day.
    """
    if date is None:
        date = dt.date.today()

    start = date
    end = date + dt.timedelta(days=1)

    df = fetch_prices_for_universe(
        tickers=tickers,
        start=start.isoformat(),
        end=end.isoformat(),
    )
    write_partitions_by_date(df, dest_root=dest_root)
