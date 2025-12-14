import os
import pandas as pd
from dataclasses import dataclass
from typing import Dict
from pathlib import Path


def load_closes_from_folder(folder: str | Path, tickers: list[str]) -> pd.DataFrame:
    folder = Path(folder)
    dfs = {}
    for ticker in tickers:
        # You can change the pattern if your naming differs
        candidates = list(folder.glob(f"{ticker}_ohlcv.csv"))
        if not candidates:
            continue
        path = candidates[0]
        df = pd.read_csv(path)
        cols_lower = {c.lower(): c for c in df.columns}
        date_col = cols_lower.get("date", list(df.columns)[0])
        close_col = cols_lower.get("adj_close") or cols_lower.get("close")
        if close_col is None:
            raise ValueError(f"No adj_close/close in {path}")

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        dfs[ticker] = df[close_col].rename(ticker)

    if not dfs:
        raise ValueError("No price files found for requested tickers")

    closes = pd.concat(dfs.values(), axis=1).dropna()
    return closes

@dataclass
class Asset:
    ticker: str
    name: str
    asset_class: str   # equity / credit / commodity / crypto / etc.
    role: str          # beta / sector_tech / gold / etc.
    region: str        # US / Europe / EM / Global / ...
    max_weight: float  # e.g. 0.25
    min_weight: float  # e.g. 0.00
    include: bool      # from CSV


def load_universe(csv_path: str | Path) -> Dict[str, Asset]:
    df = pd.read_csv(csv_path)
    assets: Dict[str, Asset] = {}
    for _, row in df.iterrows():
        if not bool(row.get("include", 1)):
            continue
        asset = Asset(
            ticker=row["ticker"],
            name=row.get("name", row["ticker"]),
            asset_class=row.get("asset_class", "unknown"),
            role=row.get("role", "unknown"),
            region=row.get("region", "unknown"),
            max_weight=float(row.get("max_weight", 1.0)),
            min_weight=float(row.get("min_weight", 0.0)),
            include=bool(row.get("include", 1)),
        )
        assets[asset.ticker] = asset
    return assets