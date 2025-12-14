# stats_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf 

@dataclass
class AssetStats:
    ticker: str
    ann_return: float      # annualized mean return
    ann_vol: float         # annualized volatility
    sharpe: float
    max_drawdown: float
    skew: float
    kurtosis: float


def compute_daily_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Simple daily returns from adj_close prices.
    closes: DataFrame indexed by date, columns=tickers.
    """
    rets = closes.pct_change().dropna(how="all")
    return rets


def compute_asset_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-asset stats from a returns DataFrame.
    returns: DataFrame indexed by date, columns=tickers, values = daily returns.
    Returns a DataFrame with one row per ticker.
    """
    stats_rows: list[Dict[str, Any]] = []

    for ticker in returns.columns:
        r = returns[ticker].dropna()
        if len(r) < 50:
            # skip assets with too little history
            continue

        daily_mean = r.mean()
        daily_std = r.std()
        ann_return = daily_mean * 252
        ann_vol = daily_std * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

        # max drawdown on cumulative wealth curve
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = cum / peak - 1
        max_dd = dd.min()

        stats_rows.append(
            dict(
                ticker=ticker,
                ann_return=ann_return,
                ann_vol=ann_vol,
                sharpe=sharpe,
                max_drawdown=max_dd,
                skew=float(r.skew()),
                kurtosis=float(r.kurtosis()),
            )
        )

    stats_df = pd.DataFrame(stats_rows).set_index("ticker").sort_index()
    return stats_df


def compute_corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix of asset returns.
    """
    return returns.corr()

def compute_lw_cov_df(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Ledoit–Wolf shrinkage covariance matrix for the return series.
    Returns a DataFrame with same index/columns as returns.
    """
    clean = returns.dropna(how="any")
    lw = LedoitWolf().fit(clean.values)
    cov = lw.covariance_
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)