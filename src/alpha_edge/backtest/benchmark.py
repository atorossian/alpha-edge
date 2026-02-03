# backtest/benchmark.py
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_proxy_daily_returns(
    returns_wide: pd.DataFrame,
    proxy_tickers: list[str],
) -> pd.Series:
    """
    Equal-weight proxy return = mean of component returns each day (ignoring NaNs).
    returns_wide columns are tickers.
    """
    cols = [t for t in proxy_tickers if t in returns_wide.columns]
    if not cols:
        raise RuntimeError("Proxy tickers not found in returns_wide columns.")
    r = returns_wide[cols].copy()
    proxy = r.mean(axis=1, skipna=True)
    proxy = proxy.replace([np.inf, -np.inf], np.nan).dropna()
    proxy.name = "proxy_ret"
    return proxy

def annualized_return_from_daily_returns(r: pd.Series) -> float | None:
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 50:
        return None
    return float(r.mean() * 252.0)
