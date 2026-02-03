# backtest/sampling.py
from __future__ import annotations
import numpy as np
import pandas as pd

def sample_as_of_date_from_returns_cache(
    *,
    returns_wide_path: str,
    warmup_days: int = 252 * 5,
    min_forward_days: int = 252,       # e.g. must have at least 1y after start
    seed: int | None = None,
) -> str:
    """
    Samples a random as_of date uniformly from the available dates in returns_wide,
    while ensuring enough warmup history and forward horizon.
    Returns YYYY-MM-DD.
    """
    wide = pd.read_parquet(returns_wide_path, engine="pyarrow")
    idx = pd.to_datetime(wide.index, errors="coerce").dropna()
    idx = pd.DatetimeIndex(idx).sort_values().unique()

    if len(idx) < (warmup_days + min_forward_days + 10):
        raise RuntimeError("Not enough dates in returns_wide to sample with given constraints.")

    lo = warmup_days
    hi = len(idx) - min_forward_days - 1
    if hi <= lo:
        raise RuntimeError("Sampling window empty. Reduce warmup_days or min_forward_days.")

    rng = np.random.default_rng(seed)
    pos = int(rng.integers(lo, hi + 1))
    as_of = pd.Timestamp(idx[pos]).normalize()
    return as_of.strftime("%Y-%m-%d")
