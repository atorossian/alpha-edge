# preprocess_prices.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def align_and_clean_closes(
    closes: pd.DataFrame,
    max_ffill_days: int = 5,
    min_history_days: int = 252,
    max_missing_frac: float = 0.05,
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    - Bounded forward-fill (up to max_ffill_days).
    - Drops assets with too little history or too many missing values.
    Returns: (clean_closes, kept_tickers)
    """
    closes = closes.sort_index()

    # 1) Bounded forward-fill
    gaps = closes.isna().astype(int)
    gap_streak = gaps.groupby((gaps != gaps.shift()).cumsum()).cumsum()
    ffilled = closes.ffill()
    ffilled[gap_streak > max_ffill_days] = np.nan

    # 2) Filter by coverage
    keep_cols = []
    for col in ffilled.columns:
        s = ffilled[col]
        n_total = len(s)
        n_valid = s.notna().sum()
        if n_valid < min_history_days:
            continue
        missing_frac = 1 - n_valid / n_total
        if missing_frac > max_missing_frac:
            continue
        keep_cols.append(col)

    clean = ffilled[keep_cols].dropna(how="all")
    return clean, clean.columns
