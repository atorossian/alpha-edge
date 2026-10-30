"""Independent return-gap expectations derived from source campaign OHLCV dates."""
from __future__ import annotations

import pandas as pd


def expected_return_break_dates(source_dates, quarantined_dates):
    """First retained observation after each excluded source observation.

    Source dates are taken from the *original* campaign, not the repaired
    campaign or its declared return_break_dates. End-of-history quarantine
    correctly produces no break.
    """
    source = pd.DatetimeIndex(pd.to_datetime(list(source_dates), utc=True)).normalize()
    excluded = pd.DatetimeIndex(pd.to_datetime(list(quarantined_dates), utc=True)).normalize()
    if source.has_duplicates or excluded.has_duplicates:
        raise ValueError("Duplicate dates in source timeline or quarantine")
    if not excluded.isin(source).all():
        raise ValueError("Quarantine contains a date absent from source timeline")
    retained = source[~source.isin(excluded)].sort_values()
    positions = retained.searchsorted(excluded, side="right")
    return sorted({
        retained[int(pos)].strftime("%Y-%m-%d")
        for pos in positions if int(pos) < len(retained)
    })
