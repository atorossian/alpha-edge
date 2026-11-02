"""Reusable, fail-closed validation of historical rebuild publication evidence."""
from __future__ import annotations
import pandas as pd
import numpy as np


def validate_candidate(prices, returns, inception_date=None):
    issues = []
    if prices.empty or returns.empty:
        return ["EMPTY_CANDIDATE"]
    for frame, required in ((prices, ("date", "asset_id", "close_adjusted_usd")),
                            (returns, ("date", "asset_id", "ret_close_adjusted_usd"))):
        if not set(required).issubset(frame):
            return ["MISSING_REQUIRED_COLUMNS"]
        dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        if dates.isna().any() or dates.duplicated().any():
            issues.append("INVALID_OR_DUPLICATE_DATES")
    if issues:
        return sorted(set(issues))
    p = prices.sort_values("date")
    r = returns.sort_values("date")
    if not pd.Index(pd.to_datetime(p["date"], utc=True)).equals(pd.Index(pd.to_datetime(r["date"], utc=True))):
        issues.append("PRICE_RETURN_DATE_MISMATCH")
    px = pd.to_numeric(p["close_adjusted_usd"], errors="coerce")
    if (~np.isfinite(px)).any() or px.le(0).any():
        issues.append("MISSING_OR_INVALID_ADJUSTED_USD_QUOTES")
    if inception_date and pd.to_datetime(p["date"], utc=True).lt(pd.Timestamp(inception_date, tz="UTC")).any():
        issues.append("PRE_INCEPTION_SOURCE_ROWS")
    expected = px.pct_change(fill_method=None)
    actual = pd.to_numeric(r["ret_close_adjusted_usd"], errors="coerce")
    if pd.notna(actual.iloc[0]):
        issues.append("FIRST_RETURN_MUST_BE_NULL")
    if ((expected.iloc[1:] - actual.iloc[1:]).abs() > 1e-8).any():
        issues.append("CANDIDATE_RETURN_MISMATCH")
    return sorted(set(issues))


def validate_old_only_against_calendar(old_dates, new_dates, calendar_name):
    """Verifies ALL retired dates are non-sessions, not merely weekdays."""
    import pandas_market_calendars as mcal
    old = pd.DatetimeIndex(pd.to_datetime(old_dates, utc=True)).tz_localize(None).normalize()
    new = pd.DatetimeIndex(pd.to_datetime(new_dates, utc=True)).tz_localize(None).normalize()
    extras = old.difference(new)
    if len(extras) == 0:
        return {"calendar": calendar_name, "old_only": 0, "sessions": [], "non_sessions": [], "status": "PASSED"}
    cal = mcal.get_calendar(calendar_name)
    valid = pd.DatetimeIndex(cal.valid_days(start_date=extras.min(), end_date=extras.max())).tz_localize(None).normalize()
    sessions = extras.intersection(valid)
    non_sessions = extras.difference(valid)
    return {"calendar": calendar_name, "old_only": len(extras),
            "sessions": [str(x.date()) for x in sessions],
            "non_sessions": [str(x.date()) for x in non_sessions],
            "status": "PASSED" if len(sessions) == 0 else "REVIEW_REQUIRED"}
