"""Reusable source-currency FX reconstruction and publication-readiness diagnostics.

S3 artifacts are staged by caller. No active writes or inferred missing prices.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def fx_symbol(currency):
    c = str(currency or "").upper()
    if c == "USD":
        return None, None
    if len(c) != 3 or not c.isalpha():
        raise ValueError("Unsupported quote currency: " + repr(currency))
    return c + "USD=X", "USD" + c + "=X"


def normalize_fx_history(frame, inverse=False):
    x = frame.copy()
    if "Close" not in x or "date" not in x:
        raise ValueError("FX history lacks Close or date")
    x["date"] = pd.to_datetime(x["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    x["Close"] = pd.to_numeric(x["Close"], errors="coerce")
    x = x.dropna(subset=["date", "Close"])
    x = x.loc[np.isfinite(x["Close"]) & x["Close"].gt(0)]
    if x.duplicated("date").any():
        raise ValueError("Duplicate FX date")
    x["fx_to_usd"] = 1 / x["Close"] if inverse else x["Close"]
    return x[["date", "fx_to_usd"]].sort_values("date")


def convert_raw_with_fx(raw, fx, max_lag_days=3):
    """Backward-only FX as-of matching; never invent or interpolate quotations.

    Each equity observation retains the FX observation date and calendar lag.
    A missing or excessively stale quote remains missing, and staging then fails closed.
    """
    if not 0 <= max_lag_days <= 7:
        raise ValueError("max_lag_days must be 0..7")
    x = raw.copy()
    x["date"] = pd.to_datetime(x["date"], utc=True).dt.tz_localize(None).dt.normalize()
    f = fx.copy()
    f["date"] = pd.to_datetime(f["date"], utc=True).dt.tz_localize(None).dt.normalize()
    if x.duplicated("date").any() or f.duplicated("date").any():
        raise ValueError("Duplicate date in equity or FX input")
    f = f.rename(columns={"date": "fx_quote_date"})
    x = pd.merge_asof(x.sort_values("date"), f.sort_values("fx_quote_date"),
                      left_on="date", right_on="fx_quote_date", direction="backward",
                      tolerance=pd.Timedelta(days=max_lag_days))
    x["fx_age_calendar_days"] = (x["date"] - x["fx_quote_date"]).dt.days
    for col in ("Open", "High", "Low", "Close", "Adj Close"):
        x[col + "_usd"] = pd.to_numeric(x[col], errors="coerce") * x["fx_to_usd"]
    return x


def corporate_actions_report(raw):
    out = {}
    for column in ("Stock Splits", "Dividends", "Capital Gains"):
        if column not in raw:
            out[column] = {"events": 0, "available": False}
            continue
        values = pd.to_numeric(raw[column], errors="coerce").fillna(0)
        out[column] = {"events": int(values.ne(0).sum()), "available": True,
                       "examples": [{"date": str(d.date()), "value": float(v)}
                                    for d, v in zip(raw.loc[values.ne(0), "date"].head(20),
                                                    values[values.ne(0)].head(20))]}
    return out


def coverage_review(old, candidate, comparison, max_disagreement=0):
    old_days = set(pd.to_datetime(old["date"]).dt.normalize()) if not old.empty else set()
    new_days = set(pd.to_datetime(candidate["date"]).dt.normalize())
    old_only = sorted(old_days - new_days)
    new_only = sorted(new_days - old_days)
    # Old-only can include bad synthetic weekend prices, but cannot be retired automatically.
    reasons = []
    if comparison.get("disagreements_gt_1pct", 0) > max_disagreement:
        reasons.append("MATERIAL_OVERLAPPING_PRICE_DIFFERENCE")
    if not new_days:
        reasons.append("EMPTY_CANDIDATE")
    if old_only:
        reasons.append("OLD_ONLY_DATES_REQUIRE_EXCHANGE_CALENDAR_VALIDATION")
    if pd.to_numeric(candidate["close_adjusted_usd"], errors="coerce").isna().any():
        reasons.append("MISSING_ADJUSTED_USD_QUOTES")
    if pd.to_numeric(candidate["close_adjusted_usd"], errors="coerce").le(0).any():
        reasons.append("INVALID_ADJUSTED_USD_QUOTES")
    return {"status": "REVIEW_REQUIRED" if reasons else "COVERAGE_CHECK_PASSED",
            "not_publication_approval": True,
            "reasons": reasons,
            "old_only_dates": [str(x.date()) for x in old_only],
            "new_only_dates": [str(x.date()) for x in new_only],
            "old_only_weekend_count": sum(x.weekday() >= 5 for x in old_only),
            "old_only_weekday_count": sum(x.weekday() < 5 for x in old_only)}
