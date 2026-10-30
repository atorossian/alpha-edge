"""Regression tests for AEA-87 quarantine return discontinuities."""
import numpy as np
import pandas as pd


def test_return_break_after_quarantined_observation():
    # Jan 3 was present in source history but could not be reconstructed.
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"], utc=True),
        "close_adjusted_usd": [100.0, np.nan, 103.0],
    })
    unresolved = prices["close_adjusted_usd"].isna()
    break_dates = prices["date"].shift(-1).loc[unresolved].dropna()
    assert break_dates.dt.strftime("%Y-%m-%d").tolist() == ["2026-01-04"]
    clean = prices.loc[~unresolved].copy()
    ret = np.log(clean["close_adjusted_usd"] / clean["close_adjusted_usd"].shift(1))
    valid = ret.loc[~clean["date"].isin(break_dates)].dropna()
    assert valid.empty


def test_derived_usd_price_from_approved_fx():
    local = pd.Series([43.127037048339844])
    fx = pd.Series([1.299680233001709])
    result = local * fx
    assert np.isfinite(result).all()
    assert (result > 0).all()
    assert np.isclose(result.iloc[0], 56.0514, atol=0.01)
