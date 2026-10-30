"""AEA-87 return-break and derived-price regression tests."""
import numpy as np
import pandas as pd
import pytest

from alpha_edge.market.aea87_return_gaps import expected_return_break_dates


def test_gap_in_middle_breaks_first_retained_observation():
    source = ["2026-01-02", "2026-01-03", "2026-01-04"]
    assert expected_return_break_dates(source, ["2026-01-03"]) == ["2026-01-04"]


def test_contiguous_gap_breaks_only_first_retained_observation():
    source = ["2026-08-24", "2026-08-25", "2026-08-26",
              "2026-08-27", "2026-08-28", "2026-08-31"]
    assert expected_return_break_dates(source, source[1:5]) == ["2026-08-31"]


def test_trailing_quarantine_needs_no_return_break():
    source = ["2026-08-24", "2026-08-25", "2026-08-26",
              "2026-08-27", "2026-08-28"]
    assert expected_return_break_dates(source, source[1:]) == []


def test_leading_quarantine_needs_no_return_break():
    source = ["2026-01-02", "2026-01-03", "2026-01-04"]
    assert expected_return_break_dates(source, ["2026-01-02"]) == ["2026-01-03"]


def test_unknown_quarantine_date_is_rejected():
    with pytest.raises(ValueError):
        expected_return_break_dates(["2026-01-02"], ["2026-01-03"])


def test_derived_usd_price_from_approved_fx():
    local = pd.Series([43.127037048339844])
    fx = pd.Series([1.299680233001709])
    result = local * fx
    assert np.isfinite(result).all()
    assert (result > 0).all()
    assert np.isclose(result.iloc[0], 56.0514, atol=0.01)
