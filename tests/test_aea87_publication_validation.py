import pandas as pd
from alpha_edge.market.aea87_publication_validation import (
    validate_candidate,
    validate_old_only_against_calendar,
)


def test_valid_candidate():
    dates = pd.to_datetime(["2024-11-29", "2024-11-30"])
    prices = pd.DataFrame({"date": dates, "asset_id": ["H"] * 2, "close_adjusted_usd": [6.0, 7.0]})
    returns = pd.DataFrame({"date": dates, "asset_id": ["H"] * 2, "ret_close_adjusted_usd": [None, 1 / 6]})
    assert validate_candidate(prices, returns, "2024-11-29") == []


def test_missing_adjusted_quote_blocks_publication():
    dates = pd.to_datetime(["2024-11-29", "2024-11-30"])
    prices = pd.DataFrame({"date": dates, "asset_id": ["H"] * 2, "close_adjusted_usd": [6.0, None]})
    returns = pd.DataFrame({"date": dates, "asset_id": ["H"] * 2, "ret_close_adjusted_usd": [None, None]})
    assert "MISSING_OR_INVALID_ADJUSTED_USD_QUOTES" in validate_candidate(prices, returns)


def test_nyse_holidays_are_not_sessions():
    dates = pd.to_datetime(["2025-01-08", "2025-01-09", "2025-01-10"])
    fresh = pd.to_datetime(["2025-01-08", "2025-01-10"])
    result = validate_old_only_against_calendar(dates, fresh, "NYSE")
    assert result["status"] == "PASSED"
    assert result["non_sessions"] == ["2025-01-09"]


def test_nyse_valid_session_blocks_retirement():
    old = pd.to_datetime(["2025-01-06", "2025-01-07"])
    new = pd.to_datetime(["2025-01-06"])
    result = validate_old_only_against_calendar(old, new, "NYSE")
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["sessions"] == ["2025-01-07"]
