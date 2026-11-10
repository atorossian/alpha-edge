"""Tests for missing price fields, exchange-session coverage and source preservation."""
import pandas as pd

from alpha_edge.market.audit_missing_ohlcv_v2 import report_missing


def test_missing_price_and_trading_session_are_both_reported():
    bars = pd.DataFrame([
        {"asset_id": "A", "provider_symbol": "GLD", "session_date": "2026-10-05",
         "open": 100, "high": 101, "low": 99, "close": 100, "adj_close": 100, "volume": 1000},
        {"asset_id": "A", "provider_symbol": "GLD", "session_date": "2026-10-07",
         "open": 101, "high": 102, "low": 100, "close": 101, "adj_close": None, "volume": 1100},
    ])
    u = pd.DataFrame([{"asset_id": "A", "yahoo_ticker": "GLD", "calendar_id": "NYSE"}])
    fields, sessions = report_missing(bars, u)
    assert fields["field"].tolist() == ["adj_close"]
    assert sessions["session_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-06"]
    assert sessions["issue"].tolist() == ["missing_expected_session"]


def test_invalid_nonnumeric_close_is_reported_not_fatal():
    bars = pd.DataFrame([{
        "asset_id": "A", "provider_symbol": "GLD", "session_date": "2026-10-05",
        "open": 100, "high": 101, "low": 99, "close": "invalid",
        "adj_close": 100, "volume": 1000,
    }])
    u = pd.DataFrame([{"asset_id": "A", "yahoo_ticker": "GLD", "calendar_id": "NYSE"}])
    fields, sessions = report_missing(bars, u)
    assert fields["field"].tolist() == ["close"]
    assert sessions.empty
