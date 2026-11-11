import pandas as pd
import pytest
from alpha_edge.market.aea87_rebuild_publication_prepare import (
    align_verified_metadata,
    publication_schema_issues,
)
from alpha_edge.market.aea87_hype_price_diagnostic import diagnose


def test_prepare_inherits_verified_identity_and_log_returns():
    active = pd.DataFrame(
        {
            "asset_id": ["A"],
            "ticker": ["HAE"],
            "yahoo_ticker": ["HAE"],
            "currency": ["USD"],
            "date": pd.to_datetime(["2026-01-05"]),
            "ret_log_close_adjusted_usd": [0.01],
        }
    )
    source = pd.DataFrame(
        {
            "asset_id": ["A", "A"],
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "ret_close_adjusted_usd": [None, 0.05],
            "ret_adj_close_usd": [None, 0.05],
        }
    )
    ready = align_verified_metadata(source, active, "returns_usd", provider_symbol="HAE", provider_currency="USD")
    assert publication_schema_issues(ready, active, "returns_usd") == []
    assert pd.isna(ready["ret_log_close_adjusted_usd"].iloc[0])
    assert ready["ret_log_close_adjusted_usd"].iloc[1] == pytest.approx(0.048790164169432)


def test_prepare_rejects_symbol_conflict():
    existing = pd.DataFrame({"ticker": ["HAE"], "yahoo_ticker": ["OTHER"], "currency": ["USD"]})
    candidate = pd.DataFrame(
        {"asset_id": ["A"], "date": [pd.Timestamp("2026-01-05")], "ret_close_adjusted_usd": [None]}
    )
    with pytest.raises(ValueError, match="Provider identity conflict"):
        align_verified_metadata(candidate, existing, "returns_usd", provider_symbol="HAE", provider_currency="USD")


def test_publication_missing_columns_fails_closed():
    candidate = pd.DataFrame({"asset_id": ["A"], "date": [pd.Timestamp("2026-01-05")]})
    issues = publication_schema_issues(candidate, pd.DataFrame(), "ohlcv_usd")
    assert "MISSING_REQUIRED_COLUMN:close_adjusted_usd" in issues


def test_hype_diagnostic_reports_missing_adjusted_source():
    dates = pd.to_datetime(["2024-11-29", "2024-11-30"])
    raw = pd.DataFrame({"date": dates, "Close": [1.0, 2.0], "Adj Close": [None, 2.0]})
    candidate = pd.DataFrame({"date": dates, "close_raw_usd": [1.0, 2.0], "close_adjusted_usd": [None, 2.0]})
    findings = diagnose(raw, candidate, pd.DataFrame())
    assert findings["missing_adjusted_rows"] == 1
    assert findings["observations"][0]["date"] == "2024-11-29"


def test_hae_legacy_price_fields_are_derived_exactly_from_usd_candidate():
    active = pd.DataFrame(
        {
            "asset_id": ["A"],
            "ticker": ["HAE"],
            "yahoo_ticker": ["HAE"],
            "currency": ["USD"],
            "date": [pd.Timestamp("2025-01-06")],
            "open": [1.0],
            "high": [2.0],
            "low": [1.0],
            "close": [2.0],
            "adj_close": [2.0],
            "fx_to_usd": [1.0],
        }
    )
    source = pd.DataFrame(
        {
            "asset_id": ["A"],
            "date": [pd.Timestamp("2025-01-06")],
            "open_raw_usd": [30.0],
            "high_raw_usd": [34.0],
            "low_raw_usd": [29.0],
            "close_raw_usd": [33.0],
            "close_adjusted_usd": [32.0],
            "close_usd": [33.0],
            "adj_close_usd": [32.0],
        }
    )
    out = align_verified_metadata(source, active, "ohlcv_usd", provider_symbol="HAE", provider_currency="USD")
    assert publication_schema_issues(out, active, "ohlcv_usd") == []
    assert float(out["open"].iloc[0]) == 30
    assert float(out["high"].iloc[0]) == 34
    assert float(out["low"].iloc[0]) == 29
    assert float(out["close"].iloc[0]) == 33
    assert float(out["adj_close"].iloc[0]) == 32
    assert float(out["fx_to_usd"].iloc[0]) == 1
