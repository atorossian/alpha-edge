"""Actual current-universe configuration contract regression.

These five files are checked into the repo and are the ONLY registry
configuration sources. They are NOT market-price data or historical lake
artifacts. CI reads them without creating any local data outputs.
"""

from pathlib import Path

import pandas_market_calendars as mcal

from alpha_edge.market.lakehouse_v3.registry_v3 import (
    MARKET_POLICY,
    SOURCE_FILES,
    build_registry,
    validate_registry,
)

BASE = Path(__file__).resolve().parents[2] / "data" / "universe"


def test_full_current_universe_exactly_five_files_and_membership():
    files = {name: (BASE / name).read_bytes() for name in SOURCE_FILES}
    doc = build_registry(files)
    validate_registry(doc)
    counts = doc["summary"]
    assert counts["universe_total"] == 2207
    assert counts["included"] == 2028
    assert counts["excluded"] == 179
    assert counts["configured"] + counts["blocked"] == 2028
    assert len(doc["assets"]) == 2028
    assert len(doc["excluded"]) == 179
    assert len({a["asset_id"] for a in doc["assets"]}) == 2028
    assert len({a["yahoo_ticker"] for a in doc["assets"]}) == 2028
    assert doc["refresh_tickers"] == ["HYPE-USD"]
    by_ticker = {a["ticker"]: a for a in doc["assets"]}
    for ticker in ("SPY", "GLD", "BTC-USD", "RENT3", "TRAXIONA", "LOG", "REY", "EUR-USD"):
        assert ticker in by_ticker
    assert by_ticker["EUR-USD"]["yahoo_ticker"] == "EURUSD=X"
    assert not by_ticker["EUR-USD"]["blockers"]
    for ticker in ("HYPE-USD", "KMB"):
        assert by_ticker[ticker]["currency"] == "USD"
        assert not by_ticker[ticker]["blockers"]
    assert by_ticker["DB1"]["yahoo_ticker"] == "DB1.DE"
    assert by_ticker["INBR32"]["yahoo_ticker"] == "INBR34.SA"
    assert not {a["yahoo_ticker"] for a in doc["assets"] if a["blockers"]}
    assert {a["asset_id"] for a in doc["excluded"] if a["exclusion_reason"] == "historical_only"} == {
        "EQH2dea64b7904460c4",
        "EQH4113c94384740eeb",
        "EQH50281759392ec254",
        "EQH0816028bb3741194",
    }
    for ticker, currency in {"USD-CNY": "CNY", "USD-BRL": "BRL", "USD-JPY": "JPY"}.items():
        assert by_ticker[ticker]["currency"] == currency
        assert "missing_or_invalid_quote_currency" not in by_ticker[ticker]["blockers"]
    assert {a["currency"] for a in doc["assets"] if a["currency"]} == {"USD", "EUR", "BRL", "MXN", "CNY", "JPY"}


def test_all_proposed_market_policy_calendar_identifiers_exist():
    invalid = {}
    for market, (calendar_id, _) in MARKET_POLICY.items():
        try:
            mcal.get_calendar(calendar_id)
        except (KeyError, RuntimeError, ValueError) as exc:
            invalid[market] = (calendar_id, type(exc).__name__)
    assert not invalid, f"Unsupported calendar identifiers: {invalid}"
