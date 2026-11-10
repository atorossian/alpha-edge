"""Full population configuration must preserve unresolved assets as findings."""
import pandas as pd

from alpha_edge.market.prepare_full_universe_registry_v2 import build


def test_unmapped_exchange_and_currency_remain_visible_not_silently_skipped():
    source = pd.DataFrame([
        {"asset_id": "A", "yahoo_ticker": "SPY", "currency": "USD",
         "exchange": "PCX", "include": 1},
        {"asset_id": "B", "yahoo_ticker": "UNKNOWN.X", "currency": "ZZZ",
         "exchange": "UNKNOWN", "include": 1},
    ])
    mapping = pd.DataFrame([{"exchange": "PCX", "calendar_id": "NYSE"}])
    fx = pd.DataFrame([{"currency": "EUR", "provider_symbol": "EURUSD=X",
                        "direction": "direct"}])
    ready, unresolved, all_rows = build(source, mapping, fx)
    assert ready["asset_id"].tolist() == ["A"]
    assert unresolved["asset_id"].tolist() == ["B"]
    assert "missing_calendar_id" in unresolved.iloc[0]["review_reason"]
    assert "unmapped_fx_currency" in unresolved.iloc[0]["review_reason"]
    assert len(all_rows) == 2


def test_duplicate_assets_are_flagged():
    source = pd.DataFrame([
        {"asset_id": "A", "yahoo_ticker": "SPY", "currency": "USD",
         "exchange": "PCX", "include": 1},
        {"asset_id": "A", "yahoo_ticker": "GLD", "currency": "USD",
         "exchange": "PCX", "include": 1},
    ])
    mapping = pd.DataFrame([{"exchange": "PCX", "calendar_id": "NYSE"}])
    fx = pd.DataFrame(columns=["currency", "provider_symbol", "direction"])
    ready, unresolved, _ = build(source, mapping, fx)
    assert ready.empty
    assert len(unresolved) == 2
