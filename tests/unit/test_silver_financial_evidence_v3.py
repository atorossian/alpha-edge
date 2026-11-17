"""Pure Silver financial evidence tests, no local data artifacts."""

import pandas as pd
import pytest

from alpha_edge.market.silver_financial_evidence_v3 import (
    SilverFinancialError,
    extract_source_actions,
    normalize_fx_candidate,
    source_adjustment_flags,
)


def _bars():
    return pd.DataFrame(
        {
            "asset_id": ["EQH-SPY"] * 3,
            "provider_symbol": ["SPY"] * 3,
            "session_date": ["2026-11-12", "2026-11-13", "2026-11-16"],
            "currency_original": ["USD"] * 3,
            "close": [100.0, 101.0, 25.5],
            "adj_close": [100.0, 101.0, 25.5],
            "dividends": [0.0, 0.3, 0.0],
            "stock_splits": [0.0, 0.0, 4.0],
        }
    )


def test_actions_are_source_events_not_price_repairs():
    raw = _bars()
    pristine = raw.copy(deep=True)
    table, info = extract_source_actions(raw)
    assert len(table) == 2
    assert info["reported_splits"] == 1
    assert info["reported_dividends"] == 1
    assert not info["adjustment_applied"]
    assert not info["independently_verified"]
    split = table.query("action_type == 'SPLIT'").iloc[0]
    dividend = table.query("action_type == 'DIVIDEND'").iloc[0]
    assert split["source_value"] == pytest.approx(4.0)
    assert split["effective_date"] == "2026-11-16"
    assert dividend["source_value"] == pytest.approx(0.3)
    assert dividend["source_currency"] == "USD"
    pd.testing.assert_frame_equal(raw, pristine)


def test_missing_actions_do_not_prove_action_free_history():
    raw = _bars()
    raw["stock_splits"] = 0.0
    raw["dividends"] = 0.0
    events, info = extract_source_actions(raw)
    assert events.empty
    assert info["reported_splits"] == 0
    assert not info["independently_verified"]


def test_malformed_action_and_duplicate_source_fail_closed():
    raw = _bars()
    raw.loc[0, "stock_splits"] = -2.0
    with pytest.raises(SilverFinancialError, match="split"):
        extract_source_actions(raw)
    duplicate = pd.concat([_bars(), _bars().iloc[[0]]], ignore_index=True)
    with pytest.raises(SilverFinancialError, match="Duplicate"):
        extract_source_actions(duplicate)


def test_ratio_audit_reuses_v2_without_fixing_prices():
    raw = _bars()
    events, _ = extract_source_actions(raw)
    flags = source_adjustment_flags(raw, events)
    assert isinstance(flags, pd.DataFrame)
    assert flags.empty


def test_explicit_inverse_jpy_and_direct_eur_quotes():
    q = pd.DataFrame(
        {
            "provider_symbol": ["JPY=X", "JPY=X", "EURUSD=X", "EURUSD=X"],
            "session_date": ["2026-11-12", "2026-11-13"] * 2,
            "quote_close": [150.0, 151.0, 1.1, 1.2],
        }
    )
    rates, evidence, info = normalize_fx_candidate(
        q, pairs={"JPY": ("JPY=X", "inverse"), "EUR": ("EURUSD=X", "direct")}
    )
    jpy = rates.query("currency_original == 'JPY'").iloc[0]
    eur = rates.query("currency_original == 'EUR'").iloc[0]
    assert jpy["fx_to_usd"] == pytest.approx(1 / 150)
    assert eur["fx_to_usd"] == pytest.approx(1.1)
    assert len(evidence) == 4
    assert not info["conversion_to_underlying_asset_prices_applied"]
    assert not info["observation_time_verified"]
    assert not info["production_eligible"]


def test_fx_missing_configuration_fails_closed():
    q = pd.DataFrame(columns=["provider_symbol", "session_date", "quote_close"])
    with pytest.raises(SilverFinancialError, match="explicit"):
        normalize_fx_candidate(q, pairs={})
