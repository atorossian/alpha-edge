"""Stage-3 FX normalization, exchange-session and continuity tests."""
import math

import pandas as pd
import pytest

from alpha_edge.market.data_lake_v2 import DataContractError, QualityPolicy
from alpha_edge.market.standardize_bronze_v2 import (
    audit_asset_sessions,
    build_silver_gold_from_bronze,
    normalize_bronze_fx_quotes,
)


def quotes():
    return pd.DataFrame({
        "provider_symbol": ["EURUSD=X"] * 3 + ["USDBRL=X"] * 3 + ["USDMXN=X"] * 3,
        "session_date": ["2026-10-05", "2026-10-06", "2026-10-07"] * 3,
        "quote_close": [1.1, 1.12, 1.13, 5.0, 4.9, 4.8, 20.0, 19.8, 19.7],
    })


def bars():
    rows = []
    for symbol, asset, ccy in [
        ("SPY", "A_US", "USD"),
        ("B3SA3.SA", "A_BR", "BRL"),
        ("KESKOB.HE", "A_EU", "EUR"),
        ("ALSEA.MX", "A_MX", "MXN"),
    ]:
        for day, price in [("2026-10-05", 100.0),
                           ("2026-10-06", 101.0),
                           ("2026-10-07", 102.0)]:
            rows.append({
                "asset_id": asset, "provider_symbol": symbol, "currency_original": ccy,
                "session_date": day, "open": price, "high": price + 1,
                "low": price - 1, "close": price, "adj_close": price,
                "volume": 1000, "source_timestamp": day + "T00:00:00+02:00",
            })
    return pd.DataFrame(rows)


def test_quote_directions_and_exact_original_source_values():
    normalized, evidence = normalize_bronze_fx_quotes(quotes())
    assert len(normalized) == 9
    eur = normalized.query("currency_original == 'EUR'").iloc[0]
    brl = normalized.query("currency_original == 'BRL'").iloc[0]
    mxn = normalized.query("currency_original == 'MXN'").iloc[0]
    assert eur["fx_to_usd"] == pytest.approx(1.1)
    assert brl["fx_to_usd"] == pytest.approx(0.2)
    assert mxn["fx_to_usd"] == pytest.approx(0.05)
    assert brl["quote_close"] == 5.0
    assert brl["fx_quote_direction"] == "inverse"
    assert evidence["fx_validation_status"].eq("accepted").all()


def test_duplicate_fx_dates_do_not_enter_silver():
    q = pd.concat([quotes(), quotes().iloc[[0]]], ignore_index=True)
    accepted, evidence = normalize_bronze_fx_quotes(q)
    assert evidence["fx_validation_status"].eq("duplicate_quote_date").sum() == 2
    assert not accepted.duplicated(["currency_original", "fx_date"]).any()


def test_missing_fx_symbol_fails_closed():
    with pytest.raises(DataContractError, match="Missing FX symbol"):
        normalize_bronze_fx_quotes(quotes().query("provider_symbol != 'USDBRL=X'"))


def test_extreme_fx_observation_quarantined():
    q = quotes()
    q.loc[1, "quote_close"] = 4.0
    accepted, evidence = normalize_bronze_fx_quotes(q)
    row = evidence.loc[evidence["provider_symbol"].eq("EURUSD=X")].iloc[1]
    assert row["fx_validation_status"] == "large_fx_move"
    assert not ((accepted["currency_original"] == "EUR")
                & (accepted["fx_date"] == pd.Timestamp("2026-10-06"))).any()


def test_exchange_sessions_and_missing_expected_dates_are_flagged():
    sample = bars().query("provider_symbol == 'SPY'").copy()
    sample = sample.loc[sample["session_date"].ne("2026-10-06")]
    flags, gaps = audit_asset_sessions(sample, calendars={"SPY": "NYSE"})
    assert gaps["session_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-10-06"]
    assert flags.iloc[-1]["previous_expected_session_missing"]


def test_repaired_return_must_not_bridge_missing_expected_session():
    sample = bars().query("provider_symbol == 'SPY'").copy()
    sample = sample.loc[sample["session_date"].ne("2026-10-06")]
    # No FX quotes are required for a USD-only selected asset.
    results = build_silver_gold_from_bronze(
        sample, pd.DataFrame(columns=["provider_symbol", "session_date", "quote_close"]),
        pairs={}, calendars={"SPY": "NYSE"},
    )
    gold = results["gold"]
    assert len(gold) == 2
    assert pd.isna(gold.iloc[-1]["ret_simple_close_adjusted_usd"])
    assert gold.iloc[-1]["quality_status"] == "quarantined"
    assert len(results["session_issues"]) == 1


def test_mixed_currencies_preserve_fx_lineage_and_arithmetic_returns():
    results = build_silver_gold_from_bronze(bars(), quotes(),
                                            calendars={"SPY": "NYSE", "B3SA3.SA": "BMF",
                                                       "KESKOB.HE": "XHEL", "ALSEA.MX": "XMEX"})
    silver = results["silver"]
    assert len(silver) == 12
    assert silver["quality_status"].eq("accepted").all()
    brl = silver.loc[silver["currency_original"].eq("BRL")].sort_values("session_date")
    assert brl.iloc[0]["close_adjusted_usd"] == pytest.approx(20.0)
    assert brl.iloc[0]["fx_quote_symbol"] == "USDBRL=X"
    assert brl.iloc[0]["quote_close"] == 5.0
    assert brl.iloc[0]["fx_quote_direction"] == "inverse"
    gold_brl = results["gold"].loc[results["gold"]["currency_original"].eq("BRL")]
    assert gold_brl.iloc[1]["ret_simple_close_adjusted_usd"] == pytest.approx(
        (101 / 4.9) / (100 / 5.0) - 1
    )


def test_utc_timestamp_conversion_preserves_exchange_session_date():
    source = bars().query("provider_symbol == 'SPY'").copy()
    source["source_timestamp"] = ["2026-10-05T00:00:00+09:00",
                                  "2026-10-06T00:00:00+09:00",
                                  "2026-10-07T00:00:00+09:00"]
    result = build_silver_gold_from_bronze(
        source,
        pd.DataFrame(columns=["provider_symbol", "session_date", "quote_close"]),
        calendars={"SPY": "NYSE"}, pairs={},
    )
    silver = result["silver"].sort_values("session_date")
    assert silver.iloc[0]["session_date"] == pd.Timestamp("2026-10-05")
    assert silver.iloc[0]["source_timestamp_utc"] == pd.Timestamp(
        "2026-10-04T15:00:00Z"
    )
    assert silver.iloc[0]["source_timestamp_semantics"] == (
        "provider_daily_bar_label_not_close_time"
    )
