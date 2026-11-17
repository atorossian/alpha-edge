"""Regression tests: Yahoo metadata refresh must not erase existing universe fields."""

from __future__ import annotations

import pandas as pd

from alpha_edge.universe import universe as module


def _universe_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "row_id": "FX-CNY", "ticker": "USD-CNY", "yahoo_ticker": "CNY=X",
            "asset_class": "fx", "currency": "", "include": 1,
            "market": "ccy_market", "shortName": "USD/CNY",
        },
        {
            "row_id": "FX-BRL", "ticker": "USD-BRL", "yahoo_ticker": "BRL=X",
            "asset_class": "fx_pair", "currency": "BRL", "include": 1,
            "market": "ccy_market", "shortName": "USD/BRL",
        },
        {
            "row_id": "FX-JPY", "ticker": "USD-JPY", "yahoo_ticker": "JPY=X",
            "asset_class": "fx", "currency": "JPY", "include": 1,
            "market": "ccy_market", "shortName": "USD/JPY",
        },
        {
            "row_id": "EQ-SPY", "ticker": "SPY", "yahoo_ticker": "SPY",
            "asset_class": "equity", "currency": "USD", "include": 1,
            "market": "us_market", "shortName": "SPDR S&P 500",
        },
        {
            "row_id": "EQ-BKH", "ticker": "BKH", "yahoo_ticker": "HZN",
            "asset_class": "equity", "currency": "USD", "include": 1,
            "market": "us_market", "shortName": "Black Hills",
        },
    ])


def test_metadata_refresh_preserves_untouched_rows_and_fx_quote_ccy(monkeypatch, tmp_path):
    source = tmp_path / "universe.csv"
    original = _universe_rows()
    original.to_csv(source, index=False)
    calls = []

    def fake_fetch(symbol):
        calls.append(symbol)
        return {
            "symbol": symbol,
            "currency": None,
            "shortName": None,
            "yahoo_ok": False,
            "quoteType": "CURRENCY" if symbol == "CNY=X" else "EQUITY",
        }

    monkeypatch.setattr(module, "fetch_yahoo_meta", fake_fetch)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    result = module.enrich_universe_csv(
        universe_csv=str(source), out_csv=str(source), sleep_s=0,
        ticker_col="row_id", yahoo_col="yahoo_ticker",
        tickers_subset=["FX-CNY"],
    )
    assert calls == ["CNY=X"]
    rows = result.set_index("row_id")
    assert rows.loc["FX-CNY", "currency"] == "CNY"
    assert rows.loc["FX-BRL", "currency"] == "BRL"
    assert rows.loc["FX-JPY", "currency"] == "JPY"
    assert rows.loc["EQ-SPY", "currency"] == "USD"
    assert rows.loc["EQ-BKH", "currency"] == "USD"
    assert rows.loc["EQ-SPY", "shortName"] == "SPDR S&P 500"
    assert rows.loc["FX-CNY", "shortName"] == "USD/CNY"
    assert rows.loc["EQ-BKH", "yahoo_ticker"] == "HZN"
    assert rows["include"].tolist() == [1, 1, 1, 1, 1]
    saved = pd.read_csv(source).set_index("row_id")
    assert saved.loc["FX-CNY", "currency"] == "CNY"
    assert saved.loc["EQ-SPY", "currency"] == "USD"


def test_yahoo_currency_overrides_existing_for_equity_but_never_blank(monkeypatch, tmp_path):
    source = tmp_path / "universe.csv"
    frame = _universe_rows()
    frame.to_csv(source, index=False)

    def fake_fetch(symbol):
        return {
            "symbol": symbol,
            "currency": "USD" if symbol == "SPY" else "",
            "shortName": "Updated" if symbol == "SPY" else "",
            "yahoo_ok": True,
        }

    monkeypatch.setattr(module, "fetch_yahoo_meta", fake_fetch)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    result = module.enrich_universe_csv(
        universe_csv=str(source), out_csv=str(source), sleep_s=0,
        ticker_col="row_id", yahoo_col="yahoo_ticker",
        tickers_subset=["EQ-SPY", "FX-BRL"],
    )
    rows = result.set_index("row_id")
    assert rows.loc["EQ-SPY", "currency"] == "USD"
    assert rows.loc["EQ-SPY", "shortName"] == "Updated"
    assert rows.loc["FX-BRL", "currency"] == "BRL"
    assert rows.loc["FX-BRL", "shortName"] == "USD/BRL"


def test_fx_derivation_does_not_assign_currency_to_unrelated_symbol(monkeypatch, tmp_path):
    source = tmp_path / "universe.csv"
    rows = _universe_rows()
    rows.loc[rows["ticker"].eq("USD-CNY"), "yahoo_ticker"] = "UNKNOWN=X"
    rows.to_csv(source, index=False)
    monkeypatch.setattr(module, "fetch_yahoo_meta", lambda symbol: {"currency": None})
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    result = module.enrich_universe_csv(
        universe_csv=str(source), out_csv=str(source), sleep_s=0,
        ticker_col="row_id", yahoo_col="yahoo_ticker", tickers_subset=["FX-CNY"],
    )
    assert pd.isna(result.set_index("row_id").loc["FX-CNY", "currency"])
