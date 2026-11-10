"""Offline contract tests: Yahoo Bronze acquisition must preserve source evidence."""
from pathlib import Path

import pandas as pd
import pytest

from alpha_edge.market.ingest_bronze_yahoo_v2 import (
    _save_batch,
    acquire_yahoo_bars,
    acquire_yahoo_fx_quote,
    raw_bars_from_frame,
    run_batch,
)


class FakeYahooTicker:
    def __init__(self, symbol):
        self.symbol = symbol
        self.fast_info = {"currency": "EUR", "exchange": "XETRA"}
        self.calls = []

    def history(self, **kwargs):
        self.calls.append(kwargs)
        index = pd.DatetimeIndex([
            "2026-10-01 00:00:00+09:00",
            "2026-10-02 00:00:00+09:00",
        ])
        return pd.DataFrame({
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [100.5, 101.5],
            "Volume": [10, 11],
            "Dividends": [0.0, 0.5],
            "Stock Splits": [0, 0],
        }, index=index)


def test_raw_bars_keep_source_session_date_and_quote_values():
    fake = FakeYahooTicker("TEST.DE")
    result, meta = acquire_yahoo_bars(
        "EQ123", "TEST.DE", start="2026-10-01", end="2026-10-03",
        batch_id="batch_1", ticker_factory=lambda _: fake,
    )
    assert result["session_date"].tolist() == ["2026-10-01", "2026-10-02"]
    assert result["source_timestamp"].iloc[0].endswith("+09:00")
    assert result["close"].tolist() == [101.0, 102.0]
    assert result["adj_close"].tolist() == [100.5, 101.5]
    assert result["currency_original"].eq("EUR").all()
    assert result["dividends"].tolist() == [0.0, 0.5]
    assert result["stock_splits"].tolist() == [0, 0]
    assert meta["currency_metadata_source"] == "provider"
    assert fake.calls[0]["auto_adjust"] is False
    assert fake.calls[0]["repair"] is False
    assert fake.calls[0]["actions"] is True


def test_provider_currency_disagreement_fails_closed():
    with pytest.raises(ValueError, match="conflicts"):
        acquire_yahoo_bars(
            "EQ123", "TEST.DE", start="2026-10-01", end=None,
            batch_id="batch_1", ticker_factory=FakeYahooTicker,
            currency_hint="USD",
        )


def test_missing_currency_never_assumes_usd():
    class UnknownCurrency(FakeYahooTicker):
        def __init__(self, symbol):
            super().__init__(symbol)
            self.fast_info = {}
            self.info = {}
    with pytest.raises(ValueError, match="currency unknown"):
        acquire_yahoo_bars(
            "EQ123", "TEST.DE", start="2026-10-01", end=None,
            batch_id="batch_1", ticker_factory=UnknownCurrency,
        )


def test_source_duplicate_dates_are_retained():
    frame = FakeYahooTicker("TEST.DE").history()
    duplicate = pd.concat([frame, frame.iloc[[1]]])
    raw = raw_bars_from_frame(
        duplicate, asset_id="EQ123", symbol="TEST.DE",
        currency="EUR", batch_id="abc",
        retrieved_at_utc="2026-11-10T00:00:00Z",
    )
    assert len(raw) == 3
    assert raw["session_date"].tolist().count("2026-10-02") == 2


def test_raw_fx_quote_not_inverted():
    raw, _ = acquire_yahoo_fx_quote(
        "EURUSD=X", start="2026-10-01", end="2026-10-03",
        batch_id="fx1", ticker_factory=FakeYahooTicker,
    )
    assert raw["quote_close"].tolist() == [101.0, 102.0]
    assert "fx_to_usd" not in raw.columns
    assert raw["provider_symbol"].eq("EURUSD=X").all()


def test_batch_writes_source_tables_and_manifest_last(tmp_path):
    manifest = run_batch(
        universe=pd.DataFrame({
            "asset_id": ["EQ123"], "yahoo_ticker": ["TEST.DE"],
        }),
        fx_symbols=["EURUSD=X"], start="2026-10-01",
        end="2026-10-03", batch_id="pilot_1",
        output=str(tmp_path), ticker_factory=FakeYahooTicker,
    )
    root = Path(manifest["local_path"])
    assert (root / "manifest.json").is_file()
    assert (root / "source_ohlcv.parquet").is_file()
    assert (root / "source_fx_quotes.parquet").is_file()
    assert len(pd.read_parquet(root / "source_ohlcv.parquet")) == 2
    with pytest.raises(FileExistsError):
        run_batch(
            universe=pd.DataFrame({
                "asset_id": ["EQ123"], "yahoo_ticker": ["TEST.DE"],
            }),
            fx_symbols=[], start="2026-10-01", end="2026-10-03",
            batch_id="pilot_1", output=str(tmp_path),
            ticker_factory=FakeYahooTicker,
        )


def test_s3_legacy_prefix_rejected_before_local_side_effects(tmp_path):
    with pytest.raises(ValueError, match="legacy production"):
        _save_batch(
            destination=str(tmp_path), batch_id="x", tables={},
            provenance=[], bucket="any", upload=True, market_root="market",
        )
    assert not (tmp_path / "x").exists()
