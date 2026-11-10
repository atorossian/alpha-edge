"""Immutable local corporate-action events and reconciliation contracts."""
import pandas as pd
import pytest

from alpha_edge.market.ingest_corporate_actions_v2 import (
    acquire_actions, create_local_batch, load_verified_batch,
)
from alpha_edge.market.reconcile_corporate_actions_v2 import reconcile_events


class MockTicker:
    fast_info = {"currency": "EUR"}

    def history(self, **kwargs):
        assert kwargs["auto_adjust"] is False
        assert kwargs["actions"] is True
        assert kwargs["repair"] is False
        index = pd.DatetimeIndex([
            "2026-10-13", "2026-10-14", "2026-10-15"
        ]).tz_localize("Europe/Helsinki")
        return pd.DataFrame({
            "Close": [23.3, 23.4, 23.5],
            "Dividends": [0, 0.223813, 0],
            "Stock Splits": [0, 0, 0],
        }, index=index)


def factory(symbol):
    assert symbol == "KESKOB.HE"
    return MockTicker()


def test_dividend_event_preserves_local_session_and_currency():
    out = acquire_actions(
        "A", "KESKOB.HE", "EUR", start="2026-10-13", end="2026-10-16",
        batch_id="pilot", ticker_factory=factory, retrieved_at="2026-11-10T00:00:00Z",
    )
    assert len(out) == 1
    assert out.iloc[0]["effective_date"] == "2026-10-14"
    assert out.iloc[0]["action_type"] == "DIVIDEND"
    assert out.iloc[0]["unit"] == "currency_per_share"
    assert out.iloc[0]["currency"] == "EUR"
    assert out.iloc[0]["value"] == pytest.approx(.223813)
    assert "+03:00" in out.iloc[0]["source_timestamp"]


def test_candidate_is_immutable_and_checksum_verified(tmp_path):
    universe = pd.DataFrame([{
        "asset_id": "A", "yahoo_ticker": "KESKOB.HE", "currency": "EUR"
    }])
    output = tmp_path / "batches"
    info = create_local_batch(
        universe=universe, start="2026-10-13", end="2026-10-16",
        output=output, batch_id="pilot", ticker_factory=factory,
        retrieved_at="2026-11-10T00:00:00Z",
    )
    events, manifest = load_verified_batch(output / "pilot")
    assert info["s3_writes"] == 0
    assert len(events) == 1 and manifest["dividends"] == 1
    with pytest.raises(FileExistsError):
        create_local_batch(
            universe=universe, start="2026-10-13", end="2026-10-16",
            output=output, batch_id="pilot", ticker_factory=factory,
        )
    with (output / "pilot" / "events.parquet").open("ab") as f:
        f.write(b"extra")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_batch(output / "pilot")


def test_same_day_event_is_correlated_not_automatically_approved():
    gold = pd.DataFrame([
        {"asset_id": "A", "session_date": "2026-10-13", "close": 23.3, "adj_close": 22.3},
        {"asset_id": "A", "session_date": "2026-10-14", "close": 23.4, "adj_close": 22.7},
    ])
    actions = pd.DataFrame([{
        "asset_id": "A", "effective_date": "2026-10-14",
        "action_type": "DIVIDEND", "value": .223813,
    }])
    out = reconcile_events(gold=gold, events=actions)
    assert len(out) == 1
    assert out.iloc[0]["nearest_action_type"] == "DIVIDEND"
    assert out.iloc[0]["evidence_status"] == "same_day_reported_action"


def test_absent_yahoo_actions_columns_is_not_treated_as_zero_events():
    class MissingActions(MockTicker):
        def history(self, **kwargs):
            return super().history(**kwargs).drop(columns=["Dividends"])
    with pytest.raises(ValueError, match="action columns missing"):
        acquire_actions(
            "A", "KESKOB.HE", "EUR", start="2026-10-13", end="2026-10-16",
            batch_id="pilot", ticker_factory=lambda _: MissingActions(),
        )
