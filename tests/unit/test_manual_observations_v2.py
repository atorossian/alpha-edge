"""Manual OHLCV correction contract regression tests."""
from pathlib import Path

import pandas as pd
import pytest

from alpha_edge.market.data_lake_v2 import DataContractError
from alpha_edge.market.manual_observations_v2 import append_manual_observations
from alpha_edge.market.standardize_bronze_v2 import build_silver_gold_from_bronze


def sample():
    return pd.DataFrame([
        dict(asset_id="EQHd491d3c90f3be544", provider_symbol="ALSEA.MX",
             currency_original="MXN", session_date=day, open=close,
             high=close + 0.5, low=close - 0.5, close=close,
             adj_close=close, volume=100000,
             source_timestamp=f"{day}T00:00:00-06:00")
        for day, close in [
            ("2026-04-21", 51.10),
            ("2026-04-23", 51.35),
            ("2026-04-24", 52.53),
        ]
    ])


def corrections():
    return pd.DataFrame([{
        "asset_id": "EQHd491d3c90f3be544", "provider_symbol": "ALSEA.MX",
        "currency_original": "MXN", "session_date": "2026-04-22",
        "open": 51.50, "high": 52.06, "low": 51.10, "close": 51.80,
        "volume": 140975, "adj_close": float("nan"),
        "correction_id": "manual-alsea-20260422-v1",
        "reported_by": "user_supplied",
        "verification_status": "pending_independent_verification",
    }])


def fx():
    return pd.DataFrame({
        "provider_symbol": ["USDMXN=X"] * 4,
        "session_date": ["2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24"],
        "quote_close": [17.4, 17.5, 17.6, 17.7],
    })


def test_correction_is_append_only_and_keeps_unknown_adjusted_close():
    original = sample()
    candidate, audit = append_manual_observations(original, corrections())
    assert len(original) == 3
    assert original["session_date"].tolist() == [
        "2026-04-21", "2026-04-23", "2026-04-24"
    ]
    assert len(candidate) == 4
    row = candidate.loc[candidate["observation_origin"].eq("user_supplied_manual")].iloc[0]
    assert row["close"] == 51.80
    assert row["high"] == 52.06
    assert row["low"] == 51.10
    assert row["open"] == 51.50
    assert row["volume"] == 140975
    assert pd.isna(row["adj_close"])
    assert pd.isna(row["source_timestamp"])
    assert audit.iloc[0]["verification_status"] == "pending_independent_verification"


def test_candidate_calendar_gap_closes_but_unverified_adjusted_return_does_not():
    candidate, _ = append_manual_observations(sample(), corrections())
    result = build_silver_gold_from_bronze(
        candidate, fx(), calendars={"ALSEA.MX": "XMEX"},
        pairs={"MXN": ("USDMXN=X", "inverse")},
    )
    gold = result["gold"].sort_values("session_date")
    assert result["session_issues"].empty
    manual = gold.loc[gold["observation_origin"].eq("user_supplied_manual")].iloc[0]
    assert manual["quality_status"] == "quarantined"
    assert "invalid_price" in manual["quality_reason"]
    assert pd.isna(manual["close_adjusted_usd"])
    following = gold.loc[gold["session_date"].eq(pd.Timestamp("2026-04-23"))].iloc[0]
    assert following["quality_status"] == "accepted"
    assert pd.isna(following["ret_simple_close_adjusted_usd"])
    last = gold.loc[gold["session_date"].eq(pd.Timestamp("2026-04-24"))].iloc[0]
    assert last["return_quality_status"] == "accepted"


def test_reject_existing_date_without_replacing_provider_row():
    correction = corrections()
    correction.loc[0, "session_date"] = "2026-04-23"
    with pytest.raises(DataContractError, match="Refusing to overwrite"):
        append_manual_observations(sample(), correction)


def test_reject_invalid_ohlc_and_duplicate_corrections():
    correction = corrections()
    correction.loc[0, "high"] = 51.2
    with pytest.raises(DataContractError, match="range"):
        append_manual_observations(sample(), correction)
    with pytest.raises(DataContractError, match="unique"):
        append_manual_observations(sample(), pd.concat([corrections()] * 2))


def test_reject_unverified_adjusted_close():
    correction = corrections()
    correction.loc[0, "adj_close"] = 51.80
    with pytest.raises(DataContractError, match="adjusted close"):
        append_manual_observations(sample(), correction)
