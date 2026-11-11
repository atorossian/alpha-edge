"""Contract tests for the isolated market-data lake v2 transformations."""

import pandas as pd
import pytest

from alpha_edge.market.data_lake_v2 import (
    DataContractError,
    QualityPolicy,
    calculate_returns,
    quality_summary,
    standardize_bars,
    validate_raw_bars,
)


def bars():
    return pd.DataFrame(
        {
            "asset_id": ["EQ:EXAMPLE"] * 3,
            "provider_symbol": ["EXAMPLE"] * 3,
            "currency_original": ["EUR"] * 3,
            "session_date": ["2026-10-01", "2026-10-02", "2026-10-05"],
            "open": [99.0, 100.0, 101.0],
            "high": [102.0, 103.0, 104.0],
            "low": [98.0, 99.0, 100.0],
            "close": [100.0, 101.0, 102.0],
            "adj_close": [100.0, 101.0, 102.0],
            "volume": [1000, 1200, 1100],
        }
    )


def fx():
    return pd.DataFrame(
        {
            "currency_original": ["EUR", "EUR"],
            "fx_date": ["2026-10-01", "2026-10-02"],
            "fx_to_usd": [1.10, 1.20],
        }
    )


def test_fx_weekend_policy_and_returns():
    silver = standardize_bars(validate_raw_bars(bars()), fx())
    assert silver["quality_status"].tolist() == ["accepted"] * 3
    assert silver["fx_age_days"].tolist() == [0, 0, 3]
    assert silver["close_adjusted_usd"].tolist() == pytest.approx([110, 121.2, 122.4])
    gold = calculate_returns(silver)
    assert pd.isna(gold.iloc[0]["ret_simple_close_adjusted_usd"])
    assert gold.iloc[1]["ret_simple_close_adjusted_usd"] == pytest.approx(121.2 / 110 - 1)
    assert gold.iloc[2]["ret_log_close_adjusted_usd"] == pytest.approx(__import__("math").log(122.4 / 121.2))


def test_stale_fx_quarantined_and_not_carried_into_returns():
    silver = standardize_bars(validate_raw_bars(bars()), fx(), policy=QualityPolicy(max_fx_age_days=1))
    assert silver.iloc[2]["quality_status"] == "quarantined"
    assert "fx_stale" in silver.iloc[2]["quality_reason"]
    assert pd.isna(silver.iloc[2]["close_adjusted_usd"])
    assert pd.isna(calculate_returns(silver).iloc[2]["ret_simple_close_adjusted_usd"])


def test_duplicate_raw_rows_are_preserved_quarantined_and_refused_for_returns():
    doubled = pd.concat([bars(), bars().iloc[[1]]], ignore_index=True)
    bronze = validate_raw_bars(doubled)
    assert quality_summary(bronze)["duplicate_asset_sessions"] == 2
    assert bronze["quality_status"].eq("quarantined").sum() == 2
    silver = standardize_bars(bronze, fx())
    assert len(silver) == 4
    with pytest.raises(DataContractError, match="duplicate"):
        calculate_returns(silver)


def test_missing_close_does_not_create_zero_return():
    source = bars()
    source.loc[1, "close"] = None
    silver = standardize_bars(validate_raw_bars(source), fx())
    assert silver.iloc[1]["quality_status"] == "quarantined"
    gold = calculate_returns(silver)
    assert gold["ret_simple_close_adjusted_usd"].isna().all()


def test_utc_timezone_daily_labels_do_not_shift_exchange_session():
    source = bars()
    source.loc[0, "session_date"] = "2026-10-01T00:00:00+09:00"
    bronze = validate_raw_bars(source)
    assert str(bronze.iloc[0]["session_date"].date()) == "2026-10-01"


def test_extreme_return_flag_does_not_silently_repair_source():
    source = bars()
    source["currency_original"] = "USD"
    source.loc[2, "adj_close"] = 500
    silver = standardize_bars(
        validate_raw_bars(source), pd.DataFrame(columns=["currency_original", "fx_date", "fx_to_usd"])
    )
    gold = calculate_returns(silver)
    assert gold.iloc[2]["return_quality_status"] == "review"
    assert gold.iloc[2]["close_adjusted_usd"] == 500


def test_fx_direction_is_explicit_and_missing_currency_quarantined():
    source = bars()
    missing = standardize_bars(
        validate_raw_bars(source), pd.DataFrame(columns=["currency_original", "fx_date", "fx_to_usd"])
    )
    assert missing["quality_status"].eq("quarantined").all()
    assert missing["quality_reason"].str.contains("fx_missing").all()
