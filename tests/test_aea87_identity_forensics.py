import numpy as np
import pandas as pd

from alpha_edge.market.aea87_identity_forensics import normalize


def test_missing_prices_not_invalid():
    dates = pd.date_range("2024-11-29", periods=3)
    p = pd.DataFrame({"asset_id": ["A"] * 3, "date": dates, "close_adjusted_usd": [np.nan, 100.0, 110.0]})
    r = pd.DataFrame()
    flags, series = normalize(p, r, "A")
    assert flags.empty
    assert series["missing_adjusted_price"].sum() == 1


def test_log_return_matches_log_convention():
    dates = pd.date_range("2024-11-29", periods=2)
    p = pd.DataFrame({"asset_id": ["A"] * 2, "date": dates, "close_adjusted_usd": [100.0, 110.0]})
    r = pd.DataFrame({"asset_id": ["A"] * 2, "date": dates, "ret_log_close_adjusted_usd": [np.nan, np.log(1.1)]})
    flags, series = normalize(p, r, "A")
    assert flags.empty


def test_hype_pre_inception_flagged():
    dates = pd.to_datetime(["2021-01-01", "2024-11-29", "2024-11-30"])
    p = pd.DataFrame({"asset_id": ["CRYPTO:HYPE-USD"] * 3, "date": dates, "close_adjusted_usd": [1.0, 6.0, 7.0]})
    flags, _ = normalize(p, pd.DataFrame(), "CRYPTO:HYPE-USD", inception="2024-11-29")
    assert "pre_inception" in flags.iloc[0]["flag"]
    assert flags.iloc[0]["date"] == pd.Timestamp("2021-01-01")
