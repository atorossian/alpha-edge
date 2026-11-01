import pandas as pd
from alpha_edge.market.aea87_anomaly_audit import inspect_asset


def _prices(values):
    return pd.DataFrame({
        "asset_id": ["CRYPTO:TEST"] * len(values),
        "date": pd.date_range("2026-10-01", periods=len(values)),
        "close_adjusted_usd": values,
    })


def _returns(values):
    return pd.DataFrame({
        "asset_id": ["CRYPTO:TEST"] * len(values),
        "date": pd.date_range("2026-10-01", periods=len(values)),
        "ret_adj_close_usd": values,
    })


def test_matched_returns_no_flags():
    flags, frame = inspect_asset(_prices([100, 110, 121]),
                                 _returns([None, .1, .1]), "CRYPTO:TEST")
    assert flags.empty
    assert len(frame) == 3


def test_wrong_stored_return_flagged():
    flags, _ = inspect_asset(_prices([100, 110]),
                             _returns([None, .2]), "CRYPTO:TEST")
    assert len(flags) == 1
    assert "return_mismatch" in flags.iloc[0]["flag"]
    assert flags.iloc[0]["classification"] == "REVIEW_REQUIRED"


def test_large_legitimate_move_is_review_not_corruption():
    flags, _ = inspect_asset(_prices([100, 170]),
                             _returns([None, .7]), "CRYPTO:TEST")
    assert len(flags) == 1
    assert flags.iloc[0]["flag"] == "large_return_review"
    assert flags.iloc[0]["classification"] == "REVIEW_REQUIRED"
