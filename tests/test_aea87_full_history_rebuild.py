import pytest
import pandas as pd
from alpha_edge.market.aea87_full_history_rebuild import build_candidate, compare_histories


def sample():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-11-28", "2024-11-29", "2024-12-01"]),
        "Open": [1., 6., 8.], "High": [1., 6., 8.],
        "Low": [1., 6., 8.], "Close": [1., 6., 8.],
        "Adj Close": [1., 6., 8.], "Volume": [5., 6., 7.]})


def test_inception_and_return_no_pre_inception_bridge():
    prices, returns, status = build_candidate(sample(), "CRYPTO:HYPE-USD", "USD", "2024-11-29")
    assert status == "STAGED_FOR_REVIEW"
    assert list(prices["date"].dt.strftime("%Y-%m-%d")) == ["2024-11-29", "2024-12-01"]
    assert pd.isna(returns["ret_close_adjusted_usd"].iloc[0])
    assert abs(returns["ret_close_adjusted_usd"].iloc[1] - (8/6-1)) < 1e-9


def test_non_usd_requires_fx():
    prices, returns, status = build_candidate(sample(), "A", "EUR")
    assert prices is None and returns is None
    assert status == "NON_USD_REQUIRES_HISTORICAL_FX"


def test_missing_days_not_filled():
    prices, _, _ = build_candidate(sample(), "A", "USD")
    assert len(prices) == 3


def test_compare_detects_bad_existing_price():
    new, _, _ = build_candidate(sample(), "A", "USD")
    old = new.copy()
    old.loc[1, "close_adjusted_usd"] = .01
    report = compare_histories(old, new)
    assert report["disagreements_gt_1pct"] == 1


def test_all_null_yahoo_placeholder_not_in_usd_candidate():
    import pandas as pd
    from alpha_edge.market.aea87_full_history_rebuild import build_candidate
    frame = pd.DataFrame({
        "date":pd.to_datetime(["2026-02-01", "2026-02-02", "2026-02-03"]),
        "Open":[10.0, None, 12.0], "High":[11.0, None, 13.0],
        "Low":[9.0, None, 11.0], "Close":[10.0, None, 12.0],
        "Adj Close":[10.0, None, 12.0], "Volume":[1.0, 0.0, 1.0]})
    prices, returns, status = build_candidate(frame, "CRYPTO:HYPE-USD", "USD")
    assert status == "STAGED_FOR_REVIEW"
    assert prices["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-02-01", "2026-02-03"]
    assert returns["ret_close_adjusted_usd"].iloc[1] == pytest.approx(0.2)
