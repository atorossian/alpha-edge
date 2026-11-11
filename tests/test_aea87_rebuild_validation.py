import pandas as pd
from alpha_edge.market.aea87_rebuild_validation import normalize_fx_history, coverage_review
from alpha_edge.market.aea87_full_history_rebuild import build_candidate


def _raw():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "Open": [10.0, 11.0],
            "High": [10.0, 11.0],
            "Low": [10.0, 11.0],
            "Close": [10.0, 11.0],
            "Adj Close": [10.0, 11.0],
            "Volume": [5, 6],
        }
    )


def test_inverse_fx():
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-05"]), "Close": [2.0, 4.0]})
    fx = normalize_fx_history(frame, inverse=True)
    prices, returns, status = build_candidate(_raw(), "A", "EUR", fx=fx)
    assert status == "STAGED_FOR_REVIEW"
    assert list(prices["close_adjusted_usd"]) == [5.0, 2.75]


def test_recent_fx_is_carried_backward_only_with_provenance():
    fx = normalize_fx_history(pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "Close": [1.1]}))
    prices, returns, status = build_candidate(_raw(), "A", "EUR", fx=fx)
    assert status == "STAGED_FOR_REVIEW"
    assert prices is not None and returns is not None
    assert list(prices["fx_age_calendar_days"]) == [0, 3]
    assert list(prices["fx_quote_date"].dt.strftime("%Y-%m-%d")) == ["2026-01-02"] * 2


def test_old_extra_dates_require_review():
    fresh, _, _ = build_candidate(_raw(), "A", "USD")
    old = pd.concat([fresh, fresh.iloc[[0]].assign(date=pd.Timestamp("2026-01-03"))], ignore_index=True)
    from alpha_edge.market.aea87_full_history_rebuild import compare_histories

    report = coverage_review(old, fresh, compare_histories(old, fresh))
    assert report["old_only_weekend_count"] == 1
    assert report["status"] == "REVIEW_REQUIRED"


def test_fx_weekend_gap_uses_recent_prior_quote():
    raw = _raw()
    fx = normalize_fx_history(pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "Close": [1.2]}))
    prices, returns, status = build_candidate(raw, "A", "EUR", fx=fx)
    assert status == "STAGED_FOR_REVIEW"
    assert list(prices["close_adjusted_usd"].round(6)) == [12.0, 13.2]


def test_fx_stale_quote_rejected():
    fx = normalize_fx_history(pd.DataFrame({"date": [pd.Timestamp("2025-12-25")], "Close": [1.2]}))
    prices, returns, status = build_candidate(_raw(), "A", "EUR", fx=fx)
    assert status == "MISSING_RECENT_HISTORICAL_FX"
    assert prices is None and returns is None
