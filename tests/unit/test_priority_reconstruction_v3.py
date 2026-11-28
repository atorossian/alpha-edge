import pandas as pd
import pytest

from alpha_edge.market.build_priority_reconstruction_v3 import reconstruct


def fixture():
    dates = pd.date_range("2020-01-01", periods=11).strftime("%Y-%m-%d")
    raw = pd.DataFrame({"session_date": dates, "volume": 100, "stock_splits": 0.0, "dividends": 0.0})
    for col in ("open", "high", "low", "close"):
        raw[col] = 10.0
    raw.loc[5, "stock_splits"] = 2.0
    e = raw.rename(columns={"session_date": "date"}).copy()
    e.loc[:4, ["open", "high", "low", "close"]] = 20.0
    cal = pd.DataFrame({"session_date": dates, "is_expected_session": True})
    return raw, e, cal, dates[5]


def test_gross_cash_on_existing_normalized_basis():
    raw, e, cal, day = fixture()
    entry = {"date": day, "route": "issuer_gross_cash_candidate_verify_exact_share_chain", "issuer_gross_cash_candidate": {"gross_old": 2.0}}
    frame, reports = reconstruct(raw, e.to_dict("records"), cal, [entry])
    assert reports[0]["status"] == "gross_cash_candidate"
    assert frame.loc[5, "candidate_cash"] == 1.0
    assert frame.loc[5, "dividends"] == 0
    assert not frame.production_eligible.any()


def test_cash_with_inconsistent_share_chain_is_held():
    raw, e, cal, day = fixture()
    raw.loc[5, "stock_splits"] = 3
    entry = {"date": day, "route": "issuer_gross_cash_candidate_verify_exact_share_chain", "issuer_gross_cash_candidate": {"gross_old": 2.0}}
    frame, reports = reconstruct(raw, e.to_dict("records"), cal, [entry])
    assert reports[0]["status"] == "retained_pending_validation"
    assert frame.loc[5, "candidate_cash"] == 0


def test_price_prefix_is_candidate_only():
    raw, e, cal, day = fixture()
    raw.loc[:4, ["open", "high", "low", "close"]] = 5.0
    entry = {"date": day, "route": "quote_basis_break_estimate_pre_segment_factor", "price_multiplier_hypothesis": 2.0}
    frame, reports = reconstruct(raw, e.to_dict("records"), cal, [entry])
    assert reports[0]["multiplier"] == pytest.approx(2)
    assert frame.loc[4, "candidate_close"] == 10
    assert frame.loc[4, "close"] == 5
    assert frame.candidate_review.all()


def test_stale_window_requires_two_independent_anchors():
    raw, e, cal, day = fixture()
    raw.loc[4:6, "volume"] = 0
    entry = {"date": day, "route": "stale_yahoo_replace_only_verified_trading_sessions"}
    frame, reports = reconstruct(raw, e.to_dict("records"), cal, [entry])
    assert reports[0]["status"] == "stale_window_candidate"
    assert reports[0]["sessions"] == 3
    assert frame.loc[4:6, "candidate_volume"].isna().all()
    assert frame.loc[4:6, "candidate_close"].eq(10).all()
    e.loc[10, "close"] = 20
    frame, reports = reconstruct(raw, e.to_dict("records"), cal, [entry])
    assert reports[0]["status"] == "retained_pending_validation"
    assert frame.loc[4:6, "candidate_volume"].eq(0).all()
