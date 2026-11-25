import pandas as pd

from alpha_edge.market.audit_combined_cash_share_v3 import assess_prior, screen_cash_share
from alpha_edge.market.audit_matching_split_basis_v3 import prepare


def fixture():
    days = pd.bdate_range("2020-01-01", periods=10).strftime("%Y-%m-%d").tolist()
    day = days[5]
    y = pd.DataFrame({"session_date": days, "open": 50., "high": 51., "low": 49., "close": 50.,
                      "adj_close": [45.]*5+[45./.98]*5, "stock_splits": [0.]*5+[2.]+[0.]*4,
                      "dividends": [0.]*5+[1.]+[0.]*4, "currency_original": "USD"})
    responses = {"eod": [{"date": d, "open": 50*f, "high": 51*f, "low": 49*f, "close": 50*f,
                          "adjusted_close": a} for d, f, a in zip(days, [2]*5+[1]*5, y.adj_close)],
                 "splits": [{"date": day, "split": "2/1"}],
                 "div": [{"date": day, "unadjustedValue": 1., "currency": "USD"}]}
    return y, responses, {"date": day, "ratio": 2.}, {"date_status": "exact_date", "value_status": "ratio_match"}


def test_cash_and_share_steps_must_both_match():
    y, responses, row, event = fixture()
    yy, ee = prepare(y, responses)
    r = screen_cash_share(yy, ee, responses, row, event)
    assert r["status"] == "combined_cash_share_price_basis_candidate"
    assert r["production_eligible"] is False
    responses["div"][0]["unadjustedValue"] = 2
    assert screen_cash_share(yy, ee, responses, row, event)["status"] == "cash_amount_or_share_basis_disagreement"
    responses["div"][0]["unadjustedValue"] = 1
    y["adj_close"] = 45.
    yy, ee = prepare(y, responses)
    assert screen_cash_share(yy, ee, responses, row, event)["status"] == "cash_and_share_factor_disagreement"


def test_nearby_and_missing_cash_not_treated_as_same_day():
    y, responses, row, event = fixture()
    responses["div"][0]["date"] = responses["eod"][4]["date"]
    yy, ee = prepare(y, responses)
    assert screen_cash_share(yy, ee, responses, row, event)["status"] == "nearby_cash_requires_separate_reconciliation"
    responses["div"] = []
    assert screen_cash_share(yy, ee, responses, row, event)["status"] == "cash_not_uniquely_reported_by_both"


def test_documented_price_coefficient_is_not_a_share_ratio():
    row = {"ratio": 2., "bronze_sha256": "sha"}
    factors = [{"source_sha256": "sha", "kind": "price_coefficient", "factor": "0.5", "status": "verified"},
               {"source_sha256": "sha", "kind": "share_ratio", "factor": "2", "status": "provisional"},
               {"source_sha256": "stale", "kind": "share_ratio", "factor": "2", "status": "verified"}]
    assert [r["comparison"] for r in assess_prior(row, factors)] == [
        "documented_price_coefficient_reciprocal_match", "provisional_factor_review"]
