import pandas as pd

from alpha_edge.market.audit_matching_split_basis_v3 import prepare
from alpha_edge.market.reconcile_corporate_actions_v3 import investigate, shared_windows


def fixture(cash=1.0):
    days = pd.bdate_range("2020-01-01", periods=10).strftime("%Y-%m-%d").tolist()
    day = days[5]
    y = pd.DataFrame({"session_date": days, "open": 50., "high": 51., "low": 49., "close": 50.,
                      "adj_close": [45.]*5+[45./(1-cash/50)]*5, "stock_splits": [0.]*5+[2.]+[0.]*4,
                      "dividends": [0.]*5+[cash]+[0.]*4, "currency_original": "USD"})
    responses = {"eod": [{"date": d, "open": 50*f, "high": 51*f, "low": 49*f, "close": 50*f,
                          "adjusted_close": a} for d, f, a in zip(days, [2]*5+[1]*5, y.adj_close)],
                 "splits": [{"date": day, "split": "2/1"}],
                 "div": [{"date": day, "unadjustedValue": 2*cash, "value": cash, "currency": "USD"}] if cash else []}
    row = {"date": day, "ratio": 2., "group": "integer_forward", "same_day_dividend": bool(cash),
           "root_category": "cash_and_share" if cash else "continuity", "continuity": "ambiguous"}
    event = {"date_status": "exact_date", "value_status": "ratio_match", "yahoo_date": day,
             "yahoo_value": 2., "eodhd_date": day, "eodhd_value": 2.}
    return y, responses, row, event


def test_pre_event_cash_basis_independently_passes():
    y, responses, row, event = fixture()
    yy, ee = prepare(y, responses)
    result = investigate(yy, ee, responses, row, event, [])
    assert result["status"] == "reconciled_price_basis_candidate"
    assert result["chosen_hypothesis"]["method"] == "cash_pre"
    assert any(a["status"] == "cash_amount_or_share_basis_disagreement" for a in result["attempts"])
    assert result["production_eligible"] is False


def test_continuity_recheck_and_date_conflict_do_not_change_source():
    y, responses, row, event = fixture(cash=0)
    yy, ee = prepare(y, responses)
    assert investigate(yy, ee, responses, row, event, [])["status"] == "reconciled_price_basis_candidate"
    previous = responses["eod"][4]["date"]
    responses["splits"][0]["date"] = previous
    event.update({"date_status": "near_date_candidate", "eodhd_date": previous})
    result = investigate(yy, ee, responses, row, event, [])
    assert result["status"] == "price_basis_candidate_date_review"
    assert responses["splits"][0]["date"] == previous


def test_prior_conflict_and_ratio_disagreement_remain_held():
    y, responses, row, event = fixture(cash=0)
    yy, ee = prepare(y, responses)
    result = investigate(yy, ee, responses, row, event, [{"comparison": "documented_price_coefficient_other"}])
    assert result["status"] == "prior_factor_requires_review"
    event["value_status"] = "ratio_mismatch"
    event["eodhd_value"] = 3.
    result = investigate(yy, ee, responses, row, event, [])
    assert result["status"] == "evidence_exhausted"
    assert "yahoo" in result["attempts"][0]["matching_hypotheses"]


def test_missing_predecessor_is_not_bridged():
    y, responses, row, event = fixture()
    yy, ee = prepare(y, responses)
    assert shared_windows(yy, ee.drop(ee.index[4]), responses, row["date"]) is None
