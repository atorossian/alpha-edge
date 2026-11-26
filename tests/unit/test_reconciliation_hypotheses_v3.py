import pandas as pd

from alpha_edge.market.audit_matching_split_basis_v3 import prepare
from alpha_edge.market.reconciliation_hypotheses_v3 import analyze_window, calendar_windows, CANDIDATE


def fixture(ratio=2., cash=0., move=1.):
    days = pd.bdate_range("2020-01-01", periods=10).strftime("%Y-%m-%d").tolist()
    day = days[5]
    prices = [50.]*5+[50.*move]*5
    adj = [45.]*5+[45.*move/(1-cash/50)]*5
    y = pd.DataFrame({"session_date": days, "open": prices, "high": prices, "low": prices, "close": prices,
                      "adj_close": adj, "stock_splits": [0.]*5+[ratio]+[0.]*4,
                      "dividends": [0.]*5+[cash]+[0.]*4, "currency_original": "USD"})
    responses = {"eod": [{"date": d, "open": p*f, "high": p*f, "low": p*f, "close": p*f,
                          "adjusted_close": a} for d,p,f,a in zip(days, prices, [ratio]*5+[1]*5, adj)],
                 "splits": [{"date": day, "split": f"{ratio}/1"}],
                 "div": [{"date": day,"unadjustedValue": 3*cash,"value": cash,"currency": "USD"}] if cash else []}
    cal = pd.DataFrame({"session_date": days, "is_expected_session": True, "ohlc_eligible": True})
    row = {"date": day, "ratio": ratio}
    event = {"date_status": "exact_date", "value_status": "ratio_match"}
    return y, responses, cal, row, event


def test_adjusted_cash_and_large_return_are_independently_checked():
    y, responses, cal, row, event = fixture(cash=1., move=1.2)
    yy, ee = prepare(y, responses)
    result = analyze_window(yy, ee, responses, row, event, 5)
    assert result["status"] == CANDIDATE
    assert result["cash_raw_basis_review_required"] is True
    assert result["large_return_review_required"] is True
    responses["eod"][5]["adjusted_close"] *= .8
    yy, ee = prepare(y, responses)
    assert analyze_window(yy, ee, responses, row, event, 5)["status"] == "advanced_independent_factor_step_disagrees"


def test_small_signal_and_calendar_missing_session():
    y, responses, cal, row, event = fixture(ratio=1.005)
    yy, ee = prepare(y, responses)
    assert analyze_window(yy, ee, responses, row, event, 5)["status"] == CANDIDATE
    assert len(calendar_windows(yy, ee, cal, responses, row["date"])) == 1
    assert not calendar_windows(yy, ee, cal, responses, row["date"], [yy.index[2]])
    assert not calendar_windows(yy, ee.drop(ee.index[2]), cal, responses, row["date"])


def test_field_disagreement_not_waived_by_rounding():
    y, responses, cal, row, event = fixture()
    responses["eod"][4]["high"] *= 1.1
    yy, ee = prepare(y, responses)
    result = analyze_window(yy, ee, responses, row, event, 5)
    assert result["status"] == "advanced_field_scale_disagrees"
    assert result["field_max_relative_error"]["high"] > .05


def test_close_only_does_not_certify_disputed_ohlc_or_volume():
    from alpha_edge.market.reconciliation_hypotheses_v3 import CLOSE_CANDIDATE
    y, responses, cal, row, event = fixture()
    responses["eod"][4]["open"] *= 1.1
    yy, ee = prepare(y, responses)
    assert analyze_window(yy, ee, responses, row, event, 5)["status"] == "advanced_field_scale_disagrees"
    result = analyze_window(yy, ee, responses, row, event, 5, close_only=True)
    assert result["status"] == CLOSE_CANDIDATE
    assert result["candidate_eligible_fields"] == ["close"]
    assert not result["field_agreement_hypothesis"]["open"]
    assert not result["field_agreement_hypothesis"]["volume"]
    responses["eod"][5]["adjusted_close"] *= .8
    yy, ee = prepare(y, responses)
    assert analyze_window(yy, ee, responses, row, event, 5, close_only=True)["status"] != CLOSE_CANDIDATE


def test_cash_sequence_diagnostic_does_not_create_missing_cash():
    from alpha_edge.market.reconciliation_hypotheses_v3 import cash_sequence_diagnostic
    y, responses, cal, row, event = fixture(cash=1.)
    responses["div"] = []
    yy, ee = prepare(y, responses)
    result = cash_sequence_diagnostic(yy, ee, responses, row)
    target = next(x for x in result["sessions"] if x["date"] == row["date"])
    assert target["cash_implied_agreement"]
    assert target["issuer_evidence_required"]
    assert target["eodhd_reported_cash"] == []
    assert result["status"] == "cash_sequence_diagnostic_only"
