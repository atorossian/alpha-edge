"""Local split basis screening and conservative quarantine upper bounds."""

import pandas as pd

from alpha_edge.market.audit_matching_split_basis_v3 import CANDIDATE, prepare, recovery_bound, screen_event


def fixture():
    days = pd.bdate_range("2020-01-01", periods=10).strftime("%Y-%m-%d").tolist()
    day = days[5]
    y = pd.DataFrame({"session_date": days, "open": 50., "high": 51., "low": 49., "close": 50.,
                      "adj_close": 45., "stock_splits": [0.]*5+[2.]+[0.]*4})
    responses = {"eod": [{"date": d, "open": 50*f, "high": 51*f, "low": 49*f, "close": 50*f,
                          "adjusted_close": 45.} for d, f in zip(days, [2]*5+[1]*5)],
                 "splits": [{"date": day, "split": "2/1"}], "div": []}
    event = {"yahoo_date": day, "yahoo_value": 2.}
    screen = {"same_day_dividend": False, "continuity": "provider_normalized_signal"}
    return y, responses, event, screen


def test_integer_split_basis_matches_across_ten_sessions():
    y, responses, event, screen = fixture()
    yy, ee = prepare(y, responses)
    result = screen_event(yy, ee, responses, event, screen)
    assert result["status"] == CANDIDATE
    assert result["implied_ratio"] == 2
    assert result["production_eligible"] is False


def test_same_day_cash_and_fractional_events_require_review():
    y, responses, event, screen = fixture()
    yy, ee = prepare(y, responses)
    responses["div"] = [{"date": event["yahoo_date"], "value": 1}]
    assert screen_event(yy, ee, responses, event, screen)["status"] == "same_day_cash_event"
    event["yahoo_value"] = 1.328
    assert screen_event(yy, ee, responses, event, screen)["status"] == "non_integer_economic_review"


def test_split_agreement_does_not_override_wrong_price_basis():
    y, responses, event, screen = fixture()
    responses["eod"][0]["open"] = 80
    yy, ee = prepare(y, responses)
    assert screen_event(yy, ee, responses, event, screen)["status"] == "ohlc_window_disagreement"
    y, responses, event, screen = fixture()
    responses["eod"][5]["adjusted_close"] = 30
    yy, ee = prepare(y, responses)
    assert screen_event(yy, ee, responses, event, screen)["status"] == "adjustment_factor_disagreement"


def test_recovery_bound_respects_non_split_blocker():
    y, _, event, _ = fixture()
    findings = pd.DataFrame([
        {"scope": "prefix", "reason": "split_price_basis_unverified", "effective_date": event["yahoo_date"]},
        {"scope": "prefix", "reason": "other", "effective_date": y.session_date.iloc[2]},
    ])
    assert recovery_bound(y, findings, {event["yahoo_date"]}) == 3
    assert recovery_bound(y, findings, set()) == 0


def test_fractional_price_basis_is_not_integer_approval():
    y, responses, event, screen = fixture()
    y.loc[5, "stock_splits"] = 1.5
    event["yahoo_value"] = 1.5
    responses["splits"][0]["split"] = "3/2"
    for row in responses["eod"][:5]:
        for field in ("open", "high", "low", "close"):
            row[field] *= .75
    yy, ee = prepare(y, responses)
    assert screen_event(yy, ee, responses, event, screen)["status"] == "non_integer_economic_review"
    result = screen_event(yy, ee, responses, event, screen, fractional=True)
    assert result["status"] == "fractional_price_basis_candidate"
    assert result["production_eligible"] is False
    responses["div"] = [{"date": event["yahoo_date"], "value": 1}]
    assert screen_event(yy, ee, responses, event, screen, fractional=True)["status"] == "same_day_cash_event"


def test_small_fraction_uses_cross_provider_evidence_and_preserves_cash_blocker():
    for ratio, label in ((1.05, "21/20"), (1.10, "11/10")):
        y, responses, event, screen = fixture()
        y.loc[5, "stock_splits"] = ratio
        event["yahoo_value"] = ratio
        responses["splits"][0]["split"] = label
        screen["continuity"] = "ambiguous"
        for row in responses["eod"][:5]:
            for field in ("open", "high", "low", "close"):
                row[field] *= ratio / 2
        yy, ee = prepare(y, responses)
        assert screen_event(yy, ee, responses, event, screen, fractional=True)["status"] == "yahoo_continuity_not_confirmed"
        assert screen_event(yy, ee, responses, event, screen, small_fraction=True)["status"] == "small_fraction_price_basis_candidate"
        responses["div"] = [{"date": event["yahoo_date"], "value": 1}]
        assert screen_event(yy, ee, responses, event, screen, small_fraction=True)["status"] == "same_day_cash_event"
        responses["div"] = []
        responses["eod"][5]["adjusted_close"] = 30
        yy, ee = prepare(y, responses)
        assert screen_event(yy, ee, responses, event, screen, small_fraction=True)["status"] == "adjustment_factor_disagreement"


def test_irregular_ratio_requires_scope_and_independent_price_evidence():
    y, responses, event, screen = fixture()
    ratio = 1.328
    y.loc[5, "stock_splits"] = ratio
    event["yahoo_value"] = ratio
    responses["splits"][0]["split"] = "166/125"
    screen["continuity"] = "ambiguous"
    for row in responses["eod"][:5]:
        for field in ("open", "high", "low", "close"):
            row[field] *= ratio / 2
    yy, ee = prepare(y, responses)
    assert screen_event(yy, ee, responses, event, screen, fractional=True)["status"] == "non_integer_economic_review"
    result = screen_event(yy, ee, responses, event, screen, irregular=True)
    assert result["status"] == "irregular_price_basis_candidate"
    assert result["production_eligible"] is False
    responses["div"] = [{"date": event["yahoo_date"], "value": 1}]
    assert screen_event(yy, ee, responses, event, screen, irregular=True)["status"] == "same_day_cash_event"
    responses["div"] = []
    responses["eod"][0]["open"] *= 1.1
    yy, ee = prepare(y, responses)
    assert screen_event(yy, ee, responses, event, screen, irregular=True)["status"] == "ohlc_window_disagreement"


def test_yahoo_only_requires_implicit_price_and_factor_evidence():
    y, responses, event, screen = fixture()
    responses["splits"] = []
    yy, ee = prepare(y, responses)
    result = screen_event(yy, ee, responses, event, screen, yahoo_only=True)
    assert result["status"] == "yahoo_only_implicit_price_basis_candidate"
    assert result["production_eligible"] is False
    # Equal price bases cannot independently establish the reported split.
    for row in responses["eod"][:5]:
        for field in ("open", "high", "low", "close"):
            row[field] /= 2
    yy, ee = prepare(y, responses)
    assert screen_event(yy, ee, responses, event, screen, yahoo_only=True)["status"] == "no_independent_split_basis_change"


def test_yahoo_only_rejects_conflicting_actions_and_adjustment_factor():
    y, responses, event, screen = fixture()
    yy, ee = prepare(y, responses)
    assert screen_event(yy, ee, responses, event, screen, yahoo_only=True)["status"] == "multiple_or_conflicting_share_events_in_window"
    responses["splits"] = []
    responses["div"] = [{"date": event["yahoo_date"], "value": 1}]
    assert screen_event(yy, ee, responses, event, screen, yahoo_only=True)["status"] == "same_day_cash_event"
    responses["div"] = [{"date": responses["eod"][4]["date"], "value": 1}]
    assert screen_event(yy, ee, responses, event, screen, yahoo_only=True)["status"] == "nearby_eodhd_cash_event"
    responses["div"] = []
    responses["eod"][5]["adjusted_close"] = 30
    yy, ee = prepare(y, responses)
    assert screen_event(yy, ee, responses, event, screen, yahoo_only=True)["status"] == "adjustment_factor_disagreement"
