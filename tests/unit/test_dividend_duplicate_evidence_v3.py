import pandas as pd
import pytest

from alpha_edge.market import dividend_duplicate_evidence_v3 as duplicate, historical_quarantine_v3 as quarantine
from alpha_edge.market.silver_financial_evidence_v3 import extract_source_actions


def bars(factor, *, amount=1.0):
    return pd.DataFrame(
        {
            "asset_id": ["A"] * 3,
            "provider_symbol": ["AAA"] * 3,
            "currency_original": ["USD"] * 3,
            "session_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "close": [100.0] * 3,
            "adj_close": [100 * factor, 100.0, 100.0],
            "dividends": [0.0, amount, 0.0],
            "stock_splits": [0.0] * 3,
        }
    )


@pytest.mark.parametrize(
    ("factor", "reason"),
    [
        (0.98, "possible_double_adjustment"),
        (0.99**2, "possible_double_adjustment"),
        (0.995, "possible_doubled_reported_amount"),
        (0.97, "dividend_factor_mismatch"),
        (0.99, None),
    ],
)
def test_factor_hypotheses(factor, reason):
    frame = bars(factor)
    pristine = frame.copy()
    actions, _ = extract_source_actions(frame)
    findings, stats = duplicate.assess(frame, actions)
    assert findings.reason.tolist() == ([reason] if reason else [])
    assert stats["corrections_applied"] == 0
    assert not stats["production_eligible"]
    pd.testing.assert_frame_equal(frame, pristine)


def test_nearby_equal_events_need_review_even_if_ratios_agree():
    frame = bars(0.99**2)
    frame.loc[1, "adj_close"] = 99.0
    frame.loc[2, "dividends"] = 1.0
    actions, _ = extract_source_actions(frame)
    findings, _ = duplicate.assess(frame, actions)
    assert findings.reason.tolist() == ["nearby_equal_dividends"]
    assert findings.iloc[0].related_date == "2020-01-02"
    sessions, _, stats = quarantine.assess(
        frame, actions, asset_id="A", provider_symbol="AAA", bronze_sha256="a" * 64, duplicate_findings=findings
    )
    assert stats["quarantine_through"] == "2020-01-03"
    assert not sessions.candidate_return.any()


def test_same_day_split_and_invalid_adjusted_close_are_not_evaluable():
    frame = bars(0.98)
    frame.loc[1, "stock_splits"] = 7.7
    findings, _ = duplicate.assess(frame, extract_source_actions(frame)[0])
    assert findings.reason.tolist() == ["dividend_factor_not_evaluable"]
    frame.loc[1, "stock_splits"] = 0.0
    frame.loc[0, "adj_close"] = -1.0
    findings, _ = duplicate.assess(frame, extract_source_actions(frame)[0])
    assert findings.reason.tolist() == ["dividend_factor_not_evaluable"]
