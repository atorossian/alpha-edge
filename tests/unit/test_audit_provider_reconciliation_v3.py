"""Corporate-action reconciliation is evidence-only and never drops unmatched events."""

import json

import pandas as pd

from alpha_edge.market.audit_provider_reconciliation_v3 import compare_asset, reconcile_events


def test_exact_near_duplicate_and_outside_split_events_are_accounted_for():
    y = [{"date": d, "value": v} for d, v in [
        ("2020-01-03", 2), ("2020-02-03", 3), ("2020-03-03", 2),
        ("2020-03-03", 2), ("2019-01-01", 4), ("2020-04-03", 2),
    ]]
    e = [{"date": d, "split": v} for d, v in [
        ("2020-01-03", "2/1"), ("2020-02-04", "3/1"), ("2020-03-03", "2/1"),
        ("2020-05-03", "5/1"),
    ]]
    rows = reconcile_events(y, e, kind="split", start="2020-01-01", end="2020-12-31",
                            currency="USD", context={})
    assert sum(r["yahoo_event_rows"] for r in rows) == 6
    assert sum(r["eodhd_event_rows"] for r in rows) == 4
    assert {(r["date_status"], r["value_status"]) for r in rows} == {
        ("exact_date", "ratio_match"), ("near_date_candidate", "ratio_match"),
        ("exact_date", "duplicate_date"), ("outside_overlap", "unpaired"),
        ("yahoo_only", "unpaired"), ("eodhd_only", "unpaired"),
    }


def test_ambiguous_near_dates_are_not_merged():
    y = [{"date": "2020-02-03", "value": 1}, {"date": "2020-02-05", "value": 1}]
    e = [{"date": "2020-02-04", "value": 1, "currency": "USD"}]
    rows = reconcile_events(y, e, kind="dividend", start="2020-01-01", end="2020-12-31",
                            currency="USD", context={})
    assert len(rows) == 3
    assert all(r["date_status"] == "ambiguous_near_date" for r in rows)


def test_dividend_amount_basis_and_currency_are_distinct():
    y = [{"date": "2020-02-03", "value": 0.5, "currency": "USD"}]
    e = [{"date": "2020-02-03", "value": 0.5, "unadjustedValue": 2, "currency": "USD"}]
    options = dict(kind="dividend", start="2020-01-01", end="2020-12-31", currency="USD", context={})
    assert reconcile_events(y, e, **options)[0]["value_status"] == "adjusted_amount_match"
    e[0]["currency"] = "EUR"
    assert reconcile_events(y, e, **options)[0]["value_status"] == "currency_mismatch"


def test_price_comparison_does_not_adjust_yahoo_twice():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    y = pd.DataFrame({"session_date": dates, "close": [50, 51, 52],
                      "open": [50, 51, 52], "high": [50, 51, 52], "low": [50, 51, 52],
                      "adj_close": [50, 51, 52], "volume": [10, 20, 30],
                      "stock_splits": [0, 2, 0], "dividends": [0, 0, 0], "currency_original": "USD"})
    responses = {
        "eod": [{"date": d, "open": p, "high": p, "low": p, "close": p,
                 "adjusted_close": a, "volume": v}
                for d, p, a, v in zip(dates, [100, 51, 52], [50, 51, 52], [10, 20, 30])],
        "splits": [{"date": dates[1], "split": "2/1"}], "div": [],
    }
    summary, events = compare_asset(y, responses, currency="USD")
    fields = json.loads(summary["field_metrics"])
    assert fields["close"]["as_reported"]["outside_tolerance"] == 1
    assert fields["close"]["eodhd_split_normalized"]["within_tolerance"] == 3
    assert events[0]["cross_provider_step_ratio"] == 0.5
    assert list(y.close) == [50, 51, 52]
    responses["splits"] = []
    _, events = compare_asset(y, responses, currency="USD")
    assert events[0]["date_status"] == "yahoo_only"
