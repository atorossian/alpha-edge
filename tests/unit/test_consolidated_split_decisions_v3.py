"""Residual approvals cannot bypass independent financial validation."""

import json

import pandas as pd
import pytest

from alpha_edge.market import consolidated_split_decisions_v3 as residual
from alpha_edge.market.build_silver_adjusted_v3 import reconstruct
from test_build_silver_adjusted_v3 import fixture


def decision(day="2025-04-02", status="close_only_pending_integration"):
    source = {
        "asset_id": "a",
        "symbol": "A",
        "date": day,
        "ratio": 2.0,
        "bronze_sha256": "a" * 64,
        "eodhd_sha256": "b" * 64,
    }
    return {
        "asset_id": "a",
        "symbol": "A",
        "date": day,
        "financial_status": status,
        "decision_approved": True,
        "production_eligible": False,
        "quarantine_released": False,
        "decision": source,
        "source_entry": source,
        "source_doc": None,
        "ratio": 2.0,
        "bronze_sha256": "a" * 64,
        "eodhd_sha256": "b" * 64,
    }


def test_registry_identity_and_approval_are_strict(monkeypatch):
    e = decision()
    doc = {
        "schema": "alpha_edge_consolidated_split_decisions_v1",
        "status": "approved_dispositions_pending_financial_validation",
        "consolidated_index": {"sha256": "i"},
        "source_refs": {"bundle": {"sha256": "b"}},
        "production_eligible": False,
        "quarantine_released": False,
        "baseline_residual_count": 1,
        "approved_count": 1,
        "financial_status_counts": {e["financial_status"]: 1},
        "entries": [e],
    }
    monkeypatch.setattr(residual, "verified_bytes", lambda *a: json.dumps(doc).encode())

    def load():
        return residual.load(None, "bucket", "v", consolidated_version="i", bundle_version="b", expected_count=1)

    assert load()["a"][e["date"]]["bronze_sha256"] == "a" * 64
    e["decision_approved"] = False
    with pytest.raises(ValueError, match="Invalid or duplicate"):
        load()


def test_execution_plan_rejects_wrong_population(monkeypatch):
    args = {
        "consolidated_version": "i",
        "bundle_version": "b",
        "split_basis_approvals": ["a"],
        "deterministic_basis_approvals": ["d"],
        "expected_split_approvals": 3014,
        "residual_version": "r",
        "expected_residual_count": 596,
        "eodhd_run_id": "eod",
    }
    doc = {
        "schema": "alpha_edge_consolidated_financial_execution_plan_v1",
        "total_cases": 3610,
        "production_eligible": False,
        "quarantine_released": False,
        "builder_arguments": args,
    }
    monkeypatch.setattr(residual, "verified_bytes", lambda *a: json.dumps(doc).encode())
    assert residual.load_execution_plan(None, "bucket", "v") == args
    doc["total_cases"] = 3609
    with pytest.raises(ValueError, match="population mismatch"):
        residual.load_execution_plan(None, "bucket", "v")


def test_legacy_manual_source_identity_and_checksum_pin(monkeypatch):
    e = decision()
    e["decision"]["source"] = {"key": "legacy", "sha256": "l"}
    doc = {
        "schema": "alpha_edge_consolidated_split_decisions_v1",
        "status": "approved_dispositions_pending_financial_validation",
        "consolidated_index": {"sha256": "i"},
        "source_refs": {"bundle": {"sha256": "b"}},
        "production_eligible": False,
        "quarantine_released": False,
        "baseline_residual_count": 1,
        "approved_count": 1,
        "financial_status_counts": {e["financial_status"]: 1},
        "entries": [e],
    }
    source = {
        "asset_id": "a",
        "production_eligible": False,
        "quarantine_released": False,
        "yahoo": {"sha256": "a" * 64},
        "eodhd": {"sha256": "b" * 64},
        "entries": [{"yahoo_date": e["date"], "ratio": 2.0}],
    }
    monkeypatch.setattr(
        residual, "verified_bytes", lambda s, b, ref: json.dumps(source if ref["key"] == "legacy" else doc).encode()
    )
    result = residual.load(None, "bucket", "v", consolidated_version="i", bundle_version="b", expected_count=1)
    assert result["a"][e["date"]]["bronze_sha256"] == "a" * 64


def test_full_price_normalized_basis_requires_all_field_agreement():
    days = pd.bdate_range("2025-04-01", periods=16).strftime("%Y-%m-%d")
    day = days[8]
    source = pd.DataFrame(
        {
            "session_date": days,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "adj_close": 100.0,
            "stock_splits": 0.0,
            "dividends": 0.0,
            "currency_original": "USD",
        }
    )
    source.loc[8, "stock_splits"] = 2.0
    raw = source.set_index("session_date", drop=False)
    prices = source.rename(columns={"session_date": "date", "adj_close": "adjusted_close"})
    responses = {"eod": prices.to_dict("records"), "div": [], "splits": []}
    cal = pd.DataFrame({"session_date": days, "is_expected_session": True, "ohlc_eligible": True})
    findings = pd.DataFrame({"issue": [], "session_date": []})
    e = decision(day, "approved_price_basis_pending_integration")
    assert (
        residual.close_evidence(raw, cal, findings, {day: e}, responses)[day]["status"]
        == "already_normalized_price_basis_reverified"
    )
    responses["eod"][7]["open"] = 95.0
    assert (
        residual.close_evidence(raw, cal, findings, {day: e}, responses)[day]["status"] == "basis_reverification_failed"
    )


def test_residual_registry_integrates_without_releasing_quarantine():
    bars, cal, findings, bundle = fixture()
    day = "2025-04-02"
    bars.loc[bars.session_date.eq(day), "stock_splits"] = 2.0
    original = bars.copy(deep=True)
    entry = decision(day)
    bundle.update(residual_decisions={"a": {day: entry}}, residual_version="v")
    result, _ = reconstruct(bars, cal, findings, bundle, asset_id="a", symbol="A", source_key="key", sha="a" * 64)
    pd.testing.assert_frame_equal(bars, original)
    report = residual.validate_candidate(result, bars, {day: entry})
    assert report["residual_decision_events"] == 1
    assert report["residual_financial_blocker_events"] == 1
    assert report["residual_close_reverified_events"] == 0
    assert not result.production_eligible.any()
    assert result.candidate_close_only_price_return.isna().all()
    assert result.loc[result.session_date.eq(day), "historical_quarantine"].all()
    with pytest.raises(ValueError, match="source mismatch"):
        reconstruct(bars, cal, findings, bundle, asset_id="a", symbol="A", source_key="key", sha="b" * 64)


def test_close_basis_never_promotes_ohlc_or_cash_return():
    day = "2025-04-02"
    raw = pd.DataFrame({"close": [100.0, 105.0], "stock_splits": [0.0, 2.0]}, index=["2025-04-01", day])
    f = raw.copy()
    f["candidate_expected_session"] = True
    f["correction_sources"] = ""
    f["historical_quarantine"] = True
    f["candidate_total_return"] = 0.123
    e = decision(day)
    check = {"date": day, "yahoo_cash": 0.0, "cash_evidence": "none", "yahoo_adjusted_return": 0.05}
    evidence = {day: {"status": "close_basis_only_candidate", "daily_checks": [check]}}
    residual.annotate(f, raw, {day: e}, evidence, version="v")
    assert f.at[day, "candidate_close_only_price_return"] == pytest.approx(0.05)
    assert f.at[day, "candidate_total_return"] == 0.123
    assert f.historical_quarantine.all()
    check["yahoo_cash"] = 1.0
    residual.annotate(f, raw, {day: e}, evidence, version="v")
    assert f.candidate_close_only_price_return.isna().all()


def test_treasury_delivery_duplicate_only_changes_candidate_cash():
    bars, cal, findings, bundle = fixture()
    day, delivery = "2025-04-01", "2025-04-02"
    bars.loc[bars.session_date.eq(day), ["stock_splits", "dividends"]] = [2.0, 0.0]
    bars.loc[bars.session_date.eq(delivery), "dividends"] = 2.0
    e = decision(day, "reconstruction_pending")
    e["decision"] = {
        "previous_evidence": e["source_entry"],
        "proposed_decision": {
            "decision": "single_treasury_share_distribution",
            "ratio": 2.0,
            "duplicate_reported_cash_date": delivery,
            "duplicate_reported_cash_adjusted": 2.0,
        },
    }
    bundle.update(residual_decisions={"a": {day: e}}, residual_version="v")
    result, _ = reconstruct(bars, cal, findings, bundle, asset_id="a", symbol="A", source_key="key", sha="a" * 64)
    row = result.set_index("session_date").loc[delivery]
    assert row.dividends == 2.0 and row.candidate_cash == 0.0
    assert result.candidate_residual_reconstruction_applied.sum() == 1
    assert not result.production_eligible.any()
