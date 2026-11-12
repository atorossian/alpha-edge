"""Regression tests for conservative universe collision recommendations."""
from __future__ import annotations

import pandas as pd

from alpha_edge.jobs.universe_identity_decisions import identify_collisions, report_collisions


def _row(*, rid, aid, ticker, name, region, yahoo, include=1, role="stock"):
    return {
        "row_id": rid, "asset_id": aid, "ticker": ticker,
        "yahoo_ticker": yahoo, "name": name, "asset_class": "equity",
        "role": role, "region": region, "include": include,
    }


def test_ambev_duplicate_proposes_later_exclusion_without_mutation():
    rows = pd.DataFrame([
        _row(rid="FIRST", aid="EQH6ecf933f2adfa3f7", ticker="ABEV3",
             name="Ambev SA", region="B3", yahoo="ABEV3.SA"),
        _row(rid="SECOND", aid="EQH78bd0291b1513b79", ticker="ABEV3",
             name="Ambev S.A.", region="B3", yahoo="ABEV3.SA"),
    ])
    original = rows.copy(deep=True)
    result = identify_collisions(rows)
    assert len(result) == 1
    match = result.iloc[0]
    assert match["proposed_action"] == "PROPOSE_EXCLUSION"
    assert match["retain_asset_id"] == "EQH6ecf933f2adfa3f7"
    assert match["exclude_row_id"] == "SECOND"
    assert not bool(match["approved"])
    assert not bool(match["applied"])
    pd.testing.assert_frame_equal(rows, original)


def test_adp_collision_recommends_provider_review_without_network():
    rows = pd.DataFrame([
        _row(rid="US", aid="EQH5cd6ee5dd8d979a3", ticker="ADP",
             name="Automatic Data Processing Inc.", region="NASDAQ", yahoo="ADP"),
        _row(rid="FR", aid="EQH8983943c3dcf91dc", ticker="ADP",
             name="Aeroports de Paris SA", region="Cboe Europe", yahoo="ADP"),
    ])
    result = identify_collisions(rows)
    assert result.iloc[0]["proposed_action"] == "REVIEW_PROVIDER_MAPPING"
    assert result.iloc[0]["exclude_row_id"] == ""
    assert result.iloc[0]["suggested_yahoo_ticker"] == ""


def test_adp_provider_verified_distinct_mapping_is_a_proposal_only():
    rows = pd.DataFrame([
        _row(rid="US", aid="A", ticker="ADP", name="Automatic Data Processing Inc.",
             region="NASDAQ", yahoo="ADP"),
        _row(rid="FR", aid="B", ticker="ADP", name="Aeroports de Paris SA",
             region="Cboe Europe", yahoo="ADP"),
    ])
    seen = []

    def resolver(raw, **kwargs):
        seen.append((raw, kwargs["expected_name"], kwargs["expected_region"]))
        return "ADP.PA", {"verified": True}

    result = identify_collisions(rows, provider_resolver=resolver)
    assert seen == [("ADP", "Aeroports de Paris SA", "Cboe Europe")]
    assert result.iloc[0]["proposed_action"] == "PROPOSE_OVERRIDE"
    assert result.iloc[0]["suggested_yahoo_ticker"] == "ADP.PA"
    assert result.iloc[0]["override_row_id"] == "FR"
    assert not bool(result.iloc[0]["applied"])


def test_same_company_different_market_does_not_exclude():
    rows = pd.DataFrame([
        _row(rid="US", aid="A", ticker="WMT", name="Walmart Inc.",
             region="NYSE", yahoo="WMT"),
        _row(rid="EU", aid="B", ticker="WMT", name="Walmart Inc.",
             region="Cboe Europe", yahoo="WMT"),
    ])
    result = identify_collisions(rows)
    assert result.iloc[0]["proposed_action"] == "REVIEW_IDENTITY"
    assert result.iloc[0]["exclude_row_id"] == ""


def test_exclusion_ledger_is_applied_by_row_id():
    rows = pd.DataFrame([
        _row(rid="ONE", aid="A", ticker="ABEV3", name="Ambev SA",
             region="B3", yahoo="ABEV3.SA"),
        _row(rid="TWO", aid="B", ticker="ABEV3", name="Ambev SA",
             region="B3", yahoo="ABEV3.SA"),
    ])
    exclusions = pd.DataFrame([{
        "target_row_id": "TWO", "ticker": "ABEV3",
        "asset_class": "equity", "reason": "reviewed duplicate",
    }])
    assert identify_collisions(rows, exclusions=exclusions).empty


def test_triple_collision_never_proposes_automatic_exclusion():
    rows = pd.DataFrame([
        _row(rid=str(i), aid=str(i), ticker="ABC", name="ABC Holdings",
             region="NYSE", yahoo="ABC")
        for i in range(3)
    ])
    result = identify_collisions(rows)
    assert len(result) == 2
    assert "PROPOSE_EXCLUSION" not in result["proposed_action"].tolist()


def test_report_writes_only_audit_csv(tmp_path):
    rows = pd.DataFrame([
        _row(rid="ONE", aid="A", ticker="ABEV3", name="Ambev SA",
             region="B3", yahoo="ABEV3.SA"),
        _row(rid="TWO", aid="B", ticker="ABEV3", name="Ambev SA",
             region="B3", yahoo="ABEV3.SA"),
    ])
    report = report_collisions(rows, output_csv=tmp_path / "decision.csv")
    assert report["collision_groups"] == 1
    assert report["proposed_actions"] == {"PROPOSE_EXCLUSION": 1}
    assert report["universe_mutations"] == 0
    assert report["ledger_mutations"] == 0
    assert set(tmp_path.iterdir()) == {tmp_path / "decision.csv"}
