import pytest

from alpha_edge.market.audit_remaining_split_cases_v3 import consolidate, match_evidence, classify_reason


def test_consolidation_overlays_targeted_cases_and_never_overwrites_approved():
    a = {"asset_id": "a", "date": "2020-01-01", "ratio": 1.05, "bronze_sha256": "a"}
    b = {**a, "asset_id": "b"}
    c = {**a, "asset_id": "c"}
    old = {**a, "status": "yahoo_continuity_not_confirmed"}
    new = {**a, "status": "small_fraction_price_basis_candidate"}
    unmatched = {**b, "reason": "yahoo_only_within_overlap"}
    latest = consolidate([a, b, c], [unmatched], [[old, {**c, "status": "ohlc_window_disagreement"}], [new],
                          [{**b, "status": "same_day_cash_event"}]])
    assert latest[("a", a["date"])]["status"] == "small_fraction_price_basis_candidate"
    assert classify_reason(latest[("b", b["date"])]["status"]) == "cash_and_share"
    assert len(latest) == 3
    with pytest.raises(ValueError, match="replace validated"):
        consolidate([a], [], [[new], [old]])
    with pytest.raises(ValueError, match="source mismatch"):
        consolidate([a], [], [[{**new, "bronze_sha256": "different"}]])


def test_registry_match_distinguishes_nearby_source_and_exact():
    row = {"asset_id": "a", "date": "2020-01-05", "bronze_sha256": "hash"}
    index = {"a": [
        {"date": "2020-01-05", "sha256": "other", "family": "currency"},
        {"date": "2020-01-06", "sha256": "hash", "family": "complex"},
        {"start": "2020-01-01", "end": "2020-01-10", "sha256": "hash", "family": "events"},
        {"date": "2020-01-15", "sha256": "hash", "family": "unrelated"},
    ]}
    matches = match_evidence(row, index)
    assert [(m["relation"], m["source_matches"]) for m in matches] == [
        ("exact_date", False), ("near_date", True), ("within_event_window", True)]
