"""AEA87 identity mapping must never silently merge historical instruments."""

import pandas as pd

from alpha_edge.market.asset_level_planning_v2 import (
    build_plan_bundle,
    crosswalk,
    recovery_inventory,
    supplementary_preflight,
)


def registry():
    rows = [
        {
            "asset_id": "CURRENT-A",
            "current_universe_present": True,
            "in_original_plan": True,
            "identity_source": "current_and_archived_plan",
            "current_yahoo_ticker": "A",
            "historical_yahoo_ticker": "A",
            "current_eligible_config": True,
            "currency": "USD",
            "exchange": "PCX",
            "research_cohort_state": "marker_present_unverified",
            "original_cohort": 1,
        },
        {
            "asset_id": "CURRENT-B",
            "current_universe_present": True,
            "in_original_plan": False,
            "identity_source": "current_only",
            "current_yahoo_ticker": "B",
            "historical_yahoo_ticker": "",
            "current_eligible_config": True,
            "currency": "",
            "exchange": "",
            "research_cohort_state": "not_planned",
            "original_cohort": None,
        },
        {
            "asset_id": "HIST-A",
            "current_universe_present": False,
            "in_original_plan": False,
            "identity_source": "legacy_key_only",
            "current_yahoo_ticker": "",
            "historical_yahoo_ticker": "",
            "current_eligible_config": False,
            "currency": "",
            "exchange": "",
            "research_cohort_state": "not_planned",
            "original_cohort": None,
        },
        {
            "asset_id": "HIST-B",
            "current_universe_present": False,
            "in_original_plan": False,
            "identity_source": "legacy_key_only",
            "current_yahoo_ticker": "",
            "historical_yahoo_ticker": "",
            "current_eligible_config": False,
            "currency": "",
            "exchange": "",
            "research_cohort_state": "not_planned",
            "original_cohort": None,
        },
        {
            "asset_id": "MISSING",
            "current_universe_present": True,
            "in_original_plan": True,
            "identity_source": "current_and_archived_plan",
            "current_yahoo_ticker": "M",
            "historical_yahoo_ticker": "M",
            "current_eligible_config": False,
            "currency": "USD",
            "exchange": "PCX",
            "research_cohort_state": "missing",
            "original_cohort": 2,
        },
    ]
    return pd.DataFrame(rows)


def exchange_mapping():
    return pd.DataFrame([{"exchange": "PCX", "calendar_id": "NYSE"}])


def fx_config():
    return pd.DataFrame([{"currency": "EUR", "provider_symbol": "EURUSD=X", "direction": "direct"}])


def test_crosswalk_preserves_legacy_ids_without_canonicalizing():
    match = crosswalk(registry(), {"HIST-A": "A", "HIST-B": "UNKNOWN"})
    a, b = match.set_index("historical_asset_id").loc[["HIST-A", "HIST-B"]].to_dict("records")
    assert a["state"] == "single_symbol_match_requires_instrument_review"
    assert a["candidate_current_asset_ids"] == ["CURRENT-A"]
    assert a["canonical_asset_id"] is None
    assert not a["identity_confirmed"]
    assert b["state"] == "historical_symbol_not_in_current_universe"


def test_ambiguous_historical_provider_symbol_cannot_be_auto_merged():
    match = crosswalk(registry(), {"HIST-A": "A", "HIST-B": "A"})
    assert set(match["state"]) == {"ambiguous_historical_symbol"}
    assert not match["identity_confirmed"].any()


def test_supplementary_preflight_fails_closed_on_missing_metadata():
    cross = crosswalk(registry(), {"HIST-A": "B", "HIST-B": "OTHER"})
    ready, blocked = supplementary_preflight(registry(), cross, exchange_mapping(), fx_config())
    assert ready.empty
    assert len(blocked) == 1
    assert "currency" in blocked.iloc[0]["blockers"]
    assert "historical_asset_id_collision_requires_review" in blocked.iloc[0]["blockers"]


def test_recovery_task_is_single_original_asset_and_non_executing():
    cross = crosswalk(registry(), {"HIST-A": "A", "HIST-B": "OTHER"})
    tasks = recovery_inventory(registry(), cross)
    assert tasks["asset_id"].tolist() == ["MISSING"]
    assert tasks.iloc[0]["original_cohort"] == 2
    assert not tasks.iloc[0]["execution_requested"]


def test_planning_bundle_contains_five_reconciled_artifacts_and_zero_approvals():
    payloads, manifest = build_plan_bundle(
        registry(),
        {"HIST-A": "A", "HIST-B": "OTHER"},
        exchange_mapping(),
        fx_config(),
        {"upstream_manifest_sha256": "a" * 64},
    )
    assert len(payloads) == 6  # Five JSONL artifacts plus immutable manifest.
    assert manifest["counts"]["historical_identity_candidates"] == 2
    assert manifest["counts"]["missing_cohort_asset_tasks"] == 1
    assert manifest["counts"]["historical_coverage_tasks"] == 2
    assert manifest["market_data_approved"] is False
    assert manifest["acquisitions_executed"] == 0



def test_supplementary_symbol_already_in_original_plan_is_blocked():
    data = registry()
    data.loc[data["asset_id"].eq("CURRENT-B"), "current_yahoo_ticker"] = "A"
    mapping = crosswalk(data, {"HIST-A": "UNKNOWN", "HIST-B": "DIFFERENT"})
    _, blocked = supplementary_preflight(data, mapping, exchange_mapping(), fx_config())
    assert "original_plan_provider_symbol_collision" in blocked.iloc[0]["blockers"]


def test_empty_legacy_crosswalk_has_stable_columns():
    data = registry()
    data = data.loc[data["identity_source"].ne("legacy_key_only")]
    mapping = crosswalk(data, {})
    assert mapping.empty
    assert "historical_asset_id" in mapping.columns
    assert "state" in mapping.columns
