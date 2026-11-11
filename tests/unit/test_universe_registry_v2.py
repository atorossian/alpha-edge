"""Current exclusion and historical retention are independent contracts."""

import hashlib
import io
import json

import pandas as pd
import pytest

from alpha_edge.market.universe_registry_v2 import (
    UniverseRegistryError,
    cohort_commit_presence,
    load_archived_plan,
    reconcile,
)


def config():
    return {
        "universe.csv": pd.DataFrame(
            [
                {"row_id": "R1", "asset_id": "A", "ticker": "LIVE", "yahoo_ticker": "LIVE", "include": "1"},
                {"row_id": "R2", "asset_id": "B", "ticker": "OLD", "yahoo_ticker": "OLD", "include": "0"},
                {"row_id": "R3", "asset_id": "C", "ticker": "MAN", "yahoo_ticker": "MAN.N", "include": "1"},
                {"row_id": "R4", "asset_id": "D", "ticker": "WHY", "yahoo_ticker": "WHY", "include": "0"},
            ]
        ),
        "asset_excluded.csv": pd.DataFrame(
            [{"target_row_id": "R2", "ticker": "OLD", "reason": "Not broker tradable"}]
        ),
        "universe_overrides.csv": pd.DataFrame(
            [{"target_row_id": "R3", "ticker": "MAN", "yahoo_ticker": "MAN.N", "lock_yahoo_ticker": "1"}]
        ),
        "universe_manual_additions.csv": pd.DataFrame([{"ticker": "MAN"}]),
        "ingest_force_refresh.csv": pd.DataFrame([{"ticker": "LIVE"}]),
    }


def plan():
    return pd.DataFrame(
        [
            {"asset_id": "A", "yahoo_ticker": "LIVE", "original_cohort": 1},
            {"asset_id": "B", "yahoo_ticker": "OLD", "original_cohort": 2},
            {"asset_id": "E", "yahoo_ticker": "DISAPPEARED", "original_cohort": 2},
        ]
    )


def test_excluded_asset_retained_historically_and_missing_cohort_exposed():
    rows, report = reconcile(config(), plan(), {1: "marker_present_unverified", 2: "missing"})
    by_id = rows.set_index("asset_id")
    assert not bool(by_id.loc["B", "current_eligible_config"])
    assert bool(by_id.loc["B", "historical_retention_required"])
    assert by_id.loc["B", "research_cohort_state"] == "missing"
    assert not bool(by_id.loc["B", "historical_price_approved"])
    assert bool(by_id.loc["E", "historical_retention_required"])
    assert not bool(by_id.loc["E", "current_eligible_config"])
    assert report["historical_only_assets"] == 1
    assert report["asset_issue_counts"]["missing_exclusion_reason"] == 1


def test_v1_only_is_inventory_evidence_not_approval():
    rows, report = reconcile(config(), plan(), {}, legacy_asset_ids={"Z", "B"})
    asset = rows.set_index("asset_id").loc["Z"]
    assert bool(asset["historical_retention_required"])
    assert not bool(asset["historical_price_approved"])
    assert report["legacy_only_assets"] == 1


def test_locked_symbol_drift_is_reported():
    cfg = config()
    cfg["universe.csv"].loc[2, "yahoo_ticker"] = "WRONG"
    _, report = reconcile(cfg, plan(), {})
    assert "locked_symbol_conflict:C" in report["configuration_issues"]


def test_duplicate_current_identity_fails_closed():
    cfg = config()
    cfg["universe.csv"].loc[2, "asset_id"] = "B"
    with pytest.raises(UniverseRegistryError, match="duplicate asset ID"):
        reconcile(cfg, plan(), {})


class S3:
    def __init__(self, values):
        self.values = values

    def get_object(self, Bucket, Key):
        if Key not in self.values:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        return {"Body": io.BytesIO(self.values[Key])}


def test_archived_plan_requires_hashes_and_unique_ids():
    prefix = "dev/market/lake_v2/archives/archive-a"
    root = "full_universe_preflight_v2/first_review"
    cohort = b"asset_id,yahoo_ticker\nA,ABC\n"
    fx = b"currency,provider_symbol,direction\n"
    names = {"cohort_0001.csv": cohort, "fx_config.csv": fx}
    plan_manifest = {
        "schema": "alpha_edge_lake_v2_universe_preflight_1",
        "cohorts": 1,
        "assets_included": 1,
        "files_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in names.items()},
    }
    names["manifest.json"] = json.dumps(plan_manifest).encode()
    values = {f"{prefix}/files/{root}/{k}": v for k, v in names.items()}
    inventory = {
        "schema": "alpha_edge_local_output_archive_v2_1",
        "files": [
            {"path": f"{root}/{k}", "sha256": hashlib.sha256(v).hexdigest()} for k, v in names.items()
        ],
    }
    values[f"{prefix}/_verified_inventory.json"] = json.dumps(inventory).encode()
    df, meta = load_archived_plan(S3(values), "bucket", "archive-a")
    assert df["asset_id"].tolist() == ["A"]
    assert meta["expected_cohorts"] == 1
    values[f"{prefix}/files/{root}/cohort_0001.csv"] = b"asset_id,yahoo_ticker\nB,ABC\n"
    with pytest.raises(UniverseRegistryError, match="SHA mismatch"):
        load_archived_plan(S3(values), "bucket", "archive-a")


def test_commit_marker_identity_invalid_is_not_success():
    class Fake:
        def get_object(self, Bucket, Key):
            return {"Body": io.BytesIO(json.dumps({"schema": "bad"}).encode())}

    states = cohort_commit_presence(Fake(), "bucket", "a" * 64, "2026-11-10", 1)
    assert states == {1: "invalid_marker"}
