"""Build an evidence-based, non-production v2 release inventory.

Combines original cohorts, recovered source assets, reviewed single-asset
backfills and historical-retention requirements. Does not import v1 prices or
activate a production pointer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import boto3
import pandas as pd

from alpha_edge.market.asset_level_recovery_v2 import RECOVERY_ROOT, _json_bytes
from alpha_edge.market.run_s3_full_history_backfill_v2 import (
    upload_verified, verify_commit, _get_remote,
)
from alpha_edge.market.universe_registry_v2 import (
    _sha, cohort_commit_presence, load_archived_plan,
)

ROOT = "dev/market/lake_v2/release_candidates"
ORIGINAL_END = "2026-11-10"
REVIEWED = {
    "EQH4a3a0282d0f4690f": (
        "KMB", "06aae8efc7fb712b2aed410d", "2026-11-11",
    ),
    "EQHe874aca04f607837": (
        "EURUSD=X", "6e117c996ce8375eb1985659", "2026-11-11",
    ),
}
HISTORICAL_RETAIN = {
    "EQH9b41827cef0e0979": "KVUE",
    "EQH19f7f86807d8c9ce": "WBD",
    "EQH0198c6d7d97c1097": "CSAN",
    "EQHadc710ac75ecda67": "QRVO",
}


def _manifest(s3, bucket: str, key: str):
    raw = _get_remote(s3, bucket, key)
    return (json.loads(raw), _sha(raw)) if raw is not None else (None, None)


def make_inventory(
    s3, bucket: str, *, archive_id: str, universe_csv: Path,
    original_end: str = ORIGINAL_END,
) -> tuple[list[dict], dict]:
    original, info = load_archived_plan(s3, bucket, archive_id)
    states = cohort_commit_presence(
        s3, bucket, info["plan_sha256"], original_end, info["expected_cohorts"]
    )
    current = pd.read_csv(universe_csv, dtype=str).fillna("")
    if current["asset_id"].duplicated().any():
        raise ValueError("Duplicate asset_id in current universe")
    configured = current.set_index("asset_id")
    # Discover already-committed per-asset candidates by their immutable marker.
    # No Parquet objects are downloaded. Presence is NOT approval.
    recovered_candidate_markers = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="dev/market/lake_v2/recovered_candidates/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith("/_committed.json") or "/asset_id=" not in key:
                continue
            aid = key.rsplit("/asset_id=", 1)[1].split("/", 1)[0]
            recovered_candidate_markers.setdefault(aid, []).append(key)
    records = []
    all_ids = sorted(set(original["asset_id"]) | set(current["asset_id"]) | set(HISTORICAL_RETAIN))
    original_by_id = original.set_index("asset_id")
    for aid in all_ids:
        old = original_by_id.loc[aid] if aid in original_by_id.index else None
        now = configured.loc[aid] if aid in configured.index else None
        record = {
            "asset_id": aid,
            "current_eligible": bool(now is not None and now["include"] == "1"),
            "original_plan_present": old is not None,
            "original_cohort": int(old["original_cohort"]) if old is not None else None,
            "historical_retention_required": aid in HISTORICAL_RETAIN,
            "source_state": "not_acquired_in_v2",
            "candidate_state": "not_standardized",
            "external_evidence_approved": False,
            "production_eligible": False,
        }
        if old is not None:
            cohort = int(old["original_cohort"])
            if states[cohort] == "marker_present_unverified":
                record["source_state"] = "original_cohort_marker_unverified"
                record["candidate_state"] = "original_candidate_marker_unverified"
            elif states[cohort] == "missing":
                source = {
                    "asset_id": aid,
                    "original_cohort": cohort,
                    "original_plan_sha256": info["plan_sha256"],
                    "provider_symbol": str(old["yahoo_ticker"]),
                    "currency": str(old["currency"]),
                    "calendar_id": str(old["calendar_id"]),
                    "end_exclusive": original_end,
                }
                run_id = _sha(_json_bytes({"schema": "aea87_recovery_identity_v1", **source}))
                key = f"{RECOVERY_ROOT}/run-{run_id}/manifest.json"
                recovery, digest = _manifest(s3, bucket, key)
                if recovery:
                    record["source_state"] = (
                        "bronze_recovered" if recovery.get("schema") == "aea87_asset_recovery_candidate_v1"
                        else "source_acquisition_failed"
                    )
                    record["source_manifest"] = f"s3://{bucket}/{key}"
                    record["source_manifest_sha256"] = digest
        if aid in recovered_candidate_markers:
            markers = recovered_candidate_markers[aid]
            record["candidate_state"] = "recovered_candidate_marker_unverified"
            record["recovered_candidate_markers"] = [
                f"s3://{bucket}/{key}" for key in sorted(markers)
            ]
        if aid in REVIEWED:
            symbol, prefix_hash, end = REVIEWED[aid]
            key = (f"dev/market/lake_v2/backfills/{prefix_hash}/"
                   f"end-{end}/cohort-0001/_committed.json")
            reviewed, digest = _manifest(s3, bucket, key)
            if reviewed:
                record["source_state"] = "reviewed_full_history_committed"
                record["candidate_state"] = "reviewed_candidate_marker_unverified"
                record["reviewed_symbol"] = symbol
                record["source_manifest"] = f"s3://{bucket}/{key}"
                record["source_manifest_sha256"] = digest
        if aid in HISTORICAL_RETAIN:
            record["historical_symbol"] = HISTORICAL_RETAIN[aid]
            record["legacy_history_retained_in_v1"] = True
            record["historical_data_migration_complete"] = False
        records.append(record)
    summary = {
        "schema": "aea87_release_inventory_v2_1",
        "assets": len(records),
        "source_states": dict(Counter(r["source_state"] for r in records)),
        "candidate_states": dict(Counter(r["candidate_state"] for r in records)),
        "current_eligible": sum(r["current_eligible"] for r in records),
        "historical_retention_pending": sum(r.get("legacy_history_retained_in_v1", False) for r in records),
        "original_plan_sha256": info["plan_sha256"],
        "original_end_exclusive": original_end,
        "current_universe_sha256": hashlib.sha256(universe_csv.read_bytes()).hexdigest(),
        "production_eligible": False,
        "release_state": "blocked_pending_verified_candidates_and_external_evidence",
        "limitations": [
            "Original cohort marker presence is not full content verification",
            "Recovered Bronze alone does not establish Silver/Gold acceptance",
            "Excluded legacy historical observations remain in v1 until approved migration",
            "Independent historical source and corporate-action evidence remain outstanding",
            "No production pointer changes or legacy data imports",
        ],
    }
    return records, summary


def run(
    s3, bucket: str, *, archive_id: str, universe_csv: Path,
    original_end: str = ORIGINAL_END, execute: bool = False,
) -> dict:
    records, summary = make_inventory(
        s3, bucket, archive_id=archive_id, universe_csv=universe_csv,
        original_end=original_end,
    )
    payload = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode()
        for row in records
    )
    material = _json_bytes({
        "schema": summary["schema"],
        "original_plan_sha256": summary["original_plan_sha256"],
        "current_universe_sha256": summary["current_universe_sha256"],
        "inventory_sha256": _sha(payload),
    })
    release_sha = _sha(material)
    prefix = f"{ROOT}/run-{release_sha}"
    if not execute:
        return {**summary, "status": "dry_run",
                "candidate_prefix": f"s3://{bucket}/{prefix}", "s3_writes": 0}
    if verify_commit(s3, bucket, prefix, release_sha) is None:
        with tempfile.TemporaryDirectory(prefix="aea87_release_inventory_") as temp:
            root = Path(temp)
            (root / "asset_inventory.jsonl").write_bytes(payload)
            (root / "summary.json").write_bytes(_json_bytes(summary))
            (root / "source_contract.json").write_bytes(material)
            upload_verified(
                s3, bucket=bucket, prefix=prefix, local_dir=root,
                plan_hash=release_sha, end=original_end,
            )
    return {**summary, "status": "published_blocked_release_inventory",
            "candidate_prefix": f"s3://{bucket}/{prefix}", "production_eligible": False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--archive-id", default="old-outputs-20261110")
    p.add_argument("--universe", type=Path, default=Path("data/universe/universe.csv"))
    p.add_argument("--end", default=ORIGINAL_END)
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(
        boto3.client("s3", region_name=args.region), args.bucket,
        archive_id=args.archive_id, universe_csv=args.universe,
        original_end=args.end, execute=args.execute,
    ), indent=2))


if __name__ == "__main__":
    main()
