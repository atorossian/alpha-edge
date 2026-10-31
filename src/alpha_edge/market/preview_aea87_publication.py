"""AEA-87 publication dry-run and immutable cutover manifest.

No active production manifests, caches, or snapshots are changed. This module
produces an audit-ready deployment inventory for the approved release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone

import boto3

APPROVED_PLAN = (
    "market/repairs/aea87/staging/20261030T190922Z_03657655/"
    "release_plans/20261031T125221550000Z_ae9af437/plan.json"
)


def read_document(s3, bucket, key):
    data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def select_candidates(reconciliation, plan):
    if reconciliation["run_id"] != plan["reconciliation_id"]:
        raise ValueError("Reconciliation and release plan do not match")
    assets = {item["asset_id"]: item for item in reconciliation["assets"]}
    if len(assets) != len(reconciliation["assets"]):
        raise ValueError("Duplicate reconciled assets")
    selected = []
    blocked = []
    for item in plan["assets"]:
        aid = item["asset_id"]
        asset = assets[aid]
        if item["status"] == "BLOCKED":
            blocked.append(aid)
            continue
        if (item["status"] != "CANDIDATE_ONLY" or item["blockers"]
                or asset["status"] != "VALID" or asset.get("flagged_retained")
                or asset.get("quarantine_rows")):
            raise ValueError(f"Unapproved publication candidate: {aid}")
        selected.append(aid)
    if len(selected) != 842 or len(blocked) != 11:
        raise ValueError("Unexpected candidate/quarantine coverage")
    if len(selected) + len(blocked) != len(assets):
        raise ValueError("Incomplete publication inventory")
    return sorted(selected), sorted(blocked)


def prepare(s3, bucket, plan_key):
    plan, plan_sha = read_document(s3, bucket, plan_key)
    prefix = "market/repairs/aea87/staging/"
    recon_key = prefix + plan["reconciliation_id"] + "/reconciliation.json"
    recon, recon_sha = read_document(s3, bucket, recon_key)
    if recon_sha != plan["reconciliation_sha256"]:
        raise ValueError("Reconciliation changed since release planning")
    candidates, blocked = select_candidates(recon, plan)
    manifest = {
        "kind": "AEA87_PUBLICATION_DRY_RUN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_plan_key": plan_key,
        "release_plan_sha256": plan_sha,
        "reconciliation_key": recon_key,
        "reconciliation_sha256": recon_sha,
        "candidate_asset_ids": candidates,
        "excluded_asset_ids": blocked,
        "publication_allowed": False,
        "required_controls": [
            "pause_ingestion_and_all_readers",
            "create_only_partition_publication",
            "backup_active_manifests",
            "manifest_concurrency_guard",
            "hype_obsolete_year_retirement",
            "recovery_journal_and_rollback_tests",
            "cache_and_snapshot_rebuild",
            "post_publication_integrity_check",
        ],
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    key = prefix + plan["reconciliation_id"] + "/publication_previews/" + run_id + "/manifest.json"
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(manifest, indent=2).encode(),
                  ContentType="application/json", IfNoneMatch="*")
    print(json.dumps({"candidates": len(candidates), "excluded": len(blocked),
                      "publication_allowed": False}, indent=2))
    print(f"PUBLICATION_PREVIEW s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser(description="Read-only AEA-87 publication dry run")
    parser.add_argument("--plan-key", default=APPROVED_PLAN)
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    args = parser.parse_args()
    prepare(boto3.client("s3", region_name=args.region), args.bucket, args.plan_key)


if __name__ == "__main__":
    main()
