"""Explicit, maintenance-only AEA87 single-asset reconstructed-history cutover.

Requires a fully prepared plan, existing manifests with immutable backups,
and operator attestations of exclusive reader/writer maintenance.
This code cannot itself stop remote readers or scheduled jobs.
"""
from __future__ import annotations
import argparse
import json

import boto3

from alpha_edge.market.aea87_publication_transaction import (
    checksum, fetch, put_once, switch_manifests, rollback, write_event, entries,
)
from alpha_edge.market.aea87_release_finalize import finalize

ROOT = "market/repairs/aea87/rebuild_publications/v1"


def require_ready_plan(s3, bucket, plan):
    if plan.get("status") != "PREPARED_NOT_PUBLISHED" or plan.get("active_histories_modified") is not False:
        raise ValueError("Plan was not successfully prepared")
    operations = plan.get("operations") or []
    if not operations or {op["table"] for op in operations} != {"ohlcv_usd", "returns_usd"}:
        raise ValueError("Incomplete market publication plan")
    if len({op["manifest_key"] for op in operations}) != len(operations):
        raise ValueError("Duplicate manifest target")
    aid = plan["asset_id"]
    for op in operations:
        if op["asset_id"] != aid:
            raise ValueError("Mixed asset identities in plan")
        if op["before_missing"]:
            # HYPE needs a separate legacy-retirement transaction with a
            # verifiable backup/listing of orphan partitions first.
            raise ValueError("Missing original manifest requires legacy-retirement workflow")
        if checksum(fetch(s3, bucket, op["manifest_key"])) != op["before_sha"]:
            raise ValueError("Active manifest drift: " + op["manifest_key"])
        if checksum(fetch(s3, bucket, op["backup_key"])) != op["before_sha"]:
            raise ValueError("Original manifest backup unavailable: " + op["manifest_key"])
        if op["part_key"] and checksum(fetch(s3, bucket, op["part_key"])) != op["part_sha"]:
            raise ValueError("Prepared partition failed checksum: " + op["part_key"])
        if checksum(op["after"].encode()) != op["after_sha"]:
            raise ValueError("New manifest checksum mismatch")
    return operations


def convert_to_transaction_ops(operations):
    return [{
        "asset_id": op["asset_id"], "manifest_key": op["manifest_key"],
        "before_sha": op["before_sha"], "after": op["after"],
        "after_sha": op["after_sha"], "backup_key": op["backup_key"],
        "part_key": op["part_key"], "staged_key": op["part_key"],
        "staged_sha": op["part_sha"],
    } for op in operations]


def activate(s3, bucket, plan, publication_id):
    records = require_ready_plan(s3, bucket, plan)
    root = f"{ROOT}/{publication_id}/transaction"
    ops = convert_to_transaction_ops(records)
    if entries(s3, bucket, root):
        raise RuntimeError("Publication already has journal events; use controlled recovery")
    put_once(s3, bucket, root + "/operations.json",
             json.dumps(ops, sort_keys=True).encode())
    write_event(s3, bucket, root, 0, "READY")
    switch_manifests(s3, bucket, root, ops, workers=1, batch_size=1)
    # The existing finalizer backs up snapshots and cache, and rebuilds cache.
    # An error must not be treated as success; keep the maintenance window open
    # until manifests and downstream artifacts are verified or restored.
    publication = {"candidates": [plan["asset_id"]], "operations": ops}
    try:
        result = finalize(s3, bucket, root, publication)
    except Exception as exc:
        # Restore original manifests immediately. Cache/snapshot restoration
        # must be assessed separately because it may already have been modified.
        rollback(s3, bucket, root, ops, workers=1)
        raise RuntimeError(
            "Manifest rollback executed after downstream failure; "
            "keep readers stopped and verify/restore backed-up cache and snapshots"
        ) from exc
    return result


def main():
    p = argparse.ArgumentParser(description="Maintenance-only explicit AEA87 rebuild activation")
    p.add_argument("--publication-id", required=True)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--confirm-exclusive-maintenance", action="store_true")
    p.add_argument("--confirm-downstream-refresh", action="store_true")
    p.add_argument("--confirm-prod-write", action="store_true")
    a = p.parse_args()
    if not (a.confirm_exclusive_maintenance and a.confirm_downstream_refresh and a.confirm_prod_write):
        p.error("All three confirmations required; stop external readers, schedulers and ingestion first")
    s3 = boto3.client("s3", region_name=a.region)
    key = f"{ROOT}/{a.publication_id}/plan.json"
    blob = fetch(s3, a.bucket, key)
    if blob is None:
        p.error("Prepared publication plan not found")
    plan = json.loads(blob)
    print(json.dumps(activate(s3, a.bucket, plan, a.publication_id), indent=2))


if __name__ == "__main__":
    main()
