"""AEA-87 guarded release planner.

Reads a reconciled campaign and builds an immutable per-asset deployment plan.
No production object is modified. Publication is intentionally NOT available in
this module: sequential manifest replacement cannot provide atomic reader views.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections import Counter
from datetime import datetime, timezone

import boto3


def _raw(s3, bucket, key):
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def _json(s3, bucket, key):
    data = _raw(s3, bucket, key)
    return json.loads(data), hashlib.sha256(data).hexdigest()


def _active_hash(s3, bucket, key):
    try:
        return hashlib.sha256(_raw(s3, bucket, key)).hexdigest()
    except s3.exceptions.NoSuchKey:
        return None


def classify_asset(asset, staged_records, active_hashes):
    """Fail closed on quarantine, stale active refs and missing retire inventory."""
    reasons = []
    if asset.get("status") != "VALID":
        reasons.append("QUARANTINE_OR_NONVALID_STATUS")
    if asset.get("flagged_retained") or asset.get("quarantine_rows"):
        reasons.append("UNRESOLVED_DATA_QUALITY")
    if not staged_records:
        reasons.append("NO_STAGED_RECORDS")
    record_keys = set()
    for record in staged_records:
        key = record["manifest_key"]
        if key in record_keys:
            reasons.append("DUPLICATE_MANIFEST_TARGET")
        record_keys.add(key)
        if active_hashes.get(key) != record.get("original_manifest_sha256"):
            reasons.append("STALE_ACTIVE_MANIFEST")
    if asset.get("identity_reset"):
        retire_keys = set(asset.get("superseded_manifest_keys", []))
        if not retire_keys:
            reasons.append("MISSING_RESET_RETIREMENT_INVENTORY")
        for key in retire_keys:
            if key not in active_hashes:
                reasons.append("UNVERIFIED_RETIREMENT_MANIFEST")
        # All old manifests (including years with a new replacement) must be
        # accounted for; the publisher must retire old years atomically.
    return sorted(set(reasons))


def plan(s3, bucket, prefix, reconciliation_id):
    root = f"{prefix.strip('/')}/repairs/aea87/staging/"
    reconciliation, reconciliation_hash = _json(
        s3, bucket, root + reconciliation_id + "/reconciliation.json")
    if reconciliation.get("run_id") != reconciliation_id:
        raise ValueError("Reconciliation ID mismatch")
    sources = {}
    for role, source in reconciliation["sources"].items():
        manifest, manifest_hash = _json(s3, bucket, source["manifest_key"])
        validation, validation_hash = _json(s3, bucket, source["validation_key"])
        if manifest_hash != source["manifest_sha256"] or validation_hash != source["validation_sha256"]:
            raise ValueError(f"Changed staged source or evidence: {role}")
        if manifest["run_id"] != source["run_id"] or validation["run_id"] != source["run_id"]:
            raise ValueError("Campaign/validation identity mismatch")
        prepared = {a["asset_id"]: a for a in manifest["prepared"]}
        validated = {a["asset_id"]: a for a in validation["assets"]}
        if len(prepared) != len(manifest["prepared"]) or len(validated) != len(validation["assets"]):
            raise ValueError("Duplicated asset identities")
        sources[source["run_id"]] = (prepared, validated)
    active = {}
    output = []
    seen = set()
    for asset in reconciliation["assets"]:
        aid = asset["asset_id"]
        if aid in seen:
            raise ValueError(f"Duplicate reconciled asset: {aid}")
        seen.add(aid)
        prepared, validated = sources[asset["campaign"]]
        if aid not in prepared or aid not in validated:
            raise ValueError(f"Missing source/evidence: {aid}")
        staged = prepared[aid]
        v = validated[aid]
        if v["status"] != asset["status"] or v.get("errors"):
            raise ValueError(f"Validation disagreement for {aid}")
        records = staged["records"]
        keys = {record["manifest_key"] for record in records}
        keys.update(asset.get("superseded_manifest_keys", []))
        for key in keys - active.keys():
            active[key] = _active_hash(s3, bucket, key)
        reasons = classify_asset(asset, records, active)
        output.append({
            "asset_id": aid, "source_campaign": asset["campaign"],
            "status": "BLOCKED" if reasons else "CANDIDATE_ONLY",
            "blockers": reasons,
            "expected_manifest_hashes": {k: active[k] for k in sorted(keys)},
            "record_count": len(records),
            "retire_manifest_keys": sorted(set(asset.get("superseded_manifest_keys", []))),
        })
    counts = Counter(x["status"] for x in output)
    blockers = Counter(reason for x in output for reason in x["blockers"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    report = {
        "kind": "AEA87_GUARDED_RELEASE_PLAN", "run_id": run_id,
        "reconciliation_id": reconciliation_id, "reconciliation_sha256": reconciliation_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {"total": len(output), "asset_statuses": dict(counts),
                    "blocker_counts": dict(blockers)},
        "assets": output, "production_promotion_allowed": False,
        "publication_implementation_status": "NOT_ENABLED",
        "required_before_promotion": [
            "Classify and resolve all quarantined historical observations",
            "Rebase stale partitions using current active manifest data and repeat validation",
            "Implement reader-consistent release switching, including HYPE year retirement",
            "Test interruption, restart, rollback, caches and snapshots against a disposable namespace",
        ],
    }
    key = root + reconciliation_id + "/release_plans/" + run_id + "/plan.json"
    data = json.dumps(report, indent=2, sort_keys=True).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/json",
                  IfNoneMatch="*")
    print(json.dumps(report["summary"], indent=2))
    print(f"RELEASE_PLAN s3://{bucket}/{key}")
    return report


def main():
    p = argparse.ArgumentParser(description="AEA-87 guarded read-only promotion planning")
    p.add_argument("--reconciliation-id", required=True)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--market-prefix", default="market")
    args = p.parse_args()
    plan(boto3.client("s3", region_name=args.region),
         args.bucket, args.market_prefix, args.reconciliation_id)


if __name__ == "__main__":
    main()
