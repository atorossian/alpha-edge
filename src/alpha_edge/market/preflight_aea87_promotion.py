"""AEA-87 Stage 5C: read-only promotion preflight, never writes production.

Checks active manifest optimistic-lock hashes and retirement inventory before
any future release implementation may be considered. Writes evidence to staging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections import Counter
from datetime import datetime, timezone

import boto3


def _load(s3, bucket, key):
    data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def preflight(s3, bucket, prefix, reconciliation_id):
    base = f"{prefix.strip('/')}/repairs/aea87/staging/"
    manifest_key = base + reconciliation_id + "/reconciliation.json"
    rec, rec_hash = _load(s3, bucket, manifest_key)
    if rec.get("run_id") != reconciliation_id or rec.get("kind") != "AEA87_STAGE5_RECONCILIATION":
        raise ValueError("Wrong reconciliation manifest")
    source_entries = {}
    for role, source in rec["sources"].items():
        campaign, sha = _load(s3, bucket, source["manifest_key"])
        validation, vsha = _load(s3, bucket, source["validation_key"])
        if sha != source["manifest_sha256"] or vsha != source["validation_sha256"]:
            raise ValueError(f"Changed source evidence: {role}")
        if campaign["run_id"] != source["run_id"] or validation["run_id"] != source["run_id"]:
            raise ValueError(f"Invalid campaign/validation identity: {role}")
        valid = {a["asset_id"]: a for a in validation["assets"]}
        source_entries[source["run_id"]] = (campaign, valid)
    findings, parts, seen_assets = [], 0, set()
    manifest_targets = {}
    for item in rec["assets"]:
        aid = item["asset_id"]
        if aid in seen_assets:
            findings.append({"asset_id": aid, "reason": "DUPLICATE_ASSET"})
        seen_assets.add(aid)
        campaign, valid = source_entries[item["campaign"]]
        staged = next((a for a in campaign["prepared"] if a["asset_id"] == aid), None)
        ev = valid.get(aid)
        if staged is None or ev is None or ev["status"] not in ("VALID", "VALID_WITH_QUARANTINE"):
            findings.append({"asset_id": aid, "reason": "MISSING_VALIDATED_SOURCE"})
            continue
        if len(staged["records"]) != item["record_count"]:
            findings.append({"asset_id": aid, "reason": "PARTITION_COUNT_MISMATCH"})
        if item["status"] == "VALID_WITH_QUARANTINE" or item["flagged_retained"] or item["quarantine_rows"]:
            findings.append({"asset_id": aid, "reason": "UNRESOLVED_QUARANTINE",
                             "flagged": item["flagged_retained"], "excluded": len(item["quarantine_rows"])})
        for part in staged["records"]:
            parts += 1
            key = part["manifest_key"]
            if key in manifest_targets:
                findings.append({"asset_id": aid, "reason": "DUPLICATE_TARGET_MANIFEST", "key": key})
            manifest_targets[key] = aid
            try:
                _, actual = _load(s3, bucket, key)
            except s3.exceptions.NoSuchKey:
                actual = None
            except Exception as exc:
                findings.append({"asset_id": aid, "reason": "MANIFEST_READ_FAILED",
                                 "key": key, "error": str(exc)})
                continue
            if actual != part.get("original_manifest_sha256"):
                findings.append({"asset_id": aid, "reason": "ACTIVE_MANIFEST_CHANGED", "key": key})
    hype = next((x for x in rec["assets"] if x["asset_id"] == "CRYPTO:HYPE-USD"), None)
    if not hype or not hype.get("identity_reset") or not hype.get("superseded_manifest_keys"):
        findings.append({"reason": "HYPE_RESET_RETIREMENT_EVIDENCE_MISSING"})
    else:
        for key in hype["superseded_manifest_keys"]:
            if key in manifest_targets:
                # An old year may also have a newly staged replacement, but
                # every old reference still requires an explicit retirement step.
                pass
            findings.append({"asset_id": "CRYPTO:HYPE-USD", "reason": "RETIRE_OLD_MANIFEST_REQUIRED",
                             "key": key})
    if rec.get("production_promotion_allowed") is not False:
        findings.append({"reason": "UNEXPECTED_SOURCE_PROMOTION_PERMISSION"})
    if rec.get("blockers"):
        findings.append({"reason": "RECONCILIATION_BLOCKERS", "items": rec["blockers"]})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    result = {
        "kind": "AEA87_PROMOTION_PREFLIGHT",
        "run_id": run_id,
        "reconciliation_id": reconciliation_id,
        "reconciliation_sha256": rec_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "assets_checked": len(seen_assets),
        "parts_checked": parts,
        "finding_counts": dict(Counter(f["reason"] for f in findings)),
        "findings": findings,
        "production_promotion_allowed": False,
        "read_only": True,
    }
    key = base + reconciliation_id + "/preflights/" + run_id + "/report.json"
    data = json.dumps(result, indent=2, sort_keys=True).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/json", IfNoneMatch="*")
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, indent=2))
    print(f"PREFLIGHT_REPORT s3://{bucket}/{key}")
    return result


def main():
    p = argparse.ArgumentParser(description="AEA-87 read-only pre-promotion integrity and blocker inventory")
    p.add_argument("--reconciliation-id", required=True)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--market-prefix", default="market")
    args = p.parse_args()
    preflight(boto3.client("s3", region_name=args.region), args.bucket,
              args.market_prefix, args.reconciliation_id)


if __name__ == "__main__":
    main()
