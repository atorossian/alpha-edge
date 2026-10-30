"""AEA-87 Stage 5: immutable, fail-closed reconciliation of staged campaigns.

No Yahoo downloads, no market production writes or manifest switching.
A combined campaign is a plan/evidence index, NOT permission to promote.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone

import boto3

ORIGINAL = "20261028T115951Z_2a5399da"
BACKFILL = "20261030T120920Z_d48daac4"
SUPPLEMENTAL = "20261028T200842Z_e3cdf5d3"
EXCLUDED = "EQH57f251f2d5f83e9b"
HYPE = "CRYPTO:HYPE-USD"
PPLI = "EQH01e3d9d4c84963e1"
LKFT = "EQH273a4d84ac70f156"
OVERRIDES = {PPLI, LKFT}


def _read_json(s3, bucket, key):
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body), hashlib.sha256(body).hexdigest()


def _index(items):
    out = {}
    for item in items:
        aid = item["asset_id"]
        if aid in out:
            raise ValueError(f"Duplicate asset ID in source: {aid}")
        out[aid] = item
    return out


def _validated(s3, bucket, root, run_id, report_key):
    expected_prefix = root + run_id + "/validations/"
    if not report_key.startswith(expected_prefix) or not report_key.endswith("/report.json"):
        raise ValueError(f"Invalid validation evidence location: {report_key}")
    report, sha = _read_json(s3, bucket, report_key)
    if report.get("run_id") != run_id:
        raise ValueError(f"Validation evidence wrong campaign: {run_id}")
    entries = _index(report["assets"])
    if len(entries) != report["assets_checked"]:
        raise ValueError("Validation asset count mismatch")
    return entries, sha


def latest_validation(s3, bucket, root, run_id):
    """Discover latest completed full validation, never a selected-subset run."""
    prefix = root + run_id + "/validations/"
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", [])
                    if obj["Key"].endswith("/report.json"))
    if not keys:
        raise ValueError(
            f"No persisted full validation found for {run_id}; run validate_aea87_campaign first"
        )
    # New IDs are UTC sortable. Reject stale/incomplete reports instead of
    # falling back to older, potentially unverified output.
    key = max(keys)
    report, _ = _read_json(s3, bucket, key)
    if report.get("run_id") != run_id:
        raise ValueError("Validation run ID mismatch")
    return key


def reconcile(s3, bucket, market_prefix, original, backfill, supplemental,
              original_evidence, backfill_evidence, supplemental_evidence,
              output_id, lkft_repair=None, lkft_evidence=None):
    root = f"{market_prefix.strip('/')}/repairs/aea87/staging/"
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,80}", output_id):
        raise ValueError("Invalid reconciliation ID")
    sources = {}
    for role, rid in (("original", original), ("backfill", backfill), ("supplemental", supplemental)):
        if rid in (x["id"] for x in sources.values()):
            raise ValueError("Campaign IDs must differ")
        key = root + rid + "/prepared_campaign.json"
        manifest, sha = _read_json(s3, bucket, key)
        if manifest.get("run_id") != rid:
            raise ValueError("Source manifest run ID mismatch")
        sources[role] = {"id": rid, "key": key, "sha256": sha,
                         "manifest": manifest, "assets": _index(manifest["prepared"])}
    if lkft_repair:
        if lkft_repair in (original, backfill, supplemental):
            raise ValueError("LKFT correction run must be distinct from all source campaigns")
        key = root + lkft_repair + "/prepared_campaign.json"
        manifest, sha = _read_json(s3, bucket, key)
        if manifest.get("run_id") != lkft_repair or manifest.get("source_campaign") != supplemental:
            raise ValueError("LKFT repair must derive from supplemental campaign")
        entries = _index(manifest["prepared"])
        if set(entries) != {LKFT}:
            raise ValueError("LKFT repair must contain exactly LKFT")
        sources["lkft_repair"] = {"id": lkft_repair, "key": key, "sha256": sha,
                                  "manifest": manifest, "assets": entries}
    orig = sources["original"]["assets"]
    corrected = sources["backfill"]["assets"]
    extra = sources["supplemental"]["assets"]
    if sources["backfill"]["manifest"].get("source_campaign") != original:
        raise ValueError("Backfill does not point to the original campaign")
    if not set(corrected).issubset(orig):
        raise ValueError("Backfill includes assets absent from original")
    if set(orig) & OVERRIDES:
        raise ValueError("Original campaign unexpectedly includes supplemental-only IDs")
    if not OVERRIDES.issubset(extra):
        raise ValueError("Corrected PPLI/LKFT assets missing from supplemental campaign")
    if HYPE not in extra:
        raise ValueError("Expected staged HYPE candidate missing")
    if EXCLUDED in orig or EXCLUDED in corrected:
        raise ValueError("Excluded APLS appears as prepared")

    # Do not silently accept partially validated campaigns: each chosen asset
    # must have matching evidence in its own campaign.
    evidence = {}
    for role, path in (("original", original_evidence), ("backfill", backfill_evidence),
                       ("supplemental", supplemental_evidence)):
        entries, sha = _validated(s3, bucket, root, sources[role]["id"], path)
        if set(entries) != set(sources[role]["assets"]):
            raise ValueError(f"{role} validation does not cover all prepared assets")
        evidence[role] = {"key": path, "sha256": sha, "assets": entries}

    if lkft_repair:
        if not lkft_evidence:
            raise ValueError("LKFT repair validation report is required")
        entries, sha = _validated(s3, bucket, root, lkft_repair, lkft_evidence)
        if set(entries) != {LKFT}:
            raise ValueError("LKFT correction validation does not cover exactly LKFT")
        evidence["lkft_repair"] = {"key": lkft_evidence, "sha256": sha, "assets": entries}
    chosen = {}
    decisions = []
    for aid in sorted(orig):
        role = "backfill" if aid in corrected else "original"
        item = sources[role]["assets"][aid]
        valid = evidence[role]["assets"].get(aid)
        if not valid or valid["status"] not in ("VALID", "VALID_WITH_QUARANTINE") or valid.get("errors"):
            raise ValueError(f"Selected {role} asset {aid} lacks passing validation evidence")
        chosen[aid] = {"asset_id": aid, "campaign": sources[role]["id"],
                       "status": valid["status"], "record_count": len(item["records"]),
                       "quarantine_rows": item.get("quarantine_rows", []),
                       "flagged_retained": item.get("flagged_retained", 0),
                       "return_break_dates": item.get("return_break_dates", [])}
    for aid in sorted(OVERRIDES):
        role = "lkft_repair" if aid == LKFT and lkft_repair else "supplemental"
        valid = evidence[role]["assets"].get(aid)
        if valid is None or valid["status"] not in ("VALID", "VALID_WITH_QUARANTINE") or valid.get("errors"):
            raise ValueError(f"Selected {role} {aid} lacks passing validation evidence")
        item = sources[role]["assets"][aid]
        chosen[aid] = {"asset_id": aid, "campaign": sources[role]["id"], "status": valid["status"],
                       "record_count": len(item["records"]),
                       "quarantine_rows": item.get("quarantine_rows", []),
                       "flagged_retained": item.get("flagged_retained", 0),
                       "return_break_dates": item.get("return_break_dates", [])}
    # The existing HYPE campaign contains retained prices from the wrong
    # instrument. Never select it in the combined data.
    decisions.extend([
        {"asset_id": HYPE, "status": "IDENTITY_RESET_REQUIRED",
         "reason": "Existing supplemental HYPE history may be mixed with a different instrument"},
        {"asset_id": EXCLUDED, "status": "EXCLUDED",
         "reason": "APLS delisted/acquired; not a trusted repair candidate"},
    ])
    if set(chosen) & {HYPE, EXCLUDED}:
        raise ValueError("Blocked HYPE/APLS was selected")
    for aid, entry in chosen.items():
        if entry["status"] not in ("VALID", "VALID_WITH_QUARANTINE"):
            raise ValueError(f"Unexpected validation status for {aid}")
    blocked = sum(v["status"] == "VALID_WITH_QUARANTINE" for v in chosen.values())
    counts = dict(Counter(v["campaign"] for v in chosen.values()))
    result = {
        "kind": "AEA87_STAGE5_RECONCILIATION",
        "run_id": output_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            role: {"run_id": value["id"], "manifest_key": value["key"],
                   "manifest_sha256": value["sha256"], "validation_key": evidence[role]["key"],
                   "validation_sha256": evidence[role]["sha256"]}
            for role, value in sources.items()
        },
        "summary": {"selected_assets": len(chosen), "campaign_asset_counts": counts,
                    "assets_with_quarantine": blocked, "blocked_assets": len(decisions)},
        "assets": [chosen[a] for a in sorted(chosen)],
        "decisions": decisions,
        "production_promotion_allowed": False,
        "blockers": ["HYPE_REQUIRES_FRESH_ONLY_RESET",
                     "UNRESOLVED_QUARANTINE",
                     "PRODUCTION_PUBLICATION_AND_ROLLBACK_UNTESTED"],
    }
    key = root + output_id + "/reconciliation.json"
    data = json.dumps(result, sort_keys=True, indent=2, default=str).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/json",
                  IfNoneMatch="*")
    check = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if hashlib.sha256(check).digest() != hashlib.sha256(data).digest():
        raise IOError("Reconciliation S3 readback hash mismatch")
    print(json.dumps(result["summary"], indent=2))
    print(f"RECONCILIATION_REPORT s3://{bucket}/{key}")
    return result


def main():
    p = argparse.ArgumentParser(description="AEA-87 read-only source / staging-only reconciliation")
    p.add_argument("--original-validation", help="Original full validation report S3 key")
    p.add_argument("--backfill-validation", help="Corrected full validation report S3 key")
    p.add_argument("--supplemental-validation", help="Supplemental full validation report S3 key")
    p.add_argument("--original", default=ORIGINAL)
    p.add_argument("--backfill", default=BACKFILL)
    p.add_argument("--supplemental", default=SUPPLEMENTAL)
    p.add_argument("--lkft-repair", help="Validated LKFT-only derived-price backfill campaign ID")
    p.add_argument("--lkft-validation", help="Optional validation report key for LKFT correction")
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--market-prefix", default="market")
    args = p.parse_args()
    rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    s3 = boto3.client("s3", region_name=args.region)
    root = f"{args.market_prefix.strip('/')}/repairs/aea87/staging/"
    reports = [
        explicit or latest_validation(s3, args.bucket, root, source)
        for explicit, source in (
            (args.original_validation, args.original),
            (args.backfill_validation, args.backfill),
            (args.supplemental_validation, args.supplemental),
        )
    ]
    if args.lkft_repair:
        reports.append(args.lkft_validation or latest_validation(s3, args.bucket, root, args.lkft_repair))
    for label, path in zip(("original", "backfill", "supplemental", "lkft_repair"), reports):
        print(f"[stage5] {label} validation: s3://{args.bucket}/{path}", flush=True)
    reconcile(s3, args.bucket, args.market_prefix, args.original, args.backfill, args.supplemental,
              *reports[:3], rid, lkft_repair=args.lkft_repair,
              lkft_evidence=reports[3] if args.lkft_repair else None)


if __name__ == "__main__":
    main()
