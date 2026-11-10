"""S3-native immutable market-data v2 release controller.

A release is a complete, hash-pinned mapping of cohort snapshots. Staging is
safe even with unresolved quality. Activation is fail-closed: ALL cohorts must
have accepted independent reports and explicit source-evidence gate approval.
Production v1 keys are never changed. Uses Alpha Edge's existing S3 audit log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from typing import Any

from botocore.exceptions import ClientError

from alpha_edge.core.audit import build_audit_event, write_audit_event, make_run_id
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation

ROOT = "market/lake_v2"
CURRENT = ROOT + "/releases/current.json"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,100}$")
EVIDENCE_GATES = (
    "independent_source_observation_verification",
    "corporate_action_adjustment_factor_validation",
    "historical_provider_revision_comparison",
    "exchange_specific_fx_observation_timing_policy",
    "endpoints_and_market_close_completeness",
)


class ReleaseError(RuntimeError):
    pass


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encoded(obj: dict) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()


def get(s3, bucket: str, key: str) -> bytes | None:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def immutable_put(s3, bucket: str, key: str, body: bytes) -> None:
    if get(s3, bucket, key) is not None:
        raise ReleaseError(f"Immutable key occupied: {key}")
    s3.put_object(Bucket=bucket, Key=key, Body=body,
                  IfNoneMatch="*", ContentType="application/json")


def verify_object(s3, bucket: str, record: dict) -> bytes:
    key, digest = record.get("key"), record.get("sha256")
    if not isinstance(key, str) or not (
        key.startswith(ROOT + "/")
        or key.startswith("dev/market/lake_v2/")
    ) or "/../" in key or "/./" in key:
        raise ReleaseError(f"Untrusted market data key: {key!r}")
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ReleaseError("Missing verified source SHA-256")
    body = get(s3, bucket, key)
    if body is None or sha(body) != digest:
        raise ReleaseError(f"Missing/corrupt data artifact: {key}")
    return body


def validate_release(s3, bucket: str, release: dict, *, require_approval: bool) -> dict:
    if release.get("schema") != "alpha_edge_v2_release_candidate_1":
        raise ReleaseError("Invalid release schema")
    if not SAFE_ID.fullmatch(str(release.get("release_id", ""))):
        raise ReleaseError("Invalid release ID")
    as_of = release.get("as_of")
    if not isinstance(as_of, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of):
        raise ReleaseError("Invalid release as_of")
    expected = release.get("expected_cohorts")
    cohorts = release.get("cohorts")
    if not isinstance(expected, int) or expected < 1 or not isinstance(cohorts, list):
        raise ReleaseError("Missing expected cohort inventory")
    ids = [x.get("cohort") for x in cohorts]
    if len(ids) != expected or sorted(ids) != list(range(1, expected + 1)):
        raise ReleaseError("Incomplete/duplicate cohort coverage")
    totals = {"cohorts": 0, "assets": 0, "rows": 0}
    seen_assets = set()
    for c in cohorts:
        if not isinstance(c.get("assets"), list) or not c["assets"]:
            raise ReleaseError("Missing cohort asset reconciliation")
        if seen_assets.intersection(c["assets"]):
            raise ReleaseError("Asset overlaps between release cohorts")
        seen_assets.update(c["assets"])
        for field in ("bronze", "silver", "gold", "fx", "acceptance"):
            if not isinstance(c.get(field), dict):
                raise ReleaseError(f"Missing {field} lineage")
            data = verify_object(s3, bucket, c[field])
            if field == "acceptance":
                report = json.loads(data)
                if report.get("internal_integrity_status") != "passed":
                    raise ReleaseError("Independent arithmetic integrity not passed")
                if require_approval:
                    if report.get("acceptance") != "accepted":
                        raise ReleaseError(f"Independent acceptance blocked for cohort {c['cohort']}")
                    if any(report.get("blocker_counts", {}).values()):
                        raise ReleaseError("Nonzero blocker counts")
                    if report.get("open_external_evidence_gates"):
                        raise ReleaseError("Outstanding independent evidence gates")
        if require_approval:
            signed = c.get("independent_approval")
            if not isinstance(signed, dict) or not isinstance(signed.get("reviewer"), str) or not signed["reviewer"].strip():
                raise ReleaseError("Explicit independent reviewer approval required")
            if signed.get("acceptance_manifest_sha256") != c["acceptance"]["sha256"]:
                raise ReleaseError("Reviewer approval not pinned to acceptance evidence")
        totals["cohorts"] += 1
        totals["assets"] += len(c["assets"])
        totals["rows"] += int(c.get("gold_rows", 0))
    if totals["assets"] != release.get("expected_assets"):
        raise ReleaseError("Release asset population differs from expected universe")
    if require_approval:
        if release.get("indicator_validation") != "passed":
            raise ReleaseError("Market indicator validation not passed")
        if release.get("consumer_contract_validation") != "passed":
            raise ReleaseError("Gold consumer schema/contract validation not passed")
        if release.get("universe_reconciliation") != "passed":
            raise ReleaseError("Universe reconciliation not approved")
        if not isinstance(release.get("independent_signoff"), str) or not release["independent_signoff"].strip():
            raise ReleaseError("Final release signoff required")
    return totals


def stage(s3, *, bucket: str, release: dict, execute: bool) -> dict:
    totals = validate_release(s3, bucket, release, require_approval=False)
    release_key = f"{ROOT}/releases/candidates/{release['release_id']}.json"
    body = encoded(release)
    if execute:
        existing = get(s3, bucket, release_key)
        if existing is None:
            immutable_put(s3, bucket, release_key, body)
        elif existing != body:
            raise ReleaseError("Existing release ID has different immutable contents")
        if get(s3, bucket, release_key) != body:
            raise ReleaseError("Staged release read-back mismatch")
    return {"state": "staged" if execute else "dry_run", "release_key": release_key,
            "release_sha256": sha(body), **totals, "consumer_cutover": False}


def activate(s3, *, bucket: str, release_key: str, approved_sha256: str,
             execute: bool) -> dict:
    if not re.fullmatch(r"market/lake_v2/releases/candidates/[A-Za-z0-9_.-]+\.json", release_key):
        raise ReleaseError("Activation only accepts immutable v2 release candidates")
    if not re.fullmatch("[0-9a-f]{64}", approved_sha256):
        raise ReleaseError("Explicit 64-char SHA-256 approval required")
    body = get(s3, bucket, release_key)
    if body is None or sha(body) != approved_sha256:
        raise ReleaseError("Approval hash mismatches immutable staged release")
    release = json.loads(body)
    validate_release(s3, bucket, release, require_approval=True)
    current = get(s3, bucket, CURRENT)
    previous = json.loads(current) if current else None
    if previous is not None and previous.get("release_id") == release["release_id"]:
        return {"state": "already_current", "release_id": release["release_id"]}
    pointer = {
        "schema": "alpha_edge_v2_release_pointer_1",
        "release_id": release["release_id"],
        "release_key": release_key, "release_sha256": approved_sha256,
        "previous_release_id": previous.get("release_id") if previous else None,
        "previous_release_key": previous.get("release_key") if previous else None,
        "as_of": release["as_of"],
    }
    if not execute:
        return {"state": "dry_run", "pointer": pointer}
    params = {"IfNoneMatch": "*"} if current is None else {}
    if current is not None:
        etag = s3.head_object(Bucket=bucket, Key=CURRENT)["ETag"]
        if get(s3, bucket, CURRENT) != current:
            raise ReleaseError("Concurrent pointer modification")
        params["IfMatch"] = etag
    s3.put_object(Bucket=bucket, Key=CURRENT, Body=encoded(pointer),
                  ContentType="application/json", **params)
    if get(s3, bucket, CURRENT) != encoded(pointer):
        raise ReleaseError("Pointer activation verification failed")
    return {"state": "activated", "pointer": pointer}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["stage", "activate", "verify"])
    p.add_argument("--bucket", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--release-key")
    p.add_argument("--approval-sha256")
    p.add_argument("--candidate-s3-key", help="JSON release candidate stored in S3")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm-prod-write", action="store_true")
    a = p.parse_args()
    cfg = load_runtime_config("prod")
    bucket = a.bucket or cfg.bucket
    if a.execute:
        require_prod_confirmation(cfg, a.confirm_prod_write)
    import boto3
    s3 = boto3.client("s3", region_name=a.region or cfg.region)
    run_id = make_run_id("market_v2_release")
    try:
        if a.action in {"stage", "verify"}:
            if not a.candidate_s3_key:
                p.error("--candidate-s3-key required")
            candidate = get(s3, bucket, a.candidate_s3_key)
            if candidate is None:
                raise ReleaseError("Candidate S3 JSON does not exist")
            release = json.loads(candidate)
            result = stage(s3, bucket=bucket, release=release,
                           execute=a.execute and a.action == "stage")
        else:
            if not a.release_key or not a.approval_sha256:
                p.error("--release-key and --approval-sha256 required")
            result = activate(s3, bucket=bucket, release_key=a.release_key,
                              approved_sha256=a.approval_sha256, execute=a.execute)
    except Exception as exc:
        if a.execute:
            event = build_audit_event(
                cfg=cfg, event_type="market_v2_release", entity_type="market_lake",
                entity_id=a.release_key or a.candidate_s3_key, as_of=None,
                source_script="alpha_edge.market.production_gold_release_v2",
                source_mode=a.action, status="failed", run_id=run_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event)
        raise
    if a.execute:
        event = build_audit_event(
            cfg=cfg, event_type="market_v2_release", entity_type="market_lake",
            entity_id=a.release_key or a.candidate_s3_key, as_of=None,
            source_script="alpha_edge.market.production_gold_release_v2",
            source_mode=a.action, status="success", run_id=run_id,
            output_keys=[result.get("release_key", CURRENT)],
            metadata=result,
        )
        write_audit_event(cfg=cfg, event=event)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
