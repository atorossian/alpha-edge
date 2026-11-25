"""Record an explicitly approved split price-basis candidate registry."""
from __future__ import annotations

import argparse
import json

import boto3

from alpha_edge.market.audit_matching_split_basis_v3 import CANDIDATE, POLICY, FRACTIONAL, FRACTIONAL_POLICY, SMALL_CANDIDATE, SMALL_POLICY, is_small_fraction
from alpha_edge.market.lakehouse_v3.storage import ROOT, digest, json_bytes, put_immutable, verified_bytes

SCOPE = f"{ROOT}/config/split_basis_approval_registries/v1"


def approve(s3, bucket, version, *, expected_count, execute=False, fractional=False, small_fraction=False):
    if len(version) != 64 or any(c not in "0123456789abcdef" for c in version):
        raise ValueError("Invalid candidate digest")
    key = f"{ROOT}/config/split_basis_candidate_registries/v1/objects/{version}.json"
    source = {"key": key, "sha256": version}
    registry = json.loads(verified_bytes(s3, bucket, source))
    entries = registry["entries"]
    if fractional and small_fraction:
        raise ValueError("Choose one approval mode")
    policy = SMALL_POLICY if small_fraction else FRACTIONAL_POLICY if fractional else POLICY
    status = SMALL_CANDIDATE if small_fraction else FRACTIONAL if fractional else CANDIDATE
    groups = {"simple_fraction"} if fractional or small_fraction else {"integer_forward", "integer_reverse"}
    identities = {(x["asset_id"], x["date"]) for x in entries}
    if (registry["schema"] != "alpha_edge_split_basis_candidate_registry_v1"
            or registry["policy"] != policy or len(entries) != expected_count
            or len(identities) != expected_count or expected_count <= 0
            or any(x["status"] != status or x["group"] not in groups
                   or x["policy"] != policy or (small_fraction and not is_small_fraction(float(x["ratio"]))) for x in entries)):
        raise ValueError("Unexpected candidate population")
    approval = {"schema": "alpha_edge_split_basis_approval_v1",
                "candidate_registry": source, "approved_count": expected_count,
                "decision": "approved_price_basis_only", "production_eligible": False,
                "quarantine_released": False}
    if fractional or small_fraction:
        approval["economic_review_required"] = True
    body = json_bytes(approval)
    output = f"{SCOPE}/objects/{digest(body)}.json"
    if execute:
        put_immutable(s3, bucket, output, body, content_type="application/json", scope=SCOPE)
    return {**approval, "approval_version": digest(body), "key": output,
            "status": "approval_written" if execute else "dry_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--fractional", action="store_true")
    parser.add_argument("--small-fraction", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(approve(boto3.client("s3", region_name=args.region), args.bucket,
                             args.candidate_version, expected_count=args.expected_count,
                             execute=args.execute, fractional=args.fractional, small_fraction=args.small_fraction), indent=2))


if __name__ == "__main__":
    main()
