"""Record an explicitly approved integer split price-basis candidate registry."""
from __future__ import annotations

import argparse
import json

import boto3

from alpha_edge.market.audit_matching_split_basis_v3 import CANDIDATE, POLICY
from alpha_edge.market.lakehouse_v3.storage import ROOT, digest, json_bytes, put_immutable, verified_bytes

SCOPE = f"{ROOT}/config/split_basis_approval_registries/v1"


def approve(s3, bucket, version, *, expected_count, execute=False):
    if len(version) != 64 or any(c not in "0123456789abcdef" for c in version):
        raise ValueError("Invalid candidate digest")
    key = f"{ROOT}/config/split_basis_candidate_registries/v1/objects/{version}.json"
    source = {"key": key, "sha256": version}
    registry = json.loads(verified_bytes(s3, bucket, source))
    entries = registry["entries"]
    identities = {(x["asset_id"], x["date"]) for x in entries}
    if (registry["schema"] != "alpha_edge_split_basis_candidate_registry_v1"
            or registry["policy"] != POLICY or len(entries) != expected_count
            or len(identities) != expected_count or expected_count <= 0
            or any(x["status"] != CANDIDATE or x["group"] not in {"integer_forward", "integer_reverse"}
                   or x["policy"] != POLICY for x in entries)):
        raise ValueError("Unexpected integer candidate population")
    approval = {"schema": "alpha_edge_split_basis_approval_v1",
                "candidate_registry": source, "approved_count": expected_count,
                "decision": "approved_price_basis_only", "production_eligible": False,
                "quarantine_released": False}
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
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(approve(boto3.client("s3", region_name=args.region), args.bucket,
                             args.candidate_version, expected_count=args.expected_count,
                             execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
