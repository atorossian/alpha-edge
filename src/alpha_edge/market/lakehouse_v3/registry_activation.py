"""Activate only a verified, five-file S3 configuration registry.

This updates configuration authority, not Gold/market-data approval.
By default the command performs read-only validation and prints a preview.
"""

from __future__ import annotations

import argparse
import json

from alpha_edge.market.lakehouse_v3.registry_v3 import activate_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditionally activate an immutable S3 asset registry")
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--registry-key", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--expected-previous-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    import boto3

    result = activate_registry(
        boto3.client("s3", region_name="eu-west-1"),
        args.bucket,
        key=args.registry_key,
        sha256=args.registry_sha256,
        expected_previous_sha256=args.expected_previous_sha256,
        execute=args.execute,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
