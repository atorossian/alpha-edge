"""Operator-facing S3-only market-data pipeline.

Acquisition and transformations may run together. Production publication
requires an independent approval record, so it is a separate explicit step.
"""

from __future__ import annotations

import argparse
import json

from alpha_edge.market.lakehouse_v3.bootstrap import build_registry
from alpha_edge.market.lakehouse_v3.ingest import acquire
from alpha_edge.market.lakehouse_v3.transform import process
from alpha_edge.market.lakehouse_v3.warehouse import publish


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha Edge S3-only market lakehouse v3")
    parser.add_argument("action", choices=["bootstrap", "build", "publish"])
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--registry-key")
    parser.add_argument("--archive-id", default="old-outputs-20261110")
    parser.add_argument("--asset-ids", default="")
    parser.add_argument("--run-id")
    parser.add_argument("--start")
    parser.add_argument("--end", help="Exclusive end date")
    parser.add_argument("--approval-key")
    parser.add_argument("--athena-workgroup", default="primary")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    import boto3
    s3 = boto3.client("s3", region_name="eu-west-1")
    if args.action == "bootstrap":
        result = build_registry(
            s3, args.bucket, archive_id=args.archive_id,
            selected_asset_ids=[x.strip() for x in args.asset_ids.split(",") if x.strip()] or None,
            execute=args.execute,
        )
    elif args.action == "build":
        if not (args.registry_key and args.run_id and args.start and args.end):
            parser.error("build requires --registry-key --run-id --start --end")
        source = acquire(
            s3=s3, bucket=args.bucket, registry_key=args.registry_key,
            run_id=args.run_id, start=args.start, end=args.end, execute=args.execute,
        )
        candidate = process(
            s3, args.bucket, bronze_run_id=args.run_id, execute=args.execute,
        ) if args.execute else {"state": "not_executed", "reason": "dry_run"}
        result = {"bronze": source, "gold_candidate": candidate}
    else:
        if not (args.run_id and args.approval_key):
            parser.error("publish requires --run-id and --approval-key")
        result = publish(
            s3, boto3.client("athena", region_name="eu-west-1"),
            bucket=args.bucket, run_id=args.run_id,
            approval_key=args.approval_key, workgroup=args.athena_workgroup,
            execute=args.execute,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
