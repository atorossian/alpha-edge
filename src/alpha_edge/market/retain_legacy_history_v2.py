"""Immutable v2 quarantine for historical v1 assets no longer served by Yahoo.

Copies original OHLCV/returns Parquet bytes and source manifests unchanged,
with per-file SHA-256 lineage. Never transforms or approves v1 price data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import boto3

from alpha_edge.market.build_release_inventory_v2 import HISTORICAL_RETAIN
from alpha_edge.market.run_s3_full_history_backfill_v2 import upload_verified, verify_commit

ROOT = "dev/market/lake_v2/historical_retention"


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def collect_one(s3, bucket: str, asset_id: str, destination: Path) -> dict:
    if asset_id not in HISTORICAL_RETAIN:
        raise ValueError("Asset not approved for historical-only retention")
    files = []
    for table in ("ohlcv_usd", "returns_usd"):
        manifest_prefix = f"market/manifests/v1/{table}/asset_id={asset_id}/"
        paginator = s3.get_paginator("list_objects_v2")
        keys = sorted(
            obj["Key"]
            for page in paginator.paginate(Bucket=bucket, Prefix=manifest_prefix)
            for obj in page.get("Contents", [])
            if obj["Key"].endswith("/manifest.json")
        )
        if not keys:
            raise ValueError(f"Missing legacy {table} manifests for {asset_id}")
        for key in keys:
            payload = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            manifest = json.loads(payload)
            if str(manifest.get("asset_id")) != asset_id:
                raise ValueError(f"Manifest identity mismatch: {key}")
            year = int(manifest["year"])
            if f"/year={year}/" not in key:
                raise ValueError(f"Manifest year mismatch: {key}")
            parts = manifest.get("parts", [])
            if not parts or len(set(parts)) != len(parts):
                raise ValueError(f"Missing or duplicate legacy parts: {key}")
            directory = destination / table / f"year={year}"
            directory.mkdir(parents=True, exist_ok=False)
            saved_manifest = directory / "original_manifest.json"
            saved_manifest.write_bytes(payload)
            files.append({
                "source_key": key,
                "source_sha256": _sha(payload),
                "copy_path": saved_manifest.relative_to(destination).as_posix(),
            })
            for index, part in enumerate(parts):
                if not isinstance(part, str) or not part.startswith(
                    f"market/{table}/v1/asset_id={asset_id}/year={year}/"
                ):
                    raise ValueError(f"Unsafe or unexpected legacy part key: {part}")
                raw = s3.get_object(Bucket=bucket, Key=part)["Body"].read()
                target = directory / f"part-{index:04d}.parquet"
                target.write_bytes(raw)
                files.append({
                    "source_key": part,
                    "source_sha256": _sha(raw),
                    "copy_path": target.relative_to(destination).as_posix(),
                })
    return {
        "schema": "aea87_historical_retention_v2_1",
        "asset_id": asset_id,
        "symbol": HISTORICAL_RETAIN[asset_id],
        "preservation": "byte_for_byte_unapproved_legacy_archive",
        "historical_prices_approved": False,
        "production_eligible": False,
        "files": files,
    }


def retain_one(s3, bucket: str, asset_id: str, *, execute: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="aea87_retention_") as temp:
        root = Path(temp)
        contract = collect_one(s3, bucket, asset_id, root)
        identity = hashlib.sha256(
            (json.dumps(contract, sort_keys=True, indent=2) + "\n").encode()
        ).hexdigest()
        prefix = f"{ROOT}/run-{identity}/asset_id={asset_id}"
        result = {
            "asset_id": asset_id,
            "symbol": HISTORICAL_RETAIN[asset_id],
            "source_file_count": len(contract["files"]),
            "candidate_prefix": f"s3://{bucket}/{prefix}",
            "production_eligible": False,
        }
        if not execute:
            return {**result, "status": "dry_run", "s3_writes": 0}
        if verify_commit(s3, bucket, prefix, identity) is not None:
            return {**result, "status": "verified_existing"}
        (root / "retention_manifest.json").write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n"
        )
        upload_verified(
            s3, bucket=bucket, prefix=prefix, local_dir=root,
            plan_hash=identity, end="2026-11-11",
        )
        return {**result, "status": "historical_bytes_retained_in_v2"}


def run(s3, bucket: str, *, execute: bool = False) -> dict:
    results = []
    for aid in sorted(HISTORICAL_RETAIN):
        try:
            results.append(retain_one(s3, bucket, aid, execute=execute))
        except Exception as exc:
            results.append({
                "asset_id": aid, "status": "retention_failed",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            })
    return {
        "status": "historical_retention_completed_with_review"
        if execute else "dry_run",
        "results": results,
        "production_eligible": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(
        boto3.client("s3", region_name=args.region), args.bucket, execute=args.execute,
    ), indent=2))


if __name__ == "__main__":
    main()
