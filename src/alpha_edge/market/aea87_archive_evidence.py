"""Immutable S3 archive of local AEA-87 evidence, including historical outputs.

Archive source is read-only; writes are limited to a separate evidence prefix.
Each run has a manifest of SHA-256 hashes and relative paths. Existing mismatched
objects fail closed instead of being overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from alpha_edge import paths

PREFIX = "market/repairs/aea87/evidence_archive/v1"


def inventory(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("Not a directory: " + str(root))
    files = sorted(p for p in root.rglob("*") if p.is_file())
    result = []
    for p in files:
        relative = p.relative_to(root).as_posix()
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        result.append({"path": relative, "bytes": p.stat().st_size, "sha256": h.hexdigest()})
    return result


def archive(directory, bucket, region, archive_id, workers=8, dry_run=False):
    root = Path(directory).resolve()
    records = inventory(root)
    if not records:
        raise ValueError("No files to archive")
    prefix = f"{PREFIX}/{archive_id}"
    payload = {
        "kind": "AEA87_EVIDENCE_ARCHIVE_V1",
        "archive_id": archive_id,
        "source_directory_name": root.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    # Manifest bytes are deterministic within this invocation.
    manifest = json.dumps(payload, indent=2, sort_keys=True).encode()
    print(
        f"[archive] files={len(records)} size={sum(r['bytes'] for r in records):,} bytes "
        f"destination=s3://{bucket}/{prefix}/ dry_run={dry_run}",
        flush=True,
    )
    if dry_run:
        return payload
    s3 = boto3.client("s3", region_name=region, config=Config(max_pool_connections=max(16, workers * 2)))

    def upload(item):
        key = prefix + "/files/" + item["path"]
        try:
            obj = s3.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
            obj = None
        if obj is not None:
            if obj.get("Metadata", {}).get("sha256") == item["sha256"] and obj["ContentLength"] == item["bytes"]:
                return "reused"
            raise RuntimeError("Existing archive object cannot be verified: " + key)
        data = (root / item["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError("Local file changed during upload: " + item["path"])
        kwargs = {
            "Bucket": bucket,
            "Key": key,
            "Body": data,
            "IfNoneMatch": "*",
            "Metadata": {"sha256": item["sha256"]},
            "ContentType": mimetypes.guess_type(item["path"])[0] or "application/octet-stream",
        }
        try:
            s3.put_object(**kwargs)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "PreconditionFailed":
                raise
            obj = s3.head_object(Bucket=bucket, Key=key)
            if obj.get("Metadata", {}).get("sha256") != item["sha256"] or obj["ContentLength"] != item["bytes"]:
                raise
        return "uploaded"

    uploaded = reused = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(upload, record): record for record in records}
        for n, future in enumerate(as_completed(futures), 1):
            outcome = future.result()
            uploaded += outcome == "uploaded"
            reused += outcome == "reused"
            if n % 25 == 0 or n == len(records):
                print(f"[archive] {n}/{len(records)} uploaded={uploaded} reused={reused}", flush=True)
    manifest_key = prefix + "/manifest.json"
    existing = None
    try:
        existing = s3.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("NoSuchKey", "404"):
            raise
    # Manifest created_at differs across retries; validate file inventory instead.
    if existing is None:
        s3.put_object(Bucket=bucket, Key=manifest_key, Body=manifest, IfNoneMatch="*", ContentType="application/json")
    elif json.loads(existing)["files"] != records:
        raise RuntimeError("Archive ID already used for different evidence: " + archive_id)
    print("[archive] completed: s3://" + bucket + "/" + manifest_key, flush=True)
    return payload


def main():
    p = argparse.ArgumentParser(description="Archive existing and future AEA87 local evidence to S3")
    p.add_argument("--source-dir", default=str(paths.local_outputs_dir() / "aea87_forensics"))
    p.add_argument("--archive-id", required=True, help="Unique ID, e.g. forensic_20261101_01")
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-prod-write", action="store_true")
    a = p.parse_args()
    if not (1 <= a.workers <= 16):
        p.error("--workers must be 1..16")
    if not a.archive_id or not all(c.isalnum() or c in "-_" for c in a.archive_id):
        p.error("--archive-id must contain only letters, digits, dash or underscore")
    if not a.dry_run and not a.confirm_prod_write:
        p.error("S3 archive requires --confirm-prod-write (or --dry-run)")
    archive(a.source_dir, a.bucket, a.region, a.archive_id, a.workers, a.dry_run)


if __name__ == "__main__":
    main()
