"""Publish a reproducible five-file current-universe snapshot for Bronze.

The five committed CSV inputs are read from the repository and preserved
byte-for-byte. Market data NEVER touches the local filesystem in this step.
The snapshot fingerprint is the exact one required by ingest_bronze_full_history.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from alpha_edge.market.ingest_bronze_full_history import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    SNAPSHOT_ROOT,
    SOURCE_NAMES,
    BronzeIngestError,
    _digest,
    _json_bytes,
    _source_records,
    load_universe,
)

_SHA40 = re.compile(r"^[a-f0-9]{40}$")


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def collect_sources(root: Path, *, require_clean: bool = True) -> tuple[dict[str, bytes], str, str]:
    """Collect only the five config inputs; refuse to snapshot uncommitted edits."""
    root = root.resolve()
    if not (root / ".git").exists():
        raise BronzeIngestError(f"Not a Git checkout: {root}")
    paths = [f"data/universe/{name}" for name in SOURCE_NAMES]
    if require_clean:
        status = _git(root, "status", "--porcelain", "--", *paths)
        if status:
            raise BronzeIngestError(
                "Current universe files contain uncommitted changes; commit them before "
                f"publishing this S3 snapshot:\n{status}"
            )
    git_commit = _git(root, "rev-parse", "HEAD")
    if not _SHA40.fullmatch(git_commit):
        raise BronzeIngestError("Unable to identify a valid pinned source commit")
    blobs = {}
    metadata = {}
    for filename in SOURCE_NAMES:
        path = root / "data" / "universe" / filename
        blob = path.read_bytes()
        _source_records(blob, filename)
        blobs[filename] = blob
        metadata[filename] = {"bytes": len(blob), "sha256": _digest(blob)}
    fingerprint = _digest(_json_bytes(metadata))
    return blobs, git_commit, fingerprint


def publish_snapshot(s3: Any, bucket: str, *, blobs: dict[str, bytes], commit: str,
                     fingerprint: str, execute: bool = False) -> dict:
    if set(blobs) != set(SOURCE_NAMES):
        raise BronzeIngestError("Expected precisely the five current universe configurations")
    if not _SHA40.fullmatch(commit):
        raise BronzeIngestError("A single pinned Git SHA is required")
    expected = {
        filename: {"bytes": len(blobs[filename]), "sha256": _digest(blobs[filename])}
        for filename in SOURCE_NAMES
    }
    if fingerprint != _digest(_json_bytes(expected)):
        raise BronzeIngestError("Source fingerprint does not match input file contents")
    prefix = f"{SNAPSHOT_ROOT}/{fingerprint}"
    result = {
        "snapshot_fingerprint": fingerprint,
        "source_git_commit": commit,
        "bucket": bucket,
        "snapshot_s3_uri": f"s3://{bucket}/{prefix}/",
        "file_count": len(blobs),
        "file_hashes": expected,
        "executed": execute,
    }
    if not execute:
        return result
    for filename in SOURCE_NAMES:
        key = f"{prefix}/{filename}"
        payload = blobs[filename]
        try:
            s3.put_object(
                Bucket=bucket, Key=key, Body=payload, ContentType="text/csv",
                IfNoneMatch="*",
                Metadata={"sha256": _digest(payload), "source-git-commit": commit},
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {
                "PreconditionFailed", "412", "ConditionalRequestConflict", "409",
            }:
                raise
            previous = s3.get_object(Bucket=bucket, Key=key)
            if (
                previous["Body"].read() != payload
                or previous.get("Metadata", {}).get("source-git-commit") != commit
                or previous.get("Metadata", {}).get("sha256") != _digest(payload)
            ):
                raise BronzeIngestError(f"Existing snapshot entry is incompatible: {key}") from exc
    included, meta = load_universe(s3, bucket, fingerprint)
    result.update(
        {
            "universe_total": meta["universe_total"],
            "included": meta["included"],
            "excluded": meta["excluded"],
            "readback_verified": True,
            "warning_count": len(meta["configuration_warnings"]),
            "selected_asset_count": len(included),
        }
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Snapshot the five COMMITTED universe config files to S3")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--execute", action="store_true", help="Publish immutable S3 objects, then verify all five")
    options = ap.parse_args()
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))
    blobs, commit, fingerprint = collect_sources(repo)
    s3 = boto3.client("s3", region_name=options.region) if options.execute else None
    report = publish_snapshot(
        s3, options.bucket, blobs=blobs, commit=commit,
        fingerprint=fingerprint, execute=options.execute,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
