"""Versioned, isolated S3 publication for Market Data Lake v2 candidates.

This module never writes legacy market/ohlcv_usd/v1, returns_usd/v1,
snapshots, active AEA-87 manifests, or production pointers.

Staging is immutable, SHA-256 verified, with a commit marker written last.
Promotion is an *explicit*, conditional update to a distinct lake_v2 pointer
and is refused for any candidate not fully accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from botocore.exceptions import ClientError


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]*$")
PROTECTED_ROOTS = {"", "market", "prod/market"}
REQUIRED_FILES = {
    "silver.parquet",
    "gold.parquet",
    "bronze_validated.parquet",
    "fx_normalized.parquet",
    "fx_evidence.parquet",
    "session_issues.parquet",
    "asset_quality_summary.csv",
}


class PublicationError(RuntimeError):
    """Candidate publication or promotion gate failure."""


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _is_missing(exc: ClientError) -> bool:
    return exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound", "NoSuchVersion")


def _root(value: str) -> str:
    root = value.strip().strip("/")
    if root in PROTECTED_ROOTS or not root.startswith("dev/market"):
        raise PublicationError("Only isolated dev/market namespace is permitted")
    if root != "dev/market" and not root.startswith("dev/market/"):
        raise PublicationError("Invalid namespace")
    if any(segment in ("", ".", "..") for segment in root.split("/")):
        raise PublicationError("Invalid namespace path")
    return root


def _run_id(value: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise PublicationError("Unsafe candidate run ID")
    return value


def _canonical_manifest_name(name: str) -> str:
    """Canonical portable artifact key; tolerate existing Windows candidates.

    Legacy manifests may use Windows separators. Normalize only after
    rejecting traversal, absolute paths, drive letters, empty segments and
    ambiguous keys. New manifests are always emitted with POSIX separators.
    """
    if not isinstance(name, str):
        raise PublicationError(f"Unsafe artifact filename: {name}")
    portable = name.replace("\\", "/")
    segments = portable.split("/")
    if (
        not SAFE_FILE.fullmatch(portable)
        or portable.startswith("/")
        or any(segment in ("", ".", "..") for segment in segments)
        or ":" in portable
    ):
        raise PublicationError(f"Unsafe artifact filename: {name}")
    return PurePosixPath(portable).as_posix()


def _verified_artifacts(local: Path, manifest: dict) -> dict[str, tuple[str, Path]]:
    """Map canonical S3 keys to (original manifest key, verified local file)."""
    declared = manifest.get("files_sha256")
    if not isinstance(declared, dict) or not REQUIRED_FILES.issubset(declared):
        raise PublicationError("Candidate manifest lacks required file checksums")
    if manifest.get("schema") != "market_data_lake_v2_silver_gold_candidate_1":
        raise PublicationError("Candidate schema mismatch")
    if manifest.get("publication_state") != "candidate_only":
        raise PublicationError("Expected unpublished local candidate")
    if manifest.get("legacy_production_unchanged") is not True:
        raise PublicationError("Production isolation flag missing")
    if not manifest.get("bronze_manifest_sha256"):
        raise PublicationError("Missing Bronze provenance")
    if manifest.get("source_rows", 0) <= 0:
        raise PublicationError("Empty source history")
    if manifest.get("candidate_input_rows", manifest["source_rows"]) != manifest.get("silver_rows") or manifest.get(
        "silver_rows"
    ) != manifest.get("gold_rows"):
        raise PublicationError("Bronze/Silver/Gold row count mismatch")
    assets = {}
    for original, digest in declared.items():
        canonical = _canonical_manifest_name(original)
        if canonical in assets:
            raise PublicationError(f"Ambiguous duplicate artifact key: {canonical}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PublicationError(f"Invalid artifact digest: {original}")
        path = local.joinpath(*canonical.split("/"))
        # Reject symlink parents as well as final files.
        current = local
        for part in canonical.split("/"):
            current = current / part
            if current.is_symlink():
                raise PublicationError(f"Unsafe symlink artifact: {original}")
        if not path.is_file():
            raise PublicationError(f"Missing artifact file: {original}")
        if _sha(path.read_bytes()) != digest:
            raise PublicationError(f"Artifact SHA-256 mismatch: {original}")
        assets[canonical] = (original, path)
    return assets


def _manifest_paths(local: Path, manifest: dict) -> list[str]:
    return sorted(_verified_artifacts(local, manifest))


def validate_local_candidate(local: Path) -> tuple[dict, list[str]]:
    manifest_file = local / "manifest.json"
    if not manifest_file.is_file() or manifest_file.is_symlink():
        raise PublicationError("Candidate manifest missing")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    return manifest, _manifest_paths(local, manifest)


def _read(s3: Any, bucket: str, key: str) -> bytes | None:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if _is_missing(exc):
            return None
        raise


def _put_once(s3: Any, bucket: str, key: str, body: bytes, content_type: str) -> None:
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in (
            "PreconditionFailed",
            "412",
            "ConditionalRequestConflict",
            "409",
        ):
            raise PublicationError(f"Object already exists or concurrent write: {key}") from exc
        raise


def _candidate_prefix(root: str, run_id: str) -> str:
    return f"{_root(root)}/lake_v2/candidates/{_run_id(run_id)}"


def stage_candidate(s3: Any, bucket: str, local: Path, *, run_id: str, root: str = "dev/market") -> dict:
    """Upload once. No overwrite, retry needs a *new* run ID after failure."""
    manifest, paths = validate_local_candidate(local)
    artifacts = _verified_artifacts(local, manifest)
    prefix = _candidate_prefix(root, run_id)
    commit_key = f"{prefix}/_committed.json"
    if _read(s3, bucket, commit_key) is not None:
        raise PublicationError("Candidate already committed")
    # Existence checks occur before writing, to fail early on a reused ID.
    for name in [*paths, "manifest.json"]:
        if _read(s3, bucket, f"{prefix}/{name}") is not None:
            raise PublicationError("Candidate run ID already partially staged; use a new ID")
    for name in paths:
        data = artifacts[name][1].read_bytes()
        _put_once(
            s3,
            bucket,
            f"{prefix}/{name}",
            data,
            "text/csv" if name.endswith(".csv") else "application/octet-stream",
        )
    source_manifest = (local / "manifest.json").read_bytes()
    _put_once(s3, bucket, f"{prefix}/manifest.json", source_manifest, "application/json")

    # Re-read all artifacts from S3 before publishing the final commit marker.
    for name in paths:
        value = _read(s3, bucket, f"{prefix}/{name}")
        if value is None or _sha(value) != manifest["files_sha256"][artifacts[name][0]]:
            raise PublicationError(f"Staged artifact mismatch: {name}")
    persisted_manifest = _read(s3, bucket, f"{prefix}/manifest.json")
    if persisted_manifest != source_manifest:
        raise PublicationError("Staged candidate manifest mismatch")

    receipt = {
        "schema": "lake_v2_staged_candidate_v1",
        "run_id": run_id,
        "bucket": bucket,
        "prefix": prefix,
        "candidate_manifest_sha256": _sha(source_manifest),
        "file_count": len(paths),
        "publication_state": "staged_unpromoted",
        "acceptance": manifest.get("acceptance"),
    }
    _put_once(s3, bucket, commit_key, _json_bytes(receipt), "application/json")
    return receipt


def verify_staged_candidate(s3: Any, bucket: str, *, run_id: str, root: str = "dev/market") -> tuple[dict, dict]:
    prefix = _candidate_prefix(root, run_id)
    marker_bytes = _read(s3, bucket, f"{prefix}/_committed.json")
    if marker_bytes is None:
        raise PublicationError("Candidate not committed")
    receipt = json.loads(marker_bytes)
    if receipt.get("schema") != "lake_v2_staged_candidate_v1":
        raise PublicationError("Invalid staged receipt")
    if receipt.get("run_id") != run_id or receipt.get("bucket") != bucket or receipt.get("prefix") != prefix:
        raise PublicationError("Staged receipt identity mismatch")
    manifest_bytes = _read(s3, bucket, f"{prefix}/manifest.json")
    if manifest_bytes is None or _sha(manifest_bytes) != receipt.get("candidate_manifest_sha256"):
        raise PublicationError("Staged manifest missing or corrupt")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "market_data_lake_v2_silver_gold_candidate_1":
        raise PublicationError("Staged candidate schema mismatch")
    files = manifest.get("files_sha256", {})
    if not REQUIRED_FILES.issubset(files):
        raise PublicationError("Staged manifest missing mandatory tables")
    if len(files) != receipt.get("file_count"):
        raise PublicationError("Staged file count differs from receipt")
    canonical_names = set()
    for name, digest in files.items():
        canonical = _canonical_manifest_name(name)
        if canonical in canonical_names:
            raise PublicationError("Ambiguous stored artifact keys")
        canonical_names.add(canonical)
        data = _read(s3, bucket, f"{prefix}/{canonical}")
        if data is None or _sha(data) != digest:
            raise PublicationError(f"Published data integrity failure: {name}")
    return receipt, manifest


def promote_candidate(
    s3: Any,
    bucket: str,
    *,
    run_id: str,
    root: str = "dev/market",
    approval: str,
) -> dict:
    """Explicitly switch a *dev-only* pointer after full verification.

    The approval string must be the candidate manifest SHA-256, reviewed out
    of band. S3 IfMatch/IfNoneMatch provides optimistic concurrency.
    """
    receipt, manifest = verify_staged_candidate(s3, bucket, run_id=run_id, root=root)
    if approval != receipt["candidate_manifest_sha256"]:
        raise PublicationError("Approval digest does not match staged manifest")
    if (
        manifest.get("acceptance") != "initial_contract_pass"
        or manifest.get("manual_correction_rows", 0) != 0
        or manifest.get("missing_expected_sessions", 0) != 0
        or manifest.get("unexpected_sessions", 0) != 0
        or manifest.get("fx_rejected_rows", 0) != 0
        or any(x.get("quarantined_prices", 0) or x.get("review_returns", 0) for x in manifest.get("assets", []))
    ):
        raise PublicationError("Candidate not eligible for dev pointer promotion")

    pointer_key = f"{_root(root)}/lake_v2/current.json"
    existing = _read(s3, bucket, pointer_key)
    if existing is not None:
        current = json.loads(existing)
        if current.get("run_id") == run_id:
            raise PublicationError("Candidate is already the current dev version")
    else:
        current = None
    pointer = {
        "schema": "lake_v2_dev_pointer_v1",
        "run_id": run_id,
        "candidate_prefix": receipt["prefix"],
        "candidate_manifest_sha256": approval,
        "previous_run_id": current.get("run_id") if current else None,
    }
    body = _json_bytes(pointer)
    params = {"IfNoneMatch": "*"} if existing is None else {}
    if existing is not None:
        # Use the exact ETag of the observed version for the conditional write.
        head = s3.head_object(Bucket=bucket, Key=pointer_key)
        if _read(s3, bucket, pointer_key) != existing:
            raise PublicationError("Pointer changed concurrently")
        params["IfMatch"] = head["ETag"]
    try:
        s3.put_object(
            Bucket=bucket,
            Key=pointer_key,
            Body=body,
            ContentType="application/json",
            **params,
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in (
            "PreconditionFailed",
            "412",
            "ConditionalRequestConflict",
            "409",
        ):
            raise PublicationError("Concurrent dev pointer update") from exc
        raise
    return pointer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify-local", "stage", "verify-staged", "promote-dev"])
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--bucket")
    parser.add_argument("--run-id")
    parser.add_argument("--root", default="dev/market")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--approval", default="")
    args = parser.parse_args()
    _root(args.root)
    if args.action == "verify-local":
        if args.candidate_dir is None:
            parser.error("--candidate-dir required")
        manifest, files = validate_local_candidate(args.candidate_dir)
        output = {"files_verified": len(files), "acceptance": manifest.get("acceptance")}
    else:
        if not args.bucket or not args.run_id:
            parser.error("--bucket and --run-id required")
        import boto3

        s3 = boto3.client("s3", region_name=args.region)
        if args.action == "stage":
            if args.candidate_dir is None:
                parser.error("--candidate-dir required")
            output = stage_candidate(s3, args.bucket, args.candidate_dir, run_id=args.run_id, root=args.root)
        elif args.action == "verify-staged":
            receipt, manifest = verify_staged_candidate(s3, args.bucket, run_id=args.run_id, root=args.root)
            output = {"receipt": receipt, "acceptance": manifest.get("acceptance")}
        else:
            if not args.approval:
                parser.error("--approval requires independently reviewed manifest SHA-256")
            output = promote_candidate(s3, args.bucket, run_id=args.run_id, root=args.root, approval=args.approval)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
