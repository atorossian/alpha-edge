"""Archive local runtime outputs to an isolated S3 namespace, verify, then optionally prune.

This is a *migration utility*, not production lake publication. Defaults to
inventory-only. Upload is explicit. Prune requires a separate invocation after
full remote byte-stream verification; never deletes unverified or new files.
Does not touch Git-tracked files. Never follows symlinks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

SAFE_ARCHIVE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,90}$")
BLOCKED = {".git", ".venv", "__pycache__", ".env", ".aws", ".ssh"}
RESTRICTED_EXTENSIONS = {".pem", ".key", ".p12", ".pfx"}


class ArchiveError(RuntimeError):
    pass


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(piece)
    return sha.hexdigest()


def inventory(root: Path) -> list[dict]:
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise ArchiveError(f"Expected real output directory: {root}")
    files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        if parent.is_symlink():
            raise ArchiveError(f"Symbolic-link directory: {parent}")
        if any(name in BLOCKED or (parent / name).is_symlink() for name in dirs):
            raise ArchiveError(f"Sensitive or symlink directory under {parent}")
        for name in names:
            path = parent / name
            if path.is_symlink() or not path.is_file():
                raise ArchiveError(f"Non-regular file: {path}")
            if name in BLOCKED or path.suffix.lower() in RESTRICTED_EXTENSIONS:
                raise ArchiveError(f"Sensitive-looking file requires separate review: {path}")
            relative = path.relative_to(root).as_posix()
            if any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts):
                raise ArchiveError("Invalid path")
            files.append({"path": relative, "size": path.stat().st_size,
                          "sha256": digest(path)})
    return sorted(files, key=lambda x: x["path"])


def archive_prefix(archive_id: str, *, root: str = "dev/market") -> str:
    if not SAFE_ARCHIVE.fullmatch(archive_id):
        raise ArchiveError("Invalid archive ID")
    if root != "dev/market":
        raise ArchiveError("Archives only permitted in isolated dev/market")
    return f"{root}/lake_v2/archives/{archive_id}"


def _read_remote_sha(s3, bucket: str, key: str) -> str | None:
    from botocore.exceptions import ClientError
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    sha = hashlib.sha256()
    with response["Body"] as body:
        for chunk in body.iter_chunks(chunk_size=8 * 1024 * 1024):
            if chunk:
                sha.update(chunk)
    return sha.hexdigest()


def _verify_existing(s3, bucket, key, entry):
    remote = _read_remote_sha(s3, bucket, key)
    if remote != entry["sha256"]:
        raise ArchiveError(f"Remote bytes missing or mismatched: {key}")


def transfer(*, root: Path, bucket: str, archive_id: str,
             s3, upload: bool = False, prune_verified: bool = False) -> dict:
    if upload and prune_verified:
        raise ArchiveError("Upload and prune must be separate invocations")
    if not bucket:
        raise ArchiveError("Explicit S3 bucket required")
    prefix = archive_prefix(archive_id)
    entries = inventory(root)
    if not entries:
        raise ArchiveError("Empty output directory")
    manifest = {
        "schema": "alpha_edge_local_output_archive_v2_1",
        "archive_id": archive_id,
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(row["size"] for row in entries),
        "purpose": "nonproduction_historical_output_archive",
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    manifest_key = f"{prefix}/_verified_inventory.json"
    if not upload and not prune_verified:
        return {"status": "dry_run", "bucket": bucket, "prefix": prefix,
                "files": len(entries), "bytes": manifest["total_bytes"],
                "s3_writes": 0, "local_deletions": 0}
    if upload:
        if _read_remote_sha(s3, bucket, manifest_key) is not None:
            raise ArchiveError("Archive already finalized; use a new archive ID")
        for entry in entries:
            path = root / entry["path"]
            key = f"{prefix}/files/{entry['path']}"
            existing = _read_remote_sha(s3, bucket, key)
            if existing is None:
                s3.upload_file(str(path), bucket, key)
            elif existing != entry["sha256"]:
                raise ArchiveError(f"Archive key collision: {key}")
            _verify_existing(s3, bucket, key, entry)
            # Fail if file changed during upload, even when remote had old bytes.
            if digest(path) != entry["sha256"] or path.stat().st_size != entry["size"]:
                raise ArchiveError(f"Local file modified during archive: {path}")
        s3.put_object(Bucket=bucket, Key=manifest_key, Body=manifest_bytes,
                      ContentType="application/json", IfNoneMatch="*")
        if _read_remote_sha(s3, bucket, manifest_key) != hashlib.sha256(manifest_bytes).hexdigest():
            raise ArchiveError("Archive inventory verification failed")
        return {"status": "verified_in_s3", "bucket": bucket,
                "prefix": prefix, "files": len(entries), "bytes": manifest["total_bytes"],
                "local_deletions": 0}
    # Prune: fetch the exact verified manifest, compare complete inventory
    # then independently re-read every object before deleting *any* file.
    remote_inventory = s3.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
    if hashlib.sha256(remote_inventory).hexdigest() != hashlib.sha256(manifest_bytes).hexdigest():
        raise ArchiveError("Local inventory has changed since verified archive")
    for entry in entries:
        _verify_existing(s3, bucket, f"{prefix}/files/{entry['path']}", entry)
    # Git-tracked files must never be removed by this utility, even if they
    # happen to be under outputs/. Check git in the CLI before invoking.
    for entry in entries:
        file = root / entry["path"]
        if file.is_symlink() or digest(file) != entry["sha256"]:
            raise ArchiveError(f"File changed before pruning: {file}")
    for entry in entries:
        (root / entry["path"]).unlink()
    for directory, _, _ in os.walk(root, topdown=False):
        directory = Path(directory)
        if directory != root:
            try:
                directory.rmdir()
            except OSError:
                pass
    return {"status": "local_files_pruned_after_s3_verification",
            "files_removed": len(entries), "bucket": bucket, "prefix": prefix}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-root", type=Path, default=Path("outputs"))
    p.add_argument("--bucket", required=True)
    p.add_argument("--archive-id", required=True)
    p.add_argument("--region", default="eu-west-1")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--upload", action="store_true")
    group.add_argument("--prune-verified", action="store_true")
    a = p.parse_args()
    # Only the dedicated root outputs/ is supported for deletion. Never
    # delete any directories with source code or Git-tracked data.
    root = a.local_root.resolve()
    if root.name != "outputs" or root == Path.cwd().resolve():
        p.error("Archive root must be a directory named outputs")
    if a.prune_verified:
        import subprocess
        proc = subprocess.run(
            ["git", "ls-files", "--", str(root)], capture_output=True, text=True,
            check=False, cwd=Path.cwd(),
        )
        if proc.returncode != 0 or proc.stdout.strip():
            p.error("Refusing prune: Git-tracked output files or Git check failed")
    import boto3
    client = boto3.client("s3", region_name=a.region)
    print(json.dumps(transfer(root=root, bucket=a.bucket,
                              archive_id=a.archive_id, s3=client,
                              upload=a.upload,
                              prune_verified=a.prune_verified), indent=2))


if __name__ == "__main__":
    main()
