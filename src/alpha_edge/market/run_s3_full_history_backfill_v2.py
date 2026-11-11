"""Run full-history cohorts with ephemeral disk and immutable S3 persistence.

No persistent outputs/ tree. Acquires Bronze, audits gaps, builds Silver/Gold,
runs independent arithmetic validation, then publishes a *research candidate*
bundle to isolated dev/market/lake_v2/backfills. Never activates a pointer
or promotes production. Skips already verified S3 commits on resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from alpha_edge.market.archive_local_outputs_v2 import inventory, _read_remote_sha
from alpha_edge.market.audit_missing_ohlcv_v2 import generate as audit_missing
from alpha_edge.market.run_cohort_bronze_v2 import load_plan, run as run_cohort
from alpha_edge.market.run_standardization_pilot_v2 import generate as standardize
from alpha_edge.market.independent_acceptance_v2 import generate as acceptance


class S3BackfillError(RuntimeError):
    pass


def _sha_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _get_remote(s3, bucket: str, key: str) -> bytes | None:
    from botocore.exceptions import ClientError

    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def _commit_key(prefix):
    return f"{prefix}/_committed.json"


def verify_commit(s3, bucket: str, prefix: str, plan_hash: str) -> dict | None:
    marker = _get_remote(s3, bucket, _commit_key(prefix))
    if marker is None:
        return None
    committed = json.loads(marker)
    if committed.get("schema") != "alpha_edge_s3_backfill_commit_v2_1":
        raise S3BackfillError("Remote commit schema mismatch")
    if committed.get("plan_manifest_sha256") != plan_hash:
        raise S3BackfillError("S3 cohort from different preflight plan")
    if committed.get("prefix") != prefix:
        raise S3BackfillError("S3 cohort prefix mismatch")
    for entry in committed.get("files", []):
        key = f"{prefix}/files/{entry['path']}"
        if _read_remote_sha(s3, bucket, key) != entry["sha256"]:
            raise S3BackfillError(f"Remote checksum mismatch: {key}")
    return committed


def upload_verified(s3, *, bucket: str, prefix: str, local_dir: Path, plan_hash: str, end: str) -> dict:
    if verify_commit(s3, bucket, prefix, plan_hash) is not None:
        raise S3BackfillError("Cohort already committed")
    entries = inventory(local_dir)
    if not entries:
        raise S3BackfillError("Cannot publish empty cohort")
    for item in entries:
        key = f"{prefix}/files/{item['path']}"
        current = _read_remote_sha(s3, bucket, key)
        if current is None:
            s3.upload_file(str(local_dir / item["path"]), bucket, key)
        elif current != item["sha256"]:
            raise S3BackfillError(f"Remote object collision: {key}")
        if _read_remote_sha(s3, bucket, key) != item["sha256"]:
            raise S3BackfillError(f"Remote verification failed: {key}")
    marker = {
        "schema": "alpha_edge_s3_backfill_commit_v2_1",
        "plan_manifest_sha256": plan_hash,
        "prefix": prefix,
        "end_exclusive": end,
        "files": entries,
        "storage_state": "verified_research_candidate",
        "production_eligible": False,
    }
    encoded = (json.dumps(marker, sort_keys=True, indent=2) + "\n").encode()
    s3.put_object(Bucket=bucket, Key=_commit_key(prefix), Body=encoded, ContentType="application/json", IfNoneMatch="*")
    if _read_remote_sha(s3, bucket, _commit_key(prefix)) != _sha_bytes(encoded):
        raise S3BackfillError("S3 commit marker verification failed")
    return marker


def restore_plan_from_archive(s3, *, bucket: str, archive_id: str, relative_dir: str, destination: Path) -> Path:
    """Retrieve an already verified archive's plan without persistent local files."""
    from alpha_edge.market.archive_local_outputs_v2 import archive_prefix
    from pathlib import PurePosixPath

    prefix = archive_prefix(archive_id)
    manifest_blob = _get_remote(s3, bucket, f"{prefix}/_verified_inventory.json")
    if manifest_blob is None:
        raise S3BackfillError("No finalized S3 outputs archive")
    manifest = json.loads(manifest_blob)
    if manifest.get("schema") != "alpha_edge_local_output_archive_v2_1":
        raise S3BackfillError("Unexpected archive schema")
    parts = PurePosixPath(relative_dir).parts
    if (
        not parts
        or any(p in ("", ".", "..") for p in parts)
        or relative_dir.startswith(("/", "\\"))
        or "\\" in relative_dir
        or ":" in relative_dir
    ):
        raise S3BackfillError("Unsafe S3 plan relative directory")
    begin = "/".join(parts) + "/"
    entries = [x for x in manifest["files"] if x["path"].startswith(begin)]
    if not entries or not any(x["path"] == begin + "manifest.json" for x in entries):
        raise S3BackfillError("Archive lacks the requested preflight plan")
    destination.mkdir(parents=True, exist_ok=False)
    for entry in entries:
        name = entry["path"][len(begin) :]
        segments = PurePosixPath(name).parts
        if not segments or any(p in ("", ".", "..") for p in segments):
            raise S3BackfillError("Unsafe archived plan entry")
        blob = _get_remote(s3, bucket, f"{prefix}/files/{entry['path']}")
        if blob is None or _sha_bytes(blob) != entry["sha256"]:
            raise S3BackfillError(f"Archived plan checksum mismatch: {name}")
        out = destination.joinpath(*segments)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
    # load_plan validates every manifest-pinned cohort CSV and FX config.
    load_plan(destination)
    return destination


def run(
    *, plan_dir: Path, bucket: str, end: str, execute: bool, start_cohort: int, max_cohorts: int, s3, retries: int = 2
) -> dict:
    if not bucket or not end or not 0 <= retries <= 3:
        raise S3BackfillError("Invalid S3 full-history parameters")
    plan, cohorts, _ = load_plan(plan_dir)
    if start_cohort < 1 or max_cohorts < 1 or start_cohort > len(cohorts):
        raise S3BackfillError("Invalid cohort selection")
    last = min(len(cohorts), start_cohort + max_cohorts - 1)
    indices = range(start_cohort, last + 1)
    plan_hash = _sha_bytes((plan_dir / "manifest.json").read_bytes())
    base = f"dev/market/lake_v2/backfills/{plan_hash[:24]}/end-{end}"
    if not execute:
        return {
            "status": "dry_run",
            "plan_assets": plan["assets_included"],
            "selected_cohorts": len(indices),
            "first": start_cohort,
            "last": last,
            "s3_target": f"s3://{bucket}/{base}",
            "s3_writes": 0,
        }
    summaries = []
    for index in indices:
        prefix = f"{base}/cohort-{index:04d}"
        complete = verify_commit(s3, bucket, prefix, plan_hash)
        if complete:
            summaries.append({"cohort": index, "state": "s3_verified_existing", "files": len(complete["files"])})
            continue
        # Local bytes only live under OS temp directory and are removed at
        # context exit; uploads must be verified before we mark success.
        with tempfile.TemporaryDirectory(prefix="alpha_edge_s3_backfill_") as temp:
            temp_root = Path(temp)
            run_dir = temp_root / "execution"
            result = run_cohort(
                plan_dir=plan_dir,
                output=run_dir,
                batch_prefix=f"s3bf-{index:04d}",
                start=None,
                end=end,
                full_history=True,
                execute=True,
                start_cohort=index,
                max_cohorts=1,
                retries=retries,
            )
            if result["state"] != "complete":
                summaries.append({"cohort": index, "state": "acquisition_failed"})
                continue
            bronze = run_dir / "bronze_batches" / f"s3bf-{index:04d}-{index:04d}"
            if not (bronze / "manifest.json").is_file():
                raise S3BackfillError("No complete immutable Bronze batch")
            audit_dir = temp_root / "audit"
            audit_missing(bronze, plan_dir / f"cohort_{index:04d}.csv", plan_dir / "fx_config.csv", audit_dir)
            candidate_dir = temp_root / "candidate"
            standardize(
                bronze,
                output=candidate_dir,
                asset_registry_csv=plan_dir / f"cohort_{index:04d}.csv",
                fx_registry_csv=plan_dir / "fx_config.csv",
            )
            acceptance(
                candidate_dir=candidate_dir,
                bronze_dir=bronze,
                output=temp_root / "acceptance",
                asset_registry_csv=plan_dir / f"cohort_{index:04d}.csv",
                fx_registry_csv=plan_dir / "fx_config.csv",
            )
            committed = upload_verified(
                s3, bucket=bucket, prefix=prefix, local_dir=temp_root, plan_hash=plan_hash, end=end
            )
            summaries.append({"cohort": index, "state": "s3_verified", "files": len(committed["files"])})
    return {
        "schema": "alpha_edge_s3_backfill_execution_v2_1",
        "plan_assets": plan["assets_included"],
        "attempted_cohorts": len(indices),
        "s3_verified_cohorts": sum(row["state"].startswith("s3_verified") for row in summaries),
        "failed_cohorts": sum(row["state"] == "acquisition_failed" for row in summaries),
        "results": summaries,
        "production_eligible": False,
        "persistent_repo_outputs": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan-dir", type=Path)
    source.add_argument("--plan-archive-id")
    p.add_argument("--plan-relative-path", default="full_universe_preflight_v2/first_review")
    p.add_argument("--bucket", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--start-cohort", type=int, default=1)
    p.add_argument("--max-cohorts", type=int, default=1)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    import boto3

    s3 = boto3.client("s3", region_name=args.region)
    if args.plan_archive_id:
        with tempfile.TemporaryDirectory(prefix="alpha_edge_preflight_") as temp:
            plan = restore_plan_from_archive(
                s3,
                bucket=args.bucket,
                archive_id=args.plan_archive_id,
                relative_dir=args.plan_relative_path,
                destination=Path(temp) / "plan",
            )
            result = run(
                plan_dir=plan,
                bucket=args.bucket,
                end=args.end,
                execute=args.execute,
                start_cohort=args.start_cohort,
                max_cohorts=args.max_cohorts,
                retries=args.retries,
                s3=s3,
            )
    else:
        result = run(
            plan_dir=args.plan_dir,
            bucket=args.bucket,
            end=args.end,
            execute=args.execute,
            start_cohort=args.start_cohort,
            max_cohorts=args.max_cohorts,
            retries=args.retries,
            s3=s3,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
