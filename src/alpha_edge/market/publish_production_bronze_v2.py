"""Copy verified S3 research Bronze to immutable *production Bronze storage*.

This is not Silver/Gold approval, not a production pointer update, and never
writes legacy v1 objects. Each cohort's raw Yahoo files are byte-identical to
the SHA-256 pinned acquisition manifest. The production Bronze receipt is
written last, after re-reading every staged byte. Existing complete cohorts
are verified and safely skipped. Incomplete writes fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

from botocore.exceptions import ClientError

from alpha_edge.market.archive_local_outputs_v2 import _read_remote_sha


class BronzePublicationError(RuntimeError):
    pass


SAFE_PLAN = re.compile(r"^[0-9a-f]{24}$")
REQUIRED_BRONZE = {"source_ohlcv.parquet", "source_fx_quotes.parquet", "manifest.json"}


def _get(s3, bucket, key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def _hash(blob):
    return hashlib.sha256(blob).hexdigest()


def _json(blob, what):
    try:
        return json.loads(blob)
    except (ValueError, TypeError) as exc:
        raise BronzePublicationError(f"Invalid {what} JSON") from exc


def _put_once(s3, bucket, key, blob):
    try:
        s3.put_object(
            Bucket=bucket, Key=key, Body=blob,
            IfNoneMatch="*", ContentType="application/octet-stream",
        )
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in ("PreconditionFailed", "412", "409"):
            raise BronzePublicationError(f"Existing immutable Bronze key: {key}") from error
        raise


def _prefix(plan_prefix, end, cohort, *, environment):
    if not SAFE_PLAN.fullmatch(plan_prefix):
        raise BronzePublicationError("Invalid pinned plan hash prefix")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", end):
        raise BronzePublicationError("Invalid acquisition end date")
    if environment == "source":
        return (f"dev/market/lake_v2/backfills/{plan_prefix}/"
                f"end-{end}/cohort-{cohort:04d}")
    return (f"market/lake_v2/bronze/full_history/{plan_prefix}/"
            f"end-{end}/cohort-{cohort:04d}")


def _verify_production(s3, bucket, destination, plan_prefix, end, cohort):
    receipt_blob = _get(s3, bucket, destination + "/_bronze_committed.json")
    if receipt_blob is None:
        return None
    receipt = _json(receipt_blob, "production Bronze receipt")
    if (receipt.get("schema") != "alpha_edge_production_bronze_commit_v2_1"
            or receipt.get("plan_hash_prefix") != plan_prefix
            or receipt.get("end_exclusive") != end
            or receipt.get("cohort") != cohort
            or receipt.get("destination_prefix") != destination):
        raise BronzePublicationError("Production Bronze receipt identity mismatch")
    for name, sha in receipt.get("sha256", {}).items():
        if name not in REQUIRED_BRONZE:
            raise BronzePublicationError("Unexpected production Bronze artifact")
        if _read_remote_sha(s3, bucket, destination + "/" + name) != sha:
            raise BronzePublicationError(f"Corrupted production Bronze: {name}")
    if set(receipt.get("sha256", {})) != REQUIRED_BRONZE:
        raise BronzePublicationError("Incomplete production Bronze receipt")
    return receipt


def publish_one(s3, *, bucket, plan_prefix, end, cohort, execute):
    source = _prefix(plan_prefix, end, cohort, environment="source")
    destination = _prefix(plan_prefix, end, cohort, environment="production")
    existing = _verify_production(s3, bucket, destination, plan_prefix, end, cohort)
    if existing is not None:
        return {"cohort": cohort, "state": "production_bronze_verified_existing"}
    commit_blob = _get(s3, bucket, source + "/_committed.json")
    if commit_blob is None:
        return {"cohort": cohort, "state": "source_not_committed"}
    commit = _json(commit_blob, "S3 research commit")
    if (commit.get("schema") != "alpha_edge_s3_backfill_commit_v2_1"
            or commit.get("plan_manifest_sha256", "")[:24] != plan_prefix
            or commit.get("end_exclusive") != end
            or commit.get("prefix") != source):
        raise BronzePublicationError(f"Source commit identity mismatch: cohort {cohort}")
    entries = commit.get("files", [])
    root = "execution/bronze_batches/"
    matches = [e for e in entries if e["path"].startswith(root)
               and e["path"].endswith("/manifest.json")]
    if len(matches) != 1:
        raise BronzePublicationError("Exactly one Bronze source manifest required")
    batch_path = matches[0]["path"].rsplit("/", 1)[0]
    candidates = {e["path"].rsplit("/", 1)[-1]: e for e in entries
                  if e["path"].startswith(batch_path + "/")}
    if set(candidates) != REQUIRED_BRONZE:
        raise BronzePublicationError("Incomplete source Bronze artifact inventory")
    original_manifest = _get(s3, bucket, source + "/files/" + batch_path + "/manifest.json")
    if original_manifest is None or _hash(original_manifest) != candidates["manifest.json"]["sha256"]:
        raise BronzePublicationError("Acquisition manifest checksum mismatch")
    acquisition = _json(original_manifest, "acquisition manifest")
    if acquisition.get("schema_version") != "bronze-v2-1":
        raise BronzePublicationError("Source Bronze schema mismatch")
    if acquisition.get("batch_id") != batch_path.rsplit("/", 1)[-1]:
        raise BronzePublicationError("Bronze batch identity mismatch")
    hashes = {"manifest.json": _hash(original_manifest)}
    for name in ("source_ohlcv.parquet", "source_fx_quotes.parquet"):
        expected = acquisition.get("files", {}).get(name, {}).get("sha256")
        if expected != candidates[name]["sha256"]:
            raise BronzePublicationError(f"Source manifest SHA differs: {name}")
        key = source + "/files/" + candidates[name]["path"]
        if _read_remote_sha(s3, bucket, key) != expected:
            raise BronzePublicationError(f"Corrupt source artifact: {name}")
        hashes[name] = expected
    if not execute:
        return {"cohort": cohort, "state": "ready_for_production_bronze",
                "source": source, "destination": destination, "sha256": hashes}
    for name in sorted(REQUIRED_BRONZE):
        dest_key = destination + "/" + name
        src_key = source + "/files/" + candidates[name]["path"]
        current = _read_remote_sha(s3, bucket, dest_key)
        if current is not None:
            if current != hashes[name]:
                raise BronzePublicationError(f"Conflicting destination artifact: {dest_key}")
        else:
            # Read bytes and issue an immutable conditional put. Bronze batches
            # are bounded by the cohort plan; for very large objects use an
            # equivalent conditional multipart implementation.
            blob = _get(s3, bucket, src_key)
            if blob is None or _hash(blob) != hashes[name]:
                raise BronzePublicationError(f"Source changed: {src_key}")
            _put_once(s3, bucket, dest_key, blob)
        if _read_remote_sha(s3, bucket, dest_key) != hashes[name]:
            raise BronzePublicationError(f"Stored object SHA mismatch: {dest_key}")
    receipt = {
        "schema": "alpha_edge_production_bronze_commit_v2_1",
        "cohort": cohort, "plan_hash_prefix": plan_prefix,
        "end_exclusive": end, "destination_prefix": destination,
        "source_commit_sha256": _hash(commit_blob),
        "source_prefix": source, "batch_id": acquisition["batch_id"],
        "sha256": hashes, "data_quality_status": "unreviewed_raw_source",
        "silver_gold_approved": False, "consumer_cutover": False,
    }
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    _put_once(s3, bucket, destination + "/_bronze_committed.json", encoded)
    if _hash(_get(s3, bucket, destination + "/_bronze_committed.json")) != _hash(encoded):
        raise BronzePublicationError("Production Bronze commit confirmation failure")
    return {"cohort": cohort, "state": "production_bronze_committed",
            "source": source, "destination": destination}


def run(s3, *, bucket, plan_prefix, end, start_cohort, max_cohorts, execute):
    if start_cohort < 1 or max_cohorts < 1 or start_cohort + max_cohorts > 10000:
        raise BronzePublicationError("Invalid cohort selection")
    results = []
    for index in range(start_cohort, start_cohort + max_cohorts):
        results.append(publish_one(s3, bucket=bucket, plan_prefix=plan_prefix,
                                   end=end, cohort=index, execute=execute))
    return {
        "schema": "alpha_edge_production_bronze_copy_summary_v2_1",
        "results": results,
        "committed_or_existing": sum(r["state"] in (
            "production_bronze_committed", "production_bronze_verified_existing") for r in results),
        "missing_sources": sum(r["state"] == "source_not_committed" for r in results),
        "silver_gold_published": False,
        "production_pointer_changed": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", required=True)
    p.add_argument("--plan-hash-prefix", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--start-cohort", type=int, default=1)
    p.add_argument("--max-cohorts", type=int, default=1)
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    import boto3
    s3 = boto3.client("s3", region_name=args.region)
    result = run(s3, bucket=args.bucket, plan_prefix=args.plan_hash_prefix,
                 end=args.end, start_cohort=args.start_cohort,
                 max_cohorts=args.max_cohorts, execute=args.execute)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
