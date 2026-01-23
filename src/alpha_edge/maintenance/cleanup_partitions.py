from __future__ import annotations

import argparse
import datetime as dt
from typing import List, Tuple

import boto3


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    Convert 's3://bucket/prefix/...' into (bucket, prefix).
    Prefix returned has no leading slash and no trailing slash (normalized).
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {uri}")
    rest = uri[len("s3://") :]
    if "/" not in rest:
        return rest, ""
    bucket, prefix = rest.split("/", 1)
    prefix = prefix.strip("/")
    return bucket, prefix


def list_objects(bucket: str, prefix: str) -> List[dict]:
    s3 = boto3.client("s3")
    out: List[dict] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        out.extend(resp.get("Contents", []) or [])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


def delete_keys(bucket: str, keys: List[str], dry_run: bool) -> int:
    keys = sorted(set(keys))
    if not keys:
        return 0

    if dry_run:
        for k in keys[:80]:
            print(f"[DRY_RUN] would delete s3://{bucket}/{k}")
        if len(keys) > 80:
            print(f"[DRY_RUN] ... and {len(keys) - 80} more")
        return len(keys)

    s3 = boto3.client("s3")
    deleted = 0
    for i in range(0, len(keys), 1000):
        chunk = keys[i : i + 1000]
        resp = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
        )
        deleted += len(chunk)
        errors = resp.get("Errors") or []
        if errors:
            print("[WARN] delete errors (showing up to 20):")
            for e in errors[:20]:
                print(f"  - {e}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")
    return deleted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="alpha-edge-algo")
    ap.add_argument("--date", required=True, help="UTC date of the bad run: YYYY-MM-DD")
    ap.add_argument("--year", type=int, default=None, help="Optional: restrict to keys containing 'year=YYYY/'")
    ap.add_argument("--dry-run", action="store_true", help="List what would be deleted (recommended first)")
    ap.add_argument(
        "--confirm-delete",
        action="store_true",
        help="Required for actual delete (safety)",
    )

    args = ap.parse_args()

    target_date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
    year = args.year

    # These match your MarketStore exactly:
    #   ohlcv_prefix = s3://{bucket}/market/ohlcv_usd/v1
    #   returns_prefix = s3://{bucket}/market/returns_usd/v1
    ohlcv_uri = f"s3://{args.bucket}/market/ohlcv_usd/v1"
    returns_uri = f"s3://{args.bucket}/market/returns_usd/v1"

    bucket1, p1 = parse_s3_uri(ohlcv_uri)
    bucket2, p2 = parse_s3_uri(returns_uri)
    if bucket1 != bucket2:
        raise SystemExit("Unexpected: prefixes point to different buckets")

    bucket = bucket1
    prefixes = [p1 + "/", p2 + "/"]  # ensure trailing slash for listing

    keys_to_delete: List[str] = []

    for pref in prefixes:
        objs = list_objects(bucket=bucket, prefix=pref)
        matched = 0

        for o in objs:
            key = o["Key"]
            lm: dt.datetime = o["LastModified"]  # tz-aware
            lm_date = lm.astimezone(dt.timezone.utc).date()

            if lm_date != target_date:
                continue
            if year is not None and f"year={year}/" not in key:
                continue
            # only delete parquet parts (extra safety)
            if not key.endswith(".parquet"):
                continue

            keys_to_delete.append(key)
            matched += 1

        print(f"[SCAN] s3://{bucket}/{pref} objects={len(objs)} matched={matched}")

    keys_to_delete = sorted(set(keys_to_delete))
    print(f"\n[TOTAL] matched parquet objects to delete: {len(keys_to_delete)} for utc_date={target_date} year={year}")

    if args.dry_run:
        delete_keys(bucket=bucket, keys=keys_to_delete, dry_run=True)
        print("\n[DRY_RUN DONE]")
        return

    if not args.confirm_delete:
        raise SystemExit("Refusing to delete without --confirm-delete. Run with --dry-run first.")

    n = delete_keys(bucket=bucket, keys=keys_to_delete, dry_run=False)
    print(f"\n[DELETE DONE] deleted={n}")


if __name__ == "__main__":
    main()
