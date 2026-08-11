from __future__ import annotations

import argparse
import datetime as dt
from typing import List, Tuple

import boto3

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_MARKET_ROOT = "market"


# ----------------------------
# Runtime helpers
# ----------------------------
def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET)).strip()


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION)).strip()


def cfg_env(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "env", "dev")).strip()


def cfg_market_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "market_root", DEFAULT_MARKET_ROOT)).strip("/")


# ----------------------------
# S3 helpers
# ----------------------------
def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    Convert 's3://bucket/prefix/...' into (bucket, prefix).
    Prefix returned has no leading slash and no trailing slash.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 uri: {uri}")

    rest = uri[len("s3://") :]
    if "/" not in rest:
        return rest, ""

    bucket, prefix = rest.split("/", 1)
    return bucket, prefix.strip("/")


def list_objects(*, bucket: str, prefix: str, region: str) -> List[dict]:
    s3 = boto3.client("s3", region_name=region)

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


def delete_keys(*, bucket: str, keys: List[str], region: str, dry_run: bool) -> int:
    keys = sorted(set(keys))
    if not keys:
        return 0

    if dry_run:
        for k in keys[:80]:
            print(f"[DRY RUN] would delete s3://{bucket}/{k}")
        if len(keys) > 80:
            print(f"[DRY RUN] ... and {len(keys) - 80} more")
        return len(keys)

    s3 = boto3.client("s3", region_name=region)

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
            print("[WARN] delete errors, showing up to 20:")
            for e in errors[:20]:
                print(f"  - {e}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")

    return deleted


def _validate_safe_prefixes(*, market_root: str, prefixes: list[str]) -> None:
    """
    Prevent accidentally scanning/deleting the whole bucket or the wrong environment root.
    """
    market_root = str(market_root).strip("/")
    allowed = {
        f"{market_root}/ohlcv_usd/v1/",
        f"{market_root}/returns_usd/v1/",
    }

    for p in prefixes:
        pp = str(p).strip("/")
        pp_slash = pp + "/"

        if pp_slash not in allowed:
            raise SystemExit(
                "Refusing unsafe cleanup prefix.\n"
                f"Got: {p!r}\n"
                f"Allowed: {sorted(allowed)}"
            )

        if pp in {"", "market", "dev/market", "staging/market"}:
            raise SystemExit(f"Refusing broad cleanup prefix: {p!r}")


def _print_safety_banner(
    *,
    cfg: RuntimeConfig,
    bucket: str,
    region: str,
    market_root: str,
    target_date: dt.date,
    year: int | None,
    dry_run: bool,
    prefixes: list[str],
) -> None:
    print("\n" + "=" * 88)
    print("DANGER: MARKET PARTITION CLEANUP")
    print("=" * 88)
    print("This script selects parquet objects by S3 LastModified UTC date and deletes them.")
    print("It is intended only for bad market ingestion runs.")
    print("")
    print(f"env:         {cfg_env(cfg)}")
    print(f"bucket:      {bucket}")
    print(f"region:      {region}")
    print(f"market_root: {market_root}")
    print(f"utc_date:    {target_date}")
    print(f"year_filter: {year}")
    print(f"dry_run:     {dry_run}")
    print("")
    print("prefixes:")
    for p in prefixes:
        print(f"  s3://{bucket}/{p}")
    print("=" * 88 + "\n")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Delete bad-run market parquet objects by S3 LastModified UTC date."
    )

    ap.add_argument("--bucket", default=None, help="Override runtime bucket. Usually avoid this.")
    ap.add_argument("--region", default=None, help="Override runtime region. Usually avoid this.")
    ap.add_argument("--market-root", default=None, help="Override runtime market_root. Usually avoid this.")

    ap.add_argument("--date", required=True, help="UTC date of the bad run: YYYY-MM-DD")
    ap.add_argument("--year", type=int, default=None, help="Optional: restrict to keys containing 'year=YYYY/'")

    # Safer than --dry-run because default is dry-run.
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matching objects. Default is dry-run.",
    )

    # Kept for backwards compatibility, but no longer required.
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Deprecated compatibility flag. Default behavior is already dry-run unless --execute is passed.",
    )

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument("--reason", default=None, help="Required for real maintenance writes/deletes. Stored in audit trail.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)

    bucket = str(args.bucket or cfg_bucket(cfg)).strip()
    region = str(args.region or cfg_region(cfg)).strip()
    market_root = str(args.market_root or cfg_market_root(cfg)).strip("/")

    target_date = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
    year = args.year

    # Default is dry-run. Only --execute deletes.
    dry_run = not bool(args.execute)

    if bool(args.execute) and bool(args.dry_run):
        raise SystemExit("Conflicting flags: use either --execute or --dry-run, not both.")

    if bool(args.execute):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    ohlcv_uri = f"s3://{bucket}/{market_root}/ohlcv_usd/v1"
    returns_uri = f"s3://{bucket}/{market_root}/returns_usd/v1"

    bucket1, p1 = parse_s3_uri(ohlcv_uri)
    bucket2, p2 = parse_s3_uri(returns_uri)

    if bucket1 != bucket2:
        raise SystemExit("Unexpected: prefixes point to different buckets.")

    prefixes = [p1 + "/", p2 + "/"]
    _validate_safe_prefixes(market_root=market_root, prefixes=prefixes)

    _print_safety_banner(
        cfg=cfg,
        bucket=bucket,
        region=region,
        market_root=market_root,
        target_date=target_date,
        year=year,
        dry_run=dry_run,
        prefixes=prefixes,
    )

    keys_to_delete: List[str] = []

    for prefix in prefixes:
        objs = list_objects(bucket=bucket, prefix=prefix, region=region)
        matched = 0

        for obj in objs:
            key = obj.get("Key")
            if not isinstance(key, str):
                continue

            lm = obj.get("LastModified")
            if not isinstance(lm, dt.datetime):
                continue

            lm_date = lm.astimezone(dt.timezone.utc).date()

            if lm_date != target_date:
                continue
            if year is not None and f"year={year}/" not in key:
                continue
            if not key.endswith(".parquet"):
                continue

            keys_to_delete.append(key)
            matched += 1

        print(f"[SCAN] s3://{bucket}/{prefix} objects={len(objs)} matched={matched}")

    keys_to_delete = sorted(set(keys_to_delete))

    print("")
    print(f"[TOTAL] matched parquet objects: {len(keys_to_delete)}")

    if dry_run:
        delete_keys(bucket=bucket, keys=keys_to_delete, region=region, dry_run=True)
        print("\n[DRY RUN DONE] No objects deleted.")
        print("To delete these objects, rerun with --execute.")
        return

    if not keys_to_delete:
        print("\n[DELETE SKIPPED] No matching objects.")
        return

    print("\n[EXECUTE] Deleting matched objects now.")
    deleted = delete_keys(bucket=bucket, keys=keys_to_delete, region=region, dry_run=False)
    print(f"\n[DELETE DONE] deleted={deleted}")



# ----------------------------
# Audit/logging entrypoint wrapper
# ----------------------------
def main_with_audit() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = not bool(getattr(args, "execute", False))
    if bool(getattr(args, "execute", False)) and not getattr(args, "reason", None):
        raise SystemExit("--reason is required when using --execute for maintenance cleanup.")

    captured_delete_keys: list[str] = []
    original_delete_keys = globals()["delete_keys"]

    def _audited_delete_keys(*, bucket: str, keys: List[str], region: str, dry_run: bool) -> int:
        captured_delete_keys.extend([str(k) for k in keys])
        return original_delete_keys(bucket=bucket, keys=keys, region=region, dry_run=dry_run)

    with capture_script_run(cfg=cfg, script_name="cleanup_partitions.py", input_args=vars(args), dry_run=is_dry_run) as run_id:
        globals()["delete_keys"] = _audited_delete_keys
        try:
            main()
            planned = sorted(set(captured_delete_keys))
            event = build_audit_event(
                cfg=cfg, run_id=run_id, event_type="delete", entity_type="market_partitions_cleanup",
                entity_id=None, as_of=str(getattr(args, "date", None) or ""),
                source_script="cleanup_partitions.py", source_mode="cleanup_partitions",
                status=("dry_run" if is_dry_run else "success"), reason=getattr(args, "reason", None),
                input_args=vars(args), deleted_keys=([] if is_dry_run else planned),
                metadata={"tier": "tier_2", "payload_policy": "large_dataset_metadata_only", "planned_deleted_keys": planned,
                          "planned_deleted_count": len(planned), "target_date": str(getattr(args, "date", "")),
                          "year_filter": getattr(args, "year", None), "execute": bool(getattr(args, "execute", False))},
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
        except Exception as exc:
            planned = sorted(set(captured_delete_keys))
            event = build_audit_event(
                cfg=cfg, run_id=run_id, event_type="delete", entity_type="market_partitions_cleanup",
                entity_id=None, as_of=str(getattr(args, "date", None) or ""),
                source_script="cleanup_partitions.py", source_mode="cleanup_partitions", status="failed",
                reason=getattr(args, "reason", None), input_args=vars(args),
                metadata={"tier": "tier_2", "payload_policy": "large_dataset_metadata_only", "planned_deleted_keys": planned,
                          "planned_deleted_count": len(planned), "execute": bool(getattr(args, "execute", False))},
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise
        finally:
            globals()["delete_keys"] = original_delete_keys


if __name__ == "__main__":
    main_with_audit()