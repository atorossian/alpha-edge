# dedupe_ohlcv_partitions_keep_one.py
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import boto3

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_MARKET_ROOT = "market"


@dataclass
class Obj:
    key: str
    size: int
    last_modified: object


# Supports:
#   market/ohlcv_usd/v1/asset_id=XYZ/year=2024/file.parquet
#   market/ohlcv_usd/v1/ticker=SPY/year=2024/file.parquet
#   dev/market/ohlcv_usd/v1/asset_id=XYZ/year=2024/file.parquet
PART_RE_TEMPLATE = r"^{root}/ohlcv_usd/v1/(asset_id|ticker)=([^/]+)/year=(\d{{4}})/.*\.parquet$"


# ----------------------------
# Runtime helpers
# ----------------------------
def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET)).strip()


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION)).strip()


def cfg_market_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "market_root", DEFAULT_MARKET_ROOT)).strip("/")


def cfg_env(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "env", "dev")).strip()


# ----------------------------
# S3 helpers
# ----------------------------
def list_objects(*, bucket: str, prefix: str, region: str) -> List[Obj]:
    s3 = boto3.client("s3", region_name=region)

    out: List[Obj] = []
    token = None

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)

        for it in resp.get("Contents", []) or []:
            k = it.get("Key")
            if isinstance(k, str) and k.endswith(".parquet"):
                out.append(
                    Obj(
                        key=k,
                        size=int(it.get("Size", 0)),
                        last_modified=it.get("LastModified"),
                    )
                )

        if not resp.get("IsTruncated"):
            break

        token = resp.get("NextContinuationToken")

    return out


def choose_keep(objs: List[Obj], *, keep_strategy: str) -> Obj:
    strategy = str(keep_strategy).strip().lower()

    if strategy == "largest":
        return max(objs, key=lambda o: (int(o.size), str(o.last_modified), o.key))

    if strategy == "newest":
        return max(objs, key=lambda o: (o.last_modified, int(o.size), o.key))

    raise ValueError(f"Unsupported keep strategy: {keep_strategy!r}")


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


def _compile_partition_regex(*, market_root: str) -> re.Pattern:
    root = re.escape(str(market_root).strip("/"))
    return re.compile(PART_RE_TEMPLATE.format(root=root))


def _print_safety_banner(
    *,
    cfg: RuntimeConfig,
    bucket: str,
    region: str,
    market_root: str,
    root_prefix: str,
    keep_strategy: str,
    dry_run: bool,
    year: int | None,
    partition_kind: str,
) -> None:
    print("\n" + "=" * 88)
    print("DANGER: OHLCV PARTITION DEDUPE")
    print("=" * 88)
    print("This script deletes duplicate parquet files inside each OHLCV partition.")
    print("Default mode is dry-run. Deletion requires --execute.")
    print("")
    print(f"env:            {cfg_env(cfg)}")
    print(f"bucket:         {bucket}")
    print(f"region:         {region}")
    print(f"market_root:    {market_root}")
    print(f"root_prefix:    s3://{bucket}/{root_prefix}")
    print(f"partition_kind: {partition_kind}")
    print(f"year_filter:    {year}")
    print(f"keep_strategy:  {keep_strategy}")
    print(f"dry_run:        {dry_run}")
    print("=" * 88 + "\n")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Dedupe OHLCV parquet partitions by keeping one file per asset/year or ticker/year."
    )

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument("--bucket", default=None, help="Override runtime bucket. Usually avoid this.")
    ap.add_argument("--region", default=None, help="Override runtime region. Usually avoid this.")
    ap.add_argument("--market-root", default=None, help="Override runtime market root. Usually avoid this.")

    ap.add_argument(
        "--keep-strategy",
        choices=["newest", "largest"],
        default="newest",
        help="Which file to keep inside each duplicate partition.",
    )

    ap.add_argument(
        "--partition-kind",
        choices=["asset_id", "ticker", "both"],
        default="asset_id",
        help="Partition layout to dedupe. Current lake should use asset_id.",
    )

    ap.add_argument("--year", type=int, default=None, help="Optional: only process year=YYYY partitions.")

    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete duplicate files. Default is dry-run.",
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Deprecated compatibility flag. Default behavior is already dry-run unless --execute is passed.",
    )

    ap.add_argument("--reason", default=None, help="Required for real maintenance writes/deletes. Stored in audit trail.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)

    bucket = str(args.bucket or cfg_bucket(cfg)).strip()
    region = str(args.region or cfg_region(cfg)).strip()
    market_root = str(args.market_root or cfg_market_root(cfg)).strip("/")
    keep_strategy = str(args.keep_strategy).strip().lower()
    partition_kind = str(args.partition_kind).strip().lower()
    year_filter = args.year

    dry_run = not bool(args.execute)

    if bool(args.execute) and bool(args.dry_run):
        raise SystemExit("Conflicting flags: use either --execute or --dry-run, not both.")

    if bool(args.execute):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    root_prefix = f"{market_root}/ohlcv_usd/v1/"
    if root_prefix in {"ohlcv_usd/v1/", "market/", "dev/market/", "staging/market/"}:
        raise SystemExit(f"Refusing unsafe/broad root prefix: {root_prefix!r}")

    part_re = _compile_partition_regex(market_root=market_root)

    _print_safety_banner(
        cfg=cfg,
        bucket=bucket,
        region=region,
        market_root=market_root,
        root_prefix=root_prefix,
        keep_strategy=keep_strategy,
        dry_run=dry_run,
        year=year_filter,
        partition_kind=partition_kind,
    )

    objs = list_objects(bucket=bucket, prefix=root_prefix, region=region)

    groups: Dict[Tuple[str, str, str], List[Obj]] = defaultdict(list)
    skipped_nonmatching = 0
    skipped_kind = 0
    skipped_year = 0

    for o in objs:
        m = part_re.match(o.key)
        if not m:
            skipped_nonmatching += 1
            continue

        kind = m.group(1)
        ident = m.group(2)
        year = m.group(3)

        if partition_kind != "both" and kind != partition_kind:
            skipped_kind += 1
            continue

        if year_filter is not None and int(year) != int(year_filter):
            skipped_year += 1
            continue

        groups[(kind, ident, year)].append(o)

    print(
        f"[scan] parquet_objects={len(objs)} "
        f"partitions_matched={len(groups)} "
        f"skipped_nonmatching={skipped_nonmatching} "
        f"skipped_kind={skipped_kind} "
        f"skipped_year={skipped_year}"
    )

    total_partitions_with_dupes = 0
    total_delete = 0
    all_delete_keys: list[str] = []

    for (kind, ident, year) in sorted(groups.keys(), key=lambda x: (x[0], x[1], x[2])):
        items = groups[(kind, ident, year)]
        if len(items) <= 1:
            continue

        keep = choose_keep(items, keep_strategy=keep_strategy)
        to_delete = [o.key for o in items if o.key != keep.key]

        total_partitions_with_dupes += 1
        total_delete += len(to_delete)
        all_delete_keys.extend(to_delete)

        print(f"\n[{kind}={ident} year={year}] files={len(items)}")
        print(f"  KEEP   : {keep.key} (size={keep.size} last_modified={keep.last_modified})")
        for k in to_delete[:20]:
            print(f"  DELETE : {k}")
        if len(to_delete) > 20:
            print(f"  ... and {len(to_delete) - 20} more")

    print("\n[summary]")
    print(f"  partitions_with_dupes={total_partitions_with_dupes}")
    print(f"  files_to_delete={total_delete}")

    if not all_delete_keys:
        print("\n[DONE] No duplicate parquet files found.")
        return

    deleted = delete_keys(bucket=bucket, keys=all_delete_keys, region=region, dry_run=dry_run)

    if dry_run:
        print(f"\n[DRY RUN DONE] matched_for_delete={deleted}. No deletions executed.")
        print("To delete these files, rerun with --execute.")
    else:
        print(f"\n[DELETE DONE] deleted={deleted}")



# ----------------------------
# Audit/logging entrypoint wrapper
# ----------------------------
def main_with_audit() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = not bool(getattr(args, "execute", False))
    if bool(getattr(args, "execute", False)) and not getattr(args, "reason", None):
        raise SystemExit("--reason is required when using --execute for OHLCV dedupe.")

    captured_delete_keys: list[str] = []
    original_delete_keys = globals()["delete_keys"]

    def _audited_delete_keys(*, bucket: str, keys: List[str], region: str, dry_run: bool) -> int:
        captured_delete_keys.extend([str(k) for k in keys])
        return original_delete_keys(bucket=bucket, keys=keys, region=region, dry_run=dry_run)

    with capture_script_run(cfg=cfg, script_name="dedupe_ohlcv_partitions_keep_one.py", input_args=vars(args), dry_run=is_dry_run) as run_id:
        globals()["delete_keys"] = _audited_delete_keys
        try:
            main()
            planned = sorted(set(captured_delete_keys))
            event = build_audit_event(
                cfg=cfg, run_id=run_id, event_type="delete", entity_type="ohlcv_partition_dedupe",
                entity_id=None, as_of=None, source_script="dedupe_ohlcv_partitions_keep_one.py",
                source_mode="dedupe_ohlcv_partitions", status=("dry_run" if is_dry_run else "success"),
                reason=getattr(args, "reason", None), input_args=vars(args), deleted_keys=([] if is_dry_run else planned),
                metadata={"tier": "tier_2", "payload_policy": "large_dataset_metadata_only", "planned_deleted_keys": planned,
                          "planned_deleted_count": len(planned), "partition_kind": getattr(args, "partition_kind", None),
                          "keep_strategy": getattr(args, "keep_strategy", None), "year_filter": getattr(args, "year", None),
                          "execute": bool(getattr(args, "execute", False))},
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
        except Exception as exc:
            planned = sorted(set(captured_delete_keys))
            event = build_audit_event(
                cfg=cfg, run_id=run_id, event_type="delete", entity_type="ohlcv_partition_dedupe",
                entity_id=None, as_of=None, source_script="dedupe_ohlcv_partitions_keep_one.py",
                source_mode="dedupe_ohlcv_partitions", status="failed", reason=getattr(args, "reason", None),
                input_args=vars(args), metadata={"tier": "tier_2", "payload_policy": "large_dataset_metadata_only", "planned_deleted_keys": planned,
                                               "planned_deleted_count": len(planned), "execute": bool(getattr(args, "execute", False))},
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise
        finally:
            globals()["delete_keys"] = original_delete_keys


if __name__ == "__main__":
    main_with_audit()