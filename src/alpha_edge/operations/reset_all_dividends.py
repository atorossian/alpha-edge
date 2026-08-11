from __future__ import annotations

import argparse
import datetime as dt
import json

import boto3

from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation


def list_dividend_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            k = it["Key"]
            if k.endswith(".json") and "/dividend_" in k:
                keys.append(k)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return sorted(keys)


def main() -> None:
    ap = argparse.ArgumentParser(description="Back up and delete every dividend JSON object in the active engine root.")
    ap.add_argument("--env", default="prod", choices=["dev", "staging", "prod"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm-prod-write", action="store_true")
    ap.add_argument("--backup-tag", default=None)
    args = ap.parse_args()

    cfg = load_runtime_config(args.env)
    s3 = boto3.client("s3", region_name=cfg.region)

    if not args.dry_run:
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    prefix = f"{cfg.engine_root.strip('/')}/dividends/dt="
    keys = list_dividend_keys(s3, bucket=cfg.bucket, prefix=prefix)

    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    tag = args.backup_tag or f"full_reset_{ts}"
    backup_root = f"{cfg.engine_root.strip('/')}/dividends_audit/{tag}"

    print("=== DIVIDEND FULL RESET ===")
    print("env:", cfg.env)
    print("bucket:", cfg.bucket)
    print("region:", cfg.region)
    print("engine_root:", cfg.engine_root)
    print("prefix:", prefix)
    print("dividend_objects_found:", len(keys))
    print("backup_root:", backup_root)
    print("dry_run:", bool(args.dry_run))
    print("")

    manifest = {
        "created_at_utc": ts,
        "env": cfg.env,
        "bucket": cfg.bucket,
        "engine_root": cfg.engine_root,
        "source_prefix": prefix,
        "backup_root": backup_root,
        "objects": [],
    }

    for k in keys:
        backup_key = f"{backup_root}/{k}"
        manifest["objects"].append({"source_key": k, "backup_key": backup_key})

        print("[DELETE]", f"s3://{cfg.bucket}/{k}")
        print("         backup:", f"s3://{cfg.bucket}/{backup_key}")

        if args.dry_run:
            continue

        s3.copy_object(
            Bucket=cfg.bucket,
            CopySource={"Bucket": cfg.bucket, "Key": k},
            Key=backup_key,
            MetadataDirective="COPY",
        )
        s3.delete_object(Bucket=cfg.bucket, Key=k)

    manifest_key = f"{backup_root}/manifest.json"
    if not args.dry_run:
        s3.put_object(
            Bucket=cfg.bucket,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    print("")
    print("objects_processed:", len(keys))
    if args.dry_run:
        print("[DRY RUN] no objects were modified.")
    else:
        print("[OK] backup manifest:", f"s3://{cfg.bucket}/{manifest_key}")


if __name__ == "__main__":
    main()
