from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone

import boto3

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import (
    load_runtime_config,
    print_runtime_config,
    require_prod_confirmation,
    runtime_engine_key,
    runtime_market_key,
    runtime_warehouse_key,
)


def s3_client(region: str):
    return boto3.client("s3", region_name=region)


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def s3_get_json(s3, *, bucket: str, key: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_delete(s3, *, bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)


def s3_head_bucket(s3, *, bucket: str) -> None:
    s3.head_bucket(Bucket=bucket)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Smoke-test Alpha Edge runtime environment.")
    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")
    ap.add_argument("--write-test", action="store_true", help="Write/read/delete a small test JSON.")
    return ap.parse_args()


def _main_impl(args: argparse.Namespace, cfg) -> dict:
    print_runtime_config(cfg)

    s3 = s3_client(cfg.region)

    print("[check] S3 bucket access")
    s3_head_bucket(s3, bucket=cfg.bucket)
    print(f"[OK] bucket reachable: s3://{cfg.bucket}")

    print("\n[check] resolved canonical paths")
    print(f"engine test path:    s3://{cfg.bucket}/{runtime_engine_key(cfg, '_smoke_tests')}/")
    print(f"market test path:    s3://{cfg.bucket}/{runtime_market_key(cfg, '_smoke_tests')}/")
    print(f"warehouse test path: s3://{cfg.bucket}/{runtime_warehouse_key(cfg, '_smoke_tests')}/")

    if not args.write_test:
        print("\n[OK] read-only smoke test passed.")
        return {"write_test": False}

    require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    test_id = uuid.uuid4().hex[:12]
    key = runtime_engine_key(cfg, "_smoke_tests", f"smoke_{test_id}.json")

    payload = {
        "test_id": test_id,
        "env": cfg.env,
        "bucket": cfg.bucket,
        "engine_root": cfg.engine_root,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    print("\n[check] write/read/delete test object")
    print(f"[write] s3://{cfg.bucket}/{key}")
    s3_put_json(s3, bucket=cfg.bucket, key=key, payload=payload)

    loaded = s3_get_json(s3, bucket=cfg.bucket, key=key)
    if loaded.get("test_id") != test_id:
        raise RuntimeError("Smoke test readback mismatch.")

    print("[read] OK")

    s3_delete(s3, bucket=cfg.bucket, key=key)
    print("[delete] OK")

    print("\n[OK] write smoke test passed.")
    return {"write_test": True, "test_key": key, "test_id": test_id}


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(args.env)

    with capture_script_run(
        cfg=cfg,
        script_name="scripts/smoke_test_env.py",
        input_args=vars(args),
        dry_run=False,
    ) as run_id:
        result = _main_impl(args, cfg)

        if result.get("write_test"):
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="smoke_test",
                entity_type="runtime_environment",
                entity_id=result.get("test_id"),
                as_of=None,
                source_script="scripts/smoke_test_env.py",
                source_mode="write_test",
                status="success",
                input_args=vars(args),
                output_keys=[],
                deleted_keys=[result.get("test_key")],
                metadata={
                    "test_key": result.get("test_key"),
                    "operation": "write_read_delete",
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=False)


if __name__ == "__main__":
    main()