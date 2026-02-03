from __future__ import annotations

import argparse
import datetime as dt
import json
import uuid
from typing import Optional
from alpha_edge.core.data_loader import s3_get_json

import boto3
import pandas as pd

BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"

PORTFOLIO_RUNS_TABLE = "portfolio_search/runs"
CHOICES_TABLE = "portfolio_choices"
TARGETS_TABLE = "targets"   # NEW


def s3_client(region: str = REGION):
    return boto3.client("s3", region_name=region)

def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def s3_list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            keys.append(it["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def s3_latest_dt(s3, *, bucket: str, table_prefix: str) -> Optional[str]:
    prefix = table_prefix.rstrip("/") + "/dt="
    keys = s3_list_keys(s3, bucket=bucket, prefix=prefix)
    dts = set()
    for k in keys:
        parts = k.split("/")
        for p in parts:
            if p.startswith("dt=") and len(p) == len("dt=YYYY-MM-DD"):
                dts.add(p.replace("dt=", ""))
    if not dts:
        return None
    return sorted(dts)[-1]


def s3_latest_run_key(s3, *, bucket: str, runs_table_prefix: str) -> str:
    latest_dt = s3_latest_dt(s3, bucket=bucket, table_prefix=runs_table_prefix)
    if not latest_dt:
        raise RuntimeError(f"No dt partitions found under s3://{bucket}/{runs_table_prefix}/dt=YYYY-MM-DD/")

    dt_prefix = f"{runs_table_prefix.rstrip('/')}/dt={latest_dt}/"
    keys = s3_list_keys(s3, bucket=bucket, prefix=dt_prefix)
    json_keys = [k for k in keys if k.endswith(".json")]
    if not json_keys:
        raise RuntimeError(f"No JSON files found under s3://{bucket}/{dt_prefix}")

    run_keys = [k for k in json_keys if k.split("/")[-1].startswith("run_")]
    pick_from = run_keys if run_keys else json_keys
    return sorted(pick_from)[-1]


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def dt_key(table: str, dt_str: str, filename: str) -> str:
    return engine_key(table, f"dt={dt_str}", filename)


def apply_latest_portfolio_choice(*, dry_run: bool = False) -> None:
    s3 = s3_client(REGION)

    runs_prefix = engine_key(PORTFOLIO_RUNS_TABLE)
    run_key = s3_latest_run_key(s3, bucket=BUCKET, runs_table_prefix=runs_prefix)
    run_payload = s3_get_json(s3, bucket=BUCKET, key=run_key)

    run_id = run_payload.get("run_id")
    run_as_of = run_payload.get("as_of")
    outputs = (run_payload.get("outputs") or {})
    disc = (outputs.get("discrete_allocation") or {})
    shares = disc.get("shares") or {}

    if not run_id or not run_as_of or not isinstance(shares, dict) or not shares:
        raise RuntimeError(
            "Latest portfolio search run payload missing required fields. "
            "Need run_id, as_of, outputs.discrete_allocation.shares"
        )

    today = pd.Timestamp(dt.date.today())
    dt_str = today.strftime("%Y-%m-%d")
    choice_id = f"{dt_str}-{uuid.uuid4().hex[:8]}"

    # Targets payload: ONLY intent (no prices)
    targets_payload = {
        "as_of": dt_str,
        "choice_id": choice_id,
        "source": {
            "portfolio_search_run_key": run_key,
            "run_id": run_id,
            "run_as_of": run_as_of,
            "variant": "outputs.discrete_allocation.shares",
        },
        "targets": {
            "shares": {str(t): float(q) for t, q in shares.items() if float(q) > 0.0},
        },
    }

    choice_payload = {
        "choice_id": choice_id,
        "as_of": dt_str,
        "picked_from": {
            "portfolio_search_run_key": run_key,
            "run_id": run_id,
            "run_as_of": run_as_of,
            "variant": "outputs.discrete_allocation.shares",
        },
        "targets_ref": {
            "table": TARGETS_TABLE,
            "dt": dt_str,
            "filename": f"targets_{choice_id}.json",
        },
        "note": "This record stores target holdings intent only. Execution & prices are recorded in trades.",
    }

    targets_key = dt_key(TARGETS_TABLE, dt_str, f"targets_{choice_id}.json")
    targets_latest_key = engine_key(TARGETS_TABLE, "latest.json")
    choice_key = dt_key(CHOICES_TABLE, dt_str, f"choice_{choice_id}.json")
    choice_latest_key = engine_key(CHOICES_TABLE, "latest.json")

    print("\n=== APPLY PORTFOLIO CHOICE (TARGETS ONLY) ===")
    print(f"Run key:    s3://{BUCKET}/{run_key}")
    print(f"Run id:     {run_id}")
    print(f"Apply dt:   {dt_str}")
    print(f"Choice id:  {choice_id}")
    print(f"Targets n:  {len(targets_payload['targets']['shares'])}")
    print("")

    if dry_run:
        print("[DRY RUN] Would write:")
        print(f"  s3://{BUCKET}/{targets_key}")
        print(f"  s3://{BUCKET}/{targets_latest_key}")
        print(f"  s3://{BUCKET}/{choice_key}")
        print(f"  s3://{BUCKET}/{choice_latest_key}")
        return

    s3_put_json(s3, bucket=BUCKET, key=targets_key, payload=targets_payload)
    s3_put_json(s3, bucket=BUCKET, key=targets_latest_key, payload=targets_payload)
    s3_put_json(s3, bucket=BUCKET, key=choice_key, payload=choice_payload)
    s3_put_json(s3, bucket=BUCKET, key=choice_latest_key, payload=choice_payload)

    print("[OK] Wrote targets:")
    print(f"  s3://{BUCKET}/{targets_key}")
    print(f"  s3://{BUCKET}/{targets_latest_key}")
    print("[OK] Wrote choice record:")
    print(f"  s3://{BUCKET}/{choice_key}")
    print(f"  s3://{BUCKET}/{choice_latest_key}")
    print("")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    apply_latest_portfolio_choice(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
