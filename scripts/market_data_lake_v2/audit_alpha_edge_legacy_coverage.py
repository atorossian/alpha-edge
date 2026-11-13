#!/usr/bin/env python3
"""Read-only comparison of legacy/current Alpha Edge OHLCV manifest date coverage.

Outputs locally (no S3 writes):
    legacy_coverage_summary.json
    legacy_coverage_pairs.jsonl

This proves only date-set containment in the published manifests. It does not
verify price values, corporate actions, instrument equivalence, or authorize
S3 deletions/merges.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import json
from pathlib import Path

import boto3
from botocore.config import Config

RECONCILIATION_ID = "3b37aa4db63b6f54a30b7b4c4147ef55754e1b31447c16b9b16a396ab130905e"
REGISTRY_KEY = f"dev/market/lake_v2/reconciliation/run-{RECONCILIATION_ID}/asset_registry.jsonl"
PROVIDER_KEY = "market/state/v1/provider_symbol_by_asset_id.json"
MANIFEST_ROOT = "market/manifests/v1/ohlcv_usd"


def _read_json(s3, bucket: str, key: str):
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def _manifest_dates(s3, bucket: str, asset_id: str, end: str) -> dict:
    """Only manifest-listed dates, never counts of Parquet objects."""
    prefix = f"{MANIFEST_ROOT}/asset_id={asset_id}/"
    dates: set[str] = set()
    manifest_count = 0
    keys: list[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []) if obj["Key"].endswith("/manifest.json"))
    for key in sorted(keys):
        data = _read_json(s3, bucket, key)
        if data.get("asset_id") != asset_id:
            raise ValueError(f"Manifest asset-ID mismatch: {key}")
        year = key.rsplit("year=", 1)[-1].split("/", 1)[0]
        declared = data.get("dates")
        if not isinstance(declared, list):
            raise ValueError(f"Manifest dates missing or not a list: {key}")
        for date in declared:
            if not isinstance(date, str) or len(date) != 10 or date[:4] != year:
                raise ValueError(f"Invalid date or year mismatch: {key}: {date!r}")
            dt.date.fromisoformat(date)
            if date < end:
                dates.add(date)
        manifest_count += 1
    return {"asset_id": asset_id, "dates": dates, "manifest_count": manifest_count}


def _compare(pair: dict, old: dict, current: dict) -> dict:
    a, b = old["dates"], current["dates"]
    if not a or not b:
        status = "missing_manifest_dates"
        preferred = None
    elif a < b:
        status = "current_strict_superset"
        preferred = pair["current_asset_id"]
    elif b < a:
        status = "legacy_strict_superset"
        preferred = pair["legacy_asset_id"]
    elif a == b:
        status = "equal_manifest_coverage"
        preferred = pair["current_asset_id"]  # active identity, not a price decision
    else:
        status = "non_nested_coverage_review"
        preferred = None
    return {
        **pair,
        "coverage_status": status,
        "provisional_preferred_asset_id": preferred,
        "legacy_date_count": len(a),
        "current_date_count": len(b),
        "overlap_date_count": len(a & b),
        "legacy_unique_date_count": len(a - b),
        "current_unique_date_count": len(b - a),
        "legacy_first": min(a) if a else None,
        "legacy_last": max(a) if a else None,
        "current_first": min(b) if b else None,
        "current_last": max(b) if b else None,
        "legacy_manifest_count": old["manifest_count"],
        "current_manifest_count": current["manifest_count"],
        "price_observations_validated": False,
        "overlap_price_values_compared": False,
        "same_instrument_approved": False,
        "deletion_authorized": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default="alpha-edge-algo")
    ap.add_argument("--region", default="eu-west-1")
    ap.add_argument("--end", default="2026-11-10", help="Exclusive ISO cutoff")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Optional small pilot; 0=all 879 pairs")
    ap.add_argument("--output-dir", default="legacy_coverage_audit")
    args = ap.parse_args()
    dt.date.fromisoformat(args.end)
    if args.workers < 1 or args.workers > 32:
        ap.error("--workers must be from 1 to 32")
    s3 = boto3.client("s3", region_name=args.region, config=Config(
        max_pool_connections=args.workers + 4,
        retries={"max_attempts": 8, "mode": "standard"},
    ))
    registry = [json.loads(line) for line in s3.get_object(
        Bucket=args.bucket, Key=REGISTRY_KEY)["Body"].iter_lines() if line]
    provider = _read_json(s3, args.bucket, PROVIDER_KEY)
    active_by_symbol: dict[str, list[dict]] = collections.defaultdict(list)
    for item in registry:
        if item.get("current_eligible_config") is True and item.get("current_yahoo_ticker"):
            active_by_symbol[item["current_yahoo_ticker"]].append(item)
    legacy = [item for item in registry if item.get("issues") == "legacy_only_identity_unresolved"]
    pairs, unmatched = [], []
    for row in legacy:
        symbol = provider.get(row["asset_id"])
        current = active_by_symbol.get(symbol, [])
        if len(current) != 1 or current[0]["asset_id"] == row["asset_id"]:
            unmatched.append({"legacy_asset_id": row["asset_id"], "legacy_symbol": symbol,
                              "current_match_count": len(current)})
            continue
        pairs.append({"legacy_asset_id": row["asset_id"],
                      "current_asset_id": current[0]["asset_id"],
                      "provider_symbol": symbol})
    pairs.sort(key=lambda row: (row["provider_symbol"], row["legacy_asset_id"]))
    selected = pairs[:args.limit] if args.limit else pairs
    ids = sorted({x[k] for x in selected for k in ("legacy_asset_id", "current_asset_id")})
    print(f"Identified {len(pairs)} pairs and {len(unmatched)} unmatched legacy IDs; auditing {len(selected)} pairs / {len(ids)} distinct histories")
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_manifest_dates, s3, args.bucket, asset_id, args.end): asset_id for asset_id in ids}
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            asset_id = futures[future]
            try:
                results[asset_id] = future.result()
            except Exception as exc:
                failures[asset_id] = f"{type(exc).__name__}: {exc}"
            if n % 100 == 0 or n == len(ids):
                print(f"Read {n}/{len(ids)} histories; errors={len(failures)}")
    rows = []
    for pair in selected:
        old_id, current_id = pair["legacy_asset_id"], pair["current_asset_id"]
        if old_id in failures or current_id in failures:
            rows.append({**pair, "coverage_status": "read_error", "provisional_preferred_asset_id": None,
                         "errors": [failures[k] for k in (old_id, current_id) if k in failures],
                         "same_instrument_approved": False, "deletion_authorized": False})
        else:
            rows.append(_compare(pair, results[old_id], results[current_id]))
    by_current = collections.Counter(x["current_asset_id"] for x in selected)
    for row in rows:
        row["multiple_legacy_ids_for_current"] = by_current[row["current_asset_id"]] > 1
    summary = {
        "reconciliation_run_id": RECONCILIATION_ID,
        "bucket": args.bucket, "end_exclusive": args.end,
        "legacy_only_ids_in_registry": len(legacy), "candidate_pairs": len(pairs),
        "unmatched_legacy_ids": len(unmatched), "pairs_audited": len(rows),
        "coverage_status_counts": dict(collections.Counter(x["coverage_status"] for x in rows)),
        "read_errors": failures,
        "unmatched": unmatched,
        "scope": "S3 OHLCV manifest date coverage only",
        "limitations": ["Manifest dates do not prove non-null, valid or correct prices",
                        "Equal dates do not prove equal OHLCV values or corporate-action adjustments",
                        "No asset-ID merges, S3 deletions or production promotions performed"],
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "legacy_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "legacy_coverage_pairs.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"counts": summary["coverage_status_counts"], "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()