"""AEA-87 P0 read-only post-publication trading-session date audit.

Uses immutable publication journals and provider metadata, never downloads
market prices or modifies active market manifests, snapshots, or caches.
A journal proves published dates at cutover; it is not a guarantee that no
subsequent market ingest changed active histories. The report says so.
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone

import boto3

from alpha_edge.market.aea87_publication_transaction import fetch

REPAIRS = "market/repairs/aea87"
ROOT = REPAIRS + "/rebuild_publications/v1"
SOURCE_ROOT = REPAIRS + "/rebuilds/v1/runs"
AUDIT_ROOT = REPAIRS + "/session_date_audit/v1/runs"


def _json(s3, bucket, key):
    raw = fetch(s3, bucket, key)
    if raw is None:
        raise RuntimeError("Missing required immutable evidence: " + key)
    return json.loads(raw)


def _run_source_ids(repair_report):
    return list(repair_report["source_runs"])


def _batch_info(s3, bucket, release, n):
    prefix = f"{ROOT}/{release}_batch_{n:04d}/transaction"
    ops = _json(s3, bucket, prefix + "/operations.json")
    journal = s3.get_paginator("list_objects_v2")
    states = []
    for page in journal.paginate(Bucket=bucket, Prefix=prefix + "/events/"):
        states += [item["Key"] for item in page.get("Contents", [])]
    if not states:
        raise RuntimeError("Publication journal missing for batch " + str(n))
    last = _json(s3, bucket, max(states))
    if last.get("state") != "FINALIZED":
        raise RuntimeError("Batch not finalized: " + str(n))
    return ops


def _source_info(s3, bucket, run_id):
    summary = _json(s3, bucket, f"{SOURCE_ROOT}/{run_id}/summary.json")
    if summary.get("run_id") != run_id or summary.get("status") != "STAGING_ONLY":
        raise RuntimeError("Unverified source summary " + run_id)
    return summary["assets"]


def _dates_from_ops(operations, published):
    per_asset = defaultdict(lambda: defaultdict(list))
    for op in operations:
        aid = op["asset_id"]
        if aid not in published:
            continue
        key = op["manifest_key"]
        if "/ohlcv_usd/" in key:
            table = "ohlcv_usd"
        elif "/returns_usd/" in key:
            table = "returns_usd"
        else:
            raise RuntimeError("Unknown market manifest table " + key)
        after = json.loads(op["after"]) if op.get("after") else {}
        dates = after.get("dates", [])
        if after.get("asset_id") != aid:
            raise RuntimeError("Manifest asset identity mismatch: " + key)
        per_asset[aid][table].extend(dates)
    return per_asset


def audit_asset(aid, tables, source):
    """Pure function: checks session-date validity, not price correctness."""
    p = tables.get("ohlcv_usd", [])
    r = tables.get("returns_usd", [])
    problems = []
    if not p or not r:
        problems.append("MISSING_PRICE_OR_RETURN_DATES")
    if len(set(p)) != len(p):
        problems.append("DUPLICATE_PRICE_DATES")
    if len(set(r)) != len(r):
        problems.append("DUPLICATE_RETURN_DATES")
    if set(p) != set(r):
        problems.append("PRICE_RETURN_DATE_SET_MISMATCH")
    meta = (source or {}).get("provider_metadata") or {}
    asset_class = "CRYPTO" if aid.startswith("CRYPTO:") else (meta.get("quote_type") or "UNKNOWN").upper()
    exchange = meta.get("exchange")
    invalid = []
    weekend = []
    for value in sorted(set(p) | set(r)):
        try:
            d = date.fromisoformat(value)
        except (ValueError, TypeError):
            invalid.append(value)
            continue
        if d.weekday() >= 5 and asset_class not in {"CRYPTO", "CRYPTOCURRENCY"}:
            weekend.append(value)
    if invalid:
        problems.append("INVALID_DATE_FORMAT")
    if weekend:
        problems.append("WEEKEND_DATES_NONCRYPTO")
    if not exchange:
        problems.append("UNKNOWN_EXCHANGE_CALENDAR")
    if asset_class in {"UNKNOWN", ""}:
        problems.append("UNKNOWN_ASSET_CLASS")
    return {
        "asset_id": aid,
        "provider_symbol": (source or {}).get("yahoo_ticker"),
        "exchange": exchange,
        "asset_class": asset_class,
        "price_dates": len(p),
        "return_dates": len(r),
        "weekend_count": len(weekend),
        "weekend_dates_sample": weekend[:12],
        "invalid_dates_sample": invalid[:12],
        "issues": sorted(set(problems)),
        "status": "REVIEW_REQUIRED" if problems else "PASSED_WEEKEND_AND_DATE_CONSISTENCY",
        "calendar_scope": "WEEKEND_ONLY; EXCHANGE_HOLIDAYS_NOT_YET_VERIFIED",
    }


def run(s3, bucket, release, *, workers=8, write_report=True):
    result = _json(s3, bucket, f"{ROOT}/{release}/result.json")
    if result.get("release_id") != release or result.get("status") not in ("COMPLETED", "COMPLETED_WITH_EXCEPTIONS"):
        raise RuntimeError("Release is incomplete")
    published = set(result["published"])
    source_ids = _run_source_ids(result)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_batch_info, s3, bucket, release, i) for i in range(1, len(source_ids) + 1)]
        ops = [op for future in as_completed(jobs) for op in future.result()]
        jobs = [pool.submit(_source_info, s3, bucket, run_id) for run_id in source_ids]
        sources = {}
        for future in as_completed(jobs):
            batch_source = future.result()
            for aid, info in batch_source.items():
                if aid in sources:
                    raise RuntimeError("Duplicate source identity in distinct batches: " + aid)
                sources[aid] = info
    by_asset = _dates_from_ops(ops, published)
    if set(by_asset) != published:
        raise RuntimeError(
            "Published assets are missing from publication evidence: " + str(len(published - set(by_asset)))
        )
    details = [audit_asset(aid, by_asset[aid], sources.get(aid)) for aid in sorted(published)]
    counts = Counter(x["status"] for x in details)
    problems = Counter(issue for x in details for issue in x["issues"])
    report = {
        "schema": "aea87_session_date_audit_v1",
        "audit_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_id": release,
        "status": "COMPLETED",
        "published_assets": len(published),
        "inspected_assets": len(details),
        "status_counts": dict(counts),
        "issue_counts": dict(problems),
        "scope": (
            "IMMUTABLE_PUBLICATION_EVIDENCE; WEEKEND_AND_DATE_SET_AUDIT;"
            " NOT_EXCHANGE_HOLIDAY_VERIFICATION; NOT_CURRENT_LIVE_DRIFT_PROOF"
        ),
        "market_data_modified": False,
        "assets": details,
    }
    if write_report:
        key = (
            AUDIT_ROOT
            + "/"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:8]
            + "/report.json"
        )
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(report, indent=2).encode(),
            ContentType="application/json",
            IfNoneMatch="*",
        )
        report["report_uri"] = f"s3://{bucket}/{key}"
    print(json.dumps({k: v for k, v in report.items() if k != "assets"}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Read-only AEA-87 published session-date audit")
    parser.add_argument("--release-id", default="aea87_rebuild_20261103T140216Z_57fb3b66")
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be 1..16")
    s3 = boto3.client("s3", region_name=args.region)
    run(s3, args.bucket, args.release_id, workers=args.workers)


if __name__ == "__main__":
    main()
