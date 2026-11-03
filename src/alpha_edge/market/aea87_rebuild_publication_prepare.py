"""AEA-87 rebuilt-history publication preparation.

Creates immutable candidate parts, manifest backups and a drift-checked operations
plan in S3. NEVER switches active manifests, caches or snapshots.

The separate activation workflow must require verified maintenance isolation,
schema compatibility, and a working end-to-end downstream refresh/rollback.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import uuid
from datetime import datetime, timezone

import boto3
import pandas as pd
import numpy as np
from botocore.exceptions import ClientError

from alpha_edge.core.market_store import MarketStore
from alpha_edge.market.aea87_publication_transaction import fetch, checksum, put_once
from alpha_edge.market.aea87_rebuild_publication_preflight import check_asset

ROOT = "market/repairs/aea87/rebuilds/v1/runs"
PREPARED = "market/repairs/aea87/rebuild_publications/v1"
TABLES = {"ohlcv_usd": "candidate_ohlcv_usd.parquet",
          "returns_usd": "candidate_returns_usd.parquet"}
# These fields have downstream consumers. Other columns may also be required;
# enforce existing data schema compatibility below.
REQUIRED = {
    "ohlcv_usd": {"asset_id", "date", "ticker", "yahoo_ticker", "currency",
                  "close_raw_usd", "close_adjusted_usd", "close_usd", "adj_close_usd"},
    "returns_usd": {"asset_id", "date", "ticker", "ret_adj_close_usd",
                    "ret_close_adjusted_usd"},
}


def publication_schema_issues(candidate, active, table):
    missing = sorted(REQUIRED[table] - set(candidate.columns))
    issues = ["MISSING_REQUIRED_COLUMN:" + col for col in missing]
    if not active.empty:
        # Field-by-field comparison of columns in use by the active dataset.
        # A missing column is a publication blocker, even if the raw Yahoo source
        # itself was valid. Never silently manufacture identity metadata.
        removed = sorted(set(active.columns) - set(candidate.columns))
        issues.extend("ACTIVE_COLUMN_NOT_PRESERVED:" + col for col in removed)
    if "asset_id" in candidate and candidate["asset_id"].astype(str).nunique() != 1:
        issues.append("MULTIPLE_ASSET_IDENTITIES")
    if "date" in candidate and candidate["date"].duplicated().any():
        issues.append("DUPLICATE_CANDIDATE_DATES")
    return sorted(set(issues))


def align_verified_metadata(candidate, active, table, *, provider_symbol, provider_currency):
    """Add verified identity and mathematically derived fields, never prices.

    Identity metadata must be invariant in active data, and the Yahoo symbol
    must not conflict with the active provider symbol. Fail closed otherwise.
    """
    out = candidate.copy()
    if "asset_id" not in out or out["asset_id"].nunique() != 1:
        raise ValueError("Candidate asset identity is missing or ambiguous")
    expected = {"yahoo_ticker": provider_symbol, "currency": provider_currency}
    for field in ("ticker", "yahoo_ticker", "currency"):
        if field in active and active[field].notna().any():
            values = {str(x).strip() for x in active[field].dropna().unique() if str(x).strip()}
            if len(values) != 1:
                raise ValueError("Ambiguous active identity field: " + field)
            actual = next(iter(values))
            if field in expected and expected[field] and actual.upper() != str(expected[field]).upper():
                raise ValueError("Provider identity conflict: " + field)
            if field in out and out[field].notna().any() and out[field].astype(str).ne(actual).any():
                raise ValueError("Candidate identity conflict: " + field)
            out[field] = actual
        elif field in expected and expected[field]:
            out[field] = expected[field]
    if "ticker" not in out:
        raise ValueError("Missing authoritative broker ticker")
    if table == "returns_usd":
        # Derived log-return identity: log(1+r); first return remains null.
        r = pd.to_numeric(out["ret_close_adjusted_usd"], errors="coerce")
        if (r.dropna() <= -1).any():
            raise ValueError("Impossible return <= -100%")
        out["ret_log_close_adjusted_usd"] = np.log1p(r)
    return out


def _parquet(frame):
    out = io.BytesIO()
    frame.to_parquet(out, index=False)
    return out.getvalue()


def _put_verified(s3, bucket, key, payload):
    previous = fetch(s3, bucket, key)
    if previous is None:
        put_once(s3, bucket, key, payload)
        previous = fetch(s3, bucket, key)
    if checksum(previous) != checksum(payload):
        raise RuntimeError("Immutable staging object changed: " + key)


def prepare(s3, store, bucket, source_run, asset_id, end, publication_id):
    preflight = check_asset(s3, bucket, store, source_run, asset_id, end)
    if preflight["blockers"]:
        return {"asset_id": asset_id, "status": "BLOCKED_PREFLIGHT",
                "blockers": preflight["blockers"], "operations": []}
    folder = "".join(c if c.isalnum() else "_" for c in asset_id)
    prefix = f"{ROOT}/{source_run}/assets/{folder}/"
    frames = {}
    for table, name in TABLES.items():
        raw = s3.get_object(Bucket=bucket, Key=prefix + name)["Body"].read()
        frames[table] = pd.read_parquet(io.BytesIO(raw))
    active = {
        "ohlcv_usd": store.read_ohlcv_usd([asset_id], start="2010-01-01", end=end),
        "returns_usd": store.read_returns_usd([asset_id], start="2010-01-01", end=end),
    }
    source_summary = json.loads(s3.get_object(Bucket=bucket, Key=f"{ROOT}/{source_run}/summary.json")["Body"].read())
    source = source_summary["assets"][asset_id]
    if source.get("status") != "STAGED_FOR_REVIEW":
        raise ValueError("Source reconstruction is not staged for review")
    provider = source.get("provider_metadata") or {}
    for table in TABLES:
        frames[table] = align_verified_metadata(frames[table], active[table], table,
            provider_symbol=source["yahoo_ticker"], provider_currency=provider.get("currency"))
    # Production may have advanced after the staged Yahoo run. A full-history
    # replacement must never discard later live market sessions.
    current_day = datetime.now(timezone.utc).date().isoformat()
    stale = []
    for table in TABLES:
        later = (store.read_ohlcv_usd([asset_id], start=end, end=current_day)
                 if table == "ohlcv_usd" else
                 store.read_returns_usd([asset_id], start=end, end=current_day))
        if not later.empty:
            live_latest = pd.to_datetime(later["date"], utc=True).max()
            candidate_latest = pd.to_datetime(frames[table]["date"], utc=True).max()
            if live_latest > candidate_latest:
                stale.append(f"{table}:ACTIVE_DATA_NEWER_THAN_CANDIDATE:{live_latest.date()}")
    if stale:
        return {"asset_id": asset_id, "status": "BLOCKED_STALE_SOURCE",
                "blockers": stale, "operations": []}
    problems = []
    for table in TABLES:
        problems += [table + ":" + x for x in publication_schema_issues(
            frames[table], active[table], table)]
    if problems:
        return {"asset_id": asset_id, "status": "BLOCKED_SCHEMA",
                "blockers": sorted(set(problems)), "operations": []}
    root = f"{PREPARED}/{publication_id}"
    proposed = []
    for table, frame in frames.items():
        dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        if dates.isna().any():
            raise ValueError("Invalid source dates")
        years = set(dates.dt.year)
        old = active[table]
        old_years = set(pd.to_datetime(old["date"], utc=True).dt.year) if not old.empty else set()
        for year in sorted(years | old_years):
            manifest_key = store._manifest_key(table=table, asset_id=asset_id, year=year)
            before = fetch(s3, bucket, manifest_key)
            # Never activate without explicitly knowing what preceded this publication.
            prior_sha = checksum(before)
            if year in years:
                selected = frame[dates.dt.year == year].copy()
                payload = _parquet(selected)
                part_key = f"{root}/parts/{table}/asset_id={asset_id}/year={year}/data.parquet"
                new_manifest = {"asset_id": asset_id, "year": int(year),
                                "parts": [part_key], "dates": sorted(selected["date"].dt.strftime("%Y-%m-%d").tolist()),
                                "reconstruction_run_id": source_run,
                                "publication_id": publication_id}
            else:
                payload = None
                part_key = None
                new_manifest = {"asset_id": asset_id, "year": int(year),
                                "parts": [], "dates": [], "status": "RETIRED",
                                "publication_id": publication_id}
            after = json.dumps(new_manifest, sort_keys=True).encode()
            proposed.append({"table": table, "year": int(year), "asset_id": asset_id,
                             "manifest_key": manifest_key, "before_sha": prior_sha,
                             "before_missing": before is None,
                             "backup_key": f"{root}/backups/{table}/year={year}/manifest.json" if before is not None else None,
                             "part_key": part_key, "part_sha": checksum(payload),
                             "part_payload": payload, "after_sha": checksum(after),
                             "after": after.decode()})
    # Require no concurrent changes, and immutable source material before writing.
    for op in proposed:
        if checksum(fetch(s3, bucket, op["manifest_key"])) != op["before_sha"]:
            raise RuntimeError("Active manifest changed during preparation: " + op["manifest_key"])
    for op in proposed:
        if op["backup_key"]:
            original = fetch(s3, bucket, op["manifest_key"])
            if checksum(original) != op["before_sha"]:
                raise RuntimeError("Manifest drift during backup")
            _put_verified(s3, bucket, op["backup_key"], original)
        if op["part_key"]:
            _put_verified(s3, bucket, op["part_key"], op["part_payload"])
    clean = [{k: v for k, v in op.items() if k != "part_payload"} for op in proposed]
    result = {"asset_id": asset_id, "status": "PREPARED_NOT_PUBLISHED",
              "source_run": source_run, "operations": clean,
              "active_histories_modified": False,
              "required_before_activation": [
                  "EXCLUSIVE_INGESTION_AND_READER_MAINTENANCE",
                  "DOWNSTREAM_CACHE_SNAPSHOT_REFRESH_VALIDATED",
                  "ATOMIC_VISIBILITY_OR_MAINTENANCE_ENFORCEMENT",
                  "TESTED_ROLLBACK_AND_DRIFT_RECHECK"]}
    plan_key = f"{root}/plan.json"
    _put_verified(s3, bucket, plan_key, json.dumps(result, indent=2).encode())
    result["plan_uri"] = f"s3://{bucket}/{plan_key}"
    return result


def main():
    p = argparse.ArgumentParser(description="Prepare immutable reconstructed market partitions; no active writes")
    p.add_argument("--source-run-id", required=True)
    p.add_argument("--asset-id", default="EQH028b839246b50817")
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--end", default="2026-11-01")
    p.add_argument("--publication-id")
    a = p.parse_args()
    pub_id = a.publication_id or ("rebuild_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    if not all(c.isalnum() or c in "-_" for c in pub_id):
        p.error("Unsafe publication ID")
    s3 = boto3.client("s3", region_name=a.region)
    store = MarketStore(bucket=a.bucket, region=a.region)
    outcome = prepare(s3, store, a.bucket, a.source_run_id, a.asset_id, a.end, pub_id)
    if outcome["status"].startswith("BLOCKED"):
        report_key = f"{PREPARED}/{pub_id}/blocked_report.json"
        _put_verified(s3, a.bucket, report_key, json.dumps(outcome, indent=2).encode())
        outcome["s3_report"] = f"s3://{a.bucket}/{report_key}"
    print(json.dumps({k: v for k, v in outcome.items() if k != "operations"} |
                     {"operation_count": len(outcome["operations"])}, indent=2))
    if outcome["status"].startswith("BLOCKED"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
