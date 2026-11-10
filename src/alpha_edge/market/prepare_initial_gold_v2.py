"""Prepare an immutable initial S3 Gold release candidate from full-history cohorts.

No local artifact files. No legacy v1 writes. Does not claim that historical
quality blockers are resolved. The initial release requires ALL expected
research cohort commits and production raw Bronze commit receipts. Stage
Silver/Gold and indicator *candidates*, carrying raw provenance, quality
reports, correction registry and original asset/FX mappings. It cannot
activate a production pointer without independent acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import uuid
from datetime import date

import pandas as pd

from alpha_edge.core.audit import make_run_id, build_audit_event, write_audit_event
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.market.archive_local_outputs_v2 import archive_prefix
from alpha_edge.market.production_gold_release_v2 import (
    ROOT, encoded, get, sha, immutable_put, verify_object, stage,
)
from alpha_edge.market.run_daily_pipeline_v2 import _indicators, _put_parquet


class InitializationError(RuntimeError):
    pass


def _read(s3, bucket: str, key: str, digest: str) -> bytes:
    blob = get(s3, bucket, key)
    if blob is None or sha(blob) != digest:
        raise InitializationError(f"Missing or SHA-corrupt immutable S3 input: {key}")
    return blob


def _archive_files(s3, bucket: str, archive_id: str,
                   relative_prefix: str) -> dict[str, bytes]:
    prefix = archive_prefix(archive_id)
    marker = get(s3, bucket, prefix + "/_verified_inventory.json")
    if marker is None:
        raise InitializationError("Missing finalized S3 archive")
    archive = json.loads(marker)
    if archive.get("schema") != "alpha_edge_local_output_archive_v2_1":
        raise InitializationError("Unexpected archive schema")
    records = archive.get("files", [])
    root = relative_prefix.strip("/") + "/"
    selected = [f for f in records if f["path"].startswith(root)]
    if not selected:
        raise InitializationError("Requested cohort plan not present in archive")
    result = {}
    for item in selected:
        suffix = item["path"][len(root):]
        if "/" in suffix or not re.fullmatch(r"[A-Za-z0-9_.-]+", suffix):
            raise InitializationError("Invalid archived plan filename")
        result[suffix] = _read(
            s3, bucket, prefix + "/files/" + item["path"], item["sha256"])
    return result


def _store(s3, bucket: str, key: str, blob: bytes) -> dict:
    previous = get(s3, bucket, key)
    if previous is None:
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=blob,
                          IfNoneMatch="*", ContentType="application/octet-stream")
        except Exception:
            if get(s3, bucket, key) != blob:
                raise
    elif previous != blob:
        raise InitializationError(f"Immutable snapshot collision: {key}")
    if get(s3, bucket, key) != blob:
        raise InitializationError(f"Snapshot checksum mismatch: {key}")
    return {"key": key, "sha256": sha(blob)}


def build(*, s3, bucket: str, archive_id: str, plan_hash_prefix: str,
          end: str, run_id: str, execute: bool) -> dict:
    if not re.fullmatch("[0-9a-f]{24}", plan_hash_prefix):
        raise InitializationError("Invalid plan prefix")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", end):
        raise InitializationError("Invalid end date")
    files = _archive_files(
        s3, bucket, archive_id, "full_universe_preflight_v2/first_review")
    if "manifest.json" not in files or "fx_config.csv" not in files:
        raise InitializationError("Missing pinned cohort plan files")
    plan = json.loads(files["manifest.json"])
    if plan.get("schema") != "alpha_edge_lake_v2_universe_preflight_1":
        raise InitializationError("Unexpected plan schema")
    if sha(files["manifest.json"])[:24] != plan_hash_prefix:
        raise InitializationError("Plan hash differs from requested research run")
    for filename, digest in plan.get("files_sha256", {}).items():
        if filename not in files or sha(files[filename]) != digest:
            raise InitializationError(f"Pinned cohort source differs: {filename}")
    count = int(plan["cohorts"])
    fx_mapping = pd.read_csv(io.BytesIO(files["fx_config.csv"])).to_dict("records")
    source_prefix = (
        f"dev/market/lake_v2/backfills/{plan_hash_prefix}/end-{end}")
    raw_prefix = (
        f"{ROOT}/bronze/full_history/{plan_hash_prefix}/end-{end}")
    # PREFLIGHT FULL COMPLETENESS before a single production write.
    inventory = []
    missing = []
    for index in range(1, count + 1):
        cohort = f"cohort-{index:04d}"
        research = f"{source_prefix}/{cohort}"
        production = f"{raw_prefix}/{cohort}"
        research_raw = get(s3, bucket, research + "/_committed.json")
        bronze_raw = get(s3, bucket, production + "/_bronze_committed.json")
        if research_raw is None or bronze_raw is None:
            missing.append(index)
            continue
        research_commit = json.loads(research_raw)
        bronze_commit = json.loads(bronze_raw)
        if (research_commit.get("schema") != "alpha_edge_s3_backfill_commit_v2_1"
                or research_commit.get("prefix") != research
                or bronze_commit.get("schema") != "alpha_edge_production_bronze_commit_v2_1"
                or bronze_commit.get("source_commit_sha256") != sha(research_raw)):
            raise InitializationError(f"Source/production Bronze mismatch: {cohort}")
        entries = {item["path"]: item["sha256"] for item in research_commit["files"]}
        registry = pd.read_csv(io.BytesIO(files[f"cohort_{index:04d}.csv"])).to_dict("records")
        for row in registry:
            if pd.isna(row.get("currency")) or pd.isna(row.get("calendar_id")):
                raise InitializationError(f"Invalid configured asset in cohort {index}")
        inventory.append({
            "index": index, "research": research, "production": production,
            "entries": entries, "registry": registry,
            "bronze_commit": bronze_commit,
        })
    if missing:
        return {
            "status": "blocked_missing_cohorts", "missing_count": len(missing),
            "missing_cohorts": missing, "expected_cohorts": count,
            "s3_writes": 0,
        }
    if not execute:
        return {
            "status": "ready_to_stage_unapproved_initial_release",
            "cohorts": len(inventory), "assets": plan["assets_included"],
            "release_id": run_id, "s3_writes": 0,
        }
    result = []
    for item in inventory:
        index, research, production = (
            item["index"], item["research"], item["production"])
        prefix = f"{ROOT}/releases/data/{run_id}/cohort-{index:04d}"
        contents = {}
        for filename, output_name in (
            ("candidate/silver.parquet", "silver"),
            ("candidate/gold.parquet", "gold"),
            ("acceptance/manifest.json", "acceptance"),
        ):
            expected = item["entries"].get("files/" + filename)
            # Backfill commit inventory uses paths relative to ephemeral root,
            # not the additional 'files/' S3 key namespace.
            if expected is None:
                expected = item["entries"].get(filename)
            if expected is None:
                raise InitializationError(f"Missing research artifact: {filename}")
            blob = _read(s3, bucket, research + "/files/" + filename, expected)
            extension = ".json" if output_name == "acceptance" else ".parquet"
            contents[output_name] = _store(
                s3, bucket, f"{prefix}/{output_name}{extension}", blob)
        gold = pd.read_parquet(io.BytesIO(
            _read(s3, bucket, contents["gold"]["key"], contents["gold"]["sha256"])))
        ids = [x["asset_id"] for x in item["registry"]]
        if set(gold["asset_id"]) != set(ids):
            raise InitializationError(f"Gold population mismatch cohort {index}")
        indicator = _indicators(gold)
        contents["indicator_candidate"] = _put_parquet(
            s3, bucket, f"{prefix}/indicators.parquet", indicator)
        corrections_name = "candidate/manual_corrections_audit.parquet"
        if corrections_name in item["entries"]:
            blob = _read(
                s3, bucket, research + "/files/" + corrections_name,
                item["entries"][corrections_name])
            contents["corrections"] = _store(
                s3, bucket, f"{prefix}/corrections.parquet", blob)
        else:
            if "observation_origin" in gold and gold["observation_origin"].eq(
                    "user_supplied_manual").any():
                raise InitializationError("Manual observations without pinned correction records")
            contents["corrections"] = None
        contents["bronze"] = {
            "key": production + "/source_ohlcv.parquet",
            "sha256": item["bronze_commit"]["sha256"]["source_ohlcv.parquet"],
        }
        contents["fx"] = {
            "key": production + "/source_fx_quotes.parquet",
            "sha256": item["bronze_commit"]["sha256"]["source_fx_quotes.parquet"],
        }
        result.append({
            "cohort": index, "assets": ids, "registry": item["registry"],
            "fx_mapping": fx_mapping, "gold_rows": len(gold), **contents,
        })
    release = {
        "schema": "alpha_edge_v2_release_candidate_1",
        "release_id": run_id,
        "as_of": (date.fromisoformat(end) - pd.Timedelta(days=1)).isoformat(),
        "expected_cohorts": count, "expected_assets": plan["assets_included"],
        "cohorts": result, "initial_backfill": True,
        "source_archive_id": archive_id,
        "plan_manifest_sha256": sha(files["manifest.json"]),
        "universe_reconciliation": "pending",
        "indicator_validation": "pending",
        "consumer_contract_validation": "pending",
        "source_observation_state": "requires_independent_evidence_approval",
    }
    staged = stage(s3, bucket=bucket, release=release, execute=True)
    return {
        "status": "initial_release_candidate_staged_unapproved",
        "release_id": run_id,
        "release_manifest": staged["release_key"],
        "release_sha256": staged["release_sha256"],
        "cohorts": len(result),
        "assets": plan["assets_included"],
        "production_pointer_changed": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--plan-archive-id", default="old-outputs-20261110")
    p.add_argument("--plan-hash-prefix", required=True)
    p.add_argument("--end", default="2026-11-10")
    p.add_argument("--run-id", default=None)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm-prod-write", action="store_true")
    a = p.parse_args()
    cfg = load_runtime_config("prod")
    if a.execute:
        require_prod_confirmation(cfg, a.confirm_prod_write)
    import boto3
    s3 = boto3.client("s3", region_name=a.region)
    rid = a.run_id or make_run_id("market_v2_initial")
    try:
        result = build(s3=s3, bucket=a.bucket, archive_id=a.plan_archive_id,
                       plan_hash_prefix=a.plan_hash_prefix, end=a.end,
                       run_id=rid, execute=a.execute)
        if a.execute:
            ev = build_audit_event(
                cfg=cfg, event_type="market_v2_initial_release",
                entity_type="market_lake", entity_id=rid, as_of=a.end,
                source_script="alpha_edge.market.prepare_initial_gold_v2",
                source_mode="prod_candidate", status=result["status"],
                output_keys=[result["release_manifest"]]
                if "release_manifest" in result else [], metadata=result)
            write_audit_event(cfg=cfg, event=ev)
    except Exception as exc:
        if a.execute:
            ev = build_audit_event(
                cfg=cfg, event_type="market_v2_initial_release",
                entity_type="market_lake", entity_id=rid, as_of=a.end,
                source_script="alpha_edge.market.prepare_initial_gold_v2",
                source_mode="prod_candidate", status="failed",
                error=f"{type(exc).__name__}: {exc}")
            write_audit_event(cfg=cfg, event=ev)
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
