"""Standardize existing asset-level Bronze recoveries, without Yahoo or v1 prices.

Reuses the existing v2 audit, Silver/Gold, independent acceptance, and
immutable S3 commit functions. Per-asset S3 candidates are resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.market.asset_level_recovery_v2 import RECOVERY_ROOT, _json_bytes
from alpha_edge.market.audit_missing_ohlcv_v2 import generate as audit_missing
from alpha_edge.market.independent_acceptance_v2 import generate as acceptance
from alpha_edge.market.run_s3_full_history_backfill_v2 import upload_verified, verify_commit
from alpha_edge.market.run_standardization_pilot_v2 import generate as standardize
from alpha_edge.market.universe_registry_v2 import (
    _get, _sha, cohort_commit_presence, load_archived_plan,
)

ROOT = "dev/market/lake_v2/recovered_candidates"
ORIGINAL_PLAN_SHA = "764b62e3a94f92b5ce07c4bff9771e64dc5780d6cdd6c5c1f1112c897c8d4790"


class RecoveryStandardizationError(RuntimeError):
    pass


def _optional(s3, bucket: str, key: str) -> bytes | None:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise


def load_pinned_fx(s3, bucket: str, *, original_end: str, original_plan_sha: str):
    """Read existing, committed FX quote evidence from original v2 cohort 1."""
    cohort = (
        f"dev/market/lake_v2/backfills/{original_plan_sha[:24]}/"
        f"end-{original_end}/cohort-0001"
    )
    marker = _optional(s3, bucket, cohort + "/_committed.json")
    if marker is None:
        raise RecoveryStandardizationError("Pinned FX source cohort is not committed")
    commit = json.loads(marker)
    if commit.get("plan_manifest_sha256") != original_plan_sha:
        raise RecoveryStandardizationError("FX evidence sourced from different archived plan")
    path = "execution/bronze_batches/s3bf-0001-0001/source_fx_quotes.parquet"
    files = {item["path"]: item["sha256"] for item in commit["files"]}
    if path not in files:
        raise RecoveryStandardizationError("Committed source has no FX quote evidence")
    payload = _get(s3, bucket, cohort + "/files/" + path)
    if _sha(payload) != files[path]:
        raise RecoveryStandardizationError("Pinned FX source SHA mismatch")
    return pd.read_parquet(io.BytesIO(payload)), {
        "s3_key": cohort + "/files/" + path,
        "sha256": files[path],
        "commit_sha256": _sha(marker),
    }


def process_one(
    s3, bucket: str, row: dict, *, plan_sha: str, end: str,
    fx_table: pd.DataFrame, fx_source: dict, fx_config: pd.DataFrame,
    execute: bool = False,
) -> dict:
    """Restore and validate one immutable recovery, then reuse v2 processing."""
    asset_id = str(row["asset_id"])
    symbol = str(row["yahoo_ticker"])
    currency = str(row["currency"]).upper()
    calendar = str(row["calendar_id"])
    cohort = int(row["original_cohort"])
    source = {
        "asset_id": asset_id, "provider_symbol": symbol,
        "currency": currency, "calendar_id": calendar,
        "original_cohort": cohort, "original_plan_sha256": plan_sha,
        "end_exclusive": end,
    }
    recovery_id = _sha(_json_bytes({"schema": "aea87_recovery_identity_v1", **source}))
    base = f"{RECOVERY_ROOT}/run-{recovery_id}"
    manifest_blob = _optional(s3, bucket, base + "/manifest.json")
    if manifest_blob is None:
        return {"asset_id": asset_id, "status": "source_missing", "recovery_id": recovery_id}
    manifest = json.loads(manifest_blob)
    if manifest.get("schema") != "aea87_asset_recovery_candidate_v1":
        return {"asset_id": asset_id, "status": "source_not_acquired", "recovery_id": recovery_id}
    if manifest.get("run_id") != recovery_id or any(
        manifest.get("source", {}).get(field) != value
        for field, value in source.items()
    ):
        raise RecoveryStandardizationError(f"{asset_id}: recovery identity mismatch")
    bronze_sha = manifest["artifact_sha256"]["source_ohlcv.parquet"]
    source_identity = {
        "source": source,
        "recovery_manifest_sha256": _sha(manifest_blob),
        "bronze_sha256": bronze_sha,
        "fx_evidence_sha256": fx_source["sha256"],
        "fx_config": fx_config.to_dict("records"),
        "pipeline": "recovered_bronze_standardization_v2_1",
    }
    candidate_plan_sha = _sha(_json_bytes(source_identity))
    prefix = f"{ROOT}/run-{candidate_plan_sha}/asset_id={asset_id}"
    if not execute:
        return {
            "asset_id": asset_id, "status": "ready",
            "target": f"s3://{bucket}/{prefix}/_committed.json",
        }
    existing = verify_commit(s3, bucket, prefix, candidate_plan_sha)
    if existing is not None:
        return {"asset_id": asset_id, "status": "verified_existing",
                "target": f"s3://{bucket}/{prefix}/_committed.json"}
    raw = _get(s3, bucket, base + "/source_ohlcv.parquet")
    if _sha(raw) != bronze_sha:
        raise RecoveryStandardizationError(f"{asset_id}: recovery data SHA mismatch")
    mapped = fx_config.loc[fx_config["currency"].eq(currency)]
    if currency != "USD" and len(mapped) != 1:
        raise RecoveryStandardizationError(f"{asset_id}: FX configuration missing")
    subset = (
        fx_table.loc[fx_table["provider_symbol"].eq(mapped.iloc[0]["provider_symbol"])].copy()
        if currency != "USD"
        else fx_table.iloc[0:0].copy()
    )
    if currency != "USD" and subset.empty:
        raise RecoveryStandardizationError(f"{asset_id}: pinned FX quote source empty")
    with tempfile.TemporaryDirectory(prefix="aea87_recovered_candidate_") as temp:
        base_dir = Path(temp)
        bronze = base_dir / "bronze"
        bronze.mkdir()
        (bronze / "source_ohlcv.parquet").write_bytes(raw)
        subset.to_parquet(bronze / "source_fx_quotes.parquet", index=False)
        batch_id = "recovery-" + recovery_id[:20]
        bronze_manifest = {
            "schema_version": "bronze-v2-1", "batch_id": batch_id,
            "files": {
                p.name: {"sha256": _sha(p.read_bytes()), "bytes": p.stat().st_size}
                for p in bronze.glob("*.parquet")
            },
            "sources": [manifest["source"]["provider_provenance"]],
            "recovered_from_s3": base + "/manifest.json",
            "pinned_fx_source": fx_source,
        }
        (bronze / "manifest.json").write_text(
            json.dumps(bronze_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        config = base_dir / "asset_registry.csv"
        pd.DataFrame([{
            "asset_id": asset_id, "yahoo_ticker": symbol,
            "currency": currency, "calendar_id": calendar, "include": 1,
        }]).to_csv(config, index=False)
        fx_path = base_dir / "fx_config.csv"
        fx_config.to_csv(fx_path, index=False)
        audit_missing(bronze, config, fx_path, base_dir / "audit")
        standardize(
            bronze, output=base_dir / "candidate",
            asset_registry_csv=config, fx_registry_csv=fx_path,
        )
        audit = acceptance(
            candidate_dir=base_dir / "candidate",
            bronze_dir=bronze, output=base_dir / "acceptance",
            asset_registry_csv=config, fx_registry_csv=fx_path,
        )
        (base_dir / "source_contract.json").write_bytes(_json_bytes(source_identity))
        commit = upload_verified(
            s3, bucket=bucket, prefix=prefix, local_dir=base_dir,
            plan_hash=candidate_plan_sha, end=end,
        )
    return {
        "asset_id": asset_id,
        "status": "candidate_committed",
        "acceptance": audit.get("acceptance"),
        "internal_integrity_status": audit.get("internal_integrity_status"),
        "files": len(commit["files"]),
        "target": f"s3://{bucket}/{prefix}/_committed.json",
    }


def run(
    s3, bucket: str, archive_id: str, *, original_end: str,
    start_index: int = 0, max_assets: int = 10, execute: bool = False,
) -> dict:
    if start_index < 0 or not 1 <= max_assets <= 25:
        raise RecoveryStandardizationError("Invalid bounded selection")
    plan, info = load_archived_plan(s3, bucket, archive_id)
    if info["plan_sha256"] != ORIGINAL_PLAN_SHA:
        raise RecoveryStandardizationError("Archived plan differs from recovery campaign")
    states = cohort_commit_presence(
        s3, bucket, info["plan_sha256"], original_end, info["expected_cohorts"]
    )
    selection = plan.loc[plan["original_cohort"].map(states).eq("missing")]
    selection = selection.sort_values("asset_id").reset_index(drop=True)
    if start_index >= len(selection):
        raise RecoveryStandardizationError("Start index beyond missing-cohort asset queue")
    selected = selection.iloc[start_index:start_index + max_assets]
    if not execute:
        return {"status": "dry_run", "selected": selected["asset_id"].tolist(),
                "total_assets": len(selection), "s3_writes": 0}
    fx_table, fx_source = load_pinned_fx(
        s3, bucket, original_end=original_end, original_plan_sha=info["plan_sha256"]
    )
    # This is the pinned FX configuration from the archived v2 preflight.
    from alpha_edge.market.archive_local_outputs_v2 import archive_prefix
    archive = archive_prefix(archive_id)
    base = "full_universe_preflight_v2/first_review"
    inventory = json.loads(_get(s3, bucket, archive + "/_verified_inventory.json"))
    key = base + "/fx_config.csv"
    items = {entry["path"]: entry["sha256"] for entry in inventory["files"]}
    raw = _get(s3, bucket, archive + "/files/" + key)
    if _sha(raw) != items[key]:
        raise RecoveryStandardizationError("Archived FX registry SHA mismatch")
    fx_config = pd.read_csv(io.BytesIO(raw), dtype=str)
    results = []
    for row in selected.to_dict("records"):
        try:
            results.append(process_one(
                s3, bucket, row, plan_sha=info["plan_sha256"], end=original_end,
                fx_table=fx_table, fx_source=fx_source, fx_config=fx_config, execute=True,
            ))
        except ClientError:
            raise
        except Exception as exc:
            results.append({
                "asset_id": row["asset_id"], "status": "processing_failed",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            })
    from collections import Counter
    return {
        "status": "candidate_batch_completed",
        "processed": len(results),
        "counts": dict(Counter(r["status"] for r in results)),
        "results": results,
        "production_eligible": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--archive-id", default="old-outputs-20261110")
    p.add_argument("--end", default="2026-11-10")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-assets", type=int, default=10)
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(
        boto3.client("s3", region_name=args.region), args.bucket, args.archive_id,
        original_end=args.end, start_index=args.start_index,
        max_assets=args.max_assets, execute=args.execute,
    ), indent=2))


if __name__ == "__main__":
    main()
