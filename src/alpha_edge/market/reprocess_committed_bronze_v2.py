"""Reprocess an existing immutable v2 backfill under current audit policy.

Used for EURUSD=X weekday-session correction; never downloads Yahoo or
modifies the original committed Bronze or research candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import boto3
import pandas as pd

from alpha_edge.market.audit_missing_ohlcv_v2 import generate as audit_missing
from alpha_edge.market.independent_acceptance_v2 import generate as acceptance
from alpha_edge.market.run_s3_full_history_backfill_v2 import (
    _get_remote, upload_verified, verify_commit,
)
from alpha_edge.market.run_standardization_pilot_v2 import generate as standardize

ROOT = "dev/market/lake_v2/reprocessed_candidates"
EUR_SOURCE = (
    "dev/market/lake_v2/backfills/6e117c996ce8375eb1985659/"
    "end-2026-11-11/cohort-0001"
)


def run(
    s3, bucket: str, *, universe_csv: Path, fx_csv: Path,
    source_prefix: str = EUR_SOURCE, asset_id: str = "EQHe874aca04f607837",
    symbol: str = "EURUSD=X", execute: bool = False,
) -> dict:
    commit_blob = _get_remote(s3, bucket, source_prefix + "/_committed.json")
    if commit_blob is None:
        raise ValueError("Source research cohort has no commit marker")
    commit = json.loads(commit_blob)
    if commit.get("schema") != "alpha_edge_s3_backfill_commit_v2_1":
        raise ValueError("Unexpected committed backfill schema")
    lookup = {entry["path"]: entry["sha256"] for entry in commit["files"]}
    source_root = "execution/bronze_batches/s3bf-0001-0001/"
    needed = [source_root + x for x in (
        "manifest.json", "source_ohlcv.parquet", "source_fx_quotes.parquet"
    )]
    if not set(needed).issubset(lookup):
        raise ValueError("Source commit lacks Bronze files")
    universe = pd.read_csv(universe_csv, dtype=str).fillna("")
    chosen = universe.loc[universe["asset_id"].eq(asset_id)]
    if len(chosen) != 1:
        raise ValueError("Requested asset identity is not unique")
    asset = chosen.iloc[0]
    if asset["include"] != "1" or asset["yahoo_ticker"] != symbol:
        raise ValueError("Current active source identity does not match reviewed asset")
    calendar = asset["calendar_id"]
    if not calendar:
        raise ValueError("Missing reviewed asset calendar")
    config = pd.DataFrame([{
        "asset_id": asset_id, "yahoo_ticker": symbol,
        "currency": asset["currency"], "calendar_id": calendar, "include": 1,
    }])
    registry = pd.read_csv(fx_csv, dtype=str)
    contract = {
        "source_commit_sha256": hashlib.sha256(commit_blob).hexdigest(),
        "source_bronze_files": {name: lookup[name] for name in needed},
        "asset_registry": config.to_dict("records"),
        "fx_registry_sha256": hashlib.sha256(fx_csv.read_bytes()).hexdigest(),
        "pipeline": "reprocess_committed_bronze_v2_1",
        "source_prefix": source_prefix,
        "asset_id": asset_id,
    }
    contract_bytes = (json.dumps(contract, sort_keys=True, indent=2) + "\n").encode()
    run_sha = hashlib.sha256(contract_bytes).hexdigest()
    prefix = f"{ROOT}/run-{run_sha}/asset_id={asset_id}"
    if not execute:
        return {"status": "ready", "asset_id": asset_id,
                "calendar_id": calendar, "candidate_prefix": f"s3://{bucket}/{prefix}",
                "s3_writes": 0}
    if verify_commit(s3, bucket, prefix, run_sha) is not None:
        return {"status": "verified_existing", "asset_id": asset_id,
                "candidate_prefix": f"s3://{bucket}/{prefix}"}
    with tempfile.TemporaryDirectory(prefix="aea87_reprocess_bronze_") as temp:
        root = Path(temp)
        bronze = root / "bronze"
        bronze.mkdir()
        for name in needed:
            raw = _get_remote(s3, bucket, source_prefix + "/files/" + name)
            if raw is None or hashlib.sha256(raw).hexdigest() != lookup[name]:
                raise ValueError(f"Committed source checksum mismatch: {name}")
            (bronze / Path(name).name).write_bytes(raw)
        source_manifest = json.loads((bronze / "manifest.json").read_text())
        sources = source_manifest.get("sources", [])
        if len(sources) != 1 or sources[0].get("asset_id") != asset_id or sources[0].get("provider_symbol") != symbol:
            raise ValueError("Committed Bronze source identity mismatch")
        asset_path = root / "asset_registry.csv"
        config.to_csv(asset_path, index=False)
        fx_path = root / "fx_registry.csv"
        registry.to_csv(fx_path, index=False)
        audit_missing(bronze, asset_path, fx_path, root / "audit")
        standardize(
            bronze, output=root / "candidate",
            asset_registry_csv=asset_path, fx_registry_csv=fx_path,
        )
        report = acceptance(
            candidate_dir=root / "candidate", bronze_dir=bronze,
            output=root / "acceptance", asset_registry_csv=asset_path,
            fx_registry_csv=fx_path,
        )
        (root / "source_contract.json").write_bytes(contract_bytes)
        upload_verified(
            s3, bucket=bucket, prefix=prefix, local_dir=root,
            plan_hash=run_sha, end=commit["end_exclusive"],
        )
    return {
        "status": "reprocessed_candidate_committed",
        "asset_id": asset_id, "calendar_id": calendar,
        "internal_integrity_status": report.get("internal_integrity_status"),
        "acceptance": report.get("acceptance"),
        "blocker_counts": report.get("blocker_counts"),
        "candidate_prefix": f"s3://{bucket}/{prefix}",
        "production_eligible": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--universe", type=Path, default=Path("data/universe/universe.csv"))
    p.add_argument("--fx-config", type=Path, default=Path("data/quality/market_lake_v2/pilot_fx_registry_v2.csv"))
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(
        boto3.client("s3", region_name=args.region), args.bucket,
        universe_csv=args.universe, fx_csv=args.fx_config, execute=args.execute,
    ), indent=2))


if __name__ == "__main__":
    main()
