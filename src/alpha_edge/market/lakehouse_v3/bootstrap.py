"""Bootstrap an immutable v3 S3 registry from the checksummed S3-only v2 plan.

The archived source is a historical snapshot, NOT necessarily the latest
production universe. Any eligibility/identity changes require a reviewed
new registry revision before Gold approval.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json

import pandas as pd

from alpha_edge.market.archive_local_outputs_v2 import archive_prefix
from alpha_edge.market.universe_registry_v2 import load_archived_plan
from alpha_edge.market.lakehouse_v3.storage import ROOT, LakehouseError, json_bytes, put_json, read_bytes


def build_registry(
    s3, bucket: str, *, archive_id: str, selected_asset_ids: list[str] | None = None,
    execute: bool = False,
) -> dict:
    frame, info = load_archived_plan(s3, bucket, archive_id)
    root = archive_prefix(archive_id)
    index = json.loads(read_bytes(s3, bucket, f"{root}/_verified_inventory.json"))
    path = "full_universe_preflight_v2/first_review/fx_config.csv"
    evidence = {x["path"]: x for x in index["files"]}
    if path not in evidence:
        raise LakehouseError("Archived plan lacks expected pinned FX config")
    fx_blob = read_bytes(s3, bucket, f"{root}/files/{path}")
    if hashlib.sha256(fx_blob).hexdigest() != evidence[path]["sha256"]:
        raise LakehouseError("Archived FX config checksum mismatch")
    fx = pd.read_csv(io.BytesIO(fx_blob), dtype=str, keep_default_na=False)
    if frame.duplicated("asset_id").any() or not {"asset_id", "yahoo_ticker", "currency", "calendar_id"} <= set(frame):
        raise LakehouseError("Archived asset registry missing identity/calendar contract")
    if selected_asset_ids:
        chosen = set(selected_asset_ids)
        frame = frame.loc[frame["asset_id"].isin(chosen)].copy()
        if set(frame["asset_id"]) != chosen:
            raise LakehouseError("Requested asset IDs not found in archived S3 plan")
    if frame.empty:
        raise LakehouseError("No eligible assets selected")
    needed_fx = set(frame["currency"].astype(str)) - {"USD"}
    fx = fx.loc[fx["currency"].isin(needed_fx)]
    if needed_fx - set(fx["currency"]):
        raise LakehouseError("Missing FX mappings in archived plan")
    assets = frame[["asset_id", "yahoo_ticker", "currency", "calendar_id"]].sort_values("asset_id")
    doc = {
        "schema": "alpha_edge_registry_v3_1", "source": "verified_v2_archived_plan",
        "archive_id": archive_id, "source_plan_sha256": info["plan_sha256"],
        "review_state": "historical_snapshot_unapproved",
        "assets": assets.to_dict("records"),
        "fx_pairs": fx[["currency", "provider_symbol", "direction"]].sort_values("currency").to_dict("records"),
    }
    source_hash = hashlib.sha256(json_bytes(doc)).hexdigest()
    target = f"{ROOT}/config/registry_{source_hash[:24]}.json"
    if not execute:
        return {"state": "dry_run", "target_key": target, "assets": len(assets), "writes": 0}
    put_json(s3, bucket, target, doc)
    return {"state": "registry_staged_for_review", "target_key": target, "assets": len(assets)}


def main() -> None:
    p = argparse.ArgumentParser(description="S3-only historical registry bootstrapping")
    p.add_argument("--bucket", required=True)
    p.add_argument("--archive-id", default="old-outputs-20261110")
    p.add_argument("--asset-ids", default="", help="Comma-separated canonical asset IDs; empty selects archived universe")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    import boto3
    print(json.dumps(build_registry(
        boto3.client("s3", region_name="eu-west-1"), a.bucket, archive_id=a.archive_id,
        selected_asset_ids=[x.strip() for x in a.asset_ids.split(",") if x.strip()] or None,
        execute=a.execute,
    ), indent=2))


if __name__ == "__main__":
    main()
