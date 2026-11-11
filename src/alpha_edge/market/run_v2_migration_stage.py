"""Run all four AEA-87 v2 migration steps without re-fetching Yahoo.

1. Standardize the 125 missing-cohort slots from immutable recovered Bronze.
2. Reprocess the already acquired EURUSD=X Bronze with corrected FX weekdays.
3. Reconcile the original and current universe, retaining excluded histories.
4. Publish an explicitly blocked non-production combined release inventory.

All processing writes only isolated dev/market/lake_v2 research candidates.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import boto3

from alpha_edge.market.build_release_inventory_v2 import run as release_inventory
from alpha_edge.market.reprocess_committed_bronze_v2 import run as reprocess_committed
from alpha_edge.market.standardize_recovered_assets_v2 import run as recover_candidates
from alpha_edge.market.retain_legacy_history_v2 import run as retain_legacy


def run(
    s3, bucket: str, *, universe_csv: Path, fx_config: Path,
    archive_id: str = "old-outputs-20261110", end: str = "2026-11-10",
    execute: bool = False,
) -> dict:
    stages = {}
    if execute:
        candidates = []
        for start in range(0, 125, 25):
            batch = recover_candidates(
                s3, bucket, archive_id,
                original_end=end, start_index=start,
                max_assets=25, execute=True,
            )
            candidates.extend(batch["results"])
        stages["recovered_assets"] = {
            "attempted": len(candidates),
            "outcomes": dict(Counter(item["status"] for item in candidates)),
            "failures": [
                item for item in candidates if item["status"] in {
                    "processing_failed", "source_missing", "source_not_acquired",
                }
            ],
        }
    else:
        stages["recovered_assets"] = recover_candidates(
            s3, bucket, archive_id, original_end=end,
            start_index=0, max_assets=25, execute=False,
        )
    stages["eurusd"] = reprocess_committed(
        s3, bucket, universe_csv=universe_csv,
        fx_csv=fx_config, execute=execute,
    )
    stages["historical_retention"] = retain_legacy(s3, bucket, execute=execute)
    stages["release_inventory"] = release_inventory(
        s3, bucket, archive_id=archive_id,
        universe_csv=universe_csv, original_end=end, execute=execute,
    )
    stages["production_pointer_changed"] = False
    stages["legacy_v1_modified"] = False
    stages["yahoo_downloads"] = 0
    stages["production_eligible"] = False
    return stages


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--archive-id", default="old-outputs-20261110")
    p.add_argument("--end", default="2026-11-10")
    p.add_argument("--universe", type=Path, default=Path("data/universe/universe.csv"))
    p.add_argument("--fx-config", type=Path, default=Path("data/quality/market_lake_v2/pilot_fx_registry_v2.csv"))
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    result = run(
        boto3.client("s3", region_name=args.region), args.bucket,
        universe_csv=args.universe, fx_config=args.fx_config,
        archive_id=args.archive_id, end=args.end, execute=args.execute,
    )
    print(json.dumps(result, indent=2))
    if args.execute and (
        result["recovered_assets"]["outcomes"].get("processing_failed", 0)
        or any(x["status"] == "retention_failed" for x in result["historical_retention"]["results"])
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
