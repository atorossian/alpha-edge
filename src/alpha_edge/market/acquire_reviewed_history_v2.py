"""Acquire full-history v2 research candidates for reviewed, eligible instruments.

Uses the existing preflight and S3 backfill pipeline unchanged. The cohort is
a single asset and all local files are ephemeral. No v1 observations are read,
and no production pointer is activated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path

import boto3
import pandas as pd

from alpha_edge.market.preflight_universe_v2 import prepare
from alpha_edge.market.run_s3_full_history_backfill_v2 import run as run_backfill
from alpha_edge.market.run_s3_full_history_backfill_v2 import verify_commit


class HistoricalAcquisitionError(ValueError):
    pass


def reviewed_asset(
    universe_csv: Path, exchange_map_csv: Path, *, asset_id: str, expected_symbol: str
) -> tuple[pd.DataFrame, dict]:
    """Select the exact canonical ID; no provider-symbol-only selection."""
    raw = universe_csv.read_bytes()
    universe = pd.read_csv(universe_csv, dtype=str).fillna("")
    matching = universe.loc[universe["asset_id"].eq(asset_id)].copy()
    if len(matching) != 1:
        raise HistoricalAcquisitionError(f"Expected exactly one retained asset: {asset_id}")
    asset = matching.iloc[0]
    if asset["yahoo_ticker"] != expected_symbol or asset["include"] != "1":
        raise HistoricalAcquisitionError("Asset is not included with its reviewed provider symbol")
    other_active = universe.loc[
        universe["yahoo_ticker"].eq(expected_symbol)
        & universe["asset_id"].ne(asset_id)
        & universe["include"].eq("1")
    ]
    if not other_active.empty:
        raise HistoricalAcquisitionError("Another included asset has the same Yahoo provider symbol")
    exchange = str(asset.get("exchange", "")).strip()
    currency = str(asset.get("currency", "")).strip().upper()
    calendar = str(asset.get("calendar_id", "")).strip()
    if not calendar:
        mapping = pd.read_csv(exchange_map_csv, dtype=str).fillna("")
        rows = mapping.loc[mapping["exchange"].eq(exchange), "calendar_id"].drop_duplicates()
        if len(rows) != 1:
            raise HistoricalAcquisitionError(f"Missing or ambiguous exchange calendar for {exchange}")
        calendar = str(rows.iloc[0])
    if not currency or not calendar:
        raise HistoricalAcquisitionError("Missing explicit currency or verified exchange calendar")
    configured = pd.DataFrame([{
        "asset_id": asset_id,
        "yahoo_ticker": expected_symbol,
        "currency": currency,
        "calendar_id": calendar,
        "include": 1,
    }])
    return configured, {
        "asset_id": asset_id,
        "row_id": asset["row_id"],
        "symbol": expected_symbol,
        "currency": currency,
        "calendar_id": calendar,
        "universe_csv_sha256": hashlib.sha256(raw).hexdigest(),
        "exchange_mapping_sha256": hashlib.sha256(exchange_map_csv.read_bytes()).hexdigest(),
    }


def run(
    *, s3, bucket: str, universe_csv: Path, exchange_map_csv: Path, fx_csv: Path,
    asset_id: str, expected_symbol: str, end: str, execute: bool = False,
    minimum_expected_first_date: str | None = None,
) -> dict:
    """Build one-source-asset immutable plan and optionally run existing v2 pipeline."""
    if date.fromisoformat(end) <= date(1980, 1, 1):
        raise HistoricalAcquisitionError("Invalid exclusive end")
    if minimum_expected_first_date:
        if date.fromisoformat(minimum_expected_first_date) >= date.fromisoformat(end):
            raise HistoricalAcquisitionError("Expected history start must precede exclusive end")
    selected, source = reviewed_asset(
        universe_csv, exchange_map_csv, asset_id=asset_id, expected_symbol=expected_symbol
    )
    with tempfile.TemporaryDirectory(prefix="aea87_reviewed_acquisition_") as temp:
        work = Path(temp)
        source_path = work / "selected_universe.csv"
        selected.to_csv(source_path, index=False)
        plan_dir = work / "plan"
        preflight = prepare(
            universe_csv=source_path, fx_csv=fx_csv,
            output=plan_dir, cohort_size=1,
        )
        # Extend the existing preflight manifest with pinned Git configuration
        # provenance. The backfill runner accepts additional manifest fields and
        # hashes the resulting bytes as the immutable plan identity.
        manifest_path = plan_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["reviewed_source"] = source
        manifest["end_exclusive"] = end
        manifest["minimum_expected_first_date"] = minimum_expected_first_date
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        plan_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        prefix = f"dev/market/lake_v2/backfills/{plan_sha[:24]}/end-{end}/cohort-0001"
        if not execute:
            return {
                "status": "ready_for_full_history_acquisition",
                "source": source,
                "preflight": preflight["status"],
                "plan_sha256": plan_sha,
                "candidate_s3_prefix": f"s3://{bucket}/{prefix}",
                "minimum_expected_first_date": minimum_expected_first_date,
                "s3_writes": 0,
            }
        outcome = run_backfill(
            plan_dir=plan_dir, bucket=bucket, end=end, execute=True,
            start_cohort=1, max_cohorts=1, s3=s3,
        )
        if outcome["s3_verified_cohorts"] != 1:
            return {
                "status": "acquisition_incomplete",
                "source": source,
                "outcome": outcome,
                "production_eligible": False,
            }
        commit = verify_commit(s3, bucket, prefix, plan_sha)
        if commit is None:
            raise HistoricalAcquisitionError("Research candidate has no verified S3 commit")
        manifest_key = f"{prefix}/files/execution/bronze_batches/s3bf-0001-0001/manifest.json"
        manifest_bytes = s3.get_object(Bucket=bucket, Key=manifest_key)["Body"].read()
        # verify_commit already verifies this object against the committed inventory.
        source_manifest = json.loads(manifest_bytes)
        matches = [
            item for item in source_manifest["sources"]
            if item.get("asset_id") == asset_id and item.get("provider_symbol") == expected_symbol
        ]
        if len(matches) != 1:
            raise HistoricalAcquisitionError("Committed Bronze provenance does not identify reviewed asset")
        first = matches[0]["first_observed_session"]
        shortfall = (
            bool(minimum_expected_first_date)
            and date.fromisoformat(first) > date.fromisoformat(minimum_expected_first_date)
        )
        return {
            "status": "published_research_candidate_with_coverage_shortfall"
            if shortfall else "published_research_candidate",
            "source": source,
            "plan_sha256": plan_sha,
            "first_observed_session": first,
            "last_observed_session": matches[0]["last_observed_session"],
            "source_observations": matches[0]["row_count"],
            "expected_first_date": minimum_expected_first_date,
            "earlier_history_shortfall": shortfall,
            "commit_key": f"s3://{bucket}/{prefix}/_committed.json",
            "production_eligible": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--minimum-expected-first-date")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--universe", type=Path, default=Path("data/universe/universe.csv"))
    parser.add_argument(
        "--exchange-map", type=Path,
        default=Path("data/quality/market_lake_v2/exchange_calendar_mapping_v2.csv"),
    )
    parser.add_argument(
        "--fx-config", type=Path,
        default=Path("data/quality/market_lake_v2/pilot_fx_registry_v2.csv"),
    )
    parser.add_argument("--region", default="eu-west-1")
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    result = run(
        s3=s3, bucket=args.bucket, universe_csv=args.universe,
        exchange_map_csv=args.exchange_map, fx_csv=args.fx_config,
        asset_id=args.asset_id, expected_symbol=args.symbol,
        end=args.end, execute=args.execute,
        minimum_expected_first_date=args.minimum_expected_first_date,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    if result["status"] == "acquisition_incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
