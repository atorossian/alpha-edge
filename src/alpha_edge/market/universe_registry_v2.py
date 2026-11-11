"""Read-only current/historical market universe reconciliation for AEA87.

Current eligibility comes from the assembled Git-tracked universe. Historical
membership comes from immutable S3 plan evidence, not from legacy v1 prices.
This command never writes files, modifies S3, or publishes a release.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.market.archive_local_outputs_v2 import archive_prefix

FILES = (
    "universe.csv",
    "asset_excluded.csv",
    "universe_overrides.csv",
    "universe_manual_additions.csv",
    "ingest_force_refresh.csv",
)
PLAN_DIR = "full_universe_preflight_v2/first_review"


class UniverseRegistryError(ValueError):
    """Untrusted or inconsistent registry input."""


def _read_csv(blob: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(blob), dtype=str, keep_default_na=False)


def _required(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    if missing := columns - set(frame.columns):
        raise UniverseRegistryError(f"{name} missing columns: {sorted(missing)}")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get(s3, bucket: str, key: str) -> bytes:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            raise UniverseRegistryError(f"Missing S3 evidence: {key}") from exc
        raise


def load_archived_plan(s3, bucket: str, archive_id: str, relative_dir: str = PLAN_DIR) -> tuple[pd.DataFrame, dict]:
    """Read and SHA-verify archived, immutable preflight plan and every cohort CSV."""
    prefix = archive_prefix(archive_id)
    index = json.loads(_get(s3, bucket, f"{prefix}/_verified_inventory.json"))
    if index.get("schema") != "alpha_edge_local_output_archive_v2_1":
        raise UniverseRegistryError("Invalid archive inventory schema")
    items = {row["path"]: row["sha256"] for row in index["files"]}
    base = relative_dir.strip("/")

    def archived(name: str) -> bytes:
        relative = f"{base}/{name}"
        if relative not in items:
            raise UniverseRegistryError(f"Missing archived plan entry: {relative}")
        body = _get(s3, bucket, f"{prefix}/files/{relative}")
        if _sha(body) != items[relative]:
            raise UniverseRegistryError(f"Archived plan SHA mismatch: {relative}")
        return body

    manifest_bytes = archived("manifest.json")
    plan = json.loads(manifest_bytes)
    if plan.get("schema") != "alpha_edge_lake_v2_universe_preflight_1":
        raise UniverseRegistryError("Unexpected preflight plan schema")
    count = plan.get("cohorts")
    if not isinstance(count, int) or not 1 <= count <= 10000:
        raise UniverseRegistryError("Invalid planned cohort count")
    expected_names = {"fx_config.csv"} | {f"cohort_{i:04d}.csv" for i in range(1, count + 1)}
    if set(plan.get("files_sha256", {})) != expected_names:
        raise UniverseRegistryError("Preflight file inventory mismatch")
    assets = []
    for name in sorted(expected_names):
        body = archived(name)
        if _sha(body) != plan["files_sha256"][name]:
            raise UniverseRegistryError(f"Preflight SHA mismatch: {name}")
        if name.startswith("cohort_"):
            cohort = int(name[7:11])
            frame = _read_csv(body)
            _required(frame, {"asset_id", "yahoo_ticker"}, name)
            frame["original_cohort"] = cohort
            assets.append(frame)
    result = pd.concat(assets, ignore_index=True)
    if result["asset_id"].eq("").any() or result["asset_id"].duplicated().any():
        raise UniverseRegistryError("Duplicate/missing asset IDs in original plan")
    if len(result) != plan["assets_included"]:
        raise UniverseRegistryError("Original plan asset count mismatch")
    return result, {"plan_sha256": _sha(manifest_bytes), "expected_cohorts": count, "assets": len(result)}


def cohort_commit_presence(s3, bucket: str, plan_sha: str, end: str, count: int) -> dict[int, str]:
    """Presence and marker identity only; full artifact verification remains a separate gate."""
    prefix = f"dev/market/lake_v2/backfills/{plan_sha[:24]}/end-{end}"
    states = {}
    for index in range(1, count + 1):
        key = f"{prefix}/cohort-{index:04d}/_committed.json"
        try:
            blob = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                states[index] = "missing"
                continue
            raise
        marker = json.loads(blob)
        valid = (
            marker.get("schema") == "alpha_edge_s3_backfill_commit_v2_1"
            and marker.get("plan_manifest_sha256") == plan_sha
            and marker.get("prefix") == key.removesuffix("/_committed.json")
            and marker.get("end_exclusive") == end
        )
        states[index] = "marker_present_unverified" if valid else "invalid_marker"
    return states


def reconcile(
    config: dict[str, pd.DataFrame],
    historical: pd.DataFrame,
    commit_states: dict[int, str],
    *,
    legacy_asset_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Reconcile statuses; never infer delisting or approve historical prices."""
    current = config["universe.csv"].copy()
    excluded = config["asset_excluded.csv"].copy()
    overrides = config["universe_overrides.csv"].copy()
    manual = config["universe_manual_additions.csv"].copy()
    refresh = config["ingest_force_refresh.csv"].copy()
    _required(current, {"row_id", "asset_id", "ticker", "yahoo_ticker", "include"}, "universe.csv")
    _required(excluded, {"target_row_id", "ticker", "reason"}, "asset_excluded.csv")
    _required(overrides, {"target_row_id", "ticker", "yahoo_ticker", "lock_yahoo_ticker"}, "universe_overrides.csv")
    _required(manual, {"ticker"}, "universe_manual_additions.csv")
    _required(refresh, {"ticker"}, "ingest_force_refresh.csv")
    if current["asset_id"].eq("").any() or current["asset_id"].duplicated().any():
        raise UniverseRegistryError("Missing/duplicate asset ID in current universe")
    if current["row_id"].eq("").any() or current["row_id"].duplicated().any():
        raise UniverseRegistryError("Missing/duplicate row ID in current universe")
    if not current["include"].isin({"0", "1"}).all():
        raise UniverseRegistryError("Current include must be 0 or 1")
    if historical["asset_id"].duplicated().any():
        raise UniverseRegistryError("Duplicate asset ID in archived plan")

    reasons: dict[str, str] = {}
    anomalies: list[str] = []
    for row in excluded.to_dict("records"):
        rid = row["target_row_id"].strip()
        matches = current.loc[current["row_id"].eq(rid)] if rid else current.loc[current["ticker"].eq(row["ticker"])]
        if len(matches) != 1:
            anomalies.append(f"exclusion_unmatched_or_ambiguous:{rid or row['ticker']}")
            continue
        asset = matches.iloc[0]
        if asset["include"] != "0":
            anomalies.append(f"exclusion_still_included:{asset['asset_id']}")
        reasons[asset["asset_id"]] = row["reason"]

    for row in overrides.to_dict("records"):
        rid = row["target_row_id"].strip()
        matches = current.loc[current["row_id"].eq(rid)] if rid else current.loc[current["ticker"].eq(row["ticker"])]
        if len(matches) != 1:
            anomalies.append(f"override_unmatched_or_ambiguous:{rid or row['ticker']}")
            continue
        actual = matches.iloc[0]
        if row["lock_yahoo_ticker"] == "1" and row["yahoo_ticker"] and actual["yahoo_ticker"] != row["yahoo_ticker"]:
            anomalies.append(f"locked_symbol_conflict:{actual['asset_id']}")

    for row in manual.to_dict("records"):
        if not current["ticker"].eq(row["ticker"]).any():
            anomalies.append(f"manual_addition_missing:{row['ticker']}")
    for row in refresh.to_dict("records"):
        if not current["ticker"].eq(row["ticker"]).any():
            anomalies.append(f"refresh_target_missing:{row['ticker']}")

    history_by_id = historical.set_index("asset_id")
    current_by_id = current.set_index("asset_id")
    ids = sorted(set(current_by_id.index) | set(history_by_id.index) | (legacy_asset_ids or set()))
    rows = []
    for asset_id in ids:
        exists = asset_id in current_by_id.index
        planned = asset_id in history_by_id.index
        row = current_by_id.loc[asset_id] if exists else None
        original = history_by_id.loc[asset_id] if planned else None
        current_symbol = row["yahoo_ticker"] if exists else ""
        original_symbol = original["yahoo_ticker"] if planned else ""
        state = commit_states.get(int(original["original_cohort"]), "not_checked") if planned else "not_planned"
        issues = []
        if exists and row["include"] == "0" and asset_id not in reasons:
            issues.append("missing_exclusion_reason")
        if planned and exists and original_symbol != current_symbol:
            issues.append("provider_symbol_changed_review_history")
        if not exists and planned:
            issues.append("absent_from_current_universe")
        rows.append(
            {
                "asset_id": asset_id,
                "ticker": row["ticker"] if exists else "",
                "current_yahoo_ticker": current_symbol,
                "historical_yahoo_ticker": original_symbol,
                "current_eligible_config": bool(exists and row["include"] == "1"),
                "current_exclusion_reason": reasons.get(asset_id, ""),
                "in_original_plan": planned,
                "original_cohort": int(original["original_cohort"]) if planned else None,
                "research_cohort_state": state,
                "legacy_v1_history_present_unverified": asset_id in (legacy_asset_ids or set()),
                "historical_retention_required": planned or asset_id in (legacy_asset_ids or set()),
                "historical_price_approved": False,
                "issues": ";".join(issues),
            }
        )
    registry = pd.DataFrame(rows)
    summary = {
        "schema": "aea87_universe_reconciliation_readonly_v1",
        "current_assets": len(current),
        "currently_included": int(current["include"].eq("1").sum()),
        "currently_excluded": int(current["include"].eq("0").sum()),
        "original_plan_assets": len(historical),
        "historical_or_legacy_assets": int(registry["historical_retention_required"].sum()),
        "historical_only_assets": int(
            (~registry["asset_id"].isin(current["asset_id"]) & registry["historical_retention_required"]).sum()
        ),
        "legacy_only_assets": int(
            (registry["legacy_v1_history_present_unverified"] & ~registry["in_original_plan"]).sum()
        ),
        "cohort_states": dict(sorted(Counter(commit_states.values()).items())),
        "configuration_issues": sorted(set(anomalies)),
        "asset_issue_counts": dict(
            sorted(Counter(issue for s in registry["issues"] for issue in s.split(";") if issue).items())
        ),
        "historical_prices_approved": False,
        "s3_writes": 0,
    }
    return registry, summary


def list_legacy_asset_ids(s3, bucket: str) -> set[str]:
    """Inventory-only: v1 keys can identify historical candidates, never validate prices."""
    prefix = "market/ohlcv_usd/v1/"
    paginator = s3.get_paginator("list_objects_v2")
    ids = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            suffix = item["Prefix"][len(prefix) :].removesuffix("/")
            if suffix.startswith("asset_id="):
                ids.add(suffix[len("asset_id=") :])
    return ids


def run(
    s3, bucket: str, archive_id: str, config_bytes: dict[str, bytes], *, end: str, include_legacy: bool = True
) -> dict:
    if set(config_bytes) != set(FILES):
        raise UniverseRegistryError("All five universe configuration files are required")
    config = {name: _read_csv(config_bytes[name]) for name in FILES}
    plan, info = load_archived_plan(s3, bucket, archive_id)
    states = cohort_commit_presence(s3, bucket, info["plan_sha256"], end, info["expected_cohorts"])
    legacy = list_legacy_asset_ids(s3, bucket) if include_legacy else set()
    rows, report = reconcile(config, plan, states, legacy_asset_ids=legacy)
    report["inputs_sha256"] = {name: _sha(data) for name, data in sorted(config_bytes.items())}
    report["original_plan"] = info
    report["end_exclusive"] = end
    report["legacy_inventory_scanned"] = include_legacy
    report["asset_findings"] = rows.loc[
        rows["issues"].ne("") | rows["research_cohort_state"].eq("missing")
    ].to_dict("records")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--archive-id", default="old-outputs-20261110")
    parser.add_argument("--end", default="2026-11-10", help="Exclusive acquisition end date")
    parser.add_argument(
        "--universe-dir", type=Path, default=Path("data/universe"), help="Git-tracked input configuration only"
    )
    parser.add_argument("--skip-legacy-inventory", action="store_true")
    parser.add_argument("--region", default="eu-west-1")
    args = parser.parse_args()
    inputs = {name: (args.universe_dir / name).read_bytes() for name in FILES}
    s3 = boto3.client("s3", region_name=args.region)
    print(
        json.dumps(
            run(s3, args.bucket, args.archive_id, inputs, end=args.end, include_legacy=not args.skip_legacy_inventory),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
