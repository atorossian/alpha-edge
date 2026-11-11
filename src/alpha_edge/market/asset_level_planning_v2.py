"""AEA87 S3-native identity crosswalk and supplementary acquisition preflight.

This command plans only. It never aliases asset IDs automatically, retrieves
prices, modifies old cohorts, or publishes production data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import boto3
import pandas as pd

from alpha_edge.market.universe_registry_v2 import (
    _get,
    _json_bytes,
    _records_bytes,
    _read_csv,
    _sha,
    publish_evidence,
)

ROOT = "dev/market/lake_v2/reconciliation"
PROVIDER_STATE = "market/state/v1/provider_symbol_by_asset_id.json"
CURRENCY_REQUIRED = {"asset_id", "yahoo_ticker", "currency", "calendar_id"}
VALID_FX_DIRECTION = {"direct", "inverse"}


class IdentityPlanError(ValueError):
    pass


def crosswalk(registry: pd.DataFrame, historical_symbols: dict[str, str]) -> pd.DataFrame:
    """Preserve source IDs; output candidate relationships, never approved mergers."""
    current = registry.loc[registry["current_universe_present"]].copy()
    by_symbol: dict[str, set[str]] = defaultdict(set)
    for row in current.to_dict("records"):
        symbol = str(row.get("current_yahoo_ticker") or "").strip()
        if symbol:
            by_symbol[symbol].add(row["asset_id"])

    historical_counts = Counter(
        value.strip() for row in registry.to_dict("records")
        if row["identity_source"] == "legacy_key_only"
        for value in [str(historical_symbols.get(row["asset_id"], ""))]
        if value.strip()
    )
    output = []
    for row in registry.to_dict("records"):
        if row["identity_source"] != "legacy_key_only":
            continue
        source_id = row["asset_id"]
        provider = str(historical_symbols.get(source_id, "")).strip()
        candidates = sorted(by_symbol.get(provider, [])) if provider else []
        if not provider:
            state = "missing_historical_provider_mapping"
        elif historical_counts[provider] > 1:
            state = "ambiguous_historical_symbol"
        elif len(candidates) > 1:
            state = "ambiguous_current_symbol"
        elif len(candidates) == 1:
            state = "single_symbol_match_requires_instrument_review"
        else:
            state = "historical_symbol_not_in_current_universe"
        output.append({
            "historical_asset_id": source_id,
            "historical_provider_symbol": provider,
            "candidate_current_asset_ids": candidates,
            "candidate_count": len(candidates),
            "state": state,
            "identity_confirmed": False,
            "canonical_asset_id": None,
            "evidence_key": PROVIDER_STATE,
            "source_price_verified": False,
        })
    return pd.DataFrame(output).sort_values("historical_asset_id").reset_index(drop=True)


def _explicit_metadata(row: dict, exchange_map: dict[str, str]) -> tuple[dict, list[str]]:
    # A provider-symbol match cannot supply instrument identity, exchange, or currency.
    # Only current explicit fields may satisfy the acquisition contract.
    currency = str(row.get("currency") or "").strip().upper()
    exchange = str(row.get("exchange") or "").strip()
    calendar = str(row.get("calendar_id") or "").strip()
    if not calendar and exchange:
        calendar = exchange_map.get(exchange, "")
    symbol = str(row.get("current_yahoo_ticker") or "").strip()
    missing = [
        field for field, value in (
            ("yahoo_ticker", symbol), ("currency", currency), ("calendar_id", calendar)
        ) if not value
    ]
    return {
        "asset_id": row["asset_id"],
        "yahoo_ticker": symbol,
        "currency": currency,
        "calendar_id": calendar,
        "exchange": exchange,
        "include": 1,
    }, missing


def supplementary_preflight(
    registry: pd.DataFrame, crosswalk_rows: pd.DataFrame,
    exchange_mapping: pd.DataFrame, fx_config: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create independently reviewable ready and blocked plans.

    No auto-dedup by ticker: a matching ticker is evidence for manual identity
    review, not authority to drop a historical instrument.
    """
    required = {"exchange", "calendar_id"}
    if not required.issubset(exchange_mapping):
        raise IdentityPlanError("Exchange mapping requires exchange,calendar_id")
    if not {"currency", "provider_symbol", "direction"}.issubset(fx_config):
        raise IdentityPlanError("FX configuration missing required columns")
    if exchange_mapping["exchange"].duplicated().any():
        raise IdentityPlanError("Ambiguous exchange calendar mapping")
    if fx_config["currency"].duplicated().any():
        raise IdentityPlanError("Ambiguous FX currency mapping")
    exchange_map = dict(zip(exchange_mapping["exchange"], exchange_mapping["calendar_id"]))
    fx_map = fx_config.set_index("currency").to_dict("index")
    crosswalk_current: set[str] = set()
    for x in crosswalk_rows.to_dict("records"):
        crosswalk_current.update(x["candidate_current_asset_ids"])
    planned_symbols = set(
        registry.loc[registry["in_original_plan"], "historical_yahoo_ticker"].dropna().astype(str)
    ) - {""}
    new_rows = registry.loc[
        registry["current_universe_present"] & ~registry["in_original_plan"]
    ]
    new_symbol_counts = Counter(new_rows["current_yahoo_ticker"].dropna().astype(str))
    ready, blocked = [], []
    for row in registry.to_dict("records"):
        if not row["current_universe_present"] or row["in_original_plan"]:
            continue
        meta, missing = _explicit_metadata(row, exchange_map)
        if row["asset_id"] in crosswalk_current:
            missing.append("historical_asset_id_collision_requires_review")
        if meta["yahoo_ticker"] in planned_symbols:
            missing.append("original_plan_provider_symbol_collision")
        if meta["yahoo_ticker"] and new_symbol_counts[meta["yahoo_ticker"]] > 1:
            missing.append("supplementary_provider_symbol_collision")
        ccy = meta["currency"]
        if ccy and ccy != "USD":
            fx = fx_map.get(ccy)
            if not fx or not str(fx["provider_symbol"]).endswith("=X") or fx["direction"] not in VALID_FX_DIRECTION:
                missing.append("approved_fx_mapping")
        meta["preflight_state"] = "blocked" if missing else "configuration_ready_not_acquired"
        meta["blockers"] = sorted(set(missing))
        meta["historical_identity_verified"] = False
        meta["historical_prices_approved"] = False
        if missing:
            blocked.append(meta)
        else:
            ready.append(meta)
    return pd.DataFrame(ready), pd.DataFrame(blocked)


def recovery_inventory(registry: pd.DataFrame, crosswalk_rows: pd.DataFrame) -> pd.DataFrame:
    """Convert five missing cohorts to original-asset-id singleton recovery tasks."""
    alias_candidates = {current for row in crosswalk_rows.to_dict("records")
                        for current in row["candidate_current_asset_ids"]}
    tasks = []
    for row in registry.to_dict("records"):
        if row["research_cohort_state"] != "missing":
            continue
        tasks.append({
            "asset_id": row["asset_id"],
            "original_cohort": row["original_cohort"],
            "original_provider_symbol": row["historical_yahoo_ticker"],
            "current_provider_symbol": row["current_yahoo_ticker"],
            "identity_review_required": (
                row["current_universe_present"]
                and row["historical_yahoo_ticker"] != row["current_yahoo_ticker"]
            ) or row["asset_id"] in alias_candidates,
            "acquisition_scope": "one_asset_immutable_research_candidate",
            "execution_requested": False,
            "expected_session_coverage_verified": False,
            "historical_prices_approved": False,
        })
    return pd.DataFrame(tasks)


def coverage_queue(registry: pd.DataFrame) -> pd.DataFrame:
    """All old cohorts require per-asset evidence, including dates and corporate actions."""
    included = registry.loc[registry["in_original_plan"]].copy()
    rows = []
    for row in included.to_dict("records"):
        rows.append({
            "asset_id": row["asset_id"],
            "original_cohort": row["original_cohort"],
            "research_cohort_state": row["research_cohort_state"],
            "price_source_verified": False,
            "listing_start_verified": False,
            "delisting_end_verified": False,
            "corporate_actions_verified": False,
            "calendar_sessions_verified": False,
            "unresolved_observation_gaps": None,
            "production_eligible": False,
        })
    return pd.DataFrame(rows)


def build_plan_bundle(
    registry: pd.DataFrame, historical_symbols: dict[str, str],
    exchange_mapping: pd.DataFrame, fx_config: pd.DataFrame,
    source_fingerprints: dict,
) -> tuple[dict[str, bytes], dict]:
    mapping = crosswalk(registry, historical_symbols)
    ready, blocked = supplementary_preflight(registry, mapping, exchange_mapping, fx_config)
    recovery = recovery_inventory(registry, mapping)
    coverage = coverage_queue(registry)
    artifacts = {
        "identity_crosswalk.jsonl": _records_bytes(mapping),
        "supplementary_ready.jsonl": _records_bytes(ready),
        "supplementary_blocked.jsonl": _records_bytes(blocked),
        "missing_cohort_asset_recovery.jsonl": _records_bytes(recovery),
        "historical_coverage_queue.jsonl": _records_bytes(coverage),
    }
    counts = {
        "historical_identity_candidates": len(mapping),
        "identity_states": dict(sorted(Counter(mapping["state"]).items())),
        "supplementary_ready": len(ready),
        "supplementary_blocked": len(blocked),
        "missing_cohort_asset_tasks": len(recovery),
        "historical_coverage_tasks": len(coverage),
    }
    identity = {
        "schema": "aea87_asset_level_planning_v1",
        "source_fingerprints": source_fingerprints,
        "artifact_sha256": {k: _sha(v) for k, v in sorted(artifacts.items())},
    }
    run_id = _sha(_json_bytes(identity))
    manifest = {
        "schema": "aea87_asset_level_planning_v1",
        "run_id": run_id,
        "source_fingerprints": source_fingerprints,
        "artifact_sha256": identity["artifact_sha256"],
        "counts": counts,
        "identity_merges_approved": 0,
        "acquisitions_executed": 0,
        "market_data_approved": False,
    }
    artifacts["manifest.json"] = _json_bytes(manifest)
    return artifacts, manifest


def run(
    s3, bucket: str, reconciliation_id: str,
    exchange_mapping_bytes: bytes, fx_bytes: bytes, *, publish: bool = False
) -> dict:
    if len(reconciliation_id) != 64 or any(ch not in "0123456789abcdef" for ch in reconciliation_id):
        raise IdentityPlanError("Invalid reconciliation id")
    base = f"{ROOT}/run-{reconciliation_id}"
    source_manifest_raw = _get(s3, bucket, base + "/manifest.json")
    source_manifest = json.loads(source_manifest_raw)
    if source_manifest.get("run_id") != reconciliation_id:
        raise IdentityPlanError("Reconciliation manifest identity mismatch")
    registry_raw = _get(s3, bucket, base + "/asset_registry.jsonl")
    if _sha(registry_raw) != source_manifest["artifact_sha256"]["asset_registry.jsonl"]:
        raise IdentityPlanError("Reconciliation registry hash mismatch")
    historical_raw = _get(s3, bucket, PROVIDER_STATE)
    symbols = json.loads(historical_raw)
    registry = pd.DataFrame(json.loads(line) for line in registry_raw.splitlines() if line)
    fingerprints = {
        "reconciliation_manifest_sha256": _sha(source_manifest_raw),
        "reconciliation_id": reconciliation_id,
        "historical_symbol_map_sha256": _sha(historical_raw),
        "exchange_map_sha256": _sha(exchange_mapping_bytes),
        "fx_config_sha256": _sha(fx_bytes),
    }
    artifacts, manifest = build_plan_bundle(
        registry, symbols, _read_csv(exchange_mapping_bytes), _read_csv(fx_bytes), fingerprints
    )
    # Explicit namespace; never mutate shared module state.
    if publish:
        location = publish_evidence(
            s3, bucket, artifacts, manifest, root="dev/market/lake_v2/asset_plans"
        )
    else:
        location = {"status": "dry_run_no_writes"}
    return {**manifest["counts"], "run_id": manifest["run_id"], "publication": location}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--reconciliation-id", required=True)
    parser.add_argument("--exchange-map", type=Path, default=Path("data/quality/market_lake_v2/exchange_calendar_mapping_v2.csv"))
    parser.add_argument("--fx-config", type=Path, default=Path("data/quality/market_lake_v2/pilot_fx_registry_v2.csv"))
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    print(json.dumps(run(
        s3, args.bucket, args.reconciliation_id,
        args.exchange_map.read_bytes(), args.fx_config.read_bytes(), publish=args.publish
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
