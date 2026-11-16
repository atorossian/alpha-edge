"""AEA-91: construct a review-only, immutable S3 pilot registry from v2 evidence.

Never infer an effective date for a ticker rename. An archived identity and
a current one are separate observations until independently reconciled.
The pilot registry is NOT an approved or active production registry.
"""

from __future__ import annotations

import argparse
import csv
import io
import json

from alpha_edge.market.archive_local_outputs_v2 import archive_prefix
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT,
    LakehouseError,
    digest,
    json_bytes,
    put_json,
    read_bytes,
    read_json,
)
from alpha_edge.market.universe_registry_v2 import load_archived_plan

PILOT_SCHEMA = "alpha_edge_registry_v3_2"
ARCHIVE_ID = "old-outputs-20261110"
INVENTORY_RUN = "run-873f01265f3fc8a19741bcb48d2aebc0db3de565abd767b11f3ac9907a85a434"
INVENTORY_PREFIX = f"dev/market/lake_v2/release_candidates/{INVENTORY_RUN}"
GIT_UNIVERSE_REF = "integration/aea87-market-data-lake-v2"
GIT_UNIVERSE_COMMIT = "cfb36a1deb4eaa58615018702cdf4cbee6a79cf2"

# The current ticker/calendar/quote-unit observations were cross-checked
# against the Git-tracked universe. They are bootstrap evidence, not
# runtime source-of-truth: downstream readers must load from approved S3.
# asset_id, symbol, currency, calendar, timezone, class, quote unit
PILOT_POLICY = (
    ("EQH7f9e4349dabafdd9", "SPY", "USD", "NYSE", "America/New_York", "ETF", "USD"),
    ("EQH4561ad9433b2a662", "GLD", "USD", "NYSE", "America/New_York", "ETF", "USD"),
    ("CRYPTO:BTC-USD", "BTC-USD", "USD", "24/7", "UTC", "CRYPTO", "USD"),
    ("EQH0f232d452fb06a46", "RENT3.SA", "BRL", "BMF", "America/Sao_Paulo", "EQUITY", "BRL"),
    ("EQH11f97bdc8d8dfe85", "TRAXIONA.MX", "MXN", "XMEX", "America/Mexico_City", "EQUITY", "MXN"),
    ("EQH2325a2fe8044198c", "LOG.MC", "EUR", "XMAD", "Europe/Madrid", "EQUITY", "EUR"),
    ("EQH2834655e74af1b24", "REY.MI", "EUR", "XMIL", "Europe/Rome", "EQUITY", "EUR"),
    ("EQHe874aca04f607837", "EURUSD=X", "USD", "FOREX", "UTC", "FX", "USD"),
)
PILOT_IDS = tuple(row[0] for row in PILOT_POLICY)


def _load_inventory(s3, bucket: str) -> tuple[dict[str, dict], dict]:
    """Verify the committed v2 inventory, its contract, and every source checksum."""
    prefix = INVENTORY_PREFIX
    marker = read_json(s3, bucket, prefix + "/_committed.json")
    if (
        marker.get("schema") != "alpha_edge_s3_backfill_commit_v2_1"
        or marker.get("prefix") != prefix
        or marker.get("storage_state") != "verified_research_candidate"
        or marker.get("production_eligible") is not False
    ):
        raise LakehouseError("Unsupported or unexpectedly promoted v2 inventory")
    expected = {"asset_inventory.jsonl", "source_contract.json", "summary.json"}
    entries = marker.get("files")
    if not isinstance(entries, list) or {x.get("path") for x in entries} != expected:
        raise LakehouseError("Incomplete v2 inventory commit")
    blobs = {}
    for entry in entries:
        key = prefix + "/files/" + entry["path"]
        blob = read_bytes(s3, bucket, key)
        if digest(blob) != entry.get("sha256") or len(blob) != entry.get("size"):
            raise LakehouseError(f"Changed v2 inventory source: {key}")
        blobs[entry["path"]] = blob
    contract = json.loads(blobs["source_contract.json"])
    summary = json.loads(blobs["summary.json"])
    if (
        contract.get("schema") != "aea87_release_inventory_v2_1"
        or contract.get("inventory_sha256") != digest(blobs["asset_inventory.jsonl"])
        or summary.get("current_universe_sha256") != contract.get("current_universe_sha256")
        or summary.get("assets") != 2207
        or summary.get("current_eligible") != 2078
        or summary.get("release_state") != "blocked_pending_verified_candidates_and_external_evidence"
    ):
        raise LakehouseError("v2 release inventory evidence contract changed")
    records = [json.loads(line) for line in blobs["asset_inventory.jsonl"].splitlines() if line.strip()]
    if len(records) != summary["assets"]:
        raise LakehouseError("v2 release inventory record count does not reconcile")
    ids = [x["asset_id"] for x in records]
    if len(ids) != len(set(ids)):
        raise LakehouseError("Duplicate asset identities in v2 release inventory")
    eligible = sum(x.get("current_eligible") is True for x in records)
    if eligible != summary["current_eligible"]:
        raise LakehouseError("Inventory eligibility flags do not reconcile")
    return {x["asset_id"]: x for x in records}, contract


def _archived_fx(s3, bucket: str, archive_id: str) -> list[dict]:
    root = archive_prefix(archive_id)
    index = read_json(s3, bucket, f"{root}/_verified_inventory.json")
    if index.get("schema") != "alpha_edge_local_output_archive_v2_1":
        raise LakehouseError("Unexpected archived plan inventory version")
    path = "full_universe_preflight_v2/first_review/fx_config.csv"
    matches = [f for f in index["files"] if f["path"] == path]
    if len(matches) != 1:
        raise LakehouseError("Missing/duplicated archived FX configuration")
    body = read_bytes(s3, bucket, f"{root}/files/{path}")
    if digest(body) != matches[0]["sha256"]:
        raise LakehouseError("Historical FX configuration checksum failure")
    records = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
    return [{k: row[k] for k in ("currency", "provider_symbol", "direction")} for row in records]


def build_pilot_registry(s3, bucket: str, *, archive_id: str = ARCHIVE_ID) -> dict:
    """Fail closed on eight-asset source mismatch or an unreviewed eligibility flag."""
    archived, plan = load_archived_plan(s3, bucket, archive_id)
    inventory, source = _load_inventory(s3, bucket)
    if source["original_plan_sha256"] != plan["plan_sha256"]:
        raise LakehouseError("v2 release inventory does not reference the archived plan")
    archived_by_id = archived.set_index("asset_id", verify_integrity=True)
    assets = []
    unresolved = []
    for asset_id, symbol, currency, calendar, tz, asset_class, quote_unit in PILOT_POLICY:
        if asset_id not in archived_by_id.index or asset_id not in inventory:
            raise LakehouseError(f"{asset_id}: missing historical or current evidence")
        a = archived_by_id.loc[asset_id]
        current = inventory[asset_id]
        if current.get("current_eligible") is not True:
            raise LakehouseError(f"{asset_id}: excluded or historical-only asset cannot enter active pilot")
        if current.get("production_eligible") is not False or current.get("external_evidence_approved") is not False:
            raise LakehouseError(f"{asset_id}: unexpected v2 approval state requires independent investigation")
        if a["currency"] != currency:
            raise LakehouseError(f"{asset_id}: archived currency conflicts with current quote currency")
        old = str(a["yahoo_ticker"])
        if old != symbol:
            if (
                asset_id != "EQHe874aca04f607837"
                or old != "EUR-USD"
                or a["calendar_id"] != "24/7"
                or current.get("reviewed_symbol") != symbol
            ):
                raise LakehouseError(f"{asset_id}: unreviewed provider symbol/calendar migration")
            unresolved.append("eurusd_effective_dated_alias_and_forex_calendar_review")
        elif a["calendar_id"] != calendar:
            raise LakehouseError(f"{asset_id}: unexpected historical calendar mismatch")
        aliases = [
            {
                "provider_symbol": old,
                "observation_source": "archived_20261110_plan",
                "effective_from": None,
                "effective_to": None,
                "effective_dates_verified": False,
            }
        ]
        if old != symbol:
            aliases.append(
                {
                    "provider_symbol": symbol,
                    "observation_source": "v2_reviewed_symbol_and_current_git_universe",
                    "effective_from": None,
                    "effective_to": None,
                    "effective_dates_verified": False,
                }
            )
        assets.append(
            {
                "asset_id": asset_id,
                "yahoo_ticker": symbol,
                "currency": currency,
                "quote_unit": quote_unit,
                "quote_to_currency_factor": 1.0,
                "calendar_id": calendar,
                "exchange_timezone": tz,
                "asset_class": asset_class,
                "include": True,
                "source_aliases": aliases,
                "historical_retention_required": bool(current["original_plan_present"]),
                "identity_evidence_state": current["candidate_state"],
            }
        )
    configured = set()
    for asset in assets:
        if asset["currency"] != "USD":
            configured.add(asset["currency"])
    fx = [row for row in _archived_fx(s3, bucket, archive_id) if row["currency"] in configured]
    if {row["currency"] for row in fx} != configured:
        raise LakehouseError("Incomplete FX configuration for pilot currencies")
    fx.sort(key=lambda x: x["currency"])
    assets.sort(key=lambda x: x["asset_id"])
    doc = {
        "schema": PILOT_SCHEMA,
        "scope": "pilot_eight_assets",
        "review_state": "review_required",
        "assets": assets,
        "fx_pairs": fx,
        "evidence": {
            "historical_archive_id": archive_id,
            "historical_plan_sha256": plan["plan_sha256"],
            "release_inventory_commit_key": INVENTORY_PREFIX + "/_committed.json",
            "release_inventory_sha256": source["inventory_sha256"],
            "current_universe_sha256": source["current_universe_sha256"],
            "current_git_ref": GIT_UNIVERSE_REF,
            "current_git_commit": GIT_UNIVERSE_COMMIT,
            "current_git_mapping_verification": "eight_assets_cross_checked_manually",
        },
        "release_blockers": sorted(
            set(
                [
                    "independent_market_data_evidence_missing",
                    "historical_alias_ranges_unverified",
                    "full_universe_and_legacy_crosswalk_not_reconciled",
                    *unresolved,
                ]
            )
        ),
    }
    return doc


def stage_pilot_registry(s3, bucket: str, *, archive_id: str = ARCHIVE_ID, execute: bool = False) -> dict:
    """Write an immutable, unapproved S3 candidate only when explicitly executed."""
    from alpha_edge.market.lakehouse_v3.contracts import validate_pilot_registry_document

    doc = build_pilot_registry(s3, bucket, archive_id=archive_id)
    validate_pilot_registry_document(doc)
    sha = digest(json_bytes(doc))
    key = f"{ROOT}/config/candidates/registry_{sha}.json"
    output = {
        "state": "review_required",
        "registry_key": key,
        "sha256": sha,
        "assets": len(doc["assets"]),
        "blockers": doc["release_blockers"],
        "writes": 0,
    }
    if execute:
        put_json(s3, bucket, key, doc)
        output["writes"] = 1
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="AEA-91 immutable eight-asset pilot registry")
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--archive-id", default=ARCHIVE_ID)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    import boto3

    print(
        json.dumps(
            stage_pilot_registry(
                boto3.client("s3", region_name="eu-west-1"),
                args.bucket,
                archive_id=args.archive_id,
                execute=args.execute,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
