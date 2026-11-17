"""Versioned, S3-only exchange/session calendar registry for Lake v3 Silver.

Derive proposals from the SHA-pinned universe CSV and the established v2
exchange-to-calendar map. Do NOT guess missing mappings or certify proposals.
The resulting registry is immutable and its content SHA is its version ID.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import boto3
from botocore.exceptions import ClientError

from alpha_edge.market.ingest_bronze_full_history import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    SNAPSHOT_ROOT,
    _digest,
    _json_bytes,
    _read,
    _source_records,
    load_universe,
)

REGISTRY_ROOT = "market/lake_v3/config/calendar_registries/v1"
SCHEMA = "alpha_edge_calendar_registry_v1"

# Exact historical mapping from data/quality/market_lake_v2/
# exchange_calendar_mapping_v2.csv. Do not infer from ticker suffix.
EXCHANGE_CALENDAR_V2 = {
    "NYQ": "NYSE",
    "NMS": "NYSE",
    "NGM": "NYSE",
    "NCM": "NYSE",
    "PCX": "NYSE",
    "ASE": "NYSE",
    "BTS": "NYSE",
    "SAO": "BMF",
    "GER": "XETR",
    "PAR": "XPAR",
    "BRU": "XBRU",
    "MIL": "XMIL",
    "AMS": "XAMS",
    "MCE": "XMAD",
    "VIE": "XWBO",
    "ISE": "XDUB",
    "LIS": "XLIS",
    "MEX": "XMEX",
    "HEL": "XHEL",
}

SPECIAL_POLICIES = {"CRYPTO_24_7", "YAHOO_FX_WEEKDAYS"}


class CalendarRegistryError(RuntimeError):
    """Source-universe/calendar-registry mismatch or unsafe publication."""


def _derive(row: dict) -> dict:
    explicit = (row.get("calendar_id") or "").strip().upper()
    exchange = (row.get("exchange") or "").strip().upper()
    quote_type = (row.get("quoteType") or "").strip().upper()
    asset_class = (row.get("asset_class") or "").strip().lower()
    if explicit in {"24/7", "CRYPTO_24_7"} and asset_class == "crypto":
        calendar, origin = "CRYPTO_24_7", "universe_explicit"
    elif explicit in {"FOREX", "YAHOO_FX_WEEKDAYS"} and asset_class in {"fx", "fx_pair"}:
        calendar, origin = "YAHOO_FX_WEEKDAYS", "universe_explicit"
    elif not explicit and exchange == "CCC" and quote_type == "CRYPTOCURRENCY" and asset_class == "crypto":
        calendar, origin = "CRYPTO_24_7", "universe_exchange"
    elif not explicit and exchange == "CCY" and quote_type == "CURRENCY" and asset_class in {"fx", "fx_pair"}:
        calendar, origin = "YAHOO_FX_WEEKDAYS", "universe_exchange"
    elif (
        not explicit
        and exchange in EXCHANGE_CALENDAR_V2
        and quote_type in {"ETF", "EQUITY"}
        and asset_class == "equity"
    ):
        calendar, origin = EXCHANGE_CALENDAR_V2[exchange], "v2_exchange_mapping"
    else:
        calendar, origin = None, "unresolved_or_conflicting_source"
    return {
        "asset_id": row["asset_id"].strip(),
        "ticker": row["ticker"].strip(),
        "provider_symbol": row["yahoo_ticker"].strip(),
        "exchange_code": exchange,
        "source_calendar_id": explicit,
        "quote_type": quote_type,
        "calendar_id": calendar,
        "mapping_source": origin,
        "mapping_status": "proposed_review_required" if calendar else "unresolved",
        "calendar_validation_approved": False,
    }


def make_registry(s3: Any, bucket: str, fingerprint: str) -> tuple[dict, str]:
    """Validate source snapshot and use its already-recorded exchange metadata."""
    active, source = load_universe(s3, bucket, fingerprint)
    key = f"{SNAPSHOT_ROOT}/{fingerprint}/universe.csv"
    rows = _source_records(_read(s3, bucket, key), "universe.csv")
    records = [_derive(row) for row in rows if row["include"] == "1"]
    records.sort(key=lambda row: row["asset_id"])
    if len(records) != len(active):
        raise CalendarRegistryError("Current source active-universe count mismatch")
    expected = {row["asset_id"]: row["yahoo_ticker"] for row in active}
    if {row["asset_id"]: row["provider_symbol"] for row in records} != expected:
        raise CalendarRegistryError("Current source canonical identity/provider mapping mismatch")
    status = Counter(row["mapping_status"] for row in records)
    mapping = {
        "schema": SCHEMA,
        "source_fingerprint": fingerprint,
        "source_git_commit": source["source_git_commit"],
        "mapping_code_version": "existing_exchange_calendar_mapping_v2",
        "policy_semantics": {
            "CRYPTO_24_7": "daily UTC provider labels including weekends",
            "YAHOO_FX_WEEKDAYS": "provider daily Monday-Friday labels; not intraday FX open times",
            "exchange": "pandas_market_calendars exchange valid_days; early-close times not inferred",
        },
        "approval": "candidate_not_certified",
        "counts": {
            "active": len(records),
            "proposed": status["proposed_review_required"],
            "unresolved": status["unresolved"],
        },
        "assets": records,
    }
    version = _digest(_json_bytes(mapping))
    return mapping, version


def _path(version: str) -> str:
    if len(version) != 64 or not all(c in "0123456789abcdef" for c in version):
        raise CalendarRegistryError("Expected a 64-character SHA256 calendar version")
    return f"{REGISTRY_ROOT}/{version}/registry.json"


def publish(s3: Any, bucket: str, *, registry: dict, version: str, execute: bool) -> dict:
    payload = _json_bytes(registry)
    if _digest(payload) != version or registry.get("schema") != SCHEMA:
        raise CalendarRegistryError("Registry content/version mismatch")
    key = _path(version)
    report = {
        "registry_version": version,
        "s3_uri": f"s3://{bucket}/{key}",
        "counts": registry["counts"],
        "executed": execute,
        "approval": registry["approval"],
    }
    if not execute:
        return report
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType="application/json",
            Metadata={"sha256": version},
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"PreconditionFailed", "412"}:
            raise
        if _read(s3, bucket, key) != payload:
            raise CalendarRegistryError("Immutable calendar registry already differs") from exc
    if _read(s3, bucket, key) != payload:
        raise CalendarRegistryError("Registry S3 readback differs")
    report["readback_verified"] = True
    return report


def load_registry(s3: Any, bucket: str, version: str, source_fingerprint: str) -> dict:
    raw = _read(s3, bucket, _path(version))
    if _digest(raw) != version:
        raise CalendarRegistryError("Registry S3 byte SHA mismatch")
    value = json.loads(raw)
    if value.get("schema") != SCHEMA or value.get("source_fingerprint") != source_fingerprint:
        raise CalendarRegistryError("Registry is not pinned to requested source universe")
    if _digest(_json_bytes(value)) != version:
        raise CalendarRegistryError("Registry JSON not canonical")
    ids = [row["asset_id"] for row in value["assets"]]
    if len(ids) != len(set(ids)) or len(ids) != value["counts"]["active"]:
        raise CalendarRegistryError("Corrupt registry identities/count")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Version immutable S3 calendar mapping proposals")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    mapping, version = make_registry(s3, args.bucket, args.source_fingerprint)
    print(json.dumps(publish(s3, args.bucket, registry=mapping, version=version, execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
