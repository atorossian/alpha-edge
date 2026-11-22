"""Explicit legacy-currency dividend corrections, pinned to immutable Bronze.

The price currency is not evidence of the dividend's original denomination.
Only a documented event-specific correction may change its currency basis.
No prices, split factors or returns are changed by this module.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
import re

import pandas as pd

from alpha_edge.market.lakehouse_v3.storage import ROOT, digest, json_bytes, put_immutable, verified_bytes


REGISTRY_ROOT = f"{ROOT}/config/dividend_currency_registries/v1"
SCHEMA = "alpha_edge_dividend_currency_registry_v1"
RATE_REFERENCE = "https://ec.europa.eu/eurostat/cache/metadata/en/ert_h_esms.htm"
# Irrevocably fixed legacy currency units per EUR, not floating FX observations.
UNITS_PER_EUR = {"ESP": "166.386", "ITL": "1936.27", "FRF": "6.55957"}


class DividendCurrencyError(ValueError):
    """Invalid, stale, ambiguous or unsupported correction evidence."""


def _positive(value: str, field: str) -> Decimal:
    if not isinstance(value, str):
        raise DividendCurrencyError(f"{field} must be an exact decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise DividendCurrencyError(f"Invalid {field}") from exc
    if not result.is_finite() or result <= 0:
        raise DividendCurrencyError(f"Non-positive or non-finite {field}")
    return result


def validate_registry(document: dict) -> dict:
    """Validate an explicit, versionable registry; fail closed on ambiguity."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise DividendCurrencyError("Invalid dividend currency registry schema")
    if document.get("rate_reference") != RATE_REFERENCE or document.get("units_per_eur") != UNITS_PER_EUR:
        raise DividendCurrencyError("Unexpected fixed conversion policy")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise DividendCurrencyError("Expected correction entries list")
    keys, correction_ids = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise DividendCurrencyError("Expected correction object")
        for name in (
            "asset_id",
            "provider_symbol",
            "effective_date",
            "bronze_sha256",
            "correction_id",
            "source_value",
            "original_currency",
            "price_currency",
            "declared_amount",
            "share_basis_multiplier",
            "basis_evidence",
            "evidence_url",
            "status",
        ):
            if not isinstance(entry.get(name), str) or not entry[name].strip():
                raise DividendCurrencyError(f"Missing correction field {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["bronze_sha256"]):
            raise DividendCurrencyError("Invalid Bronze checksum")
        day = pd.to_datetime(entry["effective_date"], format="%Y-%m-%d", errors="coerce")
        if pd.isna(day) or day.date().isoformat() != entry["effective_date"] or day >= pd.Timestamp("2002-01-01"):
            raise DividendCurrencyError("Correction is outside the supported legacy-currency period")
        if entry["original_currency"] not in UNITS_PER_EUR or entry["price_currency"] != "EUR":
            raise DividendCurrencyError("Unsupported legacy-currency conversion")
        if not entry["evidence_url"].startswith("https://"):
            raise DividendCurrencyError("Independent HTTPS evidence reference required")
        if entry["status"] not in {"proposed", "approved"}:
            raise DividendCurrencyError("Invalid correction status")
        if entry["status"] == "approved" and not str(entry.get("approved_by", "")).strip():
            raise DividendCurrencyError("Approved correction requires reviewer identity")
        source = _positive(entry["source_value"], "source_value")
        declared = _positive(entry["declared_amount"], "declared_amount")
        multiplier = _positive(entry["share_basis_multiplier"], "share_basis_multiplier")
        with localcontext() as ctx:
            ctx.prec = 34
            if abs(source - declared * multiplier) > source * Decimal("0.0000001"):
                raise DividendCurrencyError("Declared amount does not reconcile to source share basis")
        key = (entry["asset_id"], entry["provider_symbol"], entry["effective_date"])
        if key in keys or entry["correction_id"] in correction_ids:
            raise DividendCurrencyError("Duplicate correction key or ID")
        keys.add(key)
        correction_ids.add(entry["correction_id"])
    return document


def registry_version(document: dict) -> str:
    return digest(json_bytes(validate_registry(document)))


def publish_registry(s3, bucket: str, document: dict, *, execute: bool = False) -> dict:
    """Explicit immutable candidate/config publication; dry run by default."""
    version = registry_version(document)
    key = f"{REGISTRY_ROOT}/{version}/registry.json"
    result = {"key": key, "version": version, "entries": len(document["entries"]), "executed": execute}
    if execute:
        put_immutable(s3, bucket, key, json_bytes(document), content_type="application/json", scope=REGISTRY_ROOT)
    return result


def load_registry(s3, bucket: str, version: str) -> dict:
    """Read exactly the requested content-addressed S3 version."""
    import json

    if not re.fullmatch(r"[0-9a-f]{64}", version):
        raise DividendCurrencyError("Expected a SHA-256 registry version")
    payload = verified_bytes(s3, bucket, {"key": f"{REGISTRY_ROOT}/{version}/registry.json", "sha256": version})
    document = validate_registry(json.loads(payload))
    if registry_version(document) != version:
        raise DividendCurrencyError("Non-canonical registry bytes")
    return document


def apply_currency_corrections(
    actions: pd.DataFrame, document: dict, *, asset_id: str, provider_symbol: str, bronze_sha256: str
) -> tuple[pd.DataFrame, dict]:
    """Return separate candidate action evidence; only approved rows are accepted.

    Proposed rows expose a preview without changing the accepted value. Every
    matching correction must resolve exactly once against the pinned source.
    """
    validate_registry(document)
    candidate = actions.copy(deep=True)
    required = {"asset_id", "provider_symbol", "effective_date", "action_type", "source_value", "source_currency"}
    if not required.issubset(candidate):
        raise DividendCurrencyError("Missing source action fields")
    if not candidate["asset_id"].eq(asset_id).all() or not candidate["provider_symbol"].eq(provider_symbol).all():
        raise DividendCurrencyError("Unexpected source action identity")
    candidate["normalized_value"] = candidate["source_value"].astype(float)
    candidate["normalized_currency"] = candidate["source_currency"]
    candidate["proposed_value"] = float("nan")
    candidate["correction_id"] = pd.Series(pd.NA, index=candidate.index, dtype="string")
    candidate["currency_correction_status"] = "not_corrected"
    candidate["original_dividend_currency"] = pd.Series(pd.NA, index=candidate.index, dtype="string")
    count = {"proposed": 0, "approved": 0}
    for entry in document["entries"]:
        if entry["asset_id"] != asset_id:
            continue
        if entry["provider_symbol"] != provider_symbol or entry["bronze_sha256"] != bronze_sha256:
            raise DividendCurrencyError("Correction source identity or checksum has changed")
        mask = candidate["effective_date"].eq(entry["effective_date"]) & candidate["action_type"].eq("DIVIDEND")
        if mask.sum() != 1:
            raise DividendCurrencyError("Correction does not resolve exactly one dividend")
        idx = candidate.index[mask][0]
        if candidate.at[idx, "source_currency"] != "EUR" or Decimal(str(candidate.at[idx, "source_value"])) != Decimal(
            entry["source_value"]
        ):
            raise DividendCurrencyError("Correction source value or quote currency has changed")
        with localcontext() as ctx:
            ctx.prec = 34
            corrected = Decimal(entry["source_value"]) / Decimal(UNITS_PER_EUR[entry["original_currency"]])
        candidate.at[idx, "proposed_value"] = float(corrected)
        candidate.at[idx, "correction_id"] = entry["correction_id"]
        candidate.at[idx, "original_dividend_currency"] = entry["original_currency"]
        candidate.at[idx, "currency_correction_status"] = entry["status"]
        if entry["status"] == "approved":
            candidate.at[idx, "normalized_value"] = float(corrected)
        count[entry["status"]] += 1
    return candidate, {**count, "registry_version": registry_version(document), "production_eligible": False}
