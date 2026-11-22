"""Explicit reviewed suppression of proven duplicate provider dividends only."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation

import pandas as pd

from alpha_edge.market.lakehouse_v3.storage import ROOT, digest, json_bytes, put_immutable, verified_bytes

SCHEMA = "alpha_edge_dividend_duplicate_registry_v2"
REGISTRY_ROOT = f"{ROOT}/config/dividend_duplicate_registries/v2"


def validate_registry(doc):
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or not isinstance(doc.get("entries"), list):
        raise ValueError("Invalid duplicate dividend registry")
    keys, retained = set(), set()
    for entry in doc["entries"]:
        if not isinstance(entry, dict):
            raise TypeError("Expected duplicate correction object")
        for key in (
            "asset_id",
            "provider_symbol",
            "bronze_sha256",
            "effective_date",
            "retained_date",
            "source_value",
            "retained_source_value",
            "source_currency",
            "issuer_ex_date",
            "issuer_amount",
            "issuer_currency",
            "evidence_url",
            "reason",
            "status",
        ):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"Missing correction field {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["bronze_sha256"]):
            raise ValueError("Invalid Bronze checksum")
        for key in ("effective_date", "retained_date", "issuer_ex_date"):
            if pd.Timestamp(entry[key]).date().isoformat() != entry[key]:
                raise ValueError("Invalid correction date")
        if entry["effective_date"] == entry["retained_date"]:
            raise ValueError("Cannot suppress retained event")
        for key in ("source_value", "retained_source_value", "issuer_amount"):
            try:
                value = Decimal(entry[key])
            except InvalidOperation as exc:
                raise ValueError("Invalid exact amount") from exc
            if not value.is_finite() or value <= 0:
                raise ValueError("Expected positive finite source amount")
        if Decimal(entry["source_value"]) != Decimal(entry["retained_source_value"]):
            raise ValueError("Duplicate suppression requires equal source amounts")
        if (
            entry["issuer_ex_date"] != entry["retained_date"]
            or Decimal(entry["issuer_amount"]) != Decimal(entry["retained_source_value"])
            or entry["issuer_currency"] != entry["source_currency"]
        ):
            raise ValueError("Retained dividend must match independently documented issuer date, amount and currency")
        if not entry["evidence_url"].startswith("https://") or entry["status"] not in {"proposed", "approved"}:
            raise ValueError("Invalid correction evidence or status")
        if entry["status"] == "approved" and not str(entry.get("approved_by", "")).strip():
            raise ValueError("Approved suppression requires reviewer identity")
        key = (entry["asset_id"], entry["effective_date"])
        if key in keys:
            raise ValueError("Duplicate correction")
        keys.add(key)
        retained.add((entry["asset_id"], entry["retained_date"]))
    if keys & retained:
        raise ValueError("Retained event cannot also be suppressed")
    return doc


def publish_registry(s3, bucket, doc, *, execute=False):
    payload = json_bytes(validate_registry(doc))
    version = digest(payload)
    key = f"{REGISTRY_ROOT}/{version}/registry.json"
    if execute:
        put_immutable(s3, bucket, key, payload, content_type="application/json", scope=REGISTRY_ROOT)
    return {"key": key, "version": version, "entries": len(doc["entries"]), "executed": execute}


def load_registry(s3, bucket, version):
    if not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("Expected SHA-256 duplicate registry version")
    blob = verified_bytes(s3, bucket, {"key": f"{REGISTRY_ROOT}/{version}/registry.json", "sha256": version})
    doc = validate_registry(json.loads(blob))
    if digest(json_bytes(doc)) != version:
        raise ValueError("Non-canonical duplicate registry bytes")
    return doc


def apply_suppressions(actions, doc, *, asset_id, provider_symbol, bronze_sha256):
    """Candidate action values only; never mutate source evidence or adjust prices."""
    validate_registry(doc)
    candidate = actions.copy(deep=True)
    if not candidate.asset_id.eq(asset_id).all() or not candidate.provider_symbol.eq(provider_symbol).all():
        raise ValueError("Unexpected source action identity")
    if "normalized_value" not in candidate:
        candidate["normalized_value"] = candidate["source_value"].astype(float)
    candidate["duplicate_correction_status"] = "not_corrected"
    candidate["duplicate_proposed_value"] = float("nan")
    counts = {"approved": 0, "proposed": 0}
    for entry in doc["entries"]:
        if entry["asset_id"] != asset_id:
            continue
        if entry["provider_symbol"] != provider_symbol or entry["bronze_sha256"] != bronze_sha256:
            raise ValueError("Duplicate correction source identity or checksum changed")
        indexes = []
        for day, value in (
            (entry["effective_date"], entry["source_value"]),
            (entry["retained_date"], entry["retained_source_value"]),
        ):
            mask = candidate.effective_date.eq(day) & candidate.action_type.eq("DIVIDEND")
            if mask.sum() != 1:
                raise ValueError("Duplicate correction does not resolve one dividend")
            i = candidate.index[mask][0]
            if (
                Decimal(str(candidate.at[i, "source_value"])) != Decimal(value)
                or candidate.at[i, "source_currency"] != entry["source_currency"]
            ):
                raise ValueError("Duplicate correction source amount or currency changed")
            if Decimal(str(candidate.at[i, "normalized_value"])) != Decimal(value):
                raise ValueError("Duplicate correction requires unchanged positive candidate amounts")
            if (
                "currency_correction_status" in candidate
                and candidate.at[i, "currency_correction_status"] != "not_corrected"
            ):
                raise ValueError("Duplicate correction conflicts with currency correction")
            indexes.append(i)
        idx = indexes[0]
        candidate.at[idx, "duplicate_proposed_value"] = 0.0
        candidate.at[idx, "duplicate_correction_status"] = entry["status"]
        if entry["status"] == "approved":
            candidate.at[idx, "normalized_value"] = 0.0
        counts[entry["status"]] += 1
    return candidate, {**counts, "registry_version": digest(json_bytes(doc)), "production_eligible": False}
