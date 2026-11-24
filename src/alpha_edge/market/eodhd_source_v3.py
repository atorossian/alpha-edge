"""EODHD acquisition adapter for the shared full-history Bronze ingestion.

Provider responses are preserved as exact UTF-8 JSON text in a single immutable
asset bundle. No price adjustments, event merging or inferred currencies.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request

import certifi
import requests

from alpha_edge.market.lakehouse_v3.storage import ROOT, digest, json_bytes, verified_bytes

STANDARD_API_BASE = "https://eodhd.com/api"
MILAN_API_BASE = "https://cboe.eodhd.com/api"


def _api_base(record):
    base = record.get("api_base", STANDARD_API_BASE)
    if base == STANDARD_API_BASE:
        return base
    if base == MILAN_API_BASE and record["provider_symbol"].endswith(".MI"):
        return base
    raise EodhdSourceError("Unapproved EODHD endpoint for instrument")


MAPPING_ROOT = f"{ROOT}/config/eodhd_symbol_mappings/v1"
ENDPOINTS = ("eod", "splits", "div")
FIELDS = {
    "eod": {"date", "open", "high", "low", "close", "adjusted_close", "volume"},
    "splits": {"date", "split"},
    "div": {"date", "value"},
}


class EodhdSourceError(RuntimeError):
    """Sanitized provider acquisition or validation error."""


def load_mapping(s3, bucket, records, version):
    if not isinstance(version, str) or not re.fullmatch(r"[a-f0-9]{64}", version):
        raise EodhdSourceError("EODHD requires a pinned mapping registry SHA256")
    document = json.loads(verified_bytes(s3, bucket, {
        "key": f"{MAPPING_ROOT}/objects/{version}.json",
        "sha256": version,
    }))
    if document.get("schema") != "alpha_edge_eodhd_symbol_mapping_v1":
        raise EodhdSourceError("Unknown mapping registry schema")
    entries = document["assets"]
    by_id = {entry["asset_id"]: entry for entry in entries}
    if len(by_id) != len(entries) or len({e["eodhd_symbol"] for e in entries}) != len(entries):
        raise EodhdSourceError("Duplicate asset or provider symbol in mapping")
    selected = []
    for record in records:
        entry = by_id.get(record["asset_id"])
        if not entry or entry["yahoo_symbol"] != record["yahoo_ticker"]:
            raise EodhdSourceError(f"Missing/stale mapping: {record['asset_id']}")
        symbol = entry["eodhd_symbol"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", symbol) or "." not in symbol:
            raise EodhdSourceError("Invalid EODHD symbol")
        mapped = {
            **record,
            "provider": "eodhd",
            "provider_symbol": symbol,
            "mapping_version": version,
            "api_base": entry.get("api_base", STANDARD_API_BASE),
        }
        _api_base(mapped)
        selected.append(mapped)
    return selected


def verify_bundle(payload, record, end):
    bundle = json.loads(payload)
    if (bundle.get("schema") != "alpha_edge_eodhd_raw_bundle_v1"
            or bundle.get("asset_id") != record["asset_id"]
            or bundle.get("provider_symbol") != record["provider_symbol"]
            or bundle.get("mapping_version") != record["mapping_version"]
            or bundle.get("api_base", STANDARD_API_BASE) != _api_base(record)
            or bundle.get("end_exclusive") != end
            or set(bundle["responses"]) != set(ENDPOINTS)):
        raise EodhdSourceError("Invalid bundle identity or endpoints")
    stats = {}
    for endpoint in ENDPOINTS:
        item = bundle["responses"][endpoint]
        raw = item["raw_json"].encode("utf-8")
        if digest(raw) != item["sha256"]:
            raise EodhdSourceError("Response checksum mismatch")
        rows = json.loads(raw)
        if not isinstance(rows, list) or (endpoint == "eod" and not rows):
            raise EodhdSourceError("Invalid or empty price response")
        dates = []
        for row in rows:
            if not isinstance(row, dict) or not FIELDS[endpoint] <= set(row):
                raise EodhdSourceError(f"Invalid {endpoint} record schema")
            day = date.fromisoformat(row["date"])
            if day >= date.fromisoformat(end):
                raise EodhdSourceError("Response outside requested history")
            dates.append(day.isoformat())
        stats[endpoint] = {
            "rows": len(rows), "first_session": min(dates, default=None),
            "last_session": max(dates, default=None),
            "duplicate_date_rows": len(dates) - len(set(dates)),
        }
    return {**stats["eod"], "endpoints": stats}


def acquire_eodhd(record, end, token, *, opener=None):
    if not token:
        raise EodhdSourceError("Missing EODHD credential")
    until = (date.fromisoformat(end) - timedelta(days=1)).isoformat()
    base = _api_base(record)
    responses = {}
    for endpoint in ENDPOINTS:
        params = {"api_token": token, "fmt": "json", "to": until}
        url = f"{base}/{endpoint}/{record['provider_symbol']}"
        try:
            if opener is None:
                # Explicit CA bundle; reject redirects so credentials stay on the approved host.
                response = requests.get(
                    url, params=params, headers={"Accept": "application/json"},
                    verify=certifi.where(), timeout=60, allow_redirects=False,
                )
                if response.status_code != 200:
                    raise EodhdSourceError(f"{endpoint}: HTTP {response.status_code}")
                raw = response.content
            else:
                # Injectable transport for existing offline tests.
                request = Request(f"{url}?{urlencode(params)}", headers={"Accept": "application/json"})
                with opener(request, timeout=60) as response:
                    raw = response.read()
            responses[endpoint] = {"raw_json": raw.decode("utf-8"), "sha256": digest(raw)}
        except EodhdSourceError:
            raise
        except HTTPError as exc:
            raise EodhdSourceError(f"{endpoint}: HTTP {exc.code}") from None
        except Exception:
            raise EodhdSourceError(f"{endpoint}: request or decoding failed") from None
    payload = json_bytes({
        "schema": "alpha_edge_eodhd_raw_bundle_v1",
        "asset_id": record["asset_id"],
        "provider_symbol": record["provider_symbol"],
        "mapping_version": record["mapping_version"],
        "end_exclusive": end,
        "api_base": base,
        "responses": responses,
    })
    try:
        validation = verify_bundle(payload, record, end)
    except Exception:
        raise EodhdSourceError("Provider response validation failed") from None
    return payload, validation, {
        "provider": "eodhd", "mapping_version": record["mapping_version"],
        "api_base": base,
        "identity_status": "candidate_mapping",
        "ohlc_basis": "provider_documented_as_traded",
        "volume_basis": "provider_documented_split_adjusted",
        "adjusted_close_basis": "provider_documented_split_dividend_adjusted",
    }
