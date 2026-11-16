"""Strict, side-effect-free contracts for the Alpha Edge S3-only lakehouse.

Contracts deliberately validate *identity and lineage*, not whether Yahoo's
underlying financial observations are correct. Independent data-quality and
corporate-action approval happen in later pipeline stages.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone

from alpha_edge.market.lakehouse_v3.storage import ROOT, LakehouseError, check_id

HEX_SHA256 = re.compile(r"^[a-f0-9]{64}$")
CURRENCY = re.compile(r"^[A-Z]{3}$")
DATE_LABEL = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REGISTRY_SCHEMA = "alpha_edge_registry_v3_1"
BRONZE_SCHEMA = "alpha_edge_bronze_v3_1"
CANDIDATE_SCHEMA = "alpha_edge_gold_candidate_v3_1"

ARTIFACT_KEYS = {
    "bars": "source_ohlcv.parquet",
    "actions": "source_actions.parquet",
    "fx": "source_fx.parquet",
}
CANDIDATE_ARTIFACT_KEYS = {
    "silver": "silver/data.parquet",
    "gold": "gold/data.parquet",
    "session_issues": "quality/session_issues.parquet",
    "quality": "quality/report.json",
}


def _required_fields(payload: dict, required: set[str], context: str) -> None:
    if not isinstance(payload, dict):
        raise LakehouseError(f"{context}: expected JSON object")
    missing = required - set(payload)
    if missing:
        raise LakehouseError(f"{context}: missing fields {sorted(missing)}")


def _sha(value: object, name: str) -> None:
    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
        raise LakehouseError(f"{name}: expected lowercase SHA-256 hex digest")


def _day(value: object, name: str) -> date:
    if not isinstance(value, str) or not DATE_LABEL.fullmatch(value):
        raise LakehouseError(f"{name}: expected YYYY-MM-DD exchange session label")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise LakehouseError(f"{name}: impossible calendar date") from exc


def _instant(value: object, name: str) -> datetime:
    """Only real, offset-aware timestamps are accepted for recorded instants."""
    if not isinstance(value, str):
        raise LakehouseError(f"{name}: expected ISO8601 UTC timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LakehouseError(f"{name}: invalid ISO8601 timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise LakehouseError(f"{name}: timestamp must include timezone offset")
    return result.astimezone(timezone.utc)


def validate_registry_document(doc: dict) -> None:
    """Reject incomplete asset identity/FX policy; do not infer currencies."""
    _required_fields(doc, {"schema", "assets", "fx_pairs"}, "registry")
    if doc["schema"] != REGISTRY_SCHEMA:
        raise LakehouseError(f"Unsupported registry schema: {doc['schema']}")
    assets, pairs = doc["assets"], doc["fx_pairs"]
    if not isinstance(assets, list) or not assets or not isinstance(pairs, list):
        raise LakehouseError("Registry assets must be a nonempty list; fx_pairs must be a list")
    seen_ids: set[str] = set()
    seen_symbols: set[str] = set()
    currencies = set()
    for record in assets:
        _required_fields(record, {"asset_id", "yahoo_ticker", "currency", "calendar_id"}, "asset")
        asset_id = record["asset_id"]
        if not isinstance(asset_id, str) or not (asset_id.startswith("EQH") or asset_id.startswith("CRYPTO:")):
            raise LakehouseError(f"Invalid canonical asset ID: {asset_id!r}")
        if asset_id in seen_ids:
            raise LakehouseError(f"Duplicate canonical asset ID: {asset_id}")
        seen_ids.add(asset_id)
        symbol = record["yahoo_ticker"]
        if not isinstance(symbol, str) or symbol != symbol.strip() or not symbol:
            raise LakehouseError(f"{asset_id}: missing provider ticker")
        if symbol in seen_symbols:
            raise LakehouseError(f"Duplicate Yahoo source symbol: {symbol}")
        seen_symbols.add(symbol)
        currency = record["currency"]
        if not isinstance(currency, str) or not CURRENCY.fullmatch(currency):
            raise LakehouseError(f"{asset_id}: invalid quote currency {currency!r}")
        currencies.add(currency)
        calendar = record["calendar_id"]
        if not isinstance(calendar, str) or not calendar.strip() or calendar != calendar.strip():
            raise LakehouseError(f"{asset_id}: explicit exchange calendar required")
        if asset_id.startswith("CRYPTO:") and calendar != "24/7":
            raise LakehouseError(f"{asset_id}: crypto requires a 24/7 calendar")
        if symbol.endswith("=X") and calendar != "FOREX":
            raise LakehouseError(f"{asset_id}: FX daily bars require FOREX calendar")
    pair_currencies = set()
    pair_symbols = set()
    for pair in pairs:
        _required_fields(pair, {"currency", "provider_symbol", "direction"}, "FX pair")
        currency = pair["currency"]
        if not isinstance(currency, str) or not CURRENCY.fullmatch(currency) or currency == "USD":
            raise LakehouseError(f"Invalid FX mapping currency {currency!r}")
        if currency in pair_currencies:
            raise LakehouseError(f"Duplicate FX mapping for {currency}")
        pair_currencies.add(currency)
        symbol = pair["provider_symbol"]
        if not isinstance(symbol, str) or not symbol.endswith("=X"):
            raise LakehouseError(f"Unsupported FX provider ticker: {symbol!r}")
        if symbol in pair_symbols:
            raise LakehouseError(f"Duplicate FX quote symbol: {symbol}")
        pair_symbols.add(symbol)
        if pair["direction"] not in {"direct", "inverse"}:
            raise LakehouseError(f"Invalid FX quote direction: {pair['direction']}")
    missing = currencies - {"USD"} - pair_currencies
    if missing:
        raise LakehouseError(f"Registry currencies missing FX mappings: {sorted(missing)}")


def validate_artifacts(artifacts: dict, *, directory: str, required: dict[str, str]) -> None:
    _required_fields(artifacts, set(required), "manifest artifacts")
    if set(artifacts) != set(required):
        raise LakehouseError("Artifact manifest includes missing or unexpected table references")
    for name, filename in required.items():
        ref = artifacts[name]
        _required_fields(ref, {"key", "sha256", "bytes"}, f"{name} artifact")
        expected = directory + "/" + filename
        if ref["key"] != expected:
            raise LakehouseError(f"{name}: artifact must reside at {expected}")
        _sha(ref["sha256"], f"{name}.sha256")
        if type(ref["bytes"]) is not int or ref["bytes"] <= 0:
            raise LakehouseError(f"{name}: invalid object size")


def validate_bronze_manifest(doc: dict, *, run_id: str) -> None:
    check_id(run_id)
    _required_fields(
        doc,
        {
            "schema",
            "run_id",
            "status",
            "registry_key",
            "registry_sha256",
            "start",
            "end_exclusive",
            "assets",
            "rows",
            "artifacts",
            "provenance",
            "completed_at_utc",
            "price_basis",
        },
        "Bronze manifest",
    )
    if doc["schema"] != BRONZE_SCHEMA or doc["run_id"] != run_id or doc["status"] != "bronze_committed":
        raise LakehouseError("Bronze manifest schema, identity or commit state invalid")
    if (
        not isinstance(doc["registry_key"], str)
        or not doc["registry_key"].startswith(ROOT + "/config/")
        or "../" in doc["registry_key"]
    ):
        raise LakehouseError("Bronze manifest registry reference outside approved configuration")
    _sha(doc["registry_sha256"], "registry_sha256")
    if _day(doc["start"], "start") >= _day(doc["end_exclusive"], "end_exclusive"):
        raise LakehouseError("Bronze manifest has invalid extraction window")
    _instant(doc["completed_at_utc"], "completed_at_utc")
    if type(doc["assets"]) is not int or doc["assets"] <= 0:
        raise LakehouseError("Bronze manifest must contain at least one asset")
    _required_fields(doc["rows"], set(ARTIFACT_KEYS), "Bronze row counts")
    if set(doc["rows"]) != set(ARTIFACT_KEYS):
        raise LakehouseError("Bronze manifest has unexpected row counts")
    for name, value in doc["rows"].items():
        if type(value) is not int or value < 0 or (name == "bars" and value == 0):
            raise LakehouseError(f"Invalid Bronze {name} row count")
    if not isinstance(doc["provenance"], list) or not doc["provenance"]:
        raise LakehouseError("Bronze provenance missing")
    if doc["price_basis"] != "yahoo_unadjusted_auto_adjust_false_close_is_provider_split_adjusted":
        raise LakehouseError("Unrecognized provider adjustment basis")
    validate_artifacts(
        doc["artifacts"],
        directory=f"{ROOT}/bronze/runs/{run_id}",
        required=ARTIFACT_KEYS,
    )


def validate_candidate_manifest(doc: dict, *, run_id: str) -> None:
    check_id(run_id)
    _required_fields(
        doc,
        {
            "schema",
            "run_id",
            "status",
            "approved",
            "bronze_manifest_key",
            "bronze_registry_sha256",
            "artifacts",
            "quality",
        },
        "Gold candidate manifest",
    )
    if (
        doc["schema"] != CANDIDATE_SCHEMA
        or doc["run_id"] != run_id
        or doc["status"] != "review_required"
        or doc["approved"] is not False
    ):
        raise LakehouseError("Candidate status/schema/identity invalid")
    expected_bronze = f"{ROOT}/bronze/runs/{run_id}/manifest.json"
    if doc["bronze_manifest_key"] != expected_bronze:
        raise LakehouseError("Candidate does not reference its exact immutable Bronze source")
    _sha(doc["bronze_registry_sha256"], "candidate registry SHA-256")
    _required_fields(
        doc["quality"],
        {"bronze_rows", "silver_rows", "gold_rows", "assets"},
        "candidate quality report",
    )
    if any(
        type(doc["quality"][x]) is not int or doc["quality"][x] < 0
        for x in ("bronze_rows", "silver_rows", "gold_rows", "assets")
    ):
        raise LakehouseError("Invalid candidate row counts")
    if not (doc["quality"]["bronze_rows"] == doc["quality"]["silver_rows"] == doc["quality"]["gold_rows"]):
        raise LakehouseError("Bronze/Silver/Gold candidate row counts do not reconcile")
    validate_artifacts(
        doc["artifacts"],
        directory=f"{ROOT}/candidates/{run_id}",
        required=CANDIDATE_ARTIFACT_KEYS,
    )


def validate_pilot_registry_document(doc: dict) -> None:
    """AEA-91 review-only richer registry. Never confuse with approved v3.1."""
    _required_fields(
        doc,
        {"schema", "scope", "review_state", "assets", "fx_pairs", "evidence", "release_blockers"},
        "v3 pilot registry",
    )
    if (
        doc["schema"] != "alpha_edge_registry_v3_2"
        or doc["scope"] != "pilot_eight_assets"
        or doc["review_state"] != "review_required"
    ):
        raise LakehouseError("Unsupported, out-of-scope or prematurely approved pilot registry")
    if not isinstance(doc["release_blockers"], list) or not doc["release_blockers"]:
        raise LakehouseError("Pilot registry must retain release blockers")
    # Shared identity, currency, FX mapping, calendar uniqueness checks.
    compatible = {**doc, "schema": REGISTRY_SCHEMA}
    validate_registry_document(compatible)
    if len(doc["assets"]) != 8:
        raise LakehouseError("Pilot registry must contain exactly eight assets")
    _required_fields(
        doc["evidence"],
        {"historical_plan_sha256", "release_inventory_sha256", "current_universe_sha256",
         "release_inventory_commit_key", "historical_archive_id", "current_git_commit"},
        "pilot evidence",
    )
    for key in ("historical_plan_sha256", "release_inventory_sha256", "current_universe_sha256"):
        _sha(doc["evidence"][key], key)
    expected_tickers = {
        "SPY", "GLD", "BTC-USD", "RENT3.SA", "TRAXIONA.MX",
        "LOG.MC", "REY.MI", "EURUSD=X",
    }
    if {x["yahoo_ticker"] for x in doc["assets"]} != expected_tickers:
        raise LakehouseError("Pilot registry contains an unexpected or missing security")
    for asset in doc["assets"]:
        _required_fields(
            asset,
            {"asset_id", "yahoo_ticker", "currency", "quote_unit",
             "quote_to_currency_factor", "calendar_id", "exchange_timezone",
             "asset_class", "include", "source_aliases",
             "historical_retention_required", "identity_evidence_state"},
            "pilot asset",
        )
        if asset["include"] is not True or asset["historical_retention_required"] is not True:
            raise LakehouseError("Pilot excludes an asset or would discard its source history")
        if asset["quote_unit"] != asset["currency"] or asset["quote_to_currency_factor"] != 1.0:
            raise LakehouseError("Unverified pilot quote-unit conversion")
        if not isinstance(asset["exchange_timezone"], str):
            raise LakehouseError("Exchange timezone must be explicit")
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(asset["exchange_timezone"])
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise LakehouseError("Unknown exchange timezone") from exc
        aliases = asset["source_aliases"]
        if not isinstance(aliases, list) or not aliases:
            raise LakehouseError("Source observation alias evidence missing")
        distinct_aliases = set()
        for alias in aliases:
            _required_fields(
                alias,
                {"provider_symbol", "observation_source", "effective_from",
                 "effective_to", "effective_dates_verified"},
                "source alias",
            )
            if (
                alias["provider_symbol"] in distinct_aliases
                or not alias["observation_source"]
                or alias["effective_dates_verified"] is not False
                or alias["effective_from"] is not None
                or alias["effective_to"] is not None
            ):
                raise LakehouseError("Unexpected duplicated/approved or fabricated historical alias")
            distinct_aliases.add(alias["provider_symbol"])
        if asset["yahoo_ticker"] not in distinct_aliases:
            raise LakehouseError("Current source symbol absent from alias evidence")
        if asset["yahoo_ticker"] == "EURUSD=X":
            if distinct_aliases != {"EUR-USD", "EURUSD=X"}:
                raise LakehouseError("EUR/USD history must preserve both observed symbols")
        elif distinct_aliases != {asset["yahoo_ticker"]}:
            raise LakehouseError("Unexpected source symbol transition requires review")
