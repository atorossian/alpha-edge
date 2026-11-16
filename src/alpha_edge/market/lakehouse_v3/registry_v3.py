"""AEA-91 canonical universe from EXACTLY five versioned configuration files.

This module reads *no* previous market-data, v1/v2 backfills, release
inventories, identity crosswalks or market-data archives. A source manifest and
the constructed registry are persisted in S3; ingest consumers read only an
explicitly activated, checksum-pinned registry from S3.

The universe.csv include flag decides eligibility. Supporting source files
validate/explain it; they cannot silently resurrect excluded instruments.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from collections import Counter
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from botocore.exceptions import ClientError

from alpha_edge.market.lakehouse_v3.storage import (
    ROOT,
    LakehouseError,
    digest,
    json_bytes,
    put_json,
    read_bytes,
    read_json,
)

SOURCE_FILES = (
    "universe.csv",
    "universe_overrides.csv",
    "asset_excluded.csv",
    "ingest_force_refresh.csv",
    "universe_manual_additions.csv",
)
SCHEMA = "alpha_edge_universe_registry_v3_1"
SOURCE_SCHEMA = "alpha_edge_universe_sources_v3_1"
ACTIVE_SCHEMA = "alpha_edge_active_registry_v3_1"

# Explicit *new* normalization policy. These keys come from the current
# universe.csv 'market' field, NOT from stored historical observations.
# Missing/ambiguous markets remain blocked. AEA-93 must verify each
# venue's historical sessions before production market-data certification.
MARKET_POLICY = {
    "us_market": ("NYSE", "America/New_York"),
    "de_market": ("XETR", "Europe/Berlin"),
    "fr_market": ("XPAR", "Europe/Paris"),
    "be_market": ("XBRU", "Europe/Brussels"),
    "br_market": ("BMF", "America/Sao_Paulo"),
    "it_market": ("XMIL", "Europe/Rome"),
    "nl_market": ("XAMS", "Europe/Amsterdam"),
    "at_market": ("XWBO", "Europe/Vienna"),
    "ie_market": ("XDUB", "Europe/Dublin"),
    "mx_market": ("XMEX", "America/Mexico_City"),
    "es_market": ("XMAD", "Europe/Madrid"),
    "fi_market": ("XHEL", "Europe/Helsinki"),
    "pt_market": ("XLIS", "Europe/Lisbon"),
    "ccc_market": ("24/7", "UTC"),
    "ccy_market": ("FOREX", "UTC"),
}
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


def _read_csv(blob: bytes, filename: str) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(blob.decode("utf-8-sig"), newline=""))
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise LakehouseError(f"{filename}: missing or duplicate column headers")
        rows = list(reader)
    except (UnicodeError, csv.Error) as exc:
        raise LakehouseError(f"{filename}: malformed UTF-8 CSV") from exc
    if any(None in row for row in rows):
        raise LakehouseError(f"{filename}: surplus unquoted columns")
    return [{key: value.strip() if value is not None else "" for key, value in row.items()} for row in rows]


def _required(rows: list[dict], fields: set[str], filename: str) -> None:
    if not rows:
        raise LakehouseError(f"{filename}: no source rows")
    missing = fields - set(rows[0])
    if missing:
        raise LakehouseError(f"{filename}: missing columns {sorted(missing)}")


def _resolve(row: dict, by_id: dict, by_ticker: dict, filename: str) -> dict:
    """Prefer stable row ID; never guess among duplicate broker tickers."""
    rid = row.get("target_row_id", "")
    if rid:
        result = by_id.get(rid)
    else:
        candidates = by_ticker.get(row.get("ticker", ""), [])
        result = candidates[0] if len(candidates) == 1 else None
        # The display ticker may have changed since an override was authored.
    # A stable target_row_id takes precedence and never guesses another listing.\n    if result is None:
        raise LakehouseError(f"{filename}: unknown or ambiguous row target {rid or row.get('ticker')}")
    return result


def validate_source_files(files: dict[str, bytes]) -> dict[str, list[dict]]:
    if set(files) != set(SOURCE_FILES):
        raise LakehouseError(f"Expected only these five sources: {SOURCE_FILES}")
    frames = {name: _read_csv(files[name], name) for name in SOURCE_FILES}
    _required(
        frames["universe.csv"],
        {
            "row_id",
            "ticker",
            "asset_id",
            "yahoo_ticker",
            "include",
            "currency",
            "market",
            "calendar_id",
            "exchange",
            "asset_class",
        },
        "universe.csv",
    )
    _required(
        frames["universe_overrides.csv"],
        {"target_row_id", "ticker", "yahoo_ticker", "lock_yahoo_ticker"},
        "universe_overrides.csv",
    )
    _required(frames["asset_excluded.csv"], {"target_row_id", "ticker", "reason"}, "asset_excluded.csv")
    _required(
        frames["universe_manual_additions.csv"],
        {"ticker", "yahoo_ticker", "lock_yahoo_ticker"},
        "universe_manual_additions.csv",
    )
    _required(frames["ingest_force_refresh.csv"], {"ticker"}, "ingest_force_refresh.csv")
    rows = frames["universe.csv"]
    for col in ("row_id", "asset_id"):
        vals = [r[col] for r in rows]
        if not all(vals) or len(vals) != len(set(vals)):
            raise LakehouseError(f"universe.csv: missing or duplicate {col}")
    if any(not r["ticker"] for r in rows):
        raise LakehouseError("universe.csv: missing broker ticker")
    if any(r["include"] not in {"0", "1"} for r in rows):
        raise LakehouseError("universe.csv: include must be 0 or 1")
    return frames


def _policy(row: dict) -> tuple[str | None, str | None, list[str]]:
    issues = []
    calendar, tz = MARKET_POLICY.get(row["market"], (None, None))
    # An explicitly supplied calendar is evidence from the current universe.
    if row["calendar_id"]:
        if calendar and calendar != row["calendar_id"]:
            issues.append("market_calendar_conflict")
        else:
            calendar = row["calendar_id"]
    if not calendar or not tz:
        issues.append("unknown_market_calendar")
    else:
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError):
            issues.append("invalid_exchange_timezone")
    return calendar, tz, issues


def _fx_sources(currencies: set[str], rows: list[dict]) -> list[dict]:
    """Derive candidate FX tickers from current universe or naming convention.

    These are *not* financial-data approvals; AEA-95 verifies Yahoo values,
    currency/quote basis and time-of-availability before accepting USD data.
    """
    available = {r["yahoo_ticker"] for r in rows if r["include"] == "1"}
    output = []
    for ccy in sorted(currencies - {"USD"}):
        direct = ccy + "USD=X"
        inverse = "USD" + ccy + "=X"
        # Prefer a source actually present in the five-file universe.
        source, direction = (direct, "direct") if direct in available else (inverse, "inverse")
        output.append(
            {
                "currency": ccy,
                "provider_symbol": source,
                "direction": direction,
                "verification": "unverified_candidate",
                "source_convention": "generated_not_historical_data",
            }
        )
    return output


def build_registry(files: dict[str, bytes]) -> dict:
    """Deterministic selection from current include flags and five-file checks."""
    frames = validate_source_files(files)
    rows = frames["universe.csv"]
    by_id = {r["row_id"]: r for r in rows}
    # Broker tickers are not unique across listings (for example SW and MRK).
    by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(row)
    exceptions: dict[str, set[str]] = {r["asset_id"]: set() for r in rows}
    reasons: dict[str, str] = {}
    overrides = frames["universe_overrides.csv"]
    manual = frames["universe_manual_additions.csv"]
    excluded = frames["asset_excluded.csv"]
    refresh = frames["ingest_force_refresh.csv"]
    seen_overrides: dict[str, tuple[str, str]] = {}
    for x in overrides:
        r = _resolve(x, by_id, by_ticker, "universe_overrides.csv")
        aid = r["asset_id"]
        decision = (x["yahoo_ticker"], x["lock_yahoo_ticker"])
        if aid in seen_overrides and seen_overrides[aid] != decision:
            exceptions[aid].add("conflicting_locked_overrides")
        seen_overrides[aid] = decision
        if x["lock_yahoo_ticker"] == "1" and x["yahoo_ticker"] != r["yahoo_ticker"]:
            exceptions[aid].add("override_symbol_conflict")
    seen_manual: set[str] = set()
    for x in manual:
        candidates = by_ticker.get(x["ticker"], [])
        if not candidates:
            raise LakehouseError(f"universe_manual_additions.csv: not materialized {x['ticker']}")
        matched = [r for r in candidates if r["yahoo_ticker"] == x["yahoo_ticker"]]
        if len(matched) == 1:
            r = matched[0]
        elif len(candidates) == 1:
            r = candidates[0]
        else:
            for candidate in candidates:
                if candidate["include"] == "1":
                    exceptions[candidate["asset_id"]].add("manual_ticker_ambiguous")
            continue
        if r["asset_id"] in seen_manual:
            raise LakehouseError(f"Duplicate manual addition: {r['asset_id']}")
        seen_manual.add(r["asset_id"])
        if x["lock_yahoo_ticker"] == "1" and x["yahoo_ticker"] != r["yahoo_ticker"]:
            exceptions[r["asset_id"]].add("manual_locked_yahoo_symbol_conflict")
    seen_excluded: set[str] = set()
    for x in excluded:
        r = _resolve(x, by_id, by_ticker, "asset_excluded.csv")
        aid = r["asset_id"]
        if aid in seen_excluded:
            raise LakehouseError(f"Duplicate exclusion for {r['ticker']}")
        seen_excluded.add(aid)
        if r["include"] == "1":
            exceptions[aid].add("excluded_but_included_conflict")
        reasons[aid] = x["reason"]
    refresh_tickers: set[str] = set()
    for x in refresh:
        ticker = x["ticker"]
        if ticker not in by_ticker or ticker in refresh_tickers:
            raise LakehouseError(f"Missing/duplicate force-refresh ticker: {ticker}")
        refresh_tickers.add(ticker)
        if len(by_ticker[ticker]) != 1:
            for candidate in by_ticker[ticker]:
                if candidate["include"] == "1":
                    exceptions[candidate["asset_id"]].add("force_refresh_ticker_ambiguous")
    symbols: dict[str, list[str]] = {}
    for r in rows:
        if r["include"] == "1":
            if not r["yahoo_ticker"]:
                exceptions[r["asset_id"]].add("missing_source_ticker")
            symbols.setdefault(r["yahoo_ticker"], []).append(r["asset_id"])
    for group in symbols.values():
        if len(group) > 1:
            for aid in group:
                exceptions[aid].add("duplicate_provider_ticker")
    assets = []
    for r in sorted(rows, key=lambda x: x["asset_id"]):
        if r["include"] != "1":
            continue
        issues = set(exceptions[r["asset_id"]])
        ccy = r["currency"]
        if not CURRENCY_RE.fullmatch(ccy):
            issues.add("missing_or_invalid_quote_currency")
        calendar, tz, policy_issues = _policy(r)
        issues.update(policy_issues)
        assets.append(
            {
                "asset_id": r["asset_id"],
                "row_id": r["row_id"],
                "ticker": r["ticker"],
                "yahoo_ticker": r["yahoo_ticker"],
                "currency": ccy or None,
                "calendar_id": calendar,
                "exchange_timezone": tz,
                "market": r["market"],
                "exchange": r["exchange"],
                "asset_class": r["asset_class"],
                "source_status": "blocked" if issues else "configured",
                "blockers": sorted(issues),
                "force_refresh": (r["ticker"] in refresh_tickers and len(by_ticker[r["ticker"]]) == 1),
                "quote_unit": ccy or None,
                "quote_to_currency_factor": 1.0 if ccy else None,
                "quote_unit_policy": "same_as_source_currency_pending_source_verification",
            }
        )
    included = sum(r["include"] == "1" for r in rows)
    if len(assets) != included:
        raise LakehouseError("Included universe rows silently dropped")
    blocked = [x for x in assets if x["source_status"] == "blocked"]
    configured = [x for x in assets if x["source_status"] == "configured"]
    currencies = {x["currency"] for x in assets if x["currency"]}
    source_files = {name: {"sha256": digest(files[name]), "bytes": len(files[name])} for name in SOURCE_FILES}
    source_digest = digest(json_bytes(source_files))
    registry = {
        "schema": SCHEMA,
        "source_schema": SOURCE_SCHEMA,
        "source_fingerprint": source_digest,
        "source_files": source_files,
        "selection_rule": "universe.csv:include==1; supporting sources reconciled",
        "status": "configuration_candidate",
        "assets": assets,
        "excluded": [
            {
                "asset_id": r["asset_id"],
                "ticker": r["ticker"],
                "exclusion_reason": reasons.get(r["asset_id"], "universe_include_0"),
            }
            for r in sorted(rows, key=lambda x: x["asset_id"])
            if r["include"] == "0"
        ],
        "fx_pairs": _fx_sources(currencies, rows),
        "refresh_tickers": sorted(refresh_tickers),
        "summary": {
            "configured": len(configured), "blocked": len(blocked),
            "excluded": len(rows) - included,
            "blocker_counts": dict(sorted(Counter(
                reason for a in blocked for reason in a["blockers"]
            ).items())),
            "by_currency": dict(sorted(Counter(
                a["currency"] or "<missing>" for a in assets
            ).items())),
        },
    }
    validate_registry(registry)
    return registry


def validate_registry(doc: dict) -> None:
    """Read-only runtime type and count gate for registry candidates."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise LakehouseError("Unsupported current-universe registry schema")
    if doc.get("status") != "configuration_candidate" or doc.get("selection_rule") != (
        "universe.csv:include==1; supporting sources reconciled"
    ):
        raise LakehouseError("Unapproved or incompatible selection contract")
    source_files = doc.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != set(SOURCE_FILES):
        raise LakehouseError("Registry references unexpected sources")
    if doc.get("source_fingerprint") != digest(json_bytes(source_files)):
        raise LakehouseError("Registry source manifest fingerprint mismatch")
    assets, excluded, summary = doc["assets"], doc["excluded"], doc["summary"]
    if not assets or any(not isinstance(x, dict) for x in assets):
        raise LakehouseError("Empty or malformed active-asset selection")
    ids = [x["asset_id"] for x in assets + excluded]
    if len(ids) != len(set(ids)):
        raise LakehouseError("Duplicate canonical asset across included and excluded")
    if len(assets) != summary["included"] or len(excluded) != summary["excluded"]:
        raise LakehouseError("Registry counts differ from source selection")
    if summary["universe_total"] != len(ids):
        raise LakehouseError("Full universe count differs from included/excluded")
    if sum(a["source_status"] == "blocked" for a in assets) != summary["blocked"]:
        raise LakehouseError("Blocked asset count differs")
    for asset in assets:
        is_blocked = asset["source_status"] == "blocked"
        if bool(asset["blockers"]) != is_blocked:
            raise LakehouseError("Blocked asset has incorrect readiness flag")
        if not is_blocked and not (
            asset["currency"] and asset["calendar_id"] and asset["exchange_timezone"]
        ):
            raise LakehouseError("Configured asset lacks required market policy")
    fx_ccy = {x["currency"] for x in doc["fx_pairs"]}
    required = {a["currency"] for a in assets if a["currency"] and a["currency"] != "USD"}
    if fx_ccy != required:
        raise LakehouseError("Missing FX candidates for included source currencies")


def registry_key(doc: dict) -> str:
    validate_registry(doc)
    return f"{ROOT}/config/registries/registry_{digest(json_bytes(doc))}.json"


def stage_registry(s3, bucket: str, source_files: dict[str, bytes], *, execute: bool = False) -> dict:
    """Record immutable five-source provenance and candidate in S3; never activate."""
    doc = build_registry(source_files)
    key = registry_key(doc)
    sha = digest(json_bytes(doc))
    result = {"state": "candidate", "registry_key": key, "sha256": sha,
              "summary": doc["summary"], "writes": 0}
    if not execute:
        return result
    src_root = f"{ROOT}/config/source_snapshots/{doc['source_fingerprint']}"
    for name in SOURCE_FILES:
        from alpha_edge.market.lakehouse_v3.storage import put_immutable

        put_immutable(s3, bucket, f"{src_root}/{name}", source_files[name], content_type="text/csv")
    put_json(s3, bucket, key, doc)
    result["writes"] = len(SOURCE_FILES) + 1
    return result


def read_registry(s3, bucket: str, key: str, expected_sha256: str) -> dict:
    """Pin the exact immutable S3 registry bytes; no file or legacy fallback."""
    if not key.startswith(f"{ROOT}/config/registries/"):
        raise LakehouseError("Only versioned S3 canonical registries may be consumed")
    raw = read_bytes(s3, bucket, key)
    if digest(raw) != expected_sha256:
        raise LakehouseError("Registry contents changed under pinned key")
    doc = json.loads(raw)
    validate_registry(doc)
    if registry_key(doc) != key:
        raise LakehouseError("Registry key does not match content address")
    return doc


def load_active(s3, bucket: str) -> tuple[dict, dict]:
    """Never return a staged registry unless an explicit active pointer selects it."""
    pointer = read_json(s3, bucket, f"{ROOT}/config/active.json")
    if pointer.get("schema") != ACTIVE_SCHEMA:
        raise LakehouseError("Unsupported S3 active registry pointer")
    doc = read_registry(s3, bucket, pointer["registry_key"], pointer["registry_sha256"])
    return doc, pointer


def verify_source_snapshots(s3, bucket: str, doc: dict) -> None:
    """Every one of the five files must exist in S3 with its pinned digest."""
    validate_registry(doc)
    prefix = f"{ROOT}/config/source_snapshots/{doc['source_fingerprint']}/"
    for name in SOURCE_FILES:
        payload = read_bytes(s3, bucket, prefix + name)
        expected = doc["source_files"][name]
        if digest(payload) != expected["sha256"] or len(payload) != expected["bytes"]:
            raise LakehouseError(f"Source configuration checksum/size mismatch: {name}")
    # Reconstruct selection to detect malicious or accidental registry mutation.
    files = {name: read_bytes(s3, bucket, prefix + name) for name in SOURCE_FILES}
    regenerated = build_registry(files)
    if digest(json_bytes(regenerated)) != digest(json_bytes(doc)):
        raise LakehouseError("Registry differs from its five authoritative source files")


def activate_registry(
    s3, bucket: str, *, key: str, sha256: str,
    execute: bool = False, expected_previous_sha256: str | None = None,
) -> dict:
    """Activate CONFIG authority, not financial data approval.

    Initial activation is conditional on no prior pointer. Replacements
    require the caller to name the exact prior registry SHA and use a
    conditional S3 update. The historical config remains immutable.
    """
    doc = read_registry(s3, bucket, key, sha256)
    verify_source_snapshots(s3, bucket, doc)
    pointer_key = f"{ROOT}/config/active.json"
    try:
        prior = s3.get_object(Bucket=bucket, Key=pointer_key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404", "NotFound"}:
            raise
        prior = None
    if prior is not None:
        old = json.loads(prior["Body"].read())
        if old.get("schema") != ACTIVE_SCHEMA or old.get("registry_sha256") != expected_previous_sha256:
            raise LakehouseError("Existing registry requires exact expected previous SHA for replacement")
    elif expected_previous_sha256 is not None:
        raise LakehouseError("Previous registry expectation differs from empty S3 pointer")
    result = {
        "status": "configuration_active" if execute else "activation_dry_run",
        "registry_key": key,
        "registry_sha256": sha256,
        "source_fingerprint": doc["source_fingerprint"],
        "included": doc["summary"]["included"],
        "excluded": doc["summary"]["excluded"],
        "blocked": doc["summary"]["blocked"],
        "previous_registry_sha256": expected_previous_sha256,
        "writes": 0,
        "market_prices_approved": False,
    }
    if not execute:
        return result
    pointer = {
        "schema": ACTIVE_SCHEMA,
        "registry_key": key,
        "registry_sha256": sha256,
        "source_fingerprint": doc["source_fingerprint"],
        "included": doc["summary"]["included"],
        "blocked": doc["summary"]["blocked"],
        "previous_registry_sha256": expected_previous_sha256,
        "activated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_prices_approved": False,
    }
    kwargs = {"IfMatch": prior["ETag"]} if prior else {"IfNoneMatch": "*"}
    try:
        s3.put_object(
            Bucket=bucket, Key=pointer_key, Body=json_bytes(pointer),
            ContentType="application/json", **kwargs,
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {
            "PreconditionFailed", "412", "ConditionalRequestConflict", "409"
        }:
            raise LakehouseError("Concurrent active-registry update rejected") from exc
        raise
    active, current = load_active(s3, bucket)
    if active["source_fingerprint"] != doc["source_fingerprint"] or current != pointer:
        raise LakehouseError("Active S3 registry read-back mismatch after promotion")
    result["writes"] = 1
    return result
