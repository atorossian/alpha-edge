"""Isolated asset-level AEA87 recovery and historical coverage audit.

Explicit execution only. Never rewrites original cohorts, legacy data or
production release pointers. A successful acquisition is NOT price approval.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from datetime import date

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.market.ingest_bronze_yahoo_v2 import acquire_yahoo_bars
from alpha_edge.market.universe_registry_v2 import (
    _get,
    _json_bytes,
    _records_bytes,
    _sha,
    cohort_commit_presence,
    load_archived_plan,
    publish_evidence,
)

RECOVERY_ROOT = "dev/market/lake_v2/asset_recovery"
AUDIT_ROOT = "dev/market/lake_v2/asset_audits"
SAFE_ASSET_ID = re.compile(r"^[A-Za-z0-9:_-]{1,100}$")
SAFE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class AssetRecoveryError(ValueError):
    pass


def _check_params(asset_id: str, end: str) -> None:
    if not SAFE_ASSET_ID.fullmatch(asset_id) or not SAFE_DATE.fullmatch(end):
        raise AssetRecoveryError("Invalid asset ID or exclusive end date")
    date.fromisoformat(end)


def _original_asset(s3, bucket: str, archive_id: str, asset_id: str, end: str):
    _check_params(asset_id, end)
    plan, info = load_archived_plan(s3, bucket, archive_id)
    subset = plan.loc[plan["asset_id"].eq(asset_id)]
    if len(subset) != 1:
        raise AssetRecoveryError(f"Not exactly one asset in archived plan: {asset_id}")
    states = cohort_commit_presence(s3, bucket, info["plan_sha256"], end, info["expected_cohorts"])
    return subset.iloc[0].to_dict(), info, states


def _result_bytes(
    payloads: dict[str, bytes], metadata: dict, *, run_id: str | None = None,
    schema: str = "aea87_asset_recovery_candidate_v1",
) -> tuple[dict[str, bytes], dict]:
    fingerprint = {
        "schema": schema,
        "source": metadata,
        "sha256": {name: _sha(blob) for name, blob in sorted(payloads.items())},
    }
    run_id = run_id or _sha(_json_bytes(fingerprint))
    manifest = {
        "schema": fingerprint["schema"],
        "run_id": run_id,
        "source": metadata,
        "artifact_sha256": fingerprint["sha256"],
        "historical_prices_approved": False,
        "corporate_actions_approved": False,
        "production_eligible": False,
    }
    return {**payloads, "manifest.json": _json_bytes(manifest)}, manifest


def recover_asset(
    s3, bucket: str, archive_id: str, asset_id: str, end: str, *,
    execute: bool = False, ticker_factory=None,
) -> dict:
    """Acquire a failed-cohort asset using its ORIGINAL pinned identity.

    The run cannot silently adopt current ticker mappings or legacy v1 values.
    One failing asset does not affect any other asset or completed cohort.
    """
    row, info, states = _original_asset(s3, bucket, archive_id, asset_id, end)
    cohort = int(row["original_cohort"])
    if states[cohort] != "missing":
        raise AssetRecoveryError("Asset's original cohort is not missing; refusing duplicate acquisition")
    symbol = str(row["yahoo_ticker"]).strip()
    currency = str(row["currency"]).strip()
    calendar = str(row["calendar_id"]).strip()
    if not all((symbol, currency, calendar)):
        raise AssetRecoveryError("Original immutable plan lacks required provider/currency/calendar metadata")
    base = {
        "asset_id": asset_id,
        "original_cohort": cohort,
        "original_plan_sha256": info["plan_sha256"],
        "provider_symbol": symbol,
        "currency": currency,
        "calendar_id": calendar,
        "end_exclusive": end,
    }
    if not execute:
        return {"status": "dry_run", "source_contract": base, "s3_writes": 0}
    # Stable immutable address for this acquisition contract, independent of
    # provider timestamps. Replays verify existing bytes before any download.
    recovery_id = _sha(_json_bytes({"schema": "aea87_recovery_identity_v1", **base}))
    recovery_prefix = f"{RECOVERY_ROOT}/run-{recovery_id}"
    try:
        existing = json.loads(
            s3.get_object(Bucket=bucket, Key=recovery_prefix + "/manifest.json")["Body"].read()
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"NoSuchKey", "404", "NotFound"}:
            raise
        existing = None
    except ValueError as exc:
        raise AssetRecoveryError("Existing recovery manifest is invalid") from exc
    if existing is not None:
        if existing.get("run_id") != recovery_id or existing.get("source", {}).get("asset_id") != asset_id:
            raise AssetRecoveryError("Existing recovery contract mismatch")
        name = "source_ohlcv.parquet"
        blob = _get(s3, bucket, f"{recovery_prefix}/{name}")
        if _sha(blob) != existing.get("artifact_sha256", {}).get(name):
            raise AssetRecoveryError("Existing recovery Bronze SHA mismatch")
        return {"status": "verified_existing_research_candidate", "asset_id": asset_id,
                "run_id": recovery_id, "production_eligible": False,
                "evidence": {"s3_uri": f"s3://{bucket}/{recovery_prefix}/manifest.json"}}
    try:
        bars, provider_meta = acquire_yahoo_bars(
            asset_id, symbol, start=None, end=end,
            batch_id=f"aea87-recovery-{cohort:04d}", ticker_factory=ticker_factory,
            currency_hint=currency, full_history=True,
        )
    except Exception as exc:
        # A source-provider failure must remain visible as S3 evidence, without
        # claiming a completed acquisition or blocking unrelated assets.
        failure = {
            **base, "state": "source_acquisition_failed",
            "error_type": type(exc).__name__,
            "error_detail": str(exc)[:500],
            "historical_prices_approved": False,
        }
        artifacts, failure_manifest = _result_bytes(
            {"failure.json": _json_bytes(failure)},
            {"asset_id": asset_id, "original_plan_sha256": info["plan_sha256"],
             "end_exclusive": end, "state": "source_acquisition_failed"},
            schema="aea87_asset_recovery_failure_v1",
        )
        evidence = publish_evidence(s3, bucket, artifacts, failure_manifest, root=RECOVERY_ROOT)
        return {"status": "acquisition_failed", "asset_id": asset_id,
                "error_type": failure["error_type"], "evidence": evidence,
                "production_eligible": False}
    if set(bars["asset_id"].astype(str)) != {asset_id}:
        raise AssetRecoveryError("Provider returned unexpected asset IDs")
    if set(bars["provider_symbol"].astype(str)) != {symbol}:
        raise AssetRecoveryError("Provider returned unexpected symbols")
    if not pd.to_datetime(bars["session_date"], errors="coerce").lt(pd.Timestamp(end)).all():
        raise AssetRecoveryError("Acquisition returned invalid or out-of-window sessions")
    buffer = io.BytesIO()
    bars.to_parquet(buffer, index=False)
    files, manifest = _result_bytes(
        {"source_ohlcv.parquet": buffer.getvalue()},
        {**base, "provider_provenance": provider_meta}, run_id=recovery_id,
    )
    location = publish_evidence(s3, bucket, files, manifest, root=RECOVERY_ROOT)
    return {
        "status": "published_research_candidate_not_approved",
        "asset_id": asset_id,
        "rows": len(bars),
        "run_id": manifest["run_id"],
        "evidence": location,
        "production_eligible": False,
    }


def audit_sessions_and_actions(
    bars: pd.DataFrame, *, asset_id: str, calendar_id: str,
    declared_listing_start: str | None = None,
    declared_listing_end: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Audit expected sessions with explicit endpoint and corporate-action blockers.

    Without independent verified listing boundaries, completeness is UNKNOWN,
    even when there are zero gaps between first and last observed sessions.
    Corporate-action columns are observations only, never proof of correctness.
    """
    required = {"asset_id", "provider_symbol", "session_date", "open", "high", "low", "close", "volume"}
    if missing := required - set(bars):
        raise AssetRecoveryError(f"Bronze missing columns: {sorted(missing)}")
    selected = bars.loc[bars["asset_id"].astype(str).eq(asset_id)].copy()
    if selected.empty:
        raise AssetRecoveryError("No observations for requested asset")
    if selected["provider_symbol"].nunique() != 1:
        raise AssetRecoveryError("Multiple provider symbols under one asset ID")
    dates = pd.to_datetime(selected["session_date"], errors="coerce")
    if dates.isna().any():
        raise AssetRecoveryError("Unparseable observation session dates")
    if dates.duplicated().any():
        raise AssetRecoveryError("Duplicate observed sessions")
    observed = set(dates.dt.normalize().tolist())
    first, last = min(observed), max(observed)
    if calendar_id == "24/7":
        expected = pd.date_range(first, last, freq="D")
    else:
        try:
            import pandas_market_calendars as mcal
            expected = mcal.get_calendar(calendar_id).valid_days(start_date=first, end_date=last)
        except (KeyError, RuntimeError, ValueError, ImportError) as exc:
            raise AssetRecoveryError(f"Unverified exchange calendar: {calendar_id}") from exc
    expected = {
        pd.Timestamp(day).tz_localize(None).normalize()
        if pd.Timestamp(day).tzinfo
        else pd.Timestamp(day).normalize()
        for day in expected
    }
    missing_sessions = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    findings = [
        {"asset_id": asset_id, "session_date": value.date().isoformat(), "issue": name}
        for name, values in (
            ("missing_expected_session", missing_sessions),
            ("unexpected_observed_session", unexpected),
        )
        for value in values
    ]
    for field in ("open", "high", "low", "close", "volume"):
        values = pd.to_numeric(selected[field], errors="coerce")
        invalid = values.isna() | ~values.map(lambda x: pd.notna(x) and abs(x) != float("inf"))
        if field == "volume":
            invalid = invalid | values.lt(0)
        else:
            invalid = invalid | values.le(0)
        for session in selected.loc[invalid, "session_date"].tolist():
            findings.append({"asset_id": asset_id, "session_date": str(session)[:10], "issue": f"invalid_{field}"})
    # Source-consistency checks supplement, but never replace, independent
    # price and corporate-action verification.
    prices = selected[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    malformed = (
        prices["high"].lt(prices[["open", "close", "low"]].max(axis=1))
        | prices["low"].gt(prices[["open", "close", "high"]].min(axis=1))
    )
    for session in selected.loc[malformed, "session_date"].tolist():
        findings.append({"asset_id": asset_id, "session_date": str(session)[:10],
                         "issue": "inconsistent_ohlc_envelope"})
    if "adj_close" in selected:
        adjusted = pd.to_numeric(selected["adj_close"], errors="coerce")
        bad_adj = adjusted.isna() | ~adjusted.map(lambda x: pd.notna(x) and abs(x) != float("inf"))
        bad_adj = bad_adj | adjusted.le(0)
        for session in selected.loc[bad_adj, "session_date"].tolist():
            findings.append({"asset_id": asset_id, "session_date": str(session)[:10],
                             "issue": "invalid_adjusted_close"})
    else:
        findings.append({"asset_id": asset_id, "session_date": "",
                         "issue": "missing_adjusted_close_provenance"})
    action_events = {}
    action_missing_counts = {}
    for field in ("dividends", "stock_splits"):
        if field in selected:
            value = pd.to_numeric(selected[field], errors="coerce")
            missing_values = value.isna() | value.map(
                lambda x: pd.notna(x) and abs(x) == float("inf")
            )
            action_missing_counts[field] = int(missing_values.sum())
            if missing_values.any():
                findings.append({
                    "asset_id": asset_id, "session_date": "",
                    "issue": f"missing_or_nonfinite_{field}_observations",
                })
            action_events[field] = int(value.fillna(0).ne(0).sum())
            suspicious = value.lt(0)
            for session in selected.loc[suspicious, "session_date"].tolist():
                findings.append({"asset_id": asset_id, "session_date": str(session)[:10],
                                 "issue": f"implausible_{field}"})
        else:
            action_events[field] = None
            action_missing_counts[field] = len(selected)
            findings.append({"asset_id": asset_id, "session_date": "", "issue": f"missing_{field}_provenance"})
    endpoints_verified = bool(declared_listing_start and declared_listing_end)
    if endpoints_verified:
        start = pd.Timestamp(declared_listing_start).normalize()
        end = pd.Timestamp(declared_listing_end).normalize()
        if start > first or end < last or start > end:
            raise AssetRecoveryError("Declared listing window contradicts observed sessions")
        # Endpoints are only accepted after a separate independent listing
        # record review. Passing dates here is never an approval.
    summary = {
        "asset_id": asset_id,
        "calendar_id": calendar_id,
        "first_observed": first.date().isoformat(),
        "last_observed": last.date().isoformat(),
        "observation_count": len(selected),
        "missing_internal_sessions": len(missing_sessions),
        "unexpected_sessions": len(unexpected),
        "action_observation_counts": action_events,
        "action_missing_or_nonfinite_counts": action_missing_counts,
        "listed_history_endpoints_declared": endpoints_verified,
        "listing_dates_independently_verified": False,
        "corporate_actions_independently_verified": False,
        "provider_prices_independently_verified": False,
        "historical_coverage_approved": False,
        "production_eligible": False,
    }
    return pd.DataFrame(findings, columns=["asset_id", "session_date", "issue"]), summary


def audit_existing_asset(
    s3, bucket: str, archive_id: str, asset_id: str, end: str, *, publish: bool = False
) -> dict:
    """Read original committed Bronze (or recovery evidence), never v1 OHLCV."""
    row, info, states = _original_asset(s3, bucket, archive_id, asset_id, end)
    cohort = int(row["original_cohort"])
    if states[cohort] != "marker_present_unverified":
        raise AssetRecoveryError("Original committed cohort unavailable for audit")
    prefix = (
        f"dev/market/lake_v2/backfills/{info['plan_sha256'][:24]}/end-{end}/"
        f"cohort-{cohort:04d}"
    )
    commit = json.loads(_get(s3, bucket, prefix + "/_committed.json"))
    if commit.get("plan_manifest_sha256") != info["plan_sha256"] or commit.get("prefix") != prefix:
        raise AssetRecoveryError("Original commit marker identity invalid")
    relative = f"execution/bronze_batches/s3bf-{cohort:04d}-{cohort:04d}/source_ohlcv.parquet"
    entries = {item["path"]: item["sha256"] for item in commit.get("files", [])}
    if relative not in entries:
        raise AssetRecoveryError("Committed raw Bronze not listed in original marker")
    original = _get(s3, bucket, f"{prefix}/files/{relative}")
    if _sha(original) != entries[relative]:
        raise AssetRecoveryError("Original Bronze bytes failed marker SHA verification")
    bars = pd.read_parquet(io.BytesIO(original))
    findings, summary = audit_sessions_and_actions(
        bars, asset_id=asset_id, calendar_id=str(row["calendar_id"])
    )
    artifacts, manifest = _result_bytes(
        {"findings.jsonl": _records_bytes(findings),
         "coverage.json": _json_bytes(summary)},
        {"asset_id": asset_id, "original_plan_sha256": info["plan_sha256"],
         "original_cohort": cohort, "source_sha256": _sha(original),
         "original_bronze_key": f"{prefix}/files/{relative}"},
        schema="aea87_asset_coverage_audit_v1",
    )
    location = publish_evidence(s3, bucket, artifacts, manifest, root=AUDIT_ROOT) if publish else {
        "status": "dry_run_no_writes"
    }
    return {**summary, "findings": len(findings), "evidence": location}


def audit_recovered_asset(
    s3, bucket: str, asset_id: str, recovery_id: str, *, publish: bool = False
) -> dict:
    """Audit an immutable individual recovery candidate, not legacy v1 data."""
    if not SAFE_ASSET_ID.fullmatch(asset_id) or not re.fullmatch(r"[a-f0-9]{64}", recovery_id):
        raise AssetRecoveryError("Invalid recovery audit selection")
    prefix = f"{RECOVERY_ROOT}/run-{recovery_id}"
    source_manifest = json.loads(_get(s3, bucket, f"{prefix}/manifest.json"))
    metadata = source_manifest.get("source", {})
    if source_manifest.get("run_id") != recovery_id or metadata.get("asset_id") != asset_id:
        raise AssetRecoveryError("Recovery manifest identity mismatch")
    source_raw = _get(s3, bucket, f"{prefix}/source_ohlcv.parquet")
    if _sha(source_raw) != source_manifest.get("artifact_sha256", {}).get("source_ohlcv.parquet"):
        raise AssetRecoveryError("Recovery source Bronze SHA mismatch")
    frame = pd.read_parquet(io.BytesIO(source_raw))
    findings, summary = audit_sessions_and_actions(
        frame, asset_id=asset_id, calendar_id=str(metadata["calendar_id"])
    )
    artifacts, manifest = _result_bytes(
        {
            "findings.jsonl": _records_bytes(findings),
            "coverage.json": _json_bytes(summary),
        },
        {"asset_id": asset_id, "recovery_run_id": recovery_id, "source_sha256": _sha(source_raw)},
        schema="aea87_asset_coverage_audit_v1",
    )
    location = publish_evidence(s3, bucket, artifacts, manifest, root=AUDIT_ROOT) if publish else {
        "status": "dry_run_no_writes"
    }
    return {**summary, "findings": len(findings), "evidence": location}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--archive-id", default="old-outputs-20261110")
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--end", default="2026-11-10")
    parser.add_argument("--mode", choices=("recover", "audit"), required=True)
    parser.add_argument("--execute", action="store_true", help="Explicit single-asset provider acquisition")
    parser.add_argument("--recovery-run-id", help="Audit existing individual recovery candidate instead of cohort")
    parser.add_argument("--publish", action="store_true", help="Publish immutable audit evidence")
    parser.add_argument("--region", default="eu-west-1")
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    if args.mode == "recover":
        if args.publish:
            parser.error("Recovery uses --execute; --publish is for the audit mode")
        result = recover_asset(
            s3, args.bucket, args.archive_id, args.asset_id, args.end, execute=args.execute
        )
    else:
        if args.execute:
            parser.error("Audit does not acquire prices; use --publish only")
        if args.recovery_run_id:
            result = audit_recovered_asset(
                s3, args.bucket, args.asset_id, args.recovery_run_id, publish=args.publish
            )
        else:
            result = audit_existing_asset(
                s3, args.bucket, args.archive_id, args.asset_id, args.end, publish=args.publish
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "acquisition_failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
