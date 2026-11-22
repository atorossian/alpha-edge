"""Silver v3 stage 1: provenance-preserving session and calendar candidates.

Read one committed S3 Bronze asset at a time. Do not apply FX, dividends,
splits, corporate-action splices, deduplication or market-data filling.
Daily provider session dates are exchange-local labels, NOT UTC timestamps.
Results are candidate-only and require independent QA before Gold publication.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import uuid
from datetime import date
from importlib.metadata import version
from typing import Any

import boto3
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from alpha_edge.market.calendar_registry_v3 import load_registry
from alpha_edge.market.effective_exchange_calendars import calendar_on, effective_schedule
from alpha_edge.market.ingest_bronze_full_history import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    SAFE_RUN,
    BronzeIngestError,
    _completed,
    _digest,
    _get_optional,
    _json_bytes,
    _read,
    load_universe,
)
from alpha_edge.market.lakehouse_v3.registry_v3 import resolve_source_fingerprint
from alpha_edge.market.lakehouse_v3.storage import LakehouseError, parquet_bytes, put_immutable, verified_bytes
from alpha_edge.market.standardize_bronze_v2 import audit_asset_sessions

SILVER_ROOT = "market/lake_v3/silver/session_candidates"
SILVER_SCHEMA = "alpha_edge_silver_session_candidate_v2"
CORRECTION_ROOT = f"{SILVER_ROOT}/corrections"
CALENDAR_POLICY_ROOT = f"{SILVER_ROOT}/calendar_policies"
ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OFFSET = re.compile(r"(?:Z|[+-]\d{2}:\d{2})$")


class SilverSessionError(RuntimeError):
    """Silver source, provenance, identity or candidate validation failure."""


def _put(s3: Any, bucket: str, key: str, data: bytes, content_type: str) -> None:
    try:
        put_immutable(s3, bucket, key, data, content_type=content_type, scope=SILVER_ROOT)
    except LakehouseError as exc:
        raise SilverSessionError(str(exc)) from exc


def _day(value: Any) -> str | None:
    if not isinstance(value, str) or not ISO_DAY.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def _instant(value: Any) -> str | None:
    """Convert only offset-aware ISO instants; never infer local session UTC."""
    if not isinstance(value, str) or not OFFSET.search(value):
        return None
    try:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            return None
        return stamp.tz_convert("UTC").isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _special_session_audit(
    bars: pd.DataFrame, calendar_id: str, expected_days=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explicit source DAILY bar conventions, not intraday exchange hours."""
    if expected_days is None and calendar_id not in {"CRYPTO_24_7", "YAHOO_FX_WEEKDAYS"}:
        raise SilverSessionError(f"Unsupported source-specific daily-session policy: {calendar_id}")
    observed = pd.to_datetime(bars["session_date"], format="%Y-%m-%d")
    first, last = observed.min(), observed.max()
    expected = (
        expected_days
        if expected_days is not None
        else pd.date_range(first, last, freq="D" if calendar_id == "CRYPTO_24_7" else "B")
    )
    actual = set(observed)
    valid = set(expected)
    preceding = {date: expected[i - 1] if i else None for i, date in enumerate(expected)}
    flags = bars[["asset_id", "provider_symbol", "session_date"]].copy()
    flags["session_date"] = observed
    flags["is_expected_session"] = observed.map(lambda d: d in valid).to_numpy()
    flags["previous_expected_session_missing"] = observed.map(
        lambda d: preceding.get(d) is not None and preceding.get(d) not in actual
    ).to_numpy()
    issues = [
        {
            "asset_id": bars["asset_id"].iloc[0],
            "provider_symbol": bars["provider_symbol"].iloc[0],
            "session_date": date,
            "issue": "missing_expected_session",
            "calendar_id": calendar_id,
        }
        for date in sorted(valid - actual)
    ]
    issues.extend(
        {
            "asset_id": bars["asset_id"].iloc[0],
            "provider_symbol": bars["provider_symbol"].iloc[0],
            "session_date": date,
            "issue": "unexpected_session",
            "calendar_id": calendar_id,
        }
        for date in sorted(actual - valid)
    )
    columns = ["asset_id", "provider_symbol", "session_date", "issue", "calendar_id"]
    return flags, pd.DataFrame(issues, columns=columns)


def prepare_session_candidate(
    bronze: pd.DataFrame,
    *,
    asset_id: str,
    symbol: str,
    calendar_id: str | None = None,
    close_correction: dict | None = None,
    calendar_policy: dict | None = None,
    end_exclusive: str | None = None,
    calendar_history: list[dict] | None = None,
    txse_policy: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Preserve every source observation and emit explicit findings.

    Missing calendar mapping is REVIEW_REQUIRED, never a fabricated
    Monday-Friday schedule for equities or worldwide markets.
    """
    needed = {
        "asset_id",
        "provider_symbol",
        "session_date",
        "source_timestamp",
        "source_timezone",
        "currency_original",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "dividends",
        "stock_splits",
    }
    if missing := needed - set(bronze):
        raise SilverSessionError(f"Bronze columns missing: {sorted(missing)}")
    if bronze.empty or not bronze["asset_id"].eq(asset_id).all() or not bronze["provider_symbol"].eq(symbol).all():
        raise SilverSessionError("Empty Bronze input or wrong canonical asset/provider identity")

    out = bronze.copy(deep=True).reset_index(drop=True)
    out["exchange_session_date"] = out["session_date"].map(_day)
    out["source_timestamp_utc"] = out["source_timestamp"].map(_instant)
    out["timestamp_semantics"] = "provider_daily_bar_label_not_verified_market_close"
    out["calendar_id"] = calendar_id or ""
    if calendar_history:
        out["effective_calendar_id"] = out["exchange_session_date"].map(
            lambda day: calendar_on(calendar_history, day) if day else ""
        )
        # Filter exception decisions by the actual historical exchange, then
        # apply them to the joined schedule under the candidate's registry ID.
        if calendar_policy:
            calendar_policy = {
                **calendar_policy,
                "exceptions": [
                    {**entry, "calendar_id": calendar_id}
                    for entry in calendar_policy["exceptions"]
                    if entry["calendar_id"] == calendar_on(calendar_history, entry["session_date"])
                ],
            }
    original = out["session_date"].astype(str)
    dup = original.duplicated(keep=False) & out["exchange_session_date"].notna()
    invalid_date = out["exchange_session_date"].isna()
    missing_timezone = out["source_timestamp_utc"].isna()

    o = pd.to_numeric(out["open"], errors="coerce")
    h = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    c = pd.to_numeric(out["close"], errors="coerce")
    out["effective_close"] = c
    out["close_correction_id"] = ""
    if close_correction is not None:
        if (
            close_correction["asset_id"] != asset_id
            or close_correction["provider_symbol"] != symbol
            or not out["currency_original"].eq(close_correction["currency_original"]).all()
        ):
            raise SilverSessionError("Close correction identity mismatch")
        mask = out["session_date"].eq(close_correction["session_date"])
        if mask.sum() != 1 or not np.isclose(
            c.loc[mask].iloc[0], close_correction["original_close"], rtol=0, atol=1e-5
        ):
            raise SilverSessionError("Close correction does not match the original observation")
        replacement = float(close_correction["corrected_close"])
        if not np.isfinite(replacement) or replacement <= 0:
            raise SilverSessionError("Invalid corrected closing price")
        if close_correction.get("kind") == "close_corroboration":
            if not np.isclose(c.loc[mask].iloc[0], replacement, rtol=0, atol=1e-5):
                raise SilverSessionError("Corroboration cannot change the closing price")
        elif not low.loc[mask].iloc[0] <= replacement <= h.loc[mask].iloc[0]:
            raise SilverSessionError("Corrected closing price is outside the reported range")
        out.loc[mask, "effective_close"] = replacement
        out.loc[mask, "close_correction_id"] = close_correction["correction_id"]
    v = pd.to_numeric(out["volume"], errors="coerce")
    # These are QA flags only: provider price values never change.
    bad_ohlc = (
        ~np.isfinite(o)
        | ~np.isfinite(h)
        | ~np.isfinite(low)
        | ~np.isfinite(c)
        | o.le(0)
        | h.le(0)
        | low.le(0)
        | c.le(0)
        | h.lt(low)
        | h.lt(o)
        | h.lt(c)
        | low.gt(o)
        | low.gt(c)
    )
    bad_volume = v.lt(0)
    issues = []
    for mask, reason in (
        (invalid_date, "invalid_session_date"),
        (dup, "duplicate_source_session"),
        (missing_timezone, "missing_offset_aware_timestamp"),
        (bad_ohlc, "inconsistent_raw_ohlc"),
        (bad_volume, "negative_volume"),
    ):
        for idx in out.index[mask]:
            issues.append(
                {
                    "asset_id": asset_id,
                    "provider_symbol": symbol,
                    "source_row": int(out.index.get_loc(idx)),
                    "session_date": str(out.at[idx, "session_date"]),
                    "issue": reason,
                    "calendar_id": calendar_id or "",
                }
            )
    out["calendar_session_status"] = "not_checked"
    calendar_issue_count = 0
    if calendar_id:
        # Never mask malformed dates/duplicates by passing them to the
        # existing v2 calendar audit, which expects valid unique sessions.
        if not invalid_date.any() and not dup.any():
            try:
                if calendar_history:
                    dates = out["exchange_session_date"]
                    schedule = effective_schedule(calendar_history, dates.min(), dates.max(), txse_policy or {})
                    flags, gaps = _special_session_audit(
                        out[["asset_id", "provider_symbol", "session_date"]], calendar_id, schedule.index
                    )
                elif calendar_id in {"CRYPTO_24_7", "YAHOO_FX_WEEKDAYS"}:
                    flags, gaps = _special_session_audit(
                        out[["asset_id", "provider_symbol", "session_date"]], calendar_id
                    )
                else:
                    flags, gaps = audit_asset_sessions(
                        out[["asset_id", "provider_symbol", "session_date"]],
                        calendars={symbol: calendar_id},
                    )
            except Exception as exc:
                raise SilverSessionError(f"Unusable explicitly supplied calendar {calendar_id}: {exc}") from exc
            if calendar_policy:
                flags, gaps = apply_calendar_policy(flags, gaps, calendar_id, calendar_policy)
            out["calendar_session_status"] = "checked"
            flags = flags.copy()
            flags["exchange_session_date"] = flags["session_date"].dt.strftime("%Y-%m-%d")
            out = out.merge(
                flags[
                    ["asset_id", "exchange_session_date", "is_expected_session", "previous_expected_session_missing"]
                ],
                on=["asset_id", "exchange_session_date"],
                how="left",
                validate="one_to_one",
            )
            if out["is_expected_session"].isna().any():
                raise SilverSessionError("Calendar did not cover every source date")
            for row in gaps.itertuples(index=False):
                issues.append(
                    {
                        "asset_id": asset_id,
                        "provider_symbol": symbol,
                        "source_row": None,
                        "session_date": pd.Timestamp(row.session_date).date().isoformat(),
                        "issue": str(row.issue),
                        "calendar_id": calendar_id,
                    }
                )
            calendar_issue_count = len(gaps)
        else:
            out["calendar_session_status"] = "blocked_invalid_source_dates"
    else:
        issues.append(
            {
                "asset_id": asset_id,
                "provider_symbol": symbol,
                "source_row": None,
                "session_date": None,
                "issue": "calendar_mapping_not_configured",
                "calendar_id": "",
            }
        )
    unexpected = out.get("is_expected_session", pd.Series(pd.NA, index=out.index)).eq(False).fillna(False)
    # A closed exchange date plus a zero-volume, unchanged flat bar is evidence
    # of a provider placeholder. FX and crypto have different volume/session
    # conventions, so a zero-volume bar alone cannot establish that conclusion.
    placeholder = pd.Series(False, index=out.index)
    if unexpected.any() and calendar_id and calendar_id not in {"CRYPTO_24_7", "YAHOO_FX_WEEKDAYS"}:
        chronological = out.sort_values("exchange_session_date")
        scheduled = chronological["is_expected_session"].eq(True).fillna(False)
        prior_dates = chronological["exchange_session_date"].where(scheduled).ffill().shift()
        scheduled_rows = chronological.loc[scheduled]
        valid_close = c.where(~bad_ohlc & ~missing_timezone)
        scheduled_closes = pd.Series(
            valid_close.loc[scheduled_rows.index].to_numpy(), index=scheduled_rows["exchange_session_date"]
        )
        prior_close = prior_dates.map(scheduled_closes)
        placeholder = (
            unexpected
            & v.eq(0)
            & o.eq(h)
            & h.eq(low)
            & low.eq(c)
            & c.eq(prior_close.reindex(out.index))
            & ~bad_ohlc
            & ~missing_timezone
        )
    review_calendar = unexpected & ~placeholder
    critical = invalid_date | dup | bad_ohlc | bad_volume | placeholder
    out["session_quality_status"] = np.where(critical, "quarantined", "review_required")
    out["calendar_exception_review"] = review_calendar
    out["non_trading_placeholder"] = placeholder
    for idx in out.index[placeholder]:
        issues.append(
            {
                "asset_id": asset_id,
                "provider_symbol": symbol,
                "source_row": int(idx),
                "session_date": out.at[idx, "exchange_session_date"],
                "issue": "non_trading_placeholder",
                "calendar_id": calendar_id,
            }
        )
    if calendar_id and not (critical.any() or missing_timezone.any() or calendar_issue_count):
        out["session_quality_status"] = "internally_validated"
    if "previous_expected_session_missing" not in out:
        out["previous_expected_session_missing"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    if "is_expected_session" not in out:
        out["is_expected_session"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    # Eligibility is per field and conditional on a known, valid session.
    # An inconsistent range does not establish which raw price is wrong.
    expected = out["is_expected_session"].eq(True).fillna(False)
    session_ok = expected & ~invalid_date & ~dup & ~missing_timezone
    corrected = out["close_correction_id"].ne("")
    effective_close = out["effective_close"]
    valid_range = (
        np.isfinite(o)
        & np.isfinite(h)
        & np.isfinite(low)
        & np.isfinite(effective_close)
        & o.gt(0)
        & low.gt(0)
        & effective_close.gt(0)
        & h.ge(low)
        & h.ge(o)
        & h.ge(effective_close)
        & low.le(o)
        & low.le(effective_close)
    )
    out["close_eligible"] = session_ok & ((valid_range & ~bad_ohlc) | corrected)
    out["ohlc_eligible"] = out["close_eligible"] & valid_range
    out["volume_eligible"] = session_ok & np.isfinite(v) & v.ge(0)
    out["daily_return_eligible"] = False
    ordered = out.loc[expected].sort_values("exchange_session_date")
    pending_dates = sorted(pd.to_datetime(out.loc[review_calendar, "exchange_session_date"]))
    blocked = pd.Series(False, index=ordered.index)
    if pending_dates:
        current_dates = pd.to_datetime(ordered["exchange_session_date"])
        previous_dates = current_dates.shift()
        for day in pending_dates:
            blocked |= previous_dates.lt(day) & current_dates.gt(day)
    out.loc[ordered.index, "daily_return_eligible"] = (
        ordered["close_eligible"]
        & ordered["close_eligible"].shift(fill_value=False)
        & ~ordered["previous_expected_session_missing"].fillna(True)
        & ~blocked
    )
    out["available_at_utc"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["scheduled_market_close_utc"] = pd.Series(pd.NA, index=out.index, dtype="string")
    if calendar_id and calendar_id not in {"CRYPTO_24_7", "YAHOO_FX_WEEKDAYS"} and expected.any():
        dates = out.loc[expected, "exchange_session_date"]
        schedule = (
            effective_schedule(calendar_history, dates.min(), dates.max(), txse_policy or {})
            if calendar_history
            else mcal.get_calendar(calendar_id).schedule(start_date=dates.min(), end_date=dates.max())
        )
        closes = schedule["market_close"].map(lambda value: value.isoformat())
        closes.index = closes.index.strftime("%Y-%m-%d")
        out.loc[expected, "scheduled_market_close_utc"] = dates.map(closes)
    trailing_missing = 0
    if end_exclusive and calendar_id and expected.any():
        last = pd.Timestamp(out.loc[expected, "exchange_session_date"].max())
        cutoff = pd.Timestamp(end_exclusive)
        start, end = last + pd.Timedelta(days=1), cutoff - pd.Timedelta(days=1)
        if start <= end:
            if calendar_history:
                days = effective_schedule(calendar_history, start, end, txse_policy or {}).index
            elif calendar_id in {"CRYPTO_24_7", "YAHOO_FX_WEEKDAYS"}:
                days = pd.date_range(start, end, freq="D" if calendar_id == "CRYPTO_24_7" else "B")
            else:
                days = mcal.get_calendar(calendar_id).valid_days(start_date=start, end_date=end).tz_localize(None)
            missing_dates = {day.date().isoformat() for day in days}
            for entry in (calendar_policy or {}).get("exceptions", []):
                day = entry["session_date"]
                if entry["calendar_id"] != calendar_id or not start <= pd.Timestamp(day) <= end:
                    continue
                if entry["decision"] == "closed":
                    missing_dates.discard(day)
                elif entry["decision"] == "expected":
                    missing_dates.add(day)
            trailing_missing = len(missing_dates)
            for day in sorted(missing_dates):
                issues.append(
                    {
                        "asset_id": asset_id,
                        "provider_symbol": symbol,
                        "source_row": None,
                        "session_date": day,
                        "issue": "missing_endpoint_session",
                        "calendar_id": calendar_id,
                    }
                )
    out["source_row_number"] = np.arange(len(out), dtype="int64")
    out = out.reset_index(drop=True)
    findings = pd.DataFrame(
        issues, columns=["asset_id", "provider_symbol", "source_row", "session_date", "issue", "calendar_id"]
    )
    report = {
        "rows": len(out),
        "source_rows_preserved": len(out) == len(bronze),
        "invalid_session_dates": int(invalid_date.sum()),
        "duplicate_session_rows": int(dup.sum()),
        "missing_offset_aware_timestamps": int(missing_timezone.sum()),
        "invalid_ohlc_rows": int(bad_ohlc.sum()),
        "negative_volume_rows": int(bad_volume.sum()),
        "calendar_issue_rows": calendar_issue_count,
        "missing_endpoint_sessions": trailing_missing,
        "missing_prior_expected_session_rows": int(out["previous_expected_session_missing"].fillna(False).sum()),
        "calendar_mapped": bool(calendar_id),
        "issue_rows": len(findings),
        "quarantined_rows": int(critical.sum()),
        "non_trading_placeholder_rows": int(placeholder.sum()),
        "calendar_exception_review_rows": int(review_calendar.sum()),
        "close_eligible_rows": int(out["close_eligible"].sum()),
        "ohlc_eligible_rows": int(out["ohlc_eligible"].sum()),
        "daily_return_eligible_rows": int(out["daily_return_eligible"].sum()),
        "production_eligible": False,
    }
    return out, findings, report


def apply_calendar_policy(flags, gaps, calendar_id, policy):
    """Apply approved date-specific rules; review entries leave expected dates intact."""
    relevant = [row for row in policy["exceptions"] if row["calendar_id"] == calendar_id]
    observed = set(pd.to_datetime(flags["session_date"]))
    expected = set(pd.to_datetime(flags.loc[flags["is_expected_session"], "session_date"]))
    expected.update(pd.to_datetime(gaps.loc[gaps["issue"].eq("missing_expected_session"), "session_date"]))
    for row in relevant:
        day = pd.Timestamp(row["session_date"])
        if not min(observed) <= day <= max(observed):
            continue
        if row["decision"] == "expected":
            expected.add(day)
        elif row["decision"] == "closed":
            expected.discard(day)
    expected = sorted(expected)
    preceding = {day: expected[index - 1] if index else None for index, day in enumerate(expected)}
    flags = flags.copy()
    flags["is_expected_session"] = flags["session_date"].isin(expected)
    flags["previous_expected_session_missing"] = flags["session_date"].map(
        lambda day: preceding.get(day) is not None and preceding[day] not in observed
    )
    asset_id, symbol = flags.iloc[0][["asset_id", "provider_symbol"]]
    issues = [
        (asset_id, symbol, day, issue, calendar_id)
        for days, issue in (
            (set(expected) - observed, "missing_expected_session"),
            (observed - set(expected), "unexpected_session"),
        )
        for day in sorted(days)
    ]
    return flags, pd.DataFrame(issues, columns=["asset_id", "provider_symbol", "session_date", "issue", "calendar_id"])


def _selection(assets: list[dict], selectors: list[str]) -> list[dict]:
    if not selectors:
        raise SilverSessionError("Pilot requires explicit --only selectors (no unrestricted full-universe Silver run)")
    wanted = {s.strip() for s in selectors if s.strip()}
    matches = [a for a in assets if wanted.intersection((a["asset_id"], a["ticker"], a["yahoo_ticker"]))]
    available = {x for a in matches for x in (a["asset_id"], a["ticker"], a["yahoo_ticker"])}
    if unknown := wanted - available:
        raise SilverSessionError(f"Unknown or excluded selectors: {sorted(unknown)}")
    for value in wanted:
        if sum(value in (a["asset_id"], a["ticker"], a["yahoo_ticker"]) for a in matches) != 1:
            raise SilverSessionError(f"Ambiguous selector: {value}")
    return sorted(matches, key=lambda x: x["asset_id"])


def build(
    s3: Any,
    bucket: str,
    *,
    source_fingerprint: str,
    bronze_run_id: str,
    silver_run_id: str,
    only: list[str],
    all_assets: bool = False,
    fallback_bronze_run_ids: list[str] | None = None,
    reuse_historical_primary: bool = False,
    calendar_id: str | None = None,
    registry_version: str | None = None,
    correction_version: str | None = None,
    calendar_policy_version: str | None = None,
    execute: bool = False,
) -> dict:
    """Process only complete, cryptographically verified Bronze asset receipts."""
    if not SAFE_RUN.fullmatch(silver_run_id) or not SAFE_RUN.fullmatch(bronze_run_id):
        raise SilverSessionError("Invalid run ID")
    assets, universe_meta = load_universe(s3, bucket, source_fingerprint)
    if all_assets and (only or not registry_version or calendar_id):
        raise SilverSessionError("Full-universe processing requires a registry version and no manual selectors")
    selected = sorted(assets, key=lambda a: a["asset_id"]) if all_assets else _selection(assets, only)
    calendar_policy = None
    if calendar_policy_version:
        if not re.fullmatch(r"[0-9a-f]{64}", calendar_policy_version):
            raise SilverSessionError("Invalid calendar policy version")
        raw_policy = _read(s3, bucket, f"{CALENDAR_POLICY_ROOT}/{calendar_policy_version}.json")
        if _digest(raw_policy) != calendar_policy_version:
            raise SilverSessionError("Calendar policy checksum mismatch")
        calendar_policy = json.loads(raw_policy)
        if calendar_policy.get("schema") != "alpha_edge_calendar_exception_policy_v1":
            raise SilverSessionError("Unsupported calendar policy")
        keys = set()
        for entry in calendar_policy["exceptions"]:
            key = (entry["calendar_id"], entry["session_date"])
            if (
                key in keys
                or not _day(key[1])
                or not entry.get("source_reference")
                or entry["decision"] not in {"review", "expected", "closed"}
                or (entry["decision"] != "review" and entry.get("approval") != "approved")
            ):
                raise SilverSessionError("Invalid or unapproved calendar exception")
            keys.add(key)
    correction = None
    if correction_version:
        if not re.fullmatch(r"[0-9a-f]{64}", correction_version):
            raise SilverSessionError("Invalid correction version")
        raw_correction = _read(s3, bucket, f"{CORRECTION_ROOT}/{correction_version}.json")
        if _digest(raw_correction) != correction_version:
            raise SilverSessionError("Correction evidence checksum mismatch")
        correction = json.loads(raw_correction)
        if (
            correction.get("schema") != "alpha_edge_silver_close_correction_v1"
            or correction.get("source_fingerprint") != source_fingerprint
            or correction.get("asset_id") not in {a["asset_id"] for a in selected}
            or not correction.get("source_reference")
        ):
            raise SilverSessionError("Correction evidence is incompatible with selected inputs")
    if registry_version and calendar_id:
        raise SilverSessionError("Use either a registry version or explicit manual calendar, not both")
    registry_rows = {}
    if registry_version:
        registry = load_registry(s3, bucket, registry_version, source_fingerprint)
        registry_rows = {row["asset_id"]: row for row in registry["assets"]}
        for asset in selected:
            mapping = registry_rows.get(asset["asset_id"])
            if not mapping or mapping.get("provider_symbol") != asset["yahoo_ticker"]:
                raise SilverSessionError(f"Registry asset/provider mismatch: {asset['asset_id']}")
    source_config = json.loads(_read(s3, bucket, f"market/lake_v3/bronze/full_history/runs/{bronze_run_id}/run.json"))
    primary_assets = assets
    if source_config.get("source_fingerprint") != source_fingerprint:
        if not reuse_historical_primary or not registry_version:
            raise SilverSessionError("Bronze run's pinned source snapshot differs")
        primary_assets, _ = load_universe(s3, bucket, source_config["source_fingerprint"])
    if (
        not all_assets
        and not fallback_bronze_run_ids
        and not {a["asset_id"] for a in selected}.issubset(source_config["requested_asset_ids"])
    ):
        raise SilverSessionError("Selected asset not in pinned Bronze run")
    sources = [(bronze_run_id, source_config, {a["asset_id"]: a for a in primary_assets})]
    for fallback in fallback_bronze_run_ids or []:
        if not SAFE_RUN.fullmatch(fallback) or fallback in {run for run, _, _ in sources}:
            raise SilverSessionError("Invalid or repeated fallback Bronze run ID")
        config = json.loads(_read(s3, bucket, f"market/lake_v3/bronze/full_history/runs/{fallback}/run.json"))
        previous_assets, _ = load_universe(s3, bucket, config["source_fingerprint"])
        if config.get("end_exclusive", "") > source_config.get("end_exclusive", ""):
            raise SilverSessionError("Fallback Bronze run extends past the primary cutoff")
        sources.append((fallback, config, {a["asset_id"]: a for a in previous_assets}))
    run_config = {
        "schema": "alpha_edge_silver_session_run_v2",
        "silver_run_id": silver_run_id,
        "bronze_run_id": bronze_run_id,
        "source_fingerprint": source_fingerprint,
        "source_git_commit": universe_meta["source_git_commit"],
        "calendar_library_version": version("pandas-market-calendars"),
        "exchange_calendars_version": version("exchange-calendars"),
        "session_eligibility_policy": "exchange_placeholder_v1",
        "selected_asset_ids": [a["asset_id"] for a in selected],
        "calendar_id": calendar_id or "",
        "status": "candidate_only_not_certified",
        "bronze_sources": [
            {
                "run_id": run,
                "source_fingerprint": cfg["source_fingerprint"],
                "run_config_sha256": _digest(_json_bytes(cfg)),
            }
            for run, cfg, _ in sources
        ],
    }
    if registry_version:
        run_config["registry_version"] = registry_version
        run_config["registry_approval"] = "candidate_not_certified"
    if correction_version:
        run_config["correction_version"] = correction_version
    if calendar_policy_version:
        run_config["calendar_policy_version"] = calendar_policy_version
    root = f"{SILVER_ROOT}/runs/{silver_run_id}"
    existing = _get_optional(s3, bucket, root + "/run.json")
    if existing is not None and json.loads(existing) != run_config:
        raise SilverSessionError("Silver run ID already bound to different inputs")
    if execute and existing is None:
        _put(s3, bucket, root + "/run.json", _json_bytes(run_config), "application/json")
    rows = []
    for index, asset in enumerate(selected, 1):
        if all_assets and (index == 1 or index % 50 == 0):
            print(f"[silver] asset {index}/{len(selected)}", file=sys.stderr, flush=True)
        mapping = registry_rows.get(asset["asset_id"])
        effective_calendar = (mapping.get("calendar_id") if mapping else calendar_id) or None
        # Unmapped assets may be published as review-required candidates; never
        # infer calendars or claim completeness on their behalf.
        receipt = None
        source_run = None
        selected_config = None
        try:
            for run, cfg, records in sources:
                old = records.get(asset["asset_id"])
                if (
                    asset["asset_id"] not in cfg["requested_asset_ids"]
                    or old is None
                    or any(old[field] != asset[field] for field in ("row_id", "yahoo_ticker", "currency_hint"))
                ):
                    continue
                receipt = _completed(s3, bucket, run_id=run, record=asset, verify_hash=True)
                if receipt is not None:
                    source_run, selected_config = run, cfg
                    break
        except BronzeIngestError as exc:
            if not all_assets:
                raise
            rows.append(
                {
                    "asset_id": asset["asset_id"],
                    "ticker": asset["ticker"],
                    "calendar_id": effective_calendar,
                    "status": "source_invalid",
                    "error": str(exc),
                }
            )
            continue
        if receipt is None:
            rows.append(
                {
                    "asset_id": asset["asset_id"],
                    "ticker": asset["ticker"],
                    "status": "source_unavailable",
                    "source": None,
                    "calendar_id": effective_calendar,
                    "reason": "no_compatible_committed_bronze",
                }
            )
            continue
        prefix = f"{root}/assets/{_digest(asset['asset_id'].encode('utf-8'))}"
        asset_correction = correction if correction and correction["asset_id"] == asset["asset_id"] else None
        if asset_correction and asset_correction["source_data_sha256"] != receipt["data"]["sha256"]:
            raise SilverSessionError("Correction evidence does not match the Bronze receipt")
        prior = _get_optional(s3, bucket, prefix + "/manifest.json")
        if prior is not None:
            manifest = json.loads(prior)
            if (
                manifest.get("schema") != SILVER_SCHEMA
                or manifest.get("source_data_sha256") != receipt["data"]["sha256"]
                or manifest.get("calendar_id") != (effective_calendar or "")
                or manifest.get("registry_version", "") != (registry_version or "")
                or manifest.get("bronze_run_id") != source_run
            ):
                raise SilverSessionError(f"Existing Silver asset receipt incompatible: {asset['asset_id']}")
            rows.append(
                {
                    "asset_id": asset["asset_id"],
                    "ticker": asset["ticker"],
                    "status": "already_committed",
                    "source": receipt["data"]["key"],
                    "quality": manifest["quality"],
                    "calendar_id": effective_calendar,
                    "bronze_run_id": source_run,
                }
            )
            continue
        payload = _read(s3, bucket, receipt["data"]["key"])
        bronze_frame = pd.read_parquet(io.BytesIO(payload), engine="pyarrow")
        try:
            candidate, issues, quality = prepare_session_candidate(
                bronze_frame,
                asset_id=asset["asset_id"],
                symbol=asset["yahoo_ticker"],
                calendar_id=effective_calendar,
                close_correction=asset_correction,
                calendar_policy=calendar_policy,
                end_exclusive=source_config.get("end_exclusive"),
                calendar_history=mapping.get("calendar_history") if mapping else None,
                txse_policy=mapping.get("txse_policy") if mapping else None,
            )
        except SilverSessionError as exc:
            if not all_assets:
                raise
            rows.append(
                {
                    "asset_id": asset["asset_id"],
                    "ticker": asset["ticker"],
                    "calendar_id": effective_calendar,
                    "status": "processing_failed",
                    "error": str(exc),
                }
            )
            continue
        manifest = {
            "schema": SILVER_SCHEMA,
            "status": "candidate_only",
            "silver_run_id": silver_run_id,
            "bronze_run_id": source_run,
            "bronze_source_fingerprint": selected_config["source_fingerprint"],
            "asset_id": asset["asset_id"],
            "provider_symbol": asset["yahoo_ticker"],
            "calendar_id": effective_calendar or "",
            "source_manifest_key": f"market/lake_v3/bronze/full_history/runs/{source_run}/assets/{_digest(asset['asset_id'].encode('utf-8'))}/manifest.json",
            "source_data_key": receipt["data"]["key"],
            "source_data_sha256": receipt["data"]["sha256"],
            "quality": quality,
            "outputs": {},
            "production_eligible": False,
        }
        if registry_version:
            manifest["registry_version"] = registry_version
            manifest["calendar_mapping_status"] = mapping["mapping_status"]
            if mapping.get("calendar_history"):
                manifest["calendar_history"] = mapping["calendar_history"]
        if asset_correction:
            manifest["correction_version"] = correction_version
        if calendar_policy_version:
            manifest["calendar_policy_version"] = calendar_policy_version
        if execute:
            for name, df in (("candidate", candidate), ("findings", issues)):
                data = parquet_bytes(df)
                key = f"{prefix}/objects/{name}-{_digest(data)}.parquet"
                _put(s3, bucket, key, data, "application/vnd.apache.parquet")
                manifest["outputs"][name] = {"key": key, "sha256": _digest(data), "bytes": len(data)}
            _put(s3, bucket, prefix + "/manifest.json", _json_bytes(manifest), "application/json")
        rows.append(
            {
                "asset_id": asset["asset_id"],
                "ticker": asset["ticker"],
                "status": "candidate_written" if execute else "dry_run",
                "source": receipt["data"]["key"],
                "quality": quality,
                "calendar_id": effective_calendar,
                "bronze_run_id": source_run,
            }
        )
    report = {
        "schema": "alpha_edge_silver_session_report_v1",
        "silver_run_id": silver_run_id,
        "bronze_run_id": bronze_run_id,
        "source_fingerprint": source_fingerprint,
        "selected": len(selected),
        "processed": sum(x["status"] in {"candidate_written", "already_committed", "dry_run"} for x in rows),
        "missing_bronze": sum(x["status"] == "source_unavailable" for x in rows),
        "failed": sum(x["status"] in {"source_invalid", "processing_failed"} for x in rows),
        "assets": rows,
        "state": "candidate_only",
        "executed": execute,
        "production_eligible": False,
    }
    report["summary"] = {
        "unmapped_assets": sum(not row.get("calendar_id") for row in rows),
        "assets_with_findings": sum(bool(row.get("quality", {}).get("issue_rows")) for row in rows),
        "processed_by_bronze_run": {run: sum(row.get("bronze_run_id") == run for row in rows) for run, _, _ in sources},
        "quality_totals": {
            field: sum(row.get("quality", {}).get(field, 0) for row in rows)
            for field in (
                "rows",
                "issue_rows",
                "quarantined_rows",
                "non_trading_placeholder_rows",
                "calendar_exception_review_rows",
                "close_eligible_rows",
                "ohlc_eligible_rows",
                "daily_return_eligible_rows",
                "missing_prior_expected_session_rows",
                "missing_endpoint_sessions",
            )
        },
    }
    report["summary"]["by_calendar"] = {}
    for row in rows:
        calendar = row.get("calendar_id") or "UNRESOLVED"
        group = report["summary"]["by_calendar"].setdefault(
            calendar, {"assets": 0, "processed": 0, "finding_rows": 0, "missing_endpoint_sessions": 0}
        )
        group["assets"] += 1
        quality = row.get("quality")
        if quality:
            group["processed"] += 1
            group["finding_rows"] += quality["issue_rows"]
            group["missing_endpoint_sessions"] += quality.get("missing_endpoint_sessions", 0)
    if execute:
        key = f"{root}/reports/{uuid.uuid4().hex}.json"
        _put(s3, bucket, key, _json_bytes(report), "application/json")
        report["report_s3_uri"] = f"s3://{bucket}/{key}"
    return report


def verify_candidate_run(s3: Any, bucket: str, silver_run_id: str) -> dict:
    """Independently verify candidate and findings bytes against their manifests."""
    if not SAFE_RUN.fullmatch(silver_run_id):
        raise SilverSessionError("Invalid run ID")
    root = f"{SILVER_ROOT}/runs/{silver_run_id}"
    config = json.loads(_read(s3, bucket, f"{root}/run.json"))
    ids = config["selected_asset_ids"]
    stats = {"selected": len(ids), "verified": 0, "missing": 0, "corrupt": 0}
    exceptions = []
    for asset_id in ids:
        prefix = f"{root}/assets/{_digest(asset_id.encode('utf-8'))}"
        try:
            manifest_bytes = _get_optional(s3, bucket, f"{prefix}/manifest.json")
            if manifest_bytes is None:
                stats["missing"] += 1
                exceptions.append({"asset_id": asset_id, "status": "missing"})
                continue
            head = s3.head_object(Bucket=bucket, Key=f"{prefix}/manifest.json")
            if head["Metadata"].get("sha256") != _digest(manifest_bytes):
                raise SilverSessionError("Candidate manifest hash mismatch")
            manifest = json.loads(manifest_bytes)
            if (
                manifest["asset_id"] != asset_id
                or manifest["silver_run_id"] != silver_run_id
                or manifest["production_eligible"]
                or manifest.get("registry_version") != config.get("registry_version")
                or manifest.get("calendar_policy_version") != config.get("calendar_policy_version")
            ):
                raise SilverSessionError("Candidate manifest identity or release status mismatch")
            outputs = {}
            for name in ("candidate", "findings"):
                ref = manifest["outputs"][name]
                if not ref["key"].startswith(f"{prefix}/objects/{name}-"):
                    raise SilverSessionError("Candidate artifact reference mismatch")
                blob = verified_bytes(s3, bucket, ref)
                if len(blob) != ref["bytes"]:
                    raise SilverSessionError("Candidate artifact length mismatch")
                outputs[name] = pd.read_parquet(io.BytesIO(blob))
            source = _read(s3, bucket, manifest["source_data_key"])
            if _digest(source) != manifest["source_data_sha256"]:
                raise SilverSessionError("Source hash mismatch")
            bronze = pd.read_parquet(io.BytesIO(source))
            candidate = outputs["candidate"]
            pd.testing.assert_frame_equal(
                bronze.reset_index(drop=True), candidate[bronze.columns].reset_index(drop=True), check_dtype=False
            )
            quality = manifest["quality"]
            if len(candidate) != quality["rows"] or len(outputs["findings"]) != quality["issue_rows"]:
                raise SilverSessionError("Candidate or finding row count mismatch")
            for field in ("close_eligible", "ohlc_eligible", "daily_return_eligible"):
                if int(candidate[field].sum()) != quality[f"{field}_rows"]:
                    raise SilverSessionError(f"Eligibility count mismatch: {field}")
            stats["verified"] += 1
        except (AssertionError, KeyError, ValueError, LakehouseError, SilverSessionError) as exc:
            stats["corrupt"] += 1
            exceptions.append({"asset_id": asset_id, "status": "corrupt", "error": str(exc)[:500]})
    return {**stats, "exceptions": exceptions, "state": "pass" if stats["verified"] == len(ids) else "incomplete"}


def main() -> None:
    parser = argparse.ArgumentParser(description="S3-only Silver v3 session/date QA candidate pilot")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--source-fingerprint", help="Override the active S3 universe with a pinned snapshot SHA256")
    parser.add_argument("--bronze-run-id")
    parser.add_argument("--silver-run-id", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--only", help="Comma-separated asset selectors")
    selection.add_argument("--all", action="store_true", help="Process the pinned full universe")
    selection.add_argument("--verify", action="store_true", help="Read back an existing Silver candidate run")
    parser.add_argument(
        "--fallback-bronze-run-id",
        action="append",
        default=[],
        help="Explicit historical Bronze run to reuse for compatible assets",
    )
    parser.add_argument("--calendar", default=None, help="Explicit calendar for one manual pilot")
    parser.add_argument(
        "--reuse-historical-primary",
        action="store_true",
        help="Explicitly reuse compatible prior-snapshot Bronze with a pinned current registry",
    )
    parser.add_argument("--registry-version", default=None, help="SHA256-pinned S3 calendar registry proposal")
    parser.add_argument("--correction-version", default=None, help="SHA256-pinned S3 close correction evidence")
    parser.add_argument("--calendar-policy-version", default=None, help="SHA256-pinned S3 calendar exception policy")
    parser.add_argument("--execute", action="store_true", help="Write candidate outputs to S3")
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    if args.verify:
        if args.execute:
            parser.error("--verify is read-only")
        result = verify_candidate_run(s3, args.bucket, args.silver_run_id)
        print(json.dumps(result, indent=2))
        if result["state"] != "pass":
            raise SystemExit(2)
        return
    if not args.bronze_run_id:
        parser.error("--bronze-run-id is required when building candidates")
    fingerprint = resolve_source_fingerprint(s3, args.bucket, args.source_fingerprint)
    result = build(
        s3,
        args.bucket,
        source_fingerprint=fingerprint,
        bronze_run_id=args.bronze_run_id,
        silver_run_id=args.silver_run_id,
        only=[x.strip() for x in (args.only or "").split(",") if x.strip()],
        all_assets=args.all,
        fallback_bronze_run_ids=args.fallback_bronze_run_id,
        reuse_historical_primary=args.reuse_historical_primary,
        calendar_id=args.calendar,
        registry_version=args.registry_version,
        correction_version=args.correction_version,
        calendar_policy_version=args.calendar_policy_version,
        execute=args.execute,
    )
    display = result if not args.all else {k: v for k, v in result.items() if k != "assets"}
    if args.all:
        display["blocked_assets"] = [
            row
            for row in result["assets"]
            if row["status"] in {"source_unavailable", "source_invalid", "processing_failed"}
            or not row.get("calendar_id")
        ]
    print(json.dumps(display, indent=2, default=str))
    if result["missing_bronze"] or result["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
