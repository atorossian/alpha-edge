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
import uuid
from datetime import date
from typing import Any

import boto3
import numpy as np
import pandas as pd

from alpha_edge.market.ingest_bronze_full_history import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    SAFE_RUN,
    _completed,
    _digest,
    _get_optional,
    _json_bytes,
    _read,
    load_universe,
)
from alpha_edge.market.standardize_bronze_v2 import audit_asset_sessions

SILVER_ROOT = "market/lake_v3/silver/session_candidates"
SILVER_SCHEMA = "alpha_edge_silver_session_candidate_v1"
ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OFFSET = re.compile(r"(?:Z|[+-]\d{2}:\d{2})$")


class SilverSessionError(RuntimeError):
    """Silver source, provenance, identity or candidate validation failure."""


def _put(s3: Any, bucket: str, key: str, data: bytes, content_type: str) -> None:
    """Append-only S3 publication; no overwrites or writes to Bronze."""
    if not key.startswith(SILVER_ROOT + "/"):
        raise SilverSessionError("Refusing writes outside Silver session-candidate prefix")
    # Bronze helper enforces a different namespace; keep its immutability
    # semantics but give Silver a distinct explicit S3 namespace.
    from botocore.exceptions import ClientError

    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata={"sha256": _digest(data)},
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"PreconditionFailed", "412"}:
            raise
        if _read(s3, bucket, key) != data:
            raise SilverSessionError(f"Existing Silver object has different bytes: {key}") from exc


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


def prepare_session_candidate(
    bronze: pd.DataFrame, *, asset_id: str, symbol: str,
    calendar_id: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Preserve every source observation and emit explicit findings.

    Missing calendar mapping is REVIEW_REQUIRED, never a fabricated
    Monday-Friday schedule for equities or worldwide markets.
    """
    needed = {
        "asset_id", "provider_symbol", "session_date", "source_timestamp",
        "source_timezone", "currency_original", "open", "high", "low",
        "close", "adj_close", "volume", "dividends", "stock_splits",
    }
    if missing := needed - set(bronze):
        raise SilverSessionError(f"Bronze columns missing: {sorted(missing)}")
    if bronze.empty or not bronze["asset_id"].eq(asset_id).all() or not bronze["provider_symbol"].eq(symbol).all():
        raise SilverSessionError("Empty Bronze input or wrong canonical asset/provider identity")

    out = bronze.copy(deep=True)
    out["exchange_session_date"] = out["session_date"].map(_day)
    out["source_timestamp_utc"] = out["source_timestamp"].map(_instant)
    out["timestamp_semantics"] = "provider_daily_bar_label_not_verified_market_close"
    out["calendar_id"] = calendar_id or ""
    original = out["session_date"].astype(str)
    dup = original.duplicated(keep=False) & out["exchange_session_date"].notna()
    invalid_date = out["exchange_session_date"].isna()
    missing_timezone = out["source_timestamp_utc"].isna()

    o = pd.to_numeric(out["open"], errors="coerce")
    h = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    c = pd.to_numeric(out["close"], errors="coerce")
    v = pd.to_numeric(out["volume"], errors="coerce")
    # These are QA flags only: provider price values never change.
    bad_ohlc = (
        ~np.isfinite(o) | ~np.isfinite(h) | ~np.isfinite(low) | ~np.isfinite(c)
        | o.le(0) | h.le(0) | low.le(0) | c.le(0)
        | h.lt(low) | h.lt(o) | h.lt(c) | low.gt(o) | low.gt(c)
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
            issues.append({
                "asset_id": asset_id, "provider_symbol": symbol,
                "source_row": int(out.index.get_loc(idx)),
                "session_date": str(out.at[idx, "session_date"]),
                "issue": reason, "calendar_id": calendar_id or "",
            })
    out["calendar_session_status"] = "not_checked"
    calendar_issue_count = 0
    if calendar_id:
        # Never mask malformed dates/duplicates by passing them to the
        # existing v2 calendar audit, which expects valid unique sessions.
        if not invalid_date.any() and not dup.any():
            try:
                flags, gaps = audit_asset_sessions(
                    out[["asset_id", "provider_symbol", "session_date"]],
                    calendars={symbol: calendar_id},
                )
            except Exception as exc:
                raise SilverSessionError(f"Unusable explicitly supplied calendar {calendar_id}: {exc}") from exc
            out["calendar_session_status"] = "checked"
            flags = flags.copy()
            flags["exchange_session_date"] = flags["session_date"].dt.strftime("%Y-%m-%d")
            out = out.merge(
                flags[["asset_id", "exchange_session_date", "is_expected_session", "previous_expected_session_missing"]],
                on=["asset_id", "exchange_session_date"],
                how="left",
                validate="one_to_one",
            )
            if out["is_expected_session"].isna().any():
                raise SilverSessionError("Calendar did not cover every source date")
            for row in gaps.itertuples(index=False):
                issues.append({
                    "asset_id": asset_id, "provider_symbol": symbol,
                    "source_row": None, "session_date": pd.Timestamp(row.session_date).date().isoformat(),
                    "issue": str(row.issue), "calendar_id": calendar_id,
                })
            calendar_issue_count = len(gaps)
        else:
            out["calendar_session_status"] = "blocked_invalid_source_dates"
    else:
        issues.append({
            "asset_id": asset_id, "provider_symbol": symbol, "source_row": None,
            "session_date": None, "issue": "calendar_mapping_not_configured", "calendar_id": "",
        })
    critical = invalid_date | dup | bad_ohlc | bad_volume
    out["session_quality_status"] = np.where(critical, "quarantined", "review_required")
    if calendar_id and not (critical.any() or missing_timezone.any() or calendar_issue_count):
        out["session_quality_status"] = "internally_validated"
    if "previous_expected_session_missing" not in out:
        out["previous_expected_session_missing"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    if "is_expected_session" not in out:
        out["is_expected_session"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
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
        "missing_prior_expected_session_rows": int(out["previous_expected_session_missing"].fillna(False).sum()),
        "calendar_mapped": bool(calendar_id),
        "issue_rows": len(findings),
        "quarantined_rows": int(critical.sum()),
        "production_eligible": False,
    }
    return out, findings, report


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
    s3: Any, bucket: str, *, source_fingerprint: str,
    bronze_run_id: str, silver_run_id: str, only: list[str],
    calendar_id: str | None = None, execute: bool = False
) -> dict:
    """Process only complete, cryptographically verified Bronze asset receipts."""
    if not SAFE_RUN.fullmatch(silver_run_id) or not SAFE_RUN.fullmatch(bronze_run_id):
        raise SilverSessionError("Invalid run ID")
    assets, universe_meta = load_universe(s3, bucket, source_fingerprint)
    selected = _selection(assets, only)
    source_config = json.loads(_read(
        s3, bucket, f"market/lake_v3/bronze/full_history/runs/{bronze_run_id}/run.json"
    ))
    if source_config.get("source_fingerprint") != source_fingerprint:
        raise SilverSessionError("Bronze run's pinned source snapshot differs")
    if not set(a["asset_id"] for a in selected).issubset(source_config["requested_asset_ids"]):
        raise SilverSessionError("Selected asset not in pinned Bronze run")
    run_config = {
        "schema": "alpha_edge_silver_session_run_v1",
        "silver_run_id": silver_run_id,
        "bronze_run_id": bronze_run_id,
        "source_fingerprint": source_fingerprint,
        "source_git_commit": universe_meta["source_git_commit"],
        "selected_asset_ids": [a["asset_id"] for a in selected],
        "calendar_id": calendar_id or "",
        "status": "candidate_only_not_certified",
    }
    root = f"{SILVER_ROOT}/runs/{silver_run_id}"
    existing = _get_optional(s3, bucket, root + "/run.json")
    if existing is not None and json.loads(existing) != run_config:
        raise SilverSessionError("Silver run ID already bound to different inputs")
    if execute and existing is None:
        _put(s3, bucket, root + "/run.json", _json_bytes(run_config), "application/json")
    rows = []
    for asset in selected:
        receipt = _completed(s3, bucket, run_id=bronze_run_id, record=asset, verify_hash=True)
        if receipt is None:
            rows.append({
                "asset_id": asset["asset_id"], "ticker": asset["ticker"],
                "status": "source_unavailable", "source": None,
            })
            continue
        prefix = f"{root}/assets/{_digest(asset['asset_id'].encode('utf-8'))}"
        prior = _get_optional(s3, bucket, prefix + "/manifest.json")
        if prior is not None:
            manifest = json.loads(prior)
            if (
                manifest.get("schema") != SILVER_SCHEMA
                or manifest.get("source_data_sha256") != receipt["data"]["sha256"]
                or manifest.get("calendar_id") != (calendar_id or "")
            ):
                raise SilverSessionError(f"Existing Silver asset receipt incompatible: {asset['asset_id']}")
            rows.append({
                "asset_id": asset["asset_id"], "ticker": asset["ticker"],
                "status": "already_committed", "source": receipt["data"]["key"],
                "quality": manifest["quality"],
            })
            continue
        payload = _read(s3, bucket, receipt["data"]["key"])
        bronze_frame = pd.read_parquet(io.BytesIO(payload), engine="pyarrow")
        candidate, issues, quality = prepare_session_candidate(
            bronze_frame, asset_id=asset["asset_id"],
            symbol=asset["yahoo_ticker"], calendar_id=calendar_id,
        )
        manifest = {
            "schema": SILVER_SCHEMA, "status": "candidate_only",
            "silver_run_id": silver_run_id, "bronze_run_id": bronze_run_id,
            "asset_id": asset["asset_id"], "provider_symbol": asset["yahoo_ticker"],
            "calendar_id": calendar_id or "",
            "source_manifest_key": f"market/lake_v3/bronze/full_history/runs/{bronze_run_id}/assets/{_digest(asset['asset_id'].encode('utf-8'))}/manifest.json",
            "source_data_key": receipt["data"]["key"],
            "source_data_sha256": receipt["data"]["sha256"],
            "quality": quality, "outputs": {},
            "production_eligible": False,
        }
        if execute:
            for name, df in (("candidate", candidate), ("findings", issues)):
                buffer = io.BytesIO()
                df.to_parquet(buffer, engine="pyarrow", index=False)
                data = buffer.getvalue()
                key = f"{prefix}/objects/{name}-{_digest(data)}.parquet"
                _put(s3, bucket, key, data, "application/vnd.apache.parquet")
                manifest["outputs"][name] = {"key": key, "sha256": _digest(data), "bytes": len(data)}
            _put(s3, bucket, prefix + "/manifest.json", _json_bytes(manifest), "application/json")
        rows.append({
            "asset_id": asset["asset_id"], "ticker": asset["ticker"],
            "status": "candidate_written" if execute else "dry_run",
            "source": receipt["data"]["key"], "quality": quality,
        })
    report = {
        "schema": "alpha_edge_silver_session_report_v1",
        "silver_run_id": silver_run_id,
        "bronze_run_id": bronze_run_id,
        "source_fingerprint": source_fingerprint,
        "selected": len(selected),
        "processed": sum(x["status"] in {"candidate_written", "already_committed", "dry_run"} for x in rows),
        "missing_bronze": sum(x["status"] == "source_unavailable" for x in rows),
        "assets": rows,
        "state": "candidate_only",
        "executed": execute,
        "production_eligible": False,
    }
    if execute:
        key = f"{root}/reports/{uuid.uuid4().hex}.json"
        _put(s3, bucket, key, _json_bytes(report), "application/json")
        report["report_s3_uri"] = f"s3://{bucket}/{key}"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="S3-only Silver v3 session/date QA candidate pilot")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--bronze-run-id", required=True)
    parser.add_argument("--silver-run-id", required=True)
    parser.add_argument("--only", required=True, help="Comma-separated pilot selectors")
    parser.add_argument("--calendar", default=None, help="Explicit approved pandas-market-calendars ID")
    parser.add_argument("--execute", action="store_true", help="Write candidate outputs to S3")
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    result = build(
        s3, args.bucket, source_fingerprint=args.source_fingerprint,
        bronze_run_id=args.bronze_run_id, silver_run_id=args.silver_run_id,
        only=[x.strip() for x in args.only.split(",") if x.strip()],
        calendar_id=args.calendar, execute=args.execute,
    )
    print(json.dumps(result, indent=2, default=str))
    if result["missing_bronze"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
