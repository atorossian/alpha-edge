"""Full-history Yahoo Finance -> immutable S3 Bronze, using the CURRENT universe.

Inputs: ONLY the five versioned universe-configuration CSVs snapshotted in S3.
No historical market data or v1/v2 lake artifacts are read. No local files,
local data caches, FX conversion, adjusted-return calculation, or repair.

Reuses acquire_yahoo_bars from ingest_bronze_yahoo_v2 (the source adapter).
DO NOT import ingest_market_data.py: it creates a local cache on import and
writes transformed legacy outputs. This job has a separate S3-only contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import re
import time
import uuid
from datetime import date, datetime, timezone
from typing import Any, Callable

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.market.ingest_bronze_yahoo_v2 import acquire_yahoo_bars
from alpha_edge.market.lakehouse_v3.registry_v3 import resolve_source_fingerprint
from alpha_edge.market.lakehouse_v3.storage import LakehouseError, parquet_bytes, put_immutable

DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
SNAPSHOT_ROOT = "market/lake_v3/config/source_snapshots"
BRONZE_ROOT = "market/lake_v3/bronze/full_history"
SOURCE_NAMES = (
    "universe.csv",
    "universe_overrides.csv",
    "asset_excluded.csv",
    "ingest_force_refresh.csv",
    "universe_manual_additions.csv",
)
SAFE_RUN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,79}$")
SHA_RE = re.compile(r"^[a-f0-9]{64}$")
REQUIRED_BARS = {
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


class BronzeIngestError(RuntimeError):
    """Full-history Bronze contract or S3 publication failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(s3: Any, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def _get_optional(s3: Any, bucket: str, key: str) -> bytes | None:
    try:
        return _read(s3, bucket, key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise


def _put_immutable(s3: Any, bucket: str, key: str, payload: bytes, content_type: str) -> None:
    try:
        put_immutable(s3, bucket, key, payload, content_type=content_type, scope=BRONZE_ROOT)
    except LakehouseError as exc:
        raise BronzeIngestError(str(exc)) from exc


def _source_records(payload: bytes, filename: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise BronzeIngestError(f"{filename}: invalid CSV column headers")
        records = list(reader)
        if any(None in record for record in records):
            raise BronzeIngestError(f"{filename}: malformed CSV rows")
        return records
    except (ValueError, UnicodeDecodeError, csv.Error) as exc:
        raise BronzeIngestError(f"{filename}: invalid UTF-8 CSV") from exc


def load_universe(s3: Any, bucket: str, fingerprint: str) -> tuple[list[dict], dict]:
    """Read and cryptographically verify ONLY the five current S3 configs."""
    if not SHA_RE.fullmatch(fingerprint):
        raise BronzeIngestError("Source snapshot must be a 64-character SHA256 fingerprint")
    source_files = {}
    raw_files = {}
    source_commits = set()
    for filename in SOURCE_NAMES:
        key = f"{SNAPSHOT_ROOT}/{fingerprint}/{filename}"
        result = s3.get_object(Bucket=bucket, Key=key)
        blob = result["Body"].read()
        sha = _digest(blob)
        metadata = result.get("Metadata", {})
        if metadata.get("sha256") != sha:
            raise BronzeIngestError(f"{filename}: source S3 checksum metadata mismatch")
        if metadata.get("source-git-commit"):
            source_commits.add(metadata["source-git-commit"])
        source_files[filename] = {"bytes": len(blob), "sha256": sha}
        raw_files[filename] = _source_records(blob, filename)
    if len(source_commits) != 1:
        raise BronzeIngestError("Snapshot sources do not pin one exact Git commit")
    if _digest(_json_bytes(source_files)) != fingerprint:
        raise BronzeIngestError("Five-file source snapshot fingerprint mismatch")
    source = raw_files["universe.csv"]
    required = {"asset_id", "row_id", "ticker", "yahoo_ticker", "include", "currency"}
    if not source or required - set(source[0]):
        raise BronzeIngestError(f"universe.csv missing {sorted(required - set(source[0]) if source else required)}")
    by_row = {}
    included = []
    excluded = []
    for row in source:
        if row["include"] not in {"0", "1"}:
            raise BronzeIngestError(f"Invalid include value for {row.get('ticker')}")
        asset_id, row_id = row["asset_id"].strip(), row["row_id"].strip()
        if not asset_id or not row_id or row_id in by_row:
            raise BronzeIngestError(f"Duplicate/missing source row identifier: {row_id}")
        by_row[row_id] = row
        if row["include"] == "1":
            included.append(
                {
                    "asset_id": asset_id,
                    "row_id": row_id,
                    "ticker": row["ticker"].strip(),
                    "yahoo_ticker": row["yahoo_ticker"].strip(),
                    "currency_hint": row["currency"].strip() or None,
                }
            )
        else:
            excluded.append(row)
    if len({x["asset_id"] for x in included}) != len(included):
        raise BronzeIngestError("Duplicate included canonical asset_id")
    if len({x["asset_id"] for x in source}) != len(source):
        raise BronzeIngestError("Duplicate canonical asset_id in universe")
    # Supporting configs are validated, NOT used to resurrect excluded assets.
    for filename in ("universe_overrides.csv", "asset_excluded.csv"):
        for row in raw_files[filename]:
            rid = row.get("target_row_id", "")
            if rid and rid not in by_row:
                raise BronzeIngestError(f"{filename}: orphaned target_row_id {rid}")
            if filename == "asset_excluded.csv" and rid and by_row[rid]["include"] == "1":
                raise BronzeIngestError(f"Excluded/included conflict: {rid}")
    ticker_set = {x["ticker"] for x in source}
    for filename in ("universe_manual_additions.csv", "ingest_force_refresh.csv"):
        for row in raw_files[filename]:
            if row.get("ticker") not in ticker_set:
                raise BronzeIngestError(f"{filename}: unknown current universe ticker {row.get('ticker')}")
    # The resolved Yahoo symbol in universe.csv is authoritative; do not
    # infer historical alias dates from conflicting manual-additions rows.
    warnings = []
    for row in raw_files["universe_manual_additions.csv"]:
        candidates = [a for a in included if a["ticker"] == row.get("ticker")]
        if (
            row.get("lock_yahoo_ticker") == "1"
            and len(candidates) == 1
            and candidates[0]["yahoo_ticker"] != row.get("yahoo_ticker")
        ):
            warnings.append(
                {
                    "ticker": row.get("ticker"),
                    "reason": "manual_locked_symbol_disagrees_with_resolved_universe",
                    "resolved_yahoo_ticker": candidates[0]["yahoo_ticker"],
                }
            )
    metadata = {
        "source_fingerprint": fingerprint,
        "source_git_commit": next(iter(source_commits)),
        "source_file_hashes": source_files,
        "universe_total": len(source),
        "included": len(included),
        "excluded": len(excluded),
        "configuration_warnings": warnings,
    }
    return sorted(included, key=lambda x: x["asset_id"]), metadata


def _target_prefix(run_id: str, asset_id: str) -> str:
    token = _digest(asset_id.encode("utf-8"))
    return f"{BRONZE_ROOT}/runs/{run_id}/assets/{token}"


def _check_bars(frame: pd.DataFrame, record: dict) -> dict:
    """Inspect raw bars without repairing, filling or dropping any observation."""
    missing = REQUIRED_BARS - set(frame)
    if missing or frame.empty:
        raise BronzeIngestError(f"{record['yahoo_ticker']}: empty/malformed provider bars: {sorted(missing)}")
    if not frame["asset_id"].eq(record["asset_id"]).all():
        raise BronzeIngestError("Provider bar asset identity mismatch")
    if not frame["provider_symbol"].eq(record["yahoo_ticker"]).all():
        raise BronzeIngestError("Provider bar ticker identity mismatch")
    if frame["session_date"].isna().any():
        raise BronzeIngestError("Provider returned missing session labels")
    # Retain flagged anomalies: Bronze is not a repaired dataset.
    numeric = {
        field: pd.to_numeric(frame[field], errors="coerce")
        for field in ("open", "high", "low", "close", "adj_close", "volume")
    }
    close = numeric["close"]
    sessions = frame["session_date"].astype(str)
    return {
        "rows": int(len(frame)),
        "first_session": sessions.min(),
        "last_session": sessions.max(),
        "duplicate_session_rows": int(sessions.duplicated(keep=False).sum()),
        "invalid_close_rows": int((~np.isfinite(close) | close.le(0)).sum()),
        "missing_adjusted_close_rows": int(numeric["adj_close"].isna().sum()),
        "negative_volume_rows": int(numeric["volume"].lt(0).sum()),
        "unsorted_sessions": bool(not sessions.is_monotonic_increasing),
    }


def _write_run(s3: Any, bucket: str, *, run_id: str, run_config: dict, execute: bool) -> None:
    key = f"{BRONZE_ROOT}/runs/{run_id}/run.json"
    existing = _get_optional(s3, bucket, key)
    if existing is not None:
        current = json.loads(existing)
        if current != run_config:
            raise BronzeIngestError(f"Run ID already belongs to a different selection/configuration: {run_id}")
    elif execute:
        _put_immutable(s3, bucket, key, _json_bytes(run_config), "application/json")


def _completed(s3: Any, bucket: str, *, run_id: str, record: dict, verify_hash: bool) -> dict | None:
    prefix = _target_prefix(run_id, record["asset_id"])
    blob = _get_optional(s3, bucket, prefix + "/manifest.json")
    if blob is None:
        return None
    manifest = json.loads(blob)
    if (
        manifest.get("schema") != "alpha_edge_full_history_asset_v1"
        or manifest.get("asset_id") != record["asset_id"]
        or manifest.get("provider_symbol") != record["yahoo_ticker"]
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "committed"
    ):
        raise BronzeIngestError(f"Existing asset commit is incompatible: {prefix}")
    data_ref = manifest.get("data", {})
    data_key = data_ref.get("key", "")
    if not data_key.startswith(prefix + "/objects/") or not data_key.endswith(".parquet"):
        raise BronzeIngestError(f"Existing asset data key unsafe: {data_key}")
    if not SHA_RE.fullmatch(str(data_ref.get("sha256", ""))):
        raise BronzeIngestError(f"Existing asset data SHA malformed: {prefix}")
    if verify_hash:
        actual = _read(s3, bucket, data_key)
        if _digest(actual) != data_ref["sha256"] or len(actual) != data_ref.get("bytes"):
            raise BronzeIngestError(f"Existing Bronze source Parquet corrupt: {data_key}")
        restored = pd.read_parquet(io.BytesIO(actual), engine="pyarrow")
        if len(restored) != manifest.get("validation", {}).get("rows"):
            raise BronzeIngestError(f"Existing Bronze row count mismatch: {data_key}")
    else:
        head = s3.head_object(Bucket=bucket, Key=data_key)
        if (
            head.get("ContentLength") != data_ref.get("bytes")
            or head.get("Metadata", {}).get("sha256") != data_ref["sha256"]
        ):
            raise BronzeIngestError(f"Existing Bronze Parquet metadata mismatch: {data_key}")
    return manifest


def _acquire_one(
    s3: Any,
    bucket: str,
    *,
    run_id: str,
    record: dict,
    end: str,
    retry_count: int,
    retry_delay: float,
    ticker_factory: Callable | None,
    sleep: Callable[[float], None],
) -> dict:
    asset_id, symbol = record["asset_id"], record["yahoo_ticker"]
    if not symbol:
        raise BronzeIngestError(f"{asset_id}: missing resolved Yahoo ticker")
    failure = None
    for attempt in range(1, retry_count + 2):
        try:
            bars, provenance = acquire_yahoo_bars(
                asset_id,
                symbol,
                start=None,
                end=end,
                batch_id=run_id,
                ticker_factory=ticker_factory,
                currency_hint=record["currency_hint"],
                full_history=True,
            )
            validation = _check_bars(bars, record)
            payload = parquet_bytes(bars)
            prefix = _target_prefix(run_id, asset_id)
            sha = _digest(payload)
            data_key = f"{prefix}/objects/{sha}.parquet"
            _put_immutable(s3, bucket, data_key, payload, "application/vnd.apache.parquet")
            manifest = {
                "schema": "alpha_edge_full_history_asset_v1",
                "run_id": run_id,
                "asset_id": asset_id,
                "broker_ticker": record["ticker"],
                "provider_symbol": symbol,
                "status": "committed",
                "period_requested": "max",
                "interval": "1d",
                "end_exclusive": end,
                "source_values_preserved": True,
                "data": {"key": data_key, "sha256": sha, "bytes": len(payload)},
                "validation": validation,
                "provenance": provenance,
            }
            _put_immutable(
                s3,
                bucket,
                prefix + "/manifest.json",
                _json_bytes(manifest),
                "application/json",
            )
            _completed(s3, bucket, run_id=run_id, record=record, verify_hash=False)
            return {"asset_id": asset_id, "ticker": symbol, "status": "success", **validation}
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            failure = exc
            if attempt <= retry_count:
                sleep(retry_delay * (2 ** (attempt - 1)))
    raise BronzeIngestError(f"{asset_id} ({symbol}): {type(failure).__name__}: {failure}")


def _record_failure(s3: Any, bucket: str, run_id: str, record: dict, error: Exception) -> None:
    key = (
        f"{_target_prefix(run_id, record['asset_id'])}/failures/"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex}.json"
    )
    report = {
        "schema": "alpha_edge_bronze_failure_v1",
        "run_id": run_id,
        "asset_id": record["asset_id"],
        "provider_symbol": record["yahoo_ticker"],
        "error_type": type(error).__name__,
        "error": str(error)[:2000],
        "occurred_at_utc": _utc_now(),
    }
    _put_immutable(s3, bucket, key, _json_bytes(report), "application/json")


def _audit_event(
    s3: Any,
    bucket: str,
    *,
    run_id: str,
    status: str,
    fingerprint: str,
    end: str,
    record: dict | None = None,
    output_keys: list[str] | None = None,
    metadata: dict | None = None,
    error: str | None = None,
) -> str:
    """Immutable S3 audit, mirroring Alpha Edge's existing build_dataset event fields.

    Unlike core.audit.write_audit_event, this stays inside the dedicated
    Bronze prefix and never mutates the legacy engine/v1 audit namespace.
    Per-asset events survive interruptions and are append-only on resume.
    """
    now = datetime.now(timezone.utc)
    audit_id = f"bronze_{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex}"
    key = f"{BRONZE_ROOT}/runs/{run_id}/audit/dt={now:%Y-%m-%d}/{audit_id}.json"
    payload = {
        "schema": "alpha_edge_bronze_audit_v1",
        "audit_id": audit_id,
        "run_id": run_id,
        "event_type": "build_dataset",
        "entity_type": "bronze_full_history",
        "entity_id": record["asset_id"] if record else None,
        "source_script": "ingest_bronze_full_history.py",
        "source_mode": "yahoo_finance_full_history",
        "created_at_utc": now.isoformat(),
        "status": status,
        "input_args": {
            "source_fingerprint": fingerprint,
            "end_exclusive": end,
            "asset_id": record["asset_id"] if record else None,
            "provider_symbol": record["yahoo_ticker"] if record else None,
        },
        "output_keys": output_keys or [],
        "metadata": metadata or {},
        "error": error,
    }
    _put_immutable(s3, bucket, key, _json_bytes(payload), "application/json")
    return key


def _print_progress(
    *,
    done: int,
    total: int,
    ok: int,
    resumed: int,
    failed: int,
    started_monotonic: float,
    progress_every: int,
    force: bool = False,
) -> None:
    """Unbuffered, observed progress; no speculative ETA."""
    if not force and done != 1 and done % progress_every != 0 and done != total:
        return
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    rate = done / max(elapsed, 1e-6)
    print(
        f"[bronze][progress] done={done}/{total} ({100 * done / total:.1f}%) "
        f"ok={ok} resumed={resumed} failed={failed} "
        f"rate={rate:.2f} assets/s elapsed={elapsed / 60:.1f}m",
        flush=True,
    )


def ingest(
    s3: Any,
    bucket: str,
    *,
    fingerprint: str,
    run_id: str,
    end: str,
    execute: bool = False,
    only: list[str] | None = None,
    retry_count: int = 2,
    retry_delay: float = 2.0,
    ticker_factory: Callable | None = None,
    sleep: Callable[[float], None] = time.sleep,
    pause_seconds: float = 0.5,
    progress_every: int = 10,
    print_first_failures: int = 20,
) -> dict:
    if not SAFE_RUN.fullmatch(run_id):
        raise BronzeIngestError("run_id must be 3–80 lowercase letters/digits/hyphens/underscores")
    if date.fromisoformat(end) > datetime.now(timezone.utc).date():
        raise BronzeIngestError("Exclusive end date must not exceed current UTC date")
    if retry_count < 0 or retry_count > 8 or retry_delay < 0 or pause_seconds < 0:
        raise BronzeIngestError("Invalid retry or pacing configuration")
    if progress_every <= 0 or print_first_failures < 0:
        raise BronzeIngestError("Invalid progress reporting configuration")
    all_assets, source = load_universe(s3, bucket, fingerprint)
    selected = all_assets
    if only:
        requested = {x.strip() for x in only if x.strip()}
        matches = [
            a
            for a in all_assets
            if a["asset_id"] in requested or a["ticker"] in requested or a["yahoo_ticker"] in requested
        ]
        found = {value for a in matches for value in (a["asset_id"], a["ticker"], a["yahoo_ticker"])}
        unknown = requested - found
        if unknown:
            raise BronzeIngestError(f"Unknown or excluded asset selector(s): {sorted(unknown)}")
        # Never silently select several listings via an ambiguous bare ticker.
        for requested_name in requested:
            if sum(requested_name in (a["ticker"], a["yahoo_ticker"], a["asset_id"]) for a in matches) > 1:
                raise BronzeIngestError(f"Ambiguous asset selector {requested_name}; use asset_id")
        selected = sorted(matches, key=lambda a: a["asset_id"])
    config = {
        "schema": "alpha_edge_full_history_run_v1",
        "run_id": run_id,
        "source_fingerprint": fingerprint,
        "source_git_commit": source["source_git_commit"],
        "end_exclusive": end,
        "requested_asset_ids": [a["asset_id"] for a in selected],
        "universe_included_count": source["included"],
        "price_adjustment": "none_by_our_pipeline",
        "provider_parameters": {
            "provider": "yahoo_finance",
            "interval": "1d",
            "period": "max",
            "auto_adjust": False,
            "actions": True,
            "repair": False,
        },
    }
    _write_run(s3, bucket, run_id=run_id, run_config=config, execute=execute)
    if not execute:
        return {
            "state": "dry_run",
            "source": source,
            "run_id": run_id,
            "selected": len(selected),
            "writes": 0,
            "end_exclusive": end,
        }
    completed = []
    failures = []
    skipped = []
    started = _utc_now()
    started_monotonic = time.monotonic()
    total = len(selected)
    print(
        f"\\n=== BRONZE FULL HISTORY INGEST ===\\n"
        f"run_id:          {run_id}\\n"
        f"bucket:          {bucket}\\n"
        f"snapshot:        {fingerprint}\\n"
        f"universe_total:  {source['universe_total']}\\n"
        f"included_total:  {source['included']}\\n"
        f"selected:        {total}\\n"
        f"end_exclusive:   {end}\\n"
        f"source:          Yahoo Finance 1d (period=max)\\n"
        f"audit:           s3://{bucket}/{BRONZE_ROOT}/runs/{run_id}/audit/\\n",
        flush=True,
    )
    _audit_event(
        s3,
        bucket,
        run_id=run_id,
        status="started",
        fingerprint=fingerprint,
        end=end,
        metadata={"selected": total, "universe_included_count": source["included"]},
        output_keys=[f"{BRONZE_ROOT}/runs/{run_id}/run.json"],
    )
    for done, record in enumerate(selected, start=1):
        aid = record["asset_id"]
        symbol = record["yahoo_ticker"]
        try:
            existing = _completed(s3, bucket, run_id=run_id, record=record, verify_hash=False)
            if existing:
                skipped.append({"asset_id": aid, "ticker": symbol})
                _audit_event(
                    s3,
                    bucket,
                    run_id=run_id,
                    status="already_committed",
                    fingerprint=fingerprint,
                    end=end,
                    record=record,
                    output_keys=[
                        existing["data"]["key"],
                        f"{_target_prefix(run_id, aid)}/manifest.json",
                    ],
                    metadata=existing.get("validation", {}),
                )
            else:
                result = _acquire_one(
                    s3,
                    bucket,
                    run_id=run_id,
                    record=record,
                    end=end,
                    retry_count=retry_count,
                    retry_delay=retry_delay,
                    ticker_factory=ticker_factory,
                    sleep=sleep,
                )
                completed.append(result)
                manifest = _completed(s3, bucket, run_id=run_id, record=record, verify_hash=False)
                if manifest is None:
                    raise BronzeIngestError(f"Committed manifest missing for {aid}")
                _audit_event(
                    s3,
                    bucket,
                    run_id=run_id,
                    status="success",
                    fingerprint=fingerprint,
                    end=end,
                    record=record,
                    output_keys=[
                        manifest["data"]["key"],
                        f"{_target_prefix(run_id, aid)}/manifest.json",
                    ],
                    metadata=manifest["validation"],
                )
        except (KeyboardInterrupt, SystemExit):
            print(
                f"\\n[bronze][interrupted] done={done - 1}/{total}; rerun identical command to resume.\\n",
                flush=True,
            )
            _audit_event(
                s3,
                bucket,
                run_id=run_id,
                status="interrupted",
                fingerprint=fingerprint,
                end=end,
                record=record,
                metadata={"processed_before_interrupt": done - 1},
            )
            raise
        except Exception as exc:
            failure = {
                "asset_id": aid,
                "ticker": symbol,
                "error": f"{type(exc).__name__}: {exc}"[:2000],
            }
            failures.append(failure)
            _record_failure(s3, bucket, run_id, record, exc)
            _audit_event(
                s3,
                bucket,
                run_id=run_id,
                status="failed",
                fingerprint=fingerprint,
                end=end,
                record=record,
                error=failure["error"],
                metadata={"selected_index": done, "selected_total": total},
            )
            if len(failures) <= print_first_failures:
                print(
                    f"[bronze][fail][{len(failures)}] "
                    f"{done}/{total} asset_id={aid} yahoo={symbol} "
                    f"error={failure['error'][:320]}",
                    flush=True,
                )
        _print_progress(
            done=done,
            total=total,
            ok=len(completed),
            resumed=len(skipped),
            failed=len(failures),
            started_monotonic=started_monotonic,
            progress_every=progress_every,
        )
        if pause_seconds:
            sleep(pause_seconds)
    result = {
        "schema": "alpha_edge_full_history_report_v1",
        "run_id": run_id,
        "source_fingerprint": fingerprint,
        "end_exclusive": end,
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
        "selected": len(selected),
        "succeeded": len(completed),
        "already_committed": len(skipped),
        "failed": len(failures),
        "state": "complete" if not failures else "completed_with_failures",
        "failure_details": failures,
        "successes": completed,
    }
    key = f"{BRONZE_ROOT}/runs/{run_id}/reports/{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex}.json"
    _put_immutable(s3, bucket, key, _json_bytes(result), "application/json")
    result["report_s3_uri"] = f"s3://{bucket}/{key}"
    result["audit_final_key"] = _audit_event(
        s3,
        bucket,
        run_id=run_id,
        status="complete" if not failures else "completed_with_failures",
        fingerprint=fingerprint,
        end=end,
        output_keys=[key],
        metadata={
            "selected": len(selected),
            "succeeded": len(completed),
            "already_committed": len(skipped),
            "failed": len(failures),
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 2),
        },
    )
    _print_progress(
        done=total,
        total=total,
        ok=len(completed),
        resumed=len(skipped),
        failed=len(failures),
        started_monotonic=started_monotonic,
        progress_every=progress_every,
        force=True,
    )
    print(
        f"[bronze][done] state={result['state']} "
        f"report={result['report_s3_uri']} "
        f"audit=s3://{bucket}/{result['audit_final_key']}",
        flush=True,
    )
    return result


def verify(s3: Any, bucket: str, *, fingerprint: str, run_id: str, only: list[str] | None = None) -> dict:
    """Independent full-byte/Parquet check of every selected S3 asset commit."""
    config_key = f"{BRONZE_ROOT}/runs/{run_id}/run.json"
    config = json.loads(_read(s3, bucket, config_key))
    if config.get("source_fingerprint") != fingerprint:
        raise BronzeIngestError("Run snapshot fingerprint differs from verification request")
    assets, _ = load_universe(s3, bucket, fingerprint)
    current = {a["asset_id"]: a for a in assets}
    wanted = config.get("requested_asset_ids", [])
    if not isinstance(wanted, list) or len(wanted) != len(set(wanted)):
        raise BronzeIngestError("Invalid run-selected asset IDs")
    if only:
        selected = {s.strip() for s in only}
        wanted = [i for i in wanted if i in selected or current.get(i, {}).get("ticker") in selected]
    stats = {"expected": len(wanted), "committed": 0, "missing": 0, "corrupt": 0}
    errors = []
    for aid in wanted:
        record = current.get(aid)
        if record is None:
            raise BronzeIngestError(f"Run asset is not included in its pinned source: {aid}")
        try:
            manifest = _completed(s3, bucket, run_id=run_id, record=record, verify_hash=True)
            if manifest is None:
                stats["missing"] += 1
            else:
                stats["committed"] += 1
        except Exception as exc:
            stats["corrupt"] += 1
            errors.append({"asset_id": aid, "error": str(exc)[:1000]})
    return {
        "schema": "alpha_edge_full_history_verification_v1",
        "run_id": run_id,
        "source_fingerprint": fingerprint,
        **stats,
        "errors": errors,
        "state": "pass" if stats["expected"] == stats["committed"] else "incomplete",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="S3-only, resumable Yahoo FULL-HISTORY Bronze ingestion")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--bucket", default=DEFAULT_BUCKET)
        p.add_argument("--region", default=DEFAULT_REGION)
        p.add_argument("--source-fingerprint", help="Override the active S3 universe with a pinned snapshot SHA256")
        p.add_argument("--run-id", required=True)
        p.add_argument("--only", default="", help="Comma-separated pilot selectors; omit for ALL included assets")
        if name == "ingest":
            p.add_argument("--end", required=True, help="Exclusive date YYYY-MM-DD, before today's partial candle")
            p.add_argument("--execute", action="store_true", help="Explicit S3 writes; otherwise only dry run")
            p.add_argument("--retries", type=int, default=2)
            p.add_argument("--pause-seconds", type=float, default=0.5)
            p.add_argument("--progress-every", type=int, default=10)
            p.add_argument("--print-first-failures", type=int, default=20)
            p.add_argument("--verbose-yahoo", action="store_true", help="Show raw yfinance diagnostics")
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    fingerprint = resolve_source_fingerprint(s3, args.bucket, args.source_fingerprint)
    selectors = [x.strip() for x in args.only.split(",") if x.strip()] or None
    if args.command == "ingest":
        if not args.verbose_yahoo:
            # All failed acquisitions are still surfaced in the Bronze failure
            # stream and immutable S3 audit trail; Yahoo may emit several
            # misleading "possibly delisted" messages per one attempt.
            logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        output = ingest(
            s3,
            args.bucket,
            fingerprint=fingerprint,
            run_id=args.run_id,
            end=args.end,
            execute=args.execute,
            only=selectors,
            retry_count=args.retries,
            pause_seconds=args.pause_seconds,
            progress_every=args.progress_every,
            print_first_failures=args.print_first_failures,
        )
    else:
        output = verify(
            s3,
            args.bucket,
            fingerprint=fingerprint,
            run_id=args.run_id,
            only=selectors,
        )
    # Full results are persisted as S3 run reports. Terminal output stays short.
    display = {k: v for k, v in output.items() if k not in {"successes", "failure_details"}}
    print(json.dumps(display, indent=2, default=str))
    if output.get("failed") or output.get("state") == "incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
