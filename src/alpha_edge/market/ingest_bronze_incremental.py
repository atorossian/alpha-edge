"""Incremental Yahoo daily Bronze acquisition: read S3 committed state, request dates only.

This module deliberately does not modify the immutable full-history ingestion
contract or production v1/v2 datasets. It follows the existing-state decision
in ingest_market_data.py: read last successful observation per asset, fetch
from that date, and keep an explicit bounded overlap for revisions.

Full historical data is NEVER requested by this module. Assets lacking a
verified S3 parent commit fail closed; use a separate deliberate backfill.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.market.ingest_bronze_full_history import (
    BRONZE_ROOT,
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    SAFE_RUN,
    BronzeIngestError,
    _check_bars,
    _completed,
    _digest,
    _get_optional,
    _json_bytes,
    _print_progress,
    _read,
    load_universe,
)
from alpha_edge.market.ingest_bronze_yahoo_v2 import acquire_yahoo_bars

INCREMENTAL_ROOT = "market/lake_v3/bronze/incremental"
SCHEMA_RUN = "alpha_edge_bronze_incremental_run_v1"
SCHEMA_ASSET = "alpha_edge_bronze_incremental_asset_v1"


def _put_immutable(s3: Any, bucket: str, key: str, payload: bytes, content_type: str) -> None:
    if not key.startswith(INCREMENTAL_ROOT + "/"):
        raise BronzeIngestError(f"Refusing incremental S3 write outside {INCREMENTAL_ROOT}: {key}")
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            Metadata={"sha256": _digest(payload)},
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {
            "PreconditionFailed",
            "412",
            "ConditionalRequestConflict",
            "409",
        }:
            raise
        if _read(s3, bucket, key) != payload:
            raise BronzeIngestError(f"Immutable incremental S3 collision: {key}") from exc


def _asset_prefix(run_id: str, asset_id: str) -> str:
    return f"{INCREMENTAL_ROOT}/runs/{run_id}/assets/{_digest(asset_id.encode('utf-8'))}"


def _incremental_commit(s3: Any, bucket: str, run_id: str, record: dict, *, verify_hash: bool = False) -> dict | None:
    prefix = _asset_prefix(run_id, record["asset_id"])
    blob = _get_optional(s3, bucket, prefix + "/manifest.json")
    if blob is None:
        return None
    manifest = json.loads(blob)
    if (
        manifest.get("schema") != SCHEMA_ASSET
        or manifest.get("asset_id") != record["asset_id"]
        or manifest.get("provider_symbol") != record["yahoo_ticker"]
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "committed"
    ):
        raise BronzeIngestError(f"Incompatible incremental manifest: {prefix}")
    ref = manifest.get("data", {})
    key = ref.get("key", "")
    if not key.startswith(prefix + "/objects/") or not key.endswith(".parquet"):
        raise BronzeIngestError(f"Unsafe incremental S3 data reference: {key}")
    if verify_hash:
        payload = _read(s3, bucket, key)
        if len(payload) != ref.get("bytes") or _digest(payload) != ref.get("sha256"):
            raise BronzeIngestError(f"Incremental data bytes corrupt: {key}")
        if len(pd.read_parquet(io.BytesIO(payload))) != manifest["validation"]["rows"]:
            raise BronzeIngestError(f"Incremental Parquet row count mismatch: {key}")
    else:
        head = s3.head_object(Bucket=bucket, Key=key)
        if head.get("ContentLength") != ref.get("bytes") or head.get("Metadata", {}).get("sha256") != ref.get("sha256"):
            raise BronzeIngestError(f"Incremental data object metadata mismatch: {key}")
    return manifest


def _find_parent(s3: Any, bucket: str, parent_run_id: str, record: dict) -> tuple[dict, str]:
    """Resolve most recent committed history by walking only explicitly pinned runs.

    No silent source-ticker substitutions and no old v1/v2 market lake reads.
    """
    seen = set()
    current = parent_run_id
    while current:
        if current in seen or len(seen) >= 100 or not SAFE_RUN.fullmatch(current):
            raise BronzeIngestError(f"Circular/invalid Bronze parent run lineage: {current}")
        seen.add(current)
        new_run = _get_optional(s3, bucket, f"{INCREMENTAL_ROOT}/runs/{current}/run.json")
        if new_run is not None:
            config = json.loads(new_run)
            if config.get("schema") != SCHEMA_RUN:
                raise BronzeIngestError(f"Unrecognized incremental parent run schema: {current}")
            if record["asset_id"] in config.get("requested_asset_ids", []):
                receipt = _incremental_commit(s3, bucket, current, record)
                if receipt is not None:
                    return receipt, current
            current = config.get("parent_run_id")
            continue
        old_run = _get_optional(s3, bucket, f"{BRONZE_ROOT}/runs/{current}/run.json")
        if old_run is None:
            raise BronzeIngestError(f"Missing pinned Bronze parent run: {current}")
        config = json.loads(old_run)
        if config.get("schema") != "alpha_edge_full_history_run_v1":
            raise BronzeIngestError(f"Unrecognized full-history parent run schema: {current}")
        if record["asset_id"] not in config.get("requested_asset_ids", []):
            raise BronzeIngestError(f"No baseline for {record['asset_id']} in full-history run {current}")
        receipt = _completed(s3, bucket, run_id=current, record=record, verify_hash=False)
        if receipt is None:
            raise BronzeIngestError(f"Missing committed baseline for {record['asset_id']} in {current}")
        return receipt, current
    raise BronzeIngestError(f"No committed Bronze parent for {record['asset_id']}")


def _window(parent_manifest: dict, *, end: str, overlap_days: int, max_staleness_days: int) -> tuple[str, str]:
    session = parent_manifest.get("validation", {}).get("last_session")
    try:
        last = date.fromisoformat(str(session))
        final = date.fromisoformat(end)
    except (TypeError, ValueError) as exc:
        raise BronzeIngestError(f"Invalid parent last_session={session} or end={end}") from exc
    if last >= final:
        raise BronzeIngestError(f"Parent last_session {last} is not before exclusive end {end}")
    if (final - last).days > max_staleness_days:
        raise BronzeIngestError(
            f"Stale baseline {last}: >{max_staleness_days} calendar days before {final}; "
            "review security/source history; refusing unbounded incremental fetch"
        )
    return (last - timedelta(days=overlap_days)).isoformat(), last.isoformat()


def _audit(
    s3: Any, bucket: str, run_id: str, status: str, *, record: dict | None = None, details: dict | None = None
) -> None:
    stamp = datetime.now(timezone.utc)
    key = f"{INCREMENTAL_ROOT}/runs/{run_id}/audit/dt={stamp:%Y-%m-%d}/{stamp:%H%M%S}-{uuid.uuid4().hex}.json"
    payload = {
        "schema": "alpha_edge_bronze_incremental_audit_v1",
        "event_type": "build_dataset",
        "source_script": "ingest_bronze_incremental.py",
        "run_id": run_id,
        "timestamp_utc": stamp.isoformat(),
        "status": status,
        "asset_id": record["asset_id"] if record else None,
        "provider_symbol": record["yahoo_ticker"] if record else None,
        "metadata": details or {},
    }
    _put_immutable(s3, bucket, key, _json_bytes(payload), "application/json")


def _acquire(
    s3: Any,
    bucket: str,
    record: dict,
    *,
    run_id: str,
    start: str,
    end: str,
    baseline_session: str,
    baseline_run_id: str,
    retry_count: int,
    retry_delay: float,
    sleep: Callable[[float], None],
    ticker_factory: Callable | None,
) -> dict:
    failure = None
    for attempt in range(retry_count + 1):
        try:
            bars, provenance = acquire_yahoo_bars(
                record["asset_id"],
                record["yahoo_ticker"],
                start=start,
                end=end,
                batch_id=run_id,
                currency_hint=record["currency_hint"],
                full_history=False,
                ticker_factory=ticker_factory,
            )
            validation = _check_bars(bars, record)
            sessions = bars["session_date"].astype(str)
            validation["new_rows_since_baseline"] = int(sessions.gt(baseline_session).sum())
            validation["overlap_rows"] = int(sessions.le(baseline_session).sum())
            if not sessions.eq(baseline_session).any():
                raise BronzeIngestError(
                    f"{record['asset_id']}: Yahoo response lacks last committed "
                    f"session {baseline_session} in required overlap"
                )
            buf = io.BytesIO()
            bars.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
            payload = buf.getvalue()
            prefix = _asset_prefix(run_id, record["asset_id"])
            sha = _digest(payload)
            data_key = f"{prefix}/objects/{sha}.parquet"
            _put_immutable(s3, bucket, data_key, payload, "application/vnd.apache.parquet")
            manifest = {
                "schema": SCHEMA_ASSET,
                "run_id": run_id,
                "asset_id": record["asset_id"],
                "broker_ticker": record["ticker"],
                "provider_symbol": record["yahoo_ticker"],
                "status": "committed",
                "mode": "incremental",
                "start_inclusive": start,
                "end_exclusive": end,
                "baseline_session": baseline_session,
                "baseline_run_id": baseline_run_id,
                "overlap_is_raw_revision_evidence": True,
                "source_values_preserved": True,
                "data": {"key": data_key, "bytes": len(payload), "sha256": sha},
                "validation": validation,
                "provenance": provenance,
            }
            _put_immutable(s3, bucket, prefix + "/manifest.json", _json_bytes(manifest), "application/json")
            _incremental_commit(s3, bucket, run_id, record, verify_hash=False)
            return manifest
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            failure = exc
            if attempt < retry_count:
                sleep(retry_delay * (2**attempt))
    raise BronzeIngestError(f"{record['asset_id']} ({record['yahoo_ticker']}): {type(failure).__name__}: {failure}")


def ingest(
    s3: Any,
    bucket: str,
    *,
    fingerprint: str,
    run_id: str,
    parent_run_id: str,
    end: str,
    overlap_days: int = 10,
    max_staleness_days: int = 60,
    only: list[str] | None = None,
    execute: bool = False,
    retry_count: int = 2,
    pause_seconds: float = 0.5,
    progress_every: int = 10,
    ticker_factory: Callable | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Acquire only bounded Yahoo windows; no 'period=max' code path."""
    if not SAFE_RUN.fullmatch(run_id) or not SAFE_RUN.fullmatch(parent_run_id) or run_id == parent_run_id:
        raise BronzeIngestError("Invalid or identical run and parent-run IDs")
    if date.fromisoformat(end) > datetime.now(timezone.utc).date():
        raise BronzeIngestError("Exclusive end date exceeds today's UTC date")
    if overlap_days < 1 or overlap_days > 30 or max_staleness_days < overlap_days or max_staleness_days > 365:
        raise BronzeIngestError("Invalid bounded lookback/staleness configuration")
    if retry_count < 0 or retry_count > 8 or pause_seconds < 0 or progress_every < 1:
        raise BronzeIngestError("Invalid retries, pacing or progress settings")
    all_assets, source = load_universe(s3, bucket, fingerprint)
    if only:
        wanted = {x.strip() for x in only}
        selection = [
            a for a in all_assets if a["asset_id"] in wanted or a["ticker"] in wanted or a["yahoo_ticker"] in wanted
        ]
        matched = {x for a in selection for x in (a["asset_id"], a["ticker"], a["yahoo_ticker"])}
        if wanted - matched:
            raise BronzeIngestError(f"Unknown or excluded asset selector: {sorted(wanted - matched)}")
        for selector in wanted:
            if sum(selector in (a["ticker"], a["yahoo_ticker"], a["asset_id"]) for a in selection) != 1:
                raise BronzeIngestError(f"Ambiguous selector {selector}; use canonical asset_id")
    else:
        selection = all_assets
    config = {
        "schema": SCHEMA_RUN,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "source_fingerprint": fingerprint,
        "source_git_commit": source["source_git_commit"],
        "end_exclusive": end,
        "overlap_days": overlap_days,
        "max_staleness_days": max_staleness_days,
        "requested_asset_ids": [a["asset_id"] for a in selection],
        "provider_parameters": {
            "provider": "yahoo_finance",
            "interval": "1d",
            "mode": "date_range",
            "auto_adjust": False,
            "actions": True,
            "repair": False,
        },
    }
    key = f"{INCREMENTAL_ROOT}/runs/{run_id}/run.json"
    prior = _get_optional(s3, bucket, key)
    if prior is not None:
        if json.loads(prior) != config:
            raise BronzeIngestError("Existing incremental run ID has different immutable configuration")
    elif execute:
        _put_immutable(s3, bucket, key, _json_bytes(config), "application/json")
    if not execute:
        return {
            "state": "dry_run",
            "selected": len(selection),
            "writes": 0,
            "parent_run_id": parent_run_id,
            "end_exclusive": end,
            "overlap_days": overlap_days,
            "source_fingerprint": fingerprint,
        }

    ok, skipped, failed = [], [], []
    started = time.monotonic()
    total = len(selection)
    print(
        f"=== BRONZE INCREMENTAL ===\nrun_id: {run_id}\nparent: {parent_run_id}\n"
        f"selected: {total}\noverlap_days: {overlap_days}\n"
        "provider: Yahoo date-window ONLY; no period=max",
        flush=True,
    )
    _audit(s3, bucket, run_id, "started", details={"selected": total, "parent_run_id": parent_run_id})
    for position, record in enumerate(selection, 1):
        try:
            existing = _incremental_commit(s3, bucket, run_id, record)
            if existing is not None:
                skipped.append(record["asset_id"])
                _audit(s3, bucket, run_id, "already_committed", record=record)
            else:
                base, base_run = _find_parent(s3, bucket, parent_run_id, record)
                start, last_session = _window(
                    base,
                    end=end,
                    overlap_days=overlap_days,
                    max_staleness_days=max_staleness_days,
                )
                manifest = _acquire(
                    s3,
                    bucket,
                    record,
                    run_id=run_id,
                    start=start,
                    end=end,
                    baseline_session=last_session,
                    baseline_run_id=base_run,
                    retry_count=retry_count,
                    retry_delay=2.0,
                    ticker_factory=ticker_factory,
                    sleep=sleep,
                )
                ok.append(
                    {
                        "asset_id": record["asset_id"],
                        "start_inclusive": start,
                        "baseline_session": last_session,
                        **manifest["validation"],
                    }
                )
                _audit(
                    s3,
                    bucket,
                    run_id,
                    "success",
                    record=record,
                    details={
                        "start": start,
                        "baseline_session": last_session,
                        "baseline_run_id": base_run,
                        "data_key": manifest["data"]["key"],
                        "rows": manifest["validation"]["rows"],
                        "new_rows": manifest["validation"]["new_rows_since_baseline"],
                    },
                )
        except (KeyboardInterrupt, SystemExit):
            _audit(s3, bucket, run_id, "interrupted", record=record)
            raise
        except Exception as exc:
            item = {
                "asset_id": record["asset_id"], "ticker": record["yahoo_ticker"],
                "error": f"{type(exc).__name__}: {exc}"[:1800],
            }
            failed.append(item)
            print(f"[incremental][failed] {position}/{total}: {item}", flush=True)
            _audit(s3, bucket, run_id, "failed", record=record, details=item)
        _print_progress(
            done=position, total=total, ok=len(ok), resumed=len(skipped),
            failed=len(failed), started_monotonic=started, progress_every=progress_every,
        )
        if pause_seconds:
            sleep(pause_seconds)
    result = {
        "schema": "alpha_edge_bronze_incremental_report_v1",
        "run_id": run_id, "parent_run_id": parent_run_id,
        "source_fingerprint": fingerprint, "selected": total,
        "succeeded": len(ok), "already_committed": len(skipped),
        "failed": len(failed), "failure_details": failed,
        "successes": ok, "state": "complete" if not failed else "completed_with_failures",
    }
    report_key = (
        f"{INCREMENTAL_ROOT}/runs/{run_id}/reports/"
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex}.json"
    )
    _put_immutable(s3, bucket, report_key, _json_bytes(result), "application/json")
    _audit(s3, bucket, run_id, result["state"], details={"report": report_key, "failed": len(failed)})
    result["report_s3_uri"] = f"s3://{bucket}/{report_key}"
    print(f"[incremental][complete] {result['state']} report={result['report_s3_uri']}", flush=True)
    return result


def verify(s3: Any, bucket: str, *, fingerprint: str, run_id: str) -> dict:
    config = json.loads(_read(s3, bucket, f"{INCREMENTAL_ROOT}/runs/{run_id}/run.json"))
    if config.get("schema") != SCHEMA_RUN or config.get("source_fingerprint") != fingerprint:
        raise BronzeIngestError("Incompatible incremental run/fingerprint")
    assets, _ = load_universe(s3, bucket, fingerprint)
    indexed = {a["asset_id"]: a for a in assets}
    ids = config["requested_asset_ids"]
    result = {"expected": len(ids), "committed": 0, "missing": 0, "corrupt": 0, "errors": []}
    for aid in ids:
        try:
            if aid not in indexed:
                raise BronzeIngestError(f"Missing current asset identity: {aid}")
            manifest = _incremental_commit(s3, bucket, run_id, indexed[aid], verify_hash=True)
            if manifest is None:
                result["missing"] += 1
            else:
                result["committed"] += 1
        except Exception as exc:
            result["corrupt"] += 1
            result["errors"].append({"asset_id": aid, "error": str(exc)[:1000]})
    result.update(
        schema="alpha_edge_bronze_incremental_verification_v1",
        run_id=run_id, source_fingerprint=fingerprint,
        state="pass" if result["expected"] == result["committed"] else "incomplete",
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="S3-ledger-based incremental Yahoo Bronze acquisition")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("ingest", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--bucket", default=DEFAULT_BUCKET)
        command.add_argument("--region", default=DEFAULT_REGION)
        command.add_argument("--source-fingerprint", required=True)
        command.add_argument("--run-id", required=True)
        if name == "ingest":
            command.add_argument("--parent-run-id", required=True, help="Previous verified Bronze run ID")
            command.add_argument("--end", required=True, help="Exclusive YYYY-MM-DD")
            command.add_argument("--overlap-days", type=int, default=10)
            command.add_argument("--max-staleness-days", type=int, default=60)
            command.add_argument("--only", default="", help="Comma-separated included asset selectors")
            command.add_argument("--execute", action="store_true", help="Allow immutable S3 writes")
            command.add_argument("--retries", type=int, default=2)
            command.add_argument("--pause-seconds", type=float, default=0.5)
            command.add_argument("--progress-every", type=int, default=10)
            command.add_argument("--verbose-yahoo", action="store_true")
    args = ap.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    if args.command == "verify":
        result = verify(s3, args.bucket, fingerprint=args.source_fingerprint, run_id=args.run_id)
    else:
        if not args.verbose_yahoo:
            logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        result = ingest(
            s3, args.bucket, fingerprint=args.source_fingerprint, run_id=args.run_id,
            parent_run_id=args.parent_run_id, end=args.end,
            overlap_days=args.overlap_days, max_staleness_days=args.max_staleness_days,
            only=[s.strip() for s in args.only.split(",") if s.strip()] or None,
            execute=args.execute, retry_count=args.retries,
            pause_seconds=args.pause_seconds, progress_every=args.progress_every,
        )
    print(json.dumps({k: v for k, v in result.items() if k not in {"successes", "failure_details"}}, indent=2))
    if result.get("failed") or result.get("state") == "incomplete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
