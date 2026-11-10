"""AEA-87 single operator rebuild command: fresh reconstruction AND production S3 publication.

One operator workflow only:
  python -m alpha_edge.market.aea87_validate
  python -m alpha_edge.market.aea87_rebuild --confirm-prod-write

Internal functions are transaction implementation details, not separate steps.
S3 has no multi-object atomicity: pause all competing ingestion and consumers
during --confirm-prod-write execution. All affected prices and returns are
replaced under active manifests, snapshots/cache refreshed, and rollback
attempted on failure. Non-approvable instruments receive S3 exception reports.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import uuid
import time
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge import paths
from alpha_edge.core.audit import make_run_id, build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import load_runtime_config
from alpha_edge.core.market_store import MarketStore
from alpha_edge.market.aea87_full_history_rebuild import run as download_and_stage
from alpha_edge.market.aea87_rebuild_publication_prepare import prepare
from alpha_edge.market.aea87_rebuild_publication_activate import (
    backup_downstream, restore_downstream, convert_to_transaction_ops, require_ready_plan,
)
from alpha_edge.market.aea87_publication_transaction import (
    fetch, checksum, put_once, write_event, entries, switch_manifests, rollback,
)
from alpha_edge.market.aea87_release_finalize import finalize

VALIDATION_LATEST = "market/repairs/aea87/universe_screen/v1/latest.json"
ROOT = "market/repairs/aea87/rebuild_publications/v1"
AFFECTED = {"REBUILD_OR_DATA_REPAIR_REVIEW", "INSUFFICIENT_DATA", "INVESTIGATION_REQUIRED"}


def selected_from_latest(s3, bucket, *, only=None):
    pointer_bytes = fetch(s3, bucket, VALIDATION_LATEST)
    if pointer_bytes is None:
        raise RuntimeError("No complete-universe validation report in S3. Run aea87_validate.")
    pointer = json.loads(pointer_bytes)
    body = fetch(s3, bucket, pointer["report_key"])
    if body is None:
        raise RuntimeError("Validation report not found in S3")
    report = json.loads(body)
    if not report.get("complete_universe") or report.get("scanned") != report.get("universe_count"):
        raise RuntimeError("Only a complete-universe validation may authorize bulk rebuilding")
    statuses = {x["asset_id"]: x["status"] for x in report["assets"]}
    selected = sorted(x for x, status in statuses.items() if status in AFFECTED)
    if only:
        missing = sorted(set(only) - set(statuses))
        if missing:
            raise ValueError("Assets missing from validation report: " + ",".join(missing))
        selected = sorted(set(only))
    return selected, {"validation_run_id": report["run_id"], "classification_counts":report["classification_counts"],
                      "complete_universe": True}


SOURCE_ROOT = "market/repairs/aea87/rebuilds/v1/runs"


def _safe_asset_folder(asset_id):
    return "".join(c if c.isalnum() else "_" for c in asset_id)


def _source_file_valid(s3, bucket, index, key):
    record = index.get(key)
    if record is None:
        return False
    body = fetch(s3, bucket, key)
    return body is not None and hashlib.sha256(body).hexdigest() == record.get("sha256")


def find_reusable_source(s3, bucket, batch, args):
    """Reuse a complete immutable source run for exactly the requested batch.

    No partial source run, failed publication, changed date window, changed
    provider mapping, or corrupted candidate may be promoted implicitly.
    """
    prefix = SOURCE_ROOT + "/"
    paginator = s3.get_paginator("list_objects_v2")
    runs = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            run_id = item["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            if "_source_" in run_id:
                runs.append(run_id)
    universe = pd.read_csv(args.universe, dtype=str).fillna("")
    if universe["asset_id"].duplicated().any():
        raise ValueError("Duplicate asset identities in universe")
    mapped = universe.set_index("asset_id")
    for run_id in sorted(runs, reverse=True):
        source_prefix = prefix + run_id + "/"
        summary_key = source_prefix + "summary.json"
        manifest_key = source_prefix + "manifest.json"
        summary_bytes = fetch(s3, bucket, summary_key)
        manifest_bytes = fetch(s3, bucket, manifest_key)
        if summary_bytes is None or manifest_bytes is None:
            continue
        try:
            summary = json.loads(summary_bytes)
            manifest = json.loads(manifest_bytes)
            if (summary.get("run_id") != run_id
                    or summary.get("status") != "STAGING_ONLY"
                    or summary.get("schema") != "aea87_full_history_rebuild_v2_local_session_dates"
                    or summary.get("active_histories_modified") is not False
                    or summary.get("start") != args.start
                    or summary.get("end_exclusive") != args.end_exclusive
                    or set(summary.get("assets", {})) != set(batch)
                    or manifest.get("run_id") != run_id):
                continue
            index = {obj["key"]: obj for obj in manifest["objects"]}
            if not _source_file_valid(s3, bucket, index, summary_key):
                continue
            valid = True
            for aid in batch:
                item = summary["assets"][aid]
                if aid not in mapped.index:
                    valid = False
                    break
                expected_symbol = str(mapped.at[aid, "yahoo_ticker"]).strip()
                if expected_symbol != str(item.get("yahoo_ticker", "")).strip():
                    # Failed source downloads without a symbol cannot be reused.
                    valid = False
                    break
                if item.get("status") == "STAGED_FOR_REVIEW":
                    asset_prefix = source_prefix + "assets/" + _safe_asset_folder(aid) + "/"
                    for name in ("raw_yahoo.parquet", "candidate_ohlcv_usd.parquet",
                                 "candidate_returns_usd.parquet", "provider_metadata.json"):
                        if not _source_file_valid(s3, bucket, index, asset_prefix + name):
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                print(f"[aea87_rebuild] Reusing verified immutable source run {run_id} "
                      f"for {len(batch)} assets", flush=True)
                return summary
        except (ValueError, KeyError, TypeError, IndexError):
            continue
    return None


def find_reusable_plan(s3, bucket, source_run, asset_id):
    """Reuse staged publication plans only when all active checksums match."""
    original_release = source_run.split("_source_", 1)[0]
    key = (ROOT + "/" + original_release + "_" + _safe_asset_folder(asset_id)
           + "/plan.json")
    body = fetch(s3, bucket, key)
    if body is None:
        return None
    try:
        plan = json.loads(body)
        if plan.get("asset_id") != asset_id or plan.get("source_run") != source_run:
            return None
        require_ready_plan(s3, bucket, plan)
        return plan
    except (ValueError, KeyError, TypeError) as exc:
        print(f"[aea87_rebuild] Existing prepared plan for {asset_id} "
              f"not reusable ({exc}); reconstructing plan", flush=True)
        return None


def _write_report(s3, bucket, root, name, data):
    key = root + "/" + name
    put_once(s3, bucket, key, json.dumps(data, indent=2, default=str).encode())
    return f"s3://{bucket}/{key}"


def _verify_active_histories(store, plans):
    """After switching manifests, prove readers see exact candidate session dates."""
    for plan in plans:
        aid = plan["asset_id"]
        for table in ("ohlcv_usd", "returns_usd"):
            expected = set()
            for op in plan["operations"]:
                if op["table"] == table:
                    expected.update(json.loads(op["after"]).get("dates", []))
            if not expected:
                raise RuntimeError("Cannot verify empty published history: " + aid)
            first, last = min(expected), max(expected)
            active = (store.read_ohlcv_usd([aid], start=first, end=last)
                      if table == "ohlcv_usd" else
                      store.read_returns_usd([aid], start=first, end=last))
            if active.empty or "date" not in active:
                raise RuntimeError("Published history unreadable: " + aid + "/" + table)
            actual = pd.to_datetime(active["date"], utc=True).dt.strftime("%Y-%m-%d")
            if actual.duplicated().any() or set(actual) != expected:
                raise RuntimeError("Published dates differ from reconstructed candidate: "
                                   + aid + "/" + table)


def _publish_batch(s3, bucket, store, plans, publication_id):
    # Exactly one switch of the whole eligible group, not a separate
    # published maintenance cycle for each instrument.
    ops = []
    for plan in plans:
        ops.extend(convert_to_transaction_ops(require_ready_plan(s3, bucket, plan)))
    if not ops:
        raise RuntimeError("No validated replacement operations to publish")
    if len({op["manifest_key"] for op in ops}) != len(ops):
        raise RuntimeError("Duplicate annual manifest target")
    root = f"{ROOT}/{publication_id}/transaction"
    if entries(s3, bucket, root):
        raise RuntimeError("Publication already journaled: choose new release id")
    downstream = backup_downstream(s3, bucket, root)
    put_once(s3, bucket, root + "/operations.json", json.dumps(ops, sort_keys=True).encode())
    write_event(s3, bucket, root, 0, "READY")
    try:
        switch_manifests(s3, bucket, root, ops, workers=8, batch_size=100)
        _verify_active_histories(store, plans)
        publication = {"candidates": [plan["asset_id"] for plan in plans], "operations": ops}
        finalized = finalize(s3, bucket, root, publication)
        if finalized.get("status") != "FINALIZED":
            raise RuntimeError("Downstream finalization not confirmed")
        return finalized
    except BaseException as exc:
        history = entries(s3, bucket, root)
        if history and history[-1]["state"] not in ("ROLLED_BACK", "FINALIZED"):
            try:
                rollback(s3, bucket, root, ops, workers=8)
            except Exception as restore_exc:
                raise RuntimeError("CRITICAL: manifest rollback failed; keep all jobs stopped") from restore_exc
        try:
            restore_downstream(s3, bucket, downstream)
        except Exception as restore_exc:
            raise RuntimeError("CRITICAL: downstream rollback failed; keep all jobs stopped") from restore_exc
        # Durable audit of the rollback through the existing Operational
        # Control Layer, with the same run ID as the journal.
        cfg = load_runtime_config("prod")
        event = build_audit_event(
            cfg=cfg, run_id=publication_id.split("_batch_")[0],
            event_type="rebuild", entity_type="market_data",
            entity_id=publication_id, as_of=None,
            source_script="aea87_rebuild", source_mode="publication_rollback",
            status="failed",
            input_args={"publication_id":publication_id},
            output_keys=[op["manifest_key"] for op in ops],
            backup_keys=[op["backup_key"] for op in ops if op.get("backup_key")],
            metadata={"rollback_completed": True, "recovered_assets": [p["asset_id"] for p in plans]},
            error=f"{type(exc).__name__}: {exc}")
        write_audit_event(cfg=cfg, event=event)
        raise RuntimeError("Publication failed; original manifests and downstream backups restored") from exc


def run(args):
    if not args.confirm_prod_write:
        raise ValueError("Production rebuild requires --confirm-prod-write")
    s3 = boto3.client("s3", region_name=args.region)
    store = MarketStore(bucket=args.bucket, region=args.region)
    only = [x.strip() for x in args.asset_ids.split(",") if x.strip()] if args.asset_ids else None
    asset_ids, report_metadata = selected_from_latest(s3, args.bucket, only=only)
    if not asset_ids:
        print("Validation found no implicated assets. Production unchanged.")
        return
    release = getattr(args, "run_id", None) or ("aea87_rebuild_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    root = f"{ROOT}/{release}"
    started = {"status":"STARTED", "asset_count":len(asset_ids), "selected_assets":asset_ids,
               **report_metadata, "active_modified":False}
    _write_report(s3, args.bucket, root, "selection.json", started)
    print(f"[aea87_rebuild] {len(asset_ids)} implicated assets selected; immutable raw data -> S3", flush=True)

    exceptions = []
    published = []
    source_runs = []
    total = len(asset_ids)
    started_at = time.monotonic()
    batch_size = args.batch_size

    def progress(processed, phase, last_asset=None):
        state = {"run_id": release, "status": phase if phase in {"COMPLETED", "COMPLETED_WITH_EXCEPTIONS", "FAILED"} else "RUNNING", "phase": phase,
                 "processed": processed, "total": total,
                 "published": len(published), "exceptions": len(exceptions),
                 "last_asset_id": last_asset,
                 "elapsed_seconds": round(time.monotonic() - started_at, 1),
                 "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        # Mutable run-specific checkpoint: write during work, not only at the end.
        s3.put_object(Bucket=args.bucket, Key=root + "/progress/latest.json",
                      Body=json.dumps(state).encode(), ContentType="application/json")
        print(f"[aea87_rebuild] {phase}: {processed}/{total} "
              f"published={len(published)} exceptions={len(exceptions)} "
              f"elapsed={state['elapsed_seconds']}s", flush=True)

    progress(0, "STARTED")
    for offset in range(0, total, batch_size):
        batch = asset_ids[offset:offset + batch_size]
        print(f"[aea87_rebuild] Batch {offset//batch_size+1}: fresh history for "
              f"{len(batch)} assets", flush=True)
        download_args = SimpleNamespace(
            asset_ids=",".join(batch), universe=args.universe, bucket=args.bucket,
            region=args.region, start=args.start, end_exclusive=args.end_exclusive,
            run_id=release + "_source_" + str(offset//batch_size+1).zfill(4))
        staged = find_reusable_source(s3, args.bucket, batch, download_args)
        if staged is None:
            staged = download_and_stage(download_args)
        if not staged or not staged.get("run_id"):
            raise RuntimeError("Fresh source run did not produce an S3 summary")
        source_run = staged["run_id"]
        source_runs.append(source_run)
        candidates = []
        for aid in batch:
            entry = staged["assets"].get(aid, {})
            if entry.get("status") != "STAGED_FOR_REVIEW":
                exceptions.append({"asset_id":aid, "stage":"SOURCE",
                                   "reason":entry.get("status"), "detail":entry.get("error")})
                continue
            try:
                outcome = find_reusable_plan(s3, args.bucket, source_run, aid)
                if outcome is None:
                    outcome = prepare(s3, store, args.bucket, source_run, aid,
                                      args.end_exclusive,
                                      release + "_" + _safe_asset_folder(aid))
                if outcome.get("status") == "PREPARED_NOT_PUBLISHED":
                    candidates.append(outcome)
                else:
                    exceptions.append({"asset_id":aid, "stage":"RECONSTRUCTION",
                                       "reason":outcome.get("status"),
                                       "blockers":outcome.get("blockers")})
            except Exception as exc:
                exceptions.append({"asset_id":aid, "stage":"RECONSTRUCTION",
                                   "reason":type(exc).__name__, "detail":str(exc)})
                print(f"[aea87_rebuild] {aid}: reconstruction exception {exc}", flush=True)

        progress(offset + len(batch), "SOURCE_AND_PREPARATION_COMPLETE", batch[-1])
        if candidates:
            print(f"[aea87_rebuild] Publishing {len(candidates)} qualifying assets "
                  f"in batch {offset//batch_size+1}", flush=True)
            _publish_batch(s3, args.bucket, store, candidates,
                           release + "_batch_" + str(offset//batch_size+1).zfill(4))
            published.extend(p["asset_id"] for p in candidates)
            # Use the established Alpha Edge audit trail, not a parallel journal.
            audit_cfg = load_runtime_config("prod")
            if audit_cfg.bucket != args.bucket or audit_cfg.region != args.region:
                raise RuntimeError("Audit target differs from production data target")
            batch_id = release + "_batch_" + str(offset//batch_size+1).zfill(4)
            ops = [op for plan in candidates for op in plan["operations"]]
            event = build_audit_event(
                cfg=audit_cfg, run_id=release, event_type="rebuild",
                entity_type="market_data", entity_id=batch_id,
                as_of=args.end_exclusive, source_script="aea87_rebuild",
                source_mode="production_reconstruction", status="success",
                input_args={"batch_index":offset//batch_size+1,
                            "source_run":source_run,"asset_count":len(candidates)},
                output_keys=[op["manifest_key"] for op in ops],
                backup_keys=[op["backup_key"] for op in ops if op.get("backup_key")],
                metadata={"affected_assets":[p["asset_id"] for p in candidates],
                          "manifest_count":len(ops),
                          "before_sha256_by_key":{op["manifest_key"]:op["before_sha"] for op in ops},
                          "after_sha256_by_key":{op["manifest_key"]:op["after_sha"] for op in ops},
                          "publication_root":f"{ROOT}/{batch_id}/transaction"})
            write_audit_event(cfg=audit_cfg, event=event)
        # Preserve independent completed batches and exceptions after every cutover.
        progress(offset + len(batch), "BATCH_COMPLETE", batch[-1])
        _write_report(s3, args.bucket, root,
                      f"batches/{offset//batch_size+1:04d}.json",
                      {"batch": offset//batch_size+1, "assets": batch,
                       "source_run":source_run,
                       "published":[p["asset_id"] for p in candidates],
                       "exceptions":[x for x in exceptions if x["asset_id"] in batch]})

    completed = {"status": ("COMPLETED_WITH_EXCEPTIONS" if exceptions else "COMPLETED")
                 if published else "FAILED",
                 "release_id":release, "source_runs":source_runs,
                 "selected":total, "published":sorted(published),
                 "exceptions":exceptions, "active_modified":bool(published),
                 "validation_run_id": report_metadata["validation_run_id"]}
    uri = _write_report(s3, args.bucket, root, "result.json", completed)
    audit_cfg = load_runtime_config("prod")
    summary_event = build_audit_event(
        cfg=audit_cfg, run_id=release, event_type="rebuild",
        entity_type="market_data", entity_id=release,
        as_of=args.end_exclusive, source_script="aea87_rebuild",
        source_mode="production_reconstruction", status="success" if published else "failed",
        input_args={"asset_count":total,"validation_run_id":report_metadata["validation_run_id"]},
        output_keys=[root+"/result.json"],
        metadata={"selected":total,"published":len(published),
                  "unresolved":len(exceptions),"result_status":completed["status"],
                  "source_runs":source_runs})
    write_audit_event(cfg=audit_cfg, event=summary_event)
    progress(total, completed["status"])
    print(json.dumps({"status":completed["status"], "selected":total,
                      "published":len(published), "unresolved":len(exceptions),
                      "s3_report":uri}, indent=2), flush=True)
    if not published:
        raise RuntimeError("No defensible replacements could be published; see S3 report")


def main():
    p = argparse.ArgumentParser(description="AEA-87 fresh reconstruction and PRODUCTION S3 publication")
    p.add_argument("--universe", default=str(paths.universe_dir() / "universe.csv"))
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end-exclusive", default=datetime.now(timezone.utc).date().isoformat())
    p.add_argument("--asset-ids", help="Optional targeted selection; default ALL implicated assets from complete validation")
    p.add_argument("--batch-size", type=int, default=25, help="Internal durable publication batch size (default 25)")
    p.add_argument("--confirm-prod-write", action="store_true")
    a = p.parse_args()
    if not 1 <= a.batch_size <= 100:
        p.error("batch-size must be between 1 and 100")
    if a.end_exclusive <= a.start:
        p.error("end-exclusive must be greater than start")
    if not a.confirm_prod_write:
        p.error("Explicit --confirm-prod-write required; stop scheduled ingestion and readers during cutover")
    cfg = load_runtime_config("prod")
    if cfg.bucket != a.bucket or cfg.region != a.region:
        p.error("Production runtime audit bucket/region mismatch")
    a.run_id = make_run_id("aea87_rebuild")
    with capture_script_run(cfg=cfg, script_name="aea87_rebuild",
                            input_args=vars(a), run_id=a.run_id):
        try:
            run(a)
        except Exception as exc:
            try:
                s3 = boto3.client("s3", region_name=a.region)
                fail_key = f"{ROOT}/{a.run_id}/progress/latest.json"
                s3.put_object(Bucket=a.bucket, Key=fail_key,
                    Body=json.dumps({"run_id":a.run_id,"status":"FAILED",
                                     "timestamp_utc":datetime.now(timezone.utc).isoformat(),
                                     "error":f"{type(exc).__name__}: {exc}"}).encode(),
                    ContentType="application/json")
            except Exception:
                # Failure of progress reporting cannot conceal the original error.
                pass
            event = build_audit_event(
                cfg=cfg, run_id=a.run_id, event_type="rebuild",
                entity_type="market_data", entity_id=None,
                as_of=a.end_exclusive, source_script="aea87_rebuild",
                source_mode="production_reconstruction", status="failed",
                input_args=vars(a),
                metadata={"recovery":"check publication journal and batch checkpoints"},
                error=f"{type(exc).__name__}: {exc}")
            write_audit_event(cfg=cfg, event=event)
            raise


if __name__ == "__main__":
    main()
