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
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge import paths
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
    release = "aea87_rebuild_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    root = f"{ROOT}/{release}"
    started = {"status":"STARTED", "asset_count":len(asset_ids), "selected_assets":asset_ids,
               **report_metadata, "active_modified":False}
    _write_report(s3, args.bucket, root, "selection.json", started)
    print(f"[aea87_rebuild] {len(asset_ids)} implicated assets selected; immutable raw data -> S3", flush=True)

    # Batch ingest is internal. The old staging script is not an operator step.
    download_args = SimpleNamespace(
        asset_ids=",".join(asset_ids), universe=args.universe,
        bucket=args.bucket, region=args.region, start=args.start,
        end_exclusive=args.end_exclusive)
    staged = download_and_stage(download_args)
    if not staged or not staged.get("run_id"):
        raise RuntimeError("Reconstruction did not return a verified S3 staging run")
    source_run = staged["run_id"]

    candidates = []
    exceptions = []
    for aid in asset_ids:
        entry = staged["assets"].get(aid, {})
        if entry.get("status") != "STAGED_FOR_REVIEW":
            exceptions.append({"asset_id":aid, "stage":"SOURCE", "reason":entry.get("status"),
                               "detail":entry.get("error")})
            continue
        try:
            outcome = prepare(s3, store, args.bucket, source_run, aid,
                              args.end_exclusive, release + "_" + "".join(c if c.isalnum() else "_" for c in aid))
            if outcome.get("status") != "PREPARED_NOT_PUBLISHED":
                exceptions.append({"asset_id":aid, "stage":"RECONSTRUCTION",
                                   "reason":outcome.get("status"), "blockers":outcome.get("blockers")})
            else:
                candidates.append(outcome)
        except Exception as exc:
            exceptions.append({"asset_id":aid, "stage":"RECONSTRUCTION",
                               "reason":type(exc).__name__, "detail":str(exc)})
    disposition = {"release_id":release, "source_run":source_run,
                   "validation_run_id":report_metadata["validation_run_id"],
                   "selected":len(asset_ids), "publishable":len(candidates),
                   "exceptions":exceptions, "active_modified":False}
    _write_report(s3, args.bucket, root, "disposition.json", disposition)
    if not candidates:
        print(json.dumps({"status":"NO_SAFE_REPLACEMENTS", **disposition}, indent=2))
        raise RuntimeError("No candidate qualified for publication; production unchanged")

    # The caller is responsible for actual external maintenance isolation.
    # We do not claim to pause remote applications or scheduled market ingestion.
    print(f"[aea87_rebuild] Publishing {len(candidates)} verified candidates to production S3", flush=True)
    result = _publish_batch(s3, args.bucket, store, candidates, release)
    completed = {"status":"PRODUCTION_PUBLISHED", "release_id":release,
                 "source_run":source_run, "published":sorted(p["asset_id"] for p in candidates),
                 "exceptions":exceptions, "publication":result,
                 "active_modified":True}
    uri = _write_report(s3, args.bucket, root, "result.json", completed)
    print(json.dumps({"status":completed["status"], "published":len(candidates),
                      "unresolved":len(exceptions), "s3_report":uri}, indent=2))


def main():
    p = argparse.ArgumentParser(description="AEA-87 fresh reconstruction and PRODUCTION S3 publication")
    p.add_argument("--universe", default=str(paths.universe_dir() / "universe.csv"))
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end-exclusive", default=datetime.now(timezone.utc).date().isoformat())
    p.add_argument("--asset-ids", help="Optional targeted selection; default ALL implicated assets from complete validation")
    p.add_argument("--confirm-prod-write", action="store_true")
    a = p.parse_args()
    if a.end_exclusive <= a.start:
        p.error("end-exclusive must be greater than start")
    if not a.confirm_prod_write:
        p.error("Explicit --confirm-prod-write required; stop scheduled ingestion and readers during cutover")
    run(a)


if __name__ == "__main__":
    main()
