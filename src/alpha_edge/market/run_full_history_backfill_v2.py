"""Resumable, explicit, local-only full-history cohort backfill.

Uses approved immutable preflight plan. Creates one independent execution
directory per cohort attempt, audits missing OHLCV fields/sessions automatically
and keeps a local progress ledger. Does not publish any lake or S3 pointer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from alpha_edge.market.audit_missing_ohlcv_v2 import generate as audit_missing
from alpha_edge.market.run_cohort_bronze_v2 import load_plan, run as run_cohort


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latest_attempt(root: Path, index: int) -> tuple[int, Path | None, dict | None]:
    matches = sorted(root.glob(f"cohort_{index:04d}_attempt_*"))
    if not matches:
        return 0, None, None
    last = matches[-1]
    suffix = last.name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"Unexpected attempt directory: {last}")
    manifest_file = last / "manifest.json"
    if not manifest_file.is_file():
        # Preserve interrupted attempt immutably; create a new attempt on resume.
        return int(suffix), last, {"state": "interrupted"}
    return int(suffix), last, json.loads(manifest_file.read_text())


def run_full(*, plan_dir: Path, output: Path, end: str, execute: bool = False,
             start_cohort: int = 1, max_cohorts: int = 1, retries: int = 2,
             delay_seconds: float = 1.0) -> dict:
    plan, frames, _ = load_plan(plan_dir)
    if (not end or start_cohort < 1 or max_cohorts < 1
            or start_cohort > len(frames)):
        raise ValueError("Invalid cohort range or exclusive end date")
    indices = list(range(start_cohort, min(start_cohort + max_cohorts,
                                           len(frames) + 1)))
    plan_sha = _digest(plan_dir / "manifest.json")
    if not execute:
        return {
            "status": "dry_run", "cohorts": indices,
            "assets": sum(len(frames[i-1]) for i in indices),
            "full_history": True, "s3_writes": 0,
        }

    output.mkdir(parents=True, exist_ok=True)
    anchor = output / "plan_reference.json"
    expected_anchor = {"plan_manifest_sha256": plan_sha, "end_exclusive": end}
    if anchor.exists():
        if json.loads(anchor.read_text()) != expected_anchor:
            raise ValueError("Refusing to resume with different plan or end date")
    else:
        if any(output.iterdir()):
            raise ValueError("Nonempty output without trusted plan reference")
        anchor.write_text(json.dumps(expected_anchor, indent=2) + "\n")
    runs = output / "runs"
    audits = output / "audits"
    runs.mkdir(exist_ok=True)
    audits.mkdir(exist_ok=True)
    records = []
    for index in indices:
        last_n, last_dir, last_manifest = _latest_attempt(runs, index)
        if last_manifest and last_manifest.get("plan_manifest_sha256") != plan_sha:
            raise ValueError("Existing cohort attempt from another plan")
        if last_manifest and last_manifest.get("requested_end_exclusive") != end:
            raise ValueError("Existing cohort attempt has different source range")
        if last_manifest and last_manifest.get("state") == "complete":
            result = last_manifest
            attempt_dir = last_dir
            attempt = last_n
        else:
            attempt = last_n + 1
            if attempt > 99:
                raise ValueError("Exceeded 99 attempts per cohort")
            attempt_dir = runs / f"cohort_{index:04d}_attempt_{attempt:03d}"
            result = run_cohort(
                plan_dir=plan_dir, output=attempt_dir,
                batch_prefix=f"backfill-{index:04d}-a{attempt:03d}",
                start=None, end=end, full_history=True, execute=True,
                start_cohort=index, max_cohorts=1, retries=retries,
                delay_seconds=delay_seconds,
            )
        audit_state = "not_acquired"
        audit_path = audits / f"cohort_{index:04d}_attempt_{attempt:03d}"
        if result["state"] == "complete":
            batch_id = f"backfill-{index:04d}-a{attempt:03d}-{index:04d}"
            bronze = attempt_dir / "bronze_batches" / batch_id
            if not bronze.joinpath("manifest.json").is_file():
                raise ValueError(f"Missing acquired immutable Bronze: {bronze}")
            if not audit_path.exists():
                audit_missing(
                    bronze, plan_dir / f"cohort_{index:04d}.csv",
                    plan_dir / "fx_config.csv", audit_path,
                )
            audit_state = "complete"
        records.append({
            "cohort_number": index, "attempt": attempt,
            "assets": len(frames[index - 1]),
            "bronze_state": result["state"], "audit_state": audit_state,
            "bronze_dir": (str(bronze) if audit_state == "complete" else ""),
            "audit_dir": str(audit_path) if audit_state == "complete" else "",
        })
        # Checkpoint after every cohort. This progress file is mutable;
        # the referenced Bronze objects and audits are immutable.
        (output / "latest_progress.json").write_text(json.dumps({
            "schema": "alpha_edge_full_history_progress_v2_1",
            "plan_manifest_sha256": plan_sha, "runs": records,
            "production_eligible": False, "s3_writes": 0,
        }, indent=2) + "\n")
    return {
        "schema": "alpha_edge_full_history_progress_v2_1",
        "cohorts_attempted": len(indices),
        "cohorts_successful": sum(r["bronze_state"] == "complete" for r in records),
        "cohorts_failed": sum(r["bronze_state"] != "complete" for r in records),
        "assets_in_successful_cohorts": sum(
            r["assets"] for r in records if r["bronze_state"] == "complete"),
        "all_cohorts_in_plan": len(frames),
        "plan_assets": plan["assets_included"],
        "s3_writes": 0, "production_eligible": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--start-cohort", type=int, default=1)
    p.add_argument("--max-cohorts", type=int, default=1)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--delay-seconds", type=float, default=1.0)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    print(json.dumps(run_full(plan_dir=a.plan_dir, output=a.output,
                              end=a.end, start_cohort=a.start_cohort,
                              max_cohorts=a.max_cohorts, retries=a.retries,
                              delay_seconds=a.delay_seconds,
                              execute=a.execute), indent=2))


if __name__ == "__main__":
    main()
