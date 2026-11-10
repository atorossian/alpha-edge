"""Bounded, local-only cohort Bronze acquisition from a hashed preflight plan.

Default is plan verification only. Explicit --execute performs Yahoo requests,
one cohort at a time, with bounded retries and immutable local outputs.
Failure of one cohort is reported without admitting an incomplete batch.
No S3 upload or production publication is supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from alpha_edge.market.ingest_bronze_yahoo_v2 import run_batch


class CohortRunError(ValueError):
    pass


SAFE_RUN = re.compile(r"^[A-Za-z0-9_-]{1,60}$")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(plan_dir: Path) -> tuple[dict, list[pd.DataFrame], list[str]]:
    manifest_path = plan_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise CohortRunError("Missing immutable preflight manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "alpha_edge_lake_v2_universe_preflight_1":
        raise CohortRunError("Preflight contract mismatch")
    files = manifest.get("files_sha256")
    if not isinstance(files, dict):
        raise CohortRunError("Preflight files are not checksum-pinned")
    if set(files) != {"fx_config.csv"} | {
        f"cohort_{i:04d}.csv" for i in range(1, manifest["cohorts"] + 1)
    }:
        raise CohortRunError("Preflight file inventory differs")
    for name, expected in files.items():
        p = plan_dir / name
        if not p.is_file() or p.is_symlink() or _sha(p) != expected:
            raise CohortRunError(f"Preflight file checksum mismatch: {name}")
    frames = [pd.read_csv(plan_dir / f"cohort_{i:04d}.csv")
              for i in range(1, manifest["cohorts"] + 1)]
    if sum(len(f) for f in frames) != manifest["assets_included"]:
        raise CohortRunError("Preflight asset coverage mismatch")
    all_ids = pd.concat([f["asset_id"] for f in frames], ignore_index=True)
    if all_ids.duplicated().any():
        raise CohortRunError("Preflight duplicate asset IDs across cohorts")
    symbols = pd.read_csv(plan_dir / "fx_config.csv")["provider_symbol"].tolist()
    return manifest, frames, symbols


def run(*, plan_dir: Path, output: Path, batch_prefix: str,
        execute: bool = False, max_cohorts: int = 1, start_cohort: int = 1,
        retries: int = 2,
        delay_seconds: float = 1.0,
        start: str, end: str,
        runner: Callable = run_batch, sleeper: Callable = time.sleep) -> dict:
    if not SAFE_RUN.fullmatch(batch_prefix):
        raise CohortRunError("Invalid batch prefix")
    if max_cohorts < 1 or retries < 0 or retries > 3 or delay_seconds < 0:
        raise CohortRunError("Invalid execution limits")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise CohortRunError("Invalid ingestion date window")
    manifest, frames, symbols = load_plan(plan_dir)
    if start_cohort < 1 or start_cohort > len(frames):
        raise CohortRunError('Starting cohort out of range')
    selected = list(enumerate(frames, start=1))[start_cohort - 1:start_cohort - 1 + max_cohorts]
    if not execute:
        return {
            "status": "dry_run", "planned_cohorts": len(selected),
            "planned_assets": sum(len(f) for _, f in selected),
            "total_cohorts_in_plan": len(frames), "start_cohort": start_cohort, "s3_writes": 0,
        }
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite cohort execution: {output}")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for idx, subset in selected:
        batch_id = f"{batch_prefix}-{idx:04d}"
        error = None
        success = None
        for attempt in range(retries + 1):
            try:
                # Local-only, no --upload or S3 credentials passed.
                success = runner(
                    universe=subset, fx_symbols=symbols,
                    start=start, end=end, batch_id=batch_id,
                    output=str(output / "bronze_batches"), upload=False,
                )
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    sleeper(delay_seconds * 2 ** attempt)
        rows.append({
            "cohort_number": idx, "batch_id": batch_id,
            "assets": len(subset),
            "state": "acquired" if success else "failed",
            "error": None if success else error,
            "local_path": success.get("local_path") if success else None,
        })
    result = pd.DataFrame(rows)
    result.to_csv(output / "cohort_execution.csv", index=False)
    digest = _sha(output / "cohort_execution.csv")
    report = {
        "schema": "alpha_edge_cohort_execution_v2_1",
        "plan_manifest_sha256": _sha(plan_dir / "manifest.json"),
        "requested_start": start, "requested_end_exclusive": end,
        "attempted_cohorts": len(selected),
        "acquired_cohorts": int(result["state"].eq("acquired").sum()),
        "failed_cohorts": int(result["state"].eq("failed").sum()),
        "assets_in_successful_cohorts": int(
            result.loc[result["state"].eq("acquired"), "assets"].sum()),
        "state": "complete" if result["state"].eq("acquired").all() else "partial_failed",
        "s3_writes": 0, "production_eligible": False,
        "files_sha256": {"cohort_execution.csv": digest},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-prefix", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-cohorts", type=int, default=1)
    p.add_argument("--start-cohort", type=int, default=1)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--delay-seconds", type=float, default=1)
    a = p.parse_args()
    print(json.dumps(run(
        plan_dir=a.plan_dir, output=a.output, batch_prefix=a.batch_prefix,
        start=a.start, end=a.end, execute=a.execute,
        max_cohorts=a.max_cohorts, start_cohort=a.start_cohort,
        retries=a.retries,
        delay_seconds=a.delay_seconds,
    ), indent=2))


if __name__ == "__main__":
    main()
