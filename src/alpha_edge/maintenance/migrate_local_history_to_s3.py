# migrate_local_history_to_s3.py
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3


# -----------------------
# CONFIG
# -----------------------
LOCAL_HIST_ROOT = Path("data/history")          # your local history root
S3_BUCKET = "alpha-edge-algo"
S3_REGION = "eu-west-1"
S3_ENGINE_ROOT = "engine/v1"                   # destination prefix in S3

# Which local tables to migrate (relative to LOCAL_HIST_ROOT).
# Add/remove as you like; these match your local writes in run_daily_report/run_portfolio_search.
TABLES = [
    "inputs/positions",
    "configs/score_config",
    "daily_reports",
    "holdings",
    "health",
    "regimes/hmm",
    "portfolio_search_runs",
    "portfolio_search_candidates",
    "recommended_portfolio_weights",
]

# If True: uploads every file found under each dt= folder (json/csv/parquet/anything).
UPLOAD_ALL_FILES_IN_DT = True

# If True: after upload, create engine/v1/<table>/latest.json (and/or latest.csv/latest.parquet)
WRITE_LATEST_POINTERS = True


# -----------------------
# Helpers
# -----------------------
DT_DIR_RE = re.compile(r"^dt=(\d{4}-\d{2}-\d{2})$")


def _iter_dt_folders(table_dir: Path) -> List[Tuple[str, Path]]:
    """
    Returns list of (dt_str, path) for dt=YYYY-MM-DD folders.
    """
    if not table_dir.exists():
        return []
    out: List[Tuple[str, Path]] = []
    for p in table_dir.iterdir():
        if not p.is_dir():
            continue
        m = DT_DIR_RE.match(p.name)
        if not m:
            continue
        out.append((m.group(1), p))
    out.sort(key=lambda x: x[0])
    return out


def _guess_content_type(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".json":
        return "application/json"
    if suf == ".csv":
        return "text/csv"
    if suf == ".parquet":
        return "application/octet-stream"
    if suf == ".gz":
        # could be .csv.gz; keep generic
        return "application/gzip"
    return "application/octet-stream"


def _upload_file(s3, *, bucket: str, key: str, path: Path) -> None:
    body = path.read_bytes()
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=_guess_content_type(path),
    )


def _table_dest_prefix(table: str) -> str:
    table = table.strip("/")
    return f"{S3_ENGINE_ROOT}/{table}"


def _dest_key_for_dt_file(table: str, dt_str: str, filename: str) -> str:
    return f"{_table_dest_prefix(table)}/dt={dt_str}/{filename}"


def _write_latest_from_dt(s3, *, table: str, dt_str: str, dt_folder: Path) -> None:
    """
    Writes engine/v1/<table>/latest.<ext> for common extensions if present in newest dt folder.
    - If multiple JSON files exist, we choose:
        - 'positions.json' for inputs/positions
        - 'score_config.json' for configs/score_config
        - 'report.json' for daily_reports
        - 'health.json' for health
        - 'hmm.json' for regimes/hmm
      Else: pick the first json file in folder.
    - For CSV/Parquet: if exactly one exists, we copy it to latest.<ext>
    """
    table_prefix = _table_dest_prefix(table)

    files = [p for p in dt_folder.iterdir() if p.is_file()]
    if not files:
        return

    # ---- JSON latest.json ----
    json_files = [p for p in files if p.suffix.lower() == ".json"]
    preferred = None
    if json_files:
        preferred_name = None
        if table == "inputs/positions":
            preferred_name = "positions.json"
        elif table == "configs/score_config":
            preferred_name = "score_config.json"
        elif table == "daily_reports":
            preferred_name = "report.json"
        elif table == "health":
            preferred_name = "health.json"
        elif table == "regimes/hmm":
            preferred_name = "hmm.json"

        if preferred_name:
            for p in json_files:
                if p.name == preferred_name:
                    preferred = p
                    break

        if preferred is None:
            preferred = json_files[0]

        latest_key = f"{table_prefix}/latest.json"
        _upload_file(s3, bucket=S3_BUCKET, key=latest_key, path=preferred)

    # ---- CSV latest.csv (only if unambiguous) ----
    csv_files = [p for p in files if p.suffix.lower() == ".csv"]
    if len(csv_files) == 1:
        latest_key = f"{table_prefix}/latest.csv"
        _upload_file(s3, bucket=S3_BUCKET, key=latest_key, path=csv_files[0])

    # ---- Parquet latest.parquet (only if unambiguous) ----
    pq_files = [p for p in files if p.suffix.lower() == ".parquet"]
    if len(pq_files) == 1:
        latest_key = f"{table_prefix}/latest.parquet"
        _upload_file(s3, bucket=S3_BUCKET, key=latest_key, path=pq_files[0])


def main():
    s3 = boto3.client("s3", region_name=S3_REGION)

    if not LOCAL_HIST_ROOT.exists():
        raise RuntimeError(f"Local history root not found: {LOCAL_HIST_ROOT.resolve()}")

    print(f"Local root: {LOCAL_HIST_ROOT.resolve()}")
    print(f"S3 dest:    s3://{S3_BUCKET}/{S3_ENGINE_ROOT}/")
    print("")

    total_uploaded = 0
    total_tables = 0
    total_dt_folders = 0

    for table in TABLES:
        table_dir = LOCAL_HIST_ROOT / table
        dts = _iter_dt_folders(table_dir)
        if not dts:
            print(f"[SKIP] {table} (no dt= folders under {table_dir})")
            continue

        total_tables += 1
        print(f"\n=== {table} ===")
        print(f"dt folders: {len(dts)}  (from {dts[0][0]} to {dts[-1][0]})")

        # Upload dt partitions
        for dt_str, dt_folder in dts:
            total_dt_folders += 1
            files = [p for p in dt_folder.iterdir() if p.is_file()]
            if not files:
                continue

            if not UPLOAD_ALL_FILES_IN_DT:
                # minimal: upload only "expected" files if present, else all
                expected = {"positions.json", "score_config.json", "report.json", "health.json", "hmm.json",
                            "holdings.csv", "holdings.parquet"}
                chosen = [p for p in files if p.name in expected]
                if not chosen:
                    chosen = files
                files = chosen

            for p in files:
                dest_key = _dest_key_for_dt_file(table, dt_str, p.name)
                _upload_file(s3, bucket=S3_BUCKET, key=dest_key, path=p)
                total_uploaded += 1

            print(f"[OK] dt={dt_str} files={len(files)} -> s3://{S3_BUCKET}/{_table_dest_prefix(table)}/dt={dt_str}/")

        # Write latest pointers from newest dt folder
        if WRITE_LATEST_POINTERS:
            latest_dt, latest_folder = dts[-1]
            _write_latest_from_dt(s3, table=table, dt_str=latest_dt, dt_folder=latest_folder)
            print(f"[OK] latest pointers written from dt={latest_dt}")

    print("\n--- summary ---")
    print(f"tables_migrated={total_tables}")
    print(f"dt_folders_seen={total_dt_folders}")
    print(f"objects_uploaded={total_uploaded}")
    print("\nNow your S3-only scripts should find:")
    print(f"  s3://{S3_BUCKET}/{S3_ENGINE_ROOT}/inputs/positions/latest.json")
    print(f"  s3://{S3_BUCKET}/{S3_ENGINE_ROOT}/configs/score_config/latest.json")


if __name__ == "__main__":
    main()
