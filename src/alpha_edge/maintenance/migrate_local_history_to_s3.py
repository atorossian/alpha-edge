# migrate_local_history_to_s3.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import boto3

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT = "engine/v1"
DEFAULT_LOCAL_HIST_ROOT = "data/history"

DT_DIR_RE = re.compile(r"^dt=(\d{4}-\d{2}-\d{2})$")


DEFAULT_TABLES = [
    "inputs/positions",
    "configs/score_config",
    "daily_reports",
    "holdings",
    "health",
    "regimes/hmm",

    # Legacy local names mapped below if present.
    "portfolio_search_runs",
    "portfolio_search_candidates",
    "recommended_portfolio_weights",
]


TABLE_RENAMES = {
    # old local folder -> current S3 table
    "portfolio_search_runs": "portfolio_search/runs",
    "portfolio_search_candidates": "portfolio_search/candidates",
    "recommended_portfolio_weights": "portfolio_search/weights",
}


PREFERRED_LATEST_JSON = {
    "inputs/positions": "positions.json",
    "configs/score_config": "score_config.json",
    "daily_reports": "report.json",
    "health": "health.json",
    "regimes/hmm": "hmm.json",
}


# ----------------------------
# Runtime helpers
# ----------------------------
def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET)).strip()


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION)).strip()


def cfg_engine_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "engine_root", DEFAULT_ENGINE_ROOT)).strip("/")


def cfg_env(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "env", "dev")).strip()


# ----------------------------
# Local/S3 helpers
# ----------------------------
def _iter_dt_folders(table_dir: Path) -> List[Tuple[str, Path]]:
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
        return "application/gzip"

    return "application/octet-stream"


def _upload_file(
    s3,
    *,
    bucket: str,
    key: str,
    path: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"[DRY RUN] would upload {path} -> s3://{bucket}/{key}")
        return

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=path.read_bytes(),
        ContentType=_guess_content_type(path),
    )


def _table_dest_name(local_table: str) -> str:
    t = str(local_table).strip("/")
    return TABLE_RENAMES.get(t, t)


def _table_dest_prefix(*, engine_root: str, local_table: str) -> str:
    return f"{engine_root.strip('/')}/{_table_dest_name(local_table)}"


def _dest_key_for_dt_file(
    *,
    engine_root: str,
    local_table: str,
    dt_str: str,
    filename: str,
) -> str:
    return f"{_table_dest_prefix(engine_root=engine_root, local_table=local_table)}/dt={dt_str}/{filename}"


def _latest_candidates(files: list[Path], *, dest_table: str) -> dict[str, Path]:
    """
    Returns extension -> file to copy as latest.<ext>.
    JSON:
      Prefer known canonical file names.
    CSV/PARQUET:
      Only write latest if there is exactly one file of that extension.
    """
    out: dict[str, Path] = {}

    json_files = [p for p in files if p.suffix.lower() == ".json"]
    if json_files:
        preferred_name = PREFERRED_LATEST_JSON.get(dest_table)

        preferred = None
        if preferred_name:
            for p in json_files:
                if p.name == preferred_name:
                    preferred = p
                    break

        out["json"] = preferred or sorted(json_files, key=lambda x: x.name)[0]

    csv_files = [p for p in files if p.suffix.lower() == ".csv"]
    if len(csv_files) == 1:
        out["csv"] = csv_files[0]

    pq_files = [p for p in files if p.suffix.lower() == ".parquet"]
    if len(pq_files) == 1:
        out["parquet"] = pq_files[0]

    return out


def _write_latest_from_dt(
    s3,
    *,
    bucket: str,
    engine_root: str,
    local_table: str,
    dt_folder: Path,
    dry_run: bool,
) -> int:
    dest_table = _table_dest_name(local_table)
    table_prefix = _table_dest_prefix(engine_root=engine_root, local_table=local_table)

    files = [p for p in dt_folder.iterdir() if p.is_file()]
    if not files:
        return 0

    latest = _latest_candidates(files, dest_table=dest_table)

    written = 0
    for ext, path in latest.items():
        key = f"{table_prefix}/latest.{ext}"
        _upload_file(s3, bucket=bucket, key=key, path=path, dry_run=dry_run)
        written += 1

    return written


def _select_files(files: list[Path], *, upload_all_files: bool) -> list[Path]:
    if upload_all_files:
        return files

    expected = {
        "positions.json",
        "score_config.json",
        "report.json",
        "health.json",
        "hmm.json",
        "holdings.csv",
        "holdings.parquet",
    }

    chosen = [p for p in files if p.name in expected]
    return chosen or files


def _parse_tables(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_TABLES)

    tables = [x.strip().strip("/") for x in raw.split(",") if x.strip()]
    if not tables:
        raise ValueError("--tables was provided but no valid table names were parsed")

    return tables


def _print_safety_banner(
    *,
    cfg: RuntimeConfig,
    bucket: str,
    region: str,
    engine_root: str,
    local_root: Path,
    tables: list[str],
    dry_run: bool,
    write_latest: bool,
    upload_all_files: bool,
) -> None:
    print("\n" + "=" * 88)
    print("MAINTENANCE: MIGRATE LOCAL HISTORY TO S3")
    print("=" * 88)
    print("This uploads local historical outputs into the configured S3 engine root.")
    print("Default mode is dry-run. Upload requires --execute.")
    print("")
    print(f"env:              {cfg_env(cfg)}")
    print(f"bucket:           {bucket}")
    print(f"region:           {region}")
    print(f"engine_root:      {engine_root}")
    print(f"local_root:       {local_root.resolve()}")
    print(f"dry_run:          {dry_run}")
    print(f"write_latest:     {write_latest}")
    print(f"upload_all_files: {upload_all_files}")
    print("")
    print("tables:")
    for t in tables:
        mapped = _table_dest_name(t)
        if mapped != t:
            print(f"  - {t} -> {mapped}")
        else:
            print(f"  - {t}")
    print("=" * 88 + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Migrate local Alpha Edge history folders into runtime-aware S3 engine root."
    )

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument("--bucket", default=None, help="Override runtime bucket. Usually avoid this.")
    ap.add_argument("--region", default=None, help="Override runtime region. Usually avoid this.")
    ap.add_argument("--engine-root", default=None, help="Override runtime engine root. Usually avoid this.")

    ap.add_argument("--local-root", default=DEFAULT_LOCAL_HIST_ROOT)
    ap.add_argument(
        "--tables",
        default=None,
        help="Comma-separated local table folders. Default: built-in migration list.",
    )

    ap.add_argument(
        "--only-expected-files",
        action="store_true",
        help="Upload only expected canonical files from each dt folder where possible.",
    )
    ap.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not write latest.json/latest.csv/latest.parquet pointers.",
    )

    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually upload files. Default is dry-run.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Deprecated compatibility flag. Default behavior is already dry-run unless --execute is passed.",
    )

    ap.add_argument("--reason", default=None, help="Required for real maintenance writes/deletes. Stored in audit trail.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.execute) and bool(args.dry_run):
        raise SystemExit("Conflicting flags: use either --execute or --dry-run, not both.")

    cfg = load_runtime_config(args.env)

    bucket = str(args.bucket or cfg_bucket(cfg)).strip()
    region = str(args.region or cfg_region(cfg)).strip()
    engine_root = str(args.engine_root or cfg_engine_root(cfg)).strip("/")
    local_root = Path(args.local_root)

    dry_run = not bool(args.execute)
    write_latest = not bool(args.no_latest)
    upload_all_files = not bool(args.only_expected_files)
    tables = _parse_tables(args.tables)

    if not local_root.exists():
        raise RuntimeError(f"Local history root not found: {local_root.resolve()}")

    if bool(args.execute):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    if engine_root in {"", "/", "engine", "market", "dev", "staging"}:
        raise SystemExit(f"Refusing unsafe/broad engine_root: {engine_root!r}")

    _print_safety_banner(
        cfg=cfg,
        bucket=bucket,
        region=region,
        engine_root=engine_root,
        local_root=local_root,
        tables=tables,
        dry_run=dry_run,
        write_latest=write_latest,
        upload_all_files=upload_all_files,
    )

    s3 = boto3.client("s3", region_name=region)

    total_uploaded = 0
    total_latest = 0
    total_tables = 0
    total_dt_folders = 0

    for local_table in tables:
        table_dir = local_root / local_table
        dts = _iter_dt_folders(table_dir)

        if not dts:
            print(f"[SKIP] {local_table} no dt= folders under {table_dir}")
            continue

        total_tables += 1
        dest_table = _table_dest_name(local_table)
        print(f"\n=== {local_table} -> {dest_table} ===")
        print(f"dt folders: {len(dts)} from {dts[0][0]} to {dts[-1][0]}")

        for dt_str, dt_folder in dts:
            total_dt_folders += 1

            files = [p for p in dt_folder.iterdir() if p.is_file()]
            files = _select_files(files, upload_all_files=upload_all_files)

            if not files:
                continue

            for p in files:
                dest_key = _dest_key_for_dt_file(
                    engine_root=engine_root,
                    local_table=local_table,
                    dt_str=dt_str,
                    filename=p.name,
                )
                _upload_file(s3, bucket=bucket, key=dest_key, path=p, dry_run=dry_run)
                total_uploaded += 1

            print(
                f"[OK] dt={dt_str} files={len(files)} -> "
                f"s3://{bucket}/{_table_dest_prefix(engine_root=engine_root, local_table=local_table)}/dt={dt_str}/"
            )

        if write_latest:
            latest_dt, latest_folder = dts[-1]
            n_latest = _write_latest_from_dt(
                s3,
                bucket=bucket,
                engine_root=engine_root,
                local_table=local_table,
                dt_folder=latest_folder,
                dry_run=dry_run,
            )
            total_latest += n_latest
            print(f"[OK] latest pointers from dt={latest_dt} count={n_latest}")

    print("\n--- summary ---")
    print(f"env={cfg_env(cfg)}")
    print(f"bucket={bucket}")
    print(f"engine_root={engine_root}")
    print(f"tables_migrated={total_tables}")
    print(f"dt_folders_seen={total_dt_folders}")
    print(f"objects_uploaded_or_planned={total_uploaded}")
    print(f"latest_pointers_uploaded_or_planned={total_latest}")

    print("\nExpected key examples:")
    print(f"  s3://{bucket}/{engine_root}/inputs/positions/latest.json")
    print(f"  s3://{bucket}/{engine_root}/configs/score_config/latest.json")

    if dry_run:
        print("\n[DRY RUN DONE] No uploads executed. Rerun with --execute to upload.")
    else:
        print("\n[UPLOAD DONE]")



# ----------------------------
# Audit/logging entrypoint wrapper
# ----------------------------
def main_with_audit() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = not bool(getattr(args, "execute", False))
    if bool(getattr(args, "execute", False)) and not getattr(args, "reason", None):
        raise SystemExit("--reason is required when using --execute for local history migration.")

    captured_output_keys: list[str] = []
    original_upload_file = globals()["_upload_file"]

    def _audited_upload_file(s3, *, bucket: str, key: str, path: Path, dry_run: bool) -> None:
        captured_output_keys.append(str(key))
        return original_upload_file(s3, bucket=bucket, key=key, path=path, dry_run=dry_run)

    with capture_script_run(cfg=cfg, script_name="migrate_local_history_to_s3.py", input_args=vars(args), dry_run=is_dry_run) as run_id:
        globals()["_upload_file"] = _audited_upload_file
        try:
            main()
            output_keys = sorted(set(captured_output_keys))
            event = build_audit_event(
                cfg=cfg, run_id=run_id, event_type="migrate", entity_type="local_history_to_s3",
                entity_id=None, as_of=None, source_script="migrate_local_history_to_s3.py",
                source_mode="migrate_local_history_to_s3", status=("dry_run" if is_dry_run else "success"),
                reason=getattr(args, "reason", None), input_args=vars(args), output_keys=([] if is_dry_run else output_keys),
                metadata={"tier": "tier_2", "payload_policy": "large_dataset_metadata_only", "planned_output_keys": output_keys,
                          "planned_output_count": len(output_keys), "local_root": getattr(args, "local_root", None),
                          "tables": getattr(args, "tables", None), "write_latest": not bool(getattr(args, "no_latest", False)),
                          "execute": bool(getattr(args, "execute", False))},
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
        except Exception as exc:
            output_keys = sorted(set(captured_output_keys))
            event = build_audit_event(
                cfg=cfg, run_id=run_id, event_type="migrate", entity_type="local_history_to_s3",
                entity_id=None, as_of=None, source_script="migrate_local_history_to_s3.py",
                source_mode="migrate_local_history_to_s3", status="failed", reason=getattr(args, "reason", None),
                input_args=vars(args), metadata={"tier": "tier_2", "payload_policy": "large_dataset_metadata_only",
                                                "planned_output_keys": output_keys, "planned_output_count": len(output_keys),
                                                "execute": bool(getattr(args, "execute", False))},
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise
        finally:
            globals()["_upload_file"] = original_upload_file


if __name__ == "__main__":
    main_with_audit()