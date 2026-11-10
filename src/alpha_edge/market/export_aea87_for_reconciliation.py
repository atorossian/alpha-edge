"""Read-only exporter of the *active manifest-selected* AEA-87 histories.

S3 permissions needed: s3:GetObject only. Refuses missing manifests, missing
parts, duplicate sessions, and unexpected schemas. Never scans orphan
partitions or writes any object to S3.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import boto3
import pandas as pd

from alpha_edge.market.reconcile_aea87_lake_v2 import ReconciliationError

DEFAULT_ASSETS = Path("data/quality/market_lake_v2/pilot_six_assets.csv")
TABLES = {
    "ohlcv_usd": ("close_adjusted_usd", "aea87_prices.parquet"),
    "returns_usd": ("ret_close_adjusted_usd", "aea87_returns.parquet"),
}


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _object(s3, bucket: str, key: str) -> bytes:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception as exc:
        raise ReconciliationError(f"Missing/unreadable required S3 object: s3://{bucket}/{key}") from exc


def _session_dates(values: pd.Series) -> pd.Series:
    # Daily session labels must not be timezone-converted to UTC.
    return pd.to_datetime(values.astype(str).str[:10], format="%Y-%m-%d", errors="coerce")


def fetch_table(
    s3, *, bucket: str, root: str, version: str,
    table: str, asset_ids: list[str], years: list[int],
    start: str, end_exclusive: str,
) -> tuple[pd.DataFrame, list[dict]]:
    value_column = TABLES[table][0]
    data = []
    provenance = []
    for asset_id in asset_ids:
        for year in years:
            manifest_key = (
                f"{root}/manifests/{version}/{table}/"
                f"asset_id={asset_id}/year={year}/manifest.json"
            )
            original_manifest = _object(s3, bucket, manifest_key)
            try:
                manifest = json.loads(original_manifest)
            except (ValueError, TypeError) as exc:
                raise ReconciliationError(f"Invalid JSON manifest: {manifest_key}") from exc
            if str(manifest.get("asset_id")) != asset_id or int(manifest.get("year", -1)) != year:
                raise ReconciliationError(f"Manifest identity mismatch: {manifest_key}")
            parts = manifest.get("parts")
            if not isinstance(parts, list):
                raise ReconciliationError(f"Manifest lacks explicit parts: {manifest_key}")
            if not parts and manifest.get("status") not in ("RETIRED", "EMPTY", "TOMBSTONED") and not manifest.get("retired"):
                raise ReconciliationError(f"Empty active manifest needs investigation: {manifest_key}")
            record = {
                "asset_id": asset_id, "table": table, "year": year,
                "manifest_key": manifest_key, "manifest_sha256": sha256(original_manifest),
                "parts": [],
            }
            for part_key in parts:
                expected_prefix = f"{root}/{table}/{version}/asset_id={asset_id}/year={year}/"
                if not isinstance(part_key, str) or not part_key.startswith(expected_prefix) or not part_key.endswith(".parquet"):
                    raise ReconciliationError(f"Unsafe or unexpected active part: {part_key}")
                raw = _object(s3, bucket, part_key)
                try:
                    frame = pd.read_parquet(io.BytesIO(raw))
                except Exception as exc:
                    raise ReconciliationError(f"Corrupt active partition: {part_key}") from exc
                if frame.empty:
                    raise ReconciliationError(f"Empty active partition: {part_key}")
                required = {"asset_id", "date", value_column}
                if missing := required - set(frame.columns):
                    raise ReconciliationError(f"{part_key} missing fields: {sorted(missing)}")
                if not frame["asset_id"].astype(str).eq(asset_id).all():
                    raise ReconciliationError(f"Partition asset mismatch: {part_key}")
                frame = frame[["asset_id", "date", value_column]].copy()
                frame["date"] = _session_dates(frame["date"])
                if frame["date"].isna().any() or not frame["date"].dt.year.eq(year).all():
                    raise ReconciliationError(f"Invalid session/year in {part_key}")
                data.append(frame)
                record["parts"].append({"key": part_key, "sha256": sha256(raw), "rows": len(frame)})
            provenance.append(record)

    if not data:
        raise ReconciliationError(f"No active {table} rows in selected years")
    result = pd.concat(data, ignore_index=True)
    result = result.loc[
        result["date"].ge(pd.Timestamp(start)) &
        result["date"].lt(pd.Timestamp(end_exclusive))
    ].copy()
    if result.empty:
        raise ReconciliationError(f"No {table} rows within requested dates")
    if result.duplicated(["asset_id", "date"]).any():
        samples = result.loc[result.duplicated(["asset_id", "date"], keep=False),
                             ["asset_id", "date"]].head(5).to_dict("records")
        raise ReconciliationError(f"Duplicate active {table} asset/session rows: {samples}")
    return result.sort_values(["asset_id", "date"]).reset_index(drop=True), provenance


def export(
    *, s3, bucket: str, universe_csv: Path, start: str, end_exclusive: str,
    output: Path, root: str = "market", version: str = "v1",
) -> dict:
    if root != "market" or version != "v1":
        raise ReconciliationError("This exporter is restricted to read-only AEA-87 market/v1")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    begin, end = pd.Timestamp(start), pd.Timestamp(end_exclusive)
    if begin >= end:
        raise ReconciliationError("End date must be after start date")
    assets = pd.read_csv(universe_csv)
    if "include" in assets:
        assets = assets.loc[assets["include"].eq(1)]
    ids = assets["asset_id"].dropna().astype(str).str.strip().tolist()
    if len(ids) != 6 or len(ids) != len(set(ids)):
        raise ReconciliationError("Expected six unique pilot asset IDs")
    years = list(range(begin.year, (end - pd.Timedelta(days=1)).year + 1))
    tables, journals = {}, {}
    for name, (_, filename) in TABLES.items():
        tables[filename], journals[name] = fetch_table(
            s3, bucket=bucket, root=root, version=version, table=name,
            asset_ids=ids, years=years, start=start, end_exclusive=end_exclusive,
        )
        if set(tables[filename]["asset_id"].unique()) != set(ids):
            raise ReconciliationError(f"Missing assets from active {name} partitions")
    output.mkdir(parents=True)
    file_hashes = {}
    for filename, df in tables.items():
        target = output / filename
        df.to_parquet(target, index=False)
        file_hashes[filename] = sha256(target.read_bytes())
    metadata = {
        "schema": "aea87_active_manifest_readonly_export_v1",
        "bucket": bucket, "root": root, "version": version,
        "start_inclusive": start, "end_exclusive": end_exclusive,
        "assets": ids,
        "return_semantics": "unknown_requires_manual_provenance_check",
        "files": {key: {"rows": len(df), "sha256": file_hashes[key]}
                  for key, df in tables.items()},
        "active_source_manifests": journals,
        "s3_writes": 0,
        "note": "Snapshot of manifests and parts read sequentially; not a transactional S3 snapshot. "
                "Review manifest stability during long-running historical extracts.",
    }
    (output / "export_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--region", default="eu-west-1")
    ap.add_argument("--universe-csv", type=Path, default=DEFAULT_ASSETS)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-11-10", help="Exclusive")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    print(json.dumps(export(
        s3=s3, bucket=args.bucket, universe_csv=args.universe_csv,
        start=args.start, end_exclusive=args.end, output=args.output,
    ), indent=2))


if __name__ == "__main__":
    main()
