#!/usr/bin/env python3
"""
Extract historical price rows from a partitioned S3 OHLCV/price dataset for assets
flagged by Alpha Edge returns-quality analysis.

Typical use cases:

1) One asset full history:
   python extract_price_history_for_bad_returns.py \
     --prices-s3-prefix s3://alpha-edge-algo/market/ohlcv_usd/v1/ \
     --asset-id EQH3feb8c9273a6a72f \
     --out-dir data/quality/returns/price_extract \
     --aws-region eu-west-1

2) Sample top bad assets from returns_quality_assets.csv:
   python extract_price_history_for_bad_returns.py \
     --prices-s3-prefix s3://alpha-edge-algo/market/ohlcv_usd/v1/ \
     --asset-ids-csv data/quality/returns/latest/returns_quality_assets.csv \
     --top-n-assets 50 \
     --out-dir data/quality/returns/price_extract \
     --aws-region eu-west-1

3) Extract windows around bad cells:
   python extract_price_history_for_bad_returns.py \
     --prices-s3-prefix s3://alpha-edge-algo/market/ohlcv_usd/v1/ \
     --bad-cells-csv data/quality/returns/latest/returns_quality_cells.csv \
     --top-n-assets 50 \
     --days-around-bad-dates 7 \
     --out-dir data/quality/returns/price_extract \
     --aws-region eu-west-1

Assumptions:
- S3 dataset is partitioned by asset, commonly: .../asset_id=<ASSET>/year=<YYYY>/*.parquet
- Parquet files contain at least a date column and one price column.
- If asset_id is not inside the parquet file, it is recovered from the partition path.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote, unquote

import boto3
import numpy as np
import pandas as pd


DEFAULT_PRICE_COLUMNS = [
    "adj_close_usd",
    "close_adjusted_usd",
    "close_usd",
    "adj_close",
    "close_adjusted",
    "close",
]


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


def parse_s3_uri(uri: str) -> S3Uri:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return S3Uri(bucket=bucket, key=key.strip("/"))


def list_s3_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.strip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key and key.endswith(".parquet"):
                keys.append(key)
    return sorted(keys)


def get_bytes(s3, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def read_parquet_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    raw = get_bytes(s3, bucket, key)
    return pd.read_parquet(io.BytesIO(raw))


def asset_from_key(key: str) -> Optional[str]:
    m = re.search(r"(?:^|/)asset_id=([^/]+)", key)
    if not m:
        return None
    return unquote(m.group(1))


def candidate_asset_prefixes(base_key: str, asset_id: str) -> list[str]:
    base = base_key.strip("/")
    raw = asset_id.strip()
    encoded = quote(raw, safe="")
    candidates = [
        f"{base}/asset_id={raw}",
        f"{base}/asset_id={encoded}",
    ]
    # Some writers may use URL-safe partitioning but leave ':' unescaped; dedupe.
    out = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def select_assets_from_reports(
    *,
    explicit_assets: list[str],
    asset_ids_csv: Optional[str],
    bad_cells_csv: Optional[str],
    top_n_assets: int,
) -> tuple[list[str], pd.DataFrame]:
    selected: list[str] = []
    rows = []

    for asset in explicit_assets:
        asset = str(asset).strip()
        if asset:
            selected.append(asset)
            rows.append({"asset_id": asset, "selection_source": "explicit"})

    if asset_ids_csv:
        df = pd.read_csv(asset_ids_csv)
        if "asset_id" not in df.columns:
            raise RuntimeError(f"{asset_ids_csv} must contain asset_id")
        sort_cols = [c for c in ["severity", "abs_gt_block", "gt_100pct", "le_minus_100pct", "max_abs_return"] if c in df.columns]
        if "max_abs_return" in df.columns:
            df = df.sort_values("max_abs_return", ascending=False)
        elif sort_cols:
            df = df.sort_values(sort_cols, ascending=False)
        for _, r in df.head(top_n_assets).iterrows():
            asset = str(r["asset_id"]).strip()
            if asset:
                selected.append(asset)
                rr = r.to_dict()
                rr["selection_source"] = "asset_report"
                rows.append(rr)

    if bad_cells_csv:
        df = pd.read_csv(bad_cells_csv)
        if "asset_id" not in df.columns:
            raise RuntimeError(f"{bad_cells_csv} must contain asset_id")
        # Pick assets with the largest absolute reported return cells first.
        if "abs_return" in df.columns:
            df = df.sort_values("abs_return", ascending=False)
        elif "return" in df.columns:
            df["abs_return"] = pd.to_numeric(df["return"], errors="coerce").abs()
            df = df.sort_values("abs_return", ascending=False)
        for asset, g in df.groupby("asset_id", sort=False):
            if len(selected) >= top_n_assets and not explicit_assets and not asset_ids_csv:
                break
            asset = str(asset).strip()
            if asset:
                selected.append(asset)
                top = g.head(1).iloc[0].to_dict()
                top["selection_source"] = "bad_cells_report"
                top["bad_cell_count_for_asset"] = int(len(g))
                rows.append(top)

    # Preserve order, dedupe.
    seen = set()
    ordered = []
    for asset in selected:
        if asset not in seen:
            seen.add(asset)
            ordered.append(asset)

    if top_n_assets and not explicit_assets:
        ordered = ordered[:top_n_assets]

    return ordered, pd.DataFrame(rows).drop_duplicates(subset=["asset_id"], keep="first") if rows else pd.DataFrame({"asset_id": ordered})


def load_bad_date_windows(bad_cells_csv: Optional[str], days: int) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    if not bad_cells_csv or days <= 0:
        return {}
    df = pd.read_csv(bad_cells_csv)
    if "asset_id" not in df.columns or "date" not in df.columns:
        return {}
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for asset, g in df.groupby("asset_id"):
        lo = g["date"].min() - pd.Timedelta(days=days)
        hi = g["date"].max() + pd.Timedelta(days=days)
        windows[str(asset)] = (lo, hi)
    return windows


def normalize_price_frame(df: pd.DataFrame, *, asset_id: str, source_key: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "asset_id" not in out.columns:
        out["asset_id"] = asset_id
    if "date" not in out.columns:
        # Common alternatives.
        for candidate in ["as_of", "timestamp", "datetime", "Date"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "date"})
                break
    if "date" not in out.columns:
        raise RuntimeError(f"No date column in {source_key}; columns={out.columns.tolist()}")
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    out = out.dropna(subset=["date"])
    out["source_s3_key"] = source_key
    return out


def add_recomputed_returns(df: pd.DataFrame, price_col: Optional[str]) -> pd.DataFrame:
    out = df.sort_values(["asset_id", "date"]).copy()
    if not price_col:
        for candidate in DEFAULT_PRICE_COLUMNS:
            if candidate in out.columns:
                price_col = candidate
                break
    if not price_col or price_col not in out.columns:
        out["selected_price_col"] = None
        out["recalc_simple_return"] = np.nan
        out["recalc_log_return"] = np.nan
        return out
    price = pd.to_numeric(out[price_col], errors="coerce")
    out["selected_price_col"] = price_col
    out["price_for_return"] = price
    out["prev_price_for_return"] = out.groupby("asset_id")["price_for_return"].shift(1)
    out["next_price_for_return"] = out.groupby("asset_id")["price_for_return"].shift(-1)
    out["recalc_simple_return"] = out.groupby("asset_id")["price_for_return"].pct_change()
    out["recalc_log_return"] = out.groupby("asset_id")["price_for_return"].transform(lambda s: np.log(s).diff())
    out["recalc_next_simple_return"] = out.groupby("asset_id")["price_for_return"].transform(lambda s: s.shift(-1) / s - 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract S3 historical prices for assets flagged by returns-quality reports.")
    ap.add_argument("--prices-s3-prefix", required=True, help="Partitioned prices/OHLCV S3 prefix, e.g. s3://bucket/market/ohlcv_usd/v1/")
    ap.add_argument("--asset-id", action="append", default=[], help="Asset ID to extract. Repeatable.")
    ap.add_argument("--asset-ids-csv", default=None, help="CSV with asset_id, e.g. returns_quality_assets.csv or returns_quarantine_assets.csv")
    ap.add_argument("--bad-cells-csv", default=None, help="CSV with date, asset_id, return, e.g. returns_quality_cells.csv")
    ap.add_argument("--top-n-assets", type=int, default=25)
    ap.add_argument("--days-around-bad-dates", type=int, default=0, help="If >0, filter each asset to min/max bad dates +/- N days.")
    ap.add_argument("--price-col", default=None, help="Price column to use for recomputed returns; auto-detected if omitted.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--aws-region", default=None)
    ap.add_argument("--scan-all-partitions", action="store_true", help="Fallback: list all parquet files under prefix and match /asset_id=. More expensive.")
    args = ap.parse_args()

    uri = parse_s3_uri(args.prices_s3_prefix)
    s3 = boto3.client("s3", region_name=args.aws_region)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets, selected_meta = select_assets_from_reports(
        explicit_assets=args.asset_id,
        asset_ids_csv=args.asset_ids_csv,
        bad_cells_csv=args.bad_cells_csv,
        top_n_assets=args.top_n_assets,
    )
    if not assets:
        raise RuntimeError("No assets selected. Provide --asset-id, --asset-ids-csv, or --bad-cells-csv.")

    windows = load_bad_date_windows(args.bad_cells_csv, args.days_around_bad_dates)

    # Optional global index of all partition keys if exact prefixes fail.
    all_keys: Optional[list[str]] = None
    extracted_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []

    for i, asset in enumerate(assets, start=1):
        print(f"[{i}/{len(assets)}] extracting {asset}")
        keys: list[str] = []
        attempted_prefixes = []
        for prefix in candidate_asset_prefixes(uri.key, asset):
            attempted_prefixes.append(prefix)
            keys = list_s3_keys(s3, uri.bucket, prefix)
            if keys:
                break
        if not keys and args.scan_all_partitions:
            if all_keys is None:
                print("Listing all parquet files under base prefix for fallback matching...")
                all_keys = list_s3_keys(s3, uri.bucket, uri.key)
            keys = [k for k in all_keys if asset_from_key(k) == asset]

        if not keys:
            manifest_rows.append({
                "asset_id": asset,
                "status": "not_found",
                "n_files": 0,
                "attempted_prefixes": " | ".join(attempted_prefixes),
            })
            print(f"  ! no parquet files found for {asset}")
            continue

        frames = []
        for key in keys:
            try:
                f = read_parquet_s3(s3, uri.bucket, key)
                frames.append(normalize_price_frame(f, asset_id=asset, source_key=key))
            except Exception as exc:
                manifest_rows.append({
                    "asset_id": asset,
                    "status": "read_failed",
                    "s3_key": key,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        if not frames:
            continue

        df = pd.concat(frames, ignore_index=True, sort=False)
        df = df.sort_values(["asset_id", "date"]).drop_duplicates(subset=["asset_id", "date"], keep="last")

        if asset in windows:
            lo, hi = windows[asset]
            df = df[(df["date"] >= lo) & (df["date"] <= hi)].copy()

        df = add_recomputed_returns(df, args.price_col)
        extracted_frames.append(df)
        asset_file = out_dir / f"price_history_{asset.replace(':','_').replace('/','_')}.csv"
        df.to_csv(asset_file, index=False)

        manifest_rows.append({
            "asset_id": asset,
            "status": "ok",
            "n_files": len(keys),
            "n_rows": int(len(df)),
            "first_date": str(df["date"].min().date()) if len(df) else None,
            "last_date": str(df["date"].max().date()) if len(df) else None,
            "asset_file": str(asset_file),
            "attempted_prefixes": " | ".join(attempted_prefixes),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "price_extract_manifest.csv", index=False)
    selected_meta.to_csv(out_dir / "selected_assets_for_price_extract.csv", index=False)

    if extracted_frames:
        all_df = pd.concat(extracted_frames, ignore_index=True, sort=False)
        all_df.to_parquet(out_dir / "price_history_extracted.parquet", index=False)
        all_df.to_csv(out_dir / "price_history_extracted.csv", index=False)

    summary = {
        "prices_s3_prefix": args.prices_s3_prefix,
        "assets_requested": len(assets),
        "assets_extracted": int((manifest["status"] == "ok").sum()) if not manifest.empty else 0,
        "out_dir": str(out_dir),
        "windows_used": bool(args.bad_cells_csv and args.days_around_bad_dates > 0),
    }
    (out_dir / "price_extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
