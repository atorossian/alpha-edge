# validate_returns_quality.py
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd


@dataclass
class ReturnsQualityConfig:
    max_abs_return_warn: float = 0.20
    max_abs_return_block: float = 0.50
    max_daily_return_block: float = 1.00
    min_daily_return_block: float = -1.00
    max_bad_cells_per_date_warn: int = 10
    max_bad_cells_per_date_block: int = 50
    max_bad_cells_per_asset_warn: int = 3
    max_bad_cells_per_asset_block: int = 10
    min_obs_per_asset: int = 252


@dataclass
class ReturnsQualitySummary:
    rows: int
    assets: int
    duplicate_dates: int
    bad_row_lengths: int
    total_observations: int
    missing_observations: int
    abs_gt_warn: int
    abs_gt_block: int
    gt_100pct: int
    le_minus_100pct: int
    dates_with_block_clusters: int
    assets_with_block_cells: int
    assets_with_impossible_returns: int
    critical: bool


def _is_s3_uri(path: str) -> bool:
    return str(path).lower().startswith("s3://")


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {uri!r}. Expected s3://bucket/key")
    return parsed.netloc, parsed.path.lstrip("/")


def _s3_client(region: Optional[str] = None):
    kwargs = {}
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def _download_s3_uri(uri: str, *, region: Optional[str], tmp_dir: Optional[str]) -> str:
    bucket, key = _parse_s3_uri(uri)
    suffix = Path(key).suffix or ".data"
    tmp_parent = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
    tmp_parent.mkdir(parents=True, exist_ok=True)
    local_path = tmp_parent / f"returns_quality_input_{bucket}_{Path(key).stem}{suffix}"

    s3 = _s3_client(region)
    print(f"[s3] downloading s3://{bucket}/{key}")
    print(f"[s3] local copy: {local_path}")
    s3.download_file(bucket, key, str(local_path))
    return str(local_path)


def _upload_dir_to_s3(local_dir: str, s3_uri: str, *, region: Optional[str]) -> None:
    bucket, prefix = _parse_s3_uri(s3_uri)
    prefix = prefix.rstrip("/")
    s3 = _s3_client(region)
    root = Path(local_dir)
    if not root.exists():
        raise FileNotFoundError(local_dir)

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        print(f"[s3] uploading {p} -> s3://{bucket}/{key}")
        s3.upload_file(str(p), bucket, key)


def _read_returns(path: str, *, region: Optional[str] = None, tmp_dir: Optional[str] = None) -> pd.DataFrame:
    source_path = path
    if _is_s3_uri(path):
        source_path = _download_s3_uri(path, region=region, tmp_dir=tmp_dir)

    p = Path(source_path)
    if not p.exists():
        raise FileNotFoundError(source_path)

    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # Handle common formats: either explicit date column or date index exported as first unnamed column.
    first_col = str(df.columns[0])
    if first_col.lower() in {"date", "as_of", "datetime", "timestamp"} or first_col.startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            bad = int(df["date"].isna().sum())
            raise ValueError(f"Found {bad} rows with invalid date values.")
        df = df.set_index("date")
    else:
        # If already indexed in parquet, normalize index.
        df.index = pd.to_datetime(df.index, errors="coerce")
        if pd.isna(df.index).any():
            raise ValueError("Could not infer date index. Provide a file with date/as_of column or datetime index.")

    df = df.sort_index()
    df.columns = [str(c).strip() for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def _prefix(asset_id: str) -> str:
    if ":" in asset_id:
        return asset_id.split(":", 1)[0]
    if asset_id.startswith("EQH"):
        return "EQH"
    return "OTHER"


def validate_returns(
    returns: pd.DataFrame,
    *,
    cfg: ReturnsQualityConfig,
    out_dir: str,
    write_sanitized: bool = False,
    sanitized_path: Optional[str] = None,
) -> ReturnsQualitySummary:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, assets = returns.shape
    duplicate_dates = int(returns.index.duplicated().sum())
    bad_row_lengths = 0  # pandas would already normalize this while reading

    values = returns.to_numpy(dtype="float64", copy=True)
    finite = np.isfinite(values)
    missing = int((~finite).sum())
    total_obs = int(finite.sum())

    abs_values = np.abs(values)
    warn_mask = finite & (abs_values > cfg.max_abs_return_warn)
    block_mask = finite & (abs_values > cfg.max_abs_return_block)
    gt_100_mask = finite & (values > cfg.max_daily_return_block)
    le_minus_100_mask = finite & (values <= cfg.min_daily_return_block)
    impossible_mask = gt_100_mask | le_minus_100_mask

    date_block_counts = pd.Series(block_mask.sum(axis=1), index=returns.index, name="block_cell_count")
    asset_block_counts = pd.Series(block_mask.sum(axis=0), index=returns.columns, name="block_cell_count")
    asset_impossible_counts = pd.Series(impossible_mask.sum(axis=0), index=returns.columns, name="impossible_count")

    dates_with_block_clusters = int((date_block_counts >= cfg.max_bad_cells_per_date_block).sum())
    assets_with_block_cells = int((asset_block_counts > 0).sum())
    assets_with_impossible_returns = int((asset_impossible_counts > 0).sum())

    critical = bool(
        duplicate_dates > 0
        or int(gt_100_mask.sum()) > 0
        or int(le_minus_100_mask.sum()) > 0
        or dates_with_block_clusters > 0
    )

    # Cell-level report for all block-level anomalies.
    r_idx, c_idx = np.where(block_mask | impossible_mask)
    cell_rows = []
    for r, c in zip(r_idx, c_idx):
        v = float(values[r, c])
        reason = []
        if abs(v) > cfg.max_abs_return_block:
            reason.append("abs_gt_block")
        if v > cfg.max_daily_return_block:
            reason.append("gt_100pct")
        if v <= cfg.min_daily_return_block:
            reason.append("le_minus_100pct")
        cell_rows.append(
            {
                "date": returns.index[r].strftime("%Y-%m-%d"),
                "asset_id": returns.columns[c],
                "return": v,
                "abs_return": abs(v),
                "reason": ";".join(reason),
                "asset_prefix": _prefix(returns.columns[c]),
            }
        )
    cells_df = pd.DataFrame(cell_rows).sort_values("abs_return", ascending=False) if cell_rows else pd.DataFrame()
    cells_df.to_csv(out / "returns_quality_cells.csv", index=False)

    # Date-level report.
    max_abs_per_date = np.nanmax(np.where(finite, abs_values, np.nan), axis=1) if rows else np.array([])
    date_report = pd.DataFrame(
        {
            "date": returns.index.strftime("%Y-%m-%d"),
            "obs": finite.sum(axis=1),
            "missing": (~finite).sum(axis=1),
            "abs_gt_warn": warn_mask.sum(axis=1),
            "abs_gt_block": block_mask.sum(axis=1),
            "gt_100pct": gt_100_mask.sum(axis=1),
            "le_minus_100pct": le_minus_100_mask.sum(axis=1),
            "max_abs_return": max_abs_per_date,
        }
    )
    date_report["severity"] = np.where(
        date_report["abs_gt_block"] >= cfg.max_bad_cells_per_date_block,
        "BLOCK",
        np.where(date_report["abs_gt_warn"] >= cfg.max_bad_cells_per_date_warn, "WARN", "OK"),
    )
    date_report.to_csv(out / "returns_quality_dates.csv", index=False)

    # Asset-level report.
    if assets:
        max_abs_by_asset = np.nanmax(np.where(finite, abs_values, np.nan), axis=0)
        max_idx_by_asset = np.nanargmax(np.where(finite, abs_values, -np.inf), axis=0)
    else:
        max_abs_by_asset = np.array([])
        max_idx_by_asset = np.array([])
    asset_report = pd.DataFrame(
        {
            "asset_id": returns.columns,
            "asset_prefix": [_prefix(c) for c in returns.columns],
            "obs": finite.sum(axis=0),
            "missing": (~finite).sum(axis=0),
            "abs_gt_warn": warn_mask.sum(axis=0),
            "abs_gt_block": block_mask.sum(axis=0),
            "gt_100pct": gt_100_mask.sum(axis=0),
            "le_minus_100pct": le_minus_100_mask.sum(axis=0),
            "max_abs_return": max_abs_by_asset,
            "max_abs_date": [returns.index[i].strftime("%Y-%m-%d") for i in max_idx_by_asset],
        }
    )
    asset_report["severity"] = "OK"
    asset_report.loc[asset_report["obs"] < cfg.min_obs_per_asset, "severity"] = "WARN_LOW_HISTORY"
    asset_report.loc[asset_report["abs_gt_warn"] >= cfg.max_bad_cells_per_asset_warn, "severity"] = "WARN"
    asset_report.loc[asset_report["abs_gt_block"] >= cfg.max_bad_cells_per_asset_block, "severity"] = "BLOCK"
    asset_report.loc[(asset_report["gt_100pct"] > 0) | (asset_report["le_minus_100pct"] > 0), "severity"] = "BLOCK_IMPOSSIBLE"
    asset_report.sort_values(["severity", "max_abs_return"], ascending=[True, False]).to_csv(out / "returns_quality_assets.csv", index=False)

    # Quarantine candidates.
    quarantine = asset_report[asset_report["severity"].str.startswith("BLOCK")].copy()
    quarantine.to_csv(out / "returns_quarantine_assets.csv", index=False)

    summary = ReturnsQualitySummary(
        rows=rows,
        assets=assets,
        duplicate_dates=duplicate_dates,
        bad_row_lengths=bad_row_lengths,
        total_observations=total_obs,
        missing_observations=missing,
        abs_gt_warn=int(warn_mask.sum()),
        abs_gt_block=int(block_mask.sum()),
        gt_100pct=int(gt_100_mask.sum()),
        le_minus_100pct=int(le_minus_100_mask.sum()),
        dates_with_block_clusters=dates_with_block_clusters,
        assets_with_block_cells=assets_with_block_cells,
        assets_with_impossible_returns=assets_with_impossible_returns,
        critical=critical,
    )

    with open(out / "returns_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "summary": asdict(summary)}, f, indent=2)

    if write_sanitized:
        sanitized = returns.copy()
        # Conservative: only remove impossible values and block-level anomalies from the cache copy.
        sanitized.values[block_mask | impossible_mask] = np.nan
        target = Path(sanitized_path) if sanitized_path else out / "returns_wide_sanitized.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() in {".parquet", ".pq"}:
            sanitized.to_parquet(target)
        else:
            sanitized.to_csv(target, index_label="date")

    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate wide asset returns before portfolio search/quarantine.")
    ap.add_argument("--returns", required=True, help="Local CSV/parquet path or S3 URI, e.g. s3://bucket/key/returns_wide_min5y.parquet")
    ap.add_argument("--out-dir", default="returns_quality_report")
    ap.add_argument("--fail-on-critical", action="store_true")
    ap.add_argument("--write-sanitized", action="store_true")
    ap.add_argument("--sanitized-path", default=None, help="Local path for sanitized output. S3 output is not supported for sanitized file in this script version.")
    ap.add_argument("--max-abs-warn", type=float, default=0.20)
    ap.add_argument("--max-abs-block", type=float, default=0.50)
    ap.add_argument("--max-date-block-cells", type=int, default=50)
    ap.add_argument("--max-asset-block-cells", type=int, default=10)
    ap.add_argument("--aws-region", default=None, help="AWS region for S3 access. If omitted, boto3 default resolution is used.")
    ap.add_argument("--tmp-dir", default=None, help="Optional local directory for downloaded S3 input file.")
    ap.add_argument("--upload-report-s3", default=None, help="Optional S3 URI prefix where report files should be uploaded, e.g. s3://bucket/engine/v1/quality/returns/latest/")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ReturnsQualityConfig(
        max_abs_return_warn=float(args.max_abs_warn),
        max_abs_return_block=float(args.max_abs_block),
        max_bad_cells_per_date_block=int(args.max_date_block_cells),
        max_bad_cells_per_asset_block=int(args.max_asset_block_cells),
    )
    returns = _read_returns(args.returns, region=args.aws_region, tmp_dir=args.tmp_dir)
    summary = validate_returns(
        returns,
        cfg=cfg,
        out_dir=args.out_dir,
        write_sanitized=bool(args.write_sanitized),
        sanitized_path=args.sanitized_path,
    )

    if args.upload_report_s3:
        _upload_dir_to_s3(args.out_dir, args.upload_report_s3, region=args.aws_region)

    print("\n=== RETURNS QUALITY SUMMARY ===")
    for k, v in asdict(summary).items():
        print(f"{k}: {v}")
    print(f"\nReports written to: {args.out_dir}")
    if args.upload_report_s3:
        print(f"Reports uploaded to: {args.upload_report_s3}")

    if summary.critical:
        print("\n[CRITICAL] Returns file failed quality checks.")
        if args.fail_on_critical:
            raise SystemExit(2)
    else:
        print("\n[OK] Returns file passed critical checks.")


if __name__ == "__main__":
    main()
