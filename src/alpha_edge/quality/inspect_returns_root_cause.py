#!/usr/bin/env python3
"""Inspect return anomalies against source adjusted-close USD prices.

This script is for triage, not production promotion. It joins bad return cells to
source price_t, price_t-1 and price_t+1 to determine whether the bad return is
caused by source prices, transformation, calendar alignment, or convention mismatch.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Tuple

import boto3
import numpy as np
import pandas as pd


def is_s3_uri(path: str) -> bool:
    return str(path).lower().startswith("s3://")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def download_s3(uri: str, *, region: Optional[str], tmp_dir: Optional[str]) -> str:
    bucket, key = parse_s3_uri(uri)
    suffix = Path(key).suffix or ".data"
    root = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    local = root / f"root_cause_{Path(key).stem}{suffix}"
    boto3.client("s3", region_name=region).download_file(bucket, key, str(local)) if region else boto3.client("s3").download_file(bucket, key, str(local))
    return str(local)


def read_table(path: str, *, region: Optional[str], tmp_dir: Optional[str]) -> pd.DataFrame:
    source = download_s3(path, region=region, tmp_dir=tmp_dir) if is_s3_uri(path) else path
    p = Path(source)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    date_col = next((lower[c] for c in ["date", "as_of", "datetime", "timestamp"] if c in lower), None)
    if not date_col or "asset_id" not in lower or "adj_close_usd" not in lower:
        raise ValueError("Prices require date/as_of, asset_id, adj_close_usd columns")
    rename = {date_col: "date", lower["asset_id"]: "asset_id", lower["adj_close_usd"]: "adj_close_usd"}
    for opt in ["ticker", "yahoo_ticker", "close", "adj_close", "fx_rate", "source_timestamp"]:
        if opt in lower:
            rename[lower[opt]] = opt
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["adj_close_usd"] = pd.to_numeric(df["adj_close_usd"], errors="coerce")
    return df.dropna(subset=["date", "asset_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Join bad returns to source price rows for root-cause triage.")
    ap.add_argument("--prices", required=True, help="Local/S3 long adjusted-close USD price table")
    ap.add_argument("--bad-cells", required=True, help="returns_quality_cells.csv")
    ap.add_argument("--out", default="returns_root_cause_join.csv")
    ap.add_argument("--top-n", type=int, default=5000)
    ap.add_argument("--aws-region", default=None)
    ap.add_argument("--tmp-dir", default=None)
    args = ap.parse_args()

    prices = normalize_prices(read_table(args.prices, region=args.aws_region, tmp_dir=args.tmp_dir))
    cells = pd.read_csv(args.bad_cells)
    cells["date"] = pd.to_datetime(cells["date"], errors="coerce").dt.normalize()
    cells = cells.sort_values("abs_return", ascending=False).head(args.top_n)

    base_cols = [c for c in ["date", "asset_id", "ticker", "yahoo_ticker", "close", "adj_close", "adj_close_usd", "fx_rate", "source_timestamp"] if c in prices.columns]
    px = prices[base_cols].sort_values(["asset_id", "date"]).copy()
    px["prev_date"] = px.groupby("asset_id")["date"].shift(1)
    px["next_date"] = px.groupby("asset_id")["date"].shift(-1)
    px["prev_adj_close_usd"] = px.groupby("asset_id")["adj_close_usd"].shift(1)
    px["next_adj_close_usd"] = px.groupby("asset_id")["adj_close_usd"].shift(-1)
    px["calc_log_return"] = np.log(px["adj_close_usd"] / px["prev_adj_close_usd"])
    px["calc_arith_return"] = px["adj_close_usd"] / px["prev_adj_close_usd"] - 1.0
    px["next_log_return"] = np.log(px["next_adj_close_usd"] / px["adj_close_usd"])

    joined = cells.merge(px, on=["date", "asset_id"], how="left")
    joined["matches_log_return"] = np.isclose(joined["return"], joined["calc_log_return"], atol=1e-8, rtol=1e-6)
    joined["matches_arith_return"] = np.isclose(joined["return"], joined["calc_arith_return"], atol=1e-8, rtol=1e-6)
    joined["reversal_next_day"] = (joined["return"] * joined["next_log_return"] < 0) & (joined["next_log_return"].abs() > 0.5)
    joined["likely_root_cause"] = np.select(
        [
            joined["adj_close_usd"].isna(),
            joined["matches_log_return"] & joined["reversal_next_day"],
            joined["matches_log_return"],
            joined["matches_arith_return"],
        ],
        [
            "MISSING_SOURCE_PRICE_FOR_BAD_RETURN_CELL",
            "SOURCE_PRICE_SPIKE_THEN_REVERSAL",
            "SOURCE_PRICE_JUMP_OR_BAD_ADJUSTED_CLOSE_USD",
            "RETURN_CONVENTION_IS_ARITHMETIC",
        ],
        default="RETURN_CACHE_DOES_NOT_MATCH_SOURCE_PRICE_CALCULATION",
    )

    joined.to_csv(args.out, index=False)
    summary = joined["likely_root_cause"].value_counts(dropna=False).to_dict()
    with open(Path(args.out).with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
