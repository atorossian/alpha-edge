"""Pilot runner for market-data lake v2 (non-destructive, stage snapshots).

Input tables are raw provider observations, intentionally NOT USD-normalized.
The command does not call Yahoo or mutate legacy market/ohlcv_usd/v1.

Usage:
  python -m alpha_edge.market.run_data_lake_v2 \
    --bars ./bars.parquet --fx ./fx.parquet --output ./lake-v2
  python -m alpha_edge.market.run_data_lake_v2 \
    --bars ./bars.parquet --fx ./fx.parquet --bucket BUCKET \
    --market-root dev/market --write-s3
"""
from __future__ import annotations

import argparse
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd

from alpha_edge.market.data_lake_v2 import (
    calculate_returns, quality_summary, standardize_bars, validate_raw_bars,
)


def _read(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Expected CSV or Parquet input")


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    df.to_parquet(out, index=False)
    return out.getvalue()


def build(args: argparse.Namespace) -> dict:
    run_id = args.run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    if not all(c.isalnum() or c in "-_" for c in run_id):
        raise ValueError("Unsafe run-id")
    if args.write_s3 and (not args.bucket or not args.market_root):
        raise ValueError("S3 writes require --bucket and --market-root")
    if args.write_s3 and args.market_root.strip("/") == "market":
        raise ValueError("Refusing legacy production market root; use an isolated prefix")
    bars = _read(args.bars)
    fx = _read(args.fx)
    bronze = validate_raw_bars(bars)
    silver = standardize_bars(bronze, fx)

    # Gold retains all original rows for lineage. Duplicated asset/date keys
    # are a hard stop until manually resolved; no silent keep='last'.
    gold = calculate_returns(silver)
    report = {
        "run_id": run_id,
        "schema_version": "v2-pilot-1",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {"bars": args.bars, "fx": args.fx},
        "raw": quality_summary(bronze),
        "standardized": quality_summary(silver),
        "gold_return_accepted": int(gold["return_quality_status"].eq("accepted").sum()),
        "gold_return_review": int(gold["return_quality_status"].eq("review").sum()),
        "gold_return_unavailable": int(gold["return_quality_status"].eq("unavailable").sum()),
        "publication_state": "pilot_only",
        "legacy_production_unchanged": True,
    }
    layers = {
        "bronze/raw_ohlcv.parquet": bronze,
        "bronze/raw_fx.parquet": fx,
        "silver/ohlcv_usd.parquet": silver,
        "gold/returns.parquet": gold,
    }
    output = Path(args.output) / run_id
    output.mkdir(parents=True, exist_ok=False)
    for name, df in layers.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_parquet_bytes(df))
    (output / "manifest.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if args.write_s3:
        s3 = boto3.client("s3", region_name=args.region)
        prefix = f"{args.market_root.strip('/')}/lake_v2/pilot/runs/{run_id}"
        # All data objects are versioned by run; no stage can overwrite another run.
        # If this run ID already exists, fail before writing anything.
        manifest_key = f"{prefix}/manifest.json"
        try:
            s3.head_object(Bucket=args.bucket, Key=manifest_key)
        except s3.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            raise FileExistsError(f"Run already exists: s3://{args.bucket}/{prefix}")
        for name, df in layers.items():
            s3.put_object(
                Bucket=args.bucket, Key=f"{prefix}/{name}",
                Body=_parquet_bytes(df), ContentType="application/octet-stream",
            )
        # Manifest LAST. Readers should consider only complete manifest-bearing runs.
        s3.put_object(
            Bucket=args.bucket, Key=manifest_key,
            Body=json.dumps(report, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        report["s3_prefix"] = f"s3://{args.bucket}/{prefix}"
    report["local_output"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", required=True, help="Unconverted raw OHLCV CSV/Parquet")
    parser.add_argument("--fx", required=True, help="Raw FX rates, USD per currency unit")
    parser.add_argument("--output", default="./outputs/market_data_lake_v2")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--write-s3", action="store_true")
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--market-root", default=None)
    parser.add_argument("--region", default="eu-west-1")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
