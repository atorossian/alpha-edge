"""Diagnose why reconstructed HYPE still fails publication gates.

This script does not repair or forward-fill any price and does not write active data.
Outputs actionable date-level observations directly to immutable S3 report.
"""
from __future__ import annotations
import argparse
import io
import json
import uuid
from datetime import datetime, timezone
import boto3
import pandas as pd

ROOT = "market/repairs/aea87/rebuilds/v1/runs"


def diagnose(raw, prices, returns):
    required = ("date", "Adj Close", "Close")
    if not set(required).issubset(raw):
        raise ValueError("Raw Yahoo candle is missing required columns")
    r = raw.copy()
    r["date"] = pd.to_datetime(r["date"], utc=True).dt.tz_localize(None).dt.normalize()
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True).dt.tz_localize(None).dt.normalize()
    merged = p[["date", "close_raw_usd", "close_adjusted_usd"]].merge(
        r[["date", "Adj Close", "Close"]], how="left", on="date", validate="one_to_one")
    invalid = merged["close_adjusted_usd"].isna() | merged["close_adjusted_usd"].le(0)
    rows = merged.loc[invalid]
    missing = [{"date": str(v["date"].date()),
                "raw_yahoo_close": None if pd.isna(v["Close"]) else float(v["Close"]),
                "raw_yahoo_adjusted_close": None if pd.isna(v["Adj Close"]) else float(v["Adj Close"]),
                "candidate_close": None if pd.isna(v["close_raw_usd"]) else float(v["close_raw_usd"])}
               for _, v in rows.iterrows()]
    return {"candidate_rows": len(prices), "missing_adjusted_rows": len(missing),
            "observations": missing, "policy": "SOURCE_INVESTIGATION_REQUIRED; DO_NOT_IMPUTE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run-id", required=True)
    ap.add_argument("--asset-id", default="CRYPTO:HYPE-USD")
    ap.add_argument("--bucket", default="alpha-edge-algo")
    ap.add_argument("--region", default="eu-west-1")
    a = ap.parse_args()
    s3 = boto3.client("s3", region_name=a.region)
    safe = "".join(c if c.isalnum() else "_" for c in a.asset_id)
    folder = f"{ROOT}/{a.source_run_id}/assets/{safe}"
    def load(name):
        return pd.read_parquet(io.BytesIO(s3.get_object(Bucket=a.bucket, Key=f"{folder}/{name}.parquet")["Body"].read()))
    findings = diagnose(load("raw_yahoo"), load("candidate_ohlcv_usd"), load("candidate_returns_usd"))
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    key = f"market/repairs/aea87/diagnostics/v1/runs/{run}/adjusted_price_gaps.json"
    payload = {"asset_id": a.asset_id, "source_run_id": a.source_run_id, **findings}
    s3.put_object(Bucket=a.bucket, Key=key, Body=json.dumps(payload, indent=2).encode(), ContentType="application/json", IfNoneMatch="*")
    print(json.dumps({"s3_report": f"s3://{a.bucket}/{key}", **findings}, indent=2))


if __name__ == "__main__":
    main()
