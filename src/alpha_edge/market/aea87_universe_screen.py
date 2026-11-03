"""AEA-87 conservative S3-native whole-universe numerical integrity screen.

This screening is NOT source-identity validation. Flags do not authorize rebuild
or automatic publication. Writes a reproducible classification report to S3.
"""
from __future__ import annotations
import argparse
import io
import json
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
from alpha_edge import paths
from alpha_edge.core.market_store import MarketStore


def screen_one(aid, prices, returns, *, jump_threshold=.50):
    if prices is None or prices.empty:
        return {"asset_id": aid, "status": "INSUFFICIENT_DATA",
                "issues": {"NO_ACTIVE_PRICES": 1}}
    p = prices.copy()
    if "date" not in p or "close_adjusted_usd" not in p:
        return {"asset_id": aid, "status": "INSUFFICIENT_DATA",
                "issues": {"MISSING_CANONICAL_COLUMNS": 1}}
    p["date"] = pd.to_datetime(p["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    bad_dates = int(p["date"].isna().sum())
    duplicates = int(p["date"].duplicated().sum())
    p = p.dropna(subset=["date"]).sort_values("date")
    close = pd.to_numeric(p["close_adjusted_usd"], errors="coerce")
    invalid = int((~np.isfinite(close) | close.le(0)).sum())
    # Threshold is for inspection only; speculative and corporate-action moves can be real.
    jumps = int(close.pct_change(fill_method=None).abs().gt(jump_threshold).sum())
    issues = {}
    if bad_dates: issues["INVALID_DATES"] = bad_dates
    if duplicates: issues["DUPLICATE_DATES"] = duplicates
    if invalid: issues["INVALID_OR_MISSING_ADJUSTED_PRICE"] = invalid
    if jumps: issues["LARGE_MOVE_REVIEW"] = jumps
    if returns is None or returns.empty:
        issues["MISSING_RETURNS"] = 1
    elif {"date", "ret_close_adjusted_usd"}.issubset(returns):
        r = returns.copy()
        r["date"] = pd.to_datetime(r["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        if duplicates:
            issues["RETURN_RECONCILIATION_BLOCKED_BY_DUPLICATE_PRICES"] = duplicates
        elif r["date"].duplicated().any():
            issues["DUPLICATE_RETURN_DATES"] = int(r["date"].duplicated().sum())
        else:
            expected = p[["date"]].copy()
            expected["expected"] = close.to_numpy() / close.shift().to_numpy() - 1
            joined = expected.merge(r[["date", "ret_close_adjusted_usd"]],
                                    on="date", how="inner", validate="one_to_one")
            stored = pd.to_numeric(joined["ret_close_adjusted_usd"], errors="coerce")
            # Compare only adjacent valid observed prices, never missing adjusted
            # prices, or a return without a computable price-side denominator.
            eligible = np.isfinite(joined["expected"]) & np.isfinite(stored)
            mismatch = eligible & (stored - joined["expected"]).abs().gt(1e-5)
            issues["RETURN_PRICE_MISMATCH"] = int(mismatch.sum()) if mismatch.any() else 0
            if not eligible.any():
                issues["NO_COMPARABLE_PRICE_RETURN_PAIRS"] = 1
            issues = {k: v for k, v in issues.items() if v}
    else:
        issues["MISSING_CANONICAL_RETURNS_COLUMN"] = 1
    serious = any(k not in {"LARGE_MOVE_REVIEW"} for k in issues)
    status = "REBUILD_OR_DATA_REPAIR_REVIEW" if serious else (
        "INVESTIGATION_REQUIRED" if issues else "PASSED_NUMERICAL_SCREEN_ONLY")
    return {"asset_id": aid, "status": status, "issues": issues,
            "price_rows": len(p), "first_date": str(p["date"].min().date()),
            "last_date": str(p["date"].max().date()),
            "identity_verified": False, "publication_approved": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default=str(paths.universe_dir() / "universe.csv"))
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-11-01")
    p.add_argument("--max-assets", type=int)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if not 1 <= a.workers <= 16:
        p.error("workers must be 1..16")
    universe = pd.read_csv(a.universe, dtype=str)
    if "asset_id" not in universe:
        p.error("Universe requires asset_id")
    ids = universe["asset_id"].dropna().astype(str).drop_duplicates().tolist()
    full_universe_count = len(ids)
    if a.max_assets is not None:
        ids = ids[:a.max_assets]
    store = MarketStore(bucket=a.bucket, region=a.region)
    def job(aid):
        try:
            prices = store.read_ohlcv_usd([aid], start=a.start, end=a.end)
            returns = store.read_returns_usd([aid], start=a.start, end=a.end)
            return screen_one(aid, prices, returns)
        except Exception as exc:
            return {"asset_id": aid, "status": "INSUFFICIENT_DATA",
                    "issues": {"READ_ERROR": str(exc)}}
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(job, aid) for aid in ids]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: x["asset_id"])
    counts = dict(Counter(x["status"] for x in results))
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    root = f"market/repairs/aea87/universe_screen/v1/runs/{run}"
    s3 = boto3.client("s3", region_name=a.region)
    report = {"run_id": run, "scanned": len(results),
              "universe_count": full_universe_count,
              "complete_universe": len(results) == full_universe_count,
              "classification_counts": counts,
              "screen_policy": "NUMERICAL_ONLY; IDENTITY_NOT_VERIFIED; NO_AUTO_PUBLICATION",
              "active_data_modified": False, "assets": results}
    blob = json.dumps(report, indent=2, default=str).encode()
    s3.put_object(Bucket=a.bucket, Key=root+"/report.json", Body=blob,
                  ContentType="application/json", IfNoneMatch="*")
    if report["complete_universe"]:
        s3.put_object(Bucket=a.bucket, Key="market/repairs/aea87/universe_screen/v1/latest.json",
                      Body=json.dumps({"run_id":run, "report_key":root+"/report.json",
                                       "universe_count":full_universe_count}).encode(),
                      ContentType="application/json")
    print(json.dumps({"report": f"s3://{a.bucket}/{root}/report.json", "scanned": len(results),
                      "classification_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
