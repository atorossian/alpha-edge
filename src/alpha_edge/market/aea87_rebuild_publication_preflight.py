"""Read-only preflight for AEA87 reconstructed-history publication.

Never writes active market objects. Returns blockers in a versioned S3 report.
"""
from __future__ import annotations
import argparse
import io
import json
import uuid
from datetime import datetime, timezone
import boto3
import pandas as pd
from alpha_edge.core.market_store import MarketStore
from alpha_edge.market.aea87_publication_validation import validate_candidate, validate_old_only_against_calendar

ASSETS = ("CRYPTO:HYPE-USD", "EQH028b839246b50817")
ROOT = "market/repairs/aea87/rebuilds/v1/runs"


def check_asset(s3, bucket, store, run_id, aid, date_end):
    folder = "".join(c if c.isalnum() else "_" for c in aid)
    base = f"{ROOT}/{run_id}/assets/{folder}"
    prices = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=bucket, Key=base+"/candidate_ohlcv_usd.parquet")["Body"].read()))
    returns = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=bucket, Key=base+"/candidate_returns_usd.parquet")["Body"].read()))
    review = json.loads(s3.get_object(Bucket=bucket, Key=base+"/coverage_review.json")["Body"].read())
    issues = [x for x in review.get("reasons", []) if x != "OLD_ONLY_DATES_REQUIRE_EXCHANGE_CALENDAR_VALIDATION"]
    inception = "2024-11-29" if aid == "CRYPTO:HYPE-USD" else None
    issues.extend(validate_candidate(prices, returns, inception))
    if prices.empty or prices["date"].duplicated().any() or returns["date"].duplicated().any():
        issues.append("EMPTY_OR_DUPLICATE_CANDIDATE")
    if set(prices["date"]) != set(returns["date"]):
        issues.append("PRICE_RETURN_DATE_MISMATCH")
    years = sorted(set(pd.to_datetime(prices["date"]).dt.year))
    # Include the full previously active historic period: years disappearing
    # from the rebuild must receive explicit retired manifests.
    old = store.read_ohlcv_usd([aid], start="2010-01-01", end=date_end)
    old_years = set(pd.to_datetime(old["date"]).dt.year) if not old.empty else set()
    calendar_report = None
    if aid == "EQH028b839246b50817":
        calendar_report = validate_old_only_against_calendar(old["date"], prices["date"], "NYSE")
        if calendar_report["status"] != "PASSED":
            issues.append("OLD_ONLY_DATES_INCLUDE_VALID_NYSE_SESSIONS")
    elif review.get("old_only_dates") and aid != "CRYPTO:HYPE-USD":
        issues.append("OLD_ONLY_DATES_REQUIRE_EXCHANGE_CALENDAR_VALIDATION")
    if aid == "CRYPTO:HYPE-USD":
        bad_old_dates = pd.to_datetime(review.get("old_only_dates", []), utc=True)
        if len(bad_old_dates) and (bad_old_dates >= pd.Timestamp("2024-11-29", tz="UTC")).any():
            issues.append("OLD_ONLY_DATES_AFTER_VALIDATED_INCEPTION")
    operations = []
    for year in sorted(old_years | set(years)):
        for table in ("ohlcv_usd", "returns_usd"):
            key = store._manifest_key(table=table, asset_id=aid, year=year)
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                manifest = json.loads(obj["Body"].read())
                etag = obj.get("ETag")
                present = True
            except s3.exceptions.NoSuchKey:
                manifest = None
                etag = None
                present = False
            if not present:
                issues.append("MISSING_ACTIVE_MANIFEST_REQUIRES_LEGACY_BACKUP")
            operations.append({"year":int(year), "table":table, "key":key,
                               "active_manifest_exists":present, "active_etag":etag,
                               "candidate_rows":int((pd.to_datetime(prices["date"] if table=="ohlcv_usd" else returns["date"]).dt.year==year).sum()),
                               "retire":bool(year not in years)})
    return {"asset_id":aid,"candidate_rows":len(prices),"return_rows":len(returns),
            "calendar_validation":calendar_report,
            "blockers":sorted(set(issues)),"operations":operations,
            "status":"NOT_APPROVED" if issues else "PREFLIGHT_REQUIRES_MANUAL_APPROVAL"}


def main():
    p=argparse.ArgumentParser(description="Read-only AEA87 rebuild publication preflight")
    p.add_argument("--run-id",required=True)
    p.add_argument("--asset-ids",default=",".join(ASSETS))
    p.add_argument("--bucket",default="alpha-edge-algo")
    p.add_argument("--region",default="eu-west-1")
    p.add_argument("--end",default="2026-11-01")
    a=p.parse_args()
    s3=boto3.client("s3",region_name=a.region)
    store=MarketStore(bucket=a.bucket,region=a.region)
    results={}
    for aid in a.asset_ids.split(","):
        results[aid]=check_asset(s3,a.bucket,store,a.run_id,aid,a.end)
        print(f"[preflight] {aid}: {results[aid]['status']}",flush=True)
    run=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    key=f"market/repairs/aea87/publication_preflight/v1/runs/{run}/report.json"
    payload={"source_run_id":a.run_id,"status":"READ_ONLY_REVIEW","assets":results}
    s3.put_object(Bucket=a.bucket,Key=key,Body=json.dumps(payload,indent=2).encode(),ContentType="application/json",IfNoneMatch="*")
    print("Report: s3://"+a.bucket+"/"+key)


if __name__=="__main__":
    main()
