"""AEA-87 staged campaign validation (read-only by default).

This tool never modifies production market data or manifests. It verifies the
content-addressed staged Parquet files and reports outstanding quarantine.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd


def validate_campaign(s3, bucket: str, run_id: str, *, prefix: str = "market",
                      asset_ids: set[str] | None = None) -> dict:
    key = f"{prefix}/repairs/aea87/staging/{run_id}/prepared_campaign.json"
    campaign = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    if campaign.get("run_id") != run_id:
        raise ValueError("Campaign manifest run_id mismatch")
    items = campaign.get("prepared", [])
    report = campaign.get("report", [])
    if len({x["asset_id"] for x in items}) != len(items):
        raise ValueError("Duplicate prepared asset IDs")
    if len({x["asset_id"] for x in report}) != len(report):
        raise ValueError("Duplicate report asset IDs")
    if asset_ids is not None:
        missing = asset_ids - {x["asset_id"] for x in items}
        if missing:
            raise ValueError(f"Requested assets not prepared in this campaign: {sorted(missing)}")
        items = [x for x in items if x["asset_id"] in asset_ids]
    report_by_id = {x["asset_id"]: x for x in report}
    results = []
    for item in items:
        aid = item["asset_id"]
        errors = []
        table_frames = {"ohlcv_usd": [], "returns_usd": []}
        seen = set()
        for rec in item.get("records", []):
            table = rec.get("table")
            year = rec.get("year")
            if table not in table_frames:
                errors.append(f"Unexpected table: {table}")
                continue
            ident = (table, year)
            if ident in seen:
                errors.append(f"Duplicate table/year record: {ident}")
                continue
            seen.add(ident)
            stage_key = rec.get("staged_part", "")
            expected_prefix = f"{prefix}/repairs/aea87/staging/{run_id}/asset_id={aid}/{table}/year={year}/"
            if not stage_key.startswith(expected_prefix):
                errors.append(f"Foreign staged key: {stage_key}")
                continue
            try:
                data = s3.get_object(Bucket=bucket, Key=stage_key)["Body"].read()
                if hashlib.sha256(data).hexdigest() != rec.get("sha256"):
                    raise ValueError("SHA256 mismatch")
                frame = pd.read_parquet(io.BytesIO(data))
                if len(frame) != rec.get("rows"):
                    raise ValueError("Row count mismatch")
                if not {"asset_id", "date"}.issubset(frame.columns):
                    raise ValueError("Missing asset_id/date")
                dates = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.normalize()
                if dates.isna().any() or (frame["asset_id"].astype(str) != aid).any():
                    raise ValueError("Invalid dates or asset identity")
                if not (dates.dt.year == int(year)).all():
                    raise ValueError("Year partition mismatch")
                if dates.duplicated().any():
                    raise ValueError("Duplicate dates in partition")
                if sorted(dates.dt.strftime("%Y-%m-%d").tolist()) != rec.get("dates"):
                    raise ValueError("Dates differ from manifest")
                table_frames[table].append(frame)
            except Exception as exc:
                errors.append(f"{table}/{year}: {type(exc).__name__}: {exc}")
        combined = {}
        for table, chunks in table_frames.items():
            if not chunks:
                errors.append(f"Missing {table} partitions")
                continue
            frame = pd.concat(chunks, ignore_index=True)
            frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
            if frame["date"].duplicated().any():
                errors.append(f"{table}: duplicate dates across years")
            combined[table] = frame.sort_values("date")
        if len(combined) == 2:
            px, ret = combined["ohlcv_usd"], combined["returns_usd"]
            vals = pd.to_numeric(px.get("close_adjusted_usd"), errors="coerce")
            if vals.isna().any() or not np.isfinite(vals).all() or (vals <= 0).any():
                errors.append("Invalid adjusted USD close")
            else:
                theoretical = pd.Series(vals.to_numpy(), index=px["date"]).sort_index()
                expected = np.log(theoretical / theoretical.shift(1)).dropna()
                actual = pd.to_numeric(ret["ret_log_close_adjusted_usd"], errors="coerce")
                series = pd.Series(actual.to_numpy(), index=ret["date"]).sort_index()
                if not expected.index.equals(series.index) or not np.allclose(
                    expected.to_numpy(), series.to_numpy(), rtol=1e-9, atol=1e-11, equal_nan=False
                ):
                    errors.append("Staged returns inconsistent with staged adjusted USD prices")
        flagged = int(report_by_id.get(aid, {}).get("flagged_retained", item.get("flagged_retained", 0)))
        if flagged != int(item.get("flagged_retained", 0)):
            errors.append("Quarantine count differs from campaign manifest")
        needs_reset = aid == "CRYPTO:HYPE-USD" and int(report_by_id.get(aid, {}).get("retained_rows", 0)) > 0
        results.append({
            "asset_id": aid,
            "status": "INVALID" if errors else ("RESET_REQUIRED" if needs_reset else ("VALID_WITH_QUARANTINE" if flagged else "VALID")),
            "flagged_retained": flagged,
            "identity_reset_required": needs_reset,
            "records_checked": len(seen),
            "errors": errors,
        })
    failures = [r for r in report if r.get("status") != "PREPARED"]
    return {
        "run_id": run_id,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "assets_checked": len(results),
        "records_checked": sum(x["records_checked"] for x in results),
        "assets_valid": sum(x["status"] == "VALID" for x in results),
        "assets_with_quarantine": sum(x["status"] == "VALID_WITH_QUARANTINE" for x in results),
        "assets_invalid": sum(x["status"] == "INVALID" for x in results),
        "assets_reset_required": sum(x["status"] == "RESET_REQUIRED" for x in results),
        "flagged_retained": sum(x["flagged_retained"] for x in results),
        "campaign_failures": failures,
        "assets": results,
        "production_promotion_allowed": False,
    }


def main():
    p = argparse.ArgumentParser(description="Read-only AEA-87 S3 staged campaign validator")
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--market-prefix", default="market")
    p.add_argument("--repair-run-id", required=True)
    p.add_argument("--asset-ids", help="Comma-separated canonical asset IDs; omit to validate full campaign")
    p.add_argument("--publish-report", action="store_true",
                   help="Write validation report ONLY to this campaign's staging prefix")
    args = p.parse_args()
    ids = None if not args.asset_ids else set(x.strip() for x in args.asset_ids.split(","))
    if ids is not None and (not ids or "" in ids):
        p.error("--asset-ids must contain nonempty asset IDs")
    s3 = boto3.client("s3", region_name=args.region)
    result = validate_campaign(s3, args.bucket, args.repair_run_id,
                               prefix=args.market_prefix.strip("/"), asset_ids=ids)
    summary = {k: v for k, v in result.items() if k != "assets"}
    print(json.dumps(summary, indent=2))
    for row in result["assets"]:
        if row["status"] != "VALID":
            print(json.dumps(row))
    if args.publish_report:
        suffix = "_selected" if ids is not None else "_full"
        key = (f"{args.market_prefix.strip('/')}/repairs/aea87/staging/"
               f"{args.repair_run_id}/validation_report{suffix}.json")
        data = json.dumps(result, indent=2).encode()
        try:
            s3.head_object(Bucket=args.bucket, Key=key)
        except s3.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            raise FileExistsError(f"Validation report already exists: s3://{args.bucket}/{key}")
        s3.put_object(Bucket=args.bucket, Key=key, Body=data, ContentType="application/json")
        print(f"VALIDATION_REPORT s3://{args.bucket}/{key}")
    if result["assets_invalid"] or result["assets_reset_required"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
