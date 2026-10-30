"""Diagnose invalid USD adjusted-close cells in AEA-87 staged OHLCV.

Read-only: fetches only staged OHLCV for chosen assets, no production writes.
"""
import argparse
import io
import json

import boto3
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repair-run-id", required=True)
    p.add_argument("--asset-ids", required=True, help="Comma-separated canonical asset IDs")
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    args = p.parse_args()
    ids = [s.strip() for s in args.asset_ids.split(",") if s.strip()]
    s3 = boto3.client("s3", region_name=args.region)
    campaign_key = f"market/repairs/aea87/staging/{args.repair_run_id}/prepared_campaign.json"
    campaign = json.loads(s3.get_object(Bucket=args.bucket, Key=campaign_key)["Body"].read())
    assets = {x["asset_id"]: x for x in campaign["prepared"]}
    for aid in ids:
        if aid not in assets:
            raise ValueError(f"Asset not prepared in this run: {aid}")
        samples = []
        total = 0
        invalid = 0
        by_reason = {}
        for rec in assets[aid]["records"]:
            if rec["table"] != "ohlcv_usd":
                continue
            raw = s3.get_object(Bucket=args.bucket, Key=rec["staged_part"])["Body"].read()
            d = pd.read_parquet(io.BytesIO(raw))
            v = pd.to_numeric(d["close_adjusted_usd"], errors="coerce")
            bad = v.isna() | ~np.isfinite(v) | (v <= 0)
            total += len(d)
            invalid += int(bad.sum())
            if bad.any():
                cols = [c for c in ("date", "ticker", "yahoo_ticker", "currency", "adj_close", "fx_to_usd", "close_adjusted_usd") if c in d.columns]
                for row in d.loc[bad, cols].to_dict("records"):
                    val = row.get("close_adjusted_usd")
                    reason = "MISSING_OR_NONNUMERIC" if pd.isna(val) else ("NONFINITE" if not np.isfinite(float(val)) else "NONPOSITIVE")
                    by_reason[reason] = by_reason.get(reason, 0) + 1
                    if len(samples) < 30:
                        row["date"] = str(row.get("date"))
                        row["failure_reason"] = reason
                        samples.append({k: (None if pd.isna(v) else v) for k, v in row.items()})
        print(json.dumps({"asset_id": aid, "rows_checked": total, "invalid_rows": invalid,
                          "reasons": by_reason, "sample": samples}, indent=2, default=str))


if __name__ == "__main__":
    main()
