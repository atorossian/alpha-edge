"""AEA-87: rebuild missing derived USD prices from an immutable staged campaign.

Reads existing staged Parquet; NEVER downloads Yahoo or modifies active manifests.
Writes corrected assets into a NEW staged campaign. Assets without repairable
missing values are not copied and stay in the original campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_edge.market.historical_repair import _dates, _dump_json, _parquet, _prepare, _returns
from alpha_edge.market.ingest_market_data import _make_market_store


def repair_derived_prices(store, source_run_id: str, new_run_id: str, *,
                          asset_ids: set[str] | None = None):
    prefix = f"{store.base_prefix}/repairs/aea87/staging/"
    source = json.loads(store._get_bytes(prefix + source_run_id + "/prepared_campaign.json"))
    if source.get("run_id") != source_run_id:
        raise ValueError("Source campaign ID mismatch")
    destination_key = prefix + new_run_id + "/prepared_campaign.json"
    if store._key_exists(destination_key):
        raise FileExistsError(f"Destination campaign already exists: {destination_key}")
    prepared, report = [], []
    available = {x["asset_id"] for x in source["prepared"]}
    if asset_ids is not None and not asset_ids.issubset(available):
        raise ValueError(f"Unknown assets: {sorted(asset_ids - available)}")
    for item in source["prepared"]:
        aid = item["asset_id"]
        if asset_ids is not None and aid not in asset_ids:
            continue
        try:
            data = {"ohlcv_usd": [], "returns_usd": []}
            for rec in item["records"]:
                raw = store._get_bytes(rec["staged_part"])
                if hashlib.sha256(raw).hexdigest() != rec["sha256"]:
                    raise ValueError(f"Source staged checksum mismatch: {rec['staged_part']}")
                data[rec["table"]].append(_parquet(raw))
            if not data["ohlcv_usd"] or not data["returns_usd"]:
                raise ValueError("Missing staged prices/returns")
            prices = _dates(pd.concat(data["ohlcv_usd"], ignore_index=True)).sort_values("date")
            if prices.duplicated(["asset_id", "date"]).any() or (prices["asset_id"] != aid).any():
                raise ValueError("Duplicate dates or wrong asset ID")
            local = pd.to_numeric(prices["adj_close"], errors="coerce")
            raw_local = pd.to_numeric(prices["close"], errors="coerce")
            fx = pd.to_numeric(prices["fx_to_usd"], errors="coerce")
            current = pd.to_numeric(prices["close_adjusted_usd"], errors="coerce")
            raw_current = pd.to_numeric(prices["close_raw_usd"], errors="coerce")
            adj_bad = current.isna() | ~np.isfinite(current) | (current <= 0)
            raw_bad = raw_current.isna() | ~np.isfinite(raw_current) | (raw_current <= 0)
            to_fix = adj_bad | raw_bad
            if not to_fix.any():
                report.append({"asset_id": aid, "status": "UNCHANGED", "invalid_derived_rows": 0})
                continue
            eligible = np.isfinite(local) & (local > 0) & np.isfinite(raw_local) & (raw_local > 0) & np.isfinite(fx) & (fx > 0)
            unresolved = to_fix & ~eligible
            if unresolved.any():
                report.append({"asset_id": aid, "status": "UNRESOLVED_SOURCE_VALUES",
                               "invalid_derived_rows": int(to_fix.sum()), "unresolved_rows": int(unresolved.sum()),
                               "sample_dates": prices.loc[unresolved, "date"].dt.strftime("%Y-%m-%d").head(10).tolist()})
                continue
            # Only fill invalid cells; never silently alter existing non-null values.
            prices.loc[adj_bad, "close_adjusted_usd"] = (local * fx).loc[adj_bad].to_numpy()
            prices.loc[raw_bad, "close_raw_usd"] = (raw_local * fx).loc[raw_bad].to_numpy()
            prices.loc[adj_bad, "adj_close_usd"] = prices.loc[adj_bad, "close_adjusted_usd"]
            prices.loc[raw_bad, "close_usd"] = prices.loc[raw_bad, "close_raw_usd"]
            adjusted = pd.to_numeric(prices["close_adjusted_usd"], errors="coerce")
            if not (np.isfinite(adjusted) & (adjusted > 0)).all():
                raise ValueError("Remaining invalid adjusted USD close after backfill")
            ticker = str(prices["ticker"].iloc[-1])
            returns = _returns(prices, aid, ticker)
            if len(returns) != len(prices) - 1:
                raise ValueError("Return coverage unexpected after backfill")
            # Preserve original expected manifest hashes, not newly read production state.
            original = {"ohlcv_usd": {}, "returns_usd": {}}
            for rec in item["records"]:
                # _prepare only needs original manifest bytes to derive the hash.
                # Construct them from current active manifests but check against recorded hash.
                key = rec["manifest_key"]
                expected = rec["original_manifest_sha256"]
                if key not in original[rec["table"]] and expected is not None:
                    blob = store._get_bytes(key)
                    if hashlib.sha256(blob).hexdigest() != expected:
                        raise ValueError(f"Active manifest changed since original preparation: {key}")
                    original[rec["table"]][key] = blob
            records = _prepare(store, aid, prices, returns, original, Path(new_run_id))
            prepared.append({"asset_id": aid, "records": records, "flagged_retained": item.get("flagged_retained", 0),
                             "derived_backfill": True, "source_campaign": source_run_id})
            report.append({"asset_id": aid, "status": "PREPARED", "invalid_derived_rows": int(to_fix.sum()),
                           "adjusted_fixed": int(adj_bad.sum()), "raw_fixed": int(raw_bad.sum()),
                           "flagged_retained": item.get("flagged_retained", 0), "partitions": len(records)})
        except Exception as exc:
            report.append({"asset_id": aid, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        print(f"[backfill] {aid}: {report[-1]['status']}", flush=True)
    if not prepared:
        raise RuntimeError("No corrected assets prepared; source campaign unchanged")
    campaign = {"run_id": new_run_id, "source_campaign": source_run_id, "kind": "DERIVED_USD_BACKFILL",
                "report": report, "prepared": prepared}
    store._put_bytes(prefix + new_run_id + "/campaign_report.json", _dump_json(report), "application/json")
    store._put_bytes(destination_key, _dump_json(campaign), "application/json")
    if json.loads(store._get_bytes(destination_key)) != campaign:
        raise IOError("Campaign manifest readback differs")
    print(f"DERIVED_BACKFILL_TO_S3 {len(prepared)} assets; manifest=s3://{store.bucket}/{destination_key}")
    return campaign


def main():
    p = argparse.ArgumentParser(description="AEA-87 S3-only derived USD staging backfill")
    p.add_argument("--source-run-id", required=True)
    p.add_argument("--repair-run-id", default=None, help="New immutable output campaign ID")
    p.add_argument("--asset-ids", default=None, help="Optional comma-separated affected IDs for pilot")
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--market-root", default="market")
    args = p.parse_args()
    rid = args.repair_run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    ids = set(x.strip() for x in args.asset_ids.split(",")) if args.asset_ids else None
    if ids is not None and ("" in ids or not ids):
        p.error("Invalid --asset-ids")
    store = _make_market_store(bucket=args.bucket, region=args.region, market_root=args.market_root)
    repair_derived_prices(store, args.source_run_id, rid, asset_ids=ids)


if __name__ == "__main__":
    main()
