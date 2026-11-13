"""S3-only Yahoo acquisition: source bars and corporate actions are captured once.

Raw Yahoo bars preserve the provider's split-adjusted Close and its separate
dividend-adjusted Adj Close. Stock splits MUST NOT be applied to Yahoo Close again.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone

import pandas as pd

from alpha_edge.market.ingest_bronze_yahoo_v2 import acquire_yahoo_bars, acquire_yahoo_fx_quote
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT,
    LakehouseError,
    check_id,
    digest,
    put_json,
    put_parquet,
    read_bytes,
)

REGISTRY_COLUMNS = {"asset_id", "yahoo_ticker", "currency", "calendar_id"}
FX_COLUMNS = {"currency", "provider_symbol", "direction"}


def registry_from_s3(s3, bucket: str, registry_key: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Read one pinned S3 JSON registry; no local configuration files."""
    if not registry_key.startswith("market/lake_v3/config/"):
        raise LakehouseError("Asset configuration must live under market/lake_v3/config/")
    raw = read_bytes(s3, bucket, registry_key)
    doc = json.loads(raw)
    assets = pd.DataFrame(doc["assets"])
    fx = pd.DataFrame(doc.get("fx_pairs", []), columns=sorted(FX_COLUMNS))
    if assets.empty or REGISTRY_COLUMNS - set(assets):
        raise LakehouseError("Registry missing assets or required columns")
    if assets["asset_id"].duplicated().any() or assets["yahoo_ticker"].isna().any():
        raise LakehouseError("Duplicate asset IDs or missing provider symbols")
    if not fx.empty and (FX_COLUMNS - set(fx) or fx["currency"].duplicated().any()):
        raise LakehouseError("Invalid FX registry")
    if not fx.empty and not fx["direction"].isin(["direct", "inverse"]).all():
        raise LakehouseError("FX direction must be direct or inverse")
    for col in REGISTRY_COLUMNS:
        if assets[col].isna().any() or assets[col].astype(str).str.strip().eq("").any():
            raise LakehouseError(f"Blank registry field: {col}")
    return assets, fx, digest(raw)


def acquire(
    *,
    s3,
    bucket: str,
    registry_key: str,
    start: str,
    end: str,
    run_id: str,
    ticker_factory=None,
    execute: bool = False,
) -> dict:
    check_id(run_id)
    if not date.fromisoformat(start) < date.fromisoformat(end):
        raise LakehouseError("Start must precede end (exclusive)")
    if date.fromisoformat(end) > datetime.now(timezone.utc).date():
        raise LakehouseError("End of daily session extraction may not be in the future")
    assets, fx_map, registry_sha = registry_from_s3(s3, bucket, registry_key)
    base = f"{ROOT}/bronze/runs/{run_id}"
    if not execute:
        return {"state": "dry_run", "assets": len(assets), "fx_pairs": len(fx_map), "target": base, "writes": 0}
    ohlcv_frames, action_frames, fx_frames, provenance = [], [], [], []
    for rec in assets.sort_values("asset_id").to_dict("records"):
        bars, meta = acquire_yahoo_bars(
            str(rec["asset_id"]),
            str(rec["yahoo_ticker"]),
            start=start,
            end=end,
            batch_id=run_id,
            ticker_factory=ticker_factory,
            currency_hint=str(rec["currency"]),
        )
        # Source columns contain Dividends / Stock Splits from the same Yahoo response;
        # do not invoke a second non-atomic Yahoo action request.
        for field, kind in (("dividends", "DIVIDEND"), ("stock_splits", "SPLIT")):
            values = pd.to_numeric(bars[field], errors="coerce")
            if values.isna().any() or values.lt(0).any():
                raise LakehouseError(f"Invalid Yahoo corporate actions: {rec['asset_id']}")
            event_rows = bars.loc[values.gt(0), ["asset_id", "provider_symbol", "session_date", field]].copy()
            event_rows = event_rows.rename(columns={field: "value", "session_date": "effective_date"})
            event_rows["action_type"] = kind
            event_rows["currency"] = str(rec["currency"]) if kind == "DIVIDEND" else ""
            action_frames.append(event_rows)
        ohlcv_frames.append(bars)
        provenance.append(meta)
    for rec in fx_map.sort_values("currency").to_dict("records"):
        fx, meta = acquire_yahoo_fx_quote(
            str(rec["provider_symbol"]),
            start=start,
            end=end,
            batch_id=run_id,
            ticker_factory=ticker_factory,
        )
        fx_frames.append(fx)
        provenance.append(meta)
    bars = pd.concat(ohlcv_frames, ignore_index=True)
    action_columns = ["asset_id", "provider_symbol", "effective_date", "value", "action_type", "currency"]
    actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame(columns=action_columns)
    fx = (
        pd.concat(fx_frames, ignore_index=True)
        if fx_frames
        else pd.DataFrame(
            columns=["provider_symbol", "session_date", "quote_close"],
        )
    )
    # Acquire ALL source observations before writing any permanent record.
    # A partial write without the final committed manifest is never consumable.
    references = {
        "bars": put_parquet(s3, bucket, base + "/source_ohlcv.parquet", bars),
        "actions": put_parquet(s3, bucket, base + "/source_actions.parquet", actions),
        "fx": put_parquet(s3, bucket, base + "/source_fx.parquet", fx),
    }
    manifest = {
        "schema": "alpha_edge_bronze_v3_1",
        "run_id": run_id,
        "registry_key": registry_key,
        "registry_sha256": registry_sha,
        "start": start,
        "end_exclusive": end,
        "assets": int(len(assets)),
        "rows": {"bars": int(len(bars)), "actions": int(len(actions)), "fx": int(len(fx))},
        "artifacts": references,
        "provenance": provenance,
        "price_basis": "yahoo_unadjusted_auto_adjust_false_close_is_provider_split_adjusted",
        "status": "bronze_committed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt = put_json(s3, bucket, base + "/manifest.json", manifest)
    return {"state": "bronze_committed", "manifest": receipt, "rows": manifest["rows"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="S3-only Yahoo Bronze acquisition")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--registry-key", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    a = parser.parse_args()
    import boto3

    result = acquire(
        s3=boto3.client("s3", region_name="eu-west-1"),
        bucket=a.bucket,
        registry_key=a.registry_key,
        start=a.start,
        end=a.end,
        run_id=a.run_id,
        execute=a.execute,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
