"""Validated Silver and analytical Gold from checksum-pinned S3 Bronze.

Reuses the v2 calendar/FX/date contracts and the existing indicator engine.
All intermediate data are in memory; final datasets are immutable S3 objects.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from alpha_edge.market.data_lake_v2 import QualityPolicy
from alpha_edge.market.indicator_calculations import compute_market_indicators_for_asset
from alpha_edge.market.standardize_bronze_v2 import build_silver_gold_from_bronze
from alpha_edge.market.lakehouse_v3.ingest import registry_from_s3
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT, LakehouseError, check_id, put_json, put_parquet, read_json, verified_frame,
)

INDICATORS = (
    "sma_20", "sma_50", "sma_200", "rsi_14", "atr_14",
    "macd", "macd_signal", "macd_hist", "annualized_vol_20",
    "annualized_vol_60", "drawdown_20", "drawdown_60", "drawdown_252",
)


def validate_events(bars: pd.DataFrame, actions: pd.DataFrame) -> dict:
    """Ensure separately materialized action evidence matches the same Yahoo response."""
    expected = {"asset_id", "provider_symbol", "effective_date", "value", "action_type", "currency"}
    if expected - set(actions.columns):
        raise LakehouseError("Corporate-action evidence contract incomplete")
    if actions.duplicated(["asset_id", "effective_date", "action_type"]).any():
        raise LakehouseError("Duplicate corporate action event")
    for field, kind in (("dividends", "DIVIDEND"), ("stock_splits", "SPLIT")):
        subset = actions.loc[actions["action_type"].eq(kind)]
        observed = pd.to_numeric(bars[field], errors="coerce")
        if observed.isna().any() or observed.lt(0).any():
            raise LakehouseError(f"Invalid {field} observations in source")
        if len(subset) != int(observed.gt(0).sum()):
            raise LakehouseError(f"{kind} event count differs from source bars")
        if subset.empty:
            continue
        joined = subset.merge(
            bars[["asset_id", "session_date", field]],
            left_on=["asset_id", "effective_date"],
            right_on=["asset_id", "session_date"],
            how="left", validate="one_to_one",
        )
        if joined[field].isna().any() or not np.isclose(
            joined["value"].astype(float), joined[field].astype(float), rtol=1e-10, atol=1e-10
        ).all():
            raise LakehouseError(f"{kind} evidence differs from original provider bars")
    if not actions["action_type"].isin(["DIVIDEND", "SPLIT"]).all():
        raise LakehouseError("Unknown corporate action type")
    return {
        "events": int(len(actions)),
        "dividends": int(actions["action_type"].eq("DIVIDEND").sum()),
        "splits": int(actions["action_type"].eq("SPLIT").sum()),
    }


def build_tables(
    *,
    bars: pd.DataFrame, fx: pd.DataFrame, actions: pd.DataFrame,
    assets: pd.DataFrame, fx_registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Compute a complete cohort in memory. No S3 operation occurs here."""
    validate_events(bars, actions)
    # The v2 indicator engine falls back to close when OHLC is missing.
    # For a certified lakehouse we must never manufacture OHLC observations.
    ohlc = bars[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if (ohlc.isna().any().any() or not np.isfinite(ohlc.to_numpy()).all()
            or (ohlc <= 0).any().any()
            or (ohlc["high"] < ohlc[["open", "low", "close"]].max(axis=1)).any()
            or (ohlc["low"] > ohlc[["open", "high", "close"]].min(axis=1)).any()):
        raise LakehouseError("Invalid OHLC bars; refusing synthetic indicator inputs")
    calendars = dict(zip(assets["yahoo_ticker"], assets["calendar_id"]))
    if "calendar_id" not in assets or assets["calendar_id"].isna().any():
        raise LakehouseError("Exchange calendar required for all assets")
    pairs = {
        str(row["currency"]): (str(row["provider_symbol"]), str(row["direction"]))
        for row in fx_registry.to_dict("records")
    }
    required_fx = set(assets["currency"].astype(str).str.upper()) - {"USD"}
    if required_fx - set(pairs):
        raise LakehouseError(f"Missing FX mappings: {sorted(required_fx - set(pairs))}")
    # The v2 processor provides split-safe source dates, calendar gap detection,
    # explicit FX rates, source-quality and adjusted-close return diagnostics.
    transformed = build_silver_gold_from_bronze(
        bars, fx, policy=QualityPolicy(), calendars=calendars, pairs=pairs,
    )
    silver = transformed["silver"].copy()
    gold = transformed["gold"].copy()
    if silver.duplicated(["asset_id", "session_date"]).any():
        raise LakehouseError("Duplicate canonical asset/session key")
    if len(gold) != len(bars):
        raise LakehouseError("Non-reconciling Bronze/Gold row counts")

    gold["session_date"] = pd.to_datetime(gold["session_date"]).dt.strftime("%Y-%m-%d")
    silver["session_date"] = pd.to_datetime(silver["session_date"]).dt.strftime("%Y-%m-%d")
    gold["close_price_usd"] = gold["close_raw_usd"]
    gold["close_total_return_index_usd"] = gold["close_adjusted_usd"]
    # Yahoo Close from auto_adjust=False is normally already split-adjusted.
    # The adjustment ratio Adj Close / Close represents provider total-return
    # adjustment, not a second split operation.
    gold["ret_price_usd"] = (
        gold["close_price_usd"] /
        gold.groupby("asset_id")["close_price_usd"].shift(1) - 1
    )
    gold["ret_total_usd"] = gold["ret_simple_close_adjusted_usd"]
    gold["ret_log_total_usd"] = gold["ret_log_close_adjusted_usd"]
    unavailable = gold["return_quality_status"].ne("accepted")
    gold.loc[unavailable, ["ret_price_usd", "ret_total_usd", "ret_log_total_usd"]] = np.nan
    gold["dividend_per_share_local"] = pd.to_numeric(gold["dividends"], errors="coerce").fillna(0.0)
    gold["split_new_shares_per_old_share"] = pd.to_numeric(gold["stock_splits"], errors="coerce").fillna(0.0)
    gold["price_basis"] = "yahoo_close_provider_split_adjusted"
    gold["total_return_basis"] = "yahoo_adj_close_dividend_and_split_adjusted"
    gold["corporate_action_independently_verified"] = False
    gold["fx_timing_independently_verified"] = False

    all_indicators = []
    for asset_id, asset in gold.groupby("asset_id", sort=True):
        asset = asset.sort_values("session_date", kind="stable")
        # Reset indicator state across invalid prices and unavailable/review returns
        # so no momentum or volatility calculation spans a missing observation.
        accepted = asset["quality_status"].eq("accepted")
        blocks = (accepted.ne(accepted.shift(fill_value=False))
                  | asset["return_quality_status"].ne("accepted")).cumsum()
        for _, segment in asset.groupby(blocks, sort=False):
            if not segment["quality_status"].eq("accepted").all():
                continue
            temp = segment.copy()
            temp["date"] = pd.to_datetime(temp["session_date"])
            temp["ticker"] = temp["provider_symbol"]
            temp["open_raw_usd"] = temp["open"].astype(float) * temp["fx_to_usd"]
            temp["high_raw_usd"] = temp["high"].astype(float) * temp["fx_to_usd"]
            temp["low_raw_usd"] = temp["low"].astype(float) * temp["fx_to_usd"]
            rr = temp[["asset_id", "date", "ret_log_total_usd"]].rename(
                columns={"ret_log_total_usd": "ret_log_close_adjusted_usd"}
            )
            values = compute_market_indicators_for_asset(temp, returns=rr)
            if not values.empty:
                all_indicators.append(values[["asset_id", "date", *INDICATORS]])
    if all_indicators:
        indicators = pd.concat(all_indicators, ignore_index=True)
        indicators["session_date"] = indicators["date"].dt.strftime("%Y-%m-%d")
        indicators = indicators.drop(columns="date")
        if indicators.duplicated(["asset_id", "session_date"]).any():
            raise LakehouseError("Indicator duplicates detected")
        gold = gold.merge(indicators, on=["asset_id", "session_date"], how="left", validate="one_to_one")
    else:
        for name in INDICATORS:
            gold[name] = np.nan
    if gold.duplicated(["asset_id", "session_date"]).any():
        raise LakehouseError("Gold asset/session uniqueness broken")
    status = gold["quality_status"].value_counts().to_dict()
    returns_status = gold["return_quality_status"].value_counts().to_dict()
    evidence = {
        "assets": int(gold["asset_id"].nunique()), "bronze_rows": len(bars),
        "silver_rows": len(silver), "gold_rows": len(gold),
        "price_quality": {str(k): int(v) for k, v in status.items()},
        "return_quality": {str(k): int(v) for k, v in returns_status.items()},
        "exchange_session_issues": int(len(transformed["session_issues"])),
        "fx_quote_issues": int(transformed["fx_evidence"]["fx_validation_status"].ne("accepted").sum()),
        "actions": validate_events(bars, actions),
        "external_evidence": "not_approved",
    }
    return silver, gold, transformed["session_issues"], evidence


def process(s3, bucket: str, *, bronze_run_id: str, execute: bool = False) -> dict:
    check_id(bronze_run_id)
    bronze = read_json(s3, bucket, f"{ROOT}/bronze/runs/{bronze_run_id}/manifest.json")
    if bronze.get("schema") != "alpha_edge_bronze_v3_1" or bronze.get("status") != "bronze_committed":
        raise LakehouseError("Uncommitted or incompatible Bronze")
    assets, fx_registry, sha = registry_from_s3(s3, bucket, bronze["registry_key"])
    if sha != bronze["registry_sha256"]:
        raise LakehouseError("Configuration revision changed since ingestion")
    if not execute:
        return {"state": "dry_run", "bronze_run_id": bronze_run_id, "writes": 0}
    bars = verified_frame(s3, bucket, bronze["artifacts"]["bars"])
    actions = verified_frame(s3, bucket, bronze["artifacts"]["actions"])
    fx = verified_frame(s3, bucket, bronze["artifacts"]["fx"])
    if any(len(frame) != bronze["rows"][key] for key, frame in (
        ("bars", bars), ("actions", actions), ("fx", fx)
    )):
        raise LakehouseError("Bronze row count differs from checksum-pinned manifest")
    silver, gold, sessions, quality = build_tables(
        bars=bars, fx=fx, actions=actions, assets=assets, fx_registry=fx_registry,
    )
    prefix = f"{ROOT}/candidates/{bronze_run_id}"
    artifacts = {
        "silver": put_parquet(s3, bucket, prefix + "/silver/data.parquet", silver),
        "gold": put_parquet(s3, bucket, prefix + "/gold/data.parquet", gold),
        "session_issues": put_parquet(s3, bucket, prefix + "/quality/session_issues.parquet", sessions),
        "quality": put_json(s3, bucket, prefix + "/quality/report.json", quality),
    }
    manifest = {
        "schema": "alpha_edge_gold_candidate_v3_1",
        "run_id": bronze_run_id,
        "bronze_manifest_key": f"{ROOT}/bronze/runs/{bronze_run_id}/manifest.json",
        "bronze_registry_sha256": sha, "artifacts": artifacts,
        "quality": quality, "status": "review_required",
        "approved": False,
    }
    receipt = put_json(s3, bucket, prefix + "/manifest.json", manifest)
    return {"state": "review_required", "candidate_manifest": receipt, "quality": quality}


def main() -> None:
    p = argparse.ArgumentParser(description="Build Silver/Gold candidates from immutable S3 Bronze")
    p.add_argument("--bucket", required=True)
    p.add_argument("--bronze-run-id", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    import boto3
    print(json.dumps(process(
        boto3.client("s3", region_name="eu-west-1"), a.bucket,
        bronze_run_id=a.bronze_run_id, execute=a.execute,
    ), indent=2))


if __name__ == "__main__":
    main()
