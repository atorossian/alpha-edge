"""Incremental market-lake v2 daily candidate builder, entirely S3/in memory.

This is the *daily production orchestration entrypoint*. It requires a
fully pinned current production v2 release and refuses to fabricate a
historical baseline. All persistent bytes (Bronze observations, FX, Silver,
Gold, indicators, source/quality evidence, manifests, audit) go to S3. Raw
daily Yahoo observations are committed before transformation. A staged
release is NOT automatically activated: independent evidence signoff
is required by production_gold_release_v2.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import io
import json
import re
import uuid
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from alpha_edge.core.audit import build_audit_event, write_audit_event, make_run_id
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.market.ingest_bronze_yahoo_v2 import (
    acquire_yahoo_bars, acquire_yahoo_fx_quote, _parquet_bytes,
)
from alpha_edge.market.standardize_bronze_v2 import build_silver_gold_from_bronze
from alpha_edge.market.manual_observations_v2 import append_manual_observations
from alpha_edge.market.independent_daily_quality_v2 import review as independent_quality_review
from alpha_edge.market.indicator_calculations import compute_market_indicators_for_asset
from alpha_edge.market.production_gold_release_v2 import (
    CURRENT, ROOT, get, sha, encoded, immutable_put, verify_object,
    validate_release, stage, ReleaseError,
)

MANDATORY = ("asset_id", "provider_symbol", "currency_original", "session_date")
KEYS = ("asset_id", "session_date")


class IncrementalError(RuntimeError):
    pass


def _day(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
        raise IncrementalError("as_of must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise IncrementalError("Invalid as_of") from exc


def _frame(raw: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(raw))


def _json(s3, bucket: str, key: str) -> dict:
    blob = get(s3, bucket, key)
    if blob is None:
        raise IncrementalError(f"Missing required S3 release input: {key}")
    return json.loads(blob)


def _put_parquet(s3, bucket: str, key: str, df: pd.DataFrame) -> dict:
    body = _parquet_bytes(df)
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*",
                      ContentType="application/octet-stream")
    except Exception:
        # Never overwrite a different immutable batch. A retry may verify
        # an existing identical object and continue after partial failure.
        if get(s3, bucket, key) != body:
            raise
    stored = get(s3, bucket, key)
    if stored != body:
        raise IncrementalError(f"S3 parquet readback mismatch: {key}")
    return {"key": key, "sha256": sha(body)}


def _put_json(s3, bucket: str, key: str, record: dict) -> dict:
    body = encoded(record)
    existing = get(s3, bucket, key)
    if existing is None:
        immutable_put(s3, bucket, key, body)
    elif existing != body:
        raise IncrementalError(f"Different immutable object already stored: {key}")
    if get(s3, bucket, key) != body:
        raise IncrementalError(f"S3 JSON readback mismatch: {key}")
    return {"key": key, "sha256": sha(body)}


def _merge(base: pd.DataFrame, fresh: pd.DataFrame, *,
           keys: tuple[str, ...], overlap_start: str) -> tuple[pd.DataFrame, int]:
    if not set(keys).issubset(base) or not set(keys).issubset(fresh):
        raise IncrementalError("Missing incremental merge key(s)")
    base = base.copy()
    fresh = fresh.copy()
    for source in (base, fresh):
        source["session_date"] = pd.to_datetime(
            source["session_date"].astype(str).str[:10], errors="raise"
        ).dt.strftime("%Y-%m-%d")
        if source.duplicated(list(keys)).any():
            raise IncrementalError("Duplicate source observations in daily merge")
    if fresh.empty:
        return base.copy(), 0
    # Provider can revise dates in the requested overlap. Never silently
    # alter older dates beyond the bounded revision window.
    days = pd.to_datetime(fresh["session_date"].astype(str).str[:10])
    if days.lt(pd.Timestamp(overlap_start)).any():
        raise IncrementalError("Provider returned observations before requested overlap")
    base_key = base.set_index(list(keys))
    new_key = fresh.set_index(list(keys))
    common = base_key.index.intersection(new_key.index)
    revisions = 0
    for key in common:
        left, right = base_key.loc[key], new_key.loc[key]
        fields = set(["open", "high", "low", "close", "adj_close", "volume"]) & set(left.index) & set(right.index)
        if any(
            not (pd.isna(left[c]) and pd.isna(right[c]))
            and not np.isclose(float(left[c]), float(right[c]),
                               rtol=1e-9, atol=1e-9, equal_nan=True)
            for c in fields
        ):
            revisions += 1
    untouched = base_key.loc[~base_key.index.isin(new_key.index)]
    merged = pd.concat([untouched.reset_index(), fresh], ignore_index=True, sort=False)
    merged = merged.sort_values(list(keys), kind="stable").reset_index(drop=True)
    if merged.duplicated(list(keys)).any():
        raise IncrementalError("Duplicate asset/date after incremental merge")
    return merged, revisions


def _indicators(gold: pd.DataFrame) -> pd.DataFrame:
    """Reuse canonical indicator calculations without bridging rejected sessions."""
    accepted = gold.loc[gold["quality_status"].eq("accepted")].copy()
    if accepted.empty:
        return pd.DataFrame()
    accepted["date"] = pd.to_datetime(
        accepted["session_date"].astype(str).str[:10], errors="raise")
    accepted["ticker"] = accepted["provider_symbol"]
    accepted["open_raw_usd"] = accepted["open"].astype(float) * accepted["fx_to_usd"]
    accepted["high_raw_usd"] = accepted["high"].astype(float) * accepted["fx_to_usd"]
    accepted["low_raw_usd"] = accepted["low"].astype(float) * accepted["fx_to_usd"]
    calculated = []
    for _, asset in accepted.groupby("asset_id", sort=False):
        asset = asset.sort_values("date", kind="stable").copy()
        # Gaps and extreme review returns reset historical indicator windows.
        # Unavailable/review daily returns must not become implicit contiguous
        # volatility/momentum observations in indicator calculations.
        segments = asset["return_quality_status"].ne("accepted").cumsum()
        for _, part in asset.groupby(segments, sort=False):
            part = part.copy()
            part.loc[part["return_quality_status"].ne("accepted"),
                     "ret_log_close_adjusted_usd"] = np.nan
            returns = part[["asset_id", "date", "ret_log_close_adjusted_usd"]].copy()
            frame = compute_market_indicators_for_asset(part, returns=returns)
            if not frame.empty:
                calculated.append(frame)
    return pd.concat(calculated, ignore_index=True) if calculated else pd.DataFrame()

def _audit(cfg, *, run_id, as_of, status, metadata=None, error=None, keys=()):
    event = build_audit_event(
        cfg=cfg, event_type="market_lake_v2_daily",
        entity_type="market_data_release_candidate", entity_id=run_id,
        as_of=as_of, source_script="alpha_edge.market.run_daily_pipeline_v2",
        source_mode="prod_s3_only", status=status, run_id=run_id,
        output_keys=keys, metadata=metadata or {}, error=error,
    )
    return write_audit_event(cfg=cfg, event=event)


def run(*, s3, bucket: str, as_of: str, execute: bool,
        confirm_prod_write: bool = False, lookback_days: int = 10,
        run_id: str | None = None, ticker_factory=None, audit_writer=None) -> dict:
    if not bucket or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{2,62}", bucket):
        raise IncrementalError("Invalid target bucket")
    cfg = replace(load_runtime_config("prod"), bucket=bucket)
    if execute:
        require_prod_confirmation(cfg, confirm_prod_write)
    if not 3 <= lookback_days <= 35:
        raise IncrementalError("Overlap must be 3..35 calendar days")
    target = _day(as_of)
    if target > datetime.now(timezone.utc).date():
        raise IncrementalError("Cannot ingest future market sessions")
    pointer = _json(s3, bucket, CURRENT)
    if pointer.get("schema") != "alpha_edge_v2_release_pointer_1":
        raise IncrementalError("Missing approved v2 production pointer")
    release_raw = get(s3, bucket, pointer["release_key"])
    if release_raw is None or sha(release_raw) != pointer.get("release_sha256"):
        raise IncrementalError("Current v2 Gold pointer is corrupt")
    prior = json.loads(release_raw)
    validate_release(s3, bucket, prior, require_approval=True)
    prior_date = _day(prior["as_of"])
    if target <= prior_date:
        raise IncrementalError("Daily as_of is not newer than current production Gold")
    if (target - prior_date).days > lookback_days:
        raise IncrementalError("Daily run exceeds bounded overlap; backfill missing interval first")

    run_id = run_id or make_run_id("lake_v2_daily")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,100}", run_id):
        raise IncrementalError("Unsafe immutable run ID")
    start = (target - timedelta(days=lookback_days)).isoformat()
    end = (target + timedelta(days=1)).isoformat()
    prefix = f"{ROOT}/daily/candidates/dt={as_of}/run={run_id}"
    if not execute:
        return {
            "status": "dry_run", "as_of": as_of,
            "retrieval_start": start, "retrieval_end_exclusive": end,
            "previous_release_id": prior["release_id"],
            "expected_cohorts": prior["expected_cohorts"],
            "target_prefix": prefix, "s3_writes": 0,
            "gold_activated": False,
        }
    records = []
    output_keys = []
    try:
        for cohort in prior["cohorts"]:
            idx = int(cohort["cohort"])
            c_prefix = f"{prefix}/cohort-{idx:04d}"
            registry = cohort.get("registry")
            if not isinstance(registry, list) or not registry:
                raise IncrementalError("Previous release lacks complete asset registry")
            assets = cohort["assets"]
            if set(assets) != {x.get("asset_id") for x in registry}:
                raise IncrementalError("Prior release asset IDs do not match registry")
            raw_bars = _frame(verify_object(s3, bucket, cohort["bronze"]))
            raw_fx = _frame(verify_object(s3, bucket, cohort["fx"]))
            original_gold = _frame(verify_object(s3, bucket, cohort["gold"]))
            manual_prior = (
                original_gold.get("observation_origin", pd.Series(dtype="string"))
                .eq("user_supplied_manual").any()
            )
            corrections_ref = cohort.get("corrections")
            if manual_prior and not isinstance(corrections_ref, dict):
                raise IncrementalError("Historical manual corrections require a pinned S3 registry")
            corrections = (
                _frame(verify_object(s3, bucket, corrections_ref))
                if isinstance(corrections_ref, dict) else None
            )
            required_fx = cohort.get("fx_mapping", [])
            if not isinstance(required_fx, list):
                raise IncrementalError("Missing FX mapping")
            needed_currencies = {
                row["currency"] for row in registry if row["currency"] != "USD"
            }
            rates = {}
            symbols = {}
            for entry in required_fx:
                currency = entry["currency"]
                if currency in rates or entry["direction"] not in {"direct", "inverse"}:
                    raise IncrementalError("Duplicate/invalid FX mapping")
                if currency in needed_currencies:
                    rates[currency] = (entry["provider_symbol"], entry["direction"])
                    symbols[entry["provider_symbol"]] = entry
            if set(rates) != needed_currencies:
                raise IncrementalError("Missing explicit native-currency FX mapping")
            fresh = []
            provenance = []
            for asset in registry:
                bars, origin = acquire_yahoo_bars(
                    asset["asset_id"], asset["yahoo_ticker"],
                    start=start, end=end, batch_id=run_id,
                    currency_hint=asset["currency"], ticker_factory=ticker_factory,
                )
                fresh.append(bars)
                provenance.append(origin)
            raw_new = pd.concat(fresh, ignore_index=True)
            fx_new = []
            for symbol in sorted(symbols):
                frame, origin = acquire_yahoo_fx_quote(
                    symbol, start=start, end=end, batch_id=run_id,
                    ticker_factory=ticker_factory,
                )
                fx_new.append(frame)
                provenance.append(origin)
            new_quotes = pd.concat(fx_new, ignore_index=True) if fx_new else raw_fx.iloc[0:0].copy()

            # Record unaltered daily source FIRST in immutable production Bronze.
            bronze_daily = f"{ROOT}/bronze/incremental/dt={as_of}/run={run_id}/cohort-{idx:04d}"
            daily_bars = _put_parquet(s3, bucket, bronze_daily + "/source_ohlcv.parquet", raw_new)
            daily_fx = _put_parquet(s3, bucket, bronze_daily + "/source_fx_quotes.parquet", new_quotes)
            capture = _put_json(s3, bucket, bronze_daily + "/source_manifest.json", {
                "schema": "alpha_edge_daily_raw_capture_v2_1",
                "as_of": as_of, "run_id": run_id,
                "retrieval_start": start, "retrieval_end_exclusive": end,
                "source_bars": daily_bars, "source_fx": daily_fx,
                "provenance": provenance,
            })
            _put_json(s3, bucket, bronze_daily + "/_bronze_committed.json", {
                "schema": "alpha_edge_daily_bronze_commit_v2_1",
                "source_manifest": capture, "source_bars": daily_bars,
                "source_fx": daily_fx, "quality_approval": "not_yet_reviewed",
            })
            output_keys.extend([daily_bars["key"], daily_fx["key"]])

            combined_bars, revisions = _merge(
                raw_bars, raw_new, keys=KEYS, overlap_start=start)
            combined_fx, fx_revisions = _merge(
                raw_fx, new_quotes, keys=("provider_symbol", "session_date"),
                overlap_start=start)
            calendars = {x["yahoo_ticker"]: x["calendar_id"] for x in registry}
            corrected_bars = combined_bars
            correction_evidence = None
            if corrections is not None and not corrections.empty:
                corrected_bars, correction_evidence = append_manual_observations(
                    combined_bars, corrections
                )
            tables = build_silver_gold_from_bronze(
                corrected_bars, combined_fx, calendars=calendars, pairs=rates)
            gold = tables["gold"]
            if set(gold["asset_id"]) != set(assets):
                raise IncrementalError("Incremental Gold asset coverage mismatch")
            if gold.duplicated(list(KEYS)).any():
                raise IncrementalError("Duplicate Gold observation keys")
            # Daily snapshot stays unapproved until independently evaluated.
            snapshot = {
                "corrections": (corrections_ref if correction_evidence is None else
                                _put_parquet(s3, bucket, c_prefix + "/corrections.parquet",
                                             correction_evidence)),
                "bronze": _put_parquet(s3, bucket, c_prefix + "/raw_history.parquet", combined_bars),
                "fx": _put_parquet(s3, bucket, c_prefix + "/raw_fx_history.parquet", combined_fx),
                "silver": _put_parquet(s3, bucket, c_prefix + "/silver.parquet", tables["silver"]),
                "gold": _put_parquet(s3, bucket, c_prefix + "/gold.parquet", gold),
                "indicator_candidate": _put_parquet(s3, bucket, c_prefix + "/indicators.parquet",
                                                     _indicators(gold)),
            }
            quality = independent_quality_review(
                tables, manual_corrections=(
                    len(correction_evidence) if correction_evidence is not None else 0
                ),
            )
            quality.update({
                "as_of": as_of, "cohort": idx,
                "price_revisions_in_overlap": revisions,
                "fx_revisions_in_overlap": fx_revisions,
                "bronze_daily_capture": capture,
                "source_gold_snapshot": snapshot["gold"],
                "indicator_candidate": snapshot["indicator_candidate"],
            })
            snapshot["acceptance"] = _put_json(s3, bucket, c_prefix + "/acceptance.json", quality)
            snapshot.update({
                "cohort": idx, "assets": assets, "registry": registry,
                "fx_mapping": required_fx, "gold_rows": len(gold),
            })
            records.append(snapshot)
        candidate = {
            "schema": "alpha_edge_v2_release_candidate_1",
            "release_id": run_id, "as_of": as_of,
            "expected_cohorts": prior["expected_cohorts"],
            "expected_assets": prior["expected_assets"],
            "cohorts": records,
            "previous_release_id": prior["release_id"],
            "universe_reconciliation": "pending",
            "indicator_validation": "pending",
            "consumer_contract_validation": "pending",
            "source_observation_state": "candidate_requires_independent_approval",
        }
        # Cannot stage as an approved Gold release: this is a candidate only.
        # Immutable snapshots and the complete aggregate manifest live in S3.
        receipt = _put_json(s3, bucket, prefix + "/release_candidate.json", candidate)
        output_keys.append(receipt["key"])
        # Stage under the canonical immutable release catalog, without
        # publishing current.json. Arithmetic failures fail closed.
        staged = stage(s3, bucket=bucket, release=candidate, execute=True)
        output_keys.append(staged["release_key"])
        writer = audit_writer or _audit
        writer(cfg, run_id=run_id, as_of=as_of, status="review_required",
               keys=output_keys, metadata={
                   "candidate": receipt, "staged_release": staged,
                   "cohorts": len(records),
                   "previous_release_id": prior["release_id"],
                   "s3_only": True, "gold_activated": False,
               })
        return {
            "status": "review_required", "candidate_manifest": receipt,
            "release_manifest": staged["release_key"],
            "release_sha256": staged["release_sha256"],
            "cohorts_staged": len(records), "gold_activated": False,
            "persistent_local_files": 0,
        }
    except Exception as exc:
        writer = audit_writer or _audit
        writer(cfg, run_id=run_id, as_of=as_of, status="failed",
               keys=output_keys, error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", required=True)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--lookback-days", type=int, default=10)
    p.add_argument("--run-id")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm-prod-write", action="store_true")
    args = p.parse_args()
    import boto3
    s3 = boto3.client("s3", region_name=args.region)
    print(json.dumps(run(
        s3=s3, bucket=args.bucket, as_of=args.as_of,
        execute=args.execute, confirm_prod_write=args.confirm_prod_write,
        lookback_days=args.lookback_days, run_id=args.run_id), indent=2))


if __name__ == "__main__":
    main()
