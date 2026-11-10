"""AEA-87: fresh full-history Yahoo reconstruction; S3 staging only.

Never invokes incremental ingestion or updates active manifests.
Every provider response is preserved before normalization.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import uuid
import time
from collections import Counter
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
import yfinance as yf
from alpha_edge import paths
from alpha_edge.core.market_store import MarketStore
from alpha_edge.market.aea87_rebuild_validation import (fx_symbol, normalize_fx_history, convert_raw_with_fx, corporate_actions_report, coverage_review)

DEFAULT_ASSETS = ("CRYPTO:HYPE-USD", "EQH028b839246b50817",
                  "EQH09dbac0e89987ccc", "EQH40e5d597ac178c4e",
                  "EQH500160ab01b152f2", "EQH80c4fd2e9d0f2967")
ROOT = "market/repairs/aea87/rebuilds/v1/runs"


def parquet_bytes(frame):
    stream = io.BytesIO()
    frame.to_parquet(stream, index=False)
    return stream.getvalue()


def normalize_dates(frame):
    """Retain the exchange-local session date of each Yahoo *daily* candle.

    Yahoo provides timezone-aware midnight-at-exchange dates. Converting these
    to UTC first shifts many European session dates to the previous Sunday.
    Never use this normalization for intraday bars (which are instants).
    """
    out = frame.copy()
    dates = out["date"]
    try:
        parsed = pd.to_datetime(dates, errors="coerce")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            # Discard the timezone while retaining the provider's wall date.
            parsed = parsed.dt.tz_localize(None)
        elif not pd.api.types.is_datetime64_any_dtype(parsed):
            # Mixed timezone offsets (e.g. around DST) can produce object
            # dtype. Preserve each element's own local date, not its UTC day.
            parsed = dates.map(
                lambda v: pd.Timestamp(v).date() if pd.notna(v) else pd.NaT
            )
            parsed = pd.to_datetime(parsed, errors="coerce")
    except (TypeError, ValueError):
        parsed = dates.map(
            lambda v: pd.Timestamp(v).date() if pd.notna(v) else pd.NaT
        )
        parsed = pd.to_datetime(parsed, errors="coerce")
    out["date"] = parsed.dt.normalize()
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")


def download_fresh(symbol, start, end_exclusive):
    frame = yf.Ticker(symbol).history(start=start, end=end_exclusive, interval="1d",
                                       auto_adjust=False, actions=True, repair=False,
                                       keepna=True, raise_errors=True)
    if frame is None or frame.empty:
        raise ValueError("Yahoo returned an empty history")
    frame = frame.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
    if "date" not in frame:
        raise ValueError("Yahoo response lacks date column")
    return normalize_dates(frame)


def provider_metadata(symbol):
    ticker = yf.Ticker(symbol)
    out = {"symbol": symbol, "currency": None, "exchange": None, "quote_type": None}
    try:
        info = ticker.fast_info
        out["currency"] = info.get("currency")
        out["exchange"] = info.get("exchange")
    except Exception:
        pass
    try:
        meta = ticker.get_history_metadata() or {}
        out["currency"] = out["currency"] or meta.get("currency")
        out["exchange"] = out["exchange"] or meta.get("exchangeName")
        out["quote_type"] = meta.get("instrumentType")
    except Exception:
        pass
    return out


def build_candidate(raw, asset_id, currency, inception=None, fx=None):
    """No FX shortcuts: stage a canonical USD candidate only for USD instruments."""
    if str(currency or "").upper() != "USD" and fx is None:
        return None, None, "NON_USD_REQUIRES_HISTORICAL_FX"
    columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if not set(columns).issubset(raw.columns):
        return None, None, "INCOMPLETE_RAW_OHLCV"
    x = raw.loc[raw["date"] >= pd.Timestamp(inception)].copy() if inception else raw.copy()
    if x.empty:
        return None, None, "NO_POST_INCEPTION_OBSERVATIONS"
    for col in columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    # Yahoo keepna=True retains source placeholders (including an entirely
    # empty HYPE candle on 2026-02-02). Preserve them in raw_yahoo, but do
    # not create a fictitious trading observation or bridge returns across it.
    # All other incomplete bars remain visible and fail publication validation.
    empty_candle = x[["Open", "High", "Low", "Close", "Adj Close"]].isna().all(axis=1)
    x = x.loc[~empty_candle].copy()
    if x.empty:
        return None, None, "NO_VALID_SOURCE_CANDLES"
    if fx is not None:
        x = convert_raw_with_fx(x, fx)
        if x["fx_to_usd"].isna().any():
            return None, None, "MISSING_RECENT_HISTORICAL_FX"
        for col in ("Open", "High", "Low", "Close", "Adj Close"):
            x[col] = x[col + "_usd"]
    if (x["Close"].notna() & ~np.isfinite(x["Close"])).any() or (x["Adj Close"].notna() & ~np.isfinite(x["Adj Close"])).any() or x["Close"].dropna().le(0).any() or x["Adj Close"].dropna().le(0).any():
        return None, None, "INVALID_SOURCE_PRICES"
    prices = pd.DataFrame({"asset_id": asset_id, "date": x["date"],
                           "open_raw_usd": x["Open"], "high_raw_usd": x["High"],
                           "low_raw_usd": x["Low"], "close_raw_usd": x["Close"],
                           "close_adjusted_usd": x["Adj Close"],
                           "close_usd": x["Close"], "adj_close_usd": x["Adj Close"],
                           "volume": x["Volume"]}).reset_index(drop=True)
    if fx is not None:
        prices["fx_to_usd"] = x["fx_to_usd"].to_numpy()
        prices["fx_quote_date"] = x["fx_quote_date"].to_numpy()
        prices["fx_age_calendar_days"] = x["fx_age_calendar_days"].to_numpy()
    p = prices["close_adjusted_usd"]
    valid = p.gt(0) & p.shift().gt(0)
    returns = pd.DataFrame({"asset_id": asset_id, "date": prices["date"],
                            "ret_close_adjusted_usd": (p / p.shift() - 1).where(valid)})
    returns["ret_adj_close_usd"] = returns["ret_close_adjusted_usd"]
    return prices, returns, "STAGED_FOR_REVIEW"


def compare_histories(old, new):
    if old.empty:
        return {"old_rows": 0, "new_rows": len(new), "overlap": 0}
    old = normalize_dates(old)
    # Legacy history may have no canonical adjusted column at all. That is
    # precisely a reconstruction candidate, not a reason to abandon source
    # download after a valid fresh provider response.
    if "close_adjusted_usd" not in old:
        old["close_adjusted_usd"] = np.nan
    a = old[["date", "close_adjusted_usd"]].rename(columns={"close_adjusted_usd": "old"})
    b = new[["date", "close_adjusted_usd"]].rename(columns={"close_adjusted_usd": "new"})
    m = a.merge(b, on="date", how="outer", indicator=True)
    old_p, new_p = pd.to_numeric(m["old"], errors="coerce"), pd.to_numeric(m["new"], errors="coerce")
    valid = old_p.gt(0) & new_p.gt(0)
    def jumps(frame):
        p = pd.to_numeric(frame["close_adjusted_usd"], errors="coerce")
        return int(p.pct_change(fill_method=None).abs().gt(.50).sum())
    return {"old_rows": len(old), "new_rows": len(new), "overlap": int(valid.sum()),
            "old_missing_new_valid": int((~old_p.gt(0) & new_p.gt(0)).sum()),
            "disagreements_gt_1pct": int((valid & (old_p / new_p - 1).abs().gt(.01)).sum()),
            "old_only_dates": int(m["_merge"].eq("left_only").sum()),
            "new_only_dates": int(m["_merge"].eq("right_only").sum()),
            "old_large_moves": jumps(old), "new_large_moves": jumps(new)}


def run(args):
    ids = [x.strip() for x in args.asset_ids.split(",") if x.strip()]
    universe = pd.read_csv(args.universe, dtype=str).fillna("")
    required = {"asset_id", "yahoo_ticker"}
    if not required.issubset(universe):
        raise ValueError("Universe requires asset_id and yahoo_ticker")
    selected = universe[universe["asset_id"].isin(ids)].copy()
    if selected["asset_id"].duplicated().any():
        raise ValueError("Ambiguous asset identity: duplicate asset_id in universe")
    mapped = selected.set_index("asset_id")
    run_id = getattr(args, "run_id", None) or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    prefix = f"{ROOT}/{run_id}"
    s3 = boto3.client("s3", region_name=args.region)
    store = MarketStore(bucket=args.bucket, region=args.region)
    results, objects = {}, []
    fx_cache = {}
    started = time.monotonic()
    total = len(ids)
    print(f"[rebuild/source] run_id={run_id} assets={total} start={args.start} end_exclusive={args.end_exclusive}", flush=True)
    def put(relative, body, mime):
        if isinstance(body, str):
            body = body.encode()
        key = prefix + "/" + relative
        s3.put_object(Bucket=args.bucket, Key=key, Body=body, ContentType=mime, IfNoneMatch="*")
        objects.append({"key": key, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)})
    for completed, asset_id in enumerate(ids, 1):
        state = {"asset_id": asset_id, "status": "NOT_STARTED"}
        results[asset_id] = state
        try:
            if asset_id not in mapped.index:
                state["status"] = "NO_UNIVERSE_MAPPING"
                continue
            symbol = str(mapped.at[asset_id, "yahoo_ticker"]).strip()
            if not symbol:
                state["status"] = "NO_YAHOO_SYMBOL"
                continue
            state["yahoo_ticker"] = symbol
            safe = "".join(c if c.isalnum() else "_" for c in asset_id)
            raw = download_fresh(symbol, args.start, args.end_exclusive)
            put(f"assets/{safe}/raw_yahoo.parquet", parquet_bytes(raw), "application/octet-stream")
            state["raw_rows"] = len(raw)
            meta = provider_metadata(symbol)
            # Canonical universe metadata is an allowed fallback for absent
            # provider metadata, not a substitute for conflicting provider data.
            universe_currency = str(mapped.at[asset_id, "currency"]).strip() if "currency" in mapped else ""
            if not meta.get("currency") and universe_currency:
                meta["currency"] = universe_currency
                meta["currency_source"] = "universe_fallback"
            elif universe_currency and str(meta.get("currency")).upper() != universe_currency.upper():
                state["status"] = "PROVIDER_CURRENCY_CONFLICT"
                state["provider_currency"] = meta.get("currency")
                state["universe_currency"] = universe_currency
                continue
            if not meta.get("exchange") and "exchange" in mapped:
                meta["exchange"] = str(mapped.at[asset_id, "exchange"]).strip() or None
            state["provider_metadata"] = meta
            put(f"assets/{safe}/provider_metadata.json", json.dumps(meta, indent=2), "application/json")
            # HYPE ticker is explicitly validated; other symbols need independent identity review.
            if asset_id == "CRYPTO:HYPE-USD" and symbol != "HYPE32196-USD":
                state["status"] = "IDENTITY_SYMBOL_CONFLICT"
                continue
            put(f"assets/{safe}/corporate_actions.json", json.dumps(corporate_actions_report(raw), indent=2), "application/json")
            fx = None
            if str(meta["currency"] or "").upper() != "USD":
                direct, inverse = fx_symbol(meta["currency"])
                fx_error = None
                for fx_ticker, inverse_flag in ((direct, False), (inverse, True)):
                    try:
                        cache_key = (fx_ticker, inverse_flag)
                        if cache_key not in fx_cache:
                            fx_raw = download_fresh(fx_ticker, args.start, args.end_exclusive)
                            normalized = normalize_fx_history(fx_raw, inverse=inverse_flag)
                            fx_cache[cache_key] = (fx_raw, normalized)
                        fx_raw, fx = fx_cache[cache_key]
                        put(f"assets/{safe}/fx_{'inverse' if inverse_flag else 'direct'}_raw.parquet", parquet_bytes(fx_raw), "application/octet-stream")
                        state["fx_symbol"] = fx_ticker
                        break
                    except Exception as exc:
                        fx_error = str(exc)
                if fx is None:
                    state["status"] = "FX_DOWNLOAD_UNAVAILABLE"
                    state["fx_error"] = fx_error
                    continue
            candidate, returns, status = build_candidate(
                raw, asset_id, meta["currency"], fx=fx,
                inception="2024-11-29" if asset_id == "CRYPTO:HYPE-USD" else None)
            state["status"] = status
            if candidate is not None:
                state["source_placeholder_rows_excluded"] = int(len(raw) - len(candidate) - (int((raw["date"] < pd.Timestamp("2024-11-29")).sum()) if asset_id == "CRYPTO:HYPE-USD" else 0))
            if candidate is None:
                continue
            old = store.read_ohlcv_usd([asset_id], start=args.start, end=args.end_exclusive)
            state["comparison"] = compare_histories(old, candidate)
            state["coverage_review"] = coverage_review(old, candidate, state["comparison"])
            put(f"assets/{safe}/coverage_review.json", json.dumps(state["coverage_review"], indent=2), "application/json")
            put(f"assets/{safe}/candidate_ohlcv_usd.parquet", parquet_bytes(candidate), "application/octet-stream")
            put(f"assets/{safe}/candidate_returns_usd.parquet", parquet_bytes(returns), "application/octet-stream")
            put(f"assets/{safe}/comparison.json", json.dumps(state["comparison"], indent=2), "application/json")
        except Exception as exc:
            state["status"] = "ERROR"
            state["error"] = str(exc)
        finally:
            elapsed = max(time.monotonic() - started, 0.001)
            counts = Counter(x["status"] for x in results.values())
            progress = {"run_id": run_id, "status": "RUNNING",
                        "processed": completed, "total": total, "elapsed_seconds": round(elapsed, 1),
                        "as_of_utc": datetime.now(timezone.utc).isoformat(),
                        "status_counts": dict(counts), "last_asset_id": asset_id,
                        "last_asset_status": state["status"]}
            # Overwrite only the run-scoped progress pointer; never live history.
            s3.put_object(Bucket=args.bucket, Key=prefix+"/progress/latest.json",
                          Body=json.dumps(progress, default=str).encode(),
                          ContentType="application/json")
            print(f"[rebuild/source] {completed}/{total} ({completed/total:.1%}) "
                  f"{asset_id}: {state['status']} elapsed={elapsed:.0f}s "
                  f"success={counts.get('STAGED_FOR_REVIEW', 0)} "
                  f"exceptions={completed-counts.get('STAGED_FOR_REVIEW', 0)}", flush=True)
    summary = {"schema": "aea87_full_history_rebuild_v2_local_session_dates", "run_id": run_id,
               "status": "STAGING_ONLY", "source": "fresh_yahoo_daily",
               "start": args.start, "end_exclusive": args.end_exclusive,
               "active_histories_modified": False, "assets": results}
    put("summary.json", json.dumps(summary, indent=2, default=str), "application/json")
    put("manifest.json", json.dumps({"run_id": run_id, "objects": objects}, indent=2), "application/json")
    s3.put_object(Bucket=args.bucket, Key=prefix+"/progress/latest.json",
                  Body=json.dumps({"run_id": run_id, "status": "COMPLETED",
                                   "processed": total, "total": total,
                                   "status_counts": dict(Counter(v["status"] for v in results.values())),
                                   "elapsed_seconds": round(time.monotonic()-started, 1)}, default=str).encode(),
                  ContentType="application/json")
    print(json.dumps({"s3_prefix": f"s3://{args.bucket}/{prefix}/",
                      "statuses": {k: v["status"] for k, v in results.items()}}, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description="AEA-87 fresh full-history reconstruction pilot; S3 staging only")
    p.add_argument("--universe", default=str(paths.universe_dir() / "universe.csv"))
    p.add_argument("--asset-ids", default=",".join(DEFAULT_ASSETS))
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end-exclusive", default=datetime.now(timezone.utc).date().isoformat())
    args = p.parse_args()
    if args.end_exclusive <= args.start:
        p.error("end-exclusive must be after start")
    run(args)


if __name__ == "__main__":
    main()
