"""AEA-87 historical OHLCV/return repair engine.

Called only through ingest_market_data's explicit repair options. No market
writes in prepare mode. A per-asset commit can be recovered from its journal.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_edge.core.market_store import MarketStore


def _dates(frame):
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.normalize()
    if out["date"].isna().any():
        raise ValueError("Invalid dates")
    return out


def _parquet(data):
    return pd.read_parquet(io.BytesIO(data))


def _json(data):
    return json.loads(data.decode("utf-8"))


def _dump_json(doc):
    return json.dumps(doc, indent=2, sort_keys=True, default=str).encode("utf-8")


def _read_manifest(store, key):
    try:
        return store._get_bytes(key)
    except store._client().exceptions.NoSuchKey:
        return None


def _asset_partitions(store, table, aid):
    """Read only manifests' referenced parts; do not trust orphaned files."""
    prefix = f"{store.manifests_prefix}/{table}/asset_id={aid}/"
    keys = sorted(k for k in store._list_keys(prefix) if k.endswith("/manifest.json"))
    chunks = []
    manifests = {}
    table_prefix = store.ohlcv_prefix if table == "ohlcv_usd" else store.returns_prefix
    for key in keys:
        raw = store._get_bytes(key)
        man = _json(raw)
        parts = man.get("parts", [])
        if not parts:
            raise ValueError(f"{aid}: active manifest has no parts: {key}")
        for part in parts:
            expected = f"{table_prefix}/asset_id={aid}/year={man['year']}/"
            if not part.startswith(expected):
                raise ValueError(f"Manifest contains foreign part {part}")
            chunks.append(_parquet(store._get_bytes(part)))
        manifests[key] = raw
    if not chunks:
        return pd.DataFrame(), manifests
    frame = _dates(pd.concat(chunks, ignore_index=True))
    if frame.duplicated(["asset_id", "date"]).any():
        raise ValueError(f"{aid} {table}: duplicate active historical rows")
    return frame.sort_values("date"), manifests


def _merge_prices(downloaded, old, aid, ticker, yahoo, ccy, fx):
    fresh = _dates(downloaded)
    if fresh["date"].duplicated().any():
        raise ValueError("Yahoo returned duplicate dates")
    source_cols = ("open", "high", "low", "close", "adj_close", "volume")
    for col in source_cols:
        if col not in fresh:
            raise ValueError(f"Yahoo missing {col}")
        fresh[col] = pd.to_numeric(fresh[col], errors="coerce")
    if ccy == "USD":
        fresh["fx_to_usd"] = 1.0
    else:
        if ccy not in fx:
            raise ValueError(f"No approved FX for {ccy}")
        fresh["fx_to_usd"] = fx[ccy].reindex(fresh["date"]).to_numpy()
    fresh = fresh.loc[(fresh["close"] > 0) & (fresh["adj_close"] > 0)
                      & (fresh["fx_to_usd"] > 0)
                      & np.isfinite(fresh["close"]) & np.isfinite(fresh["adj_close"])
                      & np.isfinite(fresh["fx_to_usd"])].copy()
    if fresh.empty:
        raise ValueError("No usable Yahoo observations with approved FX")
    fresh["asset_id"] = aid
    fresh["ticker"] = ticker
    fresh["yahoo_ticker"] = yahoo
    fresh["currency"] = ccy
    fresh["close_raw_usd"] = fresh["close"] * fresh["fx_to_usd"]
    fresh["close_adjusted_usd"] = fresh["adj_close"] * fresh["fx_to_usd"]
    fresh["close_usd"] = fresh["close_raw_usd"]
    fresh["adj_close_usd"] = fresh["close_adjusted_usd"]
    columns = ["date", "asset_id", "ticker", "yahoo_ticker", "open", "high",
               "low", "close", "adj_close", "volume", "currency", "fx_to_usd",
               "close_raw_usd", "close_adjusted_usd", "close_usd", "adj_close_usd"]
    fresh = fresh[columns]
    old = _dates(old) if not old.empty else pd.DataFrame(columns=columns)
    if not old.empty and (old["asset_id"].astype(str) != aid).any():
        raise ValueError("Existing S3 data is associated with wrong asset ID")
    retained = old.loc[~old["date"].isin(fresh["date"])].copy()
    for col in columns:
        if col not in retained:
            retained[col] = pd.NA
    retained = retained[columns]
    if ccy != "USD" and not retained.empty and ccy in fx:
        rates = fx[ccy].reindex(retained["date"])
        ok = rates.notna().to_numpy()
        retained.loc[ok, "fx_to_usd"] = rates.to_numpy()[ok]
        for local, usd in (("close", "close_raw_usd"), ("adj_close", "close_adjusted_usd"),
                           ("close", "close_usd"), ("adj_close", "adj_close_usd")):
            retained.loc[ok, usd] = (
                pd.to_numeric(retained.loc[ok, local], errors="coerce")
                * pd.to_numeric(retained.loc[ok, "fx_to_usd"], errors="coerce"))
    combined = pd.concat([retained, fresh], ignore_index=True).sort_values("date")
    if combined.duplicated(["asset_id", "date"]).any():
        raise ValueError("Merged dataset contains duplicate dates")
    old_dates = set(old["date"]) if not old.empty else set()
    if not old_dates.issubset(set(combined["date"])):
        raise ValueError("Merged history lost stored dates")
    return combined, len(fresh), len(retained), set(retained["date"])


def _returns(prices, aid, ticker):
    px = prices[["date", "close_adjusted_usd"]].copy()
    px["close_adjusted_usd"] = pd.to_numeric(px["close_adjusted_usd"], errors="coerce")
    px = px.loc[px["close_adjusted_usd"].gt(0)].sort_values("date")
    log = np.log(px["close_adjusted_usd"] / px["close_adjusted_usd"].shift(1))
    out = pd.DataFrame({"date": px["date"], "asset_id": aid, "ticker": ticker,
                        "ret_log_close_adjusted_usd": log})
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_log_close_adjusted_usd"])
    out["ret_close_adjusted_usd"] = np.expm1(out["ret_log_close_adjusted_usd"])
    out["ret_adj_close_usd"] = out["ret_close_adjusted_usd"]
    return out


def _prepare(store, aid, prices, returns, old_manifests, root):
    records = []
    for table, frame in (("ohlcv_usd", prices), ("returns_usd", returns)):
        base = store.ohlcv_prefix if table == "ohlcv_usd" else store.returns_prefix
        previous = old_manifests[table]
        years = set(frame["date"].dt.year.astype(int))
        existing_years = {int(k.split("/year=")[1].split("/")[0]) for k in previous}
        if table == "ohlcv_usd" and not existing_years.issubset(years):
            raise ValueError("Merged history has lost existing OHLCV years")
        for year, group in frame.groupby(frame["date"].dt.year):
            year = int(year)
            key = store._manifest_key(table=table, asset_id=aid, year=year)
            blob = io.BytesIO()
            group.to_parquet(blob, index=False)
            data = blob.getvalue()
            record = {"table": table, "year": year, "manifest_key": key,
                      "original_manifest_sha256": (
                          hashlib.sha256(previous[key]).hexdigest() if key in previous else None),
                      "rows": len(group),
                      "dates": sorted(group["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "new_part": store._part_key(base, aid, year)}
            stage_key = (
                f"{store.base_prefix}/repairs/aea87/staging/{root.name}/"
                f"asset_id={aid}/{table}/year={year}/part.parquet"
            )
            if store._key_exists(stage_key):
                if hashlib.sha256(store._get_bytes(stage_key)).hexdigest() != record["sha256"]:
                    raise FileExistsError(f"Staging key already has different content: {stage_key}")
            else:
                store._put_bytes(stage_key, data)
                if hashlib.sha256(store._get_bytes(stage_key)).hexdigest() != record["sha256"]:
                    raise IOError(f"S3 staged partition checksum mismatch: {stage_key}")
            record["staged_part"] = stage_key
            records.append(record)
    return records


def _promote(store, aid, records, root, run_id):
    """Prestage parts and originals, then switch manifests, rollback on error."""
    prefix = f"{store.base_prefix}/repairs/aea87/{run_id}/asset_id={aid}"
    journal_key = f"{prefix}/journal.json"
    if store._key_exists(journal_key):
        raise ValueError(f"Already staged; recover or resume asset {aid} instead of writing twice")
    originals = {}
    for item in records:
        key = item["manifest_key"]
        originals[key] = _read_manifest(store, key)
        actual_sha = hashlib.sha256(originals[key]).hexdigest() if originals[key] is not None else None
        if actual_sha != item["original_manifest_sha256"]:
            raise RuntimeError(f"Concurrent manifest change: {key}")
    journal = {"asset_id": aid, "run_id": run_id, "state": "PREPARING",
               "records": records, "backups": {}, "committed": []}
    store._put_bytes(journal_key, _dump_json(journal), "application/json")
    for item in records:
        key = item["manifest_key"]
        if originals[key] is not None:
            backup = f"{prefix}/backups/{item['table']}_{item['year']}.json"
            store._put_bytes(backup, originals[key], "application/json")
            if store._get_bytes(backup) != originals[key]:
                raise IOError(f"Manifest backup failed: {key}")
            journal["backups"][key] = backup
        data = store._get_bytes(item["staged_part"])
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"S3 staged partition changed: {key}")
        store._client().copy_object(
            Bucket=store.bucket,
            CopySource={"Bucket": store.bucket, "Key": item["staged_part"]},
            Key=item["new_part"],
        )
        if hashlib.sha256(store._get_bytes(item["new_part"])).hexdigest() != item["sha256"]:
            raise IOError(f"Promoted part checksum mismatch: {key}")
    journal["state"] = "READY"
    store._put_bytes(journal_key, _dump_json(journal), "application/json")
    try:
        for item in records:
            key = item["manifest_key"]
            if _read_manifest(store, key) != originals[key]:
                raise RuntimeError(f"Concurrent manifest change: {key}")
            updated = {"asset_id": aid, "year": item["year"],
                       "dates": item["dates"], "parts": [item["new_part"]],
                       "repair_run_id": run_id}
            store._put_bytes(key, _dump_json(updated), "application/json")
            journal["committed"].append(key)
            journal["state"] = "COMMITTING"
            store._put_bytes(journal_key, _dump_json(journal), "application/json")
        journal["state"] = "COMMITTED"
        store._put_bytes(journal_key, _dump_json(journal), "application/json")
    except Exception:
        journal["state"] = "ROLLING_BACK"
        store._put_bytes(journal_key, _dump_json(journal), "application/json")
        failed = []
        # Restore every original, including a potentially switched manifest
        # whose journal update may have failed.
        for item in reversed(records):
            key = item["manifest_key"]
            try:
                if originals[key] is None:
                    store._client().delete_object(Bucket=store.bucket, Key=key)
                else:
                    store._put_bytes(key, originals[key], "application/json")
            except Exception as exc:
                failed.append(f"{key}: {exc}")
        journal["state"] = "ROLLBACK_FAILED" if failed else "ROLLED_BACK"
        journal["rollback_errors"] = failed
        store._put_bytes(journal_key, _dump_json(journal), "application/json")
        raise


def _repair_assets_from_universe(csv, selected):
    u = pd.read_csv(csv)
    if "asset_id" not in u:
        raise ValueError("Universe CSV missing asset_id")
    u["asset_id"] = u["asset_id"].astype(str).str.strip()
    u = u[u["asset_id"].isin(selected)]
    if u["asset_id"].duplicated().any():
        raise ValueError("Duplicate asset_id: mapping is ambiguous")
    return u


def run(*, cfg, bad_cells, fx_parquet, universe_path, start, end,
        max_assets, confirm_prod_write, mode="prepare", run_id=None):
    from alpha_edge.market.ingest_market_data import (
        download_ohlcv, _clean_ccy, _normalize_crypto_symbol_to_yahoo,
        _normalize_fx_symbol_to_yahoo, _is_fx_pair_like_symbol, _make_market_store,
        _cfg_bucket, _cfg_region, _cfg_market_root,
    )

    if mode not in ("prepare", "promote"):
        raise ValueError("Repair mode must be prepare or promote")
    if mode == "promote":
        from alpha_edge.core.runtime import require_prod_confirmation
        require_prod_confirmation(cfg, bool(confirm_prod_write))
        if not confirm_prod_write or str(getattr(cfg, "env", "")) != "prod":
            raise ValueError("Production promotion requires --env prod --confirm-prod-write")
    if mode == "promote" and not run_id:
        raise ValueError("--repair-run-id is required to promote the same prepared campaign")
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,80}", run_id):
        raise ValueError("Invalid repair run ID")
    root = Path("../data/quality/returns/aea87_campaign") / run_id
    # Promotion consumes the exact local immutable preparation manifest.
    # Never re-download Yahoo data during publication.
    if mode == "promote":
        store = _make_market_store(bucket=_cfg_bucket(cfg), region=_cfg_region(cfg),
                                   market_root=_cfg_market_root(cfg))
        campaign_key = f"{store.base_prefix}/repairs/aea87/staging/{run_id}/prepared_campaign.json"
        campaign = _json(store._get_bytes(campaign_key))
        if campaign.get("run_id") != run_id:
            raise ValueError("Prepared S3 campaign run ID mismatch")
        report = list(campaign["report"])
        for item in campaign["prepared"]:
            aid = item["asset_id"]
            if item["flagged_retained"]:
                continue
            try:
                _promote(store, aid, item["records"], root, run_id)
                status, error = "COMMITTED", ""
            except Exception as exc:
                status, error = "FAILED_OR_ROLLED_BACK", str(exc)
            for row in report:
                if row["asset_id"] == aid:
                    row.update(status=status, error=error)
            report_key = f"{store.base_prefix}/repairs/aea87/staging/{run_id}/campaign_report.json"
            store._put_bytes(report_key, _dump_json(report), "application/json")
        print(f"Promotion result: {pd.Series([x['status'] for x in report]).value_counts().to_dict()}")
        if any(x["status"] != "COMMITTED" for x in report):
            print("PARTIAL_CAMPAIGN: trusted cache publication deferred")
            return
        cmd = [sys.executable, "-m", "alpha_edge.market.build_returns_wide_cache",
               "--env", "prod", "--confirm-prod-write", "--mode", "full", "--force"]
        subprocess.run(cmd, check=True)
        print(f"COMMITTED_AND_CACHE_REBUILT; report={f's3://{store.bucket}/{report_key}'}")
        return
    bad = pd.read_csv(bad_cells)
    if not {"asset_id", "date"}.issubset(bad.columns):
        raise ValueError("Anomaly CSV needs asset_id and date")
    bad["asset_id"] = bad["asset_id"].astype(str).str.strip()
    bad["date"] = pd.to_datetime(bad["date"], utc=True, errors="coerce").dt.normalize()
    selected = set(bad["asset_id"].dropna())
    u = _repair_assets_from_universe(universe_path, selected)
    unmapped = sorted(selected - set(u["asset_id"]))
    if max_assets:
        u = u.head(max_assets)
    if not str(fx_parquet).startswith("s3://"):
        raise ValueError("AEA-87 requires an S3 FX snapshot URI; local FX Parquet paths are not accepted")
    from urllib.parse import urlparse
    store = _make_market_store(bucket=_cfg_bucket(cfg), region=_cfg_region(cfg),
                               market_root=_cfg_market_root(cfg))
    fx_uri = urlparse(str(fx_parquet))
    fx_bucket, fx_key = fx_uri.netloc, fx_uri.path.lstrip("/")
    if fx_bucket != store.bucket:
        raise ValueError("FX snapshot must be in the configured market S3 bucket")
    if not fx_key.endswith("/fx_history_all.parquet"):
        raise ValueError("FX URI must refer to a versioned fx_history_all.parquet")
    manifest_key = fx_key.rsplit("/", 1)[0] + "/manifest.json"
    fx_manifest = _json(store._get_bytes(manifest_key))
    if fx_manifest.get("status") != "STAGING_ONLY":
        raise ValueError("Unrecognized FX snapshot manifest status")
    fx_payload = store._get_bytes(fx_key)
    if hashlib.sha256(fx_payload).hexdigest() != fx_manifest.get("sha256"):
        raise ValueError("FX snapshot S3 checksum differs from published manifest")
    fx_df = _dates(_parquet(fx_payload))
    allowed = {"VALID", "VALID_AFTER_GAP"}
    if not set(fx_df["validation_status"]).issubset(allowed):
        raise ValueError("Unapproved FX records present")
    fx_df["currency"] = fx_df["currency"].astype(str).str.upper()
    if fx_df.duplicated(["currency", "date"]).any():
        raise ValueError("Duplicate FX date/currency")
    fx = {ccy: g.set_index("date")["fx_to_usd"] for ccy, g in fx_df.groupby("currency")}
    report = [{"asset_id": aid, "status": "UNMAPPED"} for aid in unmapped]
    prepared = []
    for _, row in u.iterrows():
        aid = row["asset_id"]
        sym = _normalize_crypto_symbol_to_yahoo(_normalize_fx_symbol_to_yahoo(str(
            row.get("yahoo_ticker") or row.get("ticker") or "").upper()))
        ccy = _clean_ccy(row.get("currency"))
        if aid.startswith("CRYPTO:") or _is_fx_pair_like_symbol(sym):
            ccy = "USD"
        try:
            if not sym or not ccy:
                raise ValueError("Missing source symbol or currency")
            old_prices, op = _asset_partitions(store, "ohlcv_usd", aid)
            _, or_ = _asset_partitions(store, "returns_usd", aid)
            downloaded = download_ohlcv(sym, start=start, end=end, allow_cache=False)
            if downloaded.empty:
                raise ValueError("Source history unavailable")
            prices, refreshed, retained, retained_dates = _merge_prices(
                downloaded, old_prices, aid, str(row.get("ticker") or sym), sym, ccy, fx)
            returns = _returns(prices, aid, str(row.get("ticker") or sym))
            flagged_preserved = int(bad.loc[bad["asset_id"].eq(aid), "date"].isin(retained_dates).sum())
            records = _prepare(store, aid, prices, returns,
                               {"ohlcv_usd": op, "returns_usd": or_}, root)
            prepared.append((aid, records, flagged_preserved))
            report.append({"asset_id": aid, "status": "PREPARED", "fresh_rows": refreshed,
                           "retained_rows": retained, "flagged_retained": flagged_preserved,
                           "partitions": len(records)})
        except Exception as exc:
            report.append({"asset_id": aid, "status": "SOURCE_UNAVAILABLE_OR_FAILED",
                           "error": str(exc)})
        report_key = f"{store.base_prefix}/repairs/aea87/staging/{run_id}/campaign_report.json"
    store._put_bytes(report_key, _dump_json(report), "application/json")
    if mode == "prepare":
        manifest = {"run_id": run_id, "report": report, "prepared": [
            {"asset_id": aid, "records": records, "flagged_retained": flagged}
            for aid, records, flagged in prepared
        ]}
        campaign_key = f"{store.base_prefix}/repairs/aea87/staging/{run_id}/prepared_campaign.json"
        if store._key_exists(campaign_key):
            raise FileExistsError(f"Campaign manifest already exists: {campaign_key}")
        store._put_bytes(campaign_key, _dump_json(manifest), "application/json")
        if _json(store._get_bytes(campaign_key)) != manifest:
            raise IOError("S3 campaign manifest readback mismatch")
        print(f"PREPARED_TO_S3 {len(prepared)} assets; {len(unmapped)} unmapped; manifest=s3://{store.bucket}/{campaign_key}")
        return
