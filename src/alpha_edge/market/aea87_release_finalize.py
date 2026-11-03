"""AEA-87 post-cutover refresh. Must run during exclusive maintenance.

Refresh affected assets in the four price snapshots and latest returns, then
rebuild the historical returns-wide cache. All prior snapshot bytes are backed up.
"""
import io
import json
import hashlib
import numpy as np

import pandas as pd

from alpha_edge.market.aea87_publication_transaction import fetch, put_once, checksum, write_event, entries

SNAPSHOTS = {
    "latest_prices": "ohlcv_usd",
    "latest_prices_raw": "ohlcv_usd",
    "latest_prices_adjusted": "ohlcv_usd",
    "latest_returns": "returns_usd",
}


def _frame(data):
    if data is None:
        return pd.DataFrame()
    return pd.read_parquet(io.BytesIO(data))


def _last_for_asset(s3, bucket, ops, asset_id, table):
    matches = [x for x in ops if x["asset_id"] == asset_id
               and x["staged_key"] and ("/" + table + "/") in x["manifest_key"]]
    if not matches:
        return None
    # Each asset/year has a verified staged replacement. Only the newest
    # partition can contain its latest observation, so avoid rereading 30+
    # years of history for each snapshot.
    latest = max(matches, key=lambda op: int(
        op["manifest_key"].split("/year=")[-1].split("/")[0]))
    part = fetch(s3, bucket, latest["part_key"])
    if checksum(part) != latest["staged_sha"]:
        raise ValueError("Published partition content mismatch: " + latest["part_key"])
    frame = _frame(part)
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"])
    if frame["asset_id"].astype(str).ne(asset_id).any():
        raise ValueError("Unexpected asset identity")
    return frame.sort_values("date").iloc[-1].to_dict()


def refresh_repaired_returns_cache(s3, bucket, root, ops, asset_ids):
    """Replace only repaired asset columns in the existing canonical wide cache.

    Full-universe cache reconstruction is not safe while other assets remain
    flagged by AEA87. This incremental *full-history per asset* replacement
    preserves untouched columns, removes stale repaired dates, and never
    fills source gaps or manufactures returns.
    """
    cache_key = "market/cache/v1/returns_wide_min5y.parquet"
    meta_key = "market/cache/v1/returns_wide_min5y.meta.json"
    prior = fetch(s3, bucket, cache_key)
    if prior is None:
        raise RuntimeError("No existing canonical returns-wide cache to update")
    wide = _frame(prior)
    if "date" in wide.columns:
        wide["date"] = pd.to_datetime(wide["date"], utc=True, errors="coerce")
        wide = wide.set_index("date")
    else:
        wide.index = pd.to_datetime(wide.index, utc=True, errors="coerce")
    if wide.index.isna().any() or wide.index.duplicated().any():
        raise ValueError("Existing returns-wide index is invalid")
    wide.index = wide.index.tz_convert(None).normalize()
    if wide.index.duplicated().any():
        raise ValueError("Existing returns-wide dates duplicate after normalization")
    if wide.columns.duplicated().any():
        raise ValueError("Existing returns-wide contains duplicate asset columns")
    original_cols = set(wide.columns.astype(str))
    refreshed, excluded = {}, []
    for aid in asset_ids:
        relevant = [op for op in ops if op["asset_id"] == aid and
                    "/returns_usd/" in op["manifest_key"] and op.get("part_key")]
        if not relevant:
            raise RuntimeError("Missing freshly published return partitions: "+aid)
        rows = []
        for op in relevant:
            raw = fetch(s3, bucket, op["part_key"])
            if checksum(raw) != op["staged_sha"]:
                raise RuntimeError("Repaired return partition changed: "+op["part_key"])
            rows.append(_frame(raw))
        frame = pd.concat(rows, ignore_index=True)
        if frame["asset_id"].astype(str).ne(aid).any():
            raise ValueError("Mixed identities in repaired return parts")
        dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        if dates.isna().any() or dates.duplicated().any():
            raise ValueError("Repaired returns contain missing or duplicate dates")
        if "ret_log_close_adjusted_usd" in frame:
            log_value = pd.to_numeric(frame["ret_log_close_adjusted_usd"], errors="coerce")
            values = pd.Series(np.expm1(log_value.to_numpy()), index=dates.dt.tz_convert(None).dt.normalize())
        else:
            simple = pd.to_numeric(frame["ret_close_adjusted_usd"], errors="coerce")
            values = pd.Series(simple.to_numpy(), index=dates.dt.tz_convert(None).dt.normalize())
        if (values.dropna() < -1).any() or not np.isfinite(values.dropna()).all():
            raise ValueError("Invalid repaired simple returns: "+aid)
        values = values.dropna().sort_index()
        # Respect the existing cache's five-year/1260-obs eligibility policy.
        eligible = (len(values) >= 1260 and not values.empty and
                    (values.index.max() - values.index.min()).days >= 5*365)
        if eligible:
            refreshed[aid] = values
        else:
            excluded.append(aid)
        if aid in wide:
            wide = wide.drop(columns=[aid])
    if refreshed:
        combined_index = wide.index
        for series in refreshed.values():
            combined_index = combined_index.union(series.index)
        wide = wide.reindex(combined_index).sort_index()
        for aid, values in refreshed.items():
            wide[aid] = values.reindex(wide.index).astype("float32")
    if wide.empty or wide.shape[1] == 0:
        raise RuntimeError("Refresh would leave canonical cache empty")
    old_meta = fetch(s3, bucket, meta_key)
    meta = json.loads(old_meta) if old_meta else {}
    meta.update({"n_days": int(len(wide)), "n_assets": int(len(wide.columns)),
                 "first_date": str(wide.index.min().date()),
                 "last_date": str(wide.index.max().date()),
                 "as_of_utc": pd.Timestamp.utcnow().isoformat(),
                 "aea87_incremental_repaired_full_history": {
                     "run_root": root, "repaired_assets": sorted(refreshed),
                     "excluded_below_5y": sorted(excluded),
                     "unchanged_assets_not_revalidated": True}})
    out = io.BytesIO()
    wide.sort_index().to_parquet(out, index=True)
    s3.put_object(Bucket=bucket, Key=cache_key, Body=out.getvalue(),
                  ContentType="application/octet-stream")
    s3.put_object(Bucket=bucket, Key=meta_key,
                  Body=json.dumps(meta, indent=2, default=str).encode(),
                  ContentType="application/json")
    # Prove that repaired values (including NaNs) actually landed.
    observed = _frame(fetch(s3, bucket, cache_key))
    if "date" in observed.columns:
        observed = observed.set_index("date")
    observed.index = pd.to_datetime(observed.index, utc=True).tz_convert(None).normalize()
    for aid, source in refreshed.items():
        if aid not in observed:
            raise RuntimeError("Repaired asset missing from published cache: "+aid)
        actual = observed[aid].reindex(source.index)
        if not np.allclose(actual.to_numpy(dtype=float), source.to_numpy(dtype=float),
                           rtol=1e-6, atol=1e-7, equal_nan=True):
            raise RuntimeError("Repaired returns-wide values differ from source: "+aid)
    for aid in excluded:
        if aid in observed:
            raise RuntimeError("Stale short-history asset remained in returns-wide cache: "+aid)
    return {"refreshed":len(refreshed), "excluded_below_5y":excluded,
            "unchanged_columns":len(original_cols-set(asset_ids))}


def finalize(s3, bucket, root, publication):
    history = entries(s3, bucket, root)
    if not history or history[-1]["state"] not in ("CACHE_REBUILD_REQUIRED", "SNAPSHOTS_PREPARED"):
        raise RuntimeError("Expected switched manifests with cache rebuild pending")
    ops = publication["operations"]
    if any(checksum(fetch(s3, bucket, op["manifest_key"])) != op["after_sha"] for op in ops):
        raise RuntimeError("Manifest verification failed; do not rebuild")
    candidates = set(publication["candidates"])
    for name, table in SNAPSHOTS.items():
        key = "market/snapshots/v1/" + name + ".parquet"
        old = fetch(s3, bucket, key)
        backup_key = root + "/snapshot_backups/" + name + ".parquet"
        if old is not None:
            existing_backup = fetch(s3, bucket, backup_key)
            if existing_backup is None:
                put_once(s3, bucket, backup_key, old)
            elif checksum(existing_backup) != checksum(old):
                raise RuntimeError("Live snapshot changed since backup: " + name)
        original = _frame(old)
        if not original.empty and "asset_id" not in original:
            raise ValueError("Snapshot has no asset_id: " + name)
        retained = original.loc[~original["asset_id"].astype(str).isin(candidates)].copy() if not original.empty else original.copy()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as pool:
            rows = pool.map(lambda aid: _last_for_asset(s3, bucket, ops, aid, table), publication["candidates"])
            updated = [row for row in rows if row is not None]
        print(f"[finalize] prepared {name}: {len(updated):,} affected asset rows", flush=True)
        newer = pd.DataFrame(updated)
        if not newer.empty:
            # Only preserve columns compatible with the destination snapshot.
            # Missing rows are not manufactured from an unrelated schema.
            if not original.empty:
                common = set(original.columns).intersection(newer.columns)
                if "asset_id" not in common or "date" not in common:
                    raise ValueError("Incompatible snapshot schema: " + name)
                required_columns = {
                    "latest_prices_raw": "close_raw_usd",
                    "latest_prices_adjusted": "close_adjusted_usd",
                    "latest_returns": "ret_adj_close_usd",
                    "latest_prices": "close_adjusted_usd",
                }
                required = {required_columns[name]}
                if not required.issubset(set(newer.columns)):
                    raise ValueError("Required snapshot values missing: " + name)
                newer = newer.reindex(columns=original.columns)
            retained = pd.concat([retained, newer], ignore_index=True)
        if retained.empty:
            raise RuntimeError("Empty rebuilt snapshot " + name)
        retained["date"] = pd.to_datetime(retained["date"], utc=True, errors="coerce")
        if retained["date"].isna().any():
            raise ValueError("Snapshot contains invalid dates: " + name)
        retained = retained.sort_values("date").drop_duplicates("asset_id", keep="last")
        output = io.BytesIO()
        retained.to_parquet(output, index=False)
        # Retain immutable prepared replacement for post-crash recovery.
        prepared_key = root + "/prepared_snapshots/" + name + ".parquet"
        existing = fetch(s3, bucket, prepared_key)
        if existing is None:
            put_once(s3, bucket, prepared_key, output.getvalue())
        elif checksum(existing) != checksum(output.getvalue()):
            raise RuntimeError("Prepared snapshot differs on retry: " + name)
    if history[-1]["state"] == "CACHE_REBUILD_REQUIRED":
        write_event(s3, bucket, root, history[-1]["step"] + 1, "SNAPSHOTS_PREPARED")
    # Back up both canonical wide-cache files before invoking the rebuild.
    for suffix in (".parquet", ".meta.json"):
        live_key = "market/cache/v1/returns_wide_min5y" + suffix
        previous = fetch(s3, bucket, live_key)
        if previous is not None:
            backup_key = root + "/cache_backups/returns_wide_min5y" + suffix
            existing_backup = fetch(s3, bucket, backup_key)
            if existing_backup is None:
                put_once(s3, bucket, backup_key, previous)
            elif checksum(existing_backup) != checksum(previous):
                raise RuntimeError("Wide cache changed since original backup: " + live_key)
    # Internal full-history replacement only for successfully repaired assets.
    cache_result = refresh_repaired_returns_cache(s3, bucket, root, ops, candidates)
    print(f"[finalize] affected returns-wide columns refreshed: {cache_result}", flush=True)
    for name in SNAPSHOTS:
        key = "market/snapshots/v1/" + name + ".parquet"
        blob = fetch(s3, bucket, root + "/prepared_snapshots/" + name + ".parquet")
        if blob is None:
            raise RuntimeError("Prepared snapshot missing: " + name)
        s3.put_object(Bucket=bucket, Key=key, Body=blob,
                      ContentType="application/octet-stream")
        if checksum(fetch(s3, bucket, key)) != checksum(blob):
            raise RuntimeError("Snapshot publication failed: " + key)
    history = entries(s3, bucket, root)
    write_event(s3, bucket, root, history[-1]["step"] + 1, "FINALIZED")
    return {"status": "FINALIZED", "snapshots": len(SNAPSHOTS),
            "assets": len(candidates), "cache_rebuilt": True}
