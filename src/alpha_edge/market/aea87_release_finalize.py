"""AEA-87 post-cutover refresh. Must run during exclusive maintenance.

Refresh affected assets in the four price snapshots and latest returns, then
rebuild the historical returns-wide cache. All prior snapshot bytes are backed up.
"""
import io
import json
import subprocess
import sys
import hashlib

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


def finalize(s3, bucket, root, publication):
    history = entries(s3, bucket, root)
    if not history or history[-1]["state"] != "CACHE_REBUILD_REQUIRED":
        raise RuntimeError("Manifest switch not verified or release already finalized")
    ops = publication["operations"]
    if any(checksum(fetch(s3, bucket, op["manifest_key"])) != op["after_sha"] for op in ops):
        raise RuntimeError("Manifest verification failed; do not rebuild")
    candidates = set(publication["candidates"])
    for name, table in SNAPSHOTS.items():
        key = "market/snapshots/v1/" + name + ".parquet"
        old = fetch(s3, bucket, key)
        backup_key = root + "/snapshot_backups/" + name + ".parquet"
        if old is not None:
            put_once(s3, bucket, backup_key, old)
        original = _frame(old)
        if not original.empty and "asset_id" not in original:
            raise ValueError("Snapshot has no asset_id: " + name)
        retained = original.copy()
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
                required = {"close_raw_usd"} if name == "latest_prices_raw" else ({"close_adjusted_usd"} if name == "latest_prices_adjusted" else ({"ret_adj_close_usd"} if name == "latest_returns" else {"close_adjusted_usd"})
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
        put_once(s3, bucket, root + "/prepared_snapshots/" + name + ".parquet", output.getvalue())
    write_event(s3, bucket, root, history[-1]["step"] + 1, "SNAPSHOTS_PREPARED")
    # Back up both canonical wide-cache files before invoking the rebuild.
    for suffix in (".parquet", ".meta.json"):
        live_key = "market/cache/v1/returns_wide_min5y" + suffix
        previous = fetch(s3, bucket, live_key)
        if previous is not None:
            put_once(s3, bucket, root + "/cache_backups/returns_wide_min5y" + suffix, previous)
    # Cache build is done with all readers stopped.
    subprocess.run([sys.executable, "-m", "alpha_edge.market.build_returns_wide_cache",
                    "--env", "prod", "--confirm-prod-write", "--force"], check=True)
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
