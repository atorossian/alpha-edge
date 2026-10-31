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
    data = []
    for op in matches:
        part = fetch(s3, bucket, op["part_key"])
        if checksum(part) != op["staged_sha"]:
            raise ValueError("Published partition content mismatch: " + op["part_key"])
        frame = _frame(part)
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["date"])
        if frame["asset_id"].astype(str).ne(asset_id).any():
            raise ValueError("Unexpected asset identity")
        data.append(frame.sort_values("date").iloc[-1])
    if not data:
        return None
    return max(data, key=lambda row: row["date"]).to_dict()


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
        retained = original.loc[~original["asset_id"].astype(str).isin(candidates)].copy() if not original.empty else original
        updated = []
        for aid in publication["candidates"]:
            row = _last_for_asset(s3, bucket, ops, aid, table)
            if row is not None:
                updated.append(row)
        newer = pd.DataFrame(updated)
        if not newer.empty:
            # Only preserve columns compatible with the destination snapshot.
            # Missing rows are not manufactured from an unrelated schema.
            if not original.empty:
                common = set(original.columns).intersection(newer.columns)
                if "asset_id" not in common or "date" not in common:
                    raise ValueError("Incompatible snapshot schema: " + name)
                newer = newer.reindex(columns=original.columns)
            retained = pd.concat([retained, newer], ignore_index=True)
        if retained.empty:
            raise RuntimeError("Empty rebuilt snapshot " + name)
        retained["date"] = pd.to_datetime(retained["date"], utc=True, errors="coerce")
        retained = retained.sort_values("date").drop_duplicates("asset_id", keep="last")
        output = io.BytesIO()
        retained.to_parquet(output, index=False)
        # Retain immutable prepared replacement for post-crash recovery.
        put_once(s3, bucket, root + "/prepared_snapshots/" + name + ".parquet", output.getvalue())
    write_event(s3, bucket, root, history[-1]["step"] + 1, "SNAPSHOTS_PREPARED")
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
