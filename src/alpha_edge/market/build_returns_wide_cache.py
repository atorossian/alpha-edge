# build_returns_wide_cache.py
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from alpha_edge.core.market_store import MarketStore
from alpha_edge import paths


@dataclass
class CacheConfig:
    bucket: str = "alpha-edge-algo"
    cache_prefix: str = "market/cache/v1"   # S3 KEY prefix (no s3://)
    min_years: float = 5.0
    start: str = "2010-01-01"
    end: Optional[str] = None
    min_obs: int = 252 * 5
    dtype: str = "float32"
    force: bool = False
    progress_every: int = 100

    # speed knobs
    max_workers: int = 8
    s3_max_concurrency: int = 8
    strict_window: bool = True

    # NEW: universe filtering (fixes "assets_discovered > universe")
    universe_csv: str = (paths.universe_dir() / "universe.csv").as_posix()
    excluded_csv: str = (paths.universe_dir() / "asset_excluded.csv").as_posix()
    filter_to_active_universe: bool = True  # set False to build for all discovered assets


def _safe_date(s: str | None) -> pd.Timestamp | None:
    if not s:
        return None
    x = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.isna(x):
        return None
    return x.tz_convert(None).normalize()


def _to_day_naive(x) -> pd.Timestamp:
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {x!r}")
    ts = pd.Timestamp(ts)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _put_json_s3(store: MarketStore, key: str, payload: dict) -> None:
    store._put_bytes(
        key,
        json.dumps(payload, indent=2, default=str).encode("utf-8"),
        content_type="application/json",
    )


def _get_json_s3(store: MarketStore, key: str) -> dict:
    try:
        return json.loads(store._get_bytes(key).decode("utf-8"))
    except Exception:
        return {}


def _load_active_asset_ids_from_universe(
    *,
    universe_csv: str,
    excluded_csv: str | None = None,
) -> set[str]:
    """
    Active assets = universe include=1 minus excluded asset_ids (if present).
    """
    u = pd.read_csv(universe_csv)
    u = u[u.get("include", 1).fillna(1).astype(int) == 1].copy()
    if "asset_id" not in u.columns:
        raise RuntimeError("Universe CSV must include 'asset_id' column")
    active = set(u["asset_id"].astype(str).str.strip().tolist())

    if excluded_csv:
        try:
            ex = pd.read_csv(excluded_csv)
            if "asset_id" in ex.columns:
                bad = set(ex["asset_id"].astype(str).str.strip().tolist())
                active -= bad
        except Exception:
            # excluded file is optional; ignore read errors
            pass

    return active


def build_returns_wide_cache(cfg: CacheConfig) -> None:
    store = MarketStore(bucket=cfg.bucket)

    start_ts = _to_day_naive(cfg.start)
    end_ts = _to_day_naive(cfg.end) if cfg.end else _to_day_naive(pd.Timestamp.utcnow())

    years = list(range(int(start_ts.year), int(end_ts.year) + 1))

    # ---------- staleness check ----------
    meta_key = f"{cfg.cache_prefix}/returns_wide_min{int(cfg.min_years)}y.meta.json"
    cache_key = f"{cfg.cache_prefix}/returns_wide_min{int(cfg.min_years)}y.parquet"

    if not cfg.force:
        latest_state = store.read_returns_latest_state()
        latest_date = _safe_date(latest_state.get("last_date"))

        meta = _get_json_s3(store, meta_key)
        cache_last = _safe_date(meta.get("last_date"))

        if latest_date is not None and cache_last is not None and cache_last >= latest_date:
            print(
                f"[SKIP] returns_wide cache up-to-date: cache_last={cache_last.date()} "
                f"latest_returns={latest_date.date()}"
            )
            return

    # ---------- discover asset_ids by listing returns partitions ----------
    root = store.returns_prefix  # e.g. market/returns_usd/v1
    print("[INFO] returns_root:", root)
    root_prefix = f"{root}/"

    keys = store._list_keys(root_prefix)
    if not keys:
        raise RuntimeError(f"No objects found under s3://{store.bucket}/{root_prefix}")

    asset_ids_all = set()
    for k in keys:
        if "/asset_id=" not in k or "/year=" not in k or not k.endswith(".parquet"):
            continue
        try:
            after = k.split("/asset_id=", 1)[1]
            aid = after.split("/", 1)[0].strip()
            if aid:
                asset_ids_all.add(aid)
        except Exception:
            continue

    asset_ids_all = sorted(asset_ids_all)
    if not asset_ids_all:
        raise RuntimeError("Could not discover any asset_id partitions under returns_root.")

    # ---------- optionally filter to current active universe ----------
    active = None
    asset_ids = asset_ids_all

    if cfg.filter_to_active_universe:
        active = _load_active_asset_ids_from_universe(
            universe_csv=cfg.universe_csv,
            excluded_csv=cfg.excluded_csv,
        )
        asset_ids = sorted(set(asset_ids_all).intersection(active))

    print(
        f"[INFO] assets_discovered_all={len(asset_ids_all)} "
        f"active_from_universe={(len(active) if active is not None else 'NA')} "
        f"assets_used={len(asset_ids)} "
        f"years={years[0]}..{years[-1]} window={start_ts.date()}..{end_ts.date()} "
        f"workers={cfg.max_workers} strict_window={cfg.strict_window}"
    )

    if not asset_ids:
        raise RuntimeError("No assets selected for returns_wide build after filtering (assets_used=0).")

    # ---------- wide build ----------
    s3_sem = threading.Semaphore(int(cfg.s3_max_concurrency))

    def _read_one_asset(asset_id: str) -> dict:
        """
        Returns:
          {"status":"keep", "asset_id":..., "series": pd.Series, "nobs":..., "span_days":...}
          or {"status":"drop_span"/"drop_obs"/"empty"/"fail", ...}
        """
        try:
            with s3_sem:
                df = store.read_returns_usd(
                    asset_ids=[asset_id],
                    start=str(start_ts.date()),
                    end=str(end_ts.date()),
                    columns=["date", "asset_id", "ret_adj_close_usd"],
                )

            if df is None or df.empty:
                return {"status": "empty", "asset_id": asset_id}

            # normalize dates
            d = pd.to_datetime(df["date"], errors="coerce", utc=True)
            d = d.dt.tz_convert(None).dt.normalize()

            r = pd.to_numeric(df["ret_adj_close_usd"], errors="coerce")
            tmp = pd.DataFrame({"date": d, "ret": r}).dropna(subset=["date", "ret"])
            if tmp.empty:
                return {"status": "empty", "asset_id": asset_id}

            if cfg.strict_window:
                tmp = tmp[(tmp["date"] >= start_ts) & (tmp["date"] <= end_ts)]
                if tmp.empty:
                    return {"status": "empty", "asset_id": asset_id}

            # dedupe by date
            tmp = tmp.sort_values("date").drop_duplicates(subset=["date"], keep="last")

            nobs = int(tmp.shape[0])
            first = tmp["date"].min()
            last = tmp["date"].max()
            span_days = int((last - first).days) if pd.notna(first) and pd.notna(last) else 0

            if span_days < int(cfg.min_years * 365):
                return {"status": "drop_span", "asset_id": asset_id, "span_days": span_days, "nobs": nobs}

            if nobs < int(cfg.min_obs):
                return {"status": "drop_obs", "asset_id": asset_id, "span_days": span_days, "nobs": nobs}

            s = tmp.set_index("date")["ret"].astype(cfg.dtype)
            s.name = str(asset_id)

            return {"status": "keep", "asset_id": asset_id, "series": s, "span_days": span_days, "nobs": nobs}

        except Exception as e:
            return {"status": "fail", "asset_id": asset_id, "error": str(e)[:800]}

    kept_ids: list[str] = []
    series_list: list[pd.Series] = []

    dropped_span = 0
    dropped_obs = 0
    empty_assets = 0
    failed_assets = 0

    # Run per-asset reads in parallel
    with ThreadPoolExecutor(max_workers=int(cfg.max_workers)) as ex:
        futs = [ex.submit(_read_one_asset, aid) for aid in asset_ids]

        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            done += 1

            st = res.get("status")
            if st == "keep":
                kept_ids.append(str(res["asset_id"]))
                series_list.append(res["series"])
            elif st == "drop_span":
                dropped_span += 1
            elif st == "drop_obs":
                dropped_obs += 1
            elif st == "empty":
                empty_assets += 1
            else:
                failed_assets += 1

            if (done % int(cfg.progress_every)) == 0 or done == len(asset_ids):
                print(
                    f"[cache] progress {done}/{len(asset_ids)} "
                    f"kept={len(kept_ids)} empty={empty_assets} "
                    f"drop_span={dropped_span} drop_obs={dropped_obs} fail={failed_assets}"
                )

    if not series_list:
        raise RuntimeError("No assets kept for returns_wide cache after filters.")

    wide = pd.concat(series_list, axis=1, join="outer")
    wide = wide.sort_index().astype(cfg.dtype)

    # write parquet to S3
    bio = io.BytesIO()
    wide.to_parquet(bio, index=True)
    bio.seek(0)
    store._put_bytes(cache_key, bio.read())

    meta = {
        "bucket": cfg.bucket,
        "returns_root": f"s3://{store.bucket}/{store.returns_prefix}",
        "cache_key": cache_key,
        "min_years": cfg.min_years,
        "min_obs": cfg.min_obs,
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "n_days": int(wide.shape[0]),
        "n_assets": int(wide.shape[1]),
        "dtype": cfg.dtype,
        "first_date": str(pd.Timestamp(wide.index.min()).date()),
        "last_date": str(pd.Timestamp(wide.index.max()).date()),
        "assets_sample": kept_ids[:20],
        "dropped_span": int(dropped_span),
        "dropped_obs": int(dropped_obs),
        "empty_assets": int(empty_assets),
        "failed_assets": int(failed_assets),
        "as_of_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": {
            "build_strategy": "parallel per-asset read + one-shot concat",
            "max_workers": int(cfg.max_workers),
            "s3_max_concurrency": int(cfg.s3_max_concurrency),
            "strict_window": bool(cfg.strict_window),
            "filter_to_active_universe": bool(cfg.filter_to_active_universe),
            "universe_csv": cfg.universe_csv,
            "excluded_csv": cfg.excluded_csv,
            "assets_discovered_all": int(len(asset_ids_all)),
            "active_from_universe": int(len(active)) if active is not None else None,
            "assets_used": int(len(asset_ids)),
        },
    }
    _put_json_s3(store, meta_key, meta)

    print("[OK] wrote cache: s3://%s/%s" % (store.bucket, cache_key))
    print("[OK] wrote meta : s3://%s/%s" % (store.bucket, meta_key))
    print("[OK] wide shape:", wide.shape)


if __name__ == "__main__":
    build_returns_wide_cache(CacheConfig())