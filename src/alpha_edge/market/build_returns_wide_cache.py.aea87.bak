from __future__ import annotations

import argparse
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import pandas as pd
import numpy as np

from alpha_edge import paths
from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import CacheConfig, RuntimeConfig


RETURN_COLUMNS = [
    "date",
    "asset_id",
    "ticker",
    "ret_log_close_adjusted_usd",
    "ret_close_adjusted_usd",
    "ret_adj_close_usd",
]


def _safe_date(value: Any) -> Optional[pd.Timestamp]:
    if value is None or str(value).strip() == "":
        return None

    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None

    return pd.Timestamp(ts).tz_convert(None).normalize()


def _to_day_naive(value: Any) -> pd.Timestamp:
    ts = _safe_date(value)
    if ts is None:
        raise ValueError(f"Invalid date: {value!r}")
    return ts


def _parquet_bytes(df: pd.DataFrame, *, index: bool = True) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=index)
    buffer.seek(0)
    return buffer.read()


def _read_parquet_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(raw))


def _put_json_s3(
    store: MarketStore,
    key: str,
    payload: dict,
    *,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"[DRY RUN] Would write JSON: s3://{store.bucket}/{key}")
        return

    store._put_bytes(
        key,
        json.dumps(payload, indent=2, default=str).encode("utf-8"),
        content_type="application/json",
    )


def _get_json_s3(store: MarketStore, key: str) -> dict:
    try:
        value = json.loads(store._get_bytes(key).decode("utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _make_market_store(cfg: CacheConfig) -> MarketStore:
    try:
        return MarketStore(
            bucket=cfg.bucket,
            region=cfg.region,
            base_prefix=cfg.market_root,
        )
    except TypeError:
        if cfg.market_root != "market":
            raise RuntimeError(
                "MarketStore does not support base_prefix, while "
                f"market_root={cfg.market_root!r}. Refusing to fall back "
                "to the production market root."
            )
        return MarketStore(bucket=cfg.bucket, region=cfg.region)


def _load_active_asset_ids_from_universe(
    *,
    universe_csv: str,
    excluded_csv: Optional[str],
) -> set[str]:
    universe = pd.read_csv(universe_csv)

    if "asset_id" not in universe.columns:
        raise RuntimeError("Universe CSV must include 'asset_id'.")

    universe = universe.copy()
    universe["asset_id"] = universe["asset_id"].astype(str).str.strip()

    if "include" in universe.columns:
        universe = universe[
            universe["include"].fillna(1).astype(int) == 1
        ].copy()

    active = set(
        universe.loc[universe["asset_id"] != "", "asset_id"].tolist()
    )

    if excluded_csv:
        try:
            excluded = pd.read_csv(excluded_csv)
            if "asset_id" in excluded.columns:
                excluded_ids = set(
                    excluded["asset_id"].astype(str).str.strip().tolist()
                )
                active -= excluded_ids
        except FileNotFoundError:
            pass

    return active


def _discover_return_asset_ids(store: MarketStore) -> list[str]:
    root_prefix = f"{store.returns_prefix}/"
    keys = store._list_keys(root_prefix)

    if not keys:
        raise RuntimeError(
            f"No objects found under s3://{store.bucket}/{root_prefix}"
        )

    asset_ids: set[str] = set()

    for key in keys:
        if (
            "/asset_id=" not in key
            or "/year=" not in key
            or not key.endswith(".parquet")
        ):
            continue

        try:
            value = key.split("/asset_id=", 1)[1].split("/", 1)[0].strip()
            if value:
                asset_ids.add(value)
        except Exception:
            continue

    if not asset_ids:
        raise RuntimeError(
            "Could not discover asset_id partitions under returns root."
        )

    return sorted(asset_ids)


def _normalize_existing_wide(
    wide: pd.DataFrame,
    *,
    dtype: str,
) -> pd.DataFrame:
    if wide is None or wide.empty:
        return pd.DataFrame()

    out = wide.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(
            out["date"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None).dt.normalize()
        out = out.dropna(subset=["date"]).set_index("date")
    else:
        idx = pd.to_datetime(out.index, errors="coerce", utc=True)
        valid = ~idx.isna()
        out = out.loc[valid].copy()
        out.index = idx[valid].tz_convert(None).normalize()

    out.index.name = "date"
    out.columns = [str(col).strip() for col in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="last")]
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()

    return out.astype(dtype)


def _read_existing_cache(
    *,
    store: MarketStore,
    cache_key: str,
    dtype: str,
) -> pd.DataFrame:
    if not store._key_exists(cache_key):
        return pd.DataFrame()

    try:
        raw = store._get_bytes(cache_key)
        return _normalize_existing_wide(
            _read_parquet_bytes(raw),
            dtype=dtype,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not read existing cache "
            f"s3://{store.bucket}/{cache_key}: {exc}"
        ) from exc


def _extract_return_series(
    *,
    df: pd.DataFrame,
    asset_id: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    dtype: str,
    strict_window: bool,
) -> pd.Series:
    """
    Extract one asset return series for returns_wide.

    CONTRACT:
    returns_wide must expose SIMPLE returns because portfolio evaluators use
    cumprod(1 + r) and Monte Carlo simulation on ordinary return shocks.

    Priority:
      1. ret_adj_close_usd
      2. ret_close_adjusted_usd
      3. ret_log_close_adjusted_usd converted to simple with np.expm1()

    This function must never pass raw log returns into returns_wide.
    """
    if df is None or df.empty:
        return pd.Series(dtype=dtype, name=asset_id)

    if "date" not in df.columns:
        raise RuntimeError("Returns data is missing the 'date' column.")

    dates = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()

    values_log = None
    values_simple_adj = None
    values_simple_close_adjusted = None

    if "ret_log_close_adjusted_usd" in df.columns:
        values_log = pd.to_numeric(
            df["ret_log_close_adjusted_usd"],
            errors="coerce",
        )

    if "ret_adj_close_usd" in df.columns:
        values_simple_adj = pd.to_numeric(
            df["ret_adj_close_usd"],
            errors="coerce",
        )

    if "ret_close_adjusted_usd" in df.columns:
        values_simple_close_adjusted = pd.to_numeric(
            df["ret_close_adjusted_usd"],
            errors="coerce",
        )

    if (
        values_simple_adj is None
        and values_simple_close_adjusted is None
        and values_log is None
    ):
        raise RuntimeError(
            "Returns data contains none of: "
            "'ret_adj_close_usd', 'ret_close_adjusted_usd', "
            "'ret_log_close_adjusted_usd'."
        )

    if values_simple_adj is not None:
        values = values_simple_adj.copy()
        return_source = "simple_ret_adj_close_usd"

        if values_simple_close_adjusted is not None:
            values = values.combine_first(values_simple_close_adjusted)
            return_source = "simple_ret_adj_close_usd_with_close_adjusted_fallback"

        if values_log is not None:
            values = values.combine_first(np.expm1(values_log))
            return_source = f"{return_source}_with_log_expm1_fallback"

    elif values_simple_close_adjusted is not None:
        values = values_simple_close_adjusted.copy()
        return_source = "simple_ret_close_adjusted_usd"

        if values_log is not None:
            values = values.combine_first(np.expm1(values_log))
            return_source = f"{return_source}_with_log_expm1_fallback"

    else:
        values = np.expm1(values_log)
        return_source = "log_converted_to_simple_expm1"

    normalized = pd.DataFrame(
        {
            "date": dates,
            "return_value": values,
        }
    ).dropna(subset=["date", "return_value"])

    if strict_window:
        normalized = normalized[
            (normalized["date"] >= start_ts)
            & (normalized["date"] <= end_ts)
        ]

    normalized = (
        normalized.sort_values("date", kind="stable")
        .drop_duplicates(subset=["date"], keep="last")
    )

    if normalized.empty:
        return pd.Series(dtype=dtype, name=asset_id)

    series = normalized.set_index("date")["return_value"].astype(dtype)
    series.name = asset_id

    # Optional debug metadata attached to Series attrs.
    series.attrs["return_source"] = return_source
    series.attrs["return_type"] = "simple"
    series.attrs["n_log_non_null"] = int(values_log.notna().sum()) if values_log is not None else 0
    series.attrs["n_simple_adj_non_null"] = (
        int(values_simple_adj.notna().sum())
        if values_simple_adj is not None
        else 0
    )
    series.attrs["n_simple_close_adjusted_non_null"] = (
        int(values_simple_close_adjusted.notna().sum())
        if values_simple_close_adjusted is not None
        else 0
    )
    series.attrs["n_combined_non_null"] = int(series.notna().sum())

    return series

def _qualifies_full_history(
    *,
    series: pd.Series,
    min_years: float,
    min_obs: int,
) -> tuple[bool, str, int, int]:
    if series is None or series.empty:
        return False, "empty", 0, 0

    first = pd.Timestamp(series.index.min())
    last = pd.Timestamp(series.index.max())
    span_days = int((last - first).days)
    observations = int(series.notna().sum())

    if span_days < int(min_years * 365):
        return False, "drop_span", observations, span_days

    if observations < int(min_obs):
        return False, "drop_obs", observations, span_days

    return True, "keep", observations, span_days


def _merge_incremental_cache(
    *,
    existing: pd.DataFrame,
    updates: pd.DataFrame,
    replacement_start: pd.Timestamp,
    selected_assets: list[str],
    dtype: str,
) -> pd.DataFrame:
    selected_set = set(selected_assets)

    if existing.empty:
        preserved = pd.DataFrame()
    else:
        preserved = existing.loc[
            existing.index < replacement_start
        ].copy()

    existing_overlap = (
        existing.loc[existing.index >= replacement_start].copy()
        if not existing.empty
        else pd.DataFrame()
    )

    if not existing_overlap.empty:
        existing_overlap = existing_overlap.drop(
            columns=[
                col
                for col in existing_overlap.columns
                if col in selected_set
            ],
            errors="ignore",
        )

    merged_overlap = pd.concat(
        [existing_overlap, updates],
        axis=1,
        join="outer",
    )

    merged = pd.concat(
        [preserved, merged_overlap],
        axis=0,
        join="outer",
    )

    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.loc[:, ~pd.Index(merged.columns).duplicated(keep="last")]
    merged = merged.sort_index()
    merged = merged.reindex(sorted(merged.columns), axis=1)

    return merged.astype(dtype)


def build_returns_wide_cache(
    cfg: CacheConfig,
) -> dict:
    cfg.validate()
    store = _make_market_store(cfg)

    start_ts = _to_day_naive(cfg.start)
    end_ts = (
        _to_day_naive(cfg.end)
        if cfg.end
        else _to_day_naive(pd.Timestamp.utcnow())
    )

    meta_key = (
        f"{cfg.cache_prefix}/"
        f"returns_wide_min{int(cfg.min_years)}y.meta.json"
    )
    cache_key = (
        f"{cfg.cache_prefix}/"
        f"returns_wide_min{int(cfg.min_years)}y.parquet"
    )

    asset_ids_all = _discover_return_asset_ids(store)

    active_asset_ids: Optional[set[str]] = None
    selected_asset_ids = asset_ids_all

    if cfg.filter_to_active_universe:
        active_asset_ids = _load_active_asset_ids_from_universe(
            universe_csv=cfg.universe_csv,
            excluded_csv=cfg.excluded_csv,
        )
        selected_asset_ids = sorted(
            set(asset_ids_all).intersection(active_asset_ids)
        )

    if not selected_asset_ids:
        raise RuntimeError(
            "No assets selected for returns-wide cache after filtering."
        )

    existing = _read_existing_cache(
        store=store,
        cache_key=cache_key,
        dtype=cfg.dtype,
    )

    latest_state = store.read_returns_latest_state()
    source_latest = _safe_date(latest_state.get("last_date"))
    existing_last = (
        _safe_date(existing.index.max())
        if not existing.empty
        else None
    )

    if (
        cfg.mode == "incremental"
        and not cfg.force
        and source_latest is not None
        and existing_last is not None
        and existing_last >= source_latest
    ):
        existing_assets = set(existing.columns)
        selected_assets = set(selected_asset_ids)

        if (
            existing_assets == selected_assets
            or (
                not cfg.filter_to_active_universe
                and selected_assets.issubset(existing_assets)
            )
        ):
            print(
                "[SKIP] returns-wide cache up to date: "
                f"cache_last={existing_last.date()} "
                f"latest_returns={source_latest.date()} "
                f"assets={len(existing.columns)}"
            )
            return {
                "status": "skipped",
                "cache_key": cache_key,
                "meta_key": meta_key,
                "meta": _get_json_s3(store, meta_key),
            }

    if cfg.mode == "incremental" and existing.empty:
        if len(selected_asset_ids) > cfg.max_new_assets_full_build:
            raise RuntimeError(
                "Incremental mode found no existing cache and would need "
                f"to backfill {len(selected_asset_ids)} assets. This exceeds "
                f"max_new_assets_full_build={cfg.max_new_assets_full_build}. "
                "Run --mode full explicitly for the initial build."
            )

    existing_asset_ids = set(existing.columns)
    selected_asset_set = set(selected_asset_ids)

    existing_selected_assets = sorted(
        selected_asset_set.intersection(existing_asset_ids)
    )
    new_asset_ids = sorted(
        selected_asset_set.difference(existing_asset_ids)
    )
    removed_asset_ids = sorted(
        existing_asset_ids.difference(selected_asset_set)
    )

    if cfg.mode == "full":
        existing_selected_assets = []
        new_asset_ids = selected_asset_ids
        replacement_start = start_ts
    else:
        anchor = existing_last or start_ts
        replacement_start = max(
            start_ts,
            anchor - pd.Timedelta(
                days=int(cfg.replacement_calendar_days)
            ),
        )

    print("\n=== BUILD RETURNS WIDE CACHE ===")
    print(f"mode:                   {cfg.mode}")
    print(f"bucket:                 {cfg.bucket}")
    print(f"region:                 {cfg.region}")
    print(f"market_root:            {cfg.market_root}")
    print(f"returns_root:           {store.returns_prefix}")
    print(f"cache_key:              {cache_key}")
    print(f"window:                 {start_ts.date()}..{end_ts.date()}")
    print(f"replacement_start:      {replacement_start.date()}")
    print(f"assets_discovered:      {len(asset_ids_all)}")
    print(f"assets_selected:        {len(selected_asset_ids)}")
    print(f"existing_cache_assets:  {len(existing_asset_ids)}")
    print(f"existing_assets_update: {len(existing_selected_assets)}")
    print(f"new_assets_backfill:    {len(new_asset_ids)}")
    print(f"removed_assets:         {len(removed_asset_ids)}")
    print(f"workers:                {cfg.max_workers}")
    print(f"s3_max_concurrency:     {cfg.s3_max_concurrency}")
    print(f"dry_run:                {cfg.dry_run}")
    print("")

    s3_semaphore = threading.Semaphore(
        int(cfg.s3_max_concurrency)
    )

    def _read_asset(
        asset_id: str,
        *,
        full_history: bool,
    ) -> dict:
        try:
            read_start = start_ts if full_history else replacement_start

            with s3_semaphore:
                frame = store.read_returns_usd(
                    asset_ids=[asset_id],
                    start=str(read_start.date()),
                    end=str(end_ts.date()),
                    columns=RETURN_COLUMNS,
                )

            series = _extract_return_series(
                df=frame,
                asset_id=asset_id,
                start_ts=read_start,
                end_ts=end_ts,
                dtype=cfg.dtype,
                strict_window=cfg.strict_window,
            )

            if series.empty:
                return {
                    "status": "empty",
                    "asset_id": asset_id,
                    "full_history": full_history,
                }

            if full_history:
                qualifies, status, nobs, span_days = (
                    _qualifies_full_history(
                        series=series,
                        min_years=cfg.min_years,
                        min_obs=cfg.min_obs,
                    )
                )

                if not qualifies:
                    return {
                        "status": status,
                        "asset_id": asset_id,
                        "nobs": nobs,
                        "span_days": span_days,
                        "full_history": True,
                    }

                return {
                    "status": "keep_new",
                    "asset_id": asset_id,
                    "series": series,
                    "nobs": nobs,
                    "span_days": span_days,
                    "full_history": True,
                }

            return {
                "status": "keep_update",
                "asset_id": asset_id,
                "series": series,
                "nobs": int(series.notna().sum()),
                "full_history": False,
            }

        except Exception as exc:
            return {
                "status": "fail",
                "asset_id": asset_id,
                "full_history": full_history,
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }

    jobs: list[tuple[str, bool]] = [
        (asset_id, False)
        for asset_id in existing_selected_assets
    ] + [
        (asset_id, True)
        for asset_id in new_asset_ids
    ]

    update_series: list[pd.Series] = []
    new_series: list[pd.Series] = []
    failures: list[dict] = []
    drop_span_sample: list[dict] = []
    drop_obs_sample: list[dict] = []
    empty_sample: list[dict] = []

    counters = {
        "keep_update": 0,
        "keep_new": 0,
        "empty": 0,
        "drop_span": 0,
        "drop_obs": 0,
        "fail": 0,
    }

    with ThreadPoolExecutor(
        max_workers=int(cfg.max_workers)
    ) as executor:
        futures = {
            executor.submit(
                _read_asset,
                asset_id,
                full_history=full_history,
            ): (asset_id, full_history)
            for asset_id, full_history in jobs
        }

        done = 0

        for future in as_completed(futures):
            result = future.result()
            done += 1

            status = str(result.get("status"))
            counters[status] = counters.get(status, 0) + 1
            if status == "drop_span" and len(drop_span_sample) < 50:
                drop_span_sample.append(
                    {
                        "asset_id": result.get("asset_id"),
                        "nobs": result.get("nobs"),
                        "span_days": result.get("span_days"),
                        "full_history": result.get("full_history"),
                    }
                )
            elif status == "drop_obs" and len(drop_obs_sample) < 50:
                drop_obs_sample.append(
                    {
                        "asset_id": result.get("asset_id"),
                        "nobs": result.get("nobs"),
                        "span_days": result.get("span_days"),
                        "full_history": result.get("full_history"),
                    }
                )
            elif status == "empty" and len(empty_sample) < 50:
                empty_sample.append(
                    {
                        "asset_id": result.get("asset_id"),
                        "full_history": result.get("full_history"),
                    }
                )

            if status == "keep_update":
                update_series.append(result["series"])
            elif status == "keep_new":
                new_series.append(result["series"])
            elif status == "fail":
                failures.append(
                    {
                        "asset_id": result.get("asset_id"),
                        "error": result.get("error"),
                    }
                )

            if (
                done % int(cfg.progress_every) == 0
                or done == len(jobs)
            ):
                print(
                    f"[cache] progress {done}/{len(jobs)} "
                    f"updated={counters['keep_update']} "
                    f"new={counters['keep_new']} "
                    f"empty={counters['empty']} "
                    f"drop_span={counters['drop_span']} "
                    f"drop_obs={counters['drop_obs']} "
                    f"fail={counters['fail']}"
                )

    if cfg.mode == "full":
        all_series = new_series
        if not all_series:
            raise RuntimeError(
                "No assets qualified for the full returns-wide cache."
            )

        wide = pd.concat(
            all_series,
            axis=1,
            join="outer",
        ).sort_index()
    else:
        overlap_updates = (
            pd.concat(update_series, axis=1, join="outer")
            if update_series
            else pd.DataFrame()
        )

        if new_series:
            new_asset_frame = pd.concat(
                new_series,
                axis=1,
                join="outer",
            )
        else:
            new_asset_frame = pd.DataFrame()

        wide = _merge_incremental_cache(
            existing=existing,
            updates=overlap_updates,
            replacement_start=replacement_start,
            selected_assets=existing_selected_assets,
            dtype=cfg.dtype,
        )

        if not new_asset_frame.empty:
            wide = pd.concat(
                [wide, new_asset_frame],
                axis=1,
                join="outer",
            )
            wide = wide.loc[
                :,
                ~pd.Index(wide.columns).duplicated(keep="last"),
            ]

    if cfg.filter_to_active_universe:
        kept_active_columns = [
            column
            for column in wide.columns
            if column in selected_asset_set
        ]
        wide = wide[kept_active_columns]

    wide = wide[
        (wide.index >= start_ts)
        & (wide.index <= end_ts)
    ]
    wide = wide[~wide.index.duplicated(keep="last")]
    wide = wide.loc[
        :,
        ~pd.Index(wide.columns).duplicated(keep="last"),
    ]
    wide = wide.sort_index()
    wide = wide.reindex(sorted(wide.columns), axis=1)

    # Returns-wide must be a homogeneous calendar matrix for portfolio simulation.
    #
    # IMPORTANT:
    # We fill missing returns with 0.0, not forward-fill returns.
    # Forward-filling returns would incorrectly repeat the previous trading day's
    # return over weekends/holidays. A non-trading day should contribute zero return
    # for assets that did not trade.
    wide = wide.fillna(0.0)

    wide = wide.astype(cfg.dtype)

    empty_columns = [
        column
        for column in wide.columns
        if wide[column].notna().sum() == 0
    ]
    if empty_columns:
        wide = wide.drop(columns=empty_columns)

    if wide.empty or wide.shape[1] == 0:
        raise RuntimeError(
            "Resulting returns-wide cache is empty."
        )

    if not cfg.dry_run:
        store._put_bytes(
            cache_key,
            _parquet_bytes(wide, index=True),
            content_type="application/octet-stream",
        )
    else:
        print(
            f"[DRY RUN] Would write cache: "
            f"s3://{store.bucket}/{cache_key} "
            f"shape={wide.shape}"
        )

    meta = {
        "mode": cfg.mode,
        "bucket": cfg.bucket,
        "region": cfg.region,
        "market_root": cfg.market_root,
        "returns_root": (
            f"s3://{store.bucket}/{store.returns_prefix}"
        ),
        "cache_key": cache_key,
        "cache_uri": f"s3://{store.bucket}/{cache_key}",
        "min_years": float(cfg.min_years),
        "min_obs": int(cfg.min_obs),
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "replacement_start": str(replacement_start.date()),
        "replacement_calendar_days": int(
            cfg.replacement_calendar_days
        ),
        "n_days": int(wide.shape[0]),
        "n_assets": int(wide.shape[1]),
        "dtype": cfg.dtype,
        "first_date": str(
            pd.Timestamp(wide.index.min()).date()
        ),
        "last_date": str(
            pd.Timestamp(wide.index.max()).date()
        ),
        "source_latest_date": (
            str(source_latest.date())
            if source_latest is not None
            else None
        ),
        "existing_cache_last_date": (
            str(existing_last.date())
            if existing_last is not None
            else None
        ),
        "assets_discovered_all": int(len(asset_ids_all)),
        "assets_selected": int(len(selected_asset_ids)),
        "existing_assets_updated": int(
            counters["keep_update"]
        ),
        "new_assets_backfilled": int(counters["keep_new"]),
        "removed_assets": int(len(removed_asset_ids)),
        "removed_assets_sample": removed_asset_ids[:50],
        "empty_assets": int(counters["empty"]),
        "dropped_span": int(counters["drop_span"]),
        "dropped_obs": int(counters["drop_obs"]),
        "failed_assets": int(counters["fail"]),
        "drop_span_sample": drop_span_sample,
        "drop_obs_sample": drop_obs_sample,
        "empty_sample": empty_sample,
        "failures_sample": failures[:50],
        "assets_sample": list(wide.columns[:20]),
        "as_of_utc": pd.Timestamp.utcnow().strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "notes": {
            "build_strategy": (
                "full per-asset build"
                if cfg.mode == "full"
                else (
                    "incremental date overlap for existing assets "
                    "+ full-history eligibility check for new assets"
                )
            ),
            "return_type": "simple",
            "return_column_preferred": "ret_adj_close_usd",
            "return_column_compat": "ret_close_adjusted_usd",
            "return_column_log_fallback": (
                "ret_log_close_adjusted_usd converted with np.expm1"
            ),
            "missing_return_fill_policy": "fillna_0_after_outer_join",
            "missing_return_fill_reason": (
                "Returns are filled with 0.0 for non-trading calendar dates. "
                "Returns are never forward-filled because that would repeat prior returns."
            ),
            "max_workers": int(cfg.max_workers),
            "s3_max_concurrency": int(
                cfg.s3_max_concurrency
            ),
            "strict_window": bool(cfg.strict_window),
            "filter_to_active_universe": bool(
                cfg.filter_to_active_universe
            ),
            "universe_csv": cfg.universe_csv,
            "excluded_csv": cfg.excluded_csv,
        },
    }

    _put_json_s3(
        store,
        meta_key,
        meta,
        dry_run=cfg.dry_run,
    )

    print("")
    print("[OK] returns-wide cache build finished")
    print(f"[OK] mode:               {cfg.mode}")
    print(f"[OK] shape:              {wide.shape}")
    print(f"[OK] updated_assets:     {counters['keep_update']}")
    print(f"[OK] new_assets:         {counters['keep_new']}")
    print(f"[OK] removed_assets:     {len(removed_asset_ids)}")
    print(f"[OK] failed_assets:      {counters['fail']}")
    print(f"[OK] cache:              s3://{store.bucket}/{cache_key}")
    print(f"[OK] metadata:           s3://{store.bucket}/{meta_key}")
    print("")

    return {
        "status": "dry_run" if cfg.dry_run else "success",
        "cache_key": cache_key,
        "meta_key": meta_key,
        "meta": meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Alpha Edge returns-wide cache."
    )

    parser.add_argument(
        "--env",
        default=None,
        choices=["dev", "staging", "prod"],
    )
    parser.add_argument("--confirm-prod-write", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-write", action="store_true")

    parser.add_argument("--bucket", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--market-root", default=None)

    parser.add_argument("--cache-prefix", default=None)
    parser.add_argument("--universe-csv", default=None)
    parser.add_argument("--excluded-csv", default=None)
    parser.add_argument("--no-universe-filter", action="store_true")

    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--replacement-calendar-days",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max-new-assets-full-build",
        type=int,
        default=100,
    )

    parser.add_argument("--min-years", type=float, default=5.0)
    parser.add_argument("--min-obs", type=int, default=252 * 5)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--s3-max-concurrency",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--no-strict-window",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg: RuntimeConfig = load_runtime_config(args.env)

    dry_run = bool(args.dry_run or args.no_write)

    if not dry_run:
        require_prod_confirmation(
            runtime_cfg,
            bool(args.confirm_prod_write),
        )

    bucket = str(
        args.bucket or runtime_cfg.bucket
    ).strip()
    region = str(
        args.region or runtime_cfg.region
    ).strip()
    market_root = str(
        args.market_root or runtime_cfg.market_root
    ).strip("/")

    universe_csv = str(
        args.universe_csv
        or (paths.universe_dir() / "universe.csv")
    )
    excluded_csv = str(
        args.excluded_csv
        or (paths.universe_dir() / "asset_excluded.csv")
    )

    cache_prefix = (
        str(args.cache_prefix).strip("/")
        if args.cache_prefix
        else f"{market_root}/cache/v1"
    )

    input_args = vars(args).copy()
    input_args.update(
        {
            "resolved_bucket": bucket,
            "resolved_region": region,
            "resolved_market_root": market_root,
            "resolved_universe_csv": universe_csv,
            "resolved_excluded_csv": excluded_csv,
            "resolved_cache_prefix": cache_prefix,
            "resolved_dry_run": dry_run,
        }
    )

    with capture_script_run(
        cfg=runtime_cfg,
        script_name="build_returns_wide_cache.py",
        input_args=input_args,
        dry_run=dry_run,
    ) as run_id:
        try:
            config = CacheConfig(
                bucket=bucket,
                region=region,
                market_root=market_root,
                cache_prefix=cache_prefix,
                mode=str(args.mode),
                min_years=float(args.min_years),
                start=str(args.start),
                end=str(args.end) if args.end else None,
                min_obs=int(args.min_obs),
                dtype=str(args.dtype),
                force=bool(args.force),
                replacement_calendar_days=int(
                    args.replacement_calendar_days
                ),
                max_new_assets_full_build=int(
                    args.max_new_assets_full_build
                ),
                progress_every=int(args.progress_every),
                max_workers=int(args.max_workers),
                s3_max_concurrency=int(
                    args.s3_max_concurrency
                ),
                strict_window=(
                    not bool(args.no_strict_window)
                ),
                universe_csv=universe_csv,
                excluded_csv=excluded_csv,
                filter_to_active_universe=(
                    not bool(args.no_universe_filter)
                ),
                dry_run=dry_run,
            )

            result = build_returns_wide_cache(config)

            event = build_audit_event(
                cfg=runtime_cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="returns_wide_cache",
                entity_id=(
                    f"returns_wide_min"
                    f"{int(config.min_years)}y"
                ),
                as_of=result["meta"].get("last_date"),
                source_script="build_returns_wide_cache.py",
                source_mode=config.mode,
                status=result["status"],
                input_args=input_args,
                output_keys=[
                    str(result["cache_key"]),
                    str(result["meta_key"]),
                ],
                metadata={
                    "tier": "tier_1",
                    "payload_policy": (
                        "large_dataset_metadata_only"
                    ),
                    **dict(result["meta"]),
                },
            )
            write_audit_event(
                cfg=runtime_cfg,
                event=event,
                dry_run=dry_run,
            )

        except Exception as exc:
            event = build_audit_event(
                cfg=runtime_cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="returns_wide_cache",
                entity_id=None,
                as_of=None,
                source_script="build_returns_wide_cache.py",
                source_mode=str(args.mode),
                status="failed",
                input_args=input_args,
                metadata={
                    "tier": "tier_1",
                    "payload_policy": (
                        "large_dataset_metadata_only"
                    ),
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(
                cfg=runtime_cfg,
                event=event,
                dry_run=dry_run,
            )
            raise


if __name__ == "__main__":
    main()
