from __future__ import annotations

import argparse
import io
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from alpha_edge import paths
from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import IndicatorBuildConfig, RuntimeConfig
from alpha_edge.market.indicator_calculations import (
    compute_market_indicators_for_asset,
    latest_indicator_rows,
)


def _to_day_naive(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value!r}")
    return pd.Timestamp(ts).tz_convert(None).normalize()


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy()

    if "date" not in out.columns:
        raise RuntimeError("Expected a 'date' column.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out["date"] = out["date"].dt.tz_convert(None).dt.normalize()
    out = out.dropna(subset=["date"])

    if "asset_id" in out.columns:
        out["asset_id"] = out["asset_id"].astype(str).str.strip()

    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(str)

    return out


def _read_universe_asset_ids(
    *,
    universe_csv: str,
    excluded_csv: Optional[str],
    filter_to_active_universe: bool,
) -> pd.DataFrame:
    universe = pd.read_csv(universe_csv)

    if "asset_id" not in universe.columns:
        raise RuntimeError("Universe CSV must contain 'asset_id'.")

    universe = universe.copy()
    universe["asset_id"] = universe["asset_id"].astype(str).str.strip()

    if "ticker" in universe.columns:
        universe["ticker"] = universe["ticker"].astype(str).str.upper().str.strip()
    elif "broker_ticker" in universe.columns:
        universe["ticker"] = (
            universe["broker_ticker"].astype(str).str.upper().str.strip()
        )
    else:
        universe["ticker"] = universe["asset_id"]

    if filter_to_active_universe and "include" in universe.columns:
        universe = universe[
            universe["include"].fillna(1).astype(int) == 1
        ].copy()

    if excluded_csv:
        try:
            excluded = pd.read_csv(excluded_csv)
            if "asset_id" in excluded.columns:
                excluded_ids = set(
                    excluded["asset_id"].astype(str).str.strip().tolist()
                )
                universe = universe[
                    ~universe["asset_id"].isin(excluded_ids)
                ].copy()
        except FileNotFoundError:
            pass

    universe = universe.dropna(subset=["asset_id"])
    universe = universe[universe["asset_id"] != ""]
    universe = universe.drop_duplicates(subset=["asset_id"], keep="last")
    universe = universe.sort_values("asset_id", kind="stable").reset_index(drop=True)

    return universe[["asset_id", "ticker"]]


def _parquet_bytes(df: pd.DataFrame, *, index: bool = False) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=index)
    buffer.seek(0)
    return buffer.read()


def _read_parquet_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(raw))


def _put_json(
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


def _snapshot_key(store: MarketStore, name: str) -> str:
    return f"{store.snapshots_prefix}/{name}.parquet"


def _read_existing_snapshot(
    *,
    store: MarketStore,
    name: str,
) -> pd.DataFrame:
    key = _snapshot_key(store, name)

    if not store._key_exists(key):
        return pd.DataFrame()

    try:
        return _normalize_dates(_read_parquet_bytes(store._get_bytes(key)))
    except Exception as exc:
        raise RuntimeError(
            f"Could not read existing indicator snapshot "
            f"s3://{store.bucket}/{key}: {exc}"
        ) from exc


def _write_snapshot(
    *,
    store: MarketStore,
    name: str,
    df: pd.DataFrame,
    dry_run: bool,
) -> str:
    key = _snapshot_key(store, name)
    out = _normalize_dates(df)

    if not out.empty:
        out = (
            out.sort_values(["asset_id", "date"], kind="stable")
            .drop_duplicates(["asset_id"], keep="last")
            .reset_index(drop=True)
        )

    if dry_run:
        print(
            f"[DRY RUN] Would write snapshot: "
            f"s3://{store.bucket}/{key} rows={len(out)}"
        )
        return key

    store._put_bytes(
        key,
        _parquet_bytes(out, index=False),
        content_type="application/octet-stream",
    )
    return key


def _manifest_parts(
    *,
    store: MarketStore,
    asset_id: str,
    year: int,
) -> list[str]:
    manifest = store.read_asset_year_manifest(
        table="indicators",
        asset_id=asset_id,
        year=year,
    )

    return [
        str(key)
        for key in (manifest.get("parts") or [])
        if isinstance(key, str) and key.endswith(".parquet")
    ]


def _read_existing_indicator_year(
    *,
    store: MarketStore,
    indicators_prefix: str,
    asset_id: str,
    year: int,
) -> tuple[pd.DataFrame, list[str]]:
    parts = _manifest_parts(
        store=store,
        asset_id=asset_id,
        year=year,
    )

    if not parts:
        prefix = (
            f"{indicators_prefix}/asset_id={asset_id}/year={int(year)}/"
        )
        parts = [
            key
            for key in store._list_keys(prefix)
            if key.endswith(".parquet")
        ]

    frames: list[pd.DataFrame] = []
    readable_parts: list[str] = []

    for key in parts:
        try:
            frame = _read_parquet_bytes(store._get_bytes(key))
            if frame is not None and not frame.empty:
                frames.append(frame)
                readable_parts.append(key)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(), readable_parts

    combined = _normalize_dates(pd.concat(frames, ignore_index=True))
    combined = (
        combined.drop_duplicates(["asset_id", "date"], keep="last")
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    return combined, readable_parts


def _write_compacted_indicator_year(
    *,
    store: MarketStore,
    indicators_prefix: str,
    asset_id: str,
    year: int,
    df: pd.DataFrame,
    dry_run: bool,
) -> str:
    out = _normalize_dates(df)

    if out.empty:
        raise ValueError(
            f"Refusing to write empty indicator partition for "
            f"asset_id={asset_id}, year={year}."
        )

    out = (
        out.drop_duplicates(["asset_id", "date"], keep="last")
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )

    part = uuid.uuid4().hex[:12]
    key = (
        f"{indicators_prefix}/asset_id={asset_id}/"
        f"year={int(year)}/part-{part}.parquet"
    )

    if dry_run:
        print(
            f"[DRY RUN] Would write indicators partition: "
            f"s3://{store.bucket}/{key} rows={len(out)}"
        )
        return key

    store._put_bytes(
        key,
        _parquet_bytes(out, index=False),
        content_type="application/octet-stream",
    )

    store.write_asset_year_manifest(
        table="indicators",
        asset_id=asset_id,
        year=int(year),
        dates=[
            pd.Timestamp(value).date().isoformat()
            for value in out["date"].tolist()
        ],
        parts=[key],
    )

    return key


def _write_full_history(
    *,
    store: MarketStore,
    indicators_prefix: str,
    indicators: pd.DataFrame,
    dry_run: bool,
) -> tuple[list[str], list[str]]:
    data = _normalize_dates(indicators)

    if data.empty:
        return [], []

    data["year"] = data["date"].dt.year.astype(int)

    written_keys: list[str] = []
    superseded_keys: list[str] = []

    for (asset_id, year), group in data.groupby(
        ["asset_id", "year"],
        sort=False,
    ):
        asset_id_str = str(asset_id).strip()
        existing_parts = _manifest_parts(
            store=store,
            asset_id=asset_id_str,
            year=int(year),
        )

        key = _write_compacted_indicator_year(
            store=store,
            indicators_prefix=indicators_prefix,
            asset_id=asset_id_str,
            year=int(year),
            df=group.drop(columns=["year"]),
            dry_run=dry_run,
        )

        written_keys.append(key)
        superseded_keys.extend(existing_parts)

    return written_keys, superseded_keys


def _upsert_incremental_history(
    *,
    store: MarketStore,
    indicators_prefix: str,
    asset_id: str,
    recalculated: pd.DataFrame,
    replacement_start: pd.Timestamp,
    dry_run: bool,
) -> tuple[list[str], list[str]]:
    data = _normalize_dates(recalculated)

    if data.empty:
        return [], []

    data = data[data["date"] >= replacement_start].copy()
    if data.empty:
        return [], []

    data["year"] = data["date"].dt.year.astype(int)

    written_keys: list[str] = []
    superseded_keys: list[str] = []

    for year, group in data.groupby("year", sort=False):
        year_int = int(year)
        year_start = pd.Timestamp(year=year_int, month=1, day=1)
        cutoff = max(replacement_start, year_start)

        existing, existing_parts = _read_existing_indicator_year(
            store=store,
            indicators_prefix=indicators_prefix,
            asset_id=asset_id,
            year=year_int,
        )

        if existing.empty:
            merged = group.drop(columns=["year"]).copy()
        else:
            preserved = existing[existing["date"] < cutoff].copy()
            replacement = group.drop(columns=["year"]).copy()
            merged = pd.concat(
                [preserved, replacement],
                ignore_index=True,
            )

        merged = (
            _normalize_dates(merged)
            .drop_duplicates(["asset_id", "date"], keep="last")
            .sort_values("date", kind="stable")
            .reset_index(drop=True)
        )

        key = _write_compacted_indicator_year(
            store=store,
            indicators_prefix=indicators_prefix,
            asset_id=asset_id,
            year=year_int,
            df=merged,
            dry_run=dry_run,
        )

        written_keys.append(key)
        superseded_keys.extend(existing_parts)

    return written_keys, superseded_keys


def _latest_date_map(snapshot: pd.DataFrame) -> dict[str, pd.Timestamp]:
    if snapshot is None or snapshot.empty:
        return {}

    work = _normalize_dates(snapshot)
    if "asset_id" not in work.columns:
        return {}

    latest = (
        work.sort_values(["asset_id", "date"], kind="stable")
        .drop_duplicates(["asset_id"], keep="last")
    )

    return {
        str(row.asset_id).strip(): pd.Timestamp(row.date).normalize()
        for row in latest.itertuples(index=False)
    }


def _effective_windows(
    *,
    cfg: IndicatorBuildConfig,
    stored_latest: Optional[pd.Timestamp],
    full_start: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if cfg.mode == "full" or stored_latest is None:
        return full_start, full_start

    calculation_start = max(
        full_start,
        stored_latest - pd.Timedelta(days=int(cfg.lookback_calendar_days)),
    )
    replacement_start = max(
        full_start,
        stored_latest - pd.Timedelta(days=int(cfg.replacement_calendar_days)),
    )

    return calculation_start, replacement_start


def build_market_indicators(
    *,
    cfg: IndicatorBuildConfig,
) -> dict:
    cfg.validate()

    store = MarketStore(
        bucket=cfg.bucket,
        region=cfg.region,
        base_prefix=cfg.market_root,
    )

    full_start = _to_day_naive(cfg.start)
    end_ts = (
        _to_day_naive(cfg.end)
        if cfg.end
        else _to_day_naive(pd.Timestamp.utcnow())
    )

    universe = _read_universe_asset_ids(
        universe_csv=cfg.universe_csv,
        excluded_csv=cfg.excluded_csv,
        filter_to_active_universe=cfg.filter_to_active_universe,
    )

    if cfg.max_assets is not None:
        universe = universe.head(int(cfg.max_assets)).copy()

    asset_ids = universe["asset_id"].astype(str).str.strip().tolist()

    existing_latest = _read_existing_snapshot(
        store=store,
        name=cfg.latest_snapshot_name,
    )
    stored_latest_by_asset = _latest_date_map(existing_latest)

    print("\n=== BUILD MARKET INDICATORS ===")
    print(f"mode:                {cfg.mode}")
    print(f"bucket:              {cfg.bucket}")
    print(f"region:              {cfg.region}")
    print(f"market_root:         {cfg.market_root}")
    print(f"ohlcv_prefix:        {store.ohlcv_prefix}")
    print(f"returns_prefix:      {store.returns_prefix}")
    print(f"indicators_prefix:   {cfg.indicators_prefix}")
    print(
        f"latest_snapshot:     "
        f"{store.snapshots_prefix}/{cfg.latest_snapshot_name}.parquet"
    )
    print(f"universe_csv:        {cfg.universe_csv}")
    print(f"excluded_csv:        {cfg.excluded_csv}")
    print(f"assets:              {len(asset_ids)}")
    print(f"full_window:         {full_start.date()}..{end_ts.date()}")
    print(f"min_obs:             {cfg.min_obs}")
    print(f"annualization_days:  {cfg.annualization_days}")
    print(f"lookback_days:       {cfg.lookback_calendar_days}")
    print(f"replacement_days:    {cfg.replacement_calendar_days}")
    print(f"skip_unchanged:      {cfg.skip_unchanged_assets}")
    print(f"workers:             {cfg.max_workers}")
    print(f"dry_run:             {cfg.dry_run}")
    print("")

    if not asset_ids:
        raise RuntimeError("No assets selected for indicator build.")

    failures: list[dict] = []
    latest_rows: list[pd.DataFrame] = []
    written_keys: list[str] = []
    superseded_keys: list[str] = []

    kept_assets = 0
    new_assets = 0
    unchanged_assets = 0
    empty_assets = 0
    failed_assets = 0
    dropped_min_obs = 0
    total_calculated_rows = 0
    total_written_rows = 0

    def _work(asset_id: str) -> dict:
        try:
            stored_latest = stored_latest_by_asset.get(asset_id)
            calculation_start, replacement_start = _effective_windows(
                cfg=cfg,
                stored_latest=stored_latest,
                full_start=full_start,
            )

            ohlcv = store.read_ohlcv_usd(
                asset_ids=[asset_id],
                start=str(calculation_start.date()),
                end=str(end_ts.date()),
                columns=None,
            )

            if ohlcv is None or ohlcv.empty:
                return {
                    "status": "empty",
                    "asset_id": asset_id,
                    "stored_latest": stored_latest,
                }

            ohlcv = _normalize_dates(ohlcv)
            source_latest = pd.Timestamp(ohlcv["date"].max()).normalize()

            if (
                cfg.mode == "incremental"
                and cfg.skip_unchanged_assets
                and stored_latest is not None
                and source_latest <= stored_latest
            ):
                return {
                    "status": "unchanged",
                    "asset_id": asset_id,
                    "stored_latest": stored_latest,
                    "source_latest": source_latest,
                }

            returns = store.read_returns_usd(
                asset_ids=[asset_id],
                start=str(calculation_start.date()),
                end=str(end_ts.date()),
                columns=None,
            )

            if returns is None or returns.empty:
                return {
                    "status": "fail",
                    "asset_id": asset_id,
                    "error": (
                        "No returns_usd rows found. Indicator layer "
                        "requires canonical returns."
                    ),
                }

            returns = _normalize_dates(returns)

            indicators = compute_market_indicators_for_asset(
                ohlcv=ohlcv,
                returns=returns,
                annualization_days=int(cfg.annualization_days),
            )

            if indicators is None or indicators.empty:
                return {"status": "empty", "asset_id": asset_id}

            indicators = _normalize_dates(indicators)

            n_obs = (
                int(indicators["close"].notna().sum())
                if "close" in indicators.columns
                else int(len(indicators))
            )

            if n_obs < int(cfg.min_obs):
                return {
                    "status": "drop_min_obs",
                    "asset_id": asset_id,
                    "n_obs": n_obs,
                }

            latest = latest_indicator_rows(indicators)

            rows_to_write = (
                indicators
                if cfg.mode == "full"
                else indicators[indicators["date"] >= replacement_start].copy()
            )

            return {
                "status": "keep",
                "asset_id": asset_id,
                "stored_latest": stored_latest,
                "source_latest": source_latest,
                "calculation_start": calculation_start,
                "replacement_start": replacement_start,
                "n_rows_calculated": int(len(indicators)),
                "n_rows_to_write": int(len(rows_to_write)),
                "indicators": indicators,
                "rows_to_write": rows_to_write,
                "latest": latest,
                "is_new_asset": stored_latest is None,
            }

        except Exception as exc:
            return {
                "status": "fail",
                "asset_id": asset_id,
                "error": f"{type(exc).__name__}: {exc}"[:1200],
            }

    done = 0

    with ThreadPoolExecutor(max_workers=int(cfg.max_workers)) as executor:
        futures = [executor.submit(_work, asset_id) for asset_id in asset_ids]

        for future in as_completed(futures):
            result = future.result()
            done += 1

            status = result.get("status")
            asset_id = str(result.get("asset_id"))

            if status == "keep":
                indicators = result["indicators"]
                rows_to_write = result["rows_to_write"]
                latest = result["latest"]
                replacement_start = pd.Timestamp(
                    result["replacement_start"]
                ).normalize()

                if cfg.mode == "full":
                    keys, old_keys = _write_full_history(
                        store=store,
                        indicators_prefix=cfg.indicators_prefix,
                        indicators=indicators,
                        dry_run=cfg.dry_run,
                    )
                else:
                    keys, old_keys = _upsert_incremental_history(
                        store=store,
                        indicators_prefix=cfg.indicators_prefix,
                        asset_id=asset_id,
                        recalculated=rows_to_write,
                        replacement_start=replacement_start,
                        dry_run=cfg.dry_run,
                    )

                written_keys.extend(keys)
                superseded_keys.extend(old_keys)

                if latest is not None and not latest.empty:
                    latest_rows.append(_normalize_dates(latest))

                kept_assets += 1
                if bool(result.get("is_new_asset")):
                    new_assets += 1

                total_calculated_rows += int(
                    result.get("n_rows_calculated", 0)
                )
                total_written_rows += int(
                    result.get("n_rows_to_write", 0)
                )

            elif status == "unchanged":
                unchanged_assets += 1

            elif status == "empty":
                empty_assets += 1

            elif status == "drop_min_obs":
                dropped_min_obs += 1

            else:
                failed_assets += 1
                failures.append(
                    {
                        "asset_id": asset_id,
                        "status": status,
                        "error": result.get("error"),
                    }
                )

            if (
                done % int(cfg.progress_every) == 0
                or done == len(asset_ids)
            ):
                print(
                    f"[indicators] progress {done}/{len(asset_ids)} "
                    f"kept={kept_assets} new={new_assets} "
                    f"unchanged={unchanged_assets} empty={empty_assets} "
                    f"drop_min_obs={dropped_min_obs} "
                    f"failed={failed_assets} parts={len(written_keys)}"
                )

    if latest_rows:
        refreshed_latest = _normalize_dates(
            pd.concat(latest_rows, ignore_index=True)
        )
        refreshed_assets = set(
            refreshed_latest["asset_id"].astype(str).str.strip()
        )

        if existing_latest.empty:
            latest_df = refreshed_latest
        else:
            preserved = existing_latest[
                ~existing_latest["asset_id"].astype(str).str.strip().isin(
                    refreshed_assets
                )
            ].copy()
            latest_df = pd.concat(
                [preserved, refreshed_latest],
                ignore_index=True,
            )
    else:
        latest_df = existing_latest.copy()

    latest_key = _write_snapshot(
        store=store,
        name=cfg.latest_snapshot_name,
        df=latest_df,
        dry_run=cfg.dry_run,
    )

    as_of_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    meta = {
        "as_of_utc": as_of_utc,
        "mode": cfg.mode,
        "bucket": cfg.bucket,
        "region": cfg.region,
        "market_root": cfg.market_root,
        "ohlcv_prefix": store.ohlcv_prefix,
        "returns_prefix": store.returns_prefix,
        "indicators_prefix": cfg.indicators_prefix,
        "latest_snapshot_key": latest_key,
        "start": str(full_start.date()),
        "end": str(end_ts.date()),
        "lookback_calendar_days": int(cfg.lookback_calendar_days),
        "replacement_calendar_days": int(
            cfg.replacement_calendar_days
        ),
        "skip_unchanged_assets": bool(cfg.skip_unchanged_assets),
        "assets_requested": int(len(asset_ids)),
        "assets_kept": int(kept_assets),
        "new_assets": int(new_assets),
        "unchanged_assets": int(unchanged_assets),
        "empty_assets": int(empty_assets),
        "dropped_min_obs": int(dropped_min_obs),
        "failed_assets": int(failed_assets),
        "calculated_indicator_rows": int(total_calculated_rows),
        "replacement_rows": int(total_written_rows),
        "written_parts": int(len(written_keys)),
        "written_parts_sample": written_keys[:50],
        "superseded_parts": int(len(set(superseded_keys))),
        "superseded_parts_sample": sorted(set(superseded_keys))[:50],
        "failures_sample": failures[:50],
        "annualization_days": int(cfg.annualization_days),
        "min_obs": int(cfg.min_obs),
        "filter_to_active_universe": bool(
            cfg.filter_to_active_universe
        ),
        "universe_csv": cfg.universe_csv,
        "excluded_csv": cfg.excluded_csv,
    }

    meta_key = f"{cfg.indicators_prefix}/latest_build.meta.json"
    _put_json(
        store,
        meta_key,
        meta,
        dry_run=cfg.dry_run,
    )

    print("")
    print("[OK] indicator build finished")
    print(f"[OK] mode:               {cfg.mode}")
    print(f"[OK] assets_kept:        {kept_assets}")
    print(f"[OK] new_assets:         {new_assets}")
    print(f"[OK] unchanged_assets:   {unchanged_assets}")
    print(f"[OK] calculated_rows:    {total_calculated_rows}")
    print(f"[OK] replacement_rows:   {total_written_rows}")
    print(f"[OK] written_parts:      {len(written_keys)}")
    print(f"[OK] superseded_parts:   {len(set(superseded_keys))}")
    print(
        f"[OK] latest_snapshot:    "
        f"s3://{store.bucket}/{latest_key}"
    )
    print(f"[OK] meta:               s3://{store.bucket}/{meta_key}")
    print("")

    return {
        "latest_key": latest_key,
        "meta_key": meta_key,
        "written_keys": written_keys,
        "superseded_keys": sorted(set(superseded_keys)),
        "meta": meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build historical market indicators and the latest "
            "indicator snapshot."
        )
    )

    parser.add_argument(
        "--env",
        default=None,
        choices=["dev", "staging", "prod"],
    )
    parser.add_argument("--confirm-prod-write", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--bucket", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--market-root", default=None)

    parser.add_argument("--universe-csv", default=None)
    parser.add_argument("--excluded-csv", default=None)
    parser.add_argument("--no-universe-filter", action="store_true")

    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help=(
            "full rebuilds all history; incremental recalculates a "
            "limited overlap and upserts affected partitions."
        ),
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--lookback-calendar-days",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--replacement-calendar-days",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--no-skip-unchanged-assets",
        action="store_true",
    )

    parser.add_argument("--max-assets", type=int, default=None)
    parser.add_argument("--min-obs", type=int, default=60)
    parser.add_argument("--annualization-days", type=int, default=252)

    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)

    parser.add_argument("--indicators-prefix", default=None)
    parser.add_argument(
        "--latest-snapshot-name",
        default="latest_indicators",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg: RuntimeConfig = load_runtime_config(args.env)

    if not bool(args.dry_run):
        require_prod_confirmation(
            runtime_cfg,
            bool(args.confirm_prod_write),
        )

    bucket = str(args.bucket or runtime_cfg.bucket)
    region = str(args.region or runtime_cfg.region)
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

    indicators_prefix = (
        str(args.indicators_prefix).strip("/")
        if args.indicators_prefix
        else f"{market_root}/indicators/v1"
    )

    input_args: Dict[str, Any] = vars(args).copy()
    input_args.update(
        {
            "resolved_bucket": bucket,
            "resolved_region": region,
            "resolved_market_root": market_root,
            "resolved_universe_csv": universe_csv,
            "resolved_excluded_csv": excluded_csv,
            "resolved_indicators_prefix": indicators_prefix,
        }
    )

    with capture_script_run(
        cfg=runtime_cfg,
        script_name="build_market_indicators.py",
        input_args=input_args,
        dry_run=bool(args.dry_run),
    ) as run_id:
        build_cfg = IndicatorBuildConfig(
            bucket=bucket,
            region=region,
            market_root=market_root,
            universe_csv=universe_csv,
            excluded_csv=excluded_csv,
            filter_to_active_universe=(
                not bool(args.no_universe_filter)
            ),
            mode=str(args.mode),
            start=str(args.start),
            end=str(args.end) if args.end else None,
            lookback_calendar_days=int(
                args.lookback_calendar_days
            ),
            replacement_calendar_days=int(
                args.replacement_calendar_days
            ),
            skip_unchanged_assets=(
                not bool(args.no_skip_unchanged_assets)
            ),
            max_assets=args.max_assets,
            min_obs=int(args.min_obs),
            annualization_days=int(args.annualization_days),
            max_workers=int(args.max_workers),
            progress_every=int(args.progress_every),
            indicators_prefix=indicators_prefix,
            latest_snapshot_name=str(
                args.latest_snapshot_name
            ),
            dry_run=bool(args.dry_run),
        )

        result = build_market_indicators(cfg=build_cfg)

        audit = build_audit_event(
            cfg=runtime_cfg,
            run_id=run_id,
            event_type="modify",
            entity_type="market_indicators",
            entity_id="market_indicators_v1",
            as_of=None,
            source_script="build_market_indicators.py",
            source_mode=str(args.mode),
            status=("dry_run" if args.dry_run else "success"),
            reason=(
                "build full or incremental historical market "
                "indicators and latest snapshot"
            ),
            input_args=input_args,
            output_keys=[
                str(result["latest_key"]),
                str(result["meta_key"]),
            ],
            metadata={
                **dict(result["meta"]),
                "written_keys_sample": [
                    str(key)
                    for key in result["written_keys"][:100]
                ],
                "superseded_keys_sample": [
                    str(key)
                    for key in result["superseded_keys"][:100]
                ],
            },
        )
        write_audit_event(
            cfg=runtime_cfg,
            event=audit,
            dry_run=bool(args.dry_run),
        )


if __name__ == "__main__":
    main()
