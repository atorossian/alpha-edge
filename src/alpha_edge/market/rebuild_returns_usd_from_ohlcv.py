# rebuild_returns_usd_from_ohlcv.py
from __future__ import annotations

import argparse
import io
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from alpha_edge import paths
from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import RuntimeConfig


@dataclass
class RebuildReturnsConfig:
    bucket: str
    region: str
    market_root: str
    universe_csv: str
    excluded_csv: Optional[str] = None
    filter_to_active_universe: bool = True
    start: str = "2010-01-01"
    end: Optional[str] = None
    max_assets: Optional[int] = None
    max_workers: int = 8
    progress_every: int = 100
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.bucket = str(self.bucket).strip()
        self.region = str(self.region).strip()
        self.market_root = str(self.market_root).strip("/")
        self.universe_csv = str(self.universe_csv)
        if self.excluded_csv is not None:
            self.excluded_csv = str(self.excluded_csv)


def _to_day_naive(x) -> pd.Timestamp:
    ts = pd.to_datetime(x, errors="coerce", utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {x!r}")
    return pd.Timestamp(ts).tz_convert(None).normalize()


def _read_universe_asset_ids(
    *,
    universe_csv: str,
    excluded_csv: Optional[str],
    filter_to_active_universe: bool,
) -> list[str]:
    u = pd.read_csv(universe_csv)

    if "asset_id" not in u.columns:
        raise RuntimeError("Universe CSV must contain asset_id.")

    u = u.copy()
    u["asset_id"] = u["asset_id"].astype(str).str.strip()

    if filter_to_active_universe and "include" in u.columns:
        u = u[u["include"].fillna(1).astype(int) == 1].copy()

    if excluded_csv:
        try:
            ex = pd.read_csv(excluded_csv)
            if "asset_id" in ex.columns:
                bad = set(ex["asset_id"].astype(str).str.strip().tolist())
                u = u[~u["asset_id"].isin(bad)].copy()
        except Exception:
            pass

    asset_ids = (
        u["asset_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return asset_ids


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    df.to_parquet(bio, index=False)
    bio.seek(0)
    return bio.read()


def _compute_log_returns_from_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()

    required = {"date", "asset_id", "close_adjusted_usd"}
    missing = sorted(required - set(ohlcv.columns))
    if missing:
        raise ValueError(f"OHLCV missing required columns for returns rebuild: {missing}")

    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["close_adjusted_usd"] = pd.to_numeric(df["close_adjusted_usd"], errors="coerce")

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    else:
        df["ticker"] = None

    df = (
        df.dropna(subset=["date", "asset_id", "close_adjusted_usd"])
        .sort_values(["asset_id", "date"], kind="stable")
        .drop_duplicates(subset=["asset_id", "date"], keep="last")
    )

    prev_close = df.groupby("asset_id")["close_adjusted_usd"].shift(1).replace(0.0, np.nan)

    df["ret_log_close_adjusted_usd"] = np.log(df["close_adjusted_usd"] / prev_close)
    df["ret_log_close_adjusted_usd"] = df["ret_log_close_adjusted_usd"].replace([np.inf, -np.inf], np.nan)

    # Backward-compatible aliases. These are now log returns.
    df["ret_close_adjusted_usd"] = df["ret_log_close_adjusted_usd"]
    df["ret_adj_close_usd"] = df["ret_log_close_adjusted_usd"]

    df = df.dropna(subset=["ret_log_close_adjusted_usd"])

    return df[
        [
            "date",
            "asset_id",
            "ticker",
            "ret_log_close_adjusted_usd",
            "ret_close_adjusted_usd",
            "ret_adj_close_usd",
        ]
    ].reset_index(drop=True)


def _write_returns_partitions_replace_manifest(
    *,
    store: MarketStore,
    returns: pd.DataFrame,
    dry_run: bool,
) -> list[str]:
    if returns is None or returns.empty:
        return []

    df = returns.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date", "asset_id"])
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["year"] = df["date"].dt.year.astype(int)

    written: list[str] = []

    for (asset_id, year), g in df.groupby(["asset_id", "year"], sort=False):
        out = g.drop(columns=["year"]).sort_values("date", kind="stable").reset_index(drop=True)
        part = uuid.uuid4().hex[:12]
        key = f"{store.returns_prefix}/asset_id={str(asset_id).strip()}/year={int(year)}/part-logret-{part}.parquet"

        if dry_run:
            print(f"[DRY RUN] Would write returns partition: s3://{store.bucket}/{key} rows={len(out)}")
        else:
            store._put_bytes(key, _parquet_bytes(out), content_type="application/octet-stream")

            # Replace manifest parts with this new canonical log-return part.
            store.write_asset_year_manifest(
                table="returns_usd",
                asset_id=str(asset_id),
                year=int(year),
                dates=[
                    pd.Timestamp(x).date().isoformat()
                    for x in pd.to_datetime(out["date"], errors="coerce").dropna().tolist()
                ],
                parts=[key],
            )

        written.append(key)

    return written


def _write_returns_latest_snapshot_and_state(
    *,
    store: MarketStore,
    all_latest_rows: list[pd.DataFrame],
    dry_run: bool,
) -> tuple[Optional[str], Optional[str]]:
    if not all_latest_rows:
        return None, None

    latest = pd.concat(all_latest_rows, ignore_index=True)
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
    latest = latest.dropna(subset=["date", "asset_id"])
    latest = latest.sort_values(["asset_id", "date"], kind="stable")
    latest = latest.groupby("asset_id", as_index=False, sort=False).tail(1).reset_index(drop=True)

    snapshot_key = f"{store.snapshots_prefix}/latest_returns.parquet"
    state_key = f"{store.state_prefix}/returns_latest.json"

    state = {
        "last_date": None if latest.empty else pd.Timestamp(latest["date"].max()).date().isoformat(),
        "n_assets": int(latest["asset_id"].nunique()) if not latest.empty else 0,
        "return_type": "log",
        "return_column": "ret_log_close_adjusted_usd",
        "compat_columns": ["ret_close_adjusted_usd", "ret_adj_close_usd"],
        "as_of_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if dry_run:
        print(f"[DRY RUN] Would write latest returns snapshot: s3://{store.bucket}/{snapshot_key}")
        print(f"[DRY RUN] Would write returns state: s3://{store.bucket}/{state_key}")
        return snapshot_key, state_key

    store._put_bytes(snapshot_key, _parquet_bytes(latest), content_type="application/octet-stream")
    store._put_bytes(
        state_key,
        json.dumps(state, indent=2, default=str).encode("utf-8"),
        content_type="application/json",
    )

    return snapshot_key, state_key


def rebuild_returns_usd_from_ohlcv(*, cfg: RebuildReturnsConfig) -> dict:
    cfg.__post_init__()

    store = MarketStore(bucket=cfg.bucket, region=cfg.region, base_prefix=cfg.market_root)

    start_ts = _to_day_naive(cfg.start)
    end_ts = _to_day_naive(cfg.end) if cfg.end else _to_day_naive(pd.Timestamp.utcnow())

    asset_ids = _read_universe_asset_ids(
        universe_csv=cfg.universe_csv,
        excluded_csv=cfg.excluded_csv,
        filter_to_active_universe=cfg.filter_to_active_universe,
    )

    if cfg.max_assets is not None:
        asset_ids = asset_ids[: int(cfg.max_assets)]

    print("\n=== REBUILD RETURNS USD FROM OHLCV ===")
    print(f"bucket:          {cfg.bucket}")
    print(f"region:          {cfg.region}")
    print(f"market_root:     {cfg.market_root}")
    print(f"ohlcv_prefix:    {store.ohlcv_prefix}")
    print(f"returns_prefix:  {store.returns_prefix}")
    print(f"universe_csv:    {cfg.universe_csv}")
    print(f"excluded_csv:    {cfg.excluded_csv}")
    print(f"assets:          {len(asset_ids)}")
    print(f"window:          {start_ts.date()}..{end_ts.date()}")
    print(f"workers:         {cfg.max_workers}")
    print(f"dry_run:         {cfg.dry_run}")
    print("return_type:     log")
    print("")

    written_keys: list[str] = []
    latest_rows: list[pd.DataFrame] = []
    failures: list[dict] = []

    kept = 0
    empty = 0
    failed = 0
    total_rows = 0

    def _work(asset_id: str) -> dict:
        try:
            ohlcv = store.read_ohlcv_usd(
                asset_ids=[asset_id],
                start=str(start_ts.date()),
                end=str(end_ts.date()),
                columns=None,
            )

            if ohlcv is None or ohlcv.empty:
                return {"status": "empty", "asset_id": asset_id}

            returns = _compute_log_returns_from_ohlcv(ohlcv)

            if returns is None or returns.empty:
                return {"status": "empty", "asset_id": asset_id}

            return {
                "status": "keep",
                "asset_id": asset_id,
                "returns": returns,
                "n_rows": int(len(returns)),
            }

        except Exception as e:
            return {"status": "fail", "asset_id": asset_id, "error": str(e)[:1200]}

    done = 0

    with ThreadPoolExecutor(max_workers=int(cfg.max_workers)) as ex:
        futures = [ex.submit(_work, aid) for aid in asset_ids]

        for fut in as_completed(futures):
            res = fut.result()
            done += 1

            status = res.get("status")
            aid = str(res.get("asset_id"))

            if status == "keep":
                returns = res["returns"]

                keys = _write_returns_partitions_replace_manifest(
                    store=store,
                    returns=returns,
                    dry_run=bool(cfg.dry_run),
                )
                written_keys.extend(keys)

                latest_rows.append(
                    returns.sort_values("date", kind="stable")
                    .groupby("asset_id", as_index=False, sort=False)
                    .tail(1)
                    .reset_index(drop=True)
                )

                kept += 1
                total_rows += int(res.get("n_rows", 0))

            elif status == "empty":
                empty += 1

            else:
                failed += 1
                failures.append({"asset_id": aid, "error": res.get("error")})

            if done % int(cfg.progress_every) == 0 or done == len(asset_ids):
                print(
                    f"[returns-rebuild] progress {done}/{len(asset_ids)} "
                    f"kept={kept} empty={empty} failed={failed} "
                    f"rows={total_rows} parts={len(written_keys)}"
                )

    latest_snapshot_key, state_key = _write_returns_latest_snapshot_and_state(
        store=store,
        all_latest_rows=latest_rows,
        dry_run=bool(cfg.dry_run),
    )

    meta = {
        "as_of_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bucket": cfg.bucket,
        "region": cfg.region,
        "market_root": cfg.market_root,
        "ohlcv_prefix": store.ohlcv_prefix,
        "returns_prefix": store.returns_prefix,
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "assets_requested": int(len(asset_ids)),
        "assets_kept": int(kept),
        "empty_assets": int(empty),
        "failed_assets": int(failed),
        "return_rows": int(total_rows),
        "written_parts": int(len(written_keys)),
        "return_type": "log",
        "return_column": "ret_log_close_adjusted_usd",
        "compat_columns": ["ret_close_adjusted_usd", "ret_adj_close_usd"],
        "written_parts_sample": written_keys[:50],
        "failures_sample": failures[:50],
    }

    meta_key = f"{store.returns_prefix}/latest_rebuild_log_returns.meta.json"

    if cfg.dry_run:
        print(f"[DRY RUN] Would write meta: s3://{store.bucket}/{meta_key}")
    else:
        store._put_bytes(
            meta_key,
            json.dumps(meta, indent=2, default=str).encode("utf-8"),
            content_type="application/json",
        )

    print("")
    print("[OK] returns rebuild finished")
    print(f"[OK] assets_kept:      {kept}")
    print(f"[OK] return_rows:      {total_rows}")
    print(f"[OK] written_parts:    {len(written_keys)}")
    print(f"[OK] latest_snapshot:  {latest_snapshot_key}")
    print(f"[OK] state_key:        {state_key}")
    print(f"[OK] meta:             s3://{store.bucket}/{meta_key}")
    print("")

    return {
        "written_keys": written_keys,
        "latest_snapshot_key": latest_snapshot_key,
        "state_key": state_key,
        "meta_key": meta_key,
        "meta": meta,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rebuild returns_usd from OHLCV using canonical log returns.")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--bucket", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--market-root", default=None)

    ap.add_argument("--universe-csv", default=None)
    ap.add_argument("--excluded-csv", default=None)
    ap.add_argument("--no-universe-filter", action="store_true")

    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)

    ap.add_argument("--max-assets", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--progress-every", type=int, default=100)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    runtime_cfg: RuntimeConfig = load_runtime_config(args.env)

    if not bool(args.dry_run):
        require_prod_confirmation(runtime_cfg, bool(args.confirm_prod_write))

    bucket = str(args.bucket or runtime_cfg.bucket)
    region = str(args.region or runtime_cfg.region)
    market_root = str(args.market_root or runtime_cfg.market_root).strip("/")

    universe_csv = str(args.universe_csv or (paths.universe_dir() / "universe.csv"))
    excluded_csv = str(args.excluded_csv or (paths.universe_dir() / "asset_excluded.csv"))

    input_args: Dict[str, Any] = vars(args).copy()
    input_args.update(
        {
            "resolved_bucket": bucket,
            "resolved_region": region,
            "resolved_market_root": market_root,
            "resolved_universe_csv": universe_csv,
            "resolved_excluded_csv": excluded_csv,
        }
    )

    with capture_script_run(
        cfg=runtime_cfg,
        script_name="rebuild_returns_usd_from_ohlcv.py",
        input_args=input_args,
        dry_run=bool(args.dry_run),
    ) as run_id:
        build_cfg = RebuildReturnsConfig(
            bucket=bucket,
            region=region,
            market_root=market_root,
            universe_csv=universe_csv,
            excluded_csv=excluded_csv,
            filter_to_active_universe=(not bool(args.no_universe_filter)),
            start=str(args.start),
            end=(str(args.end) if args.end else None),
            max_assets=args.max_assets,
            max_workers=int(args.max_workers),
            progress_every=int(args.progress_every),
            dry_run=bool(args.dry_run),
        )

        result = rebuild_returns_usd_from_ohlcv(cfg=build_cfg)

        audit = build_audit_event(
            cfg=runtime_cfg,
            run_id=run_id,
            event_type="rebuild_dataset",
            entity_type="returns_usd",
            entity_id="returns_usd_v1_log_returns",
            as_of=None,
            source_script="rebuild_returns_usd_from_ohlcv.py",
            source_mode="rebuild_log_returns",
            status=("dry_run" if args.dry_run else "success"),
            reason="rebuild returns_usd from OHLCV using canonical log returns",
            input_args=input_args,
            output_keys=[
                str(result["latest_snapshot_key"]),
                str(result["state_key"]),
                str(result["meta_key"]),
            ],
            metadata={
                **dict(result["meta"]),
                "written_keys_sample": [str(k) for k in result["written_keys"][:100]],
            },
        )
        write_audit_event(cfg=runtime_cfg, event=audit, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()