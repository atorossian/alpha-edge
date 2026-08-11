# compute_market_regime.py
from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run

import argparse
import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from alpha_edge import paths
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import RuntimeConfig
from alpha_edge.market.hmm_engine import (
    GaussianHMM,
    compute_state_diagnostics,
    label_states_4,
    regime_probs_from_state_probs,
    select_regime_label,
)
from alpha_edge.market.regime_filter import RegimeFilterState
from alpha_edge.market.regime_leverage import leverage_from_hmm
from alpha_edge.portfolio.report_engine import print_hmm_summary


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT = "engine/v1"
DEFAULT_MARKET_ROOT = "market"

# Option B: composite proxy equal-weight basket.
PROXY_TICKERS = ["VT", "SPY", "QQQ", "IWM", "TLT", "VCIT", "GLD"]

# History window for regime fitting.
START_HISTORY = "2015-01-01"


def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET)).strip()


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION)).strip()


def cfg_engine_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "engine_root", DEFAULT_ENGINE_ROOT)).strip("/")


def cfg_market_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "market_root", DEFAULT_MARKET_ROOT)).strip("/")


def make_market_store(cfg: RuntimeConfig) -> MarketStore:
    """
    Runtime-aware MarketStore constructor.

    If MarketStore does not yet accept base_prefix, this falls back to the older
    constructor. Ideally MarketStore should also become fully runtime-aware.
    """
    try:
        return MarketStore(
            bucket=cfg_bucket(cfg),
            region=cfg_region(cfg),
            base_prefix=cfg_market_root(cfg),
        )
    except TypeError:
        return MarketStore(
            bucket=cfg_bucket(cfg),
            region=cfg_region(cfg),
        )


def _to_day(x) -> pd.Timestamp:
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {x!r}")
    ts = pd.Timestamp(ts)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _load_universe_ticker_to_asset_id(universe_path: str | None = None) -> dict[str, str]:
    path = universe_path or str(paths.universe_dir() / "universe.csv")
    df = pd.read_csv(path)

    if df is None or df.empty:
        raise RuntimeError(f"Universe is empty: {path}")

    for c in ["ticker", "asset_id"]:
        if c not in df.columns:
            raise RuntimeError(f"Universe missing required column {c!r}: {path}")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()

    if "include" in df.columns:
        df["include"] = pd.to_numeric(df["include"], errors="coerce").fillna(1).astype(int)
    else:
        df["include"] = 1

    df = df[(df["ticker"] != "") & (df["asset_id"] != "")].copy()
    if df.empty:
        raise RuntimeError(f"Universe has no valid ticker/asset_id rows after normalization: {path}")

    df = df.sort_values(["ticker", "include"], ascending=[True, False])
    df = df.drop_duplicates(subset=["ticker"], keep="first")

    return dict(zip(df["ticker"].tolist(), df["asset_id"].tolist()))


def _pick_price_column(schema_names: set[str]) -> str:
    for c in ["adj_close_usd", "close_raw_usd", "close_usd", "adj_close", "close", "Adj Close", "Close"]:
        if c in schema_names:
            return c
    raise RuntimeError(f"OHLCV dataset missing usable close column. schema={sorted(schema_names)}")


def _load_closes_usd_from_ohlcv(
    *,
    tickers: list[str],
    start: str,
    end: str,
    cfg: RuntimeConfig,
    universe_path: str | None = None,
) -> pd.DataFrame:
    """
    Load close USD prices for tickers from the OHLCV parquet dataset and pivot wide.

    Supports both layouts:
      1. legacy/schema with ticker column
      2. current preferred schema partitioned by asset_id/year
    """
    bucket = cfg_bucket(cfg)
    market_root = cfg_market_root(cfg)

    start_ts = _to_day(start)
    end_ts = _to_day(end)

    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    if not tickers:
        return pd.DataFrame()

    ohlcv_root = f"s3://{bucket}/{market_root}/ohlcv_usd/v1"
    dataset = ds.dataset(ohlcv_root, format="parquet", partitioning="hive")

    schema_names = set(dataset.schema.names)
    price_col = _pick_price_column(schema_names)

    if "date" not in schema_names:
        raise RuntimeError(f"OHLCV dataset missing date column. schema={sorted(schema_names)}")

    using_asset_id = "asset_id" in schema_names
    using_ticker = "ticker" in schema_names

    if not using_asset_id and not using_ticker:
        raise RuntimeError(
            "OHLCV dataset must contain either asset_id or ticker column. "
            f"schema={sorted(schema_names)}"
        )

    ticker_to_asset: dict[str, str] = {}
    asset_to_ticker: dict[str, str] = {}

    if using_asset_id:
        ticker_to_asset = _load_universe_ticker_to_asset_id(universe_path)
        missing = [t for t in tickers if t not in ticker_to_asset]
        if missing:
            raise RuntimeError(
                "Proxy tickers missing from universe ticker->asset_id mapping: "
                + ", ".join(missing)
            )

        asset_ids = [ticker_to_asset[t] for t in tickers]
        asset_to_ticker = {aid: t for t, aid in ticker_to_asset.items() if t in tickers}
        filt = ds.field("asset_id").isin(asset_ids)
        columns = ["date", "asset_id", price_col]
    else:
        filt = ds.field("ticker").isin(tickers)
        columns = ["date", "ticker", price_col]

    table = dataset.to_table(filter=filt, columns=columns)
    df = table.to_pandas()

    if df is None or df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()

    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    if df.empty:
        return pd.DataFrame()

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    if using_asset_id:
        df["asset_id"] = df["asset_id"].astype(str).str.strip()
        df["ticker"] = df["asset_id"].map(asset_to_ticker).fillna(df["asset_id"])
    else:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    df = df.sort_values(["date", "ticker"])
    dup = df.duplicated(subset=["date", "ticker"], keep=False)
    if dup.any():
        n_dup = int(dup.sum())
        print(f"[ohlcv][warn] duplicate (date,ticker) rows detected: {n_dup}. Collapsing by last().")
        df = df.groupby(["date", "ticker"], as_index=False)[price_col].last()

    closes = (
        df.set_index(["date", "ticker"])[price_col]
        .sort_index()
        .unstack("ticker")
        .sort_index()
        .ffill()
    )

    return closes


def _compute_equal_weight_proxy_returns(
    closes: pd.DataFrame,
    *,
    min_assets_per_day: int = 3,
) -> tuple[pd.Series, dict]:
    """
    Compute equal-weight composite returns from wide closes.

    Returns:
      proxy_returns, meta
    """
    if closes is None or closes.empty:
        return (
            pd.Series(dtype="float64"),
            {"kept": [], "dropped": [], "min_assets_per_day": int(min_assets_per_day)},
        )

    rets = closes.pct_change()

    available = rets.notna().sum(axis=1)
    rets_ok = rets[available >= int(min_assets_per_day)].copy()

    proxy = rets_ok.mean(axis=1, skipna=True).dropna()

    kept = [c for c in closes.columns if closes[c].dropna().shape[0] >= 50]
    dropped = [c for c in closes.columns if c not in kept]

    meta = {
        "tickers_requested": list(closes.columns),
        "min_assets_per_day": int(min_assets_per_day),
        "n_days_raw": int(rets.shape[0]),
        "n_days_used": int(proxy.shape[0]),
        "assets_present_sample": {
            str(d.date()): int(available.loc[d])
            for d in available.index[-5:]
        },
        "kept": kept,
        "dropped": dropped,
    }

    return proxy.astype("float64"), meta



def _fit_market_hmm_point_in_time(
    *,
    proxy_rets: pd.Series,
    as_of: str,
    proxy_meta: dict[str, Any],
    commit_threshold: float = 0.65,
    min_obs_after_vol: int = 80,
) -> tuple[str, dict[str, Any]]:
    """
    Fit the market HMM using only proxy returns available up to `as_of`.

    This is the single source of truth for point-in-time regime calculation.
    It intentionally slices the input series to `index <= as_of` before fitting
    so historical backfills cannot use future observations.
    """
    as_of_ts = _to_day(as_of)

    s = proxy_rets.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna().sort_index()
    s = s[s.index <= as_of_ts].copy()

    if s.empty:
        raise RuntimeError(f"Market regime: no proxy returns available up to {as_of}")

    last_date = pd.Timestamp(s.index[-1]).normalize()

    if last_date > as_of_ts:
        raise RuntimeError(
            f"Lookahead violation: last_date={last_date.date()} > as_of={as_of_ts.date()}"
        )

    r = s.to_numpy(dtype=np.float64)
    vol20 = (
        pd.Series(r, index=s.index)
        .rolling(20)
        .std()
        .to_numpy(dtype=np.float64)
    )

    mask = np.isfinite(vol20)
    X = np.column_stack([r[mask], vol20[mask]])

    if X.shape[0] < int(min_obs_after_vol):
        raise RuntimeError(
            f"Market regime: not enough observations after vol window. "
            f"as_of={as_of} X={X.shape}"
        )

    r_aligned = r[mask]

    hmm = GaussianHMM(n_states=4, n_dim=2, seed=42)
    fit_res = hmm.fit(X, max_iter=150, tol=1e-4, verbose=False)

    filtered = hmm.predict_proba(X)
    p_today = filtered[-1]

    diags = compute_state_diagnostics(r_aligned, filtered)
    mapping = label_states_4(diags)
    p_label_today = regime_probs_from_state_probs(p_today, mapping)
    label_commit = select_regime_label(
        p_label_today,
        commit_threshold=float(commit_threshold),
    )

    as_of_market_date = last_date.strftime("%Y-%m-%d")

    hmm_res = {
        "n_states": 4,
        "obs_dim": 2,
        "loglik": float(fit_res.loglik),
        "n_iter": int(fit_res.n_iter),
        "converged": bool(fit_res.converged),
        "p_state_today": [float(x) for x in p_today],
        "state_to_label": {str(k): v for k, v in mapping.items()},
        "p_label_today": {k: float(v) for k, v in p_label_today.items()},
        "label_commit": label_commit,
        "state_diagnostics": {
            str(k): {
                "drift": float(diags[k].drift),
                "vol": float(diags[k].vol),
                "neg_rate": float(diags[k].neg_rate),
                "weight": float(diags[k].weight),
            }
            for k in range(4)
        },
        "params": {
            "pi": [float(x) for x in fit_res.params.pi],
            "A": [[float(x) for x in row] for row in fit_res.params.A],
            "means": [[float(x) for x in row] for row in fit_res.params.means],
            "vars": [[float(x) for x in row] for row in fit_res.params.vars],
        },
        "meta": {
            "uses": "filtered_probs",
            "commit_threshold": float(commit_threshold),
            "features": ["proxy_return_eqw", "vol20"],
            "last_date_used": as_of_market_date,
            "point_in_time": True,
            "lookahead_safe": True,
            "proxy": {
                "method": "equal_weight_basket",
                "tickers": PROXY_TICKERS,
            },
            "proxy_meta": proxy_meta,
        },
    }

    return as_of_market_date, hmm_res


def _candidate_regime_dates(
    *,
    proxy_rets: pd.Series,
    start: str | None,
    end: str | None,
) -> list[str]:
    s = proxy_rets.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna().sort_index()

    if start is not None:
        s = s[s.index >= _to_day(start)]

    if end is not None:
        s = s[s.index <= _to_day(end)]

    return [
        pd.Timestamp(x).normalize().strftime("%Y-%m-%d")
        for x in s.index
    ]


def _build_market_regime_payload(
    *,
    cfg: RuntimeConfig,
    bucket: str,
    engine_root: str,
    market_root: str,
    as_of_market_date: str,
    as_of_run_date: str,
    end_date_requested: str,
    start_history: str,
    hmm_res: dict[str, Any],
    lev_rec: dict[str, Any],
) -> dict[str, Any]:
    pricing_as_of_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "as_of": as_of_market_date,
        "proxy": {
            "method": "equal_weight_basket",
            "tickers": PROXY_TICKERS,
        },
        "hmm": hmm_res,
        "leverage_recommendation": lev_rec,
        "meta": {
            "env": getattr(cfg, "env", None),
            "as_of_market_date": as_of_market_date,
            "as_of_run_date": as_of_run_date,
            "pricing_as_of_utc": pricing_as_of_utc,
            "end_date_requested": end_date_requested,
            "start_history": start_history,
            "bucket": bucket,
            "engine_root": engine_root,
            "market_root": market_root,
            "ohlcv_root": f"s3://{bucket}/{market_root}/ohlcv_usd/v1",
            "point_in_time": True,
            "lookahead_safe": True,
        },
    }


def _stateless_filter_state() -> RegimeFilterState:
    """
    Backfill helper: never inject today's live regime filter state into a
    historical row. Each historical payload must remain point-in-time safe.
    """
    return RegimeFilterState(
        last_date=None,
        chosen_label=None,
        days_in_regime=0,
        probs_smoothed=None,
    )


def _compute_leverage_recommendation(
    *,
    hmm_res: dict[str, Any],
    filter_state: RegimeFilterState,
    as_of_market_date: str,
) -> dict[str, Any]:
    return leverage_from_hmm(
        hmm_res or {},
        default=1.0,
        risk_appetite=0.6,
        low_confidence_floor=0.2,
        hard_cap=12.0,
        filter_state=filter_state,
        as_of=as_of_market_date,
        filter_alpha=0.20,
        min_hold_days=3,
        min_prob_to_switch=0.60,
        min_margin_to_switch=0.12,
    )


def _backfill_missing_market_regimes(
    *,
    cfg: RuntimeConfig,
    market: MarketStore,
    engine_store: MarketStore,
    proxy_rets: pd.Series,
    proxy_meta: dict[str, Any],
    start: str | None,
    end: str | None,
    start_history: str,
    write_outputs: bool,
) -> dict[str, Any]:
    """
    Fill missing dates under the existing regime path:
        engine/v1/regimes/market_hmm/dt=YYYY-MM-DD/regime.json

    Each missing date is computed point-in-time using only returns available up
    to that same date. Historical rows do not update latest.json and do not use
    today's live persisted regime filter state.
    """
    bucket = cfg_bucket(cfg)
    engine_root = cfg_engine_root(cfg)
    market_root = cfg_market_root(cfg)

    candidate_dates = _candidate_regime_dates(
        proxy_rets=proxy_rets,
        start=start,
        end=end,
    )

    existing_dates = set(engine_store.list_market_hmm_regime_dates())
    missing_dates = [d for d in candidate_dates if d not in existing_dates]

    written = 0
    skipped = 0
    errors: list[dict[str, Any]] = []

    print(
        f"[PIT regime backfill] candidates={len(candidate_dates)} "
        f"existing={len(existing_dates)} missing={len(missing_dates)}"
    )

    for i, d in enumerate(missing_dates, start=1):
        try:
            as_of_market_date, hmm_res = _fit_market_hmm_point_in_time(
                proxy_rets=proxy_rets,
                as_of=d,
                proxy_meta=proxy_meta,
            )

            lev_rec = _compute_leverage_recommendation(
                hmm_res=hmm_res,
                filter_state=_stateless_filter_state(),
                as_of_market_date=as_of_market_date,
            )

            payload = _build_market_regime_payload(
                cfg=cfg,
                bucket=bucket,
                engine_root=engine_root,
                market_root=market_root,
                as_of_market_date=as_of_market_date,
                as_of_run_date=d,
                end_date_requested=d,
                start_history=start_history,
                hmm_res=hmm_res,
                lev_rec=lev_rec,
            )
            payload["meta"]["backfill_filter_policy"] = "stateless_filter_reinitialized_for_pit_backfill"
            payload["meta"]["backfill_source"] = "fill_missing_history"

            if write_outputs:
                engine_store.write_market_hmm_regime(
                    as_of=d,
                    payload=payload,
                    update_latest=False,
                )

            written += 1

        except Exception as exc:
            skipped += 1
            errors.append(
                {
                    "date": d,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        if i % 25 == 0:
            print(
                f"[PIT regime backfill] processed={i}/{len(missing_dates)} "
                f"written={written} skipped={skipped}"
            )

    return {
        "candidate_dates": int(len(candidate_dates)),
        "existing_dates": int(len(existing_dates)),
        "missing_dates": int(len(missing_dates)),
        "written": int(written),
        "skipped": int(skipped),
        "error_sample": errors[:20],
        "start": start,
        "end": end,
        "path": f"{engine_root}/regimes/market_hmm",
        "point_in_time": True,
        "lookahead_safe": True,
        "write_outputs": bool(write_outputs),
    }


def compute_market_regime(
    *,
    cfg: RuntimeConfig | None = None,
    env: str | None = None,
    as_of: str | None = None,
    start_history: str = START_HISTORY,
    universe_path: str | None = None,
    write_outputs: bool = True,
    update_latest: bool = True,
    confirm_prod_write: bool = False,
    fill_missing_history: bool = False,
    backfill_start: str | None = None,
    backfill_end: str | None = None,
) -> dict[str, Any]:
    if cfg is None:
        cfg = load_runtime_config(env)

    if write_outputs:
        require_prod_confirmation(cfg, bool(confirm_prod_write))

    bucket = cfg_bucket(cfg)
    region = cfg_region(cfg)
    engine_root = cfg_engine_root(cfg)
    market_root = cfg_market_root(cfg)

    run_dt = pd.Timestamp(as_of or dt.date.today()).tz_localize(None).normalize()
    as_of_run_date = run_dt.strftime("%Y-%m-%d")

    market = make_market_store(cfg)
    engine_store = MarketStore(
        bucket=bucket,
        region=region,
        base_prefix=engine_root,
    )

    print("\n=== COMPUTE MARKET REGIME ===")
    print(f"env:          {getattr(cfg, 'env', 'unknown')}")
    print(f"bucket:       {bucket}")
    print(f"region:       {region}")
    print(f"engine_root:  {engine_root}")
    print(f"market_root:  {market_root}")
    print(f"write_outputs:{bool(write_outputs)}")
    print(f"fill_missing_history:{bool(fill_missing_history)}")
    print("")

    # Use returns latest state if present, so regime aligns to ingestion output.
    latest_state = market.read_returns_latest_state() or {}
    end_date = str(as_of or latest_state.get("last_date") or as_of_run_date)

    closes = _load_closes_usd_from_ohlcv(
        tickers=PROXY_TICKERS,
        start=start_history,
        end=end_date,
        cfg=cfg,
        universe_path=universe_path,
    )

    if closes is None or closes.empty:
        raise RuntimeError(
            "Market regime: no closes available for proxy tickers in OHLCV dataset. "
            f"Expected data under s3://{bucket}/{market_root}/ohlcv_usd/v1"
        )

    proxy_rets, proxy_meta = _compute_equal_weight_proxy_returns(closes, min_assets_per_day=3)
    if proxy_rets.empty or proxy_rets.shape[0] < 120:
        raise RuntimeError(f"Market regime: insufficient proxy return history. n={proxy_rets.shape[0]}")

    as_of_market_date, hmm_res = _fit_market_hmm_point_in_time(
        proxy_rets=proxy_rets,
        as_of=end_date,
        proxy_meta=proxy_meta,
    )

    st_raw = market.read_regime_filter_state() or {}
    filter_state = RegimeFilterState(
        last_date=st_raw.get("last_date"),
        chosen_label=st_raw.get("chosen_label"),
        days_in_regime=int(st_raw.get("days_in_regime", 0) or 0),
        probs_smoothed=st_raw.get("probs_smoothed"),
    )

    lev_rec = _compute_leverage_recommendation(
        hmm_res=hmm_res,
        filter_state=filter_state,
        as_of_market_date=as_of_market_date,
    )

    if write_outputs and isinstance(lev_rec.get("filter_state"), dict):
        market.write_regime_filter_state(lev_rec["filter_state"])

    payload = _build_market_regime_payload(
        cfg=cfg,
        bucket=bucket,
        engine_root=engine_root,
        market_root=market_root,
        as_of_market_date=as_of_market_date,
        as_of_run_date=as_of_run_date,
        end_date_requested=end_date,
        start_history=start_history,
        hmm_res=hmm_res,
        lev_rec=lev_rec,
    )

    if fill_missing_history:
        payload["meta"]["fill_missing_history_requested"] = True
        payload["meta"]["backfill_start"] = backfill_start
        payload["meta"]["backfill_end"] = backfill_end

    written_keys: list[str] = []

    if write_outputs:
        written_keys = engine_store.write_market_hmm_regime(
            as_of=as_of_run_date,
            payload=payload,
            update_latest=update_latest,
        )

    backfill_meta: dict[str, Any] | None = None

    if fill_missing_history:
        backfill_meta = _backfill_missing_market_regimes(
            cfg=cfg,
            market=market,
            engine_store=engine_store,
            proxy_rets=proxy_rets,
            proxy_meta=proxy_meta,
            start=backfill_start,
            end=backfill_end or as_of_market_date,
            start_history=start_history,
            write_outputs=write_outputs,
        )
        payload["meta"]["backfill_missing_history"] = backfill_meta

    print_hmm_summary(hmm_res, lev_rec=lev_rec)

    if write_outputs:
        print(f"[OK] wrote market regime -> s3://{bucket}/{engine_root}/regimes/market_hmm")
        if written_keys:
            print(f"[OK] written_keys={written_keys}")
        if update_latest:
            print("[OK] latest.json updated")
    else:
        print("[NO-WRITE] market regime computed but not written")

    if backfill_meta is not None:
        print(f"[OK] PIT missing-history backfill meta={backfill_meta}")

    print(f"[OK] as_of_market_date={as_of_market_date} target_leverage={float(lev_rec.get('leverage', 1.0)):.2f}x")

    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute Alpha Edge market regime HMM.")

    ap.add_argument("--as-of", default=None, help="As-of/run date YYYY-MM-DD. Default: today or latest market state.")
    ap.add_argument("--start-history", default=START_HISTORY)
    ap.add_argument("--universe-path", default=None)

    ap.add_argument("--no-write", action="store_true", help="Compute but do not write outputs.")
    ap.add_argument("--no-latest", action="store_true", help="Write partition only; do not update latest.json.")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument(
        "--fill-missing-history",
        action="store_true",
        help=(
            "Backfill missing dt=YYYY-MM-DD/regime.json files under the existing "
            "regimes/market_hmm path. Each date is computed point-in-time using "
            "only data up to that date."
        ),
    )
    ap.add_argument(
        "--backfill-start",
        default=None,
        help="Optional start date for --fill-missing-history.",
    )
    ap.add_argument(
        "--backfill-end",
        default=None,
        help="Optional end date for --fill-missing-history.",
    )

    return ap.parse_args()


def _main_impl() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)
    write_outputs = not bool(args.no_write)

    if write_outputs:
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    compute_market_regime(
        cfg=cfg,
        as_of=args.as_of,
        start_history=str(args.start_history),
        universe_path=args.universe_path,
        write_outputs=write_outputs,
        update_latest=(not bool(args.no_latest)),
        confirm_prod_write=bool(args.confirm_prod_write),
        fill_missing_history=bool(args.fill_missing_history),
        backfill_start=args.backfill_start,
        backfill_end=args.backfill_end,
    )


# ----------------------------
# Audit/logging entrypoint wrapper
# ----------------------------
def _tier1_audit_is_dry_run(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "dry_run", False) or getattr(args, "no_write", False))


def main_with_audit() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = _tier1_audit_is_dry_run(args)

    with capture_script_run(
        cfg=cfg,
        script_name="compute_market_regime.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            _main_impl()

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="market_regime",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="compute_market_regime.py",
                source_mode="market_regime",
                status=("dry_run" if is_dry_run else "success"),
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                    "fill_missing_history": bool(getattr(args, "fill_missing_history", False)),
                    "backfill_start": getattr(args, "backfill_start", None),
                    "backfill_end": getattr(args, "backfill_end", None),
                    "note": "Tier 1 audit event is entrypoint-level. Detailed output keys/row counts are available in the script log stdout and script-specific metadata where emitted by the script.",
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
        except Exception as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="market_regime",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="compute_market_regime.py",
                source_mode="market_regime",
                status="failed",
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                    "fill_missing_history": bool(getattr(args, "fill_missing_history", False)),
                    "backfill_start": getattr(args, "backfill_start", None),
                    "backfill_end": getattr(args, "backfill_end", None),
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise


def main() -> None:
    main_with_audit()


if __name__ == "__main__":
    main()
