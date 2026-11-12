# run_daily_report.py
from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run

import argparse
import datetime as dt
from dataclasses import asdict
from typing import Dict, Any

import numpy as np
import pandas as pd

import yfinance as yf
import json
import io
import boto3

from alpha_edge import paths

from alpha_edge.portfolio.take_profit import (
    TakeProfitConfig,
    TakeProfitState,
    take_profit_policy,
)
from alpha_edge.portfolio.execution_engine import weights_to_discrete_shares
from alpha_edge.market.regime_filter import RegimeFilterState
from alpha_edge.core.schemas import ScoreConfig, Position
from alpha_edge.market.regime_leverage import leverage_from_hmm
from alpha_edge.portfolio.report_engine import (
    build_portfolio_report,
    summarize_report,
    print_hmm_summary,
    print_decision_addendum,
)
from alpha_edge.portfolio.reinvest_engine import reinvest_leftover_with_frozen_core

from alpha_edge.risk.actuarial.portfolio_search_output import (
    build_actuarial_diagnostic_from_portfolio_report,
    maybe_print_actuarial_diagnostic_section,
)

from alpha_edge.market.hmm_engine import (
    GaussianHMM,
    compute_state_diagnostics,
    label_states_4,
    regime_probs_from_state_probs,
    select_regime_label,
)
from alpha_edge.market.build_returns_wide_cache import build_returns_wide_cache, CacheConfig

from alpha_edge.portfolio.portfolio_health import (
    build_portfolio_health,
    should_reoptimize,
)
from alpha_edge.portfolio.evaluation_service import (
    build_evaluation_metadata,
    build_plausibility_guards,
    build_portfolio_behavior_regime,
    build_daily_report_execution_signals,
    compute_portfolio_health_score as canonical_compute_portfolio_health_score,
)
from alpha_edge.portfolio.alpha_report import format_alpha_report
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.data_loader import (
    s3_init,
    s3_load_latest_json,
    s3_load_latest_report_score,
    s3_write_json_event,
    s3_write_parquet_partition,
    parse_portfolio_health,
    parse_ledger_positions_obj,
    clean_returns_matrix,
    s3_get_json,
    s3_load_latest_json_asof,
)
from alpha_edge.portfolio.equity_valuation import compute_live_equity_from_ledger_and_prices as _canonical_compute_live_equity_from_ledger_and_prices
from alpha_edge.portfolio.rebalance_engine import (
    RebalanceState,
    should_rebalance,
    build_rescale_plan,
    compute_gross_notional_from_positions,
)
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation

DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT_PREFIX = "engine/v1"
DEFAULT_MARKET_ROOT = "market"

TAKE_PROFIT_STATE_TABLE = "take_profit/state"
TAKE_PROFIT_PLAN_TABLE = "take_profit/plan"

# per-asset TP state/plan
TAKE_PROFIT_ASSETS_STATE_TABLE = "take_profit/assets_state"
TAKE_PROFIT_ASSETS_PLAN_TABLE = "take_profit/assets_plan"

MARKET_RESCALE_STATE_TABLE = "regimes/market_rescale_state"
TRANSITION_ASSESSMENT_TABLE = "portfolio_transition/assessment"


def _resolve_root_prefix(*, engine_root: str, backtest_run_id: str | None) -> str:
    engine_root = str(engine_root).strip("/")
    if backtest_run_id:
        return f"{engine_root}/backtests/{backtest_run_id}"
    return engine_root


def _load_returns_wide_cache(
    *, bucket: str, market_root: str, as_of_ts: pd.Timestamp, refresh: bool = False
) -> pd.DataFrame:
    if refresh:
        cache_cfg = CacheConfig(bucket=bucket, min_years=float(5.0))
        build_returns_wide_cache(cache_cfg)
    else:
        print("[returns_wide] using existing S3 cache; pass --refresh-returns-cache to rebuild")
    path = f"s3://{bucket}/{market_root.strip('/')}/cache/v1/returns_wide_min5y.parquet"
    df = pd.read_parquet(path, engine="pyarrow").sort_index()
    df, _ = clean_returns_matrix(df)
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None).normalize()
    df = df.loc[df.index <= as_of_ts]
    if df.shape[0] < 252:
        raise RuntimeError(f"Not enough returns history up to as_of={as_of_ts.date()}: rows={df.shape[0]}")
    return df


def _compute_live_equity_from_ledger_and_prices(*, pnl_summary: dict, spot_rows: list[dict], prices_for_valuation: pd.Series) -> float:
    # Backward-compatible wrapper. Canonical implementation lives in
    # alpha_edge.portfolio.equity_valuation so all jobs can resolve equity
    # consistently instead of duplicating daily-report-specific logic.
    equity = _canonical_compute_live_equity_from_ledger_and_prices(
        pnl_summary=pnl_summary,
        spot_rows=spot_rows,
        prices_for_valuation=prices_for_valuation,
    )
    net_cashflow = float(pnl_summary.get("net_cashflow_usd", 0.0) or 0.0)
    realized = float(pnl_summary.get("realized_pnl", pnl_summary.get("realized_pnl_usd", 0.0)) or 0.0)
    dividends = float(pnl_summary.get("dividends_pnl_usd", 0.0) or 0.0)
    live_unrealized = float(equity) - net_cashflow - realized - dividends
    print("[equity] source=canonical_equity_valuation "
          f"net_cashflow={net_cashflow:.2f} realized={realized:.2f} dividends={dividends:.2f} "
          f"live_unrealized={live_unrealized:.2f} equity={equity:.2f}")
    return float(equity)


def _asset_id_ticker_maps_from_ledger_rows(*, spot_rows: list[dict], deriv_rows: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    asset_to_ticker, ticker_to_asset = {}, {}
    for r in list(spot_rows or []) + list(deriv_rows or []):
        aid = str(r.get("asset_id") or "").strip()
        t = str(r.get("ticker") or "").upper().strip()
        if not aid or not t:
            continue
        if t in ticker_to_asset and ticker_to_asset[t] != aid:
            raise RuntimeError(f"Ticker {t} maps to multiple live asset_ids: {ticker_to_asset[t]} and {aid}")
        asset_to_ticker[aid] = t
        ticker_to_asset[t] = aid
    return asset_to_ticker, ticker_to_asset


def _build_live_augmented_returns_for_portfolio(
    *, returns_wide: pd.DataFrame, spot_rows: list[dict], deriv_rows: list[dict],
    latest_close_prices: pd.Series, prices_for_valuation: pd.Series, as_of_run_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the canonical daily-report return matrix.

    The returned DataFrame is asset_id-keyed. Tickers are retained only as
    display/provider aliases in metadata and for live price lookup. This keeps
    daily-report evaluation aligned with portfolio search, where asset_id is the
    canonical identity and ticker/yahoo_ticker are not trusted as unique keys.
    """
    asset_to_ticker, ticker_to_asset = _asset_id_ticker_maps_from_ledger_rows(spot_rows=spot_rows, deriv_rows=deriv_rows)
    if not asset_to_ticker:
        raise RuntimeError("No asset_id/ticker pairs in ledger positions for returns_wide evaluation.")

    requested_asset_ids = list(asset_to_ticker.keys())
    missing = [aid for aid in requested_asset_ids if aid not in returns_wide.columns]
    if missing:
        raise RuntimeError("Portfolio asset_ids missing from returns_wide: " + ", ".join(missing[:20]))

    df = returns_wide[requested_asset_ids].copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="any")

    live: dict[str, float] = {}
    missing_live: list[str] = []
    for aid in df.columns:
        t = asset_to_ticker.get(str(aid), str(aid))
        last_close = latest_close_prices.get(t, np.nan)
        live_px = prices_for_valuation.get(t, np.nan)
        if not np.isfinite(float(last_close)) or float(last_close) <= 0 or not np.isfinite(float(live_px)):
            missing_live.append(f"{aid} ({t})")
            continue
        live[str(aid)] = float(live_px) / float(last_close) - 1.0
    if missing_live:
        raise RuntimeError("Cannot append live returns row; missing prices for " + ", ".join(sorted(set(missing_live))))

    live_idx = pd.Timestamp(as_of_run_date).tz_localize(None).normalize()
    if not df.empty:
        last_idx = pd.Timestamp(df.index.max()).tz_localize(None).normalize()
        if live_idx <= last_idx:
            live_idx = last_idx + pd.Timedelta(days=1)
    df = pd.concat([df, pd.DataFrame([live], index=[live_idx])], axis=0).sort_index()
    df = df[~df.index.duplicated(keep="last")].dropna(how="any")

    meta = {
        "source": "returns_wide_plus_live_mark_row",
        "key_type_internal": "asset_id",
        "display_key_type": "ticker",
        "columns": list(df.columns),
        "asset_id_columns": list(df.columns),
        "asset_id_to_ticker": asset_to_ticker,
        "asset_id_by_ticker": ticker_to_asset,
        "last_historical_return_date": None if len(df.index) <= 1 else str(pd.Timestamp(df.index[-2]).date()),
        "live_return_date": str(pd.Timestamp(live_idx).date()),
        "live_returns_by_asset_id": {k: float(v) for k, v in live.items()},
        "live_returns_by_ticker": {asset_to_ticker.get(k, k): float(v) for k, v in live.items()},
        "rows": int(len(df)),
    }
    print(f"[returns_eval] source=returns_wide_plus_live_row key_type=asset_id assets={len(df.columns)} rows={len(df)} live_date={meta['live_return_date']}")
    return df, meta


def _clamp01(x: float) -> float:
    try: v = float(x)
    except Exception: return 0.0
    return 0.0 if not np.isfinite(v) else float(min(1.0, max(0.0, v)))


def _safe_ratio_good(value: float, cap: float) -> float:
    try: v, c = abs(float(value)), abs(float(cap))
    except Exception: return 0.0
    if not np.isfinite(v) or not np.isfinite(c) or c <= 0: return 0.0
    return _clamp01(1.0 - v / c)


_HEALTH_LATEST_EXTRA_KEYS = {
    "schema_version",
    "health_score",
    "normalized_health_score",
    "portfolio_health_score",
    "health_grade",
    "raw_score",
    "raw_optimizer_score",
    "score_semantics",
    "health_score_semantics",
    "health_score_payload",
    "legacy_portfolio_health",
    "meta",
    "evaluation_metadata",
    "plausibility",
}


def _parse_portfolio_health_compat(raw: dict | None):
    """
    Parse the legacy PortfolioHealth payload while tolerating the extra
    normalized-health fields added to health/latest.json.

    This keeps the daily-report baseline logic backward-compatible even after
    health/latest.json starts carrying explicit normalized health_score fields.
    """
    if not raw:
        return None

    try:
        return parse_portfolio_health(raw)
    except TypeError:
        if not isinstance(raw, dict):
            raise

        cleaned = {
            k: v
            for k, v in raw.items()
            if k not in _HEALTH_LATEST_EXTRA_KEYS
        }
        return parse_portfolio_health(cleaned)


def _none_if_not_finite(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _build_health_latest_payload(
    *,
    current_health,
    health_score_payload: dict[str, Any],
    as_of_market_date: str,
    as_of_run_date: str,
    pricing_as_of_utc: str,
    returns_eval_meta: dict[str, Any],
    evaluation_metadata: dict[str, Any] | None = None,
    plausibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the canonical health/latest.json payload.

    Rules:
    - `health_score` is normalized 0-100 and is safe for transition logic.
    - `score` remains the legacy/raw daily health/evaluator score and may be negative.
    - `raw_score` and `raw_optimizer_score` make the raw score semantics explicit.
    """
    legacy = asdict(current_health)

    normalized_health_score = _none_if_not_finite(
        health_score_payload.get("health_score")
    )
    raw_optimizer_score = _none_if_not_finite(
        health_score_payload.get("raw_optimizer_score")
    )

    out = {
        **legacy,

        # Explicit normalized health fields for consumers such as transition
        # assessment and shadow validation.
        "schema_version": "portfolio_health_latest_v2",
        "health_score": normalized_health_score,
        "normalized_health_score": normalized_health_score,
        "portfolio_health_score": normalized_health_score,
        "health_grade": health_score_payload.get("health_grade"),

        # Keep the old `score` field from `current_health` for backward
        # compatibility, but expose explicit raw aliases so consumers do not
        # confuse it with normalized health.
        "raw_score": raw_optimizer_score,
        "raw_optimizer_score": raw_optimizer_score,
        "score_semantics": "legacy_raw_daily_report_score_not_normalized",
        "health_score_semantics": "normalized_0_100_daily_report_health_score",

        # Full explainability payload.
        "health_score_payload": dict(health_score_payload),
        "evaluation_metadata": dict(evaluation_metadata or {}),
        "plausibility": dict(plausibility or {}),
        "legacy_portfolio_health": legacy,
        "meta": {
            "as_of_market_date": as_of_market_date,
            "as_of_run_date": as_of_run_date,
            "pricing_as_of_utc": pricing_as_of_utc,
            "returns_eval": dict(returns_eval_meta or {}),
            "evaluation_metadata": dict(evaluation_metadata or {}),
            "producer": "run_daily_report.py",
        },
    }

    return out


def _compute_daily_health_score(*, metrics, score_cfg: ScoreConfig, goals: list[float], main_goal: float) -> dict[str, Any]:
    """Daily-report compatibility wrapper around the canonical health scorer.

    Daily report does not have executable-allocation drift diagnostics at this
    stage, so execution-quality components are set to neutral/pass values.
    Portfolio-search executable validation still passes the actual execution
    quality values into the same canonical scorer.
    """
    return canonical_compute_portfolio_health_score(
        final_metrics=metrics,
        execution_quality={
            "deployment_ratio": 1.0,
            "cash_weight": 0.0,
            "weight_drift_l1": 0.0,
            "dropped_theoretical_weight": 0.0,
        },
        score_cfg=score_cfg,
        goals=tuple(float(g) for g in goals),
        main_goal=float(main_goal),
        max_cash_weight=0.20,
        min_deployment_ratio=1.0,
        max_executable_mdd=float(getattr(score_cfg, "mdd_cap", 0.40) or 0.40),
        max_executable_cdar_95=float(getattr(score_cfg, "cdar_95_cap", 0.60) or 0.60),
        max_stability_energy=float(getattr(score_cfg, "stability_energy_cap", 2.00) or 2.00),
        max_dropped_weight=0.0 + 1e-12,
        max_weight_drift_l1=0.0 + 1e-12,
        metadata={
            "producer": "run_daily_report.py",
            "consumer": "daily_report",
            "execution_quality_mode": "neutral_current_holdings_report",
        },
    )


def s3_load_ledger_positions_dt(s3, *, bucket: str, root_prefix: str, as_of: str) -> dict | None:
    key = f"{str(root_prefix).strip('/')}/ledger/dt={as_of}/positions.json"
    return s3_get_json(s3, bucket=bucket, key=key)


def s3_load_ledger_pnl_dt(s3, *, bucket: str, root_prefix: str, as_of: str) -> dict | None:
    key = f"{str(root_prefix).strip('/')}/ledger/dt={as_of}/pnl.json"
    return s3_get_json(s3, bucket=bucket, key=key)


UNIVERSE_CSV_LOCAL = paths.universe_dir() / "universe.csv"

def _load_universe_ticker_to_asset_id() -> dict[str, str]:
    """
    Temporary ticker -> asset_id map for the still ticker-based daily-report path.

    Long-term rule: asset_id is the primary key. Duplicate tickers are allowed in
    the universe, so this helper only returns unambiguous ticker mappings. If a
    ledger position uses an ambiguous ticker, _load_closes_usd_from_ohlcv() will
    fail for that requested ticker instead of silently choosing the wrong asset.
    """
    df = pd.read_csv(UNIVERSE_CSV_LOCAL)
    if df is None or df.empty:
        raise RuntimeError(f"Universe is empty: {UNIVERSE_CSV_LOCAL}")

    for c in ["ticker", "asset_id"]:
        if c not in df.columns:
            raise RuntimeError(f"Universe missing required column '{c}': {UNIVERSE_CSV_LOCAL}")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()

    if "yahoo_ticker" not in df.columns:
        df["yahoo_ticker"] = df["ticker"]
    df["yahoo_ticker"] = df["yahoo_ticker"].astype(str).str.strip()

    if "yahoo_ticker_norm" not in df.columns:
        df["yahoo_ticker_norm"] = df["yahoo_ticker"]
    df["yahoo_ticker_norm"] = df["yahoo_ticker_norm"].astype(str).str.strip()

    if "name" not in df.columns:
        df["name"] = df["ticker"]
    df["name"] = df["name"].astype(str).str.strip()

    if "include" in df.columns:
        df["include"] = pd.to_numeric(df["include"], errors="coerce").fillna(1).astype(int)
    else:
        df["include"] = 1

    df = df[
        (df["include"] == 1)
        & (df["ticker"] != "")
        & (df["asset_id"] != "")
    ].copy()

    if df.empty:
        raise RuntimeError("Universe has no active (ticker, asset_id) pairs after normalization.")

    dup_asset = df[df["asset_id"].duplicated(keep=False)].sort_values("asset_id")
    if not dup_asset.empty:
        cols = [c for c in ["asset_id", "ticker", "yahoo_ticker_norm", "yahoo_ticker", "name", "include"] if c in dup_asset.columns]
        raise RuntimeError(
            "Duplicate active asset_id values found in universe. asset_id must be unique.\n"
            + dup_asset[cols].head(50).to_string(index=False)
        )

    counts = df.groupby("ticker")["asset_id"].nunique()
    ambiguous = sorted(counts[counts > 1].index.tolist())
    if ambiguous:
        print(
            "[universe][warn] duplicate active tickers exist and will not be used "
            f"as unique lookup keys. ambiguous_count={len(ambiguous)} sample={ambiguous[:10]}"
        )

    unique_df = df[df["ticker"].map(counts).eq(1)].copy()
    return dict(zip(unique_df["ticker"].tolist(), unique_df["asset_id"].tolist()))

def _diagnose_hmm_history(*, closes: pd.DataFrame, tickers: list[str], as_of_date: str) -> None:
    """
    Print per-ticker history stats and identify the limiting ticker for:
      closes window
      returns window after pct_change + dropna(any)
    """
    if closes is None or closes.empty:
        print("[diag][hmm] closes is empty")
        return

    c = closes.copy()
    c.index = pd.to_datetime(c.index, errors="coerce").tz_localize(None).normalize()
    c = c.loc[c.index <= pd.Timestamp(as_of_date).tz_localize(None).normalize()]

    cols = [t for t in tickers if t in c.columns]
    missing = [t for t in tickers if t not in c.columns]

    if missing:
        print(f"[diag][hmm] missing tickers in closes: {missing[:20]}{'...' if len(missing)>20 else ''}")

    if not cols:
        print("[diag][hmm] no tickers available in closes")
        return

    sub = c[cols]

    # First valid date per ticker (after ffill there may be fewer NaNs, but still check)
    first_valid = {t: sub[t].first_valid_index() for t in cols}
    last_valid = {t: sub[t].last_valid_index() for t in cols}
    n_valid = {t: int(sub[t].notna().sum()) for t in cols}
    n_nan = {t: int(sub[t].isna().sum()) for t in cols}

    # Limiting ticker = latest first_valid (it starts the latest)
    limiting_by_start = sorted(
        [(t, first_valid[t]) for t in cols],
        key=lambda x: (pd.Timestamp.max if x[1] is None else pd.Timestamp(x[1])),
        reverse=True,
    )[:5]

    limiting_by_count = sorted([(t, n_valid[t]) for t in cols], key=lambda x: x[1])[:5]

    print("[diag][hmm] closes per ticker (worst 5 by latest start):")
    for t, d in limiting_by_start:
        print(f"  - {t}: first_valid={d} last_valid={last_valid[t]} n_valid={n_valid[t]} n_nan={n_nan[t]}")

    print("[diag][hmm] closes per ticker (worst 5 by fewest valid obs):")
    for t, n in limiting_by_count:
        print(f"  - {t}: n_valid={n} first_valid={first_valid[t]} last_valid={last_valid[t]} n_nan={n_nan[t]}")

    # Now check returns window impact
    rets = sub.pct_change()

    # How many NaNs per column in returns?
    rets_nan = rets.isna().sum().sort_values(ascending=False)
    print("[diag][hmm] returns NaN counts (top 5):")
    for t, nn in rets_nan.head(5).items():
        print(f"  - {t}: nan_returns={int(nn)}")

    # Effective sample after dropna(any)
    rets_any = rets.dropna(how="any")
    print(f"[diag][hmm] returns window: rows_before={rets.shape[0]} rows_after_dropna_any={rets_any.shape[0]}")
    if not rets_any.empty:
        print(f"[diag][hmm] effective returns start={rets_any.index.min().date()} end={rets_any.index.max().date()}")
    

def _s3_list_keys(client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) or []:
            k = obj.get("Key", "")
            if k and k.lower().endswith(".parquet"):
                keys.append(k)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def _read_parquet_s3_bytes(client, bucket: str, key: str) -> pd.DataFrame:
    obj = client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return pd.read_parquet(io.BytesIO(body), engine="pyarrow")


def _load_closes_usd_from_ohlcv(
    *,
    tickers: list[str],
    start: str,
    end: str,
    s3_bucket: str = "alpha-edge-algo",
    s3_root_prefix: str = "market/ohlcv_usd/v1",
    s3_region: str = "eu-west-1",
) -> pd.DataFrame:
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        raise RuntimeError("No tickers provided to _load_closes_usd_from_ohlcv()")

    start_ts = pd.Timestamp(start).tz_localize(None).normalize()
    end_ts = pd.Timestamp(end).tz_localize(None).normalize()
    years = list(range(int(start_ts.year), int(end_ts.year) + 1))

    t2aid = _load_universe_ticker_to_asset_id()
    missing = [t for t in tickers if t not in t2aid]
    if missing:
        raise RuntimeError(
            "Some tickers in ledger are missing from universe mapping (ticker->asset_id): "
            + ", ".join(missing[:20])
        )

    ticker_asset = [(t, t2aid[t]) for t in tickers]
    asset_to_ticker = {aid: t for (t, aid) in ticker_asset}

    s3 = boto3.client("s3", region_name=s3_region)

    all_keys: list[tuple[str, str]] = []
    total_prefixes = len(ticker_asset) * len(years)
    seen_prefixes = 0

    print(f"[ohlcv] listing parquet keys assets={len(ticker_asset)} years={years[0]}..{years[-1]}")

    for (_, aid) in ticker_asset:
        for y in years:
            seen_prefixes += 1
            prefix = f"{s3_root_prefix}/asset_id={aid}/year={y}/"
            keys = _s3_list_keys(s3, s3_bucket, prefix)
            if keys:
                for k in keys:
                    all_keys.append((aid, k))

            if seen_prefixes % 20 == 0:
                print(f"[ohlcv] listed prefixes={seen_prefixes}/{total_prefixes} keys_so_far={len(all_keys)}")

    if not all_keys:
        raise RuntimeError(
            f"No parquet files found under s3://{s3_bucket}/{s3_root_prefix} "
            f"for tickers={tickers[:5]}... years={years}"
        )

    frames: list[pd.DataFrame] = []
    for (aid, key) in all_keys:
        df = _read_parquet_s3_bytes(s3, s3_bucket, key)
        if df is None or df.empty:
            continue

        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date")
        px_col = cols.get("adj_close_usd") or cols.get("close_usd") or cols.get("adj_close") or cols.get("close")
        if date_col is None or px_col is None:
            raise RuntimeError(
                f"Unexpected OHLCV parquet schema in s3://{s3_bucket}/{key}. Columns={list(df.columns)}"
            )

        out = df[[date_col, px_col]].copy()
        out.columns = ["date", "adj_close_usd"]
        out["asset_id"] = aid
        frames.append(out)

    if not frames:
        raise RuntimeError("Parquet keys were found but all read as empty frames.")

    long = pd.concat(frames, ignore_index=True)

    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.dropna(subset=["date"])
    long = long[(long["date"] >= start_ts) & (long["date"] <= end_ts)].copy()
    if long.empty:
        raise RuntimeError("OHLCV data exists but none in requested date window.")

    long["adj_close_usd"] = pd.to_numeric(long["adj_close_usd"], errors="coerce")
    long = long.dropna(subset=["adj_close_usd"])

    long["ticker"] = long["asset_id"].map(asset_to_ticker).fillna(long["asset_id"])

    long = long.sort_values(["date", "ticker"])
    if long.duplicated(subset=["date", "ticker"]).any():
        n_dup = int(long.duplicated(subset=["date", "ticker"], keep=False).sum())
        sample = long.loc[long.duplicated(subset=["date", "ticker"], keep=False), ["date", "ticker"]].head(10)
        print(f"[ohlcv][warn] found {n_dup} duplicate (date,ticker) rows; collapsing by last()")
        print(sample.to_string(index=False))
        long = long.groupby(["date", "ticker"], as_index=False)["adj_close_usd"].last()

    closes = (
        long.set_index(["date", "ticker"])["adj_close_usd"]
        .unstack("ticker")
        .sort_index()
        .ffill()
    )
    return closes


def _fetch_spot_prices_usd(
    *,
    tickers: list[str],
    provider_map: dict[str, str] | None = None,
    fallback_prices: pd.Series | None = None,
) -> pd.Series:
    provider_map = provider_map or {}
    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    if not tickers:
        return pd.Series(dtype="float64")

    internal_to_yahoo = {t: str(provider_map.get(t, t)).strip() for t in tickers}
    yahoo_list = [internal_to_yahoo[t] for t in tickers]

    df = yf.download(
        tickers=yahoo_list,
        period="1d",
        interval="1m",
        progress=True,
        threads=True,
        auto_adjust=True,
        timeout=30,
    )

    spot_by_yahoo: Dict[str, float] = {}

    try:
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                lvl0 = list(df.columns.get_level_values(0))
                lvl1 = list(df.columns.get_level_values(1))

                if any(y in lvl0 for y in yahoo_list):
                    for y in yahoo_list:
                        if y not in df.columns.get_level_values(0):
                            continue
                        sub = df[y]
                        if ("Close" in sub.columns) and (not sub["Close"].dropna().empty):
                            spot_by_yahoo[y] = float(sub["Close"].dropna().iloc[-1])
                        elif ("Adj Close" in sub.columns) and (not sub["Adj Close"].dropna().empty):
                            spot_by_yahoo[y] = float(sub["Adj Close"].dropna().iloc[-1])

                elif any(y in lvl1 for y in yahoo_list):
                    for y in yahoo_list:
                        if y not in df.columns.get_level_values(1):
                            continue
                        if ("Close" in df.columns.get_level_values(0)):
                            s = df["Close"][y]
                            if not s.dropna().empty:
                                spot_by_yahoo[y] = float(s.dropna().iloc[-1])
                                continue
                        if ("Adj Close" in df.columns.get_level_values(0)):
                            s = df["Adj Close"][y]
                            if not s.dropna().empty:
                                spot_by_yahoo[y] = float(s.dropna().iloc[-1])
            else:
                if "Close" in df.columns and not df["Close"].dropna().empty:
                    spot_by_yahoo[yahoo_list[0]] = float(df["Close"].dropna().iloc[-1])
                elif "Adj Close" in df.columns and not df["Adj Close"].dropna().empty:
                    spot_by_yahoo[yahoo_list[0]] = float(df["Adj Close"].dropna().iloc[-1])
    except Exception:
        spot_by_yahoo = {}

    out = {}
    for t in tickers:
        y = internal_to_yahoo[t]
        v = spot_by_yahoo.get(y, np.nan)
        if (not np.isfinite(v)) and (fallback_prices is not None) and (t in fallback_prices.index):
            v = float(fallback_prices.loc[t])
        out[t] = v

    return pd.Series(out, dtype="float64").replace([np.inf, -np.inf], np.nan)


# =========================
# Take profit by asset
# =========================

def _load_asset_tp_anchors(raw: dict | None) -> dict[str, dict[str, Any]]:
    """
    raw expected:
      {"as_of": "...", "anchors": {"TICK": {"anchor_price": 123.4, "anchor_date": "YYYY-MM-DD"}}}
    """
    if not isinstance(raw, dict):
        return {}
    anchors = raw.get("anchors")
    if not isinstance(anchors, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in anchors.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, dict):
            continue
        ap = v.get("anchor_price")
        ad = v.get("anchor_date")
        try:
            apf = float(ap)
        except Exception:
            apf = np.nan
        out[str(k).upper().strip()] = {
            "anchor_price": (apf if np.isfinite(apf) and apf > 0 else None),
            "anchor_date": (str(ad) if ad else None),
        }
    return out


def _directional_return_series(
    *,
    prices: pd.Series,
    anchor_price: float,
    side_sign: float,
) -> pd.Series:
    """
    Monotone "profit-directional" curve starting at 1:
      long : rel = price / anchor
      short: rel = anchor / price
    """
    p = pd.to_numeric(prices, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if p.empty or (not np.isfinite(anchor_price)) or anchor_price <= 0:
        return pd.Series(dtype="float64")

    if side_sign >= 0:
        rel = p / float(anchor_price)
    else:
        rel = float(anchor_price) / p
    rel = rel.replace([np.inf, -np.inf], np.nan).dropna()
    return rel


def _max_drawdown(rel: pd.Series) -> float | None:
    if rel is None or rel.empty:
        return None
    x = rel.astype(float)
    peak = x.cummax()
    dd = 1.0 - (x / peak)
    mdd = float(dd.max())
    return mdd if np.isfinite(mdd) else None


def build_take_profit_by_asset_plan(
    *,
    as_of: str,
    positions: dict[str, Position],
    closes: pd.DataFrame,
    exec_prices_usd: pd.Series,
    anchors_state: dict[str, dict[str, Any]],
    gross_target: float,
    min_trade_usd: float = 25.0,
    min_position_usd: float = 0.0,
    tp_return_thr: float = 0.15,
    dd_window_days: int = 63,
    dd_max: float = 0.08,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, float]]:
    """
    Option-2 behavior:
      - First try strict eligibility: r_anchor>=thr AND dd<=max
      - If nobody eligible BUT you need to reduce gross:
          fallback to profit-first trims:
            - choose tickers with r_anchor>0 (profitable vs anchor)
            - allocate reduction budget proportional to gross within that set
          if still empty:
            - proportional trim across ALL positions

    Returns:
      plan_df
      next_anchors_state
      positions_qty_target
    """
    tickers = sorted([t for t in positions.keys() if t in exec_prices_usd.index])
    if not tickers:
        empty = pd.DataFrame()
        return empty, anchors_state, {t: float(p.quantity) for t, p in positions.items()}

    px = pd.to_numeric(exec_prices_usd.reindex(tickers), errors="coerce").replace([np.inf, -np.inf], np.nan)
    qty = pd.Series({t: float(positions[t].quantity) for t in tickers}, dtype="float64")

    exp_signed = px * qty
    exp_gross = exp_signed.abs()
    gross_now = float(exp_gross.sum(skipna=True))

    need_reduce = float(max(0.0, gross_now - float(gross_target)))
    if need_reduce <= 0:
        plan = pd.DataFrame(
            {
                "ticker": tickers,
                "eligible": False,
                "reason": "gross_already<=target",
                "qty_current": qty.values,
                "qty_target": qty.values,
                "delta_qty": np.zeros(len(tickers)),
                "exp_gross_current": exp_gross.values,
                "exp_gross_reduce": np.zeros(len(tickers)),
                "exec_price_usd": px.values,
                "anchor_price_usd": [None] * len(tickers),
                "r_anchor": [None] * len(tickers),
                "dd_63": [None] * len(tickers),
                "meta_need_reduce": [0.0] * len(tickers),
            }
        )
        return plan, anchors_state, {t: float(positions[t].quantity) for t in positions.items()}

    rows: list[dict[str, Any]] = []
    eligible_strict: list[str] = []
    profitable_soft: list[str] = []

    for t in tickers:
        p_now = float(px.loc[t]) if np.isfinite(px.loc[t]) else np.nan
        q_now = float(qty.loc[t])
        side_sign = 1.0 if q_now >= 0 else -1.0

        entry_ok = positions[t].entry_price is not None and np.isfinite(float(positions[t].entry_price))
        if not entry_ok:
            rows.append(
                dict(
                    ticker=t,
                    eligible=False,
                    reason="missing_entry_price",
                    qty_current=q_now,
                    exec_price_usd=p_now,
                    anchor_price_usd=None,
                    r_anchor=None,
                    dd_63=None,
                    exp_gross_current=float(abs(p_now * q_now)) if np.isfinite(p_now) else np.nan,
                )
            )
            continue

        st = anchors_state.get(t, {})
        anchor_price = st.get("anchor_price")
        if anchor_price is None or (not np.isfinite(float(anchor_price))) or float(anchor_price) <= 0:
            anchor_price = float(positions[t].entry_price)

        if (not np.isfinite(p_now)) or p_now <= 0:
            rows.append(
                dict(
                    ticker=t,
                    eligible=False,
                    reason="missing_exec_price",
                    qty_current=q_now,
                    exec_price_usd=p_now,
                    anchor_price_usd=float(anchor_price),
                    r_anchor=None,
                    dd_63=None,
                    exp_gross_current=float(abs(p_now * q_now)) if np.isfinite(p_now) else np.nan,
                )
            )
            continue

        if side_sign >= 0:
            r_anchor = (p_now / float(anchor_price)) - 1.0
        else:
            r_anchor = (float(anchor_price) / p_now) - 1.0

        dd_63 = None
        try:
            if t in closes.columns:
                hist = closes[t].dropna()
                hist_tail = hist.iloc[-int(dd_window_days):] if len(hist) > dd_window_days else hist
                rel = _directional_return_series(prices=hist_tail, anchor_price=float(anchor_price), side_sign=side_sign)
                dd_63 = _max_drawdown(rel)
        except Exception:
            dd_63 = None

        ok_ret_strict = np.isfinite(r_anchor) and float(r_anchor) >= float(tp_return_thr)
        ok_dd_strict = (dd_63 is not None) and np.isfinite(float(dd_63)) and float(dd_63) <= float(dd_max)

        # Strict eligible
        if ok_ret_strict and ok_dd_strict:
            eligible_strict.append(t)
            rows.append(
                dict(
                    ticker=t,
                    eligible=True,
                    reason="eligible_strict",
                    qty_current=q_now,
                    exec_price_usd=p_now,
                    anchor_price_usd=float(anchor_price),
                    r_anchor=float(r_anchor),
                    dd_63=float(dd_63),
                    exp_gross_current=float(abs(p_now * q_now)),
                )
            )
            continue

        # Soft "profit-first" set (Option-2 fallback): any positive r_anchor
        is_prof = np.isfinite(r_anchor) and float(r_anchor) > 0.0
        if is_prof:
            profitable_soft.append(t)

        reason = []
        if not ok_ret_strict:
            reason.append(f"r_anchor<{tp_return_thr:.2f}")
        if not ok_dd_strict:
            reason.append(f"dd_63>{dd_max:.2f}" if dd_63 is not None else "dd_63_missing")

        rows.append(
            dict(
                ticker=t,
                eligible=False,
                reason=";".join(reason) if reason else "not_eligible",
                qty_current=q_now,
                exec_price_usd=p_now,
                anchor_price_usd=float(anchor_price),
                r_anchor=(None if not np.isfinite(r_anchor) else float(r_anchor)),
                dd_63=dd_63,
                exp_gross_current=float(abs(p_now * q_now)),
            )
        )

    # Choose reduction universe:
    #   1) strict eligible
    #   2) profitable soft (r_anchor>0)
    #   3) all tickers (last resort)
    if eligible_strict:
        reduce_set = eligible_strict
        reduce_mode = "strict"
    elif profitable_soft:
        reduce_set = profitable_soft
        reduce_mode = "profit_fallback"
    else:
        reduce_set = list(tickers)
        reduce_mode = "proportional_fallback"

    exp_set = exp_gross.reindex(reduce_set).fillna(0.0)
    denom = float(exp_set.sum())
    if denom <= 0:
        plan_df = pd.DataFrame(rows)
        plan_df["qty_target"] = plan_df["qty_current"]
        plan_df["delta_qty"] = 0.0
        plan_df["exp_gross_reduce"] = 0.0
        plan_df["meta_need_reduce"] = need_reduce
        plan_df["meta_reduce_mode"] = reduce_mode
        return plan_df, anchors_state, {t: float(p.quantity) for t, p in positions.items()}

    reduce_by = (exp_set / denom) * float(need_reduce)

    qty_target = qty.copy()
    reduce_used = 0.0
    traded: set[str] = set()

    for t in reduce_set:
        p = float(px.loc[t])
        q = float(qty.loc[t])
        if not np.isfinite(p) or p <= 0:
            continue

        exp_abs = float(abs(p * q))
        budget = float(min(float(reduce_by.loc[t]), exp_abs))

        if budget < float(min_trade_usd):
            continue

        dq = -float(np.sign(q) if q != 0 else 1.0) * (budget / p)
        q_new = q + dq

        if float(min_position_usd) > 0:
            rem_abs = float(abs(p * q_new))
            if rem_abs < float(min_position_usd):
                q_new = 0.0

        if q > 0 and q_new < 0:
            q_new = 0.0
        if q < 0 and q_new > 0:
            q_new = 0.0

        realized_budget = float(abs(p * (q - q_new)))
        if realized_budget < float(min_trade_usd):
            continue

        qty_target.loc[t] = float(q_new)
        reduce_used += realized_budget
        traded.add(t)

    plan_df = pd.DataFrame(rows).set_index("ticker", drop=False)
    plan_df["qty_target"] = plan_df["ticker"].map(lambda t: float(qty_target.get(t, np.nan)))
    plan_df["delta_qty"] = plan_df["qty_target"] - plan_df["qty_current"]

    plan_df["exp_gross_reduce"] = plan_df.apply(
        lambda r: (
            float(abs(float(r["exec_price_usd"]) * float(r["qty_current"])))
            - float(abs(float(r["exec_price_usd"]) * float(r["qty_target"])))
        )
        if np.isfinite(r.get("exec_price_usd", np.nan)) else np.nan,
        axis=1,
    )

    plan_df["meta_gross_now"] = gross_now
    plan_df["meta_gross_target"] = float(gross_target)
    plan_df["meta_need_reduce"] = float(need_reduce)
    plan_df["meta_reduce_used"] = float(reduce_used)
    plan_df["meta_reduce_mode"] = reduce_mode
    plan_df["as_of"] = as_of

    next_anchors = {k: dict(v) for k, v in anchors_state.items()}
    for t in traded:
        p_exec = float(px.loc[t])
        if not np.isfinite(p_exec) or p_exec <= 0:
            continue
        next_anchors[t] = {"anchor_price": float(p_exec), "anchor_date": str(as_of)}

    qty_target_dict = {t: float(qty_target.get(t, float(p.quantity))) for t, p in positions.items()}
    return plan_df.reset_index(drop=True), next_anchors, qty_target_dict


def run_daily_cycle_asof(
    *,
    as_of: str,
    backtest_run_id: str | None = None,
    write_outputs: bool = True,
    update_latest: bool = True,
    equity_override: float | None = None,
    goals_override: list[float] | None = None,
    main_goal_override: float | None = None,
    env: str | None = None,
    confirm_prod_write: bool = False,
    actuarial_max_allowed_leverage: float = 2.0,
    actuarial_n_paths: int = 20_000,
    refresh_returns_cache: bool = False,
) -> dict:
    cfg = load_runtime_config(env)

    if write_outputs:
        require_prod_confirmation(cfg, bool(confirm_prod_write))

    bucket = cfg.bucket
    region = cfg.region
    engine_root = cfg.engine_root
    market_root = cfg.market_root

    root_prefix = _resolve_root_prefix(
        engine_root=engine_root,
        backtest_run_id=backtest_run_id,
    )
    mode = "backtest" if backtest_run_id else "live"

    as_of_ts = pd.Timestamp(as_of).tz_localize(None).normalize()
    as_of_date = as_of_ts.strftime("%Y-%m-%d")

    # The report's logical run/as-of date must follow the explicit --as-of input.
    # Current wall-clock time is still stored separately in run_id/pricing metadata.
    # This prevents a no-write historical validation such as --as-of 2026-08-10
    # from silently appending a 2026-08-11 live-return row.
    run_dt = as_of_ts
    as_of_run_date = run_dt.strftime("%Y-%m-%d")
    requested_as_of_date = as_of_date

    # --- RETURNS_WIDE (asset_id-keyed canonical return source) ---
    returns_wide = _load_returns_wide_cache(
        bucket=bucket,
        market_root=market_root,
        as_of_ts=as_of_ts,
        refresh=bool(refresh_returns_cache),
    )

    if mode == "backtest" and equity_override is None:
        raise ValueError("backtest requires equity_override (do not rely on hardcoded equity).")

    s3 = s3_init(region)
    market = MarketStore(bucket=bucket, region=region)

    BENCH_PROXY = ["VT", "SPY", "QQQ", "IWM", "TLT", "VCIT", "GLD"]
    BENCH_NAME = "EQW(VT,SPY,QQQ,IWM,TLT,VCIT,GLD)"
    START_HISTORY = "2015-01-01"

    RESCALE_STATE_TABLE = "rescale/state"
    RESCALE_PLAN_TABLE = "rescale/plan"

    GOALS = goals_override if goals_override is not None else [7500.0, 10000.0, 12500.0]
    MAIN_GOAL = float(main_goal_override if main_goal_override is not None else 10000.0)
    # Live equity is resolved after ledger positions and current valuation prices are loaded.
    equity = float(equity_override) if equity_override is not None else None

    # ---------- Load MARKET regime (GLOBAL path) ----------
    market_hmm_payload = s3_load_latest_json(
        s3, bucket=bucket, root_prefix=engine_root, table="regimes/market_hmm"
    ) or {}

    market_as_of = market_hmm_payload.get("as_of")
    market_as_of = str(market_as_of) if market_as_of else as_of_date

    market_lev = None
    if isinstance(market_hmm_payload, dict):
        lr = market_hmm_payload.get("leverage_recommendation") or {}
        if isinstance(lr, dict) and lr.get("leverage") is not None:
            market_lev = float(lr.get("leverage"))
    if market_lev is None:
        market_lev = 1.0

    print(f"[market regime] as_of={market_as_of} target_leverage={market_lev:.2f}x")

    # ---------- Inputs ----------
    raw_ledger_positions = s3_load_ledger_positions_dt(
        s3, bucket=bucket, root_prefix=root_prefix, as_of=as_of_date
    )
    raw_pnl = s3_load_ledger_pnl_dt(
        s3, bucket=bucket, root_prefix=root_prefix, as_of=as_of_date
    ) or {}

    pnl_summary = raw_pnl.get("summary", {}) if isinstance(raw_pnl, dict) else {}
    equity_from_ledger = pnl_summary.get("equity_usd")
    if not raw_ledger_positions:
        raise RuntimeError(f"Missing S3 latest ledger positions under {root_prefix}/ledger/positions/latest.json")

    spot_rows, deriv_rows = parse_ledger_positions_obj(raw_ledger_positions)
    if not spot_rows and not deriv_rows:
        raise RuntimeError("Ledger positions payload has no spot_positions and no derivatives_positions.")

    raw_score_cfg = s3_load_latest_json(
        s3, bucket=bucket, root_prefix=root_prefix, table="configs/score_config"
    )
    if not raw_score_cfg:
        raise RuntimeError("Missing S3 latest score_config.")
    score_cfg = ScoreConfig(**raw_score_cfg)

    raw_baseline = s3_load_latest_json(s3, bucket=bucket, root_prefix=root_prefix, table="health")
    baseline = _parse_portfolio_health_compat(raw_baseline) if raw_baseline else None

    last_score = s3_load_latest_report_score(s3, bucket=bucket, root_prefix=root_prefix)
    if last_score is not None:
        print(f"[last run] previous daily report score: {last_score:.4f}")
    else:
        print("[last run] previous daily report score: N/A")

    tickers_spot = [str(r.get("ticker")).upper().strip() for r in spot_rows if r.get("ticker")]
    tickers_deriv = [str(r.get("ticker")).upper().strip() for r in deriv_rows if r.get("ticker")]
    tickers_all = sorted(set(tickers_spot + tickers_deriv))
    if not tickers_all:
        raise RuntimeError("No tickers in ledger positions.")

    # ---------- Load closes USD (as_of) ----------
    end_date = as_of_date
    closes_all = _load_closes_usd_from_ohlcv(
        tickers=tickers_all,
        start=START_HISTORY,
        end=end_date,
        s3_bucket=bucket,
        s3_root_prefix=f"{market_root.strip('/')}/ohlcv_usd/v1",
        s3_region=region,
        )
    latest_close_prices = closes_all.iloc[-1]

    pricing_as_of_utc = (
        f"{as_of_date}T23:59:59Z"
        if mode == "backtest"
        else pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    provider_state = market.read_provider_symbol_state() or {}
    if mode == "backtest":
        spot_prices = latest_close_prices.copy()
    else:
        spot_prices = _fetch_spot_prices_usd(
            tickers=tickers_all,
            provider_map=provider_state,
            fallback_prices=latest_close_prices,
        )

    prices_for_valuation = pd.to_numeric(spot_prices, errors="coerce").replace([np.inf, -np.inf], np.nan)
    prices_for_valuation = prices_for_valuation.reindex(latest_close_prices.index).combine_first(latest_close_prices)

    exec_prices = latest_close_prices.copy() if mode == "backtest" else prices_for_valuation.copy()

    # ---------- Resolve equity and canonical evaluation returns ----------
    if mode == "live":
        if equity_override is not None:
            print(f"[equity][warn] using explicit equity_override={float(equity_override):.2f}")
            equity = float(equity_override)
        else:
            equity = _compute_live_equity_from_ledger_and_prices(
                pnl_summary=pnl_summary,
                spot_rows=spot_rows,
                prices_for_valuation=prices_for_valuation,
            )
    else:
        if equity_override is None:
            raise ValueError("backtest requires equity_override.")
        equity = float(equity_override)

    returns_for_eval, returns_eval_meta = _build_live_augmented_returns_for_portfolio(
        returns_wide=returns_wide,
        spot_rows=spot_rows,
        deriv_rows=deriv_rows,
        latest_close_prices=latest_close_prices,
        prices_for_valuation=prices_for_valuation,
        as_of_run_date=as_of_run_date,
    )
    returns_eval_meta = {
        **dict(returns_eval_meta or {}),
        "requested_as_of_date": requested_as_of_date,
        "logical_run_date": as_of_run_date,
        "latest_close_date": str(pd.Timestamp(closes_all.index[-1]).date()),
        "pricing_as_of_utc": pricing_as_of_utc,
    }

    # ---------- Build Position objects ----------
    positions: dict[str, Position] = {}

    for r in spot_rows:
        t = str(r.get("ticker") or "").upper().strip()
        if not t:
            continue
        qty = float(r.get("quantity") or 0.0)
        if abs(qty) <= 0.0:
            continue
        entry = r.get("avg_cost", None)
        entry_price = None if entry is None else float(entry)
        positions[t] = Position(ticker=t, quantity=float(qty), entry_price=entry_price, currency="USD")

    for r in deriv_rows:
        t = str(r.get("ticker") or "").upper().strip()
        if not t:
            continue
        side = str(r.get("side") or "LONG").upper().strip()
        notional = float(r.get("open_notional_usd") or 0.0)
        if notional <= 0:
            continue
        last = prices_for_valuation.get(t, np.nan)
        if not np.isfinite(last) or float(last) <= 0:
            print(f"[positions][warn] missing price for derivative {t}; cannot convert notional->qty. Skipping.")
            continue
        sign = 1.0 if side == "LONG" else -1.0
        qty = sign * (notional / float(last))
        entry = r.get("avg_entry_price", None)
        entry_price = None if entry is None else float(entry)
        positions[t] = Position(ticker=t, quantity=float(qty), entry_price=entry_price, currency="USD")

    tickers = sorted([t for t in positions.keys() if t in closes_all.columns])
    if not tickers:
        raise RuntimeError("No usable tickers after building positions.")

    closes = closes_all[tickers].copy()

    asset_to_ticker, ticker_to_asset = _asset_id_ticker_maps_from_ledger_rows(
        spot_rows=spot_rows,
        deriv_rows=deriv_rows,
    )
    missing_asset_ids = [t for t in tickers if t not in ticker_to_asset]
    if missing_asset_ids:
        raise RuntimeError(
            "Daily report is asset_id-first, but these live tickers have no asset_id in ledger positions: "
            + ", ".join(missing_asset_ids[:20])
        )

    asset_ids = [ticker_to_asset[t] for t in tickers]
    missing_return_assets = [aid for aid in asset_ids if aid not in returns_for_eval.columns]
    if missing_return_assets:
        raise RuntimeError(
            "Daily report asset_id-first evaluation missing return columns for asset_id(s): "
            + ", ".join(missing_return_assets[:20])
        )

    returns_for_eval = returns_for_eval[asset_ids].dropna(how="any")
    rets_assets = returns_for_eval.copy()

    _diagnose_hmm_history(closes=closes, tickers=tickers, as_of_date=as_of_date)

    values = np.array([float(prices_for_valuation[t]) * float(positions[t].quantity) for t in tickers], dtype=np.float64)
    gross = float(np.sum(np.abs(values)))
    if not np.isfinite(gross) or gross <= 0:
        raise ValueError("Gross exposure == 0 (or non-finite) from positions/prices")
    w_vec = values / gross

    port_rets = (rets_assets[asset_ids] * w_vec).sum(axis=1).dropna()
    as_of_market_dt = pd.Timestamp(port_rets.index[-1]).normalize()
    as_of_market_date = as_of_market_dt.strftime("%Y-%m-%d")
    returns_eval_meta = {
        **dict(returns_eval_meta or {}),
        "returns_eval_end_date": as_of_market_date,
        "valuation_market_date": as_of_market_date,
    }
    print(
        f"[dates] requested_as_of_date={requested_as_of_date} | "
        f"as_of_market_date={as_of_market_date} | as_of_run_date={as_of_run_date}"
    )

    # ---------- Market regime ----------
    # Single source of truth: morning routine payload under regimes/market_hmm.
    hmm_payload_for_output = market_hmm_payload.get("hmm") if isinstance(market_hmm_payload, dict) else None
    regime_labels = None
    if hmm_payload_for_output:
        print("\n[market regime] source=regimes/market_hmm/latest.json (morning routine)")
        try:
            print_hmm_summary(hmm_payload_for_output, market_hmm_payload.get("leverage_recommendation"))
        except Exception as e:
            print(f"[market regime][warn] failed to print persisted HMM summary: {type(e).__name__}: {e}")
    else:
        print("\n[market regime][warn] missing persisted market HMM payload; daily report will not fit a local market HMM.")

    st_raw = market.read_regime_filter_state() or {}
    filter_state = RegimeFilterState(
        last_date=st_raw.get("last_date"),
        chosen_label=st_raw.get("chosen_label"),
        days_in_regime=int(st_raw.get("days_in_regime", 0) or 0),
        probs_smoothed=st_raw.get("probs_smoothed"),
    )

    # Daily report consumes the morning routine's persisted market leverage.
    # It must not recompute or mutate the market regime filter state.
    lev_rec = market_hmm_payload.get("leverage_recommendation") if isinstance(market_hmm_payload, dict) else None
    if not isinstance(lev_rec, dict):
        lev_rec = {"leverage": float(market_lev), "mode": "persisted_or_default", "label": None, "conf": 0.0}

    # ---------- Market RESCALE trigger (ONLY on regime / leverage change) ----------
    raw_mkt_state = s3_load_latest_json(
        s3, bucket=bucket, root_prefix=root_prefix, table=MARKET_RESCALE_STATE_TABLE
    ) or {}
    prev_label = raw_mkt_state.get("label")
    prev_lev = raw_mkt_state.get("leverage")

    fs = (lev_rec or {}).get("filter_state") or {}
    cur_label = str(
        fs.get("chosen_label")
        or (lev_rec or {}).get("label")
        or (hmm_payload_for_output or {}).get("label_commit")
        or "UNKNOWN"
    )
    cur_lev = float(market_lev)

    market_regime_changed = (prev_label is not None and cur_label != prev_label)
    market_lev_changed = (
        prev_lev is not None
        and abs(cur_lev - float(prev_lev)) / max(1e-9, abs(float(prev_lev))) >= 0.10
    )
    should_rescale_market = bool(market_regime_changed or market_lev_changed)

    print(
        f"[market rescale] should_rescale={should_rescale_market} "
        f"prev_label={prev_label} cur_label={cur_label} prev_lev={prev_lev} cur_lev={cur_lev:.2f}"
    )

    # ---------- Portfolio behavior regime ----------
    # This is intentionally separate from the canonical market regime.
    # It is a local diagnostic of how the current portfolio return path is behaving.
    portfolio_behavior_regime = build_portfolio_behavior_regime(
        portfolio_returns=port_rets,
        market_regime_payload=market_hmm_payload if isinstance(market_hmm_payload, dict) else {},
        min_observations=252,
        commit_threshold=0.65,
    )
    print(
        "[portfolio behavior regime] "
        f"label={portfolio_behavior_regime.get('label')} "
        f"confidence={portfolio_behavior_regime.get('confidence')} "
        f"alignment={(portfolio_behavior_regime.get('regime_alignment') or {}).get('status')}"
    )

    # ---------- Report ----------
    report = build_portfolio_report(
        closes=closes,
        positions=positions,
        equity=equity,
        goals=GOALS,
        main_goal=MAIN_GOAL,
        score_config=score_cfg,
        prices_usd=prices_for_valuation,
        asset_returns=returns_for_eval,
        asset_id_by_ticker=ticker_to_asset,
    )
    health_score_payload = _compute_daily_health_score(
        metrics=report.eval,
        score_cfg=score_cfg,
        goals=list(GOALS),
        main_goal=float(MAIN_GOAL),
    )
    evaluation_metadata = build_evaluation_metadata(
        returns_eval_meta=returns_eval_meta,
        price_source="spot_prices_usd_with_latest_close_fallback" if mode == "live" else "latest_close_prices",
        market_regime_source=f"{engine_root.strip('/')}/regimes/market_hmm/latest.json",
        score_config_version="score_config_latest",
        run_id=f"daily_report_{as_of_run_date}_{as_of_market_date}",
        as_of=as_of_market_date,
    )
    evaluation_metadata = {
        **dict(evaluation_metadata or {}),
        "requested_as_of_date": requested_as_of_date,
        "as_of_run_date": as_of_run_date,
        "as_of_market_date": as_of_market_date,
        "valuation_market_date": as_of_market_date,
        "latest_close_date": returns_eval_meta.get("latest_close_date"),
        "returns_eval_end_date": returns_eval_meta.get("returns_eval_end_date"),
        "live_return_date": returns_eval_meta.get("live_return_date"),
        "pricing_as_of_utc": pricing_as_of_utc,
    }
    health_score_payload["metadata"] = {**dict(health_score_payload.get("metadata") or {}), **evaluation_metadata}
    metric_plausibility = build_plausibility_guards(
        metrics=report.eval,
        returns_rows=int(len(returns_for_eval)),
        returns_assets=int(len(returns_for_eval.columns)),
        health_score_payload=health_score_payload,
        evaluation_metadata=evaluation_metadata,
        asset_ids=asset_ids,
    )
    health_score_payload["plausibility"] = metric_plausibility
    if not metric_plausibility.get("ok", False):
        print(f"[metrics][warn] plausibility flags: {metric_plausibility.get('flags')}")
    print(summarize_report(report))
    print(f"Health Score: {health_score_payload['health_score']:.1f} / 100 ({health_score_payload['health_grade']})")
    print(f"Raw optimizer score: {health_score_payload['raw_optimizer_score']:.4f}")

    # ---------- Actuarial diagnostics ----------
    actuarial_diagnostics = None
    actuarial_text = None

    try:
        _actuarial_report, actuarial_text, actuarial_diagnostics = (
            build_actuarial_diagnostic_from_portfolio_report(
                report=report,
                closes=closes,
                goals=GOALS,
                main_goal=float(MAIN_GOAL),
                score_config=score_cfg,
                portfolio_id="current_portfolio",
                run_id=f"daily_report_{as_of_run_date}_{as_of_market_date}",
                source="daily_report",
                terminal_title="ACTUARIAL RISK DIAGNOSTICS - DAILY REPORT",
                current_leverage=float(report.snapshot.leverage),
                max_allowed_leverage=float(actuarial_max_allowed_leverage),
                days=252,
                n_paths=int(actuarial_n_paths),
                mc_seed=97531,
                path_source="bootstrap",
                pca_k=5,
                block_size=(8, 12),
                asset_returns=returns_for_eval,
                metadata={
                    "mode": mode,
                    "requested_as_of_date": requested_as_of_date,
                    "as_of_market_date": as_of_market_date,
                    "as_of_run_date": as_of_run_date,
                    "returns_eval_end_date": returns_eval_meta.get("returns_eval_end_date"),
                    "asset_identity_mode": "asset_id_first",
                    "root_prefix": root_prefix,
                    "tolerance_policy": evaluation_metadata.get("tolerance_policy"),
                },
            )
        )
        maybe_print_actuarial_diagnostic_section(actuarial_text, enabled=True)
    except Exception as e:
        actuarial_diagnostics = {
            "status": "failed",
            "source": "daily_report",
            "error_type": type(e).__name__,
            "error": str(e),
        }
        print(f"[actuarial][daily][warn] failed to build diagnostics: {type(e).__name__}: {e}")

    # ---------- Benchmark ----------
    bench_rets = None
    bench_ann_ret = None
    bench_meta = {"name": BENCH_NAME, "tickers": BENCH_PROXY, "method": "equal_weight_daily_rebalanced"}
    cols: list[str] = []
    try:
        bench_closes_df = _load_closes_usd_from_ohlcv(
            tickers=BENCH_PROXY,
            start=START_HISTORY,
            end=end_date,
            s3_bucket=bucket,
            s3_root_prefix=f"{market_root.strip('/')}/ohlcv_usd/v1",
            s3_region=region,
        )

        cols = [c for c in BENCH_PROXY if c in bench_closes_df.columns]
        bench_meta["n_assets_used"] = int(len(cols))

        if len(cols) >= 2:
            x = bench_closes_df[cols].copy()
            r_b = x.pct_change().dropna(how="any")
            bench_rets = r_b.mean(axis=1).dropna()
            bench_ann_ret = float(bench_rets.mean() * 252.0)

            bench_meta["first_date_used"] = str(pd.Timestamp(bench_rets.index.min()).date())
            bench_meta["last_date_used"] = str(pd.Timestamp(bench_rets.index.max()).date())
            print(f"[bench] used={cols} ann={bench_ann_ret:.6f} rets_none={bench_rets is None}")
        else:
            bench_meta["error"] = "not_enough_assets"
            print(f"[bench][warn] not enough assets after filtering. cols={cols}")

    except Exception as e:
        bench_rets = None
        bench_ann_ret = None
        bench_meta = {**bench_meta, "error": f"failed_to_compute: {type(e).__name__}: {e}"}
        print(f"[bench][error] {type(e).__name__}: {e} | cols={cols}")

    if bench_rets is not None:
        print(f"[bench] rets_len={len(bench_rets)} first={bench_rets.index.min()} last={bench_rets.index.max()}")

    # ---------- Health snapshot & reopt ----------
    current_health = build_portfolio_health(
        report.eval,
        as_of=as_of_market_dt,
        benchmark_ann_return=bench_ann_ret,
        port_rets=port_rets,
        bench_rets=bench_rets,
        regime_labels=regime_labels,
    )

    health_latest_payload = _build_health_latest_payload(
        current_health=current_health,
        health_score_payload=health_score_payload,
        as_of_market_date=as_of_market_date,
        as_of_run_date=as_of_run_date,
        pricing_as_of_utc=pricing_as_of_utc,
        returns_eval_meta=returns_eval_meta,
        evaluation_metadata=evaluation_metadata,
        plausibility=metric_plausibility,
    )

    if getattr(current_health, "alpha_report_json", None):
        try:
            ar = json.loads(current_health.alpha_report_json)
            print(format_alpha_report(ar))
        except Exception:
            print("[alpha][warn] failed to parse alpha_report_json")

    reopt = False
    if baseline is None:
        print("\n[Portfolio health] No baseline set yet. Setting baseline to current health.")
        baseline = current_health
    else:
        reopt = should_reoptimize(baseline, current_health)

    # ---------- Take Profit (portfolio-level) ----------
    raw_tp_state = s3_load_latest_json(
        s3, bucket=bucket, root_prefix=root_prefix, table=TAKE_PROFIT_STATE_TABLE
    ) or {}

    tp_state = TakeProfitState(
        anchor_date=raw_tp_state.get("anchor_date"),
        anchor_equity=raw_tp_state.get("anchor_equity"),
        hwm_equity=raw_tp_state.get("hwm_equity"),
        harvest_mode=bool(raw_tp_state.get("harvest_mode", False)),
        last_harvest_date=raw_tp_state.get("last_harvest_date"),
        current_multiplier=float(raw_tp_state.get("current_multiplier", 1.0) or 1.0),
    )

    tp_cfg = TakeProfitConfig(
        enter_profit=0.10,
        exit_profit=0.07,
        max_dd=0.05,
        min_sharpe=0.75,
        max_harvest=0.25,
        k=8.0,
        m_min=0.60,
        cooldown_days=10,
        use_stability=False,
    )

    tp_res = take_profit_policy(
        cfg=tp_cfg,
        state=tp_state,
        as_of=as_of_market_date,
        equity=float(equity),
        sharpe_value=getattr(current_health, "sharpe", None),
        stability=None,
    )

    # Effective leverage target (market * TP multiplier)
    lev_target = float(market_lev) * float(tp_res.m_star)

    # Precedence rule:
    # - REOPT blocks REINVEST
    # - REINVEST only allowed when TP harvest is active AND not reopt
    do_reinvest = bool(tp_res.do_harvest) and (not bool(reopt))

    print(
        f"\n[take_profit] {'HARVEST' if tp_res.do_harvest else 'no_harvest'} "
        f"m={tp_res.m_star:.3f} "
        f"r_anchor={tp_res.r_anchor if tp_res.r_anchor is not None else 'n/a'} "
        f"dd={tp_res.dd if tp_res.dd is not None else 'n/a'} "
        f"sharpe={tp_res.sharpe if tp_res.sharpe is not None else 'n/a'}"
    )
    if tp_res.reasons:
        print("[take_profit] reasons:", ", ".join(tp_res.reasons))

    # ---------- Take Profit by asset (ONLY when tp_res.do_harvest=True) ----------
    asset_tp_plan_df = None
    next_asset_anchors = None
    positions_qty_for_rebalance = {t: float(p.quantity) for t, p in positions.items()}  # default

    if tp_res.do_harvest:
        raw_asset_state = s3_load_latest_json(
            s3, bucket=bucket, root_prefix=root_prefix, table=TAKE_PROFIT_ASSETS_STATE_TABLE
        )
        anchors_state = _load_asset_tp_anchors(raw_asset_state)

        gross_target_tp = float(equity) * float(lev_target)

        asset_tp_plan_df, next_asset_anchors, positions_qty_for_rebalance = build_take_profit_by_asset_plan(
            as_of=as_of_market_date,
            positions=positions,
            closes=closes,
            exec_prices_usd=exec_prices.reindex(sorted(exec_prices.index)).copy(),
            anchors_state=anchors_state,
            gross_target=gross_target_tp,
            min_trade_usd=25.0,
            min_position_usd=0.0,
            tp_return_thr=0.15,
            dd_window_days=63,
            dd_max=0.08,
        )

        if asset_tp_plan_df is not None and not asset_tp_plan_df.empty:
            used = float(asset_tp_plan_df["meta_reduce_used"].iloc[0]) if "meta_reduce_used" in asset_tp_plan_df.columns else 0.0
            need = float(asset_tp_plan_df["meta_need_reduce"].iloc[0]) if "meta_need_reduce" in asset_tp_plan_df.columns else 0.0
            mode_red = str(asset_tp_plan_df.get("meta_reduce_mode", pd.Series(["?"])).iloc[0]) if "meta_reduce_mode" in asset_tp_plan_df.columns else "?"
            n_traded = int((asset_tp_plan_df["exp_gross_reduce"].fillna(0.0) >= 25.0).sum()) if "exp_gross_reduce" in asset_tp_plan_df.columns else 0
            print(f"\n[take_profit_by_asset] need_reduce={need:,.2f} used={used:,.2f} traded_assets={n_traded} mode={mode_red}")

        if write_outputs and asset_tp_plan_df is not None and not asset_tp_plan_df.empty:
            s3_write_parquet_partition(
                s3,
                bucket=bucket,
                root_prefix=root_prefix,
                table=TAKE_PROFIT_ASSETS_PLAN_TABLE,
                dt=run_dt,
                filename="asset_tp_plan.parquet",
                df=asset_tp_plan_df,
            )

        if write_outputs and (next_asset_anchors is not None):
            s3_write_json_event(
                s3,
                bucket=bucket,
                root_prefix=root_prefix,
                table=TAKE_PROFIT_ASSETS_STATE_TABLE,
                dt=run_dt,
                filename="state.json",
                payload={
                    "as_of": as_of_market_date,
                    "anchors": next_asset_anchors,
                    "meta": {
                        "mode": mode,
                        "pricing_as_of_utc": pricing_as_of_utc,
                        "exec_price_source": ("close" if mode == "backtest" else "spot"),
                        "rule": {
                            "tp_return_thr": 0.15,
                            "dd_window_days": 63,
                            "dd_max": 0.08,
                            "min_trade_usd": 25.0,
                            "min_position_usd": 0.0,
                        },
                    },
                },
                update_latest=update_latest,
            )

    # ---------- RESCALE (market-regime only) then REINVEST (if allowed) ----------
    gross_target = float(equity) * float(lev_target)

    # Start from post-asset-TP quantities (THIS is current)
    qty_current = {str(t).upper().strip(): float(q) for t, q in (positions_qty_for_rebalance or {}).items()}

    # Build RESCALE PLAN (do NOT apply it yet)
    rescale_plan = None
    qty_target_rescale = None

    if should_rescale_market:
        rescale_plan = build_rescale_plan(
            as_of=as_of_market_date,
            equity=float(equity),
            recommended_leverage=float(lev_target),
            positions_qty=qty_current,              # IMPORTANT: current quantities
            prices_usd=prices_for_valuation,
            max_notional_cap=None,
        )

        df_t = rescale_plan.targets
        if ("ticker" in df_t.columns) and ("qty_target" in df_t.columns):
            qty_target_rescale = {
                str(r["ticker"]).upper().strip(): float(r["qty_target"])
                for _, r in df_t.iterrows()
            }
        else:
            print("[rescale][warn] plan.targets missing (ticker,qty_target); cannot build qty_target_rescale")


    # Base for reinvest = current holdings (post-TP), NOT rescale targets
    qty_for_reinvest = dict(qty_current)

    # Now REINVEST only if allowed (TP harvest + not reopt)
    if do_reinvest:
        qty_after_continuous, reinvest_meta = reinvest_leftover_with_frozen_core(
            as_of=as_of_market_date,
            returns_wide=returns_wide,
            exec_prices_usd=exec_prices,
            equity=float(equity),
            gross_target=float(gross_target),
            positions=positions,
            positions_qty_after_tp=qty_for_reinvest,   # <-- FIX: use current, not qty_base
            asset_tp_plan_df=asset_tp_plan_df,
            score_cfg=score_cfg,
            goals=list(GOALS),
            main_goal=float(MAIN_GOAL),
            max_assets_total=10,
            min_assets_sleeve=2,
            pop_size=60,
            generations=25,
            elite_frac=0.15,
            n_paths_init=4000,
            n_paths_final=20000,
            block_size=(8, 12),
            min_trade_usd=25.0,
            seed=123,
        )

        # ---------------------------------------------------------
        # Discretize ONLY the sleeve (keep frozen core frozen)
        # ---------------------------------------------------------
        sleeve_w = reinvest_meta.get("best_weights_sleeve")
        leftover = float(reinvest_meta.get("leftover", 0.0) or 0.0)

        # Robust core set:
        # Prefer reinvest_meta core_active if present, else infer from asset_tp_plan_df core logic is based on.
        core_set = set(str(t).upper().strip() for t in (reinvest_meta.get("core_active") or []))
        if not core_set:
            # Fallback: freeze tickers that exist in the starting qty map AND are NOT in sleeve weights
            # (This is conservative: it freezes everything except explicit sleeve tickers)
            if isinstance(sleeve_w, dict) and sleeve_w:
                core_set = set(qty_for_reinvest.keys()) - set(str(t).upper().strip() for t in sleeve_w.keys())

        if isinstance(sleeve_w, dict) and sleeve_w and np.isfinite(leftover) and leftover >= 25.0:
            px_dict = {
                str(t).upper().strip(): float(p)
                for t, p in exec_prices.items()
                if p is not None and np.isfinite(float(p)) and float(p) > 0
            }

            alloc = weights_to_discrete_shares(
                weights={str(t).upper().strip(): float(w) for t, w in sleeve_w.items()},
                prices=px_dict,
                notional=float(leftover),      # <-- critical: sleeve only
                min_weight=0.01,
                min_units_equity=1.0,
                min_units_crypto=0.0,
                min_units_weight_thr=0.03,
                crypto_decimals=8,
                nearest_step_remaining_frac=0.10,
            )

            sleeve_qty = {str(t).upper().strip(): float(q) for t, q in (alloc.shares or {}).items()}

            # Start from the reinvest base (frozen core quantities)
            qty_after = dict(qty_for_reinvest)

            # Merge sleeve target quantities as deltas (new sleeve buys)
            for t, dq in sleeve_qty.items():
                if t in core_set:
                    continue  # hard-freeze
                if not np.isfinite(dq) or abs(dq) <= 0.0:
                    continue
                qty_after[t] = float(qty_after.get(t, 0.0) + float(dq))

            reinvest_meta["discrete_allocation"] = {
                "mode": "sleeve_only_merge",
                "leftover_budget": float(leftover),
                "total_spent": float(alloc.total_spent),
                "cash_left": float(alloc.cash_left),
                "realized_weights": dict(alloc.realized_weights or {}),
                "sleeve_shares": sleeve_qty,
            }
        else:
            # If we can't discretize sleeve, keep the continuous qty output (already preserves frozen core)
            qty_after = {str(t).upper().strip(): float(q) for t, q in (qty_after_continuous or {}).items()}
            reinvest_meta["discrete_allocation"] = {
                "status": "skipped",
                "reason": (
                    "missing_best_weights_sleeve"
                    if not (isinstance(sleeve_w, dict) and sleeve_w)
                    else "leftover_too_small"
                ),
                "leftover_budget": float(leftover),
            }

    else:
        qty_after = dict(qty_for_reinvest)
        reinvest_meta = {
            "status": "skip_reinvest",
            "reason": ("reopt" if bool(reopt) else "tp_not_active"),
            "as_of": as_of_market_date,
            "gross_target": float(gross_target),
            "discrete_allocation": {
                "status": "skipped",
                "reason": ("reopt" if bool(reopt) else "tp_not_active"),
            },
        }

    positions_qty_for_rebalance = dict(qty_after)


    print(
        f"\n[reinvest] status={reinvest_meta.get('status')} "
        f"reason={reinvest_meta.get('reason', '')} "
        f"leftover={float(reinvest_meta.get('leftover', 0.0) or 0.0):.2f} "
        f"impr={float(reinvest_meta.get('improvement', 0.0) or 0.0):+.6f}"
    )

    if write_outputs:
        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="reinvest/runs",
            dt=run_dt,
            filename="reinvest.json",
            payload=reinvest_meta,
            update_latest=update_latest,
        )

    # ---------- Rebalance planning (market-rescale only) ----------
    # FIX: reb_state was missing in your latest version
    raw_reb_state = s3_load_latest_json(
        s3, bucket=bucket, root_prefix=root_prefix, table="rescale/state"
    )
    reb_state = RebalanceState(
        last_rebalance_date=(raw_reb_state or {}).get("last_rebalance_date"),
        last_rebalance_equity=(raw_reb_state or {}).get("last_rebalance_equity"),
    )

    gross_now = compute_gross_notional_from_positions(
        positions_qty=positions_qty_for_rebalance,
        prices_usd=prices_for_valuation,
    )

    # Diagnostics (optional logging)
    _diag = should_rebalance(
        as_of=as_of_market_date,
        equity=float(equity),
        gross_notional=float(gross_now),
        recommended_leverage=float(lev_target),
        state=reb_state,
        drift_threshold=0.15,
        min_days_between=3,
        time_rule_days=30,
        equity_band=None,
    )

    from alpha_edge.portfolio.rebalance_engine import RebalanceDecision
    L_real = float(gross_now) / float(equity) if float(equity) > 0 else float("inf")
    drift_ratio = (L_real / float(lev_target)) if float(lev_target) > 0 else float("inf")

    decision = RebalanceDecision(
        should_rebalance=bool(should_rescale_market),
        reasons=(
            (["market_regime_change"] if market_regime_changed else [])
            + (["market_leverage_change"] if market_lev_changed else [])
            + ([] if should_rescale_market else ["no_market_rescale"])
        ),
        leverage_real=float(L_real),
        leverage_target=float(lev_target),
        drift_ratio=float(drift_ratio),
    )

    # ---------- Rescale plan persistence ----------
    if decision.should_rebalance:
        plan = build_rescale_plan(
            as_of=as_of_market_date,
            equity=float(equity),
            recommended_leverage=float(lev_target),
            positions_qty=positions_qty_for_rebalance,
            prices_usd=prices_for_valuation,
            max_notional_cap=None,
        )

        print_decision_addendum(
            decision=decision,
            health=current_health,
            bench_ann_ret=bench_ann_ret,
            reopt=reopt,
            plan=plan,
            take_profit={
                "do_harvest": bool(tp_res.do_harvest),
                "m_star": float(tp_res.m_star),
                "r_anchor": tp_res.r_anchor,
                "dd": tp_res.dd,
                "sharpe": tp_res.sharpe,
                "reasons": tp_res.reasons,
                "cooldown_days": int(tp_cfg.cooldown_days),
            },
            execution_signals=None,
        )

        plan_df = plan.targets.copy()
        plan_df["as_of"] = plan.as_of
        plan_df["equity"] = plan.equity
        plan_df["recommended_leverage"] = plan.recommended_leverage
        plan_df["target_gross_notional"] = plan.target_gross_notional
        plan_df["used_gross_notional"] = plan.used_gross_notional
        plan_df["leftover_notional"] = plan.leftover_notional
        plan_df["gross_current"] = plan.gross_current
        plan_df["leverage_current"] = plan.leverage_current
        plan_df["decision_reasons"] = ", ".join(decision.reasons)

        tp_plan_df = None
        if tp_res.do_harvest:
            tp_plan_df = plan_df.copy()
            tp_plan_df["take_profit_m_star"] = float(tp_res.m_star)
            tp_plan_df["take_profit_r_anchor"] = tp_res.r_anchor
            tp_plan_df["take_profit_dd"] = tp_res.dd
            tp_plan_df["take_profit_sharpe"] = tp_res.sharpe

        if write_outputs:
            s3_write_parquet_partition(
                s3,
                bucket=bucket,
                root_prefix=root_prefix,
                table="rescale/plan",
                dt=run_dt,
                filename="plan.parquet",
                df=plan_df,
            )

            s3_write_json_event(
                s3,
                bucket=bucket,
                root_prefix=root_prefix,
                table="rescale/state",
                dt=run_dt,
                filename="state.json",
                payload={
                    "last_rebalance_date": as_of_market_date,
                    "last_rebalance_equity": float(equity),
                    "meta": {
                        "leverage_target": float(lev_target),
                        "leverage_real": float(decision.leverage_real),
                        "drift_ratio": float(decision.drift_ratio),
                        "reasons": decision.reasons,
                    },
                },
                update_latest=update_latest,
            )

            if tp_plan_df is not None:
                s3_write_parquet_partition(
                    s3,
                    bucket=bucket,
                    root_prefix=root_prefix,
                    table=TAKE_PROFIT_PLAN_TABLE,
                    dt=run_dt,
                    filename="plan.parquet",
                    df=tp_plan_df,
                )

    else:
        print_decision_addendum(
            decision=decision,
            health=current_health,
            bench_ann_ret=bench_ann_ret,
            reopt=reopt,
            plan=None,
            take_profit={
                "do_harvest": bool(tp_res.do_harvest),
                "m_star": float(tp_res.m_star),
                "r_anchor": tp_res.r_anchor,
                "dd": tp_res.dd,
                "sharpe": tp_res.sharpe,
                "reasons": tp_res.reasons,
                "cooldown_days": int(tp_cfg.cooldown_days),
            },
            execution_signals=None,
        )

    transition_assessment_payload = s3_load_latest_json(
        s3, bucket=bucket, root_prefix=root_prefix, table=TRANSITION_ASSESSMENT_TABLE
    ) or {}

    execution_signals = build_daily_report_execution_signals(
        rescale_decision=decision,
        reoptimization_pressure=bool(reopt),
        take_profit={
            "do_harvest": bool(tp_res.do_harvest),
            "m_star": float(tp_res.m_star),
            "r_anchor": tp_res.r_anchor,
            "dd": tp_res.dd,
            "sharpe": tp_res.sharpe,
            "reasons": tp_res.reasons,
        },
        transition_assessment=transition_assessment_payload,
        current_health=current_health,
    )

    print("\n[execution_signals]")
    print(f"  decision_authority: {execution_signals.get('decision_authority')}")
    print(f"  final_decision:     {(execution_signals.get('final_execution_decision') or {}).get('recommendation')}")
    for _name, _sig in (execution_signals.get("signals") or {}).items():
        print(f"  {_name}: triggered={bool(_sig.get('triggered'))} severity={_sig.get('severity')} reason={_sig.get('reason')}")

    # ---------- Persist outputs ----------
    if write_outputs:
        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="portfolio_behavior_regime",
            dt=run_dt,
            filename="regime.json",
            payload={
                "as_of": as_of_market_date,
                "tickers": list(tickers),
                "asset_ids": list(asset_ids),
                "portfolio_behavior_regime": portfolio_behavior_regime,
                "market_regime_source": "regimes/market_hmm/latest.json",
                "meta": {
                    "as_of_market_date": as_of_market_date,
                    "as_of_run_date": as_of_run_date,
                    "pricing_as_of_utc": pricing_as_of_utc,
                    "note": "Portfolio behavior regime is a diagnostic; market regime remains the source of truth.",
                },
            },
            update_latest=update_latest,
        )

        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="daily_reports",
            dt=run_dt,
            filename="report.json",
            payload={
                "as_of": as_of_market_date,
                "meta": {
                    "as_of_market_date": as_of_market_date,
                    "as_of_run_date": as_of_run_date,
                    "pricing_as_of_utc": pricing_as_of_utc,
                },
                "report": asdict(report),
                "actuarial_diagnostics": actuarial_diagnostics,
                "inputs": {
                    "equity": equity,
                    "evaluation_metadata": evaluation_metadata,
                    "metric_plausibility": metric_plausibility,
                    "goals": GOALS,
                    "main_goal": MAIN_GOAL,
                    "returns_eval": returns_eval_meta,
                    "health_score": health_score_payload,
                    "benchmark": {
                        "name": bench_meta.get("name"),
                        "tickers": bench_meta.get("tickers"),
                        "method": bench_meta.get("method"),
                        "ann_return": bench_ann_ret,
                        "meta": bench_meta,
                    },
                    "tickers": tickers,
                    "asset_ids": asset_ids,
                    "asset_id_by_ticker": ticker_to_asset,
                    "asset_id_to_ticker": asset_to_ticker,
                    "asset_identity_mode": "asset_id_first",
                    "start_history": START_HISTORY,
                    "spot_prices_usd": {
                        k: (None if not np.isfinite(v) else float(v))
                        for k, v in prices_for_valuation.items()
                    },
                    "market_regime": {
                        "target_leverage": float(market_lev),
                        "source_table": "regimes/market_hmm",
                        "label": cur_label,
                    },
                    "portfolio_behavior_regime": portfolio_behavior_regime,
                    "execution_signals": execution_signals,
                    "transition_assessment_ref": execution_signals.get("transition_assessment_ref"),
                },
                "flags": {
                    "should_reoptimize": bool(reopt),
                    "baseline_exists": bool(baseline is not None),
                    "daily_report_execution_authority": execution_signals.get("decision_authority"),
                },
            },
            update_latest=update_latest,
        )

        holdings_df = pd.DataFrame(report.snapshot.positions_table)
        s3_write_parquet_partition(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="holdings",
            dt=run_dt,
            filename="holdings.parquet",
            df=holdings_df,
        )

        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="health",
            dt=run_dt,
            filename="health.json",
            payload=health_latest_payload,
            update_latest=update_latest,
        )

        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="configs/score_config",
            dt=run_dt,
            filename="score_config.json",
            payload=asdict(score_cfg),
            update_latest=update_latest,
        )

        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table=MARKET_RESCALE_STATE_TABLE,
            dt=run_dt,
            filename="state.json",
            payload={"as_of": as_of_market_date, "label": cur_label, "leverage": float(cur_lev)},
            update_latest=update_latest,
        )

        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="inputs/positions",
            dt=run_dt,
            filename="positions.json",
            payload={t: asdict(p) for t, p in positions.items()},
            update_latest=update_latest,
        )

        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table=TAKE_PROFIT_STATE_TABLE,
            dt=run_dt,
            filename="state.json",
            payload={
                **asdict(tp_res.next_state),
                "as_of": as_of_market_date,
                "meta": {
                    "m_star": float(tp_res.m_star),
                    "do_harvest": bool(tp_res.do_harvest),
                    "r_anchor": tp_res.r_anchor,
                    "dd": tp_res.dd,
                    "sharpe": tp_res.sharpe,
                    "reasons": tp_res.reasons,
                },
            },
            update_latest=update_latest,
        )

        print("\n[S3] Saved daily report + holdings + health + score_config + positions + portfolio_behavior_regime + take_profit_state (+ asset_tp if triggered).")

    return {
        "mode": mode,
        "root_prefix": root_prefix,
        "run_dt": run_dt.strftime("%Y-%m-%d"),
        "requested_as_of_date": requested_as_of_date,
        "as_of_market_date": as_of_market_date,
        "as_of_run_date": as_of_run_date,
        "equity": float(equity),
        "market_target_leverage": float(market_lev),
        "rebalance": asdict(decision),  # legacy field; use execution_signals.signals.rescale for new consumers
        "execution_signals": execution_signals,
        "transition_assessment_ref": execution_signals.get("transition_assessment_ref"),
        "should_reoptimize": bool(reopt),
        "health": asdict(current_health),
        "health_latest": health_latest_payload,
        "health_score": health_score_payload,
        "evaluation_metadata": evaluation_metadata,
        "metric_plausibility": metric_plausibility,
        "portfolio_behavior_regime": portfolio_behavior_regime,
        "asset_ids": asset_ids,
        "asset_id_by_ticker": ticker_to_asset,
        "asset_identity_mode": "asset_id_first",
        "actuarial_diagnostics": actuarial_diagnostics,
        "bench_ann_return": None if bench_ann_ret is None else float(bench_ann_ret),
        "take_profit": {
            "do_harvest": bool(tp_res.do_harvest),
            "m_star": float(tp_res.m_star),
            "r_anchor": tp_res.r_anchor,
            "dd": tp_res.dd,
            "sharpe": tp_res.sharpe,
            "reasons": tp_res.reasons,
        },
        "take_profit_by_asset": None if asset_tp_plan_df is None else {
            "n_rows": int(len(asset_tp_plan_df)),
            "gross_target": float(equity) * float(lev_target),
        },
        "reinvest": reinvest_meta,
    }

def _parse_goals_arg(x: str | None) -> list[float] | None:
    if x is None:
        return None

    parts = [p.strip() for p in str(x).split(",") if p.strip()]
    if not parts:
        return None

    if len(parts) != 3:
        raise ValueError(
            f"--goals must contain exactly 3 comma-separated numbers, got {x!r}. "
            "Example: --goals 7500,10000,12500"
        )

    return [float(p) for p in parts]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Alpha Edge daily report.")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument(
        "--as-of",
        default=None,
        help="Report as-of date YYYY-MM-DD. Default: today.",
    )
    ap.add_argument(
        "--backtest-run-id",
        default=None,
        help="Optional backtest run id. Writes under <engine_root>/backtests/<id>.",
    )

    ap.add_argument(
        "--no-write",
        action="store_true",
        help="Run computation but do not write outputs to S3.",
    )
    ap.add_argument(
        "--no-latest",
        action="store_true",
        help="Write dated outputs but do not update latest.json pointers.",
    )

    ap.add_argument(
        "--equity-override",
        type=float,
        default=None,
        help="Override equity used for report computation.",
    )
    ap.add_argument(
        "--goals",
        default=None,
        help="Comma-separated goal ladder, e.g. 7500,10000,12500.",
    )
    ap.add_argument(
        "--main-goal",
        type=float,
        default=None,
        help="Main goal used by report scoring.",
    )
    ap.add_argument(
        "--actuarial-max-allowed-leverage",
        type=float,
        default=2.0,
        help="Policy cap used by actuarial safe-leverage diagnostics.",
    )
    ap.add_argument(
        "--actuarial-n-paths",
        type=int,
        default=20000,
        help="Number of Monte Carlo paths used for daily-report actuarial diagnostics.",
    )
    ap.add_argument(
        "--refresh-returns-cache",
        action="store_true",
        help="Rebuild returns_wide cache before the report. Default is to read the existing cache.",
    )

    return ap.parse_args()

def main() -> None:
    args = parse_args()

    if args.as_of:
        as_of = pd.Timestamp(args.as_of).strftime("%Y-%m-%d")
    else:
        # Live daily report should default to the latest reconciled ledger date,
        # not to today's calendar date. The market pipeline/report can run before
        # a same-day ledger exists, and defaulting to today causes NoSuchKey on
        # ledger/dt=<today>/positions.json.
        if args.backtest_run_id:
            raise ValueError(
                "--as-of is required when --backtest-run-id is provided."
            )

        cfg = load_runtime_config(args.env)
        s3 = s3_init(cfg.region)
        latest_pnl = s3_load_latest_json(
            s3,
            bucket=cfg.bucket,
            root_prefix=cfg.engine_root,
            table="ledger/pnl",
        )
        latest_summary = (
            latest_pnl.get("summary", {})
            if isinstance(latest_pnl, dict)
            else {}
        )
        resolved_as_of = (
            (latest_pnl or {}).get("as_of")
            or latest_summary.get("as_of")
        )
        if not resolved_as_of:
            raise RuntimeError(
                "Could not resolve --as-of from "
                f"s3://{cfg.bucket}/{cfg.engine_root.strip('/')}/ledger/pnl/latest.json. "
                "Pass --as-of explicitly."
            )
        as_of = pd.Timestamp(resolved_as_of).strftime("%Y-%m-%d")
        print(f"[as_of] resolved from ledger/pnl/latest.json: {as_of}")

    goals_override = _parse_goals_arg(args.goals)

    out = run_daily_cycle_asof(
        as_of=as_of,
        backtest_run_id=(args.backtest_run_id if args.backtest_run_id else None),
        write_outputs=(not bool(args.no_write)),
        update_latest=(not bool(args.no_latest)),
        equity_override=args.equity_override,
        goals_override=goals_override,
        main_goal_override=args.main_goal,
        env=args.env,
        confirm_prod_write=bool(args.confirm_prod_write),
        actuarial_max_allowed_leverage=float(args.actuarial_max_allowed_leverage),
        actuarial_n_paths=int(args.actuarial_n_paths),
        refresh_returns_cache=bool(args.refresh_returns_cache),
    )

    print("\n=== DAILY REPORT RESULT ===")
    print(f"env:                 {load_runtime_config(args.env).env}")
    print(f"mode:                {out.get('mode')}")
    print(f"root_prefix:         {out.get('root_prefix')}")
    print(f"run_dt:              {out.get('run_dt')}")
    print(f"as_of_market_date:   {out.get('as_of_market_date')}")
    print(f"as_of_run_date:      {out.get('as_of_run_date')}")
    print(f"equity:              {out.get('equity')}")
    hs = out.get("health_score") or {}
    if hs:
        print(f"health_score:        {hs.get('health_score')} / 100 ({hs.get('health_grade')})")
    print(f"market_leverage:     {out.get('market_target_leverage')}")
    print(f"should_reoptimize:   {out.get('should_reoptimize')}")

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
        script_name="run_daily_report.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            main()

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="daily_report",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="run_daily_report.py",
                source_mode="daily_report",
                status=("dry_run" if is_dry_run else "success"),
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                    "note": "Tier 1 audit event is entrypoint-level. Detailed output keys/row counts are available in the script log stdout and script-specific metadata where emitted by the script.",
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
        except Exception as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="daily_report",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="run_daily_report.py",
                source_mode="daily_report",
                status="failed",
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise


if __name__ == "__main__":
    main_with_audit()
