# run_daily_report.py
from __future__ import annotations

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
)
from alpha_edge.portfolio.rebalance_engine import (
    RebalanceState,
    should_rebalance,
    build_rescale_plan,
    compute_gross_notional_from_positions,
)

ENGINE_BUCKET = "alpha-edge-algo"
ENGINE_REGION = "eu-west-1"
ENGINE_ROOT_PREFIX = "engine/v1"

TAKE_PROFIT_STATE_TABLE = "take_profit/state"
TAKE_PROFIT_PLAN_TABLE = "take_profit/plan"

# per-asset TP state/plan
TAKE_PROFIT_ASSETS_STATE_TABLE = "take_profit/assets_state"
TAKE_PROFIT_ASSETS_PLAN_TABLE = "take_profit/assets_plan"

MARKET_RESCALE_STATE_TABLE = "regimes/market_rescale_state"

RETURNS_WIDE_CACHE_PATH = "s3://alpha-edge-algo/market/cache/v1/returns_wide_min5y.parquet"
OHLCV_USD_ROOT = "s3://alpha-edge-algo/market/ohlcv_usd/v1"


def _resolve_root_prefix(*, backtest_run_id: str | None) -> str:
    if backtest_run_id:
        return f"{ENGINE_ROOT_PREFIX}/backtests/{backtest_run_id}"
    return ENGINE_ROOT_PREFIX


UNIVERSE_CSV_LOCAL = paths.universe_dir() / "universe.csv"


def _load_universe_ticker_to_asset_id() -> dict[str, str]:
    df = pd.read_csv(UNIVERSE_CSV_LOCAL)
    if df is None or df.empty:
        raise RuntimeError(f"Universe is empty: {UNIVERSE_CSV_LOCAL}")

    for c in ["ticker", "asset_id"]:
        if c not in df.columns:
            raise RuntimeError(f"Universe missing required column '{c}': {UNIVERSE_CSV_LOCAL}")

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()

    if "include" in df.columns:
        df["include"] = pd.to_numeric(df["include"], errors="coerce").fillna(1).astype(int)
    else:
        df["include"] = 1

    df = df[(df["ticker"] != "") & (df["asset_id"] != "")].copy()
    if df.empty:
        raise RuntimeError("Universe has no (ticker,asset_id) pairs after normalization.")

    df = df.sort_values(["ticker", "include"], ascending=[True, False])
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    return dict(zip(df["ticker"].tolist(), df["asset_id"].tolist()))


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
) -> dict:
    root_prefix = _resolve_root_prefix(backtest_run_id=backtest_run_id)
    mode = "backtest" if backtest_run_id else "live"

    as_of_ts = pd.Timestamp(as_of).tz_localize(None).normalize()
    as_of_date = as_of_ts.strftime("%Y-%m-%d")

    run_dt = pd.Timestamp(dt.date.today()).normalize() if mode == "live" else as_of_ts
    as_of_run_date = run_dt.strftime("%Y-%m-%d")

    # --- RETURNS_WIDE (universe-wide) ---
    u = pd.read_csv(UNIVERSE_CSV_LOCAL)
    u = u[u.get("include", 1).fillna(1).astype(int) == 1].copy()
    u["asset_id"] = u["asset_id"].astype(str).str.strip()
    u["ticker"] = u.get("ticker", u["asset_id"]).astype(str).str.upper().str.strip()
    asset_to_ticker = dict(zip(u["asset_id"], u["ticker"]))

    cache_cfg = CacheConfig(bucket=ENGINE_BUCKET, min_years=float(5.0))
    build_returns_wide_cache(cache_cfg)

    returns_wide = pd.read_parquet(RETURNS_WIDE_CACHE_PATH, engine="pyarrow").sort_index()
    returns_wide, _ = clean_returns_matrix(returns_wide)

    returns_wide.index = pd.to_datetime(returns_wide.index, errors="coerce").tz_localize(None).normalize()
    returns_wide = returns_wide.loc[returns_wide.index <= as_of_ts]
    if returns_wide.shape[0] < 252:
        raise RuntimeError(f"Not enough returns history up to as_of={as_of_date}: rows={returns_wide.shape[0]}")

    returns_wide = returns_wide.rename(columns=lambda c: asset_to_ticker.get(str(c).strip(), str(c).strip()))
    returns_wide.columns = [str(c).upper().strip() for c in returns_wide.columns]

    if mode == "backtest" and equity_override is None:
        raise ValueError("backtest requires equity_override (do not rely on hardcoded equity).")

    s3 = s3_init(ENGINE_REGION)
    market = MarketStore(bucket=ENGINE_BUCKET)

    BENCH_PROXY = ["VT", "SPY", "QQQ", "IWM", "TLT", "VCIT", "GLD"]
    BENCH_NAME = "EQW(VT,SPY,QQQ,IWM,TLT,VCIT,GLD)"
    START_HISTORY = "2015-01-01"

    RESCALE_STATE_TABLE = "rescale/state"
    RESCALE_PLAN_TABLE = "rescale/plan"

    GOALS = goals_override if goals_override is not None else [7500.0, 10000.0, 12500.0]
    MAIN_GOAL = float(main_goal_override if main_goal_override is not None else 10000.0)
    equity = float(equity_override) if equity_override is not None else 7019.61

    # ---------- Load MARKET regime (GLOBAL path) ----------
    market_hmm_payload = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="regimes/market_hmm"
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
    raw_ledger_positions = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=root_prefix, table="ledger/positions"
    )
    if not raw_ledger_positions:
        raise RuntimeError(f"Missing S3 latest ledger positions under {root_prefix}/ledger/positions/latest.json")

    spot_rows, deriv_rows = parse_ledger_positions_obj(raw_ledger_positions)
    if not spot_rows and not deriv_rows:
        raise RuntimeError("Ledger positions payload has no spot_positions and no derivatives_positions.")

    raw_score_cfg = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=root_prefix, table="configs/score_config"
    )
    if not raw_score_cfg:
        raise RuntimeError("Missing S3 latest score_config.")
    score_cfg = ScoreConfig(**raw_score_cfg)

    raw_baseline = s3_load_latest_json(s3, bucket=ENGINE_BUCKET, root_prefix=root_prefix, table="health")
    baseline = parse_portfolio_health(raw_baseline) if raw_baseline else None

    last_score = s3_load_latest_report_score(s3, bucket=ENGINE_BUCKET, root_prefix=root_prefix)
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
    closes_all = _load_closes_usd_from_ohlcv(tickers=tickers_all, start=START_HISTORY, end=end_date)
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
    rets_assets = closes.pct_change().dropna(how="any")

    values = np.array([float(prices_for_valuation[t]) * float(positions[t].quantity) for t in tickers], dtype=np.float64)
    gross = float(np.sum(np.abs(values)))
    if not np.isfinite(gross) or gross <= 0:
        raise ValueError("Gross exposure == 0 (or non-finite) from positions/prices")
    w_vec = values / gross

    port_rets = (rets_assets[tickers] * w_vec).sum(axis=1).dropna()
    as_of_market_dt = pd.Timestamp(port_rets.index[-1]).normalize()
    as_of_market_date = as_of_market_dt.strftime("%Y-%m-%d")
    print(f"[dates] as_of_market_date={as_of_market_date} | as_of_run_date={as_of_run_date}")

    # ---------- Portfolio HMM ----------
    r = port_rets.to_numpy(dtype=np.float64)
    vol20 = pd.Series(r, index=port_rets.index).rolling(20).std().to_numpy(dtype=np.float64)
    mask = np.isfinite(vol20)
    X = np.column_stack([r[mask], vol20[mask]])

    hmm_res = None
    regime_labels = None
    if X.shape[0] >= 80:
        hmm = GaussianHMM(n_states=4, n_dim=2, seed=42)
        fit_res = hmm.fit(X, max_iter=150, tol=1e-4, verbose=False)

        filtered = hmm.predict_proba(X)
        p_today = filtered[-1]

        r_aligned = r[mask]
        diags = compute_state_diagnostics(r_aligned, filtered)
        mapping = label_states_4(diags)

        try:
            idx = port_rets.index[mask]
            labs = []
            for p_state in filtered:
                k = int(np.argmax(p_state))
                labs.append(mapping.get(k, "UNKNOWN"))
            regime_labels = pd.Series(labs, index=idx, name="regime")
        except Exception:
            regime_labels = None

        p_label_today = regime_probs_from_state_probs(p_today, mapping)
        label_commit = select_regime_label(p_label_today, commit_threshold=0.65)

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
                "commit_threshold": 0.65,
                "features": ["port_return", "vol20"],
                "last_date_used": as_of_market_date,
            },
        }
        print_hmm_summary(hmm_res)
    else:
        print("\n[HMM] Not enough history to fit 4-state HMM (need >= ~80 after vol window).")

    st_raw = market.read_regime_filter_state() or {}
    filter_state = RegimeFilterState(
        last_date=st_raw.get("last_date"),
        chosen_label=st_raw.get("chosen_label"),
        days_in_regime=int(st_raw.get("days_in_regime", 0) or 0),
        probs_smoothed=st_raw.get("probs_smoothed"),
    )

    lev_rec = leverage_from_hmm(
        hmm_res or {},
        default=1.0,
        risk_appetite=0.8,
        hard_cap=12.0,
        filter_state=filter_state,
        as_of=as_of_market_date,
        filter_alpha=0.20,
        min_hold_days=3,
        min_prob_to_switch=0.60,
        min_margin_to_switch=0.12,
    )
    if isinstance(lev_rec.get("filter_state"), dict):
        market.write_regime_filter_state(lev_rec["filter_state"])

    # ---------- Market RESCALE trigger (ONLY on regime / leverage change) ----------
    raw_mkt_state = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=root_prefix, table=MARKET_RESCALE_STATE_TABLE
    ) or {}
    prev_label = raw_mkt_state.get("label")
    prev_lev = raw_mkt_state.get("leverage")

    cur_label = str(
        (lev_rec or {}).get("filter_state", {}).get("chosen_label")
        or (hmm_res or {}).get("label_commit")
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

    # ---------- Report ----------
    report = build_portfolio_report(
        closes=closes,
        positions=positions,
        equity=equity,
        goals=GOALS,
        main_goal=MAIN_GOAL,
        score_config=score_cfg,
        prices_usd=prices_for_valuation,
    )
    print(summarize_report(report))

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
        s3, bucket=ENGINE_BUCKET, root_prefix=root_prefix, table=TAKE_PROFIT_STATE_TABLE
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
            s3, bucket=ENGINE_BUCKET, root_prefix=root_prefix, table=TAKE_PROFIT_ASSETS_STATE_TABLE
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
                bucket=ENGINE_BUCKET,
                root_prefix=root_prefix,
                table=TAKE_PROFIT_ASSETS_PLAN_TABLE,
                dt=run_dt,
                filename="asset_tp_plan.parquet",
                df=asset_tp_plan_df,
            )

        if write_outputs and (next_asset_anchors is not None):
            s3_write_json_event(
                s3,
                bucket=ENGINE_BUCKET,
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
            bucket=ENGINE_BUCKET,
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
        s3, bucket=ENGINE_BUCKET, root_prefix=root_prefix, table="rescale/state"
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
                bucket=ENGINE_BUCKET,
                root_prefix=root_prefix,
                table="rescale/plan",
                dt=run_dt,
                filename="plan.parquet",
                df=plan_df,
            )

            s3_write_json_event(
                s3,
                bucket=ENGINE_BUCKET,
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
                    bucket=ENGINE_BUCKET,
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
        )

    # ---------- Persist outputs ----------
    if write_outputs:
        s3_write_json_event(
            s3,
            bucket=ENGINE_BUCKET,
            root_prefix=root_prefix,
            table="regimes/hmm",
            dt=run_dt,
            filename="hmm.json",
            payload={
                "as_of": as_of_market_date,
                "tickers": list(tickers),
                "hmm": hmm_res,
                "leverage_recommendation": lev_rec,
                "meta": {
                    "as_of_market_date": as_of_market_date,
                    "as_of_run_date": as_of_run_date,
                    "pricing_as_of_utc": pricing_as_of_utc,
                },
            },
            update_latest=update_latest,
        )

        s3_write_json_event(
            s3,
            bucket=ENGINE_BUCKET,
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
                "inputs": {
                    "equity": equity,
                    "goals": GOALS,
                    "main_goal": MAIN_GOAL,
                    "benchmark": {
                        "name": bench_meta.get("name"),
                        "tickers": bench_meta.get("tickers"),
                        "method": bench_meta.get("method"),
                        "ann_return": bench_ann_ret,
                        "meta": bench_meta,
                    },
                    "tickers": tickers,
                    "start_history": START_HISTORY,
                    "spot_prices_usd": {
                        k: (None if not np.isfinite(v) else float(v))
                        for k, v in prices_for_valuation.items()
                    },
                    "market_regime": {"target_leverage": float(market_lev), "source_table": "regimes/market_hmm"},
                },
                "flags": {
                    "should_reoptimize": bool(reopt),
                    "baseline_exists": bool(baseline is not None),
                },
            },
            update_latest=update_latest,
        )

        holdings_df = pd.DataFrame(report.snapshot.positions_table)
        s3_write_parquet_partition(
            s3,
            bucket=ENGINE_BUCKET,
            root_prefix=root_prefix,
            table="holdings",
            dt=run_dt,
            filename="holdings.parquet",
            df=holdings_df,
        )

        s3_write_json_event(
            s3,
            bucket=ENGINE_BUCKET,
            root_prefix=root_prefix,
            table="health",
            dt=run_dt,
            filename="health.json",
            payload=asdict(current_health),
            update_latest=update_latest,
        )

        s3_write_json_event(
            s3,
            bucket=ENGINE_BUCKET,
            root_prefix=root_prefix,
            table="configs/score_config",
            dt=run_dt,
            filename="score_config.json",
            payload=asdict(score_cfg),
            update_latest=update_latest,
        )

        s3_write_json_event(
            s3,
            bucket=ENGINE_BUCKET,
            root_prefix=root_prefix,
            table=MARKET_RESCALE_STATE_TABLE,
            dt=run_dt,
            filename="state.json",
            payload={"as_of": as_of_market_date, "label": cur_label, "leverage": float(cur_lev)},
            update_latest=update_latest,
        )

        s3_write_json_event(
            s3,
            bucket=ENGINE_BUCKET,
            root_prefix=root_prefix,
            table="inputs/positions",
            dt=run_dt,
            filename="positions.json",
            payload={t: asdict(p) for t, p in positions.items()},
            update_latest=update_latest,
        )

        s3_write_json_event(
            s3,
            bucket=ENGINE_BUCKET,
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

        print("\n[S3] Saved daily report + holdings + health + score_config + positions + regimes + take_profit_state (+ asset_tp if triggered).")

    return {
        "mode": mode,
        "root_prefix": root_prefix,
        "run_dt": run_dt.strftime("%Y-%m-%d"),
        "as_of_market_date": as_of_market_date,
        "as_of_run_date": as_of_run_date,
        "equity": float(equity),
        "market_target_leverage": float(market_lev),
        "rebalance": asdict(decision),
        "should_reoptimize": bool(reopt),
        "health": asdict(current_health),
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


def main():
    as_of = pd.Timestamp(dt.date.today()).strftime("%Y-%m-%d")
    run_daily_cycle_asof(
        as_of=as_of,
        backtest_run_id=None,
        write_outputs=True,
        update_latest=True,
        equity_override=None,
        goals_override=None,
        main_goal_override=None,
    )


if __name__ == "__main__":
    main()
