# compute_market_regime.py
from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from typing import Dict, List
from alpha_edge.portfolio.report_engine import print_hmm_summary
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.data_loader import s3_init, s3_write_json_event
from alpha_edge.market.regime_filter import RegimeFilterState
from alpha_edge.market.regime_leverage import leverage_from_hmm
from alpha_edge.market.hmm_engine import (
    GaussianHMM,
    compute_state_diagnostics,
    label_states_4,
    regime_probs_from_state_probs,
    select_regime_label,
)

ENGINE_BUCKET = "alpha-edge-algo"
ENGINE_REGION = "eu-west-1"
ENGINE_ROOT_PREFIX = "engine/v1"

OHLCV_USD_ROOT = "s3://alpha-edge-algo/market/ohlcv_usd/v1"

# Option B: composite proxy (equal-weight)
PROXY_TICKERS = ["VT", "SPY", "QQQ", "IWM", "TLT", "VCIT", "GLD"]

# history window for regime fitting
START_HISTORY = "2015-01-01"


def _to_day(x) -> pd.Timestamp:
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {x!r}")
    ts = pd.Timestamp(ts)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _load_closes_usd_from_ohlcv(
    *,
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Load adj_close_usd for tickers from the OHLCV parquet dataset and pivot wide.
    """
    start_ts = _to_day(start)
    end_ts = _to_day(end)

    years = list(range(int(start_ts.year), int(end_ts.year) + 1))
    dataset = ds.dataset(OHLCV_USD_ROOT, format="parquet", partitioning="hive")

    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    if not tickers:
        return pd.DataFrame()

    filt = ds.field("ticker").isin(tickers) & ds.field("year").isin(years)

    table = dataset.to_table(
        filter=filt,
        columns=["date", "ticker", "adj_close_usd"],
    )
    df = table.to_pandas()
    if df is None or df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()

    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    if df.empty:
        return pd.DataFrame()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["adj_close_usd"] = pd.to_numeric(df["adj_close_usd"], errors="coerce")
    df = df.dropna(subset=["adj_close_usd"])

    # dedupe (date,ticker) => last
    df = df.sort_values(["date", "ticker"])
    dup = df.duplicated(subset=["date", "ticker"], keep=False)
    if dup.any():
        df = df.groupby(["date", "ticker"], as_index=False)["adj_close_usd"].last()

    closes = (
        df.set_index(["date", "ticker"])["adj_close_usd"]
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
    Returns: (proxy_returns, meta)
    """
    if closes is None or closes.empty:
        return pd.Series(dtype="float64"), {"kept": [], "dropped": [], "min_assets_per_day": min_assets_per_day}

    rets = closes.pct_change()
    # Require at least N assets available that day
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
        "assets_present_sample": {str(d.date()): int(available.loc[d]) for d in available.index[-5:]},
        "kept": kept,
        "dropped": dropped,
    }
    return proxy.astype("float64"), meta


def compute_market_regime() -> None:
    run_dt = pd.Timestamp(dt.date.today()).normalize()
    as_of_run_date = run_dt.strftime("%Y-%m-%d")

    s3 = s3_init(ENGINE_REGION)
    market = MarketStore(bucket=ENGINE_BUCKET)

    # Use your returns-latest-state if present, so we align to what ingestion produced.
    latest_state = market.read_returns_latest_state() or {}
    end_date = str(latest_state.get("last_date") or as_of_run_date)

    closes = _load_closes_usd_from_ohlcv(
        tickers=PROXY_TICKERS,
        start=START_HISTORY,
        end=end_date,
    )

    if closes is None or closes.empty:
        raise RuntimeError("Market regime: no closes available for proxy tickers in OHLCV dataset.")

    proxy_rets, proxy_meta = _compute_equal_weight_proxy_returns(closes, min_assets_per_day=3)
    if proxy_rets.empty or proxy_rets.shape[0] < 120:
        raise RuntimeError(f"Market regime: insufficient proxy return history (n={proxy_rets.shape[0]}).")

    as_of_market_dt = pd.Timestamp(proxy_rets.index[-1]).normalize()
    as_of_market_date = as_of_market_dt.strftime("%Y-%m-%d")

    r = proxy_rets.to_numpy(dtype=np.float64)
    vol20 = pd.Series(r, index=proxy_rets.index).rolling(20).std().to_numpy(dtype=np.float64)
    mask = np.isfinite(vol20)
    X = np.column_stack([r[mask], vol20[mask]])

    hmm_res = None
    if X.shape[0] >= 80:
        hmm = GaussianHMM(n_states=4, n_dim=2, seed=42)
        fit_res = hmm.fit(X, max_iter=150, tol=1e-4, verbose=False)

        filtered = hmm.predict_proba(X)
        p_today = filtered[-1]

        r_aligned = r[mask]
        diags = compute_state_diagnostics(r_aligned, filtered)
        mapping = label_states_4(diags)
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
                "features": ["proxy_return_eqw", "vol20"],
                "last_date_used": as_of_market_date,
                "proxy": {
                    "method": "equal_weight_basket",
                    "tickers": PROXY_TICKERS,
                },
                "proxy_meta": proxy_meta,
            },
        }
    else:
        raise RuntimeError(f"Market regime: not enough observations after vol window (X={X.shape}).")

    # filter state lives in MarketStore (same as your current code)
    st_raw = market.read_regime_filter_state() or {}
    filter_state = RegimeFilterState(
        last_date=st_raw.get("last_date"),
        chosen_label=st_raw.get("chosen_label"),
        days_in_regime=int(st_raw.get("days_in_regime", 0) or 0),
        probs_smoothed=st_raw.get("probs_smoothed"),
    )

    # NOTE: choose your preferred risk appetite for the market regime driver
    lev_rec = leverage_from_hmm(
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

    if isinstance(lev_rec.get("filter_state"), dict):
        market.write_regime_filter_state(lev_rec["filter_state"])

    pricing_as_of_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "as_of": as_of_market_date,
        "proxy": {
            "method": "equal_weight_basket",
            "tickers": PROXY_TICKERS,
        },
        "hmm": hmm_res,
        "leverage_recommendation": lev_rec,
        "meta": {
            "as_of_market_date": as_of_market_date,
            "as_of_run_date": as_of_run_date,
            "pricing_as_of_utc": pricing_as_of_utc,
            "end_date_requested": end_date,
            "start_history": START_HISTORY,
        },
    }

    # Write to engine/v1/regimes/market_hmm/...
    s3_write_json_event(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
        table="regimes/market_hmm",
        dt=run_dt,
        filename="regime.json",
        payload=payload,
        update_latest=True,
    )
    print_hmm_summary(hmm_res, lev_rec=lev_rec)
    print("[OK] wrote market regime -> engine/v1/regimes/market_hmm (latest.json updated)")
    print(f"[OK] as_of_market_date={as_of_market_date} target_leverage={float(lev_rec.get('leverage', 1.0)):.2f}x")


if __name__ == "__main__":
    compute_market_regime()
