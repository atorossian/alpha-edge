# run_daily_report.py
from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from typing import Dict

import numpy as np
import pandas as pd

import yfinance as yf
import json
import io
import boto3

from alpha_edge import paths  # you already have this in other scripts

from alpha_edge.market.regime_filter import RegimeFilterState
from alpha_edge.core.schemas import ScoreConfig, Position
from alpha_edge.market.regime_leverage import leverage_from_hmm
from alpha_edge.portfolio.report_engine import (
    build_portfolio_report,
    summarize_report,
    print_hmm_summary,
    print_decision_addendum
)
from alpha_edge.market.hmm_engine import (
    GaussianHMM,
    compute_state_diagnostics,
    label_states_4,
    regime_probs_from_state_probs,
    select_regime_label,
)
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

RETURNS_WIDE_CACHE_PATH = "s3://alpha-edge-algo/market/cache/v1/returns_wide_min5y.parquet"
OHLCV_USD_ROOT = "s3://alpha-edge-algo/market/ohlcv_usd/v1"  # long parquet, partitioned (ticker/year)


def _resolve_root_prefix(*, backtest_run_id: str | None) -> str:
    if backtest_run_id:
        return f"{ENGINE_ROOT_PREFIX}/backtests/{backtest_run_id}"
    return ENGINE_ROOT_PREFIX

UNIVERSE_CSV_LOCAL = paths.universe_dir() / "universe.csv"

def _load_universe_ticker_to_asset_id() -> dict[str, str]:
    """
    Loads local universe.csv and returns {TICKER -> ASSET_ID}.
    Uses include==1 preference if duplicates exist.
    """
    df = pd.read_csv(UNIVERSE_CSV_LOCAL)
    if df is None or df.empty:
        raise RuntimeError(f"Universe is empty: {UNIVERSE_CSV_LOCAL}")

    # normalize
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

    # If duplicates exist for a ticker, prefer include=1 then last
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
    """
    Loads adj_close_usd for given *tickers* from S3 OHLCV dataset that is partitioned by asset_id/year.

    Expected layout (Hive-style):
      s3://<bucket>/<root_prefix>/asset_id=<ASSET_ID>/year=<YYYY>/...parquet

    Returns:
      DataFrame indexed by date, columns are tickers, values are adj_close_usd (ffilled).
    """
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

    # Map tickers to asset_ids
    ticker_asset = [(t, t2aid[t]) for t in tickers]
    asset_to_ticker = {aid: t for (t, aid) in ticker_asset}

    s3 = boto3.client("s3", region_name=s3_region)

    # List keys
    all_keys: list[tuple[str, str]] = []  # (asset_id, key)
    total_prefixes = len(ticker_asset) * len(years)
    seen_prefixes = 0

    print(f"[ohlcv] listing parquet keys assets={len(ticker_asset)} years={years[0]}..{years[-1]}")

    for (t, aid) in ticker_asset:
        for y in years:
            seen_prefixes += 1
            prefix = f"{s3_root_prefix}/asset_id={aid}/year={y}/"
            keys = _s3_list_keys(s3, s3_bucket, prefix)
            if keys:
                for k in keys:
                    all_keys.append((aid, k))

            # lightweight progress print every ~20 prefixes
            if seen_prefixes % 20 == 0:
                print(f"[ohlcv] listed prefixes={seen_prefixes}/{total_prefixes} keys_so_far={len(all_keys)}")

    if not all_keys:
        raise RuntimeError(
            f"No parquet files found under s3://{s3_bucket}/{s3_root_prefix} "
            f"for tickers={tickers[:5]}... years={years}"
        )

    # Read parquet files
    frames: list[pd.DataFrame] = []
    for (aid, key) in all_keys:
        df = _read_parquet_s3_bytes(s3, s3_bucket, key)
        if df is None or df.empty:
            continue

        # normalize columns (be defensive)
        cols = {c.lower(): c for c in df.columns}
        # try typical names
        date_col = cols.get("date")
        px_col = cols.get("adj_close_usd") or cols.get("close_usd") or cols.get("adj_close") or cols.get("close")
        if date_col is None or px_col is None:
            # if schema is unexpected, skip but don’t silently “succeed”
            raise RuntimeError(
                f"Unexpected OHLCV parquet schema in s3://{s3_bucket}/{key}. "
                f"Columns={list(df.columns)}"
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

    # Map asset_id -> ticker for columns
    long["ticker"] = long["asset_id"].map(asset_to_ticker).fillna(long["asset_id"])

    # Deduplicate (date,ticker) if multiple parquet shards overlap
    long = long.sort_values(["date", "ticker"])
    if long.duplicated(subset=["date", "ticker"]).any():
        n_dup = int(long.duplicated(subset=["date", "ticker"], keep=False).sum())
        sample = long.loc[long.duplicated(subset=["date", "ticker"], keep=False), ["date", "ticker"]].head(10)
        print(f"[ohlcv][warn] found {n_dup} duplicate (date,ticker) rows; collapsing by last()")
        print(sample.to_string(index=False))

        long = (
            long.groupby(["date", "ticker"], as_index=False)["adj_close_usd"]
            .last()
        )

    closes = (
        long.set_index(["date", "ticker"])["adj_close_usd"]
        .unstack("ticker")
        .sort_index()
        .ffill()
    )

    return closes

def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{100.0 * float(x):.2f}%"


def _fmt_num(x: float | None, *, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{float(x):.{nd}f}"

def _fetch_spot_prices_usd(
    *,
    tickers: list[str],
    provider_map: dict[str, str] | None = None,
    fallback_prices: pd.Series | None = None,
) -> pd.Series:
    """
    Fetch spot-ish prices at runtime (best effort) using yfinance.

    IMPORTANT:
      - Use intraday bars for "price at report generation time".
      - Fallback to last close (from S3 OHLCV) if missing.
    """
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

                # Case A: (ticker, field)
                if any(y in lvl0 for y in yahoo_list):
                    for y in yahoo_list:
                        if y not in df.columns.get_level_values(0):
                            continue
                        sub = df[y]
                        if ("Close" in sub.columns) and (not sub["Close"].dropna().empty):
                            spot_by_yahoo[y] = float(sub["Close"].dropna().iloc[-1])
                        elif ("Adj Close" in sub.columns) and (not sub["Adj Close"].dropna().empty):
                            spot_by_yahoo[y] = float(sub["Adj Close"].dropna().iloc[-1])

                # Case B: (field, ticker)
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
                # Single ticker case
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

    s = pd.Series(out, dtype="float64")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def run_daily_cycle_asof(
    *,
    as_of: str,
    backtest_run_id: str | None = None,
    write_outputs: bool = True,
    update_latest: bool = True,
    # backtest knobs (optional)
    equity_override: float | None = None,
    goals_override: list[float] | None = None,
    main_goal_override: float | None = None,
) -> dict:
    """
    One daily report cycle for a given AS_OF date.

    - live (backtest_run_id=None): writes to engine/v1/...
    - backtest: writes to engine/v1/backtests/<run_id>/...

    Returns a dict of key decisions + metrics for the backtest driver.
    """
    root_prefix = _resolve_root_prefix(backtest_run_id=backtest_run_id)
    mode = "backtest" if backtest_run_id else "live"

    as_of_ts = pd.Timestamp(as_of).tz_localize(None).normalize()
    as_of_date = as_of_ts.strftime("%Y-%m-%d")

    # output partition date
    run_dt = pd.Timestamp(dt.date.today()).normalize() if mode == "live" else as_of_ts
    as_of_run_date = run_dt.strftime("%Y-%m-%d")

    # IMPORTANT: avoid “silent wrong backtest”
    if mode == "backtest" and equity_override is None:
        raise ValueError("backtest requires equity_override (do not rely on hardcoded equity).")

    s3 = s3_init(ENGINE_REGION)
    market = MarketStore(bucket=ENGINE_BUCKET)  # market data stays global

    BENCH_PROXY = ["VT", "SPY", "QQQ", "IWM", "TLT", "VCIT", "GLD"]
    BENCH_NAME = "EQW(VT,SPY,QQQ,IWM,TLT,VCIT,GLD)"
    START_HISTORY = "2015-01-01"
    RESCALE_STATE_TABLE = "rescale/state"
    RESCALE_PLAN_TABLE = "rescale/plan"

    GOALS = goals_override if goals_override is not None else [7500.0, 10000.0, 12500.0]
    MAIN_GOAL = float(main_goal_override if main_goal_override is not None else 10000.0)
    equity = float(equity_override) if equity_override is not None else 5543.10  # live fallback only

    # ---------- Load MARKET regime (GLOBAL path, not backtest root) ----------
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

    # ---------- Inputs (FROM root_prefix: live or backtest) ----------
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

    raw_baseline = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=root_prefix, table="health"
    )
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

    # ---------- Load closes USD (as_of!) ----------
    end_date = as_of_date
    closes_all = _load_closes_usd_from_ohlcv(tickers=tickers_all, start=START_HISTORY, end=end_date)
    latest_close_prices = closes_all.iloc[-1]

    # ---------- Prices for valuation ----------
    # Backtest: do NOT peek intraday, use last close.
    # Live: allow yfinance intraday.
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

    # Derivatives/notional -> synthetic qty using valuation price
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

    # IMPORTANT for shorts: use signed exposure weights normalized by GROSS
    values = np.array(
        [float(prices_for_valuation[t]) * float(positions[t].quantity) for t in tickers],
        dtype=np.float64,
    )
    gross = float(np.sum(np.abs(values)))
    if not np.isfinite(gross) or gross <= 0:
        raise ValueError("Gross exposure == 0 (or non-finite) from positions/prices")
    w_vec = values / gross

    # Portfolio returns for HMM: history-based, weighted by current (gross-normalized signed) exposures
    port_rets = (rets_assets[tickers] * w_vec).sum(axis=1).dropna()
    as_of_market_dt = pd.Timestamp(port_rets.index[-1]).normalize()
    as_of_market_date = as_of_market_dt.strftime("%Y-%m-%d")
    print(f"[dates] as_of_market_date={as_of_market_date} | as_of_run_date={as_of_run_date}")

    # ---------- Portfolio HMM (unchanged logic, just no duplication) ----------
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
        # --- regime labels time-series aligned to X (for regime-aware alpha later) ---
        regime_labels = None
        try:
            # X corresponds to r[mask] and vol20[mask]
            idx = port_rets.index[mask]  # same length/order as filtered rows

            # map state -> label using mapping computed from diagnostics
            # filtered rows correspond to idx
            labs = []
            for p_state in filtered:
                k = int(np.argmax(p_state))
                labs.append(mapping.get(k, "UNKNOWN"))

            regime_labels = pd.Series(labs, index=idx, name="regime")
        except Exception:
            regime_labels = None

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

    # ---------- Rebalance planning ----------
    raw_reb_state = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=root_prefix, table=RESCALE_STATE_TABLE
    )
    reb_state = RebalanceState(
        last_rebalance_date=(raw_reb_state or {}).get("last_rebalance_date"),
        last_rebalance_equity=(raw_reb_state or {}).get("last_rebalance_equity"),
    )

    positions_qty = {t: float(p.quantity) for t, p in positions.items()}

    gross_now = compute_gross_notional_from_positions(
        positions_qty=positions_qty,
        prices_usd=prices_for_valuation,
    )

    lev_target = float(market_lev)

    decision = should_rebalance(
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

    cols: list[str] = []  # IMPORTANT: avoid UnboundLocalError in logs/except paths

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

    # optional: print alpha report
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

    # ---------- Rescale plan persistence ----------
    if decision.should_rebalance:
        plan = build_rescale_plan(
            as_of=as_of_market_date,
            equity=float(equity),
            recommended_leverage=float(lev_target),
            positions_qty=positions_qty,
            prices_usd=prices_for_valuation,
            max_notional_cap=None,
        )
        print_decision_addendum(
            decision=decision,
            health=current_health,
            bench_ann_ret=bench_ann_ret,
            reopt=reopt,
            plan=plan,
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

        if write_outputs:
            s3_write_parquet_partition(
                s3,
                bucket=ENGINE_BUCKET,
                root_prefix=root_prefix,
                table=RESCALE_PLAN_TABLE,
                dt=run_dt,
                filename="plan.parquet",
                df=plan_df,
            )

            s3_write_json_event(
                s3,
                bucket=ENGINE_BUCKET,
                root_prefix=root_prefix,
                table=RESCALE_STATE_TABLE,
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
    else:
        print_decision_addendum(
            decision=decision,
            health=current_health,
            bench_ann_ret=bench_ann_ret,
            reopt=reopt,
            plan=None,
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
                    "spot_prices_usd": {k: (None if not np.isfinite(v) else float(v)) for k, v in prices_for_valuation.items()},
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
            table="inputs/positions",
            dt=run_dt,
            filename="positions.json",
            payload={t: asdict(p) for t, p in positions.items()},
            update_latest=update_latest,
        )

        print("\n[S3] Saved daily report + holdings + health + score_config + positions + regimes.")

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
