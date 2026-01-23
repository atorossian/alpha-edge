# run_daily_report.py
from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yfinance as yf

from alpha_edge.market.regime_filter import RegimeFilterState
from alpha_edge.core.schemas import ScoreConfig, Position
from alpha_edge.market.regime_leverage import leverage_from_hmm
from alpha_edge.portfolio.report_engine import build_portfolio_report, summarize_report, print_hmm_summary
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


def _load_closes_usd_from_ohlcv(
    *,
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Loads adj_close_usd for given tickers from the OHLCV long parquet dataset and pivots to wide closes.
    NOTE: This can be moderately heavy but is fine for "positions tickers" scale.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    years = list(range(int(start_ts.year), int(end_ts.year) + 1))
    dataset = ds.dataset(OHLCV_USD_ROOT, format="parquet", partitioning="hive")

    # Filters: ticker IN (...) AND year IN (...)
    filt = ds.field("ticker").isin([str(t) for t in tickers]) & ds.field("year").isin(years)

    table = dataset.to_table(
        filter=filt,
        columns=["date", "ticker", "adj_close_usd"],
    )
    df = table.to_pandas()
    if df.empty:
        raise RuntimeError("No OHLCV USD rows found for requested tickers/date range.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    if df.empty:
        raise RuntimeError("OHLCV data exists but none in requested date window.")

    df["ticker"] = df["ticker"].astype(str)
    df["adj_close_usd"] = pd.to_numeric(df["adj_close_usd"], errors="coerce")

    # --- dedupe (date,ticker) to make unstack safe ---
    df = df.sort_values(["date", "ticker"])
    dup = df.duplicated(subset=["date", "ticker"], keep=False)
    if dup.any():
        n_dup = int(dup.sum())
        sample = df.loc[dup, ["date", "ticker"]].head(10)
        print(f"[ohlcv][warn] found {n_dup} duplicate (date,ticker) rows; collapsing by last()")
        print(sample.to_string(index=False))

        df = (
            df.groupby(["date", "ticker"], as_index=False)["adj_close_usd"]
              .last()
        )

    closes = (
        df.set_index(["date", "ticker"])["adj_close_usd"]
          .sort_index()
          .unstack("ticker")
          .sort_index()
          .ffill()
    )
    return closes


def _compute_benchmark_ann_return_from_closes(closes: pd.Series) -> float | None:
    """
    Simple annualized return estimate from daily closes:
      ann_ret = mean(daily_returns) * 252
    """
    s = pd.to_numeric(closes, errors="coerce").dropna()
    if len(s) < 50:
        return None
    r = s.pct_change().dropna()
    if len(r) < 50:
        return None
    return float(r.mean() * 252.0)


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

    # internal -> yahoo
    internal_to_yahoo = {t: str(provider_map.get(t, t)).strip() for t in tickers}
    yahoo_list = [internal_to_yahoo[t] for t in tickers]

    df = yf.download(
        tickers=yahoo_list,
        period="1d",
        interval="1m",     # actual "now-ish" pricing
        progress=True,
        threads=True,
        auto_adjust=True,
        timeout=30,
    )

    spot_by_yahoo: Dict[str, float] = {}

    try:
        if df is not None and not df.empty:
            # Multi-ticker case => columns MultiIndex (field, ticker) OR (ticker, field) depending on yfinance version.
            # Your previous logic assumes (ticker, field). We'll handle both safely.

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
                        # prefer Close
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
                # Single ticker case => flat columns
                if "Close" in df.columns and not df["Close"].dropna().empty:
                    spot_by_yahoo[yahoo_list[0]] = float(df["Close"].dropna().iloc[-1])
                elif "Adj Close" in df.columns and not df["Adj Close"].dropna().empty:
                    spot_by_yahoo[yahoo_list[0]] = float(df["Adj Close"].dropna().iloc[-1])
    except Exception:
        spot_by_yahoo = {}

    # build internal series, fall back to last close if needed
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


def main():
    BENCH_PROXY = ["VT", "SPY", "QQQ", "IWM", "TLT", "HYG", "GLD"]
    BENCH_NAME = "EQW(VT,SPY,QQQ,IWM,TLT,HYG,GLD)"
    START_HISTORY = "2015-01-01"
    RESCALE_STATE_TABLE = "rescale/state"
    RESCALE_PLAN_TABLE = "rescale/plan"   # parquet partitioned by dt=run_dt
    GOALS = [1500.0, 2000.0, 3000.0]
    equity = 1224.57
    MAIN_GOAL = 2000.0

    # --- RUN date (execution date) ---
    run_dt = pd.Timestamp(dt.date.today()).normalize()
    as_of_run_date = run_dt.strftime("%Y-%m-%d")

    # ---- S3 clients / stores ----
    s3 = s3_init(ENGINE_REGION)
    market = MarketStore(bucket=ENGINE_BUCKET)

    # ---------- Load MARKET regime (computed outside daily report) ----------
    market_hmm_payload = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="regimes/market_hmm"
    ) or {}

    market_as_of = market_hmm_payload.get("as_of")
    market_as_of = str(market_as_of) if market_as_of else as_of_run_date

    market_lev = None
    if isinstance(market_hmm_payload, dict):
        lr = market_hmm_payload.get("leverage_recommendation") or {}
        if isinstance(lr, dict) and lr.get("leverage") is not None:
            market_lev = float(lr.get("leverage"))

    if market_lev is None:
        # hard fallback (shouldn't happen if compute_market_regime.py ran)
        market_lev = 1.0

    print(f"[market regime] as_of={market_as_of} target_leverage={market_lev:.2f}x")


    # ---------- Load latest inputs (S3-only) ----------
    raw_ledger_positions = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="ledger/positions"
    )
    if not raw_ledger_positions:
        raise RuntimeError("Missing S3 latest ledger positions. Expected engine/v1/ledger/positions/latest.json")

    spot_rows, deriv_rows = parse_ledger_positions_obj(raw_ledger_positions)
    if not spot_rows and not deriv_rows:
        raise RuntimeError("Ledger positions payload has no spot_positions and no derivatives_positions.")

    raw_score_cfg = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="configs/score_config"
    )
    if not raw_score_cfg:
        raise RuntimeError("Missing S3 latest score_config. Expected engine/v1/configs/score_config/latest.json")
    score_cfg = ScoreConfig(**raw_score_cfg)

    raw_baseline = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="health"
    )
    baseline = parse_portfolio_health(raw_baseline) if raw_baseline else None

    last_score = s3_load_latest_report_score(s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX)
    if last_score is not None:
        print(f"[last run] previous daily report score: {last_score:.4f}")
    else:
        print("[last run] previous daily report score: N/A")

    tickers_spot = [str(r.get("ticker")).upper().strip() for r in spot_rows if r.get("ticker")]
    tickers_deriv = [str(r.get("ticker")).upper().strip() for r in deriv_rows if r.get("ticker")]
    tickers_all = sorted(set(tickers_spot + tickers_deriv))
    if not tickers_all:
        raise RuntimeError("No tickers in ledger positions.")

    # ---------- Load closes USD from S3 OHLCV (history) ----------
    end_date = as_of_run_date
    closes_all = _load_closes_usd_from_ohlcv(tickers=tickers_all, start=START_HISTORY, end=end_date)
    latest_close_prices = closes_all.iloc[-1]

    # ---------- Spot prices at report generation time ----------
    pricing_as_of_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    provider_state = market.read_provider_symbol_state() or {}

    spot_prices = _fetch_spot_prices_usd(
        tickers=tickers_all,
        provider_map=provider_state,
        fallback_prices=latest_close_prices,
    )

    # Use spot for valuation; guaranteed fallback to last close where missing
    prices_for_valuation = pd.to_numeric(spot_prices, errors="coerce").replace([np.inf, -np.inf], np.nan)
    prices_for_valuation = prices_for_valuation.reindex(latest_close_prices.index).combine_first(latest_close_prices)

    # ---------- Build Position objects ----------
    positions: dict[str, Position] = {}

    # Spot positions: qty + avg_cost -> entry_price
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

    # Derivatives/notional: convert open_notional_usd to synthetic qty using valuation price (spot w/ fallback)
    for r in deriv_rows:
        t = str(r.get("ticker") or "").upper().strip()
        if not t:
            continue

        side = str(r.get("side") or "LONG").upper().strip()   # LONG/SHORT
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

    # Use only tickers that actually made it into positions and exist in closes
    tickers = sorted([t for t in positions.keys() if t in closes_all.columns])
    if not tickers:
        raise RuntimeError("No usable tickers after building positions (check missing prices or missing closes columns).")

    closes = closes_all[tickers].copy()
    rets_assets = closes.pct_change().dropna(how="any")

    # Use valuation prices for weights (report-time weights)
    values = np.array(
        [float(prices_for_valuation[t]) * float(positions[t].quantity) for t in tickers],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        bad = [t for t in tickers if not np.isfinite(float(prices_for_valuation.get(t, np.nan)))]
        raise RuntimeError(f"Non-finite valuation prices for: {bad}")
    if values.sum() == 0:
        raise ValueError("Total portfolio value == 0 from positions/prices")

    w_vec = values / values.sum()

    # Portfolio returns for HMM are history-based but weighted with current weights (ok)
    port_rets = (rets_assets[tickers] * w_vec).sum(axis=1).dropna()
    as_of_market_dt = pd.Timestamp(port_rets.index[-1]).normalize()
    as_of_market_date = as_of_market_dt.strftime("%Y-%m-%d")
    print(f"[dates] as_of_market_date={as_of_market_date} | as_of_run_date={as_of_run_date}")

    r = port_rets.to_numpy(dtype=np.float64)
    vol20 = pd.Series(r, index=port_rets.index).rolling(20).std().to_numpy(dtype=np.float64)
    mask = np.isfinite(vol20)
    X = np.column_stack([r[mask], vol20[mask]])

    hmm_res = None
    if X.shape[0] >= 80:
        hmm = GaussianHMM(n_states=4, n_dim=2, seed=42)
        fit_res = hmm.fit(X, max_iter=150, tol=1e-4, verbose=False)

        filtered = hmm.predict_proba(X)   # (T,K)
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

    # -----------------------------
    # Rebalance / rescale planning
    # -----------------------------

    # 1) load last rebalance state (json)
    raw_reb_state = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table=RESCALE_STATE_TABLE
    )
    reb_state = RebalanceState(
        last_rebalance_date=(raw_reb_state or {}).get("last_rebalance_date"),
        last_rebalance_equity=(raw_reb_state or {}).get("last_rebalance_equity"),
    )

    # 2) compute current gross + leverage_real
    positions_qty = {t: float(p.quantity) for t, p in positions.items()}

    gross_now = compute_gross_notional_from_positions(
        positions_qty=positions_qty,
        prices_usd=prices_for_valuation,
    )

    # recommended leverage comes from your HMM output:
    # IMPORTANT: leverage target comes from MARKET regime, not portfolio regime
    lev_target = float(market_lev)

    decision = should_rebalance(
        as_of=as_of_market_date,
        equity=float(equity),
        gross_notional=float(gross_now),
        recommended_leverage=float(lev_target),
        state=reb_state,
        drift_threshold=0.15,      # tune
        min_days_between=3,        # tune
        time_rule_days=30,         # tune (monthly). set None to disable
        equity_band=None,          # optional: 0.10 for +/-10%
    )

    print(
        f"[rebalance] L_real={decision.leverage_real:.2f}x "
        f"L_target={decision.leverage_target:.2f}x "
        f"drift={abs(decision.drift_ratio-1.0):.2%} "
        f"should={decision.should_rebalance} "
        f"reasons={decision.reasons}"
    )

    # ---------- Build full report (REAL SOLUTION: pass prices_usd, no patching, no overrides) ----------
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


    if decision.should_rebalance:
        plan = build_rescale_plan(
            as_of=as_of_market_date,
            equity=float(equity),
            recommended_leverage=float(lev_target),
            positions_qty=positions_qty,
            prices_usd=prices_for_valuation,
            max_notional_cap=None,  # optional global cap if you want
        )

        # Persist the plan as parquet under engine/v1/rescale/plan/dt=YYYY-MM-DD/
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

        s3_write_parquet_partition(
            s3,
            bucket=ENGINE_BUCKET,
            root_prefix=ENGINE_ROOT_PREFIX,
            table=RESCALE_PLAN_TABLE,
            dt=run_dt,
            filename="plan.parquet",
            df=plan_df,
        )

        # Update state only when a plan is produced (or when trades are actually executed — your choice)
        s3_write_json_event(
            s3,
            bucket=ENGINE_BUCKET,
            root_prefix=ENGINE_ROOT_PREFIX,
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
            update_latest=True,
        )

        print("\n" + "=" * 72)
        print("RESCALE / REBALANCE PLAN (PREVIEW)")
        print("=" * 72)
        print(f"Equity:                {plan.equity:,.2f} USD")
        print(f"Recommended leverage:  {plan.recommended_leverage:.2f}x")
        print(f"Target gross notional: {plan.target_gross_notional:,.2f} USD")
        print(f"Used gross notional:   {plan.used_gross_notional:,.2f} USD")
        print(f"Leftover notional:     {plan.leftover_notional:,.2f} USD")
        print("-" * 72)

        df = plan.targets.copy()

        view_cols = [
            "ticker",
            "price",
            "qty_current",
            "qty_target",
            "delta_qty",
            "exp_current",
            "exp_target_rounded",
            "delta_exp",
        ]

        df_view = df[view_cols].copy()

        # formatting helpers
        df_view["price"] = df_view["price"].map(lambda x: f"{x:,.2f}")
        df_view["exp_current"] = df_view["exp_current"].map(lambda x: f"{x:,.2f}")
        df_view["exp_target_rounded"] = df_view["exp_target_rounded"].map(lambda x: f"{x:,.2f}")
        df_view["delta_exp"] = df_view["delta_exp"].map(lambda x: f"{x:+,.2f}")
        df_view["delta_qty"] = df_view["delta_qty"].map(lambda x: f"{x:+.0f}")

        print("\nPosition adjustments:")
        print(df_view.to_string(index=False))
        print("-" * 72)
        # signed exposure check
        signed_sum = plan.targets["exp_target_rounded"].sum()
        gross_sum = plan.targets["exp_target_rounded"].abs().sum()

        print(f"Signed exposure sum: {signed_sum:,.2f} USD")
        print(f"Gross exposure sum:  {gross_sum:,.2f} USD")

        if abs(gross_sum - plan.used_gross_notional) > 1e-2:
            print("[WARN] Gross exposure mismatch after rounding")

        if plan.leftover_notional > 0:
            print("[INFO] Some notional could not be deployed due to rounding constraints")
    else:
        print("\n[rescale] No plan produced today.")

    # ---------- Benchmark annual return (proxy basket from OHLCV USD) ----------
    bench_ann_ret = None
    bench_meta = {"name": BENCH_NAME, "tickers": BENCH_PROXY, "method": "equal_weight_daily_rebalanced"}

    try:
        bench_closes_df = _load_closes_usd_from_ohlcv(
            tickers=BENCH_PROXY,
            start=START_HISTORY,
            end=end_date,
        )

        # Ensure we only use columns that actually exist
        cols = [c for c in BENCH_PROXY if c in bench_closes_df.columns]
        if len(cols) >= 2:
            x = bench_closes_df[cols].copy()

            # Daily returns per asset, then equal-weight across available assets each day
            r = x.pct_change().dropna(how="any")

            # Equal-weight basket return (daily rebalanced)
            bench_rets = r.mean(axis=1)

            # Annualized mean return (same style as your helper)
            bench_ann_ret = float(bench_rets.mean() * 252.0)

            bench_meta["n_assets_used"] = int(len(cols))
            bench_meta["first_date_used"] = str(pd.Timestamp(bench_rets.index.min()).date())
            bench_meta["last_date_used"] = str(pd.Timestamp(bench_rets.index.max()).date())
        else:
            bench_meta["n_assets_used"] = int(len(cols))
            bench_ann_ret = None
    except Exception:
        bench_ann_ret = None
        bench_meta = {"name": BENCH_NAME, "tickers": BENCH_PROXY, "method": "equal_weight_daily_rebalanced", "error": "failed_to_compute"}


    # ---------- Health snapshot & reopt decision ----------
    current_health = build_portfolio_health(
        report.eval,
        as_of=as_of_market_dt,   # market date
        benchmark_ann_return=bench_ann_ret,
    )

    reopt = False
    if baseline is None:
        print("\n[Portfolio health] No baseline set yet. Setting baseline to current health.")
        baseline = current_health
    else:
        reopt = should_reoptimize(baseline, current_health)
        print("\n[Portfolio health]")
        print(f"Baseline date:      {baseline.date.date()}")
        print(f"Baseline score:     {baseline.score:.4f}")
        print(f"Current score:      {current_health.score:.4f}")
        print(f"Main goal:          {current_health.main_goal:.0f}")
        print(f"Baseline P(main):   {baseline.p_hit_main_goal:.2%}")
        print(f"Current P(main):    {current_health.p_hit_main_goal:.2%}")
        print(f"Should reoptimize?  {reopt}")

    # ---------- Persist to S3 ----------
    s3_write_json_event(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
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
        update_latest=True,
    )

    s3_write_json_event(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
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
                "market_regime": {
                    "target_leverage": float(market_lev),
                    "source_table": "regimes/market_hmm",
                },
            },
            "flags": {
                "should_reoptimize": bool(reopt),
                "baseline_exists": bool(baseline is not None),
            },
        },
        update_latest=True,
    )

    holdings_df = pd.DataFrame(report.snapshot.positions_table)
    s3_write_parquet_partition(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
        table="holdings",
        dt=run_dt,
        filename="holdings.parquet",
        df=holdings_df,
    )

    s3_write_json_event(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
        table="health",
        dt=run_dt,
        filename="health.json",
        payload=asdict(current_health),
        update_latest=True,
    )

    s3_write_json_event(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
        table="configs/score_config",
        dt=run_dt,
        filename="score_config.json",
        payload=asdict(score_cfg),
        update_latest=True,
    )

    s3_write_json_event(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
        table="inputs/positions",
        dt=run_dt,
        filename="positions.json",
        payload={t: asdict(p) for t, p in positions.items()},
        update_latest=True,
    )

    print("\n[S3] Saved daily report + holdings + health + score_config + positions + regimes.")


if __name__ == "__main__":
    main()
