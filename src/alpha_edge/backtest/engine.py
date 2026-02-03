# backtest/engine.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from alpha_edge.portfolio.report_engine import build_portfolio_report
from alpha_edge.core.market_store import MarketStore
from backtest.config import BacktestConfig
from backtest.goals import GoalPath, goals3_from_goalpath
from backtest.sampling import sample_as_of_date_from_returns_cache as sample_as_of_date
from backtest.benchmark import compute_proxy_daily_returns, annualized_return_from_daily_returns
from backtest.data_view import slice_returns_wide_asof, get_close_prices_asof
from backtest.executor import Trade, target_qty_from_weights, rebalance_to_target, mark_to_market, apply_trades
from alpha_edge.portfolio.rebalance_engine import (
    RebalanceState, should_rebalance, compute_gross_notional_from_positions
)

# your existing health funcs
from alpha_edge.portfolio.portfolio_health import build_portfolio_health, should_reoptimize

# you will plug your portfolio search here
# (we keep it as a callable injected for minimal coupling)
PortfolioSearchFn = Any  # callable signature described below

@dataclass
class BacktestResult:
    as_of_start: str
    as_of_end: str
    n_steps: int
    equity_path: list[dict]
    trades: list[dict]
    events: list[dict]
    meta: dict

def build_eval_dates(index: pd.DatetimeIndex, *, start: str, end: str, freq: str) -> list[str]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce")).tz_localize(None).normalize()
    idx = idx[(idx >= start_ts) & (idx <= end_ts)]
    if idx.empty:
        return []
    # resample to calendar frequency, then snap to last available trading day <= that timestamp
    s = pd.Series(index=idx, data=np.ones(len(idx)))
    anchors = s.resample(freq).last().dropna().index
    # snap each anchor to last trading day <= anchor
    out = []
    for a in anchors:
        w = idx[idx <= a]
        if not w.empty:
            out.append(pd.Timestamp(w[-1]).strftime("%Y-%m-%d"))
    # ensure start included
    if out and out[0] != start_ts.strftime("%Y-%m-%d"):
        out = [start_ts.strftime("%Y-%m-%d")] + out
    if not out:
        out = [start_ts.strftime("%Y-%m-%d")]
    # de-dupe
    out = list(dict.fromkeys(out))
    return out

def run_backtest_once(
    *,
    cfg: BacktestConfig,
    universe_csv: str,
    portfolio_search_fn: PortfolioSearchFn,
    # window control
    as_of_start: str | None = None,
    horizon_days: int = 252 * 2,
) -> BacktestResult:
    """
    MVP:
      - sample as_of_start if not provided
      - search portfolio at as_of_start using returns history up to as_of_start
      - enter portfolio at close(as_of_start)
      - monthly evaluate; rebalance to same weights when leverage target changes materially
      - reoptimize (search new portfolio) if should_reoptimize triggers

    portfolio_search_fn contract (expected):
      portfolio_search_fn(returns_wide, as_of, equity0, notional, goals, main_goal, score_cfg, universe_csv) -> dict
        returning at least: {"weights": {ticker: w}, "eval_metrics": EvalMetrics}
    """
    store = MarketStore(bucket=cfg.bucket, region=cfg.region)

    # load returns_wide once
    wide_all = pd.read_parquet(cfg.returns_wide_cache_path, engine="pyarrow").sort_index()
    wide_all.index = pd.to_datetime(wide_all.index, errors="coerce").tz_localize(None).normalize()

    if as_of_start is None:
        as_of_start = sample_as_of_date(
            returns_wide_path=cfg.returns_wide_cache_path,
            warmup_days=cfg.warmup_days,
            min_forward_days=cfg.min_forward_days,
            seed=cfg.seed,
        )

    start_ts = pd.Timestamp(as_of_start).normalize()
    end_ts = (start_ts + pd.Timedelta(days=int(horizon_days) * 2)).normalize()  # *2 to survive non-trading days
    # cap at last date available
    last_avail = pd.Timestamp(wide_all.index.max()).normalize()
    end_ts = min(end_ts, last_avail)
    as_of_end = end_ts.strftime("%Y-%m-%d")

    # build eval dates from trading index
    eval_dates = build_eval_dates(wide_all.index, start=as_of_start, end=as_of_end, freq=cfg.eval_freq)
    if len(eval_dates) < 2:
        raise RuntimeError("Not enough evaluation dates in window.")

    # universe mapping (asset_id <-> ticker)
    u = pd.read_csv(universe_csv)
    u = u[u.get("include", 1).fillna(1).astype(int) == 1].copy()
    u["asset_id"] = u["asset_id"].astype(str).str.strip()
    u["ticker"] = u.get("ticker", u["asset_id"]).astype(str).str.upper().str.strip()
    ticker_to_asset = dict(zip(u["ticker"], u["asset_id"]))
    asset_to_ticker = dict(zip(u["asset_id"], u["ticker"]))

    # Goal path
    gp = GoalPath(list(cfg.goals), confirm_steps=cfg.goal_confirm_steps, buffer=cfg.goal_buffer)

    # state
    positions_qty: Dict[str, float] = {}
    equity = float(cfg.initial_equity)
    baseline_health = None
    current_weights: Dict[str, float] = {}

    trades: List[Trade] = []
    events: list[dict] = []
    equity_path: list[dict] = []

    # Precompute benchmark proxy returns series (as-of slices later)
    proxy_ret_all = compute_proxy_daily_returns(wide_all, list(cfg.proxy_tickers))

    def get_prices_for_tickers(as_of: str, tickers: list[str]) -> Dict[str, float]:
        aids = [ticker_to_asset[t] for t in tickers if t in ticker_to_asset]
        px_by_aid = get_close_prices_asof(store=store, asset_ids=aids, as_of=as_of, lookback_days=10)
        return {t: float(px_by_aid.get(ticker_to_asset[t], np.nan)) for t in tickers if t in ticker_to_asset}

    # --- iterate evaluation dates ---
    for step_i, as_of in enumerate(eval_dates):
        # slice returns up to this as_of for search/eval
        rets_slice = slice_returns_wide_asof(wide_all, end_date=as_of, lookback_days=cfg.warmup_days)

        # compute benchmark ann return from proxy over same slice
        proxy_slice = proxy_ret_all.loc[proxy_ret_all.index <= pd.Timestamp(as_of).normalize()]
        proxy_slice = proxy_slice.iloc[-min(len(proxy_slice), cfg.warmup_days):]
        bench_ann = annualized_return_from_daily_returns(proxy_slice)

        # choose target leverage (MVP: use constant 1.0; next step: plug your market_hmm as-of)
        # For now we keep it deterministic and simple:
        target_leverage = 1.0
        target_notional = float(equity) * float(target_leverage)

        # Portfolio search at step 0 OR after reoptimize trigger
        do_search = (step_i == 0) or any(e.get("type") == "REOPTIMIZE" and e.get("as_of") == as_of for e in events)
        if do_search:

            goals3 = goals3_from_goalpath(gp)

            search_out = portfolio_search_fn(
                returns_wide=rets_slice,
                as_of=as_of,
                equity0=float(equity),
                notional=float(target_notional),
                goals=goals3,                 # <-- IMPORTANT: 3 goals only
                main_goal=float(goals3[0]),   # <-- matches your EvalMetrics semantics
                universe_csv=universe_csv,
            )

            current_weights = dict(search_out["weights"])
            # optional eval_metrics if you return it
            eval_metrics = search_out.get("eval_metrics")

            events.append({
                "type": "PORTFOLIO_SELECTED",
                "as_of": as_of,
                "main_goal": float(gp.main_goal),
                "n_assets": len(current_weights),
            })

            # build entry trades to reach target weights
            tickers = sorted(current_weights.keys())
            prices = get_prices_for_tickers(as_of, tickers)
            target_qty = target_qty_from_weights(weights=current_weights, prices=prices, target_notional=target_notional)
            tds = rebalance_to_target(date=as_of, positions_qty=positions_qty, target_qty=target_qty, prices=prices)
            trades.extend(tds)

            positions_qty = apply_trades(positions_qty, tds)

            # set baseline health at entry if available
            if eval_metrics is not None:
                try:
                    baseline_health = build_portfolio_health(eval_metrics, as_of=pd.Timestamp(as_of), benchmark_ann_return=bench_ann)
                except Exception:
                    baseline_health = baseline_health

        # mark-to-market equity at close(as_of)
        tickers_now = sorted(positions_qty.keys())
        prices_now = get_prices_for_tickers(as_of, tickers_now) if tickers_now else {}
        port_value = mark_to_market(positions_qty, prices_now)

        # NOTE: because you’re leveraged, “equity” is not identical to gross value.
        # MVP simplification: treat equity as marked-to-market of positions (you can upgrade with margin model later).
        equity = float(port_value) if tickers_now else float(equity)
        
        reb_state = RebalanceState(
                last_rebalance_date=None,
                last_rebalance_equity=None,
            )
        goal_evt = gp.update(equity)
        if goal_evt.get("advanced"):
            events.append({"type": "GOAL_ADVANCED", "as_of": as_of, **goal_evt})

        # Evaluate portfolio health if eval_metrics is available from a daily report-style evaluator.
        # MVP: re-use eval_metrics only at selection time. Next: call your report evaluator per step.
        # We'll store alpha proxy anyway.
        equity_path.append({
            "as_of": as_of,
            "equity": float(equity),
            "target_leverage": float(target_leverage),
            "target_notional": float(target_notional),
            "bench_ann_return": None if bench_ann is None else float(bench_ann),
        })

        # Reoptimize trigger (only if we have both baseline and a "current" health snapshot)
        # MVP: we can only do this if portfolio_search_fn returns eval_metrics at each step (or we add an evaluator)
        # so we *wire the hook* but keep it dormant unless current metrics exist.

        report = build_portfolio_report(
            returns_wide=rets_slice,
            weights=current_weights,
            equity0=float(equity),
            notional=float(target_notional),
            goals=goals3_from_goalpath(gp),
            main_goal=float(gp.main_goal),
        )
        current_health = build_portfolio_health(
            report.eval_metrics,
            as_of=pd.Timestamp(as_of),
            benchmark_ann_return=report.benchmark_ann_return,
        )
        if baseline_health is not None and current_health is not None:
            if should_reoptimize(
                baseline_health,
                current_health,
                max_score_drop=cfg.max_score_drop,
                max_p_main_drop=cfg.max_p_main_drop,
                min_alpha=cfg.min_alpha,
            ):
                events.append({"type": "REOPTIMIZE", "as_of": as_of, "why": "health_trigger"})
                # next loop iteration will search because we add REOPTIMIZE event for that as_of
                # (or you can trigger immediate reselection here)

    gross_now = compute_gross_notional_from_positions(
        positions_qty=positions_qty,
        prices_usd=pd.Series(prices_now),
    )

    decision = should_rebalance(
        as_of=as_of,
        equity=float(equity),
        gross_notional=float(gross_now),
        recommended_leverage=float(target_leverage),
        state=reb_state,
        drift_threshold=cfg.rebalance_drift_threshold,
        min_days_between=cfg.rebalance_min_days_between,
        time_rule_days=cfg.rebalance_time_rule_days,
        equity_band=cfg.rebalance_equity_band,
    )

    if decision.should_rebalance and current_weights:
        tickers = sorted(current_weights.keys())
        prices = get_prices_for_tickers(as_of, tickers)
        target_qty = target_qty_from_weights(
            weights=current_weights,
            prices=prices,
            target_notional=float(equity) * float(target_leverage),
        )
        tds = rebalance_to_target(date=as_of, positions_qty=positions_qty, target_qty=target_qty, prices=prices)
        trades.extend(tds)
        positions_qty = apply_trades(positions_qty, tds)

        # update rebalance state
        reb_state = RebalanceState(last_rebalance_date=as_of, last_rebalance_equity=float(equity))
        events.append({"type": "REBALANCE", "as_of": as_of, "reasons": decision.reasons})

    return BacktestResult(
        as_of_start=as_of_start,
        as_of_end=as_of_end,
        n_steps=len(eval_dates),
        equity_path=equity_path,
        trades=[asdict(t) for t in trades],
        events=events,
        meta={
            "proxy_basket": list(cfg.proxy_tickers),
            "goals": list(cfg.goals),
            "eval_freq": cfg.eval_freq,
            "note": "MVP backtest; leverage fixed to 1.0; health trigger wired but requires per-step eval_metrics.",
        },
    )
