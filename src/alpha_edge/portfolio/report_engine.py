# report_engine.py
from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd

import json

from alpha_edge.core.schemas import Position, PortfolioSnapshot, PortfolioReport
from alpha_edge.portfolio.optimizer_engine import evaluate_portfolio  # canonical evaluator
from alpha_edge.market.stats_engine import compute_daily_returns   # to build asset returns
from alpha_edge.portfolio.alpha_report import format_alpha_report

# ---------- Core helpers ----------

def extract_p_hit_map(metrics, goals: list[float]) -> dict[float, float]:
    g1, g2, g3 = [float(g) for g in goals]
    return {
        g1: float(metrics.p_hit_goal_1_1y),
        g2: float(metrics.p_hit_goal_2_1y),
        g3: float(metrics.p_hit_goal_3_1y),
    }


def extract_p_hit_main_goal(metrics, goals: list[float], main_goal: float) -> float:
    p_hit = extract_p_hit_map(metrics, goals)
    return float(p_hit.get(float(main_goal), 0.0))


def compute_portfolio_timeseries(
    closes: pd.DataFrame,
    positions: Dict[str, float],
) -> pd.Series:
    """
    closes: DataFrame indexed by date, columns=tickers, values=adj_close
    positions: ticker -> quantity (SIGNED)
    NOTE: this produces SIGNED portfolio value series, which is not what we want
          for gross-notional leverage (use gross exposures instead).
    Kept for convenience, not used for notional in long/short mode.
    """
    cols = [t for t in positions.keys() if t in closes.columns]
    values = closes[cols].mul(pd.Series(positions), axis=1)
    port_value = values.sum(axis=1)
    return port_value


def compute_unlevered_stats(port_series: pd.Series) -> Dict[str, float]:
    rets = port_series.pct_change().dropna()
    if len(rets) < 10:
        raise ValueError("Not enough data for stats")

    daily_mean = rets.mean()
    daily_std = rets.std()
    downside = rets[rets < 0]

    ann_return = daily_mean * 252
    ann_vol = daily_std * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    downside_std = downside.std()
    sortino = (ann_return / (downside_std * np.sqrt(252))
               if downside_std > 0 else np.nan)

    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    max_dd = dd.min()

    var_95 = np.percentile(rets, 5)
    cvar_95 = rets[rets <= var_95].mean()

    return dict(
        ann_return=float(ann_return),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        var_95=float(var_95),
        cvar_95=float(cvar_95),
        rets=rets,
    )


def build_portfolio_report(
    closes: pd.DataFrame,
    positions: Dict[str, Position],
    equity: float,
    goals: list[float],
    main_goal: float,
    *,
    lw_cov: pd.DataFrame | None = None,
    days: int = 252,
    n_paths: int = 20000,
    score_config=None,
    mc_seed: int | None = None,
    block_size: int | tuple[int, int] | None = (8, 12),
    path_source: str = "bootstrap",
    pca_k: int | None = 5,
    prices_usd: pd.Series | None = None,  # valuation prices at report time
) -> PortfolioReport:
    """
    Long/short aware report:

    - Historical risk/returns are estimated from `closes` (daily history).
    - Valuation uses `prices_usd` if provided (spot at report time), else last close.
    - Notional is GROSS: sum(abs(exposure)).
    - Weights passed to evaluator are SIGNED weights scaled by gross notional:
        w_i = exposure_i / gross_notional
      These weights usually do NOT sum to 1 (can sum ~0). The evaluator must support that.
    """
    if closes is None or closes.empty:
        raise ValueError("closes is empty")

    # --- last close for fallback valuation ---
    last_close = closes.iloc[-1].copy()
    last_close = pd.to_numeric(last_close, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # --- valuation prices ---
    if prices_usd is None:
        latest_prices = last_close
    else:
        p = pd.to_numeric(prices_usd, errors="coerce").replace([np.inf, -np.inf], np.nan)
        p = p.reindex(last_close.index)
        latest_prices = p.combine_first(last_close)

    # --- exposures + gross notional ---
    exposures: dict[str, float] = {}
    gross_notional = 0.0

    for ticker, pos in positions.items():
        if ticker not in latest_prices.index:
            continue
        px = float(latest_prices.get(ticker, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue

        exp = float(px) * float(pos.quantity)  # signed exposure
        exposures[ticker] = exp
        gross_notional += abs(exp)

    gross_notional = float(gross_notional)
    if gross_notional <= 0:
        raise ValueError("Gross notional <= 0 from positions/prices")

    leverage = float(gross_notional / float(equity)) if float(equity) > 0 else float("inf")

    # --- positions table + signed weights ---
    pos_rows: List[Dict[str, Any]] = []
    weights: Dict[str, float] = {}

    for ticker, pos in positions.items():
        if ticker not in exposures:
            continue

        px = float(latest_prices.get(ticker, np.nan))
        exp = float(exposures[ticker])

        w_signed = exp / gross_notional
        w_abs = abs(exp) / gross_notional

        pos_rows.append(
            dict(
                ticker=ticker,
                quantity=float(pos.quantity),
                price=float(px),
                value=float(exp),          # signed
                weight=float(w_signed),    # signed
                weight_abs=float(w_abs),   # gross contribution
                currency=pos.currency,
            )
        )
        weights[ticker] = float(w_signed)

    snapshot = PortfolioSnapshot(
        as_of=pd.Timestamp(closes.index[-1]),
        total_notional=float(gross_notional),
        equity=float(equity),
        leverage=float(leverage),
        positions_table=pos_rows,
    )

    # --- build returns matrix for held tickers (history-based) ---
    tickers = [t for t in weights.keys() if t in closes.columns]
    if not tickers:
        raise ValueError("No tickers overlap between positions and closes")

    closes_sub = closes[tickers].dropna(how="any")
    asset_returns = compute_daily_returns(closes_sub)

    # --- canonical evaluation ---
    eval_metrics = evaluate_portfolio(
        returns=asset_returns,
        weights=weights,                    # SIGNED weights (gross-scaled)
        equity0=float(equity),
        notional=float(gross_notional),     # GROSS notional
        goals=goals,
        main_goal=float(main_goal),
        lw_cov=lw_cov,
        days=days,
        n_paths=n_paths,
        score_config=score_config,
        mc_seed=mc_seed,
        path_source=path_source,
        pca_k=pca_k,
        block_size=block_size,
        weight_mode="gross_signed",         # NEW: tells evaluator how to interpret weights
    )

    return PortfolioReport(snapshot=snapshot, eval=eval_metrics)


def summarize_report(report: PortfolioReport) -> str:
    s = report.snapshot
    m = report.eval
    g1, g2, g3 = m.goals

    return f"""
Daily Portfolio Summary
-----------------------

As of: {s.as_of.date()}
Total Notional: {s.total_notional:,.2f} USD
Equity: {s.equity:,.2f} USD
Leverage: {s.leverage:.2f}x

Unlevered Stats:
- Annual Return: {m.ann_return:.2%}
- Annual Vol: {m.ann_vol:.2%}
- Sharpe: {m.sharpe:.2f}
- Sortino: {m.sortino:.2f}
- Max Drawdown: {m.max_drawdown:.2%}

Risk Metrics:
- 1-day VaR(95): {m.var_95:.2%}
- 1-day CVaR(95): {m.cvar_95:.2%}

Leveraged Monte Carlo (1 year):
- Ruin Probability: {m.ruin_prob_1y:.2%}
- P(>= {g1:.0f}): {m.p_hit_goal_1_1y:.2%} | Median time: {m.med_t_goal_1_days or 'N/A'} days
- P(>= {g2:.0f}): {m.p_hit_goal_2_1y:.2%} | Median time: {m.med_t_goal_2_days or 'N/A'} days
- P(>= {g3:.0f}): {m.p_hit_goal_3_1y:.2%} | Median time: {m.med_t_goal_3_days or 'N/A'} days

Ending Equity Percentiles:
- P5: {m.ending_equity_p5:,.2f} USD
- P25: {m.ending_equity_p25:,.2f} USD
- P50: {m.ending_equity_p50:,.2f} USD
- P75: {m.ending_equity_p75:,.2f} USD
- P95: {m.ending_equity_p95:,.2f} USD

Score: {m.score:.4f}
"""

def print_hmm_summary(hmm_res: dict, lev_rec: dict | None = None) -> None:

    if hmm_res is None:
        print("\n[HMM] No regime data available.")
        return

    # ---------- leverage mapping + chooser (local) ----------
    REGIME_TO_BAND = {
        "STRESS_BEAR": (1.0, 3.0),
        "CHOPPY_BEAR": (3.0, 5.0),
        "CHOPPY_BULL": (5.0, 7.0),
        "CALM_BULL":   (7.0, 12.0),
    }

    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _leverage_recommendation_from_hmm(
        hmm_payload: dict,
        *,
        default: float = 1.0,
        risk_appetite: float = 0.8,         # 0..1
        low_conf_floor: float = 0.20,       # below this -> near min
        hard_cap: float | None = 12.0,
    ) -> dict:
        p_label = hmm_payload.get("p_label_today") or {}
        if not isinstance(p_label, dict) or not p_label:
            return {"leverage": default, "mode": "none", "label": None, "conf": 0.0, "band": None}

        # normalize probs
        items = [(str(k), float(v)) for k, v in p_label.items() if v is not None]
        s = sum(v for _, v in items)
        if s <= 0:
            return {"leverage": default, "mode": "none", "label": None, "conf": 0.0, "band": None}
        items = [(k, v / s) for k, v in items]
        p_norm = dict(items)

        commit = hmm_payload.get("label_commit")
        if isinstance(commit, str) and commit in REGIME_TO_BAND:
            conf = float(p_norm.get(commit, 0.0))
            conf_scaled = _clamp((conf - low_conf_floor) / (1.0 - low_conf_floor), 0.0, 1.0)
            ra = _clamp(risk_appetite * conf_scaled, 0.0, 1.0)
            lo, hi = REGIME_TO_BAND[commit]
            lev = lo + ra * (hi - lo)
            if hard_cap is not None:
                lev = min(float(lev), float(hard_cap))
            return {"leverage": float(lev), "mode": "commit", "label": commit, "conf": conf, "band": (lo, hi)}

        # expected leverage band if not committed
        exp_lo = 0.0
        exp_hi = 0.0
        for lab, prob in items:
            lo, hi = REGIME_TO_BAND.get(lab, (default, default))
            exp_lo += prob * lo
            exp_hi += prob * hi

        conf = float(max(prob for _, prob in items))
        conf_scaled = _clamp((conf - low_conf_floor) / (1.0 - low_conf_floor), 0.0, 1.0)
        ra = _clamp(risk_appetite * conf_scaled, 0.0, 1.0)
        lev = exp_lo + ra * (exp_hi - exp_lo)
        if hard_cap is not None:
            lev = min(float(lev), float(hard_cap))

        top_label = max(items, key=lambda kv: kv[1])[0]
        return {"leverage": float(lev), "mode": "expected", "label": top_label, "conf": conf, "band": (exp_lo, exp_hi)}

    if lev_rec is None:
        lev_rec = _leverage_recommendation_from_hmm(hmm_res)

    # ---------- your existing printing ----------
    p = hmm_res["p_label_today"]
    commit = hmm_res["label_commit"]
    diags = hmm_res["state_diagnostics"]
    mapping = hmm_res["state_to_label"]

    print("\n" + "─" * 44)
    print("Market Regime (HMM · 4 states)")
    print("─" * 44)

    print(f"As of: {hmm_res['meta']['last_date_used']}\n")

    if commit:
        print(f"▶ Active regime:  {commit}  (confidence: {p[commit]*100:.2f}%)")
    else:
        print("▶ Active regime:  MIXED / NEUTRAL")

    # --- leverage line (ONLY HERE) ---
    band = lev_rec.get("band")
    band_str = ""
    if isinstance(band, (tuple, list)) and len(band) == 2:
        band_str = f"{float(band[0]):.1f}x–{float(band[1]):.1f}x"
    print(
        f"▶ Recommended leverage: {float(lev_rec['leverage']):.2f}x"
        f"{f'  (band: {band_str})' if band_str else ''}"
        f"  [{lev_rec.get('mode','none')}, conf={float(lev_rec.get('conf',0.0))*100:.2f}%]\n"
    )

    print("Regime probabilities:")
    for lab, prob in sorted(p.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 24)
        print(f"  {lab:12s} {bar:<24s} {prob*100:6.2f}%")

    print("\nState diagnostics:")
    print("┌──────────────┬──────────┬──────────┬────────────┬──────────┐")
    print("│ Regime       │ Drift    │ Vol      │ Neg days % │ Weight   │")
    print("├──────────────┼──────────┼──────────┼────────────┼──────────┤")

    for k, lab in mapping.items():
        d = diags[k]
        print(
            f"│ {lab:12s} │ "
            f"{d['drift']*100:+6.3f}% │ "
            f"{d['vol']*100:6.2f}% │ "
            f"{d['neg_rate']*100:8.2f}% │ "
            f"{d['weight']*100:6.2f}% │"
        )

    print("└──────────────┴──────────┴──────────┴────────────┴──────────┘")


def print_decision_addendum(
    *,
    decision,
    health,
    bench_ann_ret: float | None,
    reopt: bool,
    plan=None,                 # RescalePlan | None
    top_n: int = 8,
    show_full_alpha_blob: bool = False,
    take_profit: dict | None = None,
) -> None:
    # decision: RebalanceDecision
    # health: PortfolioHealth
    # plan: RescalePlan (optional)

    # format_alpha_report must be in scope (import where you call this)
    # from alpha_edge.portfolio.portfolio_health import format_alpha_report

    def pct(x: float | None) -> str:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{100.0 * float(x):+.2f}%"

    def num(x: float | None, nd: int = 4) -> str:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{float(x):.{nd}f}"

    def money(x: float | None) -> str:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "n/a"
        return f"{float(x):,.2f} USD"

    print("\n" + "─" * 44)
    print("Decision Addendum")
    print("─" * 44)

    # --- REBALANCE ---
    should_rb = bool(getattr(decision, "should_rebalance", False))
    print("\nRebalance")
    print(f"▶ should_rebalance: {should_rb}")

    drift_ratio = getattr(decision, "drift_ratio", None)
    drift = None
    try:
        drift = abs(float(drift_ratio) - 1.0) if drift_ratio is not None else None
    except Exception:
        drift = None

    print(
        f"  L_real={num(getattr(decision, 'leverage_real', None), 2)}x  "
        f"L_target={num(getattr(decision, 'leverage_target', None), 2)}x  "
        f"drift={pct(drift)}"
    )
    reasons = getattr(decision, "reasons", None)
    if reasons:
        print(f"  reasons: {', '.join(reasons)}")
        
    # --- TAKE PROFIT ---
    if take_profit is not None:
        print("\nTake Profit")
        do_tp = bool(take_profit.get("do_harvest", False))
        m_star = take_profit.get("m_star", None)
        r_anchor = take_profit.get("r_anchor", None)
        dd = take_profit.get("dd", None)
        sharpe_tp = take_profit.get("sharpe", None)
        cooldown = take_profit.get("cooldown_days", None)

        print(f"▶ do_harvest: {do_tp}")
        print(
            f"  m_star={num(m_star, 3)}  "
            f"r_anchor={pct(r_anchor)}  "
            f"dd={pct(dd)}  "
            f"sharpe={num(sharpe_tp, 2)}"
        )
        if cooldown is not None:
            print(f"  cooldown_days={cooldown}")

        tp_reasons = take_profit.get("reasons", None)
        if tp_reasons:
            print(f"  reasons: {', '.join(tp_reasons)}")

    # --- HEALTH ---
    print("\nHealth")
    print(f"▶ should_reoptimize: {bool(reopt)}")
    print(
        f"  score={num(getattr(health, 'score', None), 4)}  "
        f"sharpe={num(getattr(health, 'sharpe', None), 2)}"
    )
    print(
        f"  ann_return={pct(getattr(health, 'ann_return', None))}  "
        f"ruin_prob_1y={pct(getattr(health, 'ruin_prob', None))}"
    )

    # --- BENCH + ALPHA (CAPM) ---
    print("\nBenchmark & Alpha (CAPM)")
    print(f"  bench_ann_return={pct(bench_ann_ret)}")

    # Diagnostics (active return) — do NOT call it alpha
    ex1y = getattr(health, "excess_return_1y", None)
    if ex1y is None:
        # fallback to legacy field if you still store it there
        ex1y = getattr(health, "alpha_vs_bench", None)
    print(f"  excess_return_1y={pct(ex1y)}")

    # CAPM alpha is the only “alpha”
    print(
        f"  capm_alpha_1y={pct(getattr(health, 'alpha_1y', None))}  "
        f"beta_1y={num(getattr(health, 'beta_1y', None), 3)}  "
        f"r2_1y={num(getattr(health, 'r2_1y', None), 3)}"
    )
    print(
        f"  IR_1y={num(getattr(health, 'info_ratio_1y', None), 3)}  "
        f"TE_1y={pct(getattr(health, 'tracking_error_1y', None))}"
    )

    # Optional: 3m window if your health has it
    if getattr(health, "alpha_3m", None) is not None or getattr(health, "excess_return_3m", None) is not None:
        print(
            f"  capm_alpha_3m={pct(getattr(health, 'alpha_3m', None))}  "
            f"beta_3m={num(getattr(health, 'beta_3m', None), 3)}  "
            f"r2_3m={num(getattr(health, 'r2_3m', None), 3)}"
        )
        print(
            f"  IR_3m={num(getattr(health, 'info_ratio_3m', None), 3)}  "
            f"TE_3m={pct(getattr(health, 'tracking_error_3m', None))}  "
            f"excess_return_3m={pct(getattr(health, 'excess_return_3m', None))}"
        )

    # Pretty alpha lines (short) - recommended default
    if getattr(health, "alpha_report_json", None):
        try:
            ar = json.loads(health.alpha_report_json)
            print("\n" + format_alpha_report(ar))
        except Exception:
            print("\n[alpha][warn] could not parse alpha_report_json")

    # --- RESCALE / REBALANCE PLAN (PREVIEW) ---
    # Only show if we actually decided to rebalance and we have a plan
    if should_rb and plan is not None:
        print("\n" + "=" * 72)
        print("RESCALE / REBALANCE PLAN (PREVIEW)")
        print("=" * 72)
        print(f"Equity:                {money(getattr(plan, 'equity', None))}")
        print(f"Recommended leverage:  {num(getattr(plan, 'recommended_leverage', None), 2)}x")
        print(f"Current leverage:      {num(getattr(plan, 'leverage_current', None), 2)}x")
        print(f"Target gross notional: {money(getattr(plan, 'target_gross_notional', None))}")
        print(f"Used gross notional:   {money(getattr(plan, 'used_gross_notional', None))}")
        print(f"Leftover notional:     {money(getattr(plan, 'leftover_notional', None))}")
        print("-" * 72)

        try:
            df = getattr(plan, "targets", None)
            if df is None or len(df) == 0:
                print("(plan.targets empty)")
            else:
                d = df.copy()

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
                # keep only columns that exist (defensive)
                view_cols = [c for c in view_cols if c in d.columns]
                df_view = d[view_cols].copy()

                # coerce numeric safely
                for c in ["price", "qty_current", "qty_target", "delta_qty", "exp_current", "exp_target_rounded", "delta_exp"]:
                    if c in df_view.columns:
                        df_view[c] = pd.to_numeric(df_view[c], errors="coerce")

                # formatting
                if "price" in df_view.columns:
                    df_view["price"] = df_view["price"].map(lambda x: "n/a" if not np.isfinite(x) else f"{x:,.2f}")
                for c in ["exp_current", "exp_target_rounded"]:
                    if c in df_view.columns:
                        df_view[c] = df_view[c].map(lambda x: "n/a" if not np.isfinite(x) else f"{x:,.2f}")
                if "delta_exp" in df_view.columns:
                    df_view["delta_exp"] = df_view["delta_exp"].map(lambda x: "n/a" if not np.isfinite(x) else f"{x:+,.2f}")
                if "delta_qty" in df_view.columns:
                    # if you want crypto decimals preserved, change +.0f -> +.8f when is_crypto
                    df_view["delta_qty"] = df_view["delta_qty"].map(lambda x: "n/a" if not np.isfinite(x) else f"{x:+.0f}")

                print("\nPosition adjustments:")
                print(df_view.to_string(index=False))
                print("-" * 72)

                # signed/gross checks (same as old version)
                if "exp_target_rounded" in d.columns:
                    exp_tgt = pd.to_numeric(d["exp_target_rounded"], errors="coerce")
                    signed_sum = float(exp_tgt.sum(skipna=True))
                    gross_sum = float(exp_tgt.abs().sum(skipna=True))

                    print(f"Signed exposure sum: {signed_sum:,.2f} USD")
                    print(f"Gross exposure sum:  {gross_sum:,.2f} USD")

                    used = getattr(plan, "used_gross_notional", None)
                    if used is not None and np.isfinite(float(used)):
                        if abs(gross_sum - float(used)) > 1e-2:
                            print("[WARN] Gross exposure mismatch after rounding")

                leftover = getattr(plan, "leftover_notional", None)
                if leftover is not None and np.isfinite(float(leftover)) and float(leftover) > 0:
                    print("[INFO] Some notional could not be deployed due to rounding constraints")

        except Exception as e:
            print(f"(plan preview failed: {type(e).__name__})")

        print("\n" + "=" * 72)
