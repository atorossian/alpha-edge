# report_engine.py
from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd

from alpha_edge.core.schemas import Position, PortfolioSnapshot, PortfolioReport
from alpha_edge.portfolio.optimizer_engine import evaluate_portfolio  # canonical evaluator
from alpha_edge.market.stats_engine import compute_daily_returns   # to build asset returns


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
