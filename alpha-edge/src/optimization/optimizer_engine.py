# optimizer_engine.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Sequence

import numpy as np
import pandas as pd

from stats_engine import compute_asset_stats
from report_engine import simulate_leveraged_paths  # reuse existing MC


@dataclass
class PortfolioCandidateMetrics:
    weights: Dict[str, float]

    # unlevered stats (sample-based)
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float

    # LW-based risk
    ann_vol_lw: float

    # VaR / CVaR (1d, 95%)
    var_95: float
    cvar_95: float

    # leveraged MC (1y)
    ruin_prob_1y: float
    p_hit_600_1y: float
    p_hit_800_1y: float
    p_hit_2000_1y: float
    med_t_800_days: float | None
    med_t_2000_days: float | None

    # ending equity distribution
    ending_equity_p5: float
    ending_equity_p25: float
    ending_equity_p50: float
    ending_equity_p75: float
    ending_equity_p95: float

    # scalar score
    score: float

def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    returns: DataFrame indexed by date, columns=tickers, values=daily returns.
    weights: dict[ticker -> weight], should roughly sum to 1.
    """
    # Filter to tickers we actually have in the returns matrix
    tickers = [t for t in weights.keys() if t in returns.columns]
    if not tickers:
        raise ValueError("No overlap between weights and returns columns")

    w = np.array([weights[t] for t in tickers], dtype=float)
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("Weights must sum to a positive number")
    w = w / w_sum  # normalize strictly to 1

    sub_rets = returns[tickers]
    port_rets = (sub_rets * w).sum(axis=1)
    return port_rets

def evaluate_portfolio_candidate(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    equity0: float,
    notional: float,
    lw_cov: pd.DataFrame | None = None,
    days: int = 252,
    n_paths: int = 20000,
    lambda_ruin: float = 0.5,
    gamma_vol: float = 0.0,
) -> PortfolioCandidateMetrics:
    """
    returns: daily returns matrix (Date x tickers)
    weights: candidate portfolio weights (ticker -> weight)
    equity0: starting equity
    notional: fixed notional exposure
    lw_cov: Ledoit–Wolf covariance matrix (optional but recommended)
    lambda_ruin: penalty on ruin prob in the score
    gamma_vol: penalty on LW vol in the score
    """

    # 1) Portfolio return series
    port_rets = compute_portfolio_returns(returns, weights).dropna()
    if len(port_rets) < 50:
        raise ValueError("Not enough data for this portfolio candidate")

    # 2) Unlevered stats (sample-based)
    daily_mean = port_rets.mean()
    daily_std = port_rets.std()
    ann_return = float(daily_mean * 252)
    ann_vol = float(daily_std * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + port_rets).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    max_dd = float(dd.min())

    # 2b) VaR / CVaR (1d, 95%)
    var_95 = float(np.percentile(port_rets, 5))
    cvar_95 = float(port_rets[port_rets <= var_95].mean())

    # 3) LW-based portfolio vol (if covariance provided)
    ann_vol_lw = float("nan")
    if lw_cov is not None:
        tickers = [t for t in weights.keys() if t in lw_cov.columns]
        w = np.array([weights[t] for t in tickers], dtype=float)
        w = w / w.sum()
        cov_sub = lw_cov.loc[tickers, tickers].values   # daily cov
        port_var_daily = float(w @ cov_sub @ w)         # w' Σ w
        ann_vol_lw = float(np.sqrt(port_var_daily * 252))

    # 4) Leveraged Monte Carlo (using your engine)
    mc = simulate_leveraged_paths(
        rets=port_rets,
        notional=notional,
        equity0=equity0,
        days=days,
        n_paths=n_paths,
    )

    ruin_prob = mc["ruin_prob"]
    p600 = mc["p600"]
    p800 = mc["p800"]
    p2000 = mc["p2000"]
    med_t800 = mc["med_t800"]
    med_t2000 = mc["med_t2000"]

    # 5) Score: maximize P(hit 2000), penalize ruin and (optionally) vol
    score = p2000 - lambda_ruin * ruin_prob
    if not np.isnan(ann_vol_lw) and gamma_vol > 0:
        score -= gamma_vol * ann_vol_lw

    return PortfolioCandidateMetrics(
        weights=weights,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        ann_vol_lw=ann_vol_lw,
        var_95=var_95,
        cvar_95=cvar_95,
        ruin_prob_1y=ruin_prob,
        p_hit_600_1y=p600,
        p_hit_800_1y=p800,
        p_hit_2000_1y=p2000,
        med_t_800_days=med_t800,
        med_t_2000_days=med_t2000,
        ending_equity_p5=mc["end_p5"],
        ending_equity_p25=mc["end_p25"],
        ending_equity_p50=mc["end_p50"],
        ending_equity_p75=mc["end_p75"],
        ending_equity_p95=mc["end_p95"],
        score=score,
    )
