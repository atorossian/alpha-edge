# report_engine.py
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# ---------- Data structures ----------

@dataclass
class Position:
    ticker: str
    quantity: float
    entry_price: float | None = None  # optional
    currency: str = "USD"

@dataclass
class PortfolioMetrics:
    # snapshot
    total_notional: float
    equity: float
    leverage: float
    positions_table: List[Dict[str, Any]]

    # unlevered stats
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float

    # risk metrics
    var_95: float
    cvar_95: float

    # leveraged MC
    ruin_prob_1y: float
    p_hit_600_1y: float
    p_hit_800_1y: float
    p_hit_2000_1y: float
    med_t_800_days: float | None
    med_t_2000_days: float | None

    # distribution of outcomes at 1y
    ending_equity_p5: float
    ending_equity_p25: float
    ending_equity_p50: float
    ending_equity_p75: float
    ending_equity_p95: float


# ---------- Core helpers ----------

def compute_portfolio_timeseries(
    closes: pd.DataFrame,
    positions: Dict[str, float],
) -> pd.Series:
    """
    closes: DataFrame indexed by date, columns=tickers, values=adj_close
    positions: ticker -> quantity
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

    # max drawdown
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    max_dd = dd.min()

    # VaR / CVaR (portfolio-level, 1d, 95%)
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
        rets=rets,  # return series for MC
    )


def simulate_leveraged_paths(
    rets: pd.Series,
    notional: float,
    equity0: float,
    days: int = 252,
    n_paths: int = 20000,
) -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    arr = rets.values
    n_hist = len(arr)

    eq_paths = np.empty((n_paths, days + 1))
    eq_paths[:, 0] = equity0

    hit_600 = np.zeros(n_paths, dtype=bool)
    hit_800 = np.zeros(n_paths, dtype=bool)
    hit_2000 = np.zeros(n_paths, dtype=bool)
    t_800 = np.full(n_paths, np.nan)
    t_2000 = np.full(n_paths, np.nan)

    for p in range(n_paths):
        E = equity0
        for t in range(1, days + 1):
            if E <= 0:
                eq_paths[p, t] = 0
                continue
            r = arr[rng.integers(0, n_hist)]
            dE = notional * r      # fixed-notional PnL
            E = E + dE
            if E <= 0:
                E = 0
            eq_paths[p, t] = E

            if (not hit_600[p]) and E >= 600:
                hit_600[p] = True
            if (not hit_800[p]) and E >= 800:
                hit_800[p] = True
                t_800[p] = t
            if (not hit_2000[p]) and E >= 2000:
                hit_2000[p] = True
                t_2000[p] = t

    ruin_prob = float(np.mean(eq_paths[:, -1] <= 0))
    p600 = float(np.mean(hit_600))
    p800 = float(np.mean(hit_800))
    p2000 = float(np.mean(hit_2000))

    med_t800 = float(np.nanmedian(t_800[hit_800])) if p800 > 0 else None
    med_t2000 = float(np.nanmedian(t_2000[hit_2000])) if p2000 > 0 else None

    end_eq = eq_paths[:, -1]
    percentiles = np.percentile(end_eq, [5, 25, 50, 75, 95])

    return dict(
        ruin_prob=ruin_prob,
        p600=p600,
        p800=p800,
        p2000=p2000,
        med_t800=med_t800,
        med_t2000=med_t2000,
        end_p5=float(percentiles[0]),
        end_p25=float(percentiles[1]),
        end_p50=float(percentiles[2]),
        end_p75=float(percentiles[3]),
        end_p95=float(percentiles[4]),
    )


def build_portfolio_metrics(
    closes: pd.DataFrame,
    positions: Dict[str, Position],
    equity: float,
) -> PortfolioMetrics:
    # 1) portfolio value series (unlevered)
    qty_map = {p.ticker: p.quantity for p in positions.values()}
    port_series = compute_portfolio_timeseries(closes, qty_map)

    total_notional = float(port_series.iloc[-1])
    leverage = total_notional / equity

    # 2) unlevered stats
    stats = compute_unlevered_stats(port_series)
    rets = stats.pop("rets")

    # 3) leveraged MC
    mc = simulate_leveraged_paths(
        rets=rets,
        notional=total_notional,
        equity0=equity,
        days=252,
        n_paths=20000,
    )

    # 4) build positions table for the report
    latest_prices = closes.iloc[-1]
    pos_rows = []
    for ticker, pos in positions.items():
        price = float(latest_prices[ticker])
        value = price * pos.quantity
        weight = value / total_notional
        pos_rows.append(dict(
            ticker=ticker,
            quantity=pos.quantity,
            price=price,
            value=value,
            weight=weight,
            currency=pos.currency,
        ))

    return PortfolioMetrics(
        total_notional=total_notional,
        equity=equity,
        leverage=leverage,
        positions_table=pos_rows,
        ann_return=stats["ann_return"],
        ann_vol=stats["ann_vol"],
        sharpe=stats["sharpe"],
        sortino=stats["sortino"],
        max_drawdown=stats["max_drawdown"],
        var_95=stats["var_95"],
        cvar_95=stats["cvar_95"],
        ruin_prob_1y=mc["ruin_prob"],
        p_hit_600_1y=mc["p600"],
        p_hit_800_1y=mc["p800"],
        p_hit_2000_1y=mc["p2000"],
        med_t_800_days=mc["med_t800"],
        med_t_2000_days=mc["med_t2000"],
        ending_equity_p5=mc["end_p5"],
        ending_equity_p25=mc["end_p25"],
        ending_equity_p50=mc["end_p50"],
        ending_equity_p75=mc["end_p75"],
        ending_equity_p95=mc["end_p95"],
    )

def summarize_metrics(metrics: PortfolioMetrics) -> str:
    return f"""
Daily Portfolio Summary
-----------------------

Total Notional: {metrics.total_notional:,.2f} USD
Equity: {metrics.equity:,.2f} USD
Leverage: {metrics.leverage:.2f}x

Unlevered Stats:
- Annual Return: {metrics.ann_return:.2%}
- Annual Vol: {metrics.ann_vol:.2%}
- Sharpe: {metrics.sharpe:.2f}
- Sortino: {metrics.sortino:.2f}
- Max Drawdown: {metrics.max_drawdown:.2%}

Risk Metrics:
- 1-day VaR(95): {metrics.var_95:.2%}
- 1-day CVaR(95): {metrics.cvar_95:.2%}

Leveraged Monte Carlo (1 year):
- Ruin Probability: {metrics.ruin_prob_1y:.2%}
- P(>= 800): {metrics.p_hit_800_1y:.2%}
- P(>= 2000): {metrics.p_hit_2000_1y:.2%}
- Median Time to 800: {metrics.med_t_800_days or 'N/A'} days
- Median Time to 2000: {metrics.med_t_2000_days or 'N/A'} days

Ending Equity Percentiles:
- P5: {metrics.ending_equity_p5:,.2f} USD
- P25: {metrics.ending_equity_p25:,.2f} USD
- P50: {metrics.ending_equity_p50:,.2f} USD
- P75: {metrics.ending_equity_p75:,.2f} USD
- P95: {metrics.ending_equity_p95:,.2f} USD
"""
