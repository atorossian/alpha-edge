# backtest/executor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from alpha_edge.portfolio.execution_engine import weights_to_discrete_shares


@dataclass
class Trade:
    date: str
    ticker: str
    qty_delta: float
    price: float
    notional: float


def apply_trades(positions_qty: Dict[str, float], trades: List[Trade]) -> Dict[str, float]:
    for t in trades:
        positions_qty[t.ticker] = float(positions_qty.get(t.ticker, 0.0)) + float(t.qty_delta)
        if abs(float(positions_qty[t.ticker])) < 1e-12:
            positions_qty.pop(t.ticker, None)
    return positions_qty


def mark_to_market(positions_qty: Dict[str, float], prices: Dict[str, float]) -> float:
    v = 0.0
    for t, q in positions_qty.items():
        px = float(prices.get(t, np.nan))
        if np.isfinite(px):
            v += float(q) * px
    return float(v)

def target_qty_from_weights(*, weights: Dict[str, float], prices: Dict[str, float], target_notional: float) -> Dict[str, float]:
    alloc = weights_to_discrete_shares(
        weights=weights,
        prices=prices,
        notional=float(target_notional),
        min_units=1.0,
        min_weight=0.01,
        crypto_decimals=8,
    )
    return {t: float(q) for t, q in alloc.shares.items() if t != "CASH"}

def rebalance_to_target(
    *,
    date: str,
    positions_qty: Dict[str, float],
    target_qty: Dict[str, float],
    prices: Dict[str, float],
    min_abs_notional: float = 1.0,
) -> List[Trade]:
    trades: List[Trade] = []
    all_tickers = sorted(set(positions_qty.keys()) | set(target_qty.keys()))
    for t in all_tickers:
        q0 = float(positions_qty.get(t, 0.0))
        q1 = float(target_qty.get(t, 0.0))
        dq = q1 - q0
        if abs(dq) <= 0.0:
            continue
        px = float(prices.get(t, np.nan))
        if (not np.isfinite(px)) or px <= 0:
            continue
        notional = dq * px
        if abs(notional) < float(min_abs_notional):
            continue
        trades.append(Trade(date=date, ticker=t, qty_delta=float(dq), price=px, notional=float(notional)))
        positions_qty[t] = q1
    # drop near-zero dust
    positions_qty = {t: q for t, q in positions_qty.items() if abs(q) > 1e-12}
    return trades
