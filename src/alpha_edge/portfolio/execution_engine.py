# execution_engine.py
from __future__ import annotations

from typing import Dict
import numpy as np

from alpha_edge.core.schemas import DiscreteAllocation

DEFAULT_CRYPTO_DECIMALS = 8


def is_crypto_ticker(ticker: str) -> bool:
    t = str(ticker).upper().strip()
    return t.endswith("-USD") and len(t) > 4


def _quantize_qty(ticker: str, qty: float, *, crypto_decimals: int) -> float:
    """
    Signed quantization:
      - Crypto: round to allowed decimals
      - Non-crypto: integer shares, truncate toward zero (keeps sign)
    """
    if is_crypto_ticker(ticker):
        return float(np.round(qty, int(crypto_decimals)))
    return float(int(np.trunc(qty)))


def weights_to_discrete_shares(
    weights: Dict[str, float],
    prices: Dict[str, float],
    notional: float,
    *,
    min_units: float = 1.0,
    min_weight: float = 0.01,
    crypto_decimals: int = DEFAULT_CRYPTO_DECIMALS,
) -> DiscreteAllocation:
    """
    Supports LONG/SHORT weights by targeting *gross* notional.

    Interpretation:
      - 'notional' is gross target: sum(abs(exposure)) ~= notional.
      - Weights can be negative (short). We normalize by sum(abs(w)).
      - realized_value is SIGNED exposure (price * qty).
      - total_spent is GROSS used notional (sum abs exposures).
      - cash_left is remaining gross capacity (notional - gross_used).

    Notes:
      - Applies min_units per included ticker when feasible.
      - Drops tiny weights by abs(weight) < min_weight (unless that would drop all).
      - Quantizes equity shares to integers (toward zero) and crypto to decimals.
    """
    notional = float(notional)
    if not np.isfinite(notional) or notional <= 0:
        raise ValueError("notional must be finite and > 0")

    # keep only tickers with finite positive prices
    px = {
        str(t).upper().strip(): float(prices[t])
        for t in prices
        if t is not None and np.isfinite(float(prices[t])) and float(prices[t]) > 0
    }

    # sanitize weights: keep only tickers with prices and finite weights, non-zero
    w_raw: Dict[str, float] = {}
    for t, x in (weights or {}).items():
        tt = str(t).upper().strip()
        if not tt or tt not in px:
            continue
        try:
            xv = float(x)
        except Exception:
            continue
        if not np.isfinite(xv) or xv == 0.0:
            continue
        w_raw[tt] = xv

    if not w_raw:
        raise ValueError("No valid weights/prices overlap")

    # drop tiny ABS weights (keeps sign)
    w = {t: v for t, v in w_raw.items() if abs(v) >= float(min_weight)}
    if not w:
        # keep the largest |w|
        t0 = max(w_raw.items(), key=lambda kv: abs(kv[1]))[0]
        w = {t0: w_raw[t0]}

    # normalize by gross weight
    denom = float(sum(abs(v) for v in w.values()))
    if denom <= 0:
        raise ValueError("Sum(abs(weights)) <= 0")
    w_norm = {t: float(v) / denom for t, v in w.items()}

    # helper: min qty (signed) should follow weight sign
    def signed_min_qty(t: str) -> float:
        base = float(min_units)
        if is_crypto_ticker(t):
            q = _quantize_qty(t, base, crypto_decimals=crypto_decimals)
            return float(q if abs(q) > 0 else (10.0 ** (-int(crypto_decimals))))
        # equities must be >= 1 share magnitude
        return 1.0

    # Allocate baseline min_units per ticker if affordable in gross terms.
    # For shorts, min is negative shares.
    qty: Dict[str, float] = {t: 0.0 for t in w_norm}

    # Build signed mins
    min_qty_map: Dict[str, float] = {}
    for t, wt in w_norm.items():
        sgn = 1.0 if wt >= 0 else -1.0
        min_qty_map[t] = sgn * signed_min_qty(t)

    def gross_cost_of(qty_map: Dict[str, float]) -> float:
        return float(sum(abs(float(qty_map[t]) * px[t]) for t in qty_map))

    # Check min allocation feasibility; if too expensive, drop smallest |w| / most expensive until feasible
    if gross_cost_of(min_qty_map) > notional:
        # drop by (abs(weight), -price) => drop low conviction first, expensive first
        order = sorted(w_norm.keys(), key=lambda t: (abs(w_norm[t]), -px[t]))
        kept = dict(w_norm)
        kept_min = dict(min_qty_map)
        while kept and gross_cost_of(kept_min) > notional:
            drop = order.pop(0)
            kept.pop(drop, None)
            kept_min.pop(drop, None)

        if not kept:
            # fallback: keep the cheapest instrument only
            cheapest = min(w_norm.keys(), key=lambda t: px[t])
            if px[cheapest] > notional:
                raise ValueError(f"Notional {notional:.2f} cannot buy/sell minimum unit of any selected asset")
            kept = {cheapest: w_norm[cheapest]}
            kept_min = {cheapest: min_qty_map[cheapest]}

        # renormalize by gross abs weights
        denom2 = float(sum(abs(v) for v in kept.values()))
        w_norm = {t: kept[t] / denom2 for t in kept}
        min_qty_map = kept_min
        qty = {t: 0.0 for t in w_norm}

    # apply minimums
    for t in w_norm:
        qty[t] = float(min_qty_map[t])

    used_gross = gross_cost_of(qty)
    remaining = float(max(0.0, notional - used_gross))

    # Phase 1: proportional fill toward target exposures (signed)
    # target gross exposure per ticker
    tgt_exp = {t: w_norm[t] * notional for t in w_norm}  # signed
    cur_exp = {t: qty[t] * px[t] for t in w_norm}        # signed

    # Allocate remaining gross by closing the gap in abs exposure toward target
    if remaining > 0:
        # sort by largest desired abs exposure first
        for t in sorted(w_norm.keys(), key=lambda k: -abs(tgt_exp[k])):
            p = px[t]
            if remaining <= 0:
                break

            # desired additional signed exposure (can be +/-)
            gap = float(tgt_exp[t] - cur_exp[t])
            if gap == 0:
                continue

            # gross budget chunk proportional to abs target weight
            # (simple approach, avoids solving knapsack)
            budget = remaining * (abs(w_norm[t]) / float(sum(abs(v) for v in w_norm.values())))
            desired_exp_add = np.sign(gap) * min(abs(gap), budget)

            raw_add_qty = desired_exp_add / p
            add_qty = _quantize_qty(t, raw_add_qty, crypto_decimals=crypto_decimals)

            if add_qty != 0.0:
                qty[t] += add_qty
                cur_exp[t] = qty[t] * p
                used_gross = float(sum(abs(cur_exp[k]) for k in cur_exp))
                remaining = float(max(0.0, notional - used_gross))

    # Phase 2: greedy spend remaining gross in 1-unit steps
    # choose ticker with largest abs exposure shortfall vs target
    max_iter = 200000
    it = 0
    while remaining > 0 and it < max_iter:
        it += 1

        # which tickers can afford one step in gross terms?
        affordable = []
        for t in w_norm:
            p = px[t]
            if is_crypto_ticker(t):
                step = 10.0 ** (-int(crypto_decimals))
                if p * step <= remaining:
                    affordable.append(t)
            else:
                if p <= remaining:
                    affordable.append(t)

        if not affordable:
            break

        # pick ticker with biggest abs shortfall: |target|-|current|
        def shortfall(t: str) -> float:
            return abs(tgt_exp[t]) - abs(cur_exp[t])

        best_t = max(affordable, key=shortfall)
        p = px[best_t]
        sgn = 1.0 if tgt_exp[best_t] >= 0 else -1.0

        if is_crypto_ticker(best_t):
            step = 10.0 ** (-int(crypto_decimals))
            if p * step > remaining:
                break
            qty[best_t] += sgn * step
            qty[best_t] = float(np.round(qty[best_t], int(crypto_decimals)))
        else:
            if p > remaining:
                break
            qty[best_t] += sgn * 1.0

        cur_exp[best_t] = qty[best_t] * p
        used_gross = float(sum(abs(cur_exp[k]) for k in cur_exp))
        remaining = float(max(0.0, notional - used_gross))

    # Build realized outputs
    realized_value = {t: float(qty[t]) * float(px[t]) for t in qty}  # SIGNED
    gross_used = float(sum(abs(v) for v in realized_value.values()))
    cash_left = float(max(0.0, notional - gross_used))

    denom_notional = notional if notional > 0 else 1.0
    realized_weights = {t: float(realized_value[t]) / denom_notional for t in realized_value}  # SIGNED
    realized_weights["CASH"] = cash_left / denom_notional

    target_value = {t: float(tgt_exp[t]) for t in tgt_exp}  # SIGNED target exposure

    return DiscreteAllocation(
        shares=qty,  # signed quantities
        target_value=target_value,
        realized_value=realized_value,
        realized_weights=realized_weights,  # signed
        total_spent=gross_used,             # gross used
        cash_left=cash_left,                # gross remaining
    )
