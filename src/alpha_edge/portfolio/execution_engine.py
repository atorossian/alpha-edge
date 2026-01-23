# execution_engine.py
from __future__ import annotations

from typing import Dict, Mapping
import numpy as np

from alpha_edge.core.schemas import DiscreteAllocation


DEFAULT_CRYPTO_DECIMALS = 8


def is_crypto_ticker(ticker: str) -> bool:
    """
    Yahoo-style crypto pairs: BTC-USD, ETH-USD, SUI-USD, etc.
    Adjust if you have other conventions.
    """
    t = str(ticker).upper().strip()
    return t.endswith("-USD") and len(t) > 4


def _quantize_qty(ticker: str, qty: float, *, crypto_decimals: int) -> float:
    """
    - Crypto: round to allowed decimals
    - Non-crypto: floor to integer shares (toward zero)
    """
    if is_crypto_ticker(ticker):
        return float(np.round(qty, int(crypto_decimals)))
    # equities/ETFs: integer shares only
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
    Turn continuous weights into order quantities with:
      - optionally ignoring tiny weights (< min_weight)
      - enforcing min_units for included tickers
          * for equities: min_units is in shares (typically 1)
          * for crypto: min_units is in coins (can be fractional if you want)
      - never exceeding total notional (best-effort)

    Crypto tickers (e.g., BTC-USD) are allowed to be fractional, quantized to `crypto_decimals`.
    Non-crypto tickers are truncated to integer shares.
    """
    notional = float(notional)

    # sanitize + normalize (keep only positive weights here)
    w = {
        t: float(x)
        for t, x in weights.items()
        if float(x) > 0 and t in prices and np.isfinite(prices[t]) and float(prices[t]) > 0
    }
    if not w:
        raise ValueError("No valid weights/prices overlap")

    s = float(sum(w.values()))
    if s <= 0:
        raise ValueError("Weights sum <= 0")
    w = {t: x / s for t, x in w.items()}

    # drop tiny weights
    selected = {t: x for t, x in w.items() if x >= float(min_weight)}
    if not selected:
        t0 = max(w.items(), key=lambda kv: kv[1])[0]
        selected = {t0: w[t0]}

    # normalize again
    s2 = float(sum(selected.values()))
    selected = {t: x / s2 for t, x in selected.items()}

    # base: ensure min_units per ticker (if affordable)
    qty: Dict[str, float] = {t: 0.0 for t in selected}

    def unit_cost(t: str, units: float) -> float:
        return float(prices[t]) * float(units)

    # quantize min_units appropriately per asset type
    min_qty = {t: _quantize_qty(t, float(min_units), crypto_decimals=crypto_decimals) for t in selected}
    # ensure we don't accidentally quantize equities to 0
    for t in list(min_qty.keys()):
        if not is_crypto_ticker(t) and min_qty[t] < 1.0:
            min_qty[t] = 1.0

    cost_min = float(sum(unit_cost(t, min_qty[t]) for t in selected))

    # If we can't afford min for each, drop lowest-weight / most-expensive until feasible
    if cost_min > notional:
        order = sorted(selected.keys(), key=lambda t: (selected[t], -float(prices[t])))
        kept = dict(selected)
        while kept and float(sum(unit_cost(t, min_qty[t]) for t in kept)) > notional:
            drop = order.pop(0)
            kept.pop(drop, None)

        if not kept:
            # can't afford 1 share of cheapest equity OR min crypto unit
            cheapest = min(selected.keys(), key=lambda t: float(prices[t]))
            if float(prices[cheapest]) > notional:
                raise ValueError(f"Notional {notional:.2f} cannot buy the minimum unit of any selected asset")
            kept = {cheapest: 1.0}

        s3 = float(sum(kept.values()))
        selected = {t: kept[t] / s3 for t in kept}
        qty = {t: 0.0 for t in selected}

    # allocate min units
    for t in selected:
        qty[t] = float(min_qty[t])

    spent = float(sum(unit_cost(t, qty[t]) for t in qty))
    remaining = float(notional - spent)

    # Phase 1: proportional adds (in quantized steps)
    if remaining > 0:
        for t, wt in sorted(selected.items(), key=lambda kv: -kv[1]):
            price = float(prices[t])
            if price <= 0 or not np.isfinite(price):
                continue

            target_extra_value = remaining * float(wt)
            raw_add = target_extra_value / price

            if is_crypto_ticker(t):
                add = float(np.round(raw_add, int(crypto_decimals)))
            else:
                add = float(np.floor(raw_add))

            if add > 0:
                # quantize and apply
                add_q = _quantize_qty(t, add, crypto_decimals=crypto_decimals)
                if add_q > 0:
                    qty[t] += add_q
                    remaining -= add_q * price

    # Phase 2: greedy spend rest
    # For equities: step=1 share
    # For crypto: step = 10^-crypto_decimals coin
    max_iter = 50000
    it = 0
    while remaining > 0:
        it += 1
        if it > max_iter:
            break

        affordable = [t for t in selected if float(prices[t]) <= remaining]
        if not affordable:
            break

        cur_val = {t: float(qty[t]) * float(prices[t]) for t in selected}
        tgt_val = {t: notional * float(selected[t]) for t in selected}

        best_t = max(affordable, key=lambda t: (tgt_val[t] - cur_val[t]))

        p = float(prices[best_t])
        if is_crypto_ticker(best_t):
            step = 10.0 ** (-int(crypto_decimals))
            if p * step > remaining:
                # can't afford one smallest step; stop
                break
            qty[best_t] += step
            remaining -= p * step
            # keep numerical hygiene
            qty[best_t] = float(np.round(qty[best_t], int(crypto_decimals)))
        else:
            if p > remaining:
                break
            qty[best_t] += 1.0
            remaining -= p

    realized_value = {t: float(qty[t]) * float(prices[t]) for t in qty}
    total_spent = float(sum(realized_value.values()))
    cash_left = float(notional - total_spent)

    denom = notional if notional > 0 else 1.0
    realized_weights = {t: realized_value[t] / denom for t in realized_value}
    realized_weights["CASH"] = float(notional - total_spent) / denom

    target_value = {t: notional * float(selected[t]) for t in selected}

    return DiscreteAllocation(
        shares=qty,  # NOTE: now float quantities
        target_value=target_value,
        realized_value=realized_value,
        realized_weights=realized_weights,
        total_spent=total_spent,
        cash_left=cash_left,
    )
