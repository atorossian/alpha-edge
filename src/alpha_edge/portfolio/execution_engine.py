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


def _step_size(ticker: str, *, crypto_decimals: int) -> float:
    return float(10.0 ** (-int(crypto_decimals))) if is_crypto_ticker(ticker) else 1.0


def _quantize_toward_zero_steps(ticker: str, qty: float, *, crypto_decimals: int) -> float:
    """
    Explicit toward-zero quantization in "steps" (1 share or crypto step).
    """
    step = _step_size(ticker, crypto_decimals=crypto_decimals)
    if step <= 0:
        return 0.0
    steps = float(qty) / step
    steps_q = float(np.trunc(steps))
    out = steps_q * step
    if is_crypto_ticker(ticker):
        out = float(np.round(out, int(crypto_decimals)))
    return float(out)


def _quantize_nearest_steps(ticker: str, qty: float, *, crypto_decimals: int) -> float:
    """
    Nearest-step quantization (can overshoot per-ticker), but we still enforce gross <= notional.
    """
    step = _step_size(ticker, crypto_decimals=crypto_decimals)
    if step <= 0:
        return 0.0
    steps = float(qty) / step
    steps_q = float(np.round(steps))
    out = steps_q * step
    if is_crypto_ticker(ticker):
        out = float(np.round(out, int(crypto_decimals)))
    return float(out)


def weights_to_discrete_shares(
    weights: Dict[str, float],
    prices: Dict[str, float],
    notional: float,
    *,
    min_weight: float = 0.01,
    min_units_equity: float = 1.0,
    min_units_crypto: float = 0.0,
    min_units_weight_thr: float = 0.03,
    crypto_decimals: int = DEFAULT_CRYPTO_DECIMALS,
    nearest_step_remaining_frac: float = 0.10,
) -> DiscreteAllocation:
    """
    Supports LONG/SHORT weights by targeting *gross* notional.

    Interpretation:
      - 'notional' is gross target: sum(abs(exposure)) ~= notional.
      - Weights can be negative (short). We normalize by sum(abs(w)).
      - realized_value is SIGNED exposure (price * qty).
      - total_spent is GROSS used notional (sum abs exposures).
      - cash_left is remaining gross capacity (notional - gross_used).
    """
    notional = float(notional)
    if not np.isfinite(notional) or notional <= 0:
        raise ValueError("notional must be finite and > 0")

    # keep only tickers with finite positive prices
    px: Dict[str, float] = {}
    for t, p in (prices or {}).items():
        tt = str(t).upper().strip()
        if not tt:
            continue
        try:
            pf = float(p)
        except Exception:
            continue
        if np.isfinite(pf) and pf > 0:
            px[tt] = pf

    # sanitize weights: keep only tickers with prices and finite weights, non-zero
    w_raw: Dict[str, float] = {}
    missing_px: list[str] = []

    for t, x in (weights or {}).items():
        tt = str(t).upper().strip()
        if not tt:
            continue
        if tt not in px:
            missing_px.append(tt)
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

    # SAFE gross_in calculation (avoids crashing on non-numeric original weights)
    gross_in = 0.0
    for v in (weights or {}).values():
        try:
            vf = float(v)
        except Exception:
            continue
        if np.isfinite(vf):
            gross_in += abs(vf)

    gross_kept = float(sum(abs(v) for v in w_raw.values()))

    # if more than 20% of gross weights got dropped due to missing prices -> fail loudly
    if gross_in > 0 and (gross_kept / gross_in) < 0.80:
        missing_preview = ", ".join(sorted(set(missing_px))[:20])
        raise ValueError(
            f"Too many weights missing prices (kept={gross_kept/gross_in:.1%}). "
            f"Missing_px sample: {missing_preview}"
        )

    # drop tiny ABS weights (keeps sign)
    w = {t: v for t, v in w_raw.items() if abs(v) >= float(min_weight)}
    if not w:
        t0 = max(w_raw.items(), key=lambda kv: abs(kv[1]))[0]
        w = {t0: w_raw[t0]}

    # normalize by gross abs weights (signed preserved)
    denom = float(sum(abs(v) for v in w.values()))
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Sum(abs(weights)) <= 0")
    w_norm = {t: float(v) / denom for t, v in w.items()}  # signed, gross=1

    # --- target signed exposure per ticker (gross notional basis) ---
    tgt_exp = {t: float(w_norm[t]) * notional for t in w_norm}  # signed USD exposure

    def q_safe(t: str, q: float) -> float:
        return _quantize_toward_zero_steps(t, q, crypto_decimals=crypto_decimals)

    def q_nearest(t: str, q: float) -> float:
        return _quantize_nearest_steps(t, q, crypto_decimals=crypto_decimals)

    qty: Dict[str, float] = {}
    for t in w_norm:
        q_raw = float(tgt_exp[t]) / float(px[t])
        qty[t] = q_safe(t, q_raw)

    def gross_used(qmap: Dict[str, float]) -> float:
        return float(sum(abs(float(qmap[t]) * float(px[t])) for t in qmap))

    # --- enforce gross cap (rare overshoot due to crypto rounding) ---
    g0 = gross_used(qty)
    if g0 > notional:
        scale = float(notional / g0) if g0 > 0 else 0.0
        qty2: Dict[str, float] = {}
        for t in qty:
            qty2[t] = q_safe(t, float(qty[t]) * scale)
        qty = qty2

    # --- conditional min-units baseline (asset-type aware) ---
    def baseline_min_qty(t: str, wt: float) -> float:
        if abs(float(wt)) < float(min_units_weight_thr):
            return 0.0

        sgn = 1.0 if wt >= 0 else -1.0

        if is_crypto_ticker(t):
            mu = float(min_units_crypto)
            if mu <= 0:
                return 0.0
            q = _quantize_toward_zero_steps(t, mu, crypto_decimals=crypto_decimals)
            if abs(q) <= 0:
                q = _step_size(t, crypto_decimals=crypto_decimals)
            return float(sgn * q)

        mu = float(min_units_equity)
        if mu <= 0:
            return 0.0
        return float(sgn * max(1.0, float(int(mu))))

    qty_min_add = {t: baseline_min_qty(t, w_norm[t]) for t in w_norm}
    qty_with_min = {t: float(qty.get(t, 0.0)) for t in w_norm}

    for t, qmin in qty_min_add.items():
        if qmin == 0.0:
            continue
        if np.sign(qmin) == np.sign(qty_with_min[t]) and abs(qty_with_min[t]) >= abs(qmin):
            continue
        qty_with_min[t] = float(qmin)

    if gross_used(qty_with_min) <= notional:
        qty = qty_with_min

    used = gross_used(qty)
    remaining = float(max(0.0, notional - used))

    # --- nearest-step improve tracking (only when remaining is plenty) ---
    if remaining >= float(nearest_step_remaining_frac) * notional:
        cur_exp = {t: float(qty[t]) * float(px[t]) for t in qty}

        def abs_shortfall(t: str) -> float:
            return float(abs(tgt_exp[t]) - abs(cur_exp[t]))

        order = sorted(w_norm.keys(), key=lambda t: abs_shortfall(t), reverse=True)

        for t in order:
            q_raw = float(tgt_exp[t]) / float(px[t])
            q_prop = q_nearest(t, q_raw)
            if q_prop == qty[t]:
                continue

            prop_qty = dict(qty)
            prop_qty[t] = float(q_prop)
            g_prop = gross_used(prop_qty)
            if g_prop > notional:
                continue

            cur_sf = abs(float(abs(tgt_exp[t]) - abs(cur_exp[t])))
            prop_exp = float(prop_qty[t]) * float(px[t])
            prop_sf = abs(float(abs(tgt_exp[t]) - abs(prop_exp)))

            if prop_sf <= cur_sf + 1e-12:
                qty = prop_qty
                used = g_prop
                remaining = float(max(0.0, notional - used))
                cur_exp[t] = prop_exp

    # --- final greedy top-up (1-step increments), BUT only if there is positive shortfall ---
    cur_exp = {t: float(qty[t]) * float(px[t]) for t in qty}
    max_iter = 200000
    it = 0

    while remaining > 0 and it < max_iter:
        it += 1

        affordable = []
        for t in w_norm:
            p = float(px[t])
            step = _step_size(t, crypto_decimals=crypto_decimals)
            if p * step <= remaining + 1e-12:
                affordable.append(t)
        if not affordable:
            break

        def shortfall(t: str) -> float:
            return float(abs(tgt_exp[t]) - abs(cur_exp[t]))

        # NEW: stop if there's no positive shortfall anywhere (don't waste remaining)
        best_t = max(affordable, key=shortfall)
        best_sf = float(shortfall(best_t))
        if not np.isfinite(best_sf) or best_sf <= 1e-12:
            break

        p = float(px[best_t])
        step = _step_size(best_t, crypto_decimals=crypto_decimals)
        sgn = 1.0 if float(tgt_exp[best_t]) >= 0 else -1.0

        if p * step > remaining + 1e-12:
            break

        qty[best_t] = float(qty[best_t] + sgn * step)
        if is_crypto_ticker(best_t):
            qty[best_t] = float(np.round(qty[best_t], int(crypto_decimals)))

        cur_exp[best_t] = float(qty[best_t]) * p
        used = float(sum(abs(v) for v in cur_exp.values()))
        remaining = float(max(0.0, notional - used))

    realized_value = {t: float(qty[t]) * float(px[t]) for t in qty}  # SIGNED
    gross_used_final = float(sum(abs(v) for v in realized_value.values()))
    cash_left = float(max(0.0, notional - gross_used_final))

    denom_notional = notional if notional > 0 else 1.0
    realized_weights = {t: float(realized_value[t]) / denom_notional for t in realized_value}  # SIGNED
    realized_weights["CASH"] = cash_left / denom_notional

    target_value = {t: float(tgt_exp[t]) for t in tgt_exp}  # SIGNED target exposure

    return DiscreteAllocation(
        shares=qty,                  # signed quantities
        target_value=target_value,   # signed exposure targets
        realized_value=realized_value,
        realized_weights=realized_weights,
        total_spent=gross_used_final,  # gross used
        cash_left=cash_left,           # gross remaining
    )
