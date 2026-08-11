# execution_engine.py
from __future__ import annotations

from typing import Dict
import numpy as np

from alpha_edge.core.schemas import (
    DiscreteAllocation,
    TransitionExecutionConfig,
    TransitionExecutionPlan,
    TransitionTradeDelta,
)

DEFAULT_CRYPTO_DECIMALS = 8


def is_crypto_ticker(ticker: str) -> bool:
    t = str(ticker).upper().strip()
    return t.endswith("-USD") and len(t) > 4


def _step_size(ticker: str, *, crypto_decimals: int) -> float:
    return float(10.0 ** (-int(crypto_decimals))) if is_crypto_ticker(ticker) else 1.0


def _quantize_toward_zero_steps(ticker: str, qty: float, *, crypto_decimals: int) -> float:
    """
    Explicit toward-zero quantization in "steps" (1 share or crypto step).
    Keeps sign; truncates steps toward zero.
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
    # inclusion / filtering
    min_weight: float = 0.01,
    # baseline mins (asset-aware)
    min_units_equity: float = 1.0,
    min_units_crypto: float = 0.0,
    min_units_weight_thr: float = 0.03,
    # quantization
    crypto_decimals: int = DEFAULT_CRYPTO_DECIMALS,
    nearest_step_remaining_frac: float = 0.10,
    # greedy top-up controls
    max_topup_iters: int = 200000,
    topup_chunk_max_steps: int = 5000,
) -> DiscreteAllocation:
    """
    Supports LONG/SHORT weights by targeting *gross* notional.

    Interpretation:
      - 'notional' is gross target: sum(abs(exposure)) ~= notional.
      - Weights can be negative (short). We normalize by sum(abs(w)).
      - realized_value is SIGNED exposure (price * qty).
      - total_spent is GROSS used notional (sum abs exposures).
      - cash_left is remaining gross capacity (notional - gross_used).

    Key behaviors:
      1) Safe weight sanitization + missing-price fail-fast using a robust gross_in calc.
      2) Base quantization is toward-zero steps (equities: int shares, crypto: decimals).
      3) Baseline min-units are applied BEST-EFFORT in priority order (not all-or-nothing).
      4) Optional nearest-step pass when remaining gross is “plenty”.
      5) Final greedy top-up uses chunked steps to avoid crypto micro-step slowdowns,
         and stops if there is no positive shortfall anywhere.
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

    # initial safe target qty
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

    # BEST-EFFORT: apply mins in priority order without exceeding gross cap
    # (avoids all-or-nothing dropping shorts when notional is tight)
    qty_try = dict(qty)
    for t in sorted(w_norm.keys(), key=lambda k: abs(w_norm[k]), reverse=True):
        qmin = float(qty_min_add.get(t, 0.0))
        if qmin == 0.0:
            continue

        qcur = float(qty_try.get(t, 0.0))

        # already meets min in correct direction
        if np.sign(qmin) == np.sign(qcur) and abs(qcur) >= abs(qmin):
            continue

        prop = dict(qty_try)
        prop[t] = float(qmin)
        if gross_used(prop) <= notional:
            qty_try = prop

    qty = qty_try

    used = gross_used(qty)
    remaining = float(max(0.0, notional - used))

    # --- nearest-step improve tracking (only when remaining is plenty) ---
    if remaining >= float(nearest_step_remaining_frac) * notional:
        cur_exp = {t: float(qty[t]) * float(px[t]) for t in qty}

        def signed_shortfall(t: str) -> float:
            # >0 means under-allocated (abs); <0 means over-allocated (abs)
            return float(abs(tgt_exp[t]) - abs(cur_exp.get(t, 0.0)))

        order = sorted(w_norm.keys(), key=lambda t: signed_shortfall(t), reverse=True)

        for t in order:
            q_raw = float(tgt_exp[t]) / float(px[t])
            q_prop = q_nearest(t, q_raw)
            if q_prop == qty.get(t, 0.0):
                continue

            prop_qty = dict(qty)
            prop_qty[t] = float(q_prop)
            g_prop = gross_used(prop_qty)
            if g_prop > notional:
                continue

            cur_sf = abs(float(abs(tgt_exp[t]) - abs(cur_exp.get(t, 0.0))))
            prop_exp = float(prop_qty[t]) * float(px[t])
            prop_sf = abs(float(abs(tgt_exp[t]) - abs(prop_exp)))

            if prop_sf <= cur_sf + 1e-12:
                qty = prop_qty
                used = g_prop
                remaining = float(max(0.0, notional - used))
                cur_exp[t] = prop_exp

    # --- final greedy top-up (chunked steps), BUT only if there is positive shortfall ---
    cur_exp = {t: float(qty[t]) * float(px[t]) for t in qty}
    it = 0

    def signed_shortfall_abs(t: str) -> float:
        return float(abs(tgt_exp[t]) - abs(cur_exp.get(t, 0.0)))

    while remaining > 0 and it < int(max_topup_iters):
        it += 1

        affordable = []
        for t in w_norm:
            p = float(px[t])
            step = _step_size(t, crypto_decimals=crypto_decimals)
            if p * step <= remaining + 1e-12:
                affordable.append(t)
        if not affordable:
            break

        best_t = max(affordable, key=signed_shortfall_abs)
        best_sf = float(signed_shortfall_abs(best_t))
        if not np.isfinite(best_sf) or best_sf <= 1e-12:
            break

        p = float(px[best_t])
        step = _step_size(best_t, crypto_decimals=crypto_decimals)
        sgn = 1.0 if float(tgt_exp[best_t]) >= 0 else -1.0

        step_cost = float(p * step)
        if step_cost <= 0 or step_cost > remaining + 1e-12:
            break

        # CHUNK the number of steps to reduce iterations (especially for crypto)
        max_steps_by_cash = int(remaining / step_cost)
        max_steps_by_sf = int(best_sf / step_cost)  # only fill up to the abs-target
        n_steps = max(1, min(max_steps_by_cash, max_steps_by_sf, int(topup_chunk_max_steps)))

        qty[best_t] = float(qty.get(best_t, 0.0) + sgn * step * n_steps)
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
        shares=qty,                    # signed quantities
        target_value=target_value,     # signed exposure targets
        realized_value=realized_value,
        realized_weights=realized_weights,
        total_spent=gross_used_final,  # gross used
        cash_left=cash_left,           # gross remaining
    )


def _clean_signed_quantity_map(values: Dict[str, float] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for k, v in (values or {}).items():
        t = str(k).upper().strip()
        if not t:
            continue

        try:
            q = float(v)
        except Exception:
            continue

        if not np.isfinite(q) or abs(q) <= 1e-12:
            continue

        out[t] = float(q)

    return out


def _clean_price_map(values: Dict[str, float] | None) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for k, v in (values or {}).items():
        t = str(k).upper().strip()
        if not t:
            continue

        try:
            p = float(v)
        except Exception:
            continue

        if not np.isfinite(p) or p <= 0:
            continue

        out[t] = float(p)

    return out


def _trade_direction(delta_quantity: float) -> str:
    if float(delta_quantity) > 0:
        return "BUY"
    if float(delta_quantity) < 0:
        return "SELL"
    return "HOLD"


def _gross_weight_map_from_shares(
    *,
    shares: Dict[str, float],
    prices: Dict[str, float],
    notional: float,
) -> Dict[str, float]:
    """
    Convert signed shares into signed notional weights.

    Uses the supplied notional as denominator, consistent with
    weights_to_discrete_shares(), where realized_weights are exposure / notional.
    """
    denom = float(notional)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("notional must be finite and > 0")

    out: Dict[str, float] = {}

    for t, q in _clean_signed_quantity_map(shares).items():
        if t not in prices:
            raise ValueError(f"Missing price for current holding {t!r}")

        value = float(q) * float(prices[t])
        if abs(value) <= 1e-12:
            continue

        out[t] = float(value / denom)

    return out


def _weight_turnover(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
) -> float:
    keys = set(current_weights.keys()) | set(target_weights.keys())

    return float(
        0.5
        * sum(
            abs(float(target_weights.get(k, 0.0)) - float(current_weights.get(k, 0.0)))
            for k in keys
        )
    )


def _scale_target_weights_to_daily_turnover(
    *,
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    max_daily_turnover: float,
) -> tuple[Dict[str, float], float, float]:
    """
    Move only part of the way from current weights to target weights when
    required turnover exceeds max_daily_turnover.

    This function works at the weight level. The resulting adjusted target
    is later passed into weights_to_discrete_shares(), which remains the
    source of truth for executable sizing.
    """
    full_turnover = _weight_turnover(current_weights, target_weights)

    if full_turnover <= 1e-12:
        return dict(current_weights), 0.0, 0.0

    if full_turnover <= float(max_daily_turnover):
        return dict(target_weights), float(full_turnover), 0.0

    scale = float(max_daily_turnover) / float(full_turnover)

    keys = sorted(set(current_weights.keys()) | set(target_weights.keys()))

    adjusted = {
        k: float(
            float(current_weights.get(k, 0.0))
            + scale
            * (
                float(target_weights.get(k, 0.0))
                - float(current_weights.get(k, 0.0))
            )
        )
        for k in keys
    }

    adjusted = {
        k: v
        for k, v in adjusted.items()
        if abs(float(v)) > 1e-12
    }

    daily_turnover = _weight_turnover(current_weights, adjusted)
    blocked_turnover = max(0.0, full_turnover - daily_turnover)

    return adjusted, float(daily_turnover), float(blocked_turnover)


def allocation_to_trade_deltas(
    *,
    current_shares: Dict[str, float],
    target_allocation: DiscreteAllocation,
    prices: Dict[str, float],
    notional: float,
    cfg: TransitionExecutionConfig | None = None,
) -> tuple[list[TransitionTradeDelta], list[TransitionTradeDelta]]:
    """
    Compare current signed shares against target signed shares and produce
    recommendation-only BUY/SELL deltas.

    This does not execute trades.
    This does not write to the ledger.
    """
    if cfg is None:
        cfg = TransitionExecutionConfig()

    px = _clean_price_map(prices)
    current = _clean_signed_quantity_map(current_shares)
    target = _clean_signed_quantity_map(target_allocation.shares)

    denom = float(notional)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("notional must be finite and > 0")

    tickers = sorted(set(current.keys()) | set(target.keys()))

    trades: list[TransitionTradeDelta] = []
    blocked_trades: list[TransitionTradeDelta] = []

    for t in tickers:
        if t not in px:
            raise ValueError(f"Missing price for trade delta asset {t!r}")

        price = float(px[t])

        current_qty = float(current.get(t, 0.0))
        target_qty = float(target.get(t, 0.0))
        delta_qty = float(target_qty - current_qty)

        if abs(delta_qty) <= 1e-12:
            continue

        current_value = float(current_qty * price)
        target_value = float(target_qty * price)
        delta_value = float(target_value - current_value)

        current_weight = float(current_value / denom)
        target_weight = float(target_value / denom)
        delta_weight = float(delta_value / denom)

        direction = _trade_direction(delta_qty)

        reason = "included"
        is_blocked = False

        if abs(delta_value) < float(cfg.min_trade_value):
            reason = (
                f"abs(delta_value) {abs(delta_value):.2f} below "
                f"min_trade_value {float(cfg.min_trade_value):.2f}"
            )
            is_blocked = True

        if abs(delta_qty) < float(cfg.min_trade_quantity):
            reason = (
                f"abs(delta_quantity) {abs(delta_qty):.10f} below "
                f"min_trade_quantity {float(cfg.min_trade_quantity):.10f}"
            )
            is_blocked = True

        trade = TransitionTradeDelta(
            asset_id=str(t),
            direction=direction,
            current_quantity=float(current_qty),
            target_quantity=float(target_qty),
            delta_quantity=float(delta_qty),
            price=float(price),
            current_value=float(current_value),
            target_value=float(target_value),
            delta_value=float(delta_value),
            current_weight=float(current_weight),
            target_weight=float(target_weight),
            delta_weight=float(delta_weight),
            reason=reason,
        )

        if is_blocked:
            blocked_trades.append(trade)
        else:
            trades.append(trade)

    return trades, blocked_trades


def build_transition_execution_plan(
    *,
    as_of: str,
    source: str,
    current_shares: Dict[str, float],
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    notional: float,
    cfg: TransitionExecutionConfig | None = None,
    # passthrough controls for weights_to_discrete_shares()
    min_weight: float = 0.01,
    min_units_equity: float = 1.0,
    min_units_crypto: float = 0.0,
    min_units_weight_thr: float = 0.03,
    crypto_decimals: int = DEFAULT_CRYPTO_DECIMALS,
    nearest_step_remaining_frac: float = 0.10,
    max_topup_iters: int = 200000,
    topup_chunk_max_steps: int = 5000,
) -> TransitionExecutionPlan:
    """
    Build a recommendation-only transition execution plan.

    Responsibilities:
      - apply turnover limits
      - call weights_to_discrete_shares()
      - compare target shares with current shares
      - produce BUY/SELL deltas

    Non-responsibilities:
      - portfolio search
      - portfolio health scoring
      - broker execution
      - ledger writing
    """
    if cfg is None:
        cfg = TransitionExecutionConfig()

    notional = float(notional)
    if not np.isfinite(notional) or notional <= 0:
        raise ValueError("notional must be finite and > 0")

    px = _clean_price_map(prices)
    current = _clean_signed_quantity_map(current_shares)

    if not current:
        raise ValueError("current_shares is empty; cannot build transition plan.")

    current_weights = _gross_weight_map_from_shares(
        shares=current,
        prices=px,
        notional=notional,
    )

    # Build the full target allocation first so we can measure true target turnover
    # using the same executable allocation logic that the project already trusts.
    full_target_allocation = weights_to_discrete_shares(
        weights=target_weights,
        prices=px,
        notional=notional,
        min_weight=float(min_weight),
        min_units_equity=float(min_units_equity),
        min_units_crypto=float(min_units_crypto),
        min_units_weight_thr=float(min_units_weight_thr),
        crypto_decimals=int(crypto_decimals),
        nearest_step_remaining_frac=float(nearest_step_remaining_frac),
        max_topup_iters=int(max_topup_iters),
        topup_chunk_max_steps=int(topup_chunk_max_steps),
    )

    full_target_weights = {
        str(k).upper().strip(): float(v)
        for k, v in (full_target_allocation.realized_weights or {}).items()
        if str(k).upper().strip() != "CASH" and abs(float(v)) > 1e-12
    }

    total_turnover = _weight_turnover(current_weights, full_target_weights)

    if total_turnover <= 1e-12:
        trades, blocked_trades = allocation_to_trade_deltas(
            current_shares=current,
            target_allocation=full_target_allocation,
            prices=px,
            notional=notional,
            cfg=cfg,
        )

        return TransitionExecutionPlan(
            as_of=str(as_of),
            recommendation="NO_TRADE",
            reason="Current shares already match target allocation.",
            source=str(source),
            notional=float(notional),
            total_turnover=0.0,
            daily_turnover_used=0.0,
            blocked_turnover=0.0,
            target_allocation=full_target_allocation,
            trades=trades,
            blocked_trades=blocked_trades,
            config=cfg,
            diagnostics={
                "current_weights": current_weights,
                "target_weights_full": full_target_weights,
                "partial_transition": False,
            },
        )

    if total_turnover > float(cfg.max_total_turnover):
        return TransitionExecutionPlan(
            as_of=str(as_of),
            recommendation="TRADE_BLOCKED",
            reason=(
                f"Required turnover {total_turnover:.2%} exceeds "
                f"max_total_turnover {float(cfg.max_total_turnover):.2%}."
            ),
            source=str(source),
            notional=float(notional),
            total_turnover=float(total_turnover),
            daily_turnover_used=0.0,
            blocked_turnover=float(total_turnover),
            target_allocation=full_target_allocation,
            trades=[],
            blocked_trades=[],
            config=cfg,
            diagnostics={
                "blocked_reason": "max_total_turnover_exceeded",
                "current_weights": current_weights,
                "target_weights_full": full_target_weights,
                "partial_transition": False,
            },
        )

    target_weights_for_today = dict(target_weights)
    daily_turnover_used = float(total_turnover)
    blocked_turnover = 0.0
    partial_transition = False

    if total_turnover > float(cfg.max_daily_turnover):
        if not bool(cfg.allow_partial_transition):
            return TransitionExecutionPlan(
                as_of=str(as_of),
                recommendation="TRADE_BLOCKED",
                reason=(
                    f"Required turnover {total_turnover:.2%} exceeds "
                    f"max_daily_turnover {float(cfg.max_daily_turnover):.2%} "
                    "and partial transition is disabled."
                ),
                source=str(source),
                notional=float(notional),
                total_turnover=float(total_turnover),
                daily_turnover_used=0.0,
                blocked_turnover=float(total_turnover),
                target_allocation=full_target_allocation,
                trades=[],
                blocked_trades=[],
                config=cfg,
                diagnostics={
                    "blocked_reason": "max_daily_turnover_exceeded",
                    "current_weights": current_weights,
                    "target_weights_full": full_target_weights,
                    "partial_transition": False,
                },
            )

        target_weights_for_today, daily_turnover_used, blocked_turnover = _scale_target_weights_to_daily_turnover(
            current_weights=current_weights,
            target_weights=full_target_weights,
            max_daily_turnover=float(cfg.max_daily_turnover),
        )
        partial_transition = True

    target_allocation = weights_to_discrete_shares(
        weights=target_weights_for_today,
        prices=px,
        notional=notional,
        min_weight=float(min_weight),
        min_units_equity=float(min_units_equity),
        min_units_crypto=float(min_units_crypto),
        min_units_weight_thr=float(min_units_weight_thr),
        crypto_decimals=int(crypto_decimals),
        nearest_step_remaining_frac=float(nearest_step_remaining_frac),
        max_topup_iters=int(max_topup_iters),
        topup_chunk_max_steps=int(topup_chunk_max_steps),
    )

    trades, blocked_trades = allocation_to_trade_deltas(
        current_shares=current,
        target_allocation=target_allocation,
        prices=px,
        notional=notional,
        cfg=cfg,
    )

    if not trades:
        return TransitionExecutionPlan(
            as_of=str(as_of),
            recommendation="NO_TRADE",
            reason="All computed trade deltas are below execution thresholds.",
            source=str(source),
            notional=float(notional),
            total_turnover=float(total_turnover),
            daily_turnover_used=float(daily_turnover_used),
            blocked_turnover=float(blocked_turnover),
            target_allocation=target_allocation,
            trades=[],
            blocked_trades=blocked_trades,
            config=cfg,
            diagnostics={
                "current_weights": current_weights,
                "target_weights_full": full_target_weights,
                "target_weights_for_today": target_weights_for_today,
                "partial_transition": bool(partial_transition),
                "trade_count": 0,
                "blocked_trade_count": int(len(blocked_trades)),
            },
        )

    transition_notional = float(sum(abs(t.delta_value) for t in trades))

    return TransitionExecutionPlan(
        as_of=str(as_of),
        recommendation="TRADE_RECOMMENDED",
        reason=(
            f"{len(trades)} trade recommendation(s), "
            f"transition_notional={transition_notional:.2f}, "
            f"daily_turnover={float(daily_turnover_used):.2%}."
        ),
        source=str(source),
        notional=float(notional),
        total_turnover=float(total_turnover),
        daily_turnover_used=float(daily_turnover_used),
        blocked_turnover=float(blocked_turnover),
        target_allocation=target_allocation,
        trades=trades,
        blocked_trades=blocked_trades,
        config=cfg,
        diagnostics={
            "current_weights": current_weights,
            "target_weights_full": full_target_weights,
            "target_weights_for_today": target_weights_for_today,
            "partial_transition": bool(partial_transition),
            "trade_count": int(len(trades)),
            "blocked_trade_count": int(len(blocked_trades)),
            "transition_notional": float(transition_notional),
        },
    )