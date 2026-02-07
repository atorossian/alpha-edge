# alpha_edge/portfolio/take_profit.py
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TakeProfitConfig:
    # hysteresis thresholds on anchor return
    enter_profit: float = 0.10
    exit_profit: float = 0.07

    # safety gates
    max_dd: float = 0.05
    min_sharpe: float = 0.75

    # multiplier curve
    max_harvest: float = 0.25   # at most reduce leverage by 25%
    k: float = 8.0              # curve aggressiveness
    m_min: float = 0.60         # never go below 60% of base

    # churn control
    cooldown_days: int = 10

    # (future) stability gate
    use_stability: bool = False
    max_stability: float | None = None


@dataclass
class TakeProfitState:
    # anchor for ratcheting
    anchor_date: str | None = None
    anchor_equity: float | None = None

    # high-water mark tracking (persisted)
    hwm_equity: float | None = None

    # mode + cooldown
    harvest_mode: bool = False
    last_harvest_date: str | None = None

    # last applied multiplier (for “only harvest when decreasing”)
    current_multiplier: float = 1.0


@dataclass(frozen=True)
class TakeProfitResult:
    m_star: float
    do_harvest: bool
    reasons: list[str]
    sharpe: float | None
    dd: float | None
    r_anchor: float | None
    next_state: TakeProfitState


def _days_between(a: str | None, b: str) -> int | None:
    if not a:
        return None
    try:
        da = pd.Timestamp(a).normalize()
        db = pd.Timestamp(b).normalize()
        return int((db - da).days)
    except Exception:
        return None


def take_profit_policy(
    *,
    cfg: TakeProfitConfig,
    state: TakeProfitState | None,
    as_of: str,                 # "YYYY-MM-DD" market date
    equity: float,
    sharpe_value: float | None, # use health.sharpe (already computed)
    stability: float | None = None,
) -> TakeProfitResult:
    eq = float(equity)
    if not np.isfinite(eq) or eq <= 0:
        st = state or TakeProfitState()
        return TakeProfitResult(
            m_star=1.0,
            do_harvest=False,
            reasons=["invalid_equity"],
            sharpe=None,
            dd=None,
            r_anchor=None,
            next_state=st,
        )

    st = state or TakeProfitState()

    # initialize anchor/hwm if missing
    if st.anchor_equity is None or (not np.isfinite(float(st.anchor_equity))) or float(st.anchor_equity) <= 0:
        st.anchor_equity = eq
        st.anchor_date = as_of
    if st.hwm_equity is None or (not np.isfinite(float(st.hwm_equity))) or float(st.hwm_equity) <= 0:
        st.hwm_equity = eq

    # update HWM
    hwm = float(max(float(st.hwm_equity), eq))
    st.hwm_equity = hwm

    # compute metrics
    dd = 1.0 - (eq / hwm) if hwm > 0 else None
    r_anchor = (eq / float(st.anchor_equity)) - 1.0 if float(st.anchor_equity) > 0 else None

    sharpe = None if sharpe_value is None else float(sharpe_value)
    if sharpe is not None and (not np.isfinite(sharpe)):
        sharpe = None

    reasons: list[str] = []

    # safety gates
    safe = True
    if dd is None or (not np.isfinite(dd)):
        safe = False
        reasons.append("dd_nan")
    elif dd > float(cfg.max_dd):
        safe = False
        reasons.append(f"dd_{dd:.3f}>{float(cfg.max_dd):.3f}")

    if sharpe is None:
        safe = False
        reasons.append("sharpe_missing")
    elif float(sharpe) < float(cfg.min_sharpe):
        safe = False
        reasons.append(f"sharpe_{float(sharpe):.2f}<{float(cfg.min_sharpe):.2f}")

    if cfg.use_stability:
        if stability is None or (not np.isfinite(float(stability))):
            safe = False
            reasons.append("stability_nan")
        elif cfg.max_stability is not None and float(stability) > float(cfg.max_stability):
            safe = False
            reasons.append("stability_too_high")

    # cooldown
    cooldown_ok = True
    d_since = _days_between(st.last_harvest_date, as_of)
    if d_since is not None and int(cfg.cooldown_days) > 0 and d_since < int(cfg.cooldown_days):
        cooldown_ok = False
        reasons.append(f"cooldown_{d_since}d<{int(cfg.cooldown_days)}d")

    # hysteresis threshold
    if r_anchor is None or (not np.isfinite(float(r_anchor))):
        eligible_profit = False
        reasons.append("r_anchor_nan")
    else:
        thr = float(cfg.exit_profit) if st.harvest_mode else float(cfg.enter_profit)
        eligible_profit = float(r_anchor) >= thr
        if not eligible_profit:
            reasons.append(f"r_anchor_{float(r_anchor):+.3f}<thr_{thr:+.3f}")

    eligible = bool(eligible_profit) and bool(safe)

    # compute multiplier target
    m_star = 1.0
    if eligible:
        thr = float(cfg.exit_profit) if st.harvest_mode else float(cfg.enter_profit)
        x = max(0.0, float(r_anchor) - thr)
        g = 1.0 - math.exp(-float(cfg.k) * x)
        m_star = 1.0 - float(cfg.max_harvest) * g
        m_star = max(float(cfg.m_min), min(1.0, float(m_star)))
        reasons.append(f"eligible_x={x:.3f}_g={g:.3f}_m={m_star:.3f}")

    # decide whether to actually harvest now (only if decreasing)
    eps = 1e-6
    do_harvest = bool(eligible) and bool(cooldown_ok) and (m_star < float(st.current_multiplier) - eps)

    # next state update (ratchet on harvest)
    next_state = TakeProfitState(**asdict(st))

    # manage harvest_mode
    if not safe:
        next_state.harvest_mode = False
    else:
        if r_anchor is not None and np.isfinite(float(r_anchor)) and float(r_anchor) >= float(cfg.exit_profit):
            next_state.harvest_mode = bool(st.harvest_mode)
        else:
            next_state.harvest_mode = False

    if do_harvest:
        next_state.last_harvest_date = as_of
        next_state.harvest_mode = True
        next_state.current_multiplier = float(m_star)

        # ratchet anchor to lock profits
        next_state.anchor_equity = float(eq)
        next_state.anchor_date = as_of
        reasons.append("ratchet_anchor")
    else:
        next_state.current_multiplier = float(st.current_multiplier)

    next_state.hwm_equity = float(hwm)

    return TakeProfitResult(
        m_star=float(m_star),
        do_harvest=bool(do_harvest),
        reasons=reasons,
        sharpe=(None if sharpe is None else float(sharpe)),
        dd=(None if dd is None else float(dd)),
        r_anchor=(None if r_anchor is None else float(r_anchor)),
        next_state=next_state,
    )
