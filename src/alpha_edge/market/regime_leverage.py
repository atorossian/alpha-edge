# regime_leverage.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from alpha_edge.market.regime_filter import RegimeFilterState, update_regime_filter


@dataclass(frozen=True)
class LeverageBand:
    min_lev: float
    max_lev: float


REGIME_TO_LEVERAGE: Dict[str, LeverageBand] = {
    "STRESS_BEAR": LeverageBand(1.0, 3.0),
    "CHOPPY_BEAR": LeverageBand(3.0, 5.0),
    "CHOPPY_BULL": LeverageBand(5.0, 7.0),
    "CALM_BULL":   LeverageBand(7.0, 12.0),
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def leverage_from_label(label: str, *, risk_appetite: float = 0.6) -> float:
    b = REGIME_TO_LEVERAGE.get(label)
    if b is None:
        return 1.0
    ra = _clamp(float(risk_appetite), 0.0, 1.0)
    return b.min_lev + ra * (b.max_lev - b.min_lev)


def _normalize_probs(p_label: dict) -> Dict[str, float]:
    items = [(str(k), float(v)) for k, v in (p_label or {}).items() if v is not None]
    s = sum(v for _, v in items)
    if s <= 0:
        return {}
    return {k: v / s for k, v in items}


def leverage_from_hmm(
    hmm_payload: dict,
    *,
    default: float = 1.0,
    risk_appetite: float = 0.6,
    low_confidence_floor: float = 0.2,
    hard_cap: float | None = None,
    # ---- NEW: stability layer ----
    filter_state: RegimeFilterState | None = None,
    as_of: str | None = None,   # YYYY-MM-DD (IMPORTANT)
    filter_alpha: float = 0.20,
    min_hold_days: int = 3,
    min_prob_to_switch: float = 0.60,
    min_margin_to_switch: float = 0.12,
) -> dict:
    """
    Returns dict with:
      leverage, mode, chosen_label, confidence, band,
      p_label_used, filter_state (if filtering enabled)
    """
    if not hmm_payload:
        return {
            "leverage": default,
            "mode": "none",
            "chosen_label": None,
            "confidence": 0.0,
            "band": None,
            "p_label_used": None,
            "filter_state": None,
        }

    p_label_raw = hmm_payload.get("p_label_today") or {}
    if not isinstance(p_label_raw, dict) or not p_label_raw:
        return {
            "leverage": default,
            "mode": "none",
            "chosen_label": None,
            "confidence": 0.0,
            "band": None,
            "p_label_used": None,
            "filter_state": None,
        }

    # normalize raw probs
    p_used = _normalize_probs(p_label_raw)
    if not p_used:
        return {
            "leverage": default,
            "mode": "none",
            "chosen_label": None,
            "confidence": 0.0,
            "band": None,
            "p_label_used": None,
            "filter_state": None,
        }

    # ---- NEW: apply smoothing + hysteresis if state provided ----
    new_state = None
    if filter_state is not None and as_of is not None:
        new_state = update_regime_filter(
            state=filter_state,
            as_of=str(as_of),
            probs_raw=p_used,
            alpha=float(filter_alpha),
            min_hold_days=int(min_hold_days),
            min_prob_to_switch=float(min_prob_to_switch),
            min_margin_to_switch=float(min_margin_to_switch),
        )
        if new_state.probs_smoothed:
            p_used = dict(new_state.probs_smoothed)

        chosen_label = new_state.chosen_label
        mode = "filtered"
    else:
        # fallback to your old behavior:
        chosen_label = hmm_payload.get("label_commit")
        mode = "commit" if isinstance(chosen_label, str) else "expected"

    # compute confidence proxy
    conf = 0.0
    if chosen_label and chosen_label in p_used:
        conf = float(p_used.get(chosen_label, 0.0))
    else:
        conf = float(max(p_used.values()))

    conf_scaled = _clamp((conf - float(low_confidence_floor)) / (1.0 - float(low_confidence_floor)), 0.0, 1.0)
    ra = _clamp(float(risk_appetite) * conf_scaled, 0.0, 1.0)

    # ---- leverage decision ----
    if chosen_label in REGIME_TO_LEVERAGE:
        b = REGIME_TO_LEVERAGE[chosen_label]
        lev = leverage_from_label(chosen_label, risk_appetite=ra)
        band = (b.min_lev, b.max_lev)
    else:
        # expected leverage fallback
        exp_min = 0.0
        exp_max = 0.0
        for label, p in p_used.items():
            b = REGIME_TO_LEVERAGE.get(label, LeverageBand(default, default))
            exp_min += float(p) * b.min_lev
            exp_max += float(p) * b.max_lev
        lev = exp_min + ra * (exp_max - exp_min)
        band = (float(exp_min), float(exp_max))
        if not chosen_label:
            chosen_label = max(p_used.items(), key=lambda kv: kv[1])[0]

    if hard_cap is not None:
        lev = min(float(lev), float(hard_cap))

    # return full payload + filter state for persistence
    return {
        "leverage": float(lev),
        "mode": mode,
        "chosen_label": chosen_label,
        "confidence": float(conf),
        "band": band,
        "p_label_used": p_used,
        "filter_state": {
            "last_date": new_state.last_date,
            "chosen_label": new_state.chosen_label,
            "days_in_regime": new_state.days_in_regime,
            "probs_smoothed": new_state.probs_smoothed,
        } if new_state is not None else None,
    }
