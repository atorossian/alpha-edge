# regime_filter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RegimeFilterState:
    last_date: str | None = None
    chosen_label: str | None = None
    days_in_regime: int = 0
    probs_smoothed: Dict[str, float] | None = None


def _normalize(p: Dict[str, float]) -> Dict[str, float]:
    items = {str(k): max(0.0, float(v)) for k, v in (p or {}).items()}
    s = sum(items.values())
    if s <= 0:
        n = len(items) if items else 1
        return {k: 1.0 / n for k in items.keys()}
    return {k: v / s for k, v in items.items()}


def ema_probs(prev: Optional[Dict[str, float]], cur: Dict[str, float], alpha: float) -> Dict[str, float]:
    cur = _normalize(cur)
    if not prev:
        return cur
    prev = _normalize(prev)
    keys = sorted(set(prev) | set(cur))
    out = {k: alpha * cur.get(k, 0.0) + (1.0 - alpha) * prev.get(k, 0.0) for k in keys}
    return _normalize(out)


def choose_label_hysteresis(
    *,
    probs: Dict[str, float],
    prev_label: Optional[str],
    days_in_regime: int,
    min_hold_days: int,
    min_prob_to_switch: float,
    min_margin_to_switch: float,
) -> str:
    probs = _normalize(probs)
    best_label = max(probs, key=lambda k: probs[k])
    best_p = float(probs[best_label])

    if prev_label is None or prev_label not in probs:
        return best_label

    cur_p = float(probs.get(prev_label, 0.0))

    # respect holding period
    if days_in_regime < min_hold_days:
        return prev_label

    # switch only if strong + clearly better
    if best_label != prev_label:
        if best_p >= min_prob_to_switch and (best_p - cur_p) >= min_margin_to_switch:
            return best_label

    return prev_label


def update_regime_filter(
    *,
    state: RegimeFilterState,
    as_of: str,  # YYYY-MM-DD
    probs_raw: Dict[str, float],
    alpha: float = 0.25,
    min_hold_days: int = 3,
    min_prob_to_switch: float = 0.55,
    min_margin_to_switch: float = 0.10,
) -> RegimeFilterState:
    same_day = (state.last_date == as_of)

    probs_sm = ema_probs(state.probs_smoothed, probs_raw, alpha=alpha)

    chosen = choose_label_hysteresis(
        probs=probs_sm,
        prev_label=state.chosen_label,
        days_in_regime=(state.days_in_regime if not same_day else max(1, state.days_in_regime)),
        min_hold_days=min_hold_days,
        min_prob_to_switch=min_prob_to_switch,
        min_margin_to_switch=min_margin_to_switch,
    )

    if state.chosen_label is None:
        days = 1
    elif chosen == state.chosen_label:
        days = state.days_in_regime if same_day else state.days_in_regime + 1
    else:
        days = 1

    return RegimeFilterState(
        last_date=as_of,
        chosen_label=chosen,
        days_in_regime=int(days),
        probs_smoothed=probs_sm,
    )
