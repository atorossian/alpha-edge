from __future__ import annotations

from alpha_edge.core.schemas import (
    CurrentPortfolioState,
    PortfolioTransitionConfig,
    TransitionAssessment,
)


_GRADE_ORDER = {
    "A": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "F": 1,
}


def _grade_below(current: str | None, minimum: str) -> bool:
    if not current:
        return True

    current_norm = str(current).upper().strip()[:1]
    minimum_norm = str(minimum).upper().strip()[:1]

    return _GRADE_ORDER.get(current_norm, 0) < _GRADE_ORDER.get(minimum_norm, 0)


def _regime_change_triggered(
    *,
    state: CurrentPortfolioState,
    cfg: PortfolioTransitionConfig,
) -> bool:
    """
    Regime change is a soft full-search trigger.

    Meaning:
      - it should run a full search,
      - it should create / refresh a shadow portfolio,
      - it should NOT automatically force immediate liquidation.
    """
    if not cfg.regime_change_requires_full_search:
        return False

    if not bool(state.regime_changed):
        return False

    if state.regime_confidence is None:
        return True

    return float(state.regime_confidence) >= float(cfg.min_regime_confidence_for_full_search)


def assess_transition(
    *,
    state: CurrentPortfolioState,
    cfg: PortfolioTransitionConfig,
) -> TransitionAssessment:
    diagnostics: dict[str, object] = {}

    triggers: list[str] = []

    if state.health_score is None:
        triggers.append("missing_health_score")

    elif state.health_score < cfg.min_health_score:
        triggers.append("health_score_below_threshold")

    if (
        state.health_score is not None
        and state.original_health_score is not None
        and state.original_health_score - state.health_score >= cfg.health_drop_trigger
    ):
        triggers.append("health_score_drop_vs_original")

    if _grade_below(state.grade, cfg.min_grade):
        triggers.append("grade_below_minimum")

    if (
        state.ruin_probability is not None
        and state.ruin_probability > cfg.max_ruin_probability
    ):
        triggers.append("ruin_probability_above_limit")

    if (
        state.max_drawdown is not None
        and abs(state.max_drawdown) > cfg.max_drawdown_limit
    ):
        triggers.append("max_drawdown_above_limit")

    if (
        state.days_since_full_search is not None
        and state.days_since_full_search >= cfg.full_search_refresh_days
    ):
        triggers.append("scheduled_full_search_refresh")

    if state.local_optimizer_failed_days >= cfg.shadow_confirmation_days:
        triggers.append("repeated_local_optimizer_failure")

    if _regime_change_triggered(state=state, cfg=cfg):
        triggers.append("market_regime_changed")

    diagnostics["triggers"] = triggers
    diagnostics["regime"] = {
        "current_regime": state.regime,
        "previous_regime": state.previous_regime,
        "regime_changed": bool(state.regime_changed),
        "regime_confidence": state.regime_confidence,
        "regime_change_requires_full_search": bool(cfg.regime_change_requires_full_search),
        "min_regime_confidence_for_full_search": float(cfg.min_regime_confidence_for_full_search),
    }

    hard_triggers = {
        "missing_health_score",
        "health_score_below_threshold",
        "grade_below_minimum",
        "ruin_probability_above_limit",
        "max_drawdown_above_limit",
    }

    soft_full_search_triggers = {
        "scheduled_full_search_refresh",
        "repeated_local_optimizer_failure",
        "market_regime_changed",
        "health_score_drop_vs_original",
    }

    if any(t in hard_triggers for t in triggers):
        return TransitionAssessment(
            as_of=state.as_of,
            recommendation="FULL_SEARCH_REQUIRED",
            reason=f"Full search required due to hard trigger(s): {', '.join(triggers)}",
            current_state=state,
            full_search_required=True,
            local_optimization_allowed=False,
            shadow_portfolio_required=True,
            delta_execution_allowed=False,
            diagnostics=diagnostics,
        )

    if any(t in soft_full_search_triggers for t in triggers):
        return TransitionAssessment(
            as_of=state.as_of,
            recommendation="SHADOW_PORTFOLIO_ACTIVE",
            reason=f"Run full search and evaluate shadow portfolio due to: {', '.join(triggers)}",
            current_state=state,
            full_search_required=True,
            local_optimization_allowed=True,
            shadow_portfolio_required=True,
            delta_execution_allowed=False,
            diagnostics=diagnostics,
        )

    return TransitionAssessment(
        as_of=state.as_of,
        recommendation="LOCAL_OPTIMIZATION_RECOMMENDED",
        reason="Portfolio is healthy; run daily local optimization and only trade if improvement clears thresholds.",
        current_state=state,
        full_search_required=False,
        local_optimization_allowed=True,
        shadow_portfolio_required=False,
        delta_execution_allowed=False,
        diagnostics=diagnostics,
    )