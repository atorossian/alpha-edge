from __future__ import annotations

from alpha_edge.core.schemas import (
    ShadowPortfolioAssessment,
    ShadowPortfolioConfig,
    ShadowPortfolioState,
)


def weight_turnover(
    current: dict[str, float],
    target: dict[str, float],
) -> float:
    keys = set(current.keys()) | set(target.keys())
    return float(
        0.5
        * sum(
            abs(float(target.get(k, 0.0)) - float(current.get(k, 0.0)))
            for k in keys
        )
    )


def delta_weights(
    current: dict[str, float],
    target: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(current.keys()) | set(target.keys()))
    return {
        k: float(float(target.get(k, 0.0)) - float(current.get(k, 0.0)))
        for k in keys
        if abs(float(target.get(k, 0.0)) - float(current.get(k, 0.0))) > 1e-10
    }


def assess_shadow_portfolio(
    *,
    state: ShadowPortfolioState,
    cfg: ShadowPortfolioConfig,
) -> ShadowPortfolioAssessment:
    diagnostics: dict[str, object] = {}

    health_advantage = state.health_advantage
    score_advantage = state.score_advantage
    turnover = state.turnover

    reasons: list[str] = []

    health_ok = (
        health_advantage is not None
        and float(health_advantage) >= float(cfg.min_health_advantage)
    )

    score_ok = (
        score_advantage is not None
        and float(score_advantage) >= float(cfg.min_score_advantage)
    )

    turnover_ok = (
        turnover is not None
        and float(turnover) <= float(cfg.max_turnover_to_accept)
    )

    immediate_health = (
        health_advantage is not None
        and float(health_advantage) >= float(cfg.immediate_accept_health_advantage)
    )

    immediate_score = (
        score_advantage is not None
        and float(score_advantage) >= float(cfg.immediate_accept_score_advantage)
    )

    dominates_today = bool((health_ok or score_ok) and turnover_ok)

    days_dominating = int(state.days_dominating)
    if dominates_today:
        days_dominating += 1

    diagnostics["checks"] = {
        "health_ok": bool(health_ok),
        "score_ok": bool(score_ok),
        "turnover_ok": bool(turnover_ok),
        "immediate_health": bool(immediate_health),
        "immediate_score": bool(immediate_score),
        "dominates_today": bool(dominates_today),
        "days_dominating_after_today": int(days_dominating),
    }

    if not turnover_ok:
        reasons.append(
            f"turnover {turnover:.2%} exceeds max {float(cfg.max_turnover_to_accept):.2%}"
            if turnover is not None
            else "turnover unavailable"
        )

    if not health_ok and not score_ok:
        reasons.append(
            "shadow portfolio does not clear minimum health or score advantage"
        )

    updated_state = ShadowPortfolioState(
        shadow_id=state.shadow_id,
        as_of=state.as_of,
        source_run_id=state.source_run_id,
        source_run_key=state.source_run_key,
        status=state.status,
        current_health_score=state.current_health_score,
        shadow_health_score=state.shadow_health_score,
        health_advantage=state.health_advantage,
        current_score=state.current_score,
        shadow_score=state.shadow_score,
        score_advantage=state.score_advantage,
        turnover=state.turnover,
        days_active=int(state.days_active),
        days_dominating=int(days_dominating),
        current_weights=dict(state.current_weights or {}),
        shadow_weights=dict(state.shadow_weights or {}),
        delta_weights=dict(state.delta_weights or {}),
        diagnostics=dict(state.diagnostics or {}),
    )

    if turnover_ok and (immediate_health or immediate_score):
        updated_state.status = "accepted"

        return ShadowPortfolioAssessment(
            as_of=state.as_of,
            recommendation="SHADOW_ACCEPTED",
            reason=(
                "Shadow portfolio accepted immediately because advantage is large "
                "and turnover is within limit."
            ),
            state=updated_state,
            config=cfg,
            diagnostics=diagnostics,
        )

    if dominates_today and days_dominating >= int(cfg.confirmation_days):
        updated_state.status = "accepted"

        return ShadowPortfolioAssessment(
            as_of=state.as_of,
            recommendation="SHADOW_ACCEPTED",
            reason=(
                f"Shadow portfolio accepted after {days_dominating} "
                "dominating confirmation days."
            ),
            state=updated_state,
            config=cfg,
            diagnostics=diagnostics,
        )

    if dominates_today:
        updated_state.status = "active"

        return ShadowPortfolioAssessment(
            as_of=state.as_of,
            recommendation="SHADOW_ACTIVE",
            reason=(
                f"Shadow portfolio is better but needs confirmation. "
                f"days_dominating={days_dominating}/{int(cfg.confirmation_days)}."
            ),
            state=updated_state,
            config=cfg,
            diagnostics=diagnostics,
        )

    updated_state.status = "rejected"

    return ShadowPortfolioAssessment(
        as_of=state.as_of,
        recommendation="SHADOW_REJECTED",
        reason="; ".join(reasons) if reasons else "Shadow portfolio rejected.",
        state=updated_state,
        config=cfg,
        diagnostics=diagnostics,
    )
