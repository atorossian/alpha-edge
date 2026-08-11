from __future__ import annotations

from alpha_edge.core.schemas import (
    PortfolioTransitionPlan,
    TransitionPlanConfig,
    TransitionTradeRecommendation,
)


def _clean_weights(weights: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}

    for k, v in (weights or {}).items():
        asset_id = str(k).strip()
        if not asset_id:
            continue

        value = float(v)
        if abs(value) <= 1e-12:
            continue

        out[asset_id] = float(value)

    return out


def _normalize_gross(weights: dict[str, float]) -> dict[str, float]:
    clean = _clean_weights(weights)
    gross = float(sum(abs(v) for v in clean.values()))

    if gross <= 0:
        raise ValueError("Cannot normalize empty or zero-gross weights.")

    return {k: float(v / gross) for k, v in clean.items()}


def weight_turnover(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> float:
    current = _clean_weights(current_weights)
    target = _clean_weights(target_weights)

    keys = set(current.keys()) | set(target.keys())

    return float(
        0.5
        * sum(
            abs(float(target.get(k, 0.0)) - float(current.get(k, 0.0)))
            for k in keys
        )
    )


def delta_weights(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
) -> dict[str, float]:
    current = _clean_weights(current_weights)
    target = _clean_weights(target_weights)

    keys = sorted(set(current.keys()) | set(target.keys()))

    return {
        k: float(float(target.get(k, 0.0)) - float(current.get(k, 0.0)))
        for k in keys
        if abs(float(target.get(k, 0.0)) - float(current.get(k, 0.0))) > 1e-12
    }


def _scale_delta_to_turnover_limit(
    *,
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    max_daily_turnover: float,
) -> tuple[dict[str, float], float, float]:
    """
    If the full transition requires too much turnover, move only part of the way
    toward the target portfolio.

    Example:
        full turnover = 30%
        max daily turnover = 10%
        scale = 10 / 30 = 1/3

    adjusted target = current + scale * (target - current)
    """
    full_turnover = weight_turnover(current_weights, target_weights)

    if full_turnover <= 0:
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
        if abs(v) > 1e-12
    }

    daily_turnover = weight_turnover(current_weights, adjusted)
    blocked_turnover = max(0.0, full_turnover - daily_turnover)

    return adjusted, float(daily_turnover), float(blocked_turnover)


def _trade_direction(delta_value: float) -> str:
    if delta_value > 0:
        return "BUY"
    if delta_value < 0:
        return "SELL"
    return "HOLD"


def build_transition_plan(
    *,
    as_of: str,
    source: str,
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    prices_by_asset_id: dict[str, float],
    equity: float,
    gross_notional: float | None = None,
    cfg: TransitionPlanConfig | None = None,
) -> PortfolioTransitionPlan:
    """
    Build recommendation-only transition plan.

    This function does not execute trades and does not write to any broker.
    It only converts current weights and target weights into trade deltas.
    """
    if cfg is None:
        cfg = TransitionPlanConfig()

    equity = float(equity)
    if equity <= 0:
        raise ValueError("equity must be > 0.")

    if gross_notional is None:
        gross_notional = equity

    gross_notional = float(gross_notional)
    if gross_notional <= 0:
        raise ValueError("gross_notional must be > 0.")

    current = _normalize_gross(current_weights)
    target = _normalize_gross(target_weights)

    total_turnover = weight_turnover(current, target)

    if total_turnover <= 1e-12:
        return PortfolioTransitionPlan(
            as_of=str(as_of),
            recommendation="NO_TRADE",
            reason="Current weights already match target weights.",
            source=str(source),
            equity=float(equity),
            gross_notional=float(gross_notional),
            transition_notional=0.0,
            total_turnover=0.0,
            daily_turnover_used=0.0,
            blocked_turnover=0.0,
            current_weights=current,
            target_weights=target,
            adjusted_target_weights=current,
            trades=[],
            blocked_trades=[],
            config=cfg,
            diagnostics={},
        )

    if total_turnover > float(cfg.max_total_turnover):
        return PortfolioTransitionPlan(
            as_of=str(as_of),
            recommendation="TRADE_BLOCKED",
            reason=(
                f"Required turnover {total_turnover:.2%} exceeds "
                f"max_total_turnover {float(cfg.max_total_turnover):.2%}."
            ),
            source=str(source),
            equity=float(equity),
            gross_notional=float(gross_notional),
            transition_notional=0.0,
            total_turnover=float(total_turnover),
            daily_turnover_used=0.0,
            blocked_turnover=float(total_turnover),
            current_weights=current,
            target_weights=target,
            adjusted_target_weights=current,
            trades=[],
            blocked_trades=[],
            config=cfg,
            diagnostics={
                "blocked_reason": "max_total_turnover_exceeded",
            },
        )

    if cfg.allow_partial_transition:
        adjusted_target, daily_turnover, blocked_turnover = _scale_delta_to_turnover_limit(
            current_weights=current,
            target_weights=target,
            max_daily_turnover=float(cfg.max_daily_turnover),
        )
    else:
        if total_turnover > float(cfg.max_daily_turnover):
            return PortfolioTransitionPlan(
                as_of=str(as_of),
                recommendation="TRADE_BLOCKED",
                reason=(
                    f"Required turnover {total_turnover:.2%} exceeds "
                    f"max_daily_turnover {float(cfg.max_daily_turnover):.2%} "
                    "and partial transition is disabled."
                ),
                source=str(source),
                equity=float(equity),
                gross_notional=float(gross_notional),
                transition_notional=0.0,
                total_turnover=float(total_turnover),
                daily_turnover_used=0.0,
                blocked_turnover=float(total_turnover),
                current_weights=current,
                target_weights=target,
                adjusted_target_weights=current,
                trades=[],
                blocked_trades=[],
                config=cfg,
                diagnostics={
                    "blocked_reason": "max_daily_turnover_exceeded",
                },
            )

        adjusted_target = dict(target)
        daily_turnover = float(total_turnover)
        blocked_turnover = 0.0

    trade_deltas = delta_weights(current, adjusted_target)

    trades: list[TransitionTradeRecommendation] = []
    blocked_trades: list[TransitionTradeRecommendation] = []

    transition_notional = 0.0

    for asset_id, dw in sorted(trade_deltas.items()):
        current_weight = float(current.get(asset_id, 0.0))
        target_weight = float(adjusted_target.get(asset_id, 0.0))

        current_value = float(current_weight * gross_notional)
        target_value = float(target_weight * gross_notional)
        delta_value = float(target_value - current_value)

        px_raw = prices_by_asset_id.get(asset_id)
        estimated_price = None if px_raw is None else float(px_raw)

        estimated_quantity = None
        if estimated_price is not None and estimated_price > 0:
            estimated_quantity = float(delta_value / estimated_price)

        direction = _trade_direction(delta_value)

        reason = "included"
        is_blocked = False

        if abs(delta_value) < float(cfg.min_trade_value):
            reason = (
                f"delta_value {abs(delta_value):.2f} below "
                f"min_trade_value {float(cfg.min_trade_value):.2f}"
            )
            is_blocked = True

        if abs(dw) < float(cfg.min_trade_weight):
            reason = (
                f"delta_weight {abs(dw):.6f} below "
                f"min_trade_weight {float(cfg.min_trade_weight):.6f}"
            )
            is_blocked = True

        trade = TransitionTradeRecommendation(
            asset_id=str(asset_id),
            direction=direction,
            current_weight=float(current_weight),
            target_weight=float(target_weight),
            delta_weight=float(dw),
            current_value=float(current_value),
            target_value=float(target_value),
            delta_value=float(delta_value),
            estimated_price=estimated_price,
            estimated_quantity=estimated_quantity,
            reason=reason,
        )

        if is_blocked:
            blocked_trades.append(trade)
        else:
            trades.append(trade)
            transition_notional += abs(float(delta_value))

    if not trades:
        return PortfolioTransitionPlan(
            as_of=str(as_of),
            recommendation="NO_TRADE",
            reason="All calculated trade deltas are below minimum trade thresholds.",
            source=str(source),
            equity=float(equity),
            gross_notional=float(gross_notional),
            transition_notional=0.0,
            total_turnover=float(total_turnover),
            daily_turnover_used=float(daily_turnover),
            blocked_turnover=float(blocked_turnover),
            current_weights=current,
            target_weights=target,
            adjusted_target_weights=adjusted_target,
            trades=[],
            blocked_trades=blocked_trades,
            config=cfg,
            diagnostics={
                "blocked_trade_count": int(len(blocked_trades)),
            },
        )

    return PortfolioTransitionPlan(
        as_of=str(as_of),
        recommendation="TRADE_RECOMMENDED",
        reason=(
            f"{len(trades)} trade recommendation(s), "
            f"daily turnover {daily_turnover:.2%}, "
            f"transition notional {transition_notional:.2f}."
        ),
        source=str(source),
        equity=float(equity),
        gross_notional=float(gross_notional),
        transition_notional=float(transition_notional),
        total_turnover=float(total_turnover),
        daily_turnover_used=float(daily_turnover),
        blocked_turnover=float(blocked_turnover),
        current_weights=current,
        target_weights=target,
        adjusted_target_weights=adjusted_target,
        trades=trades,
        blocked_trades=blocked_trades,
        config=cfg,
        diagnostics={
            "trade_count": int(len(trades)),
            "blocked_trade_count": int(len(blocked_trades)),
            "partial_transition": bool(blocked_turnover > 1e-12),
        },
    )