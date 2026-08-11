# src/alpha_edge/risk/actuarial/engine.py
from __future__ import annotations

from typing import Optional

import numpy as np

from alpha_edge.core.schemas import ActuarialRiskConfig, ActuarialRiskResult
from alpha_edge.risk.actuarial.path_metrics import (
    calculate_cvar_max_drawdown,
    calculate_drawdown_breach_probability,
    calculate_goal_probability,
    calculate_max_drawdowns,
    calculate_probability_goal_before_ruin,
    calculate_recovery_metrics,
    calculate_ruin_probability,
    calculate_survival_curve,
    validate_equity_paths,
)
from alpha_edge.risk.actuarial.solvency import evaluate_capital_adequacy

def _mean_or_none(values: np.ndarray) -> Optional[float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return None
    return float(np.mean(clean))


def _median_or_none(values: np.ndarray) -> Optional[float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return None
    return float(np.median(clean))


def _assign_preliminary_risk_grade(
    *,
    ruin_probability: Optional[float],
    drawdown_breach_probability: Optional[float],
    capital_target_ruin_probability: float,
    capital_target_drawdown_breach_probability: float,
) -> str:
    """
    Preliminary qualitative grade.

    This is intentionally simple for Step 2.

    Later, once the capital adequacy module exists, this should be replaced
    by a more robust scoring function.
    """
    rp = 0.0 if ruin_probability is None else float(ruin_probability)
    dp = 0.0 if drawdown_breach_probability is None else float(drawdown_breach_probability)

    if rp <= capital_target_ruin_probability * 0.50 and dp <= capital_target_drawdown_breach_probability * 0.50:
        return "A"

    if rp <= capital_target_ruin_probability and dp <= capital_target_drawdown_breach_probability:
        return "B"

    if rp <= capital_target_ruin_probability * 2.0 and dp <= capital_target_drawdown_breach_probability * 1.5:
        return "C"

    if rp <= capital_target_ruin_probability * 4.0 and dp <= capital_target_drawdown_breach_probability * 2.0:
        return "D"

    return "F"


def evaluate_actuarial_risk(
    equity_paths: object,
    *,
    config: ActuarialRiskConfig,
) -> ActuarialRiskResult:
    """
    Evaluate actuarial path-based risk metrics.

    This function does not simulate paths.
    It evaluates paths produced elsewhere, for example by portfolio search.

    Expected path shape:
        rows    = simulated paths
        columns = time steps

    Column 0 must equal config.initial_value for every path.
    """
    cfg = config.validate()

    paths = validate_equity_paths(
        equity_paths,
        horizon_days=cfg.horizon_days,
        initial_value=cfg.initial_value,
    )

    # Use only the configured horizon, including t=0.
    paths = paths[:, : cfg.horizon_days + 1]

    n_paths = int(paths.shape[0])

    warnings: list[str] = []

    ruin_threshold = cfg.ruin_threshold_value() if cfg.ruin.enabled else None

    ruin_probability = None
    expected_time_to_ruin_days = None
    median_time_to_ruin_days = None
    ruin_first_times = np.full(n_paths, np.nan, dtype=float)

    if cfg.ruin.enabled and ruin_threshold is not None:
        ruin_summary = calculate_ruin_probability(
            paths,
            ruin_threshold=ruin_threshold,
        )
        ruin_probability = ruin_summary.event_probability
        expected_time_to_ruin_days = ruin_summary.expected_time_days
        median_time_to_ruin_days = ruin_summary.median_time_days
        ruin_first_times = ruin_summary.first_hit_times

    max_drawdowns = calculate_max_drawdowns(paths)

    expected_max_drawdown = _mean_or_none(max_drawdowns)
    median_max_drawdown = _median_or_none(max_drawdowns)
    cvar_max_drawdown_95 = calculate_cvar_max_drawdown(max_drawdowns, alpha=0.95)

    drawdown_breach_probability = None
    if cfg.drawdown.enabled:
        drawdown_breach_probability = calculate_drawdown_breach_probability(
            paths,
            drawdown_limit_pct=cfg.drawdown.drawdown_limit_pct,
        )

    goal_probability = None
    median_time_to_goal_days = None
    probability_goal_before_ruin = None
    goal_first_times = np.full(n_paths, np.nan, dtype=float)

    if cfg.goal.enabled and cfg.goal.goal_value is not None:
        goal_summary = calculate_goal_probability(
            paths,
            goal_value=cfg.goal.goal_value,
        )
        goal_probability = goal_summary.event_probability
        median_time_to_goal_days = goal_summary.median_time_days
        goal_first_times = goal_summary.first_hit_times

        if cfg.ruin.enabled:
            probability_goal_before_ruin = calculate_probability_goal_before_ruin(
                goal_first_times=goal_first_times,
                ruin_first_times=ruin_first_times,
            )

    recovery_probability = None
    median_recovery_time_days = None

    if cfg.recovery.enabled and cfg.drawdown.enabled:
        recovery = calculate_recovery_metrics(
            paths,
            drawdown_limit_pct=cfg.drawdown.drawdown_limit_pct,
            recovery_level=cfg.recovery.recovery_level,
        )
        recovery_probability = recovery.recovery_probability
        median_recovery_time_days = recovery.median_recovery_time_days
    
    capital_required = None
    capital_buffer_gap = None
    solvency_ratio = None
    safe_leverage_estimate = None

    if cfg.capital_adequacy.enabled:
        capital = evaluate_capital_adequacy(
            paths,
            initial_value=cfg.initial_value,
            config=cfg.capital_adequacy,
        )

        capital_required = capital.capital_required
        capital_buffer_gap = capital.capital_buffer_gap
        solvency_ratio = capital.solvency_ratio
        safe_leverage_estimate = capital.safe_leverage_estimate
        warnings.extend(capital.warnings)


    survival_curve = []
    if cfg.survival.enabled and cfg.ruin.enabled:
        survival_curve = calculate_survival_curve(
            first_event_times=ruin_first_times,
            horizons_days=cfg.survival.horizons_days,
        )

    risk_grade = _assign_preliminary_risk_grade(
        ruin_probability=ruin_probability,
        drawdown_breach_probability=drawdown_breach_probability,
        capital_target_ruin_probability=cfg.capital_adequacy.target_ruin_probability,
        capital_target_drawdown_breach_probability=cfg.capital_adequacy.target_drawdown_breach_probability,
    )

    if ruin_probability is not None and ruin_probability > cfg.capital_adequacy.target_ruin_probability:
        warnings.append(
            "Ruin probability exceeds target_ruin_probability."
        )

    if (
        drawdown_breach_probability is not None
        and drawdown_breach_probability > cfg.capital_adequacy.target_drawdown_breach_probability
    ):
        warnings.append(
            "Drawdown breach probability exceeds target_drawdown_breach_probability."
        )

    result = ActuarialRiskResult(
        initial_value=float(cfg.initial_value),
        horizon_days=int(cfg.horizon_days),
        n_paths=n_paths,
        ruin_threshold=ruin_threshold,
        ruin_probability=ruin_probability,
        expected_time_to_ruin_days=expected_time_to_ruin_days,
        median_time_to_ruin_days=median_time_to_ruin_days,
        drawdown_limit_pct=cfg.drawdown.drawdown_limit_pct if cfg.drawdown.enabled else None,
        drawdown_breach_probability=drawdown_breach_probability,
        expected_max_drawdown=expected_max_drawdown,
        median_max_drawdown=median_max_drawdown,
        cvar_max_drawdown_95=cvar_max_drawdown_95,
        goal_value=cfg.goal.goal_value if cfg.goal.enabled else None,
        goal_probability=goal_probability,
        median_time_to_goal_days=median_time_to_goal_days,
        probability_goal_before_ruin=probability_goal_before_ruin,
        recovery_probability=recovery_probability,
        median_recovery_time_days=median_recovery_time_days,
        capital_required=capital_required,
        capital_buffer_gap=capital_buffer_gap,
        solvency_ratio=solvency_ratio,
        safe_leverage_estimate=safe_leverage_estimate,
        survival_curve=survival_curve,
        risk_grade=risk_grade,  # type: ignore[arg-type]
        warnings=warnings,
        metadata={
            "module": "alpha_edge.risk.actuarial",
            "version": "v1_path_metrics_and_capital_adequacy",
            "notes": [
                "Step 2 implements path-based actuarial metrics.",
                "Step 3 adds capital adequacy and safe leverage estimates.",
            ],
            "config": cfg.to_dict(),
        },
    )
    
    return result.validate()