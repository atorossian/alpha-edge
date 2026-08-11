# src/alpha_edge/risk/actuarial/diagnostic_report.py
from __future__ import annotations

from typing import Any, Optional

from alpha_edge.core.schemas import (
    ActuarialDiagnosticReport,
    ActuarialRiskConfig,
    ActuarialRiskResult,
)
from alpha_edge.risk.actuarial.portfolio_adapter import (
    evaluate_portfolio_search_actuarial_risk,
)


def _get_nested(data: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = data

    for key in path:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]

    return current


def _extract_config_from_result(result: ActuarialRiskResult) -> dict[str, Any]:
    config = result.metadata.get("config", {})
    return config if isinstance(config, dict) else {}


def _build_risk_flags(result: ActuarialRiskResult) -> list[str]:
    """
    Build deterministic diagnostic flags from actuarial results.

    These are not portfolio-search quarantine rules yet.
    They are report-level warnings that help explain the result.
    """
    flags: list[str] = []

    config = _extract_config_from_result(result)

    target_ruin_probability = _get_nested(
        config,
        ["capital_adequacy", "target_ruin_probability"],
    )
    target_drawdown_breach_probability = _get_nested(
        config,
        ["capital_adequacy", "target_drawdown_breach_probability"],
    )
    current_leverage = _get_nested(
        config,
        ["capital_adequacy", "current_leverage"],
    )
    drawdown_limit_pct = _get_nested(
        config,
        ["drawdown", "drawdown_limit_pct"],
    )

    if (
        result.ruin_probability is not None
        and target_ruin_probability is not None
        and result.ruin_probability > float(target_ruin_probability)
    ):
        flags.append("ruin_probability_above_target")

    if (
        result.drawdown_breach_probability is not None
        and target_drawdown_breach_probability is not None
        and result.drawdown_breach_probability > float(target_drawdown_breach_probability)
    ):
        flags.append("drawdown_breach_probability_above_target")

    if result.capital_buffer_gap is not None and result.capital_buffer_gap < 0.0:
        flags.append("negative_capital_buffer_gap")

    if result.solvency_ratio is not None and result.solvency_ratio < 1.0:
        flags.append("solvency_ratio_below_one")

    if (
        result.safe_leverage_estimate is not None
        and current_leverage is not None
        and result.safe_leverage_estimate < float(current_leverage)
    ):
        flags.append("safe_leverage_below_current_leverage")

    if (
        result.cvar_max_drawdown_95 is not None
        and drawdown_limit_pct is not None
        and result.cvar_max_drawdown_95 <= -float(drawdown_limit_pct)
    ):
        flags.append("tail_drawdown_cvar_breaches_limit")

    return flags


def _assign_diagnostic_verdict(
    *,
    risk_flags: list[str],
    warnings: list[str],
) -> str:
    """
    Convert risk flags into a simple diagnostic verdict.

    This is not a trading decision.
    This is only a reporting classification.
    """
    hard_fail_flags = {
        "ruin_probability_above_target",
        "negative_capital_buffer_gap",
        "solvency_ratio_below_one",
    }

    if any(flag in hard_fail_flags for flag in risk_flags):
        return "fail"

    if risk_flags or warnings:
        return "warn"

    return "pass"


def _headline_metrics(result: ActuarialRiskResult) -> dict[str, Optional[float]]:
    """
    Small set of metrics that should be easy to display in reports.
    """
    return {
        "ruin_probability": result.ruin_probability,
        "drawdown_breach_probability": result.drawdown_breach_probability,
        "goal_probability": result.goal_probability,
        "probability_goal_before_ruin": result.probability_goal_before_ruin,
        "solvency_ratio": result.solvency_ratio,
        "capital_buffer_gap": result.capital_buffer_gap,
        "safe_leverage_estimate": result.safe_leverage_estimate,
    }


def _detail_metrics(result: ActuarialRiskResult) -> dict[str, Optional[float]]:
    """
    Wider set of metrics for diagnostics and future warehouse/reporting use.
    """
    return {
        "initial_value": result.initial_value,
        "horizon_days": float(result.horizon_days),
        "n_paths": float(result.n_paths),
        "ruin_threshold": result.ruin_threshold,
        "expected_time_to_ruin_days": result.expected_time_to_ruin_days,
        "median_time_to_ruin_days": result.median_time_to_ruin_days,
        "drawdown_limit_pct": result.drawdown_limit_pct,
        "expected_max_drawdown": result.expected_max_drawdown,
        "median_max_drawdown": result.median_max_drawdown,
        "cvar_max_drawdown_95": result.cvar_max_drawdown_95,
        "goal_value": result.goal_value,
        "median_time_to_goal_days": result.median_time_to_goal_days,
        "recovery_probability": result.recovery_probability,
        "median_recovery_time_days": result.median_recovery_time_days,
        "capital_required": result.capital_required,
        "capital_buffer_gap": result.capital_buffer_gap,
        "solvency_ratio": result.solvency_ratio,
        "safe_leverage_estimate": result.safe_leverage_estimate,
    }


def build_actuarial_diagnostic_report(
    result: ActuarialRiskResult,
    *,
    portfolio_id: Optional[str] = None,
    run_id: Optional[str] = None,
    source: str = "actuarial_risk",
    metadata: Optional[dict[str, Any]] = None,
) -> ActuarialDiagnosticReport:
    """
    Build a diagnostic report from an ActuarialRiskResult.

    This function does not recalculate risk.
    It only summarizes and classifies an already-computed result.
    """
    risk_flags = _build_risk_flags(result)
    warnings = list(result.warnings)

    verdict = _assign_diagnostic_verdict(
        risk_flags=risk_flags,
        warnings=warnings,
    )

    report_metadata: dict[str, Any] = {
        "module": "alpha_edge.risk.actuarial",
        "version": "v1_diagnostic_report",
    }
    if metadata:
        report_metadata.update(metadata)

    integration = result.metadata.get("integration", {})
    if isinstance(integration, dict):
        portfolio_id = portfolio_id or integration.get("portfolio_id")
        run_id = run_id or integration.get("run_id")

    report = ActuarialDiagnosticReport(
        portfolio_id=portfolio_id,
        run_id=run_id,
        source=source,
        verdict=verdict,  # type: ignore[arg-type]
        risk_grade=result.risk_grade,
        risk_flags=risk_flags,
        warnings=warnings,
        headline_metrics=_headline_metrics(result),
        detail_metrics=_detail_metrics(result),
        result=result.to_dict(),
        metadata=report_metadata,
    )

    return report.validate()


def evaluate_portfolio_search_actuarial_diagnostic(
    portfolio_result: object,
    *,
    config: ActuarialRiskConfig,
    equity_paths_key: Optional[str] = None,
    portfolio_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> ActuarialDiagnosticReport:
    """
    Evaluate one portfolio-search result and return a diagnostic report.

    This is the safest portfolio-search integration point.

    It does not:
      - change candidate score,
      - quarantine candidates,
      - mutate portfolio search output,
      - write to S3.
    """
    result = evaluate_portfolio_search_actuarial_risk(
        portfolio_result,
        config=config,
        equity_paths_key=equity_paths_key,
        portfolio_id=portfolio_id,
        run_id=run_id,
    )

    return build_actuarial_diagnostic_report(
        result,
        portfolio_id=portfolio_id,
        run_id=run_id,
        source="portfolio_search",
        metadata={
            "integration_step": "portfolio_search_diagnostic",
        },
    )


def evaluate_many_portfolio_search_actuarial_diagnostics(
    portfolio_results: list[object],
    *,
    config: ActuarialRiskConfig,
    equity_paths_key: Optional[str] = None,
) -> list[ActuarialDiagnosticReport]:
    """
    Evaluate many portfolio-search results and return diagnostic reports.

    This is useful for offline candidate diagnostics.
    It intentionally does not rank or filter candidates.
    """
    reports: list[ActuarialDiagnosticReport] = []

    for i, portfolio_result in enumerate(portfolio_results):
        portfolio_id: Optional[str] = None
        run_id: Optional[str] = None

        if isinstance(portfolio_result, dict):
            if portfolio_result.get("portfolio_id") is not None:
                portfolio_id = str(portfolio_result["portfolio_id"])
            elif portfolio_result.get("candidate_id") is not None:
                portfolio_id = str(portfolio_result["candidate_id"])

            if portfolio_result.get("run_id") is not None:
                run_id = str(portfolio_result["run_id"])

        report = evaluate_portfolio_search_actuarial_diagnostic(
            portfolio_result,
            config=config,
            equity_paths_key=equity_paths_key,
            portfolio_id=portfolio_id or f"portfolio_index_{i}",
            run_id=run_id,
        )
        reports.append(report)

    return reports