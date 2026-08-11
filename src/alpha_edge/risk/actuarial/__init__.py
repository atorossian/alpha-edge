# src/alpha_edge/risk/actuarial/__init__.py
from __future__ import annotations

from alpha_edge.core.schemas import (
    ActuarialDiagnosticBatchReport,
    ActuarialDiagnosticReport,
    ActuarialRiskConfig,
    ActuarialRiskResult,
    CapitalAdequacyConfig,
    DrawdownBreachConfig,
    GoalConfig,
    RecoveryConfig,
    RuinConfig,
    SurvivalConfig,
    SurvivalCurvePoint,
)

from alpha_edge.risk.actuarial.diagnostic_report import (
    build_actuarial_diagnostic_report,
    evaluate_many_portfolio_search_actuarial_diagnostics,
    evaluate_portfolio_search_actuarial_diagnostic,
)
from alpha_edge.risk.actuarial.engine import evaluate_actuarial_risk
from alpha_edge.risk.actuarial.path_metrics import (
    calculate_drawdown_breach_probability,
    calculate_goal_probability,
    calculate_max_drawdowns,
    calculate_probability_goal_before_ruin,
    calculate_recovery_metrics,
    calculate_ruin_probability,
    calculate_survival_curve,
    first_hit_times,
)
from alpha_edge.risk.actuarial.portfolio_adapter import (
    build_actuarial_config_from_portfolio_context,
    evaluate_many_portfolio_search_actuarial_risks,
    evaluate_portfolio_search_actuarial_risk,
    extract_equity_paths_from_portfolio_result,
    normalize_equity_paths,
)
from alpha_edge.risk.actuarial.solvency import (
    CapitalAdequacyResult,
    calculate_capital_required_from_losses,
    calculate_path_capital_losses,
    calculate_solvent_capital_ratio,
    estimate_safe_leverage,
    evaluate_capital_adequacy,
)

from alpha_edge.risk.actuarial.diagnostic_persistence import (
    build_actuarial_diagnostic_batch_report,
    diagnostic_report_to_summary_row,
    evaluate_and_write_portfolio_search_actuarial_diagnostics,
    write_actuarial_diagnostic_batch_report,
)

from alpha_edge.risk.actuarial.formatters import (
    diagnostic_report_to_json_block,
    format_actuarial_diagnostic_report,
    format_actuarial_risk_result_compact,
)

from alpha_edge.risk.actuarial.portfolio_search_output import (
    attach_actuarial_diagnostic_to_output_payload,
    build_portfolio_search_actuarial_diagnostic_section,
    maybe_print_actuarial_diagnostic_section,
)

__all__ = [
    "ActuarialDiagnosticReport",
    "ActuarialRiskConfig",
    "ActuarialRiskResult",
    "CapitalAdequacyConfig",
    "DrawdownBreachConfig",
    "GoalConfig",
    "RecoveryConfig",
    "RuinConfig",
    "SurvivalConfig",
    "SurvivalCurvePoint",
    "CapitalAdequacyResult",
    "build_actuarial_config_from_portfolio_context",
    "build_actuarial_diagnostic_report",
    "calculate_drawdown_breach_probability",
    "calculate_goal_probability",
    "calculate_max_drawdowns",
    "calculate_probability_goal_before_ruin",
    "calculate_recovery_metrics",
    "calculate_ruin_probability",
    "calculate_survival_curve",
    "calculate_capital_required_from_losses",
    "calculate_path_capital_losses",
    "calculate_solvent_capital_ratio",
    "estimate_safe_leverage",
    "evaluate_actuarial_risk",
    "evaluate_capital_adequacy",
    "evaluate_many_portfolio_search_actuarial_diagnostics",
    "evaluate_many_portfolio_search_actuarial_risks",
    "evaluate_portfolio_search_actuarial_diagnostic",
    "evaluate_portfolio_search_actuarial_risk",
    "extract_equity_paths_from_portfolio_result",
    "first_hit_times",
    "normalize_equity_paths",
    "ActuarialDiagnosticBatchReport",
    "build_actuarial_diagnostic_batch_report",
    "diagnostic_report_to_summary_row",
    "evaluate_and_write_portfolio_search_actuarial_diagnostics",
    "write_actuarial_diagnostic_batch_report",
    "diagnostic_report_to_json_block",
    "format_actuarial_diagnostic_report",
    "format_actuarial_risk_result_compact",
    "attach_actuarial_diagnostic_to_output_payload",
    "build_portfolio_search_actuarial_diagnostic_section",
    "maybe_print_actuarial_diagnostic_section",
]