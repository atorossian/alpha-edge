# src/alpha_edge/risk/actuarial/formatters.py
from __future__ import annotations

from typing import Any, Optional

from alpha_edge.core.schemas import ActuarialDiagnosticReport, ActuarialRiskResult


def _fmt_pct(value: Optional[float], *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _fmt_float(value: Optional[float], *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_money(value: Optional[float], *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def _fmt_leverage(value: Optional[float], *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}x"


def _fmt_days(value: Optional[float], *, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}d"


def _metric(
    label: str,
    value: str,
    *,
    width: int = 36,
) -> str:
    return f"{label:<{width}} {value}"


def _section(title: str) -> str:
    underline = "-" * len(title)
    return f"{title}\n{underline}"


def format_actuarial_diagnostic_report(
    report: ActuarialDiagnosticReport,
    *,
    title: str = "ACTUARIAL RISK DIAGNOSTICS",
    include_detail_metrics: bool = True,
    include_flags: bool = True,
    include_warnings: bool = True,
) -> str:
    """
    Format an ActuarialDiagnosticReport for terminal/text output.

    This formatter is intentionally plain text so it can be reused in:
      - portfolio search terminal output,
      - quarantine analysis terminal output,
      - daily report text output,
      - logs,
      - future markdown-like reports.

    It does not mutate the report and does not write files.
    """
    report.validate()

    headline = report.headline_metrics
    detail = report.detail_metrics

    lines: list[str] = []

    lines.append(_section(title))
    lines.append(_metric("Source:", report.source))
    lines.append(_metric("Portfolio ID:", report.portfolio_id or "n/a"))
    lines.append(_metric("Run ID:", report.run_id or "n/a"))
    lines.append(_metric("Verdict:", report.verdict.upper()))
    lines.append(_metric("Risk grade:", report.risk_grade or "n/a"))
    lines.append("")

    lines.append("Headline metrics:")
    lines.append(
        _metric(
            "Ruin probability:",
            _fmt_pct(headline.get("ruin_probability")),
        )
    )
    lines.append(
        _metric(
            "Drawdown breach probability:",
            _fmt_pct(headline.get("drawdown_breach_probability")),
        )
    )
    lines.append(
        _metric(
            "Goal probability:",
            _fmt_pct(headline.get("goal_probability")),
        )
    )
    lines.append(
        _metric(
            "Goal-before-ruin probability:",
            _fmt_pct(headline.get("probability_goal_before_ruin")),
        )
    )
    lines.append(
        _metric(
            "Solvency ratio:",
            _fmt_float(headline.get("solvency_ratio")),
        )
    )
    lines.append(
        _metric(
            "Capital buffer gap:",
            _fmt_money(headline.get("capital_buffer_gap")),
        )
    )
    lines.append(
        _metric(
            "Safe leverage estimate:",
            _fmt_leverage(headline.get("safe_leverage_estimate")),
        )
    )

    if include_detail_metrics:
        lines.append("")
        lines.append("Detail metrics:")
        lines.append(
            _metric(
                "Initial value:",
                _fmt_money(detail.get("initial_value")),
            )
        )
        lines.append(
            _metric(
                "Horizon days:",
                _fmt_float(detail.get("horizon_days"), digits=0),
            )
        )
        lines.append(
            _metric(
                "Number of paths:",
                _fmt_float(detail.get("n_paths"), digits=0),
            )
        )
        lines.append(
            _metric(
                "Ruin threshold:",
                _fmt_money(detail.get("ruin_threshold")),
            )
        )
        lines.append(
            _metric(
                "Expected time to ruin:",
                _fmt_days(detail.get("expected_time_to_ruin_days")),
            )
        )
        lines.append(
            _metric(
                "Median time to ruin:",
                _fmt_days(detail.get("median_time_to_ruin_days")),
            )
        )
        lines.append(
            _metric(
                "Drawdown limit:",
                _fmt_pct(detail.get("drawdown_limit_pct")),
            )
        )
        lines.append(
            _metric(
                "Expected max drawdown:",
                _fmt_pct(detail.get("expected_max_drawdown")),
            )
        )
        lines.append(
            _metric(
                "Median max drawdown:",
                _fmt_pct(detail.get("median_max_drawdown")),
            )
        )
        lines.append(
            _metric(
                "CVaR max drawdown 95:",
                _fmt_pct(detail.get("cvar_max_drawdown_95")),
            )
        )
        lines.append(
            _metric(
                "Goal value:",
                _fmt_money(detail.get("goal_value")),
            )
        )
        lines.append(
            _metric(
                "Median time to goal:",
                _fmt_days(detail.get("median_time_to_goal_days")),
            )
        )
        lines.append(
            _metric(
                "Recovery probability:",
                _fmt_pct(detail.get("recovery_probability")),
            )
        )
        lines.append(
            _metric(
                "Median recovery time:",
                _fmt_days(detail.get("median_recovery_time_days")),
            )
        )
        lines.append(
            _metric(
                "Capital required:",
                _fmt_money(detail.get("capital_required")),
            )
        )

    if include_flags:
        lines.append("")
        lines.append("Risk flags:")
        if report.risk_flags:
            for flag in report.risk_flags:
                lines.append(f"- {flag}")
        else:
            lines.append("- none")

    if include_warnings:
        lines.append("")
        lines.append("Warnings:")
        if report.warnings:
            for warning in report.warnings:
                lines.append(f"- {warning}")
        else:
            lines.append("- none")

    return "\n".join(lines)


def format_actuarial_risk_result_compact(
    result: ActuarialRiskResult,
    *,
    title: str = "ACTUARIAL RISK SUMMARY",
) -> str:
    """
    Compact formatter for raw ActuarialRiskResult.

    Use this when a diagnostic report has not been built yet.
    In most production/reporting flows, prefer format_actuarial_diagnostic_report().
    """
    result.validate()

    lines: list[str] = []

    lines.append(_section(title))
    lines.append(_metric("Risk grade:", result.risk_grade or "n/a"))
    lines.append(_metric("Initial value:", _fmt_money(result.initial_value)))
    lines.append(_metric("Horizon days:", str(result.horizon_days)))
    lines.append(_metric("Number of paths:", str(result.n_paths)))
    lines.append("")

    lines.append(_metric("Ruin probability:", _fmt_pct(result.ruin_probability)))
    lines.append(
        _metric(
            "Drawdown breach probability:",
            _fmt_pct(result.drawdown_breach_probability),
        )
    )
    lines.append(_metric("Goal probability:", _fmt_pct(result.goal_probability)))
    lines.append(
        _metric(
            "Goal-before-ruin probability:",
            _fmt_pct(result.probability_goal_before_ruin),
        )
    )
    lines.append(_metric("Solvency ratio:", _fmt_float(result.solvency_ratio)))
    lines.append(_metric("Capital buffer gap:", _fmt_money(result.capital_buffer_gap)))
    lines.append(_metric("Safe leverage estimate:", _fmt_leverage(result.safe_leverage_estimate)))

    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)


def diagnostic_report_to_json_block(
    report: ActuarialDiagnosticReport,
) -> dict[str, Any]:
    """
    Compact JSON-ready block to embed into existing outputs.

    This is meant for portfolio_search result dictionaries, daily report payloads,
    or quarantine summaries.

    It intentionally excludes the full nested result by default to keep the
    embedded object compact. The full result is already available through
    report.to_dict() when needed.
    """
    report.validate()

    return {
        "portfolio_id": report.portfolio_id,
        "run_id": report.run_id,
        "source": report.source,
        "verdict": report.verdict,
        "risk_grade": report.risk_grade,
        "risk_flags": list(report.risk_flags),
        "warnings": list(report.warnings),
        "headline_metrics": dict(report.headline_metrics),
        "detail_metrics": dict(report.detail_metrics),
        "metadata": dict(report.metadata),
    }