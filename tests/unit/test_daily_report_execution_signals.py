from __future__ import annotations

from dataclasses import dataclass

from alpha_edge.portfolio.evaluation_service import build_daily_report_execution_signals


@dataclass(frozen=True)
class FakeDecision:
    should_rebalance: bool
    reasons: list[str]
    leverage_real: float
    leverage_target: float
    drift_ratio: float


@dataclass(frozen=True)
class FakeHealth:
    score: float


def test_daily_report_signals_are_diagnostic_only_without_transition_assessment():
    out = build_daily_report_execution_signals(
        rescale_decision=FakeDecision(True, ["market_leverage_change"], 3.0, 2.0, 1.5),
        reoptimization_pressure=True,
        take_profit={"do_harvest": False, "m_star": 1.0, "reasons": []},
        transition_assessment=None,
        current_health=FakeHealth(score=72.5),
    )

    assert out["schema_version"] == "daily_report_execution_signals_v1"
    assert out["decision_authority"] == "daily_report_diagnostic_only"
    assert out["final_execution_decision"]["recommendation"] is None
    assert out["signals"]["rescale"]["triggered"] is True
    assert out["signals"]["rebalance"]["triggered"] is False
    assert out["signals"]["reoptimization_pressure"]["triggered"] is True


def test_transition_assessment_is_authoritative_when_available():
    transition = {
        "schema_version": "portfolio_transition_assessment_v1",
        "as_of": "2026-08-11",
        "recommendation": "LOCAL_OPTIMIZATION_RECOMMENDED",
        "reason": "Portfolio is healthy; run daily local optimization.",
        "diagnostics": {"triggers": []},
    }
    out = build_daily_report_execution_signals(
        rescale_decision=FakeDecision(False, ["no_market_rescale"], 2.0, 2.0, 1.0),
        reoptimization_pressure=False,
        transition_assessment=transition,
    )

    assert out["decision_authority"] == "transition_assessment"
    assert out["final_execution_decision"]["recommendation"] == "LOCAL_OPTIMIZATION_RECOMMENDED"
    assert out["transition_assessment_ref"]["available"] is True
    assert out["signals"]["rescale"]["triggered"] is False
