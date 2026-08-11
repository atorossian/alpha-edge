from __future__ import annotations

import pytest

from alpha_edge.core.schemas import ShadowPortfolioConfig, ShadowPortfolioState
from alpha_edge.portfolio.shadow_portfolio_engine import (
    assess_shadow_portfolio,
    delta_weights,
    weight_turnover,
)


def test_weight_turnover_and_delta_weights() -> None:
    current = {
        "AAPL": 0.6,
        "MSFT": 0.4,
    }
    target = {
        "AAPL": 0.3,
        "GOOG": 0.7,
    }

    assert weight_turnover(current, target) == pytest.approx(0.7)

    delta = delta_weights(current, target)

    assert delta["AAPL"] == pytest.approx(-0.3)
    assert delta["MSFT"] == pytest.approx(-0.4)
    assert delta["GOOG"] == pytest.approx(0.7)


def test_assess_shadow_portfolio_accepts_immediate_large_advantage() -> None:
    state = ShadowPortfolioState(
        shadow_id="shadow-1",
        as_of="2026-07-13",
        source_run_id="run-1",
        source_run_key="key-1",
        status="active",
        current_health_score=60.0,
        shadow_health_score=72.0,
        health_advantage=12.0,
        current_score=1.00,
        shadow_score=1.08,
        score_advantage=0.08,
        turnover=0.20,
        days_active=1,
        days_dominating=0,
        current_weights={"AAPL": 1.0},
        shadow_weights={"MSFT": 1.0},
        delta_weights={"AAPL": -1.0, "MSFT": 1.0},
    )

    cfg = ShadowPortfolioConfig(
        min_health_advantage=5.0,
        min_score_advantage=0.02,
        max_turnover_to_accept=0.35,
        confirmation_days=3,
        immediate_accept_health_advantage=10.0,
        immediate_accept_score_advantage=0.05,
    )

    assessment = assess_shadow_portfolio(
        state=state,
        cfg=cfg,
    )

    assert assessment.recommendation == "SHADOW_ACCEPTED"
    assert assessment.state.status == "accepted"
    assert assessment.state.days_dominating == 1


def test_assess_shadow_portfolio_stays_active_until_confirmation_days() -> None:
    state = ShadowPortfolioState(
        shadow_id="shadow-1",
        as_of="2026-07-13",
        source_run_id="run-1",
        source_run_key="key-1",
        status="active",
        current_health_score=60.0,
        shadow_health_score=66.0,
        health_advantage=6.0,
        current_score=1.00,
        shadow_score=1.03,
        score_advantage=0.03,
        turnover=0.20,
        days_active=1,
        days_dominating=1,
        current_weights={"AAPL": 1.0},
        shadow_weights={"MSFT": 1.0},
        delta_weights={"AAPL": -1.0, "MSFT": 1.0},
    )

    cfg = ShadowPortfolioConfig(
        min_health_advantage=5.0,
        min_score_advantage=0.02,
        max_turnover_to_accept=0.35,
        confirmation_days=3,
        immediate_accept_health_advantage=10.0,
        immediate_accept_score_advantage=0.05,
    )

    assessment = assess_shadow_portfolio(
        state=state,
        cfg=cfg,
    )

    assert assessment.recommendation == "SHADOW_ACTIVE"
    assert assessment.state.status == "active"
    assert assessment.state.days_dominating == 2


def test_assess_shadow_portfolio_accepts_after_confirmation_days() -> None:
    state = ShadowPortfolioState(
        shadow_id="shadow-1",
        as_of="2026-07-13",
        source_run_id="run-1",
        source_run_key="key-1",
        status="active",
        current_health_score=60.0,
        shadow_health_score=66.0,
        health_advantage=6.0,
        current_score=1.00,
        shadow_score=1.03,
        score_advantage=0.03,
        turnover=0.20,
        days_active=2,
        days_dominating=2,
        current_weights={"AAPL": 1.0},
        shadow_weights={"MSFT": 1.0},
        delta_weights={"AAPL": -1.0, "MSFT": 1.0},
    )

    cfg = ShadowPortfolioConfig(
        min_health_advantage=5.0,
        min_score_advantage=0.02,
        max_turnover_to_accept=0.35,
        confirmation_days=3,
        immediate_accept_health_advantage=10.0,
        immediate_accept_score_advantage=0.05,
    )

    assessment = assess_shadow_portfolio(
        state=state,
        cfg=cfg,
    )

    assert assessment.recommendation == "SHADOW_ACCEPTED"
    assert assessment.state.status == "accepted"
    assert assessment.state.days_dominating == 3


def test_assess_shadow_portfolio_rejects_excess_turnover() -> None:
    state = ShadowPortfolioState(
        shadow_id="shadow-1",
        as_of="2026-07-13",
        source_run_id="run-1",
        source_run_key="key-1",
        status="active",
        current_health_score=60.0,
        shadow_health_score=75.0,
        health_advantage=15.0,
        current_score=1.00,
        shadow_score=1.10,
        score_advantage=0.10,
        turnover=0.60,
        days_active=1,
        days_dominating=0,
        current_weights={"AAPL": 1.0},
        shadow_weights={"MSFT": 1.0},
        delta_weights={"AAPL": -1.0, "MSFT": 1.0},
    )

    cfg = ShadowPortfolioConfig(
        min_health_advantage=5.0,
        min_score_advantage=0.02,
        max_turnover_to_accept=0.35,
        confirmation_days=3,
        immediate_accept_health_advantage=10.0,
        immediate_accept_score_advantage=0.05,
    )

    assessment = assess_shadow_portfolio(
        state=state,
        cfg=cfg,
    )

    assert assessment.recommendation == "SHADOW_REJECTED"
    assert assessment.state.status == "rejected"
    assert "turnover" in assessment.reason
