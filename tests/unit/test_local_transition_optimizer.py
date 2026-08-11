from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from alpha_edge.core.schemas import EvalMetrics, LocalTransitionOptimizerConfig
from alpha_edge.portfolio import local_transition_optimizer as lto


def _metrics(weights: dict[str, float], score: float) -> EvalMetrics:
    return EvalMetrics(
        weights=weights,
        goals=(7500.0, 10000.0, 12500.0),
        main_goal=10000.0,
        ann_return=0.10,
        ann_vol=0.20,
        sharpe=0.50,
        sortino=0.60,
        max_drawdown=-0.20,
        ann_vol_lw=0.20,
        var_95=-0.02,
        cvar_95=-0.03,
        ruin_prob_1y=0.05,
        p_hit_goal_1_1y=0.50,
        p_hit_goal_2_1y=0.40,
        p_hit_goal_3_1y=0.30,
        med_t_goal_1_days=100.0,
        med_t_goal_2_days=150.0,
        med_t_goal_3_days=200.0,
        ending_equity_p5=8000.0,
        ending_equity_p25=9000.0,
        ending_equity_p50=10000.0,
        ending_equity_p75=11000.0,
        ending_equity_p95=12000.0,
        score=score,
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

    assert lto.weight_turnover(current, target) == pytest.approx(0.7)

    delta = lto.delta_weights(current, target)

    assert delta["AAPL"] == pytest.approx(-0.3)
    assert delta["MSFT"] == pytest.approx(-0.4)
    assert delta["GOOG"] == pytest.approx(0.7)


def test_run_local_transition_optimizer_recommends_rebalance(monkeypatch) -> None:
    current_metrics = _metrics(
        weights={
            "AAPL": 0.6,
            "MSFT": 0.4,
        },
        score=1.00,
    )

    best_metrics = _metrics(
        weights={
            "AAPL": 0.5,
            "MSFT": 0.5,
        },
        score=1.05,
    )

    def fake_evaluate_weights_for_search(**kwargs):
        return current_metrics

    def fake_refine_portfolio_annealing(**kwargs):
        assert kwargs["base_metrics"] is current_metrics
        return best_metrics

    monkeypatch.setattr(
        lto,
        "evaluate_weights_for_search",
        fake_evaluate_weights_for_search,
    )
    monkeypatch.setattr(
        lto,
        "refine_portfolio_annealing",
        fake_refine_portfolio_annealing,
    )

    cfg = LocalTransitionOptimizerConfig(
        anneal_steps=10,
        max_turnover=0.20,
        min_score_improvement=0.02,
    )

    returns = pd.DataFrame(
        {
            "AAPL": [0.01, -0.01, 0.02],
            "MSFT": [0.02, 0.00, -0.01],
        }
    )

    universe = {
        "AAPL": object(),
        "MSFT": object(),
    }

    result = lto.run_local_transition_optimizer(
        as_of="2026-07-13",
        returns=returns,
        universe=universe,
        current_weights=current_metrics.weights,
        equity0=10000.0,
        notional=10000.0,
        goals=(7500.0, 10000.0, 12500.0),
        main_goal=10000.0,
        score_config=None,
        cfg=cfg,
        lw_cov=None,
    )

    assert result.recommendation == "LOCAL_REBALANCE_RECOMMENDED"
    assert result.best_candidate is not None
    assert result.best_candidate.score_improvement == pytest.approx(0.05)
    assert result.best_candidate.turnover == pytest.approx(0.10)
    assert result.candidates_accepted_by_turnover == 1


def test_run_local_transition_optimizer_holds_when_turnover_too_high(monkeypatch) -> None:
    current_metrics = _metrics(
        weights={
            "AAPL": 1.0,
        },
        score=1.00,
    )

    best_metrics = _metrics(
        weights={
            "MSFT": 1.0,
        },
        score=1.20,
    )

    monkeypatch.setattr(
        lto,
        "evaluate_weights_for_search",
        lambda **kwargs: current_metrics,
    )
    monkeypatch.setattr(
        lto,
        "refine_portfolio_annealing",
        lambda **kwargs: best_metrics,
    )

    cfg = LocalTransitionOptimizerConfig(
        anneal_steps=10,
        max_turnover=0.10,
        min_score_improvement=0.02,
    )

    result = lto.run_local_transition_optimizer(
        as_of="2026-07-13",
        returns=pd.DataFrame({"AAPL": [0.01], "MSFT": [0.02]}),
        universe={"AAPL": object(), "MSFT": object()},
        current_weights=current_metrics.weights,
        equity0=10000.0,
        notional=10000.0,
        goals=(7500.0, 10000.0, 12500.0),
        main_goal=10000.0,
        score_config=None,
        cfg=cfg,
        lw_cov=None,
    )

    assert result.recommendation == "HOLD"
    assert "exceeds max_turnover" in result.reason
    assert result.candidates_accepted_by_turnover == 0


def test_run_local_transition_optimizer_holds_when_score_improvement_too_small(monkeypatch) -> None:
    current_metrics = _metrics(
        weights={
            "AAPL": 0.6,
            "MSFT": 0.4,
        },
        score=1.00,
    )

    best_metrics = replace(
        current_metrics,
        weights={
            "AAPL": 0.55,
            "MSFT": 0.45,
        },
        score=1.005,
    )

    monkeypatch.setattr(
        lto,
        "evaluate_weights_for_search",
        lambda **kwargs: current_metrics,
    )
    monkeypatch.setattr(
        lto,
        "refine_portfolio_annealing",
        lambda **kwargs: best_metrics,
    )

    cfg = LocalTransitionOptimizerConfig(
        anneal_steps=10,
        max_turnover=0.20,
        min_score_improvement=0.02,
    )

    result = lto.run_local_transition_optimizer(
        as_of="2026-07-13",
        returns=pd.DataFrame({"AAPL": [0.01], "MSFT": [0.02]}),
        universe={"AAPL": object(), "MSFT": object()},
        current_weights=current_metrics.weights,
        equity0=10000.0,
        notional=10000.0,
        goals=(7500.0, 10000.0, 12500.0),
        main_goal=10000.0,
        score_config=None,
        cfg=cfg,
        lw_cov=None,
    )

    assert result.recommendation == "HOLD"
    assert "below min_score_improvement" in result.reason
    assert result.candidates_accepted_by_turnover == 1
