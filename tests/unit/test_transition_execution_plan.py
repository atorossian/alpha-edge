from __future__ import annotations

from alpha_edge.core.schemas import (
    DiscreteAllocation,
    TransitionExecutionConfig,
    TransitionExecutionPlan,
)
from alpha_edge.portfolio.execution_engine import (
    allocation_to_trade_deltas,
    build_transition_execution_plan,
)
import pytest


def test_allocation_to_trade_deltas_buy_sell() -> None:
    current_shares = {
        "AAPL": 10.0,
        "MSFT": 5.0,
    }

    target_allocation = DiscreteAllocation(
        shares={
            "AAPL": 15.0,
            "MSFT": 2.0,
        },
        target_value={
            "AAPL": 1500.0,
            "MSFT": 400.0,
        },
        realized_value={
            "AAPL": 1500.0,
            "MSFT": 400.0,
        },
        realized_weights={
            "AAPL": 0.75,
            "MSFT": 0.20,
            "CASH": 0.05,
        },
        total_spent=1900.0,
        cash_left=100.0,
    )

    prices = {
        "AAPL": 100.0,
        "MSFT": 200.0,
    }

    cfg = TransitionExecutionConfig(
        min_trade_value=0.0,
        min_trade_quantity=0.0,
    )

    trades, blocked = allocation_to_trade_deltas(
        current_shares=current_shares,
        target_allocation=target_allocation,
        prices=prices,
        notional=2000.0,
        cfg=cfg,
    )

    assert blocked == []
    assert len(trades) == 2

    by_asset = {t.asset_id: t for t in trades}

    assert by_asset["AAPL"].direction == "BUY"
    assert by_asset["AAPL"].current_quantity == 10.0
    assert by_asset["AAPL"].target_quantity == 15.0
    assert by_asset["AAPL"].delta_quantity == 5.0
    assert by_asset["AAPL"].delta_value == 500.0

    assert by_asset["MSFT"].direction == "SELL"
    assert by_asset["MSFT"].current_quantity == 5.0
    assert by_asset["MSFT"].target_quantity == 2.0
    assert by_asset["MSFT"].delta_quantity == -3.0
    assert by_asset["MSFT"].delta_value == -600.0


def test_allocation_to_trade_deltas_blocks_small_trade() -> None:
    current_shares = {
        "AAPL": 10.0,
    }

    target_allocation = DiscreteAllocation(
        shares={
            "AAPL": 10.1,
        },
        target_value={
            "AAPL": 1010.0,
        },
        realized_value={
            "AAPL": 1010.0,
        },
        realized_weights={
            "AAPL": 1.0,
            "CASH": 0.0,
        },
        total_spent=1010.0,
        cash_left=0.0,
    )

    prices = {
        "AAPL": 100.0,
    }

    cfg = TransitionExecutionConfig(
        min_trade_value=25.0,
        min_trade_quantity=0.0,
    )

    trades, blocked = allocation_to_trade_deltas(
        current_shares=current_shares,
        target_allocation=target_allocation,
        prices=prices,
        notional=2000.0,
        cfg=cfg,
    )

    assert trades == []
    assert len(blocked) == 1
    assert blocked[0].asset_id == "AAPL"
    assert blocked[0].direction == "BUY"
    assert blocked[0].delta_quantity == pytest.approx(0.1)
    assert blocked[0].delta_value == pytest.approx(10.0)
    assert "below min_trade_value" in blocked[0].reason


def test_build_transition_execution_plan_reuses_discrete_allocation() -> None:
    current_shares = {
        "AAPL": 10.0,
        "MSFT": 5.0,
    }

    target_weights = {
        "AAPL": 0.75,
        "MSFT": 0.25,
    }

    prices = {
        "AAPL": 100.0,
        "MSFT": 200.0,
    }

    cfg = TransitionExecutionConfig(
        max_total_turnover=1.0,
        max_daily_turnover=1.0,
        min_trade_value=0.0,
        min_trade_quantity=0.0,
        allow_partial_transition=True,
    )

    plan = build_transition_execution_plan(
        as_of="2026-07-06",
        source="unit_test",
        current_shares=current_shares,
        target_weights=target_weights,
        prices=prices,
        notional=2000.0,
        cfg=cfg,
        min_weight=0.0,
    )

    assert isinstance(plan, TransitionExecutionPlan)
    assert isinstance(plan.target_allocation, DiscreteAllocation)

    assert plan.as_of == "2026-07-06"
    assert plan.source == "unit_test"
    assert plan.recommendation == "TRADE_RECOMMENDED"

    by_asset = {t.asset_id: t for t in plan.trades}

    assert "AAPL" in by_asset
    assert "MSFT" in by_asset

    assert by_asset["AAPL"].direction == "BUY"
    assert by_asset["AAPL"].delta_quantity > 0

    assert by_asset["MSFT"].direction == "SELL"
    assert by_asset["MSFT"].delta_quantity < 0

    assert plan.total_turnover > 0
    assert plan.daily_turnover_used > 0
    assert plan.blocked_turnover == 0.0


def test_build_transition_execution_plan_blocks_excess_total_turnover() -> None:
    current_shares = {
        "AAPL": 20.0,
    }

    target_weights = {
        "MSFT": 1.0,
    }

    prices = {
        "AAPL": 100.0,
        "MSFT": 200.0,
    }

    cfg = TransitionExecutionConfig(
        max_total_turnover=0.10,
        max_daily_turnover=0.10,
        min_trade_value=0.0,
        min_trade_quantity=0.0,
        allow_partial_transition=True,
    )

    plan = build_transition_execution_plan(
        as_of="2026-07-06",
        source="unit_test",
        current_shares=current_shares,
        target_weights=target_weights,
        prices=prices,
        notional=2000.0,
        cfg=cfg,
        min_weight=0.0,
    )

    assert plan.recommendation == "TRADE_BLOCKED"
    assert "max_total_turnover" in plan.reason
    assert plan.trades == []
    assert plan.total_turnover > cfg.max_total_turnover


def test_build_transition_execution_plan_allows_partial_transition() -> None:
    current_shares = {
        "AAPL": 20.0,
    }

    target_weights = {
        "MSFT": 1.0,
    }

    prices = {
        "AAPL": 100.0,
        "MSFT": 200.0,
    }

    cfg = TransitionExecutionConfig(
        max_total_turnover=1.0,
        max_daily_turnover=0.10,
        min_trade_value=0.0,
        min_trade_quantity=0.0,
        allow_partial_transition=True,
    )

    plan = build_transition_execution_plan(
        as_of="2026-07-06",
        source="unit_test",
        current_shares=current_shares,
        target_weights=target_weights,
        prices=prices,
        notional=2000.0,
        cfg=cfg,
        min_weight=0.0,
    )

    assert plan.recommendation in {"TRADE_RECOMMENDED", "NO_TRADE"}
    assert plan.total_turnover > cfg.max_daily_turnover
    assert plan.blocked_turnover > 0.0
    assert plan.diagnostics["partial_transition"] is True


def test_build_transition_execution_plan_fails_on_missing_current_price() -> None:
    current_shares = {
        "AAPL": 10.0,
    }

    target_weights = {
        "MSFT": 1.0,
    }

    prices = {
        "MSFT": 200.0,
    }

    cfg = TransitionExecutionConfig()

    try:
        build_transition_execution_plan(
            as_of="2026-07-06",
            source="unit_test",
            current_shares=current_shares,
            target_weights=target_weights,
            prices=prices,
            notional=2000.0,
            cfg=cfg,
            min_weight=0.0,
        )
    except ValueError as exc:
        assert "Missing price for current holding" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing current holding price.")
