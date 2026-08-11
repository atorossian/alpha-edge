from __future__ import annotations

import pytest

from alpha_edge.jobs.run_transition_execution_plan import _select_target_weights


def test_select_target_weights_returns_local_candidate() -> None:
    local_payload = {
        "recommendation": "LOCAL_REBALANCE_RECOMMENDED",
        "reason": "local improved score",
        "best_candidate": {
            "weights": {
                "AAPL": 0.6,
                "MSFT": 0.4,
            },
            "score": 1.23,
            "score_improvement": 0.05,
            "turnover": 0.08,
        },
    }

    source, weights, summary = _select_target_weights(
        local_payload=local_payload,
        shadow_payload=None,
        prefer_shadow=True,
    )

    assert source == "local_optimizer"
    assert weights == {
        "AAPL": 0.6,
        "MSFT": 0.4,
    }
    assert summary["recommendation"] == "LOCAL_REBALANCE_RECOMMENDED"
    assert summary["score_improvement"] == 0.05


def test_select_target_weights_returns_shadow_candidate() -> None:
    shadow_payload = {
        "recommendation": "SHADOW_ACCEPTED",
        "reason": "shadow accepted",
        "state": {
            "shadow_id": "shadow-1",
            "source_run_id": "run-123",
            "source_run_key": "engine/v1/portfolio_search/runs/dt=2026-07-13/run.json",
            "shadow_weights": {
                "AAPL": 0.7,
                "MSFT": 0.3,
            },
            "score_advantage": 0.07,
            "health_advantage": 8.0,
            "turnover": 0.12,
        },
    }

    source, weights, summary = _select_target_weights(
        local_payload=None,
        shadow_payload=shadow_payload,
        prefer_shadow=True,
    )

    assert source == "shadow"
    assert weights == {
        "AAPL": 0.7,
        "MSFT": 0.3,
    }
    assert summary["recommendation"] == "SHADOW_ACCEPTED"
    assert summary["shadow_id"] == "shadow-1"
    assert summary["source_run_id"] == "run-123"


def test_select_target_weights_prefers_shadow_by_default() -> None:
    local_payload = {
        "recommendation": "LOCAL_REBALANCE_RECOMMENDED",
        "best_candidate": {
            "weights": {
                "LOCAL": 1.0,
            },
            "score": 1.0,
            "score_improvement": 0.03,
            "turnover": 0.05,
        },
    }

    shadow_payload = {
        "recommendation": "SHADOW_ACCEPTED",
        "state": {
            "shadow_id": "shadow-1",
            "shadow_weights": {
                "SHADOW": 1.0,
            },
            "score_advantage": 0.10,
            "health_advantage": 10.0,
            "turnover": 0.20,
        },
    }

    source, weights, _summary = _select_target_weights(
        local_payload=local_payload,
        shadow_payload=shadow_payload,
        prefer_shadow=True,
    )

    assert source == "shadow"
    assert weights == {
        "SHADOW": 1.0,
    }


def test_select_target_weights_prefers_local_when_requested() -> None:
    local_payload = {
        "recommendation": "LOCAL_REBALANCE_RECOMMENDED",
        "best_candidate": {
            "weights": {
                "LOCAL": 1.0,
            },
            "score": 1.0,
            "score_improvement": 0.03,
            "turnover": 0.05,
        },
    }

    shadow_payload = {
        "recommendation": "SHADOW_ACCEPTED",
        "state": {
            "shadow_id": "shadow-1",
            "shadow_weights": {
                "SHADOW": 1.0,
            },
            "score_advantage": 0.10,
            "health_advantage": 10.0,
            "turnover": 0.20,
        },
    }

    source, weights, _summary = _select_target_weights(
        local_payload=local_payload,
        shadow_payload=shadow_payload,
        prefer_shadow=False,
    )

    assert source == "local_optimizer"
    assert weights == {
        "LOCAL": 1.0,
    }


def test_select_target_weights_raises_when_no_accepted_target() -> None:
    local_payload = {
        "recommendation": "HOLD",
        "reason": "no improvement",
        "best_candidate": None,
    }

    shadow_payload = {
        "recommendation": "SHADOW_REJECTED",
        "reason": "not enough advantage",
        "state": {
            "shadow_weights": {
                "AAPL": 1.0,
            },
        },
    }

    with pytest.raises(RuntimeError, match="No accepted transition target found"):
        _select_target_weights(
            local_payload=local_payload,
            shadow_payload=shadow_payload,
            prefer_shadow=True,
        )
