from __future__ import annotations

from types import SimpleNamespace

from alpha_edge.portfolio.portfolio_search import (
    _archive_add,
    _schedule_value,
    _weight_l1_distance,
)


def _m(score: float, weights: dict[str, float]):
    return SimpleNamespace(score=float(score), weights=dict(weights))


def test_exploration_schedule_anneals_from_start_to_end() -> None:
    start = _schedule_value(gen_idx=0, generations=10, start=0.30, end=0.05, power=1.5)
    middle = _schedule_value(gen_idx=5, generations=10, start=0.30, end=0.05, power=1.5)
    end = _schedule_value(gen_idx=9, generations=10, start=0.30, end=0.05, power=1.5)

    assert start > middle > end
    assert abs(start - 0.30) < 1e-12
    assert abs(end - 0.05) < 1e-12


def test_weight_l1_distance_detects_structural_difference() -> None:
    a = {"SPY": 0.50, "GLD": 0.50}
    b = {"SPY": 0.50, "GLD": 0.50}
    c = {"BTC": 0.50, "TLT": 0.50}

    assert _weight_l1_distance(a, b) == 0.0
    assert _weight_l1_distance(a, c) > 1.0


def test_diversity_aware_archive_rejects_near_duplicate_lower_score() -> None:
    archive = {}
    first = _m(1.00, {"SPY": 0.50, "GLD": 0.50})
    near_duplicate_lower = _m(0.90, {"SPY": 0.51, "GLD": 0.49})

    _archive_add(archive, first, diversity_min_l1=0.15)
    _archive_add(archive, near_duplicate_lower, diversity_min_l1=0.15)

    assert len(archive) == 1
    assert next(iter(archive.values())).score == 1.00


def test_diversity_aware_archive_keeps_different_candidate() -> None:
    archive = {}
    first = _m(1.00, {"SPY": 0.50, "GLD": 0.50})
    different = _m(0.80, {"BTC": 0.50, "TLT": 0.50})

    _archive_add(archive, first, diversity_min_l1=0.15)
    _archive_add(archive, different, diversity_min_l1=0.15)

    assert len(archive) == 2
