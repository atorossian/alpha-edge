# src/alpha_edge/risk/actuarial/path_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from alpha_edge.core.schemas import SurvivalCurvePoint


@dataclass(frozen=True)
class EventTimeSummary:
    """
    Summary of first-hit event times.

    event_probability:
        Fraction of paths where the event happened.

    expected_time_days:
        Mean first-hit time among paths where the event happened.
        None if the event never happened.

    median_time_days:
        Median first-hit time among paths where the event happened.
        None if the event never happened.

    first_hit_times:
        Array of length n_paths.
        Contains first event time as integer day index.
        Contains np.nan for paths where the event never happened.
    """

    event_probability: float
    expected_time_days: Optional[float]
    median_time_days: Optional[float]
    first_hit_times: np.ndarray


@dataclass(frozen=True)
class RecoveryMetrics:
    """
    Recovery metrics after drawdown breach.

    recovery_probability:
        Among paths that breached the drawdown limit, fraction that later recovered.

    median_recovery_time_days:
        Median number of days from first breach to recovery.
        None if no breached path recovered.
    """

    recovery_probability: Optional[float]
    median_recovery_time_days: Optional[float]


def _as_2d_float_array(equity_paths: object) -> np.ndarray:
    arr = np.asarray(equity_paths, dtype=float)

    if arr.ndim != 2:
        raise ValueError("equity_paths must be a 2D array-like object")

    if arr.shape[0] <= 0:
        raise ValueError("equity_paths must contain at least one path")

    if arr.shape[1] <= 1:
        raise ValueError("equity_paths must contain at least two time steps, including t=0")

    if not np.all(np.isfinite(arr)):
        raise ValueError("equity_paths contains NaN or infinite values")

    return arr


def validate_equity_paths(
    equity_paths: object,
    *,
    horizon_days: Optional[int] = None,
    initial_value: Optional[float] = None,
    initial_value_tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Validate and normalize equity paths.

    Expected shape:
        rows    = paths
        columns = time steps

    Column 0 is t=0.

    If horizon_days is supplied, the array must have at least
    horizon_days + 1 columns.
    """
    arr = _as_2d_float_array(equity_paths)

    if horizon_days is not None:
        h = int(horizon_days)
        if h <= 0:
            raise ValueError("horizon_days must be > 0")

        required_cols = h + 1
        if arr.shape[1] < required_cols:
            raise ValueError(
                f"equity_paths has insufficient columns for horizon_days={h}. "
                f"Expected at least {required_cols}, got {arr.shape[1]}"
            )

    if initial_value is not None:
        iv = float(initial_value)
        if iv <= 0.0:
            raise ValueError("initial_value must be > 0")

        starts = arr[:, 0]
        if not np.allclose(starts, iv, atol=initial_value_tolerance, rtol=0.0):
            raise ValueError(
                "equity_paths column 0 must match config.initial_value for all paths. "
                f"Expected {iv}, got min={starts.min()}, max={starts.max()}"
            )

    return arr


def _none_if_empty(values: np.ndarray, *, fn: Callable[[np.ndarray], float]) -> Optional[float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return None
    return float(fn(clean))


def first_hit_times(
    equity_paths: object,
    *,
    predicate: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Return first time index where predicate is true per path.

    The predicate receives the full 2D equity path array and must return
    a boolean array of the same shape.

    Paths that never hit the event return np.nan.
    """
    arr = _as_2d_float_array(equity_paths)

    hits = np.asarray(predicate(arr), dtype=bool)

    if hits.shape != arr.shape:
        raise ValueError(
            "predicate must return a boolean array with the same shape as equity_paths"
        )

    any_hit = hits.any(axis=1)
    first_idx = np.argmax(hits, axis=1).astype(float)
    first_idx[~any_hit] = np.nan

    return first_idx


def summarize_event_times(first_times: np.ndarray) -> EventTimeSummary:
    ft = np.asarray(first_times, dtype=float)

    if ft.ndim != 1:
        raise ValueError("first_times must be a 1D array")

    if ft.size == 0:
        raise ValueError("first_times cannot be empty")

    occurred = np.isfinite(ft)
    event_probability = float(np.mean(occurred))

    event_times = ft[occurred]

    expected_time_days = None if event_times.size == 0 else float(np.mean(event_times))
    median_time_days = None if event_times.size == 0 else float(np.median(event_times))

    return EventTimeSummary(
        event_probability=event_probability,
        expected_time_days=expected_time_days,
        median_time_days=median_time_days,
        first_hit_times=ft,
    )


def calculate_ruin_probability(
    equity_paths: object,
    *,
    ruin_threshold: float,
) -> EventTimeSummary:
    """
    Ruin event:
        equity <= ruin_threshold
    """
    threshold = float(ruin_threshold)
    if threshold <= 0.0:
        raise ValueError("ruin_threshold must be > 0")

    arr = _as_2d_float_array(equity_paths)

    first_times = first_hit_times(
        arr,
        predicate=lambda x: x <= threshold,
    )

    return summarize_event_times(first_times)


def calculate_goal_probability(
    equity_paths: object,
    *,
    goal_value: float,
) -> EventTimeSummary:
    """
    Goal event:
        equity >= goal_value
    """
    goal = float(goal_value)
    if goal <= 0.0:
        raise ValueError("goal_value must be > 0")

    arr = _as_2d_float_array(equity_paths)

    first_times = first_hit_times(
        arr,
        predicate=lambda x: x >= goal,
    )

    return summarize_event_times(first_times)


def calculate_probability_goal_before_ruin(
    *,
    goal_first_times: np.ndarray,
    ruin_first_times: np.ndarray,
) -> float:
    """
    Probability that goal is reached before ruin.

    A path counts as success if:
        goal occurred and either ruin did not occur or goal_time < ruin_time.

    If both happen at the same time index, this is not counted as
    goal before ruin.
    """
    goal = np.asarray(goal_first_times, dtype=float)
    ruin = np.asarray(ruin_first_times, dtype=float)

    if goal.ndim != 1 or ruin.ndim != 1:
        raise ValueError("goal_first_times and ruin_first_times must be 1D arrays")

    if goal.shape != ruin.shape:
        raise ValueError("goal_first_times and ruin_first_times must have the same shape")

    goal_happened = np.isfinite(goal)
    ruin_happened = np.isfinite(ruin)

    success = goal_happened & (~ruin_happened | (goal < ruin))

    return float(np.mean(success))


def calculate_running_drawdowns(equity_paths: object) -> np.ndarray:
    """
    Calculate drawdown paths.

    Drawdown is expressed as a negative fraction:

        equity / running_peak - 1

    Examples:
        0.00  = no drawdown
        -0.10 = -10% drawdown
        -0.30 = -30% drawdown
    """
    arr = _as_2d_float_array(equity_paths)

    running_peak = np.maximum.accumulate(arr, axis=1)

    if np.any(running_peak <= 0.0):
        raise ValueError("running peak must stay positive to calculate drawdowns")

    return arr / running_peak - 1.0


def calculate_max_drawdowns(equity_paths: object) -> np.ndarray:
    """
    Return maximum drawdown per path as negative numbers.

    Example:
        -0.30 means max drawdown of -30%.
    """
    dd = calculate_running_drawdowns(equity_paths)
    return np.min(dd, axis=1)


def calculate_drawdown_breach_probability(
    equity_paths: object,
    *,
    drawdown_limit_pct: float,
) -> float:
    """
    Probability that max drawdown breaches the configured limit.

    drawdown_limit_pct is positive:
        0.30 means breach if drawdown <= -0.30.
    """
    limit = float(drawdown_limit_pct)
    if not 0.0 < limit <= 1.0:
        raise ValueError("drawdown_limit_pct must be > 0 and <= 1")

    max_dd = calculate_max_drawdowns(equity_paths)

    return float(np.mean(max_dd <= -limit))


def calculate_cvar_max_drawdown(
    max_drawdowns: object,
    *,
    alpha: float = 0.95,
) -> float:
    """
    CVaR-style average of the worst max drawdowns.

    max_drawdowns are negative numbers.

    For alpha=0.95, this averages the worst 5% most negative drawdowns.
    """
    mdd = np.asarray(max_drawdowns, dtype=float)

    if mdd.ndim != 1:
        raise ValueError("max_drawdowns must be 1D")

    if mdd.size == 0:
        raise ValueError("max_drawdowns cannot be empty")

    if not np.all(np.isfinite(mdd)):
        raise ValueError("max_drawdowns contains NaN or infinite values")

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be > 0 and < 1")

    cutoff = np.quantile(mdd, 1.0 - alpha)
    tail = mdd[mdd <= cutoff]

    if tail.size == 0:
        return float(cutoff)

    return float(np.mean(tail))


def calculate_survival_curve(
    *,
    first_event_times: np.ndarray,
    horizons_days: list[int],
) -> list[SurvivalCurvePoint]:
    """
    Build survival curve from first event times.

    Survival at horizon h:
        P(event has not happened by h)

    Event probability at horizon h:
        P(event has happened by h)
    """
    ft = np.asarray(first_event_times, dtype=float)

    if ft.ndim != 1:
        raise ValueError("first_event_times must be 1D")

    if ft.size == 0:
        raise ValueError("first_event_times cannot be empty")

    points: list[SurvivalCurvePoint] = []

    for h in horizons_days:
        horizon = int(h)
        if horizon <= 0:
            raise ValueError("horizons_days items must be > 0")

        event_by_horizon = np.isfinite(ft) & (ft <= horizon)
        event_probability = float(np.mean(event_by_horizon))
        survival_probability = float(1.0 - event_probability)

        points.append(
            SurvivalCurvePoint(
                horizon_days=horizon,
                survival_probability=survival_probability,
                event_probability=event_probability,
            )
        )

    return points


def calculate_recovery_metrics(
    equity_paths: object,
    *,
    drawdown_limit_pct: float,
    recovery_level: float = 1.0,
) -> RecoveryMetrics:
    """
    Calculate recovery metrics after first drawdown breach.

    For each path:
      1. Find first time drawdown breaches -drawdown_limit_pct.
      2. Record the running peak at that breach.
      3. Recovery occurs when equity later reaches:
             breach_peak * recovery_level

    With recovery_level=1.0, recovery means returning to the pre-breach peak.
    """
    arr = _as_2d_float_array(equity_paths)

    limit = float(drawdown_limit_pct)
    if not 0.0 < limit <= 1.0:
        raise ValueError("drawdown_limit_pct must be > 0 and <= 1")

    rec_level = float(recovery_level)
    if rec_level <= 0.0:
        raise ValueError("recovery_level must be > 0")

    running_peak = np.maximum.accumulate(arr, axis=1)
    dd = arr / running_peak - 1.0

    breach = dd <= -limit
    any_breach = breach.any(axis=1)

    if not np.any(any_breach):
        return RecoveryMetrics(
            recovery_probability=None,
            median_recovery_time_days=None,
        )

    first_breach_idx = np.argmax(breach, axis=1)

    recovery_times: list[float] = []
    n_breached = int(np.sum(any_breach))
    n_recovered = 0

    for path_idx in np.where(any_breach)[0]:
        b = int(first_breach_idx[path_idx])
        target = float(running_peak[path_idx, b]) * rec_level

        future = arr[path_idx, b + 1 :]
        if future.size == 0:
            continue

        recovered_mask = future >= target

        if not np.any(recovered_mask):
            continue

        first_recovery_offset = int(np.argmax(recovered_mask)) + 1
        recovery_times.append(float(first_recovery_offset))
        n_recovered += 1

    recovery_probability = float(n_recovered / n_breached)

    if len(recovery_times) == 0:
        median_recovery_time_days = None
    else:
        median_recovery_time_days = float(np.median(np.asarray(recovery_times, dtype=float)))

    return RecoveryMetrics(
        recovery_probability=recovery_probability,
        median_recovery_time_days=median_recovery_time_days,
    )