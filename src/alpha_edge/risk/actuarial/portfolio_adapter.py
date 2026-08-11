# src/alpha_edge/risk/actuarial/portfolio_adapter.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from alpha_edge.core.schemas import ActuarialRiskConfig, ActuarialRiskResult, SurvivalConfig
from alpha_edge.risk.actuarial.engine import evaluate_actuarial_risk
from alpha_edge.risk.actuarial.path_metrics import validate_equity_paths


DEFAULT_EQUITY_PATH_KEYS: tuple[str, ...] = (
    "equity_paths",
    "portfolio_paths",
    "simulated_equity_paths",
    "mc_equity_paths",
    "paths",
)

def _default_survival_horizons_for_horizon(horizon_days: int) -> list[int]:
    """
    Build safe default survival horizons that do not exceed the configured horizon.

    The global ActuarialRiskConfig default uses production-like horizons:
        [21, 63, 126, 252, 756]

    But unit tests and short portfolio-search diagnostics may use very short
    horizons, such as horizon_days=2. In that case, the default survival
    horizons must be clipped.
    """
    h = int(horizon_days)
    if h <= 0:
        raise ValueError("horizon_days must be > 0")

    candidates = [21, 63, 126, 252, 756]
    clipped = [x for x in candidates if x <= h]

    if clipped:
        return clipped

    return [h]

def _as_mapping(obj: object) -> Mapping[str, Any]:
    """
    Convert common portfolio-search result containers into a mapping.

    Supported:
      - dict-like objects
      - dataclasses
      - objects with __dict__

    This lets the adapter consume portfolio-search results without tightly
    coupling the actuarial module to one specific search result class.
    """
    if isinstance(obj, Mapping):
        return obj

    if is_dataclass(obj):
        converted = asdict(obj)
        if isinstance(converted, Mapping):
            return converted

    data = getattr(obj, "__dict__", None)
    if isinstance(data, Mapping):
        return data

    raise TypeError(
        "portfolio_result must be a mapping, dataclass instance, or object with __dict__"
    )


def _extract_by_dotted_key(data: Mapping[str, Any], dotted_key: str) -> Any:
    """
    Extract nested values using dotted keys.

    Example:
        dotted_key="simulation.equity_paths"

    Works for nested dictionaries and simple objects.
    """
    current: Any = data

    for part in dotted_key.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(dotted_key)
            current = current[part]
        else:
            if not hasattr(current, part):
                raise KeyError(dotted_key)
            current = getattr(current, part)

    return current


def extract_equity_paths_from_portfolio_result(
    portfolio_result: object,
    *,
    equity_paths_key: Optional[str] = None,
    candidate_keys: tuple[str, ...] = DEFAULT_EQUITY_PATH_KEYS,
) -> np.ndarray:
    """
    Extract simulated equity paths from a portfolio-search result.

    The adapter is intentionally flexible because portfolio-search outputs may
    evolve. It tries a set of common keys unless a specific key is supplied.

    Supported path containers:
      - numpy arrays
      - pandas DataFrames
      - list of lists
      - tuple of tuples

    Expected normalized shape:
      rows    = simulation paths
      columns = time steps

    Column 0 must be t=0 and should equal config.initial_value later when
    passed to evaluate_portfolio_search_actuarial_risk().
    """
    data = _as_mapping(portfolio_result)

    keys_to_try = (equity_paths_key,) if equity_paths_key else candidate_keys

    last_error: Optional[Exception] = None

    for key in keys_to_try:
        if key is None:
            continue

        try:
            raw = _extract_by_dotted_key(data, key)
        except KeyError as e:
            last_error = e
            continue

        try:
            return normalize_equity_paths(raw)
        except Exception as e:
            last_error = e
            raise ValueError(
                f"Found equity paths at key={key!r}, but they could not be normalized: {e}"
            ) from e

    available = sorted(str(k) for k in data.keys())
    raise KeyError(
        "Could not find simulated equity paths in portfolio_result. "
        f"Tried keys={list(keys_to_try)}. Available top-level keys={available}. "
        f"Last error={last_error}"
    )


def normalize_equity_paths(equity_paths: object) -> np.ndarray:
    """
    Normalize equity paths into a 2D float numpy array.

    DataFrame handling:
      - If columns are time steps and rows are paths, use as-is.
      - If the DataFrame has a date/time index and columns are paths, callers
        should pass transpose=True before using this function or pass `.T`.

    This function does not infer orientation automatically because silently
    transposing path matrices is dangerous.
    """
    if isinstance(equity_paths, pd.DataFrame):
        arr = equity_paths.to_numpy(dtype=float)
    else:
        arr = np.asarray(equity_paths, dtype=float)

    # Basic shape/finite validation only. Full horizon and initial-value
    # validation happens in evaluate_portfolio_search_actuarial_risk().
    return validate_equity_paths(arr)


def build_actuarial_config_from_portfolio_context(
    *,
    initial_value: float,
    horizon_days: int,
    base_config: Optional[ActuarialRiskConfig] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ActuarialRiskConfig:
    """
    Build an ActuarialRiskConfig for portfolio-search evaluation.

    If base_config is provided, this function preserves its risk settings but
    replaces initial_value, horizon_days, and merges metadata.

    This is useful when portfolio search has a known simulation horizon and
    starting capital but the actuarial thresholds come from a standard config.
    """
    meta = dict(base_config.metadata) if base_config is not None else {}
    if metadata:
        meta.update(metadata)

    if base_config is None:
        return ActuarialRiskConfig(
            initial_value=float(initial_value),
            horizon_days=int(horizon_days),
            survival=SurvivalConfig(
                horizons_days=_default_survival_horizons_for_horizon(int(horizon_days))
            ),
            metadata=meta,
        ).validate()

    return ActuarialRiskConfig(
        initial_value=float(initial_value),
        horizon_days=int(horizon_days),
        trading_days_per_year=base_config.trading_days_per_year,
        ruin=base_config.ruin,
        drawdown=base_config.drawdown,
        goal=base_config.goal,
        recovery=base_config.recovery,
        survival=base_config.survival,
        capital_adequacy=base_config.capital_adequacy,
        metadata=meta,
    ).validate()


def evaluate_portfolio_search_actuarial_risk(
    portfolio_result: object,
    *,
    config: ActuarialRiskConfig,
    equity_paths_key: Optional[str] = None,
    portfolio_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> ActuarialRiskResult:
    """
    Evaluate actuarial risk for one portfolio-search result.

    This function is the main integration point for portfolio search.

    It:
      1. Extracts simulated equity paths.
      2. Validates them through the actuarial engine.
      3. Returns an ActuarialRiskResult.
      4. Adds portfolio-search metadata to the result metadata.

    It intentionally does not:
      - modify portfolio-search scoring,
      - quarantine portfolios,
      - write to S3,
      - mutate the input object.
    """
    paths = extract_equity_paths_from_portfolio_result(
        portfolio_result,
        equity_paths_key=equity_paths_key,
    )

    result = evaluate_actuarial_risk(paths, config=config)

    metadata = dict(result.metadata)
    metadata["integration"] = {
        "source": "portfolio_search",
        "equity_paths_key": equity_paths_key,
        "portfolio_id": portfolio_id,
        "run_id": run_id,
    }

    return ActuarialRiskResult(
        initial_value=result.initial_value,
        horizon_days=result.horizon_days,
        n_paths=result.n_paths,
        ruin_threshold=result.ruin_threshold,
        ruin_probability=result.ruin_probability,
        expected_time_to_ruin_days=result.expected_time_to_ruin_days,
        median_time_to_ruin_days=result.median_time_to_ruin_days,
        drawdown_limit_pct=result.drawdown_limit_pct,
        drawdown_breach_probability=result.drawdown_breach_probability,
        expected_max_drawdown=result.expected_max_drawdown,
        median_max_drawdown=result.median_max_drawdown,
        cvar_max_drawdown_95=result.cvar_max_drawdown_95,
        goal_value=result.goal_value,
        goal_probability=result.goal_probability,
        median_time_to_goal_days=result.median_time_to_goal_days,
        probability_goal_before_ruin=result.probability_goal_before_ruin,
        recovery_probability=result.recovery_probability,
        median_recovery_time_days=result.median_recovery_time_days,
        capital_required=result.capital_required,
        capital_buffer_gap=result.capital_buffer_gap,
        solvency_ratio=result.solvency_ratio,
        safe_leverage_estimate=result.safe_leverage_estimate,
        survival_curve=result.survival_curve,
        risk_grade=result.risk_grade,
        warnings=result.warnings,
        metadata=metadata,
    ).validate()


def evaluate_many_portfolio_search_actuarial_risks(
    portfolio_results: list[object],
    *,
    config: ActuarialRiskConfig,
    equity_paths_key: Optional[str] = None,
) -> list[ActuarialRiskResult]:
    """
    Evaluate actuarial risk for many portfolio-search results.

    This is useful for offline diagnostics or reports.

    It intentionally returns only results. It does not rank, filter, score,
    quarantine, or persist anything.
    """
    results: list[ActuarialRiskResult] = []

    for i, portfolio_result in enumerate(portfolio_results):
        data = _as_mapping(portfolio_result)

        portfolio_id = (
            str(data.get("portfolio_id"))
            if data.get("portfolio_id") is not None
            else str(data.get("candidate_id"))
            if data.get("candidate_id") is not None
            else None
        )

        run_id = str(data.get("run_id")) if data.get("run_id") is not None else None

        result = evaluate_portfolio_search_actuarial_risk(
            portfolio_result,
            config=config,
            equity_paths_key=equity_paths_key,
            portfolio_id=portfolio_id or f"portfolio_index_{i}",
            run_id=run_id,
        )
        results.append(result)

    return results