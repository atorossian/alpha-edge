# src/alpha_edge/risk/actuarial/solvency.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from alpha_edge.core.schemas import CapitalAdequacyConfig


@dataclass(frozen=True)
class CapitalAdequacyResult:
    """
    Result of the capital adequacy calculation.

    capital_required:
        Estimated capital required to satisfy the configured risk tolerance.

    capital_buffer_gap:
        current capital - capital_required.
        Positive means surplus capital.
        Negative means capital shortfall.

    solvency_ratio:
        current capital / capital_required.
        Values above 1 indicate capital adequacy under this model.

    safe_leverage_estimate:
        Conservative leverage estimate based on current leverage and solvency ratio.
        Capped by max_allowed_leverage.
    """

    capital_required: Optional[float]
    capital_buffer_gap: Optional[float]
    solvency_ratio: Optional[float]
    safe_leverage_estimate: Optional[float]
    warnings: list[str]


def _validate_loss_array(losses: object) -> np.ndarray:
    arr = np.asarray(losses, dtype=float)

    if arr.ndim != 1:
        raise ValueError("losses must be a 1D array")

    if arr.size == 0:
        raise ValueError("losses cannot be empty")

    if not np.all(np.isfinite(arr)):
        raise ValueError("losses contains NaN or infinite values")

    return arr


def calculate_path_capital_losses(
    equity_paths: object,
    *,
    initial_value: float,
) -> np.ndarray:
    """
    Calculate maximum capital loss per path.

    For each path:
        capital_loss = initial_value - minimum_equity_on_path

    The value is floored at 0.

    Example:
        initial_value = 100
        path minimum = 70
        capital loss = 30

    This is not a realized PnL measure. It is a path solvency stress measure.
    """
    paths = np.asarray(equity_paths, dtype=float)

    if paths.ndim != 2:
        raise ValueError("equity_paths must be a 2D array")

    if paths.shape[0] <= 0 or paths.shape[1] <= 1:
        raise ValueError("equity_paths must contain at least one path and two time steps")

    if not np.all(np.isfinite(paths)):
        raise ValueError("equity_paths contains NaN or infinite values")

    iv = float(initial_value)
    if iv <= 0:
        raise ValueError("initial_value must be > 0")

    min_equity = np.min(paths, axis=1)
    losses = np.maximum(iv - min_equity, 0.0)

    return losses.astype(float)


def calculate_capital_required_from_losses(
    losses: object,
    *,
    target_ruin_probability: float,
    min_solvent_capital_ratio: float = 1.0,
) -> float:
    """
    Estimate required capital from simulated path losses.

    We use the loss quantile implied by target_ruin_probability.

    Example:
        target_ruin_probability = 0.05
        required capital = 95th percentile of path capital losses

    Then we multiply by min_solvent_capital_ratio.

    This answers:
        How much capital should be available to absorb losses such that only
        target_ruin_probability of simulated paths exceed that loss amount?
    """
    arr = _validate_loss_array(losses)

    p = float(target_ruin_probability)
    if not 0.0 < p < 1.0:
        raise ValueError("target_ruin_probability must be > 0 and < 1")

    ratio = float(min_solvent_capital_ratio)
    if ratio <= 0.0:
        raise ValueError("min_solvent_capital_ratio must be > 0")

    quantile_level = 1.0 - p
    required = float(np.quantile(arr, quantile_level))

    return float(required * ratio)


def calculate_solvent_capital_ratio(
    *,
    current_capital: float,
    capital_required: float,
) -> Optional[float]:
    """
    Calculate solvency ratio.

    Ratio = current_capital / capital_required.

    If capital_required is zero, return None because there is no meaningful
    denominator. The caller should interpret this as no observed simulated
    capital loss under the supplied paths.
    """
    current = float(current_capital)
    required = float(capital_required)

    if current <= 0.0:
        raise ValueError("current_capital must be > 0")

    if required < 0.0:
        raise ValueError("capital_required must be >= 0")

    if required == 0.0:
        return None

    return float(current / required)


def estimate_safe_leverage(
    *,
    current_leverage: float,
    solvency_ratio: Optional[float],
    max_allowed_leverage: float,
) -> Optional[float]:
    """
    Estimate safe leverage from solvency ratio.

    Conservative heuristic:
        safe_leverage = current_leverage * solvency_ratio

    Then cap at max_allowed_leverage.

    Important:
        current_leverage is the observed/current leverage.
        max_allowed_leverage is the policy cap.

        current_leverage may exceed max_allowed_leverage. That is not an invalid
        input. It should be reported as a diagnostic warning by the caller, not
        rejected here.
    """
    lev = float(current_leverage)
    max_lev = float(max_allowed_leverage)

    if lev <= 0.0:
        raise ValueError("current_leverage must be > 0")

    if max_lev <= 0.0:
        raise ValueError("max_allowed_leverage must be > 0")

    if solvency_ratio is None:
        return float(max_lev)

    ratio = float(solvency_ratio)

    if ratio < 0.0:
        raise ValueError("solvency_ratio must be >= 0")

    return float(min(max_lev, lev * ratio))

def evaluate_capital_adequacy(
    equity_paths: object,
    *,
    initial_value: float,
    config: CapitalAdequacyConfig,
) -> CapitalAdequacyResult:
    """
    Evaluate capital adequacy from simulated equity paths.

    This is the Step 3 solvency model.

    It estimates:
      - path capital losses,
      - required capital at configured confidence,
      - capital buffer gap,
      - solvency ratio,
      - safe leverage estimate.

    Important:
        This is based on simulated path losses, not a guarantee.
    """
    cfg = config.validate()

    warnings: list[str] = []

    if not cfg.enabled:
        return CapitalAdequacyResult(
            capital_required=None,
            capital_buffer_gap=None,
            solvency_ratio=None,
            safe_leverage_estimate=None,
            warnings=[],
        )

    iv = float(initial_value)

    losses = calculate_path_capital_losses(
        equity_paths,
        initial_value=iv,
    )

    capital_required = calculate_capital_required_from_losses(
        losses,
        target_ruin_probability=cfg.target_ruin_probability,
        min_solvent_capital_ratio=cfg.min_solvent_capital_ratio,
    )

    capital_buffer_gap = float(iv - capital_required)

    solvency_ratio = calculate_solvent_capital_ratio(
        current_capital=iv,
        capital_required=capital_required,
    )

    safe_leverage_estimate = estimate_safe_leverage(
        current_leverage=cfg.current_leverage,
        solvency_ratio=solvency_ratio,
        max_allowed_leverage=cfg.max_allowed_leverage,
    )

    if capital_buffer_gap < 0:
        warnings.append("Capital buffer gap is negative under the actuarial capital model.")

    if solvency_ratio is not None and solvency_ratio < 1.0:
        warnings.append("Solvency ratio is below 1.0.")

    if safe_leverage_estimate is not None and safe_leverage_estimate < cfg.current_leverage:
        warnings.append("Safe leverage estimate is below current leverage.")

    return CapitalAdequacyResult(
        capital_required=float(capital_required),
        capital_buffer_gap=float(capital_buffer_gap),
        solvency_ratio=solvency_ratio,
        safe_leverage_estimate=safe_leverage_estimate,
        warnings=warnings,
    )