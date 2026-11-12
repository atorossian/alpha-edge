from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from alpha_edge.core.schemas import (
    ActuarialDiagnosticReport,
    ActuarialRiskConfig,
    CapitalAdequacyConfig,
    DrawdownBreachConfig,
    GoalConfig,
    RecoveryConfig,
    RuinConfig,
    SurvivalConfig,
)
from alpha_edge.market.stats_engine import compute_daily_returns
from alpha_edge.portfolio.optimizer_engine import evaluate_portfolio_candidate_with_paths
from alpha_edge.risk.actuarial.diagnostic_report import (
    build_actuarial_diagnostic_report,
    evaluate_portfolio_search_actuarial_diagnostic,
)
from alpha_edge.risk.actuarial.engine import evaluate_actuarial_risk
from alpha_edge.risk.actuarial.formatters import (
    diagnostic_report_to_json_block,
    format_actuarial_diagnostic_report,
)


def _default_survival_horizons(horizon_days: int) -> list[int]:
    h = int(horizon_days)
    if h <= 0:
        raise ValueError("horizon_days must be > 0")

    out = [x for x in [21, 63, 126, 252, 756] if x <= h]
    return out or [h]


def build_portfolio_search_actuarial_diagnostic_section(
    portfolio_result: object,
    *,
    config: ActuarialRiskConfig,
    equity_paths_key: Optional[str] = None,
    portfolio_id: Optional[str] = None,
    run_id: Optional[str] = None,
    terminal_title: str = "ACTUARIAL RISK DIAGNOSTICS",
) -> tuple[ActuarialDiagnosticReport, str, dict[str, Any]]:
    """
    Build the actuarial diagnostic section for portfolio-search output.

    This is the main portfolio-search integration helper.

    It returns:
      1. Full ActuarialDiagnosticReport
      2. Terminal-ready text block
      3. Compact JSON-ready block for embedding in saved artifacts

    It intentionally does not:
      - change portfolio score,
      - change quarantine,
      - reject candidates,
      - mutate portfolio_result,
      - write to S3.
    """
    report = evaluate_portfolio_search_actuarial_diagnostic(
        portfolio_result,
        config=config,
        equity_paths_key=equity_paths_key,
        portfolio_id=portfolio_id,
        run_id=run_id,
    )

    terminal_text = format_actuarial_diagnostic_report(
        report,
        title=terminal_title,
        include_detail_metrics=True,
        include_flags=True,
        include_warnings=True,
    )

    json_block = diagnostic_report_to_json_block(report)

    return report, terminal_text, json_block


def build_actuarial_diagnostic_from_portfolio_report(
    *,
    report: object,
    closes: pd.DataFrame,
    goals: list[float] | tuple[float, ...],
    main_goal: float,
    score_config: object | None = None,
    portfolio_id: Optional[str] = None,
    run_id: Optional[str] = None,
    source: str = "portfolio_report",
    terminal_title: str = "ACTUARIAL RISK DIAGNOSTICS",
    current_leverage: Optional[float] = None,
    max_allowed_leverage: float = 2.0,
    ruin_threshold_fraction: float = 0.50,
    drawdown_limit_pct: float = 0.30,
    target_ruin_probability: float = 0.05,
    target_drawdown_breach_probability: float = 0.20,
    days: int = 252,
    n_paths: int = 20_000,
    mc_seed: int | None = 24681357,
    path_source: str = "bootstrap",
    pca_k: int = 5,
    block_size: int | tuple[int, int] | None = (8, 12),
    asset_returns: Optional[pd.DataFrame] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> tuple[ActuarialDiagnosticReport, str, dict[str, Any]]:
    """
    Build actuarial diagnostics from an existing PortfolioReport.

    This helper is used by daily report and quarantine analysis. It deliberately
    leaves report_engine.py unchanged and reuses the report's canonical snapshot
    and signed gross weights.

    It intentionally does not:
      - change portfolio score,
      - change quarantine decisions,
      - change daily-report decisions,
      - write to S3.
    """
    if closes is None or closes.empty:
        raise ValueError("closes is empty")

    snapshot = getattr(report, "snapshot", None)
    eval_metrics = getattr(report, "eval", None)
    if snapshot is None or eval_metrics is None:
        raise ValueError("report must expose snapshot and eval attributes")

    weights = getattr(eval_metrics, "weights", None)
    if not isinstance(weights, dict) or not weights:
        raise ValueError("report.eval.weights must be a non-empty dict")

    weight_keys = [str(k).strip() for k in weights.keys() if str(k).strip()]

    if asset_returns is not None:
        if asset_returns.empty:
            raise ValueError("asset_returns is empty")
        asset_returns_norm = asset_returns.copy()
        asset_returns_norm.columns = [str(c).strip() for c in asset_returns_norm.columns]
        cols = [k for k in weight_keys if k in asset_returns_norm.columns]
        if not cols:
            raise ValueError("No report evaluation keys overlap with asset_returns columns")
        missing = [k for k in weight_keys if k not in asset_returns_norm.columns]
        if missing:
            raise ValueError("Missing asset_returns columns for report evaluation key(s): " + ", ".join(missing[:20]))
        returns = asset_returns_norm[cols].dropna(how="any")
        weights_for_eval = {str(k).strip(): float(weights[k]) for k in cols}
        return_key_mode = "asset_returns_columns"
    else:
        if closes is None or closes.empty:
            raise ValueError("closes is empty")
        closes_norm = closes.copy()
        closes_norm.columns = [str(c).upper().strip() for c in closes_norm.columns]
        tickers = [str(t).upper().strip() for t in weight_keys if str(t).upper().strip() in closes_norm.columns]
        if not tickers:
            raise ValueError("No report tickers overlap with closes columns")
        closes_sub = closes_norm[tickers].dropna(how="any")
        if closes_sub.shape[0] < 50:
            raise ValueError("Not enough close history to run actuarial diagnostic")
        returns = compute_daily_returns(closes_sub)
        weights_for_eval = {str(k).upper().strip(): float(weights[k]) for k in tickers}
        return_key_mode = "closes_ticker_columns"

    if returns.shape[0] < 50:
        raise ValueError("Not enough return history to run actuarial diagnostic")

    equity0 = float(getattr(snapshot, "equity"))
    notional = float(getattr(snapshot, "total_notional"))
    observed_leverage = (
        float(current_leverage)
        if current_leverage is not None
        else float(getattr(snapshot, "leverage"))
    )

    eval_with_paths = evaluate_portfolio_candidate_with_paths(
        returns=returns,
        weights=weights_for_eval,
        equity0=equity0,
        notional=notional,
        goals=[float(g) for g in goals],
        main_goal=float(main_goal),
        lw_cov=None,
        days=int(days),
        n_paths=int(n_paths),
        score_config=score_config,
        mc_seed=mc_seed,
        path_source=str(path_source),
        pca_k=int(pca_k),
        block_size=block_size,
        weight_mode="gross_signed",
    )

    equity_paths = eval_with_paths.equity_paths
    if equity_paths is None:
        raise RuntimeError("Diagnostic evaluator did not return equity_paths")

    horizon_days = int(np.asarray(equity_paths).shape[1] - 1)
    if horizon_days <= 0:
        raise RuntimeError("Diagnostic evaluator returned invalid equity path horizon")

    config_metadata: dict[str, Any] = {
        "source": source,
        "run_id": run_id,
        "portfolio_id": portfolio_id,
        "diagnostic_mc_seed": mc_seed,
        "diagnostic_n_paths": int(n_paths),
        "diagnostic_note": (
            "Actuarial diagnostics are informational only and do not affect "
            "portfolio scoring, quarantine status, or daily-report decisions."
        ),
        "return_key_mode": return_key_mode,
        "return_columns": list(returns.columns),
    }
    if metadata:
        config_metadata.update(metadata)

    config = ActuarialRiskConfig(
        initial_value=equity0,
        horizon_days=horizon_days,
        ruin=RuinConfig(
            threshold_mode="fraction_of_initial",
            threshold_value=float(ruin_threshold_fraction),
        ),
        drawdown=DrawdownBreachConfig(
            drawdown_limit_pct=float(drawdown_limit_pct),
        ),
        goal=GoalConfig(
            enabled=True,
            goal_value=float(main_goal),
        ),
        recovery=RecoveryConfig(
            enabled=True,
            recovery_level=1.0,
        ),
        survival=SurvivalConfig(
            horizons_days=_default_survival_horizons(horizon_days),
        ),
        capital_adequacy=CapitalAdequacyConfig(
            enabled=True,
            target_ruin_probability=float(target_ruin_probability),
            target_drawdown_breach_probability=float(target_drawdown_breach_probability),
            current_leverage=float(observed_leverage),
            max_allowed_leverage=float(max_allowed_leverage),
        ),
        metadata=config_metadata,
    )

    result = evaluate_actuarial_risk(equity_paths, config=config)

    diagnostic_report = build_actuarial_diagnostic_report(
        result,
        portfolio_id=portfolio_id,
        run_id=run_id,
        source=source,
        metadata={"integration_step": source},
    )

    terminal_text = format_actuarial_diagnostic_report(
        diagnostic_report,
        title=terminal_title,
        include_detail_metrics=True,
        include_flags=True,
        include_warnings=True,
    )

    json_block = diagnostic_report_to_json_block(diagnostic_report)

    return diagnostic_report, terminal_text, json_block


def attach_actuarial_diagnostic_to_output_payload(
    output_payload: dict[str, Any],
    *,
    diagnostic_block: dict[str, Any],
    key: str = "actuarial_diagnostics",
) -> dict[str, Any]:
    """
    Return a copy of an output payload with an actuarial diagnostic block attached.

    This avoids mutating the original payload unless the caller explicitly assigns it.
    """
    if not isinstance(output_payload, dict):
        raise TypeError("output_payload must be a dict")

    out = dict(output_payload)
    out[key] = diagnostic_block
    return out


def maybe_print_actuarial_diagnostic_section(
    terminal_text: str,
    *,
    enabled: bool = True,
) -> None:
    """
    Print diagnostic text if enabled.
    """
    if not enabled:
        return

    print()
    print(terminal_text)
