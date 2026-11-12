# evaluation_service.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import numpy as np
import pandas as pd

from alpha_edge.market.hmm_engine import (
    GaussianHMM,
    compute_state_diagnostics,
    label_states_4,
    regime_probs_from_state_probs,
    select_regime_label,
)

from alpha_edge.core.schemas import ScoreConfig


EVALUATOR_VERSION = "canonical_portfolio_evaluator_v1"
HEALTH_SCORE_VERSION = "portfolio_health_score_v2"
ASSET_IDENTITY_MODE = "asset_id_first"
METRIC_TOLERANCE_SCHEMA_VERSION = "metric_tolerance_policy_v1"
RAW_FLOAT_ABS_TOL = 1e-6
DISPLAY_PERCENTAGE_POINT_TOL = 0.01
MONEY_ABS_TOL_USD = 0.01


@dataclass(frozen=True)
class MetricTolerancePolicy:
    schema_version: str = METRIC_TOLERANCE_SCHEMA_VERSION
    raw_float_abs_tol: float = RAW_FLOAT_ABS_TOL
    display_percentage_point_tol: float = DISPLAY_PERCENTAGE_POINT_TOL
    money_abs_tol_usd: float = MONEY_ABS_TOL_USD
    metric_drift_tolerance: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["metric_drift_tolerance"] = dict(self.metric_drift_tolerance or {})
        return out


def build_metric_tolerance_policy(
    *,
    metric_drift_tolerance: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Central tolerance policy for report/test comparisons.

    Defaults reflect the Daily Report Analytics Consistency Repair decision:
      - raw float absolute tolerance: 1e-6
      - display tolerance: 0.01 percentage points
      - money/PnL tolerance: 0.01 USD
      - metric drift tolerance: configurable per metric
    """
    drift = {str(k): float(v) for k, v in dict(metric_drift_tolerance or {}).items()}
    return MetricTolerancePolicy(metric_drift_tolerance=drift).to_dict()


def _within_abs_tolerance(value: float, target: float, tolerance: float) -> bool:
    try:
        v = float(value)
        t = float(target)
        tol = abs(float(tolerance))
    except Exception:
        return False
    if not np.isfinite(v) or not np.isfinite(t) or not np.isfinite(tol):
        return False
    return abs(v - t) <= tol


def compare_metric_values(
    *,
    actual: float,
    expected: float,
    metric_name: str,
    tolerance_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare two metric values using configurable per-metric tolerances.

    This is intentionally lightweight so tests and diagnostics do not use exact
    equality for floating/Monte-Carlo-derived metrics.
    """
    policy = dict(tolerance_policy or build_metric_tolerance_policy())
    metric_drift = dict(policy.get("metric_drift_tolerance") or {})
    tol = float(metric_drift.get(str(metric_name), policy.get("raw_float_abs_tol", RAW_FLOAT_ABS_TOL)))
    diff = float(actual) - float(expected)
    return {
        "metric": str(metric_name),
        "actual": float(actual),
        "expected": float(expected),
        "diff": float(diff),
        "abs_diff": float(abs(diff)),
        "tolerance": float(abs(tol)),
        "within_tolerance": bool(_within_abs_tolerance(float(actual), float(expected), float(tol))),
        "tolerance_policy_version": str(policy.get("schema_version", METRIC_TOLERANCE_SCHEMA_VERSION)),
    }


@dataclass(frozen=True)
class EvaluationMetadata:
    evaluator_version: str = EVALUATOR_VERSION
    health_score_version: str = HEALTH_SCORE_VERSION
    asset_identity_mode: str = ASSET_IDENTITY_MODE
    returns_source: str | None = None
    returns_source_key: str | None = None
    price_source: str | None = None
    market_regime_source: str | None = None
    score_config_version: str | None = None
    score_semantics: dict[str, str] | None = None
    tolerance_policy: dict[str, Any] | None = None
    run_id: str | None = None
    as_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _normalize_regime_label(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s:
        return None

    aliases = {
        "MIXED / NEUTRAL": "MIXED",
        "MIXED/NEUTRAL": "MIXED",
        "MIXED_NEUTRAL": "MIXED",
        "NEUTRAL / MIXED": "MIXED",
        "NEUTRAL/MIXED": "MIXED",
        "NONE": None,
        "UNKNOWN": None,
        "N/A": None,
        "NA": None,
    }
    if s in aliases:
        return aliases[s]

    for lab in ("STRESS_BEAR", "CHOPPY_BEAR", "CHOPPY_BULL", "CALM_BULL", "MIXED", "NEUTRAL"):
        if lab in s:
            return "MIXED" if lab == "NEUTRAL" else lab

    return s


def _regime_strength_rank(label: str | None) -> int | None:
    """Higher means stronger/more constructive behavior."""
    label = _normalize_regime_label(label)
    if label is None:
        return None
    ranks = {
        "STRESS_BEAR": 0,
        "CHOPPY_BEAR": 1,
        "MIXED": 2,
        "NEUTRAL": 2,
        "CHOPPY_BULL": 3,
        "CALM_BULL": 4,
    }
    return ranks.get(label)


def build_regime_alignment(
    *,
    market_regime_label: str | None,
    portfolio_behavior_label: str | None,
) -> dict[str, Any]:
    """
    Compare the canonical market regime with the portfolio's own behavior regime.

    This does not replace the market regime source of truth. It classifies whether
    the portfolio is behaving better/worse than the current market state.
    """
    market_label = _normalize_regime_label(market_regime_label)
    portfolio_label = _normalize_regime_label(portfolio_behavior_label)

    mr = _regime_strength_rank(market_label)
    pr = _regime_strength_rank(portfolio_label)

    if market_label is None or portfolio_label is None or mr is None or pr is None:
        status = "unknown"
        description = "Regime alignment could not be classified because one or both labels are missing or unrecognized."
    elif pr > mr:
        status = "positive_divergence"
        description = "Portfolio behavior is stronger than the current market regime."
    elif pr < mr:
        status = "negative_divergence"
        description = "Portfolio behavior is weaker than the current market regime."
    else:
        status = "aligned"
        description = "Portfolio behavior is broadly aligned with the current market regime."

    return {
        "status": status,
        "description": description,
        "market_regime_label": market_label,
        "portfolio_behavior_label": portfolio_label,
    }


def build_portfolio_behavior_regime(
    *,
    portfolio_returns: pd.Series,
    market_regime_payload: Mapping[str, Any] | None = None,
    min_observations: int = 252,
    commit_threshold: float = 0.65,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Fit a local HMM to the portfolio return path and compare it against the
    canonical morning market regime.

    Important naming convention:
      - market_regime is the source-of-truth macro/market state.
      - portfolio_behavior_regime is only a diagnostic of how the current
        portfolio is behaving.

    The portfolio behavior diagnostic should never overwrite or act as the
    market-regime source of truth.
    """
    market_regime_payload = dict(market_regime_payload or {})
    market_hmm = market_regime_payload.get("hmm") if isinstance(market_regime_payload, Mapping) else None
    if not isinstance(market_hmm, Mapping):
        market_hmm = {}

    # Use the committed internal HMM label when available. If no label is
    # committed, classify the market state as MIXED rather than comparing the
    # portfolio behavior regime against a low-confidence top probability label.
    market_label = (
        market_hmm.get("label_commit")
        or market_regime_payload.get("label_commit")
    )
    if not market_label:
        p_label = market_hmm.get("p_label_today") or market_regime_payload.get("p_label_today") or {}
        if isinstance(p_label, Mapping) and p_label:
            market_label = "MIXED"
        else:
            market_label = (market_regime_payload.get("leverage_recommendation") or {}).get("label")

    r = pd.Series(portfolio_returns).dropna().astype(float)
    r = r[np.isfinite(r.to_numpy(dtype=float))]

    base = {
        "source": "portfolio_hmm_diagnostic",
        "description": "Local diagnostic fitted to the current portfolio return path; not the market-regime source of truth.",
        "min_observations": int(min_observations),
        "commit_threshold": float(commit_threshold),
        "n_observations": int(len(r)),
        "market_regime": {
            "source": "regimes/market_hmm/latest.json",
            "label": _normalize_regime_label(market_label),
        },
    }

    if len(r) < int(min_observations):
        return {
            **base,
            "ok": False,
            "label": None,
            "confidence": None,
            "p_label_today": {},
            "state_diagnostics": [],
            "regime_alignment": build_regime_alignment(
                market_regime_label=market_label,
                portfolio_behavior_label=None,
            ),
            "reason": f"not_enough_observations: {len(r)} < {int(min_observations)}",
        }

    try:
        X = r.to_numpy(dtype=np.float64).reshape(-1, 1)
        hmm = GaussianHMM(n_states=4, n_dim=1, seed=int(seed))
        fit = hmm.fit(X, max_iter=100, tol=1e-4, verbose=False)
        gamma = hmm.predict_proba(X)
        diags = compute_state_diagnostics(X[:, 0], gamma)
        mapping = label_states_4(diags)
        p_label_today = regime_probs_from_state_probs(gamma[-1], mapping)
        label = select_regime_label(p_label_today, commit_threshold=float(commit_threshold)) or "MIXED"
        confidence = float(max(p_label_today.values())) if p_label_today else None

        return {
            **base,
            "ok": True,
            "label": _normalize_regime_label(label),
            "confidence": confidence,
            "p_label_today": {str(k): float(v) for k, v in p_label_today.items()},
            "state_to_label": {str(k): str(v) for k, v in mapping.items()},
            "state_diagnostics": [
                {
                    "state": int(i),
                    "label": str(mapping.get(i)),
                    "drift": float(d.drift),
                    "vol": float(d.vol),
                    "neg_rate": float(d.neg_rate),
                    "weight": float(d.weight),
                }
                for i, d in enumerate(diags)
            ],
            "fit": {
                "loglik": float(fit.loglik),
                "n_iter": int(fit.n_iter),
                "converged": bool(fit.converged),
            },
            "regime_alignment": build_regime_alignment(
                market_regime_label=market_label,
                portfolio_behavior_label=label,
            ),
            "reason": None,
        }
    except Exception as e:
        return {
            **base,
            "ok": False,
            "label": None,
            "confidence": None,
            "p_label_today": {},
            "state_diagnostics": [],
            "regime_alignment": build_regime_alignment(
                market_regime_label=market_label,
                portfolio_behavior_label=None,
            ),
            "reason": f"{type(e).__name__}: {e}",
        }

def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return float(min(1.0, max(0.0, v)))


def _safe_ratio_good(value: float, cap: float, *, lower_is_better: bool = True) -> float:
    """Return 0..1 where 1 is good and 0 is at/above cap for lower-is-better metrics."""
    try:
        v = abs(float(value))
        c = abs(float(cap))
    except Exception:
        return 0.0
    if not np.isfinite(v) or not np.isfinite(c) or c <= 0:
        return 0.0
    if lower_is_better:
        return _clamp01(1.0 - (v / c))
    return _clamp01(v / c)


def _metric_p_main(metrics: Any, goals: tuple[float, float, float] | list[float], main_goal: float) -> float:
    goal_values = [float(x) for x in goals]
    probs = [
        getattr(metrics, "p_hit_goal_1_1y", np.nan),
        getattr(metrics, "p_hit_goal_2_1y", np.nan),
        getattr(metrics, "p_hit_goal_3_1y", np.nan),
    ]

    try:
        mg = float(main_goal)
    except Exception:
        return float(probs[0])

    idx = int(np.argmin([abs(g - mg) for g in goal_values]))
    return float(probs[idx])


def _grade_from_health_score(health_score: float) -> str:
    if health_score >= 80:
        return "A"
    if health_score >= 70:
        return "B"
    if health_score >= 60:
        return "C"
    if health_score >= 50:
        return "D"
    return "F"


def build_evaluation_metadata(
    *,
    returns_eval_meta: Mapping[str, Any] | None = None,
    price_source: str | None = None,
    market_regime_source: str | None = None,
    score_config_version: str | None = None,
    run_id: str | None = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """
    Standard metadata block attached to evaluation consumers.

    `returns_eval_meta` intentionally accepts the existing daily-report/search
    metadata dictionaries so consumers can expose source paths without knowing
    the implementation details of the loader.
    """
    returns_eval_meta = dict(returns_eval_meta or {})

    return EvaluationMetadata(
        returns_source=(
            returns_eval_meta.get("source")
            or returns_eval_meta.get("returns_source")
            or returns_eval_meta.get("name")
        ),
        returns_source_key=(
            returns_eval_meta.get("key")
            or returns_eval_meta.get("path")
            or returns_eval_meta.get("returns_source_key")
        ),
        price_source=price_source,
        market_regime_source=market_regime_source,
        score_config_version=score_config_version,
        score_semantics={
            "optimizer_score": "model/ranking score from evaluate_portfolio(); may be negative and is not a 0-100 health score",
            "health_score": "human/reporting score normalized to 0-100",
        },
        tolerance_policy=build_metric_tolerance_policy(),
        run_id=run_id,
        as_of=as_of,
    ).to_dict()


def compute_portfolio_health_score(
    *,
    final_metrics: Any,
    execution_quality: Mapping[str, Any] | None,
    score_cfg: ScoreConfig,
    goals: tuple[float, float, float] | list[float],
    main_goal: float,
    max_cash_weight: float,
    min_deployment_ratio: float,
    max_executable_mdd: float,
    max_executable_cdar_95: float,
    max_stability_energy: float,
    max_dropped_weight: float,
    max_weight_drift_l1: float,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Canonical human-facing 0-100 portfolio health score.

    This is intentionally separate from `final_metrics.score`:
      - `optimizer_score` / `raw_optimizer_score` is the model ranking objective.
      - `health_score` is the normalized reporting / validation score.

    All consumers (portfolio search, quarantine, daily report, transition) should
    call this function instead of maintaining local health-score formulas.
    """
    execution_quality = dict(execution_quality or {})
    metadata = dict(metadata or {})

    goals_tuple = tuple(float(g) for g in goals)
    if len(goals_tuple) != 3:
        raise ValueError("compute_portfolio_health_score currently expects exactly three goals")

    p_main = _metric_p_main(final_metrics, goals_tuple, main_goal)

    ruin_cap = float(getattr(score_cfg, "ruin_cap", 0.10) or 0.10)
    cvar_cap = float(getattr(score_cfg, "cvar_cap", 0.03))
    path_mdd_cap = float(getattr(score_cfg, "path_mdd_mean_cap", 0.30))
    p_dd_breach_cap = float(getattr(score_cfg, "p_dd_breach_cap", 0.25))
    underwater_cap = float(getattr(score_cfg, "underwater_mean_cap", 1.00))
    ttr_cap = float(getattr(score_cfg, "ttr_cap_days", 252.0))

    final_ruin = float(getattr(final_metrics, "ruin_prob_1y", np.nan))
    final_mdd = abs(float(getattr(final_metrics, "max_drawdown", np.nan)))
    final_cvar = abs(float(getattr(final_metrics, "cvar_95", np.nan)))
    final_stability = float(getattr(final_metrics, "stability_energy", np.nan))
    final_path_mdd = float(getattr(final_metrics, "path_mdd_mean", np.nan))
    final_cdar = float(getattr(final_metrics, "cdar_95", np.nan))
    final_p_breach = float(getattr(final_metrics, "p_dd_breach", np.nan))
    final_underwater = float(getattr(final_metrics, "underwater_mean", np.nan))
    final_ttr = float(getattr(final_metrics, "ttr_mean_days", np.nan))

    components = {
        "goal_probability": _clamp01(p_main),
        "ruin": _safe_ratio_good(final_ruin, ruin_cap),
        "max_drawdown": _safe_ratio_good(final_mdd, max_executable_mdd),
        "cvar_95": _safe_ratio_good(final_cvar, cvar_cap),
        "stability_energy": _safe_ratio_good(final_stability, max_stability_energy),
        "path_mdd_mean": _safe_ratio_good(final_path_mdd, path_mdd_cap),
        "cdar_95": _safe_ratio_good(final_cdar, max_executable_cdar_95),
        "p_dd_breach": _safe_ratio_good(final_p_breach, p_dd_breach_cap),
        "underwater_mean": _safe_ratio_good(final_underwater, underwater_cap),
        "ttr_mean_days": _safe_ratio_good(final_ttr, ttr_cap),
        "deployment": _clamp01(
            float(execution_quality.get("deployment_ratio", np.nan)) / float(min_deployment_ratio)
            if float(min_deployment_ratio) > 0
            else 0.0
        ),
        "cash": _safe_ratio_good(float(execution_quality.get("cash_weight", np.nan)), max_cash_weight),
        "weight_drift": _safe_ratio_good(float(execution_quality.get("weight_drift_l1", np.nan)), max_weight_drift_l1),
        "dropped_weight": _safe_ratio_good(float(execution_quality.get("dropped_theoretical_weight", np.nan)), max_dropped_weight),
    }

    risk_component = float(np.mean([components["ruin"], components["max_drawdown"], components["cvar_95"]]))
    stability_component = float(
        np.mean(
            [
                components["stability_energy"],
                components["path_mdd_mean"],
                components["cdar_95"],
                components["p_dd_breach"],
                components["underwater_mean"],
                components["ttr_mean_days"],
            ]
        )
    )
    execution_component = float(
        np.mean([components["deployment"], components["cash"], components["weight_drift"], components["dropped_weight"]])
    )

    weights = {
        "goal_probability": 0.30,
        "risk": 0.30,
        "stability": 0.30,
        "execution": 0.10,
    }
    health_score = 100.0 * (
        weights["goal_probability"] * components["goal_probability"]
        + weights["risk"] * risk_component
        + weights["stability"] * stability_component
        + weights["execution"] * execution_component
    )

    raw_optimizer_score = float(getattr(final_metrics, "score", np.nan))

    return {
        "schema_version": HEALTH_SCORE_VERSION,
        "health_score": float(health_score),
        "health_grade": _grade_from_health_score(float(health_score)),
        "optimizer_score": raw_optimizer_score,
        "raw_optimizer_score": raw_optimizer_score,
        "score_semantics": {
            "optimizer_score": "model/ranking score from evaluate_portfolio(); may be negative and is not a 0-100 health score",
            "health_score": "human/reporting score normalized to 0-100",
        },
        "components": {
            "goal_probability": float(components["goal_probability"]),
            "risk": risk_component,
            "stability": stability_component,
            "execution": execution_component,
        },
        "component_details": {k: float(v) for k, v in components.items()},
        "weights": {k: float(v) for k, v in weights.items()},
        "metadata": metadata,
        "note": "health_score is for validation/reporting; optimizer_score/raw_optimizer_score is for candidate ranking only.",
    }


def build_plausibility_guards(
    *,
    metrics: Any,
    returns_rows: int | None = None,
    returns_assets: int | None = None,
    min_returns_rows: int = 252,
    health_score_payload: Mapping[str, Any] | None = None,
    evaluation_metadata: Mapping[str, Any] | None = None,
    asset_ids: list[str] | tuple[str, ...] | None = None,
    metric_drift_tolerance: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Common sanity checks for evaluated metric payloads and metadata.

    These guards are intentionally non-blocking diagnostics. They make strange
    daily-report/search divergences visible without silently changing the score.
    """
    flags: list[str] = []
    warnings: list[str] = []
    policy = build_metric_tolerance_policy(metric_drift_tolerance=metric_drift_tolerance)
    raw_tol = float(policy["raw_float_abs_tol"])

    def _finite_attr(name: str) -> float:
        try:
            return float(getattr(metrics, name))
        except Exception:
            return float("nan")

    finite_required = [
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "var_95",
        "cvar_95",
        "ruin_prob_1y",
        "score",
    ]
    for name in finite_required:
        if not np.isfinite(_finite_attr(name)):
            flags.append(f"non_finite_{name}")

    ann_vol = _finite_attr("ann_vol")
    if np.isfinite(ann_vol) and ann_vol < -raw_tol:
        flags.append("ann_vol_negative")

    ruin = _finite_attr("ruin_prob_1y")
    if np.isfinite(ruin) and not (-raw_tol <= ruin <= 1.0 + raw_tol):
        flags.append("ruin_prob_out_of_range")

    for prob_name in ["p_hit_goal_1_1y", "p_hit_goal_2_1y", "p_hit_goal_3_1y", "p_dd_breach"]:
        p = _finite_attr(prob_name)
        if np.isfinite(p) and not (-raw_tol <= p <= 1.0 + raw_tol):
            flags.append(f"{prob_name}_out_of_range")

    max_dd = _finite_attr("max_drawdown")
    if np.isfinite(max_dd) and max_dd > raw_tol:
        flags.append("max_drawdown_positive")

    cvar = _finite_attr("cvar_95")
    var = _finite_attr("var_95")
    if np.isfinite(cvar) and cvar > raw_tol:
        flags.append("cvar_95_positive")
    if np.isfinite(var) and var > raw_tol:
        flags.append("var_95_positive")

    if returns_rows is not None and int(returns_rows) < int(min_returns_rows):
        flags.append("insufficient_returns_rows")

    if returns_assets is not None and int(returns_assets) <= 0:
        flags.append("no_returns_assets")

    if asset_ids is not None:
        clean_asset_ids = [str(x).strip() for x in asset_ids if str(x).strip()]
        if len(clean_asset_ids) != len(asset_ids):
            flags.append("missing_asset_ids")
        if len(set(clean_asset_ids)) != len(clean_asset_ids):
            flags.append("duplicate_asset_ids")

    hp = dict(health_score_payload or {})
    if hp:
        hs = hp.get("health_score")
        try:
            hs_f = float(hs)
        except Exception:
            hs_f = float("nan")
        if not np.isfinite(hs_f):
            flags.append("non_finite_health_score")
        elif not (-raw_tol <= hs_f <= 100.0 + raw_tol):
            flags.append("health_score_out_of_range")
        if hp.get("schema_version") != HEALTH_SCORE_VERSION:
            warnings.append("unexpected_health_score_version")
        if "optimizer_score" not in hp or "raw_optimizer_score" not in hp:
            warnings.append("missing_optimizer_score_semantics")

    meta = dict(evaluation_metadata or {})
    required_meta = [
        "evaluator_version",
        "health_score_version",
        "asset_identity_mode",
        "returns_source",
        "returns_source_key",
        "market_regime_source",
        "score_config_version",
        "score_semantics",
        "tolerance_policy",
        "run_id",
        "as_of",
    ]
    missing_meta = [k for k in required_meta if meta.get(k) in (None, "", {})]
    for k in missing_meta:
        warnings.append(f"missing_metadata_{k}")

    if meta and meta.get("evaluator_version") != EVALUATOR_VERSION:
        warnings.append("unexpected_evaluator_version")
    if meta and meta.get("health_score_version") != HEALTH_SCORE_VERSION:
        warnings.append("unexpected_metadata_health_score_version")
    if meta and meta.get("asset_identity_mode") != ASSET_IDENTITY_MODE:
        flags.append("asset_identity_mode_not_asset_id_first")

    return {
        "schema_version": "portfolio_metric_plausibility_v2",
        "ok": len(flags) == 0,
        "flags": flags,
        "warnings": warnings,
        "returns_rows": None if returns_rows is None else int(returns_rows),
        "returns_assets": None if returns_assets is None else int(returns_assets),
        "min_returns_rows": int(min_returns_rows),
        "tolerance_policy": policy,
        "metadata_required": required_meta,
        "metadata_missing": missing_meta,
    }


EXECUTION_SIGNAL_SCHEMA_VERSION = "daily_report_execution_signals_v1"


def _safe_float_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return float(v) if np.isfinite(v) else None


def _severity_from_trigger(triggered: bool, *, default: str = "medium") -> str:
    return str(default if bool(triggered) else "none")


def _extract_transition_ref(transition_assessment: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(transition_assessment or {})
    available = bool(payload)
    return {
        "available": available,
        "schema_version": payload.get("schema_version") if available else None,
        "as_of": payload.get("as_of") if available else None,
        "recommendation": payload.get("recommendation") if available else None,
        "reason": payload.get("reason") if available else None,
        "full_search_required": payload.get("full_search_required") if available else None,
        "local_optimization_allowed": payload.get("local_optimization_allowed") if available else None,
        "shadow_portfolio_required": payload.get("shadow_portfolio_required") if available else None,
        "delta_execution_allowed": payload.get("delta_execution_allowed") if available else None,
        "diagnostic_triggers": (payload.get("diagnostics") or {}).get("triggers") if isinstance(payload.get("diagnostics"), Mapping) else None,
    }


def build_daily_report_execution_signals(
    *,
    rescale_decision: Any,
    reoptimization_pressure: bool,
    take_profit: Mapping[str, Any] | None = None,
    transition_assessment: Mapping[str, Any] | None = None,
    current_health: Any | None = None,
) -> dict[str, Any]:
    """
    Build daily-report execution diagnostics without making them the final
    execution decision authority.

    Authority model:
      - daily report emits diagnostic signals only;
      - transition assessment is authoritative when available;
      - if transition assessment is unavailable, the daily report still marks
        itself as diagnostic-only fallback, not as an execution instruction.
    """
    take_profit = dict(take_profit or {})
    transition_ref = _extract_transition_ref(transition_assessment)

    rescale_triggered = bool(getattr(rescale_decision, "should_rebalance", False))
    rescale_reasons = list(getattr(rescale_decision, "reasons", []) or [])
    leverage_real = _safe_float_or_none(getattr(rescale_decision, "leverage_real", None))
    leverage_target = _safe_float_or_none(getattr(rescale_decision, "leverage_target", None))
    drift_ratio = _safe_float_or_none(getattr(rescale_decision, "drift_ratio", None))
    drift_abs = None if drift_ratio is None else abs(float(drift_ratio) - 1.0)

    tp_triggered = bool(take_profit.get("do_harvest", False))
    reopt_triggered = bool(reoptimization_pressure)

    if transition_ref["available"]:
        decision_authority = "transition_assessment"
        authority_status = "authoritative_transition_available"
        final_execution_decision = {
            "source": "transition_assessment",
            "recommendation": transition_ref.get("recommendation"),
            "reason": transition_ref.get("reason"),
        }
    else:
        decision_authority = "daily_report_diagnostic_only"
        authority_status = "transition_assessment_unavailable"
        final_execution_decision = {
            "source": "none",
            "recommendation": None,
            "reason": "Transition assessment is unavailable; daily report signals are diagnostics, not final execution instructions.",
        }

    return {
        "schema_version": EXECUTION_SIGNAL_SCHEMA_VERSION,
        "decision_authority": decision_authority,
        "authority_status": authority_status,
        "semantics": {
            "rescale": "Change gross exposure/leverage while preserving relative composition.",
            "rebalance": "Change relative allocation weights among assets.",
            "reoptimization_pressure": "Diagnostic pressure indicating the portfolio may need optimizer/search reassessment.",
            "transition_assessment": "Authoritative execution decision layer when available.",
        },
        "final_execution_decision": final_execution_decision,
        "transition_assessment_ref": transition_ref,
        "signals": {
            "rescale": {
                "triggered": rescale_triggered,
                "severity": _severity_from_trigger(rescale_triggered, default="medium"),
                "reason": ", ".join(rescale_reasons) if rescale_reasons else "no_rescale_pressure",
                "reasons": rescale_reasons,
                "leverage_real": leverage_real,
                "leverage_target": leverage_target,
                "drift_ratio": drift_ratio,
                "drift_abs": drift_abs,
            },
            "rebalance": {
                "triggered": False,
                "severity": "none",
                "reason": "composition_drift_not_decided_by_daily_report; transition/local optimizer owns executable allocation changes",
            },
            "reoptimization_pressure": {
                "triggered": reopt_triggered,
                "severity": _severity_from_trigger(reopt_triggered, default="high"),
                "reason": "portfolio_health_reoptimization_rule_triggered" if reopt_triggered else "portfolio_health_reoptimization_rule_not_triggered",
                "health_score": _safe_float_or_none(getattr(current_health, "score", None)) if current_health is not None else None,
            },
            "take_profit_harvest": {
                "triggered": tp_triggered,
                "severity": _severity_from_trigger(tp_triggered, default="medium"),
                "reason": ", ".join(take_profit.get("reasons") or []) if take_profit.get("reasons") else ("take_profit_harvest" if tp_triggered else "take_profit_not_active"),
                "m_star": _safe_float_or_none(take_profit.get("m_star")),
                "r_anchor": _safe_float_or_none(take_profit.get("r_anchor")),
                "dd": _safe_float_or_none(take_profit.get("dd")),
                "sharpe": _safe_float_or_none(take_profit.get("sharpe")),
            },
        },
    }
