from __future__ import annotations

from dataclasses import dataclass

from alpha_edge.core.schemas import ScoreConfig
from alpha_edge.portfolio.evaluation_service import (
    ASSET_IDENTITY_MODE,
    EVALUATOR_VERSION,
    HEALTH_SCORE_VERSION,
    build_evaluation_metadata,
    build_metric_tolerance_policy,
    build_plausibility_guards,
    compare_metric_values,
    compute_portfolio_health_score,
)


@dataclass
class FakeMetrics:
    p_hit_goal_1_1y: float = 0.40
    p_hit_goal_2_1y: float = 0.75
    p_hit_goal_3_1y: float = 0.20
    ruin_prob_1y: float = 0.01
    max_drawdown: float = 5e-7  # tiny positive value tolerated as float noise
    ann_return: float = 0.12
    ann_vol: float = 0.20
    var_95: float = -0.02
    cvar_95: float = -0.03
    stability_energy: float = 0.10
    path_mdd_mean: float = 0.12
    cdar_95: float = 0.18
    p_dd_breach: float = 0.05
    underwater_mean: float = 0.20
    ttr_mean_days: float = 40.0
    score: float = -1.23


def _health(metrics: FakeMetrics) -> dict:
    return compute_portfolio_health_score(
        final_metrics=metrics,
        execution_quality={
            "deployment_ratio": 1.0,
            "cash_weight": 0.0,
            "weight_drift_l1": 0.0,
            "dropped_theoretical_weight": 0.0,
        },
        score_cfg=ScoreConfig(),
        goals=(7500.0, 10000.0, 12500.0),
        main_goal=10000.0,
        max_cash_weight=0.20,
        min_deployment_ratio=1.0,
        max_executable_mdd=0.40,
        max_executable_cdar_95=0.60,
        max_stability_energy=2.0,
        max_dropped_weight=1e-12,
        max_weight_drift_l1=1e-12,
    )


def test_evaluation_metadata_includes_versions_semantics_and_tolerance_policy():
    meta = build_evaluation_metadata(
        returns_eval_meta={"source": "returns_wide_min5y", "key": "engine/v1/market/cache/v1/returns.parquet"},
        price_source="latest_close_prices",
        market_regime_source="engine/v1/regimes/market_hmm/latest.json",
        score_config_version="score_config_latest",
        run_id="daily_report_test",
        as_of="2026-08-11",
    )

    assert meta["evaluator_version"] == EVALUATOR_VERSION
    assert meta["health_score_version"] == HEALTH_SCORE_VERSION
    assert meta["asset_identity_mode"] == ASSET_IDENTITY_MODE
    assert meta["score_semantics"]["optimizer_score"].startswith("model/ranking score")
    assert meta["tolerance_policy"]["raw_float_abs_tol"] == 1e-6
    assert meta["tolerance_policy"]["display_percentage_point_tol"] == 0.01
    assert meta["tolerance_policy"]["money_abs_tol_usd"] == 0.01


def test_plausibility_guards_use_tolerance_and_validate_metadata():
    metrics = FakeMetrics()
    health = _health(metrics)
    meta = build_evaluation_metadata(
        returns_eval_meta={"source": "returns_wide_min5y", "key": "engine/v1/returns.parquet"},
        price_source="latest_close_prices",
        market_regime_source="engine/v1/regimes/market_hmm/latest.json",
        score_config_version="score_config_latest",
        run_id="daily_report_test",
        as_of="2026-08-11",
    )
    health["metadata"] = meta

    guards = build_plausibility_guards(
        metrics=metrics,
        returns_rows=300,
        returns_assets=2,
        health_score_payload=health,
        evaluation_metadata=meta,
        asset_ids=["CRYPTO:BTC-USD", "CRYPTO:ETH-USD"],
    )

    assert guards["schema_version"] == "portfolio_metric_plausibility_v2"
    assert guards["ok"] is True
    assert guards["flags"] == []
    assert guards["metadata_missing"] == []


def test_plausibility_guards_flag_clear_metric_errors_and_missing_asset_ids():
    metrics = FakeMetrics(max_drawdown=0.10, ann_vol=-0.02, ruin_prob_1y=1.25)
    guards = build_plausibility_guards(
        metrics=metrics,
        returns_rows=10,
        returns_assets=0,
        asset_ids=["CRYPTO:BTC-USD", ""],
    )

    assert guards["ok"] is False
    assert "max_drawdown_positive" in guards["flags"]
    assert "ann_vol_negative" in guards["flags"]
    assert "ruin_prob_out_of_range" in guards["flags"]
    assert "insufficient_returns_rows" in guards["flags"]
    assert "no_returns_assets" in guards["flags"]
    assert "missing_asset_ids" in guards["flags"]


def test_metric_comparison_uses_configurable_drift_tolerance():
    policy = build_metric_tolerance_policy(metric_drift_tolerance={"health_score": 0.05})

    within = compare_metric_values(
        actual=75.03,
        expected=75.00,
        metric_name="health_score",
        tolerance_policy=policy,
    )
    outside = compare_metric_values(
        actual=75.08,
        expected=75.00,
        metric_name="health_score",
        tolerance_policy=policy,
    )

    assert within["within_tolerance"] is True
    assert outside["within_tolerance"] is False
