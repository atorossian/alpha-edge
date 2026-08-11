from __future__ import annotations

import pandas as pd
import pytest

from alpha_edge.portfolio.regime_asset_preferences import (
    RegimeAssetPreference,
    RegimeAssetPreferenceConfig,
    assess_portfolio_regime_fit,
    build_portfolio_regime_fit_comparison,
    compute_asset_regime_preferences,
    normalize_regime_history,
    regime_fit_advantage,
)


def test_normalize_regime_history_accepts_label_or_mixed() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "label_or_mixed": ["CALM_BULL", None],
        }
    )

    out = normalize_regime_history(raw)

    assert list(out.columns) == ["date", "regime"]
    assert out["regime"].tolist() == ["CALM_BULL", "MIXED"]


def test_compute_asset_regime_preferences_ranks_assets() -> None:
    dates = pd.date_range("2024-01-01", periods=90, freq="D")

    returns = pd.DataFrame(
        {
            "STRONG": [0.002] * 90,
            "WEAK": [-0.002] * 90,
            "NOISY": [0.01 if i % 2 == 0 else -0.01 for i in range(90)],
        },
        index=dates,
    )

    regime_history = pd.DataFrame(
        {
            "date": dates,
            "regime": ["CALM_BULL"] * 90,
        }
    )

    cfg = RegimeAssetPreferenceConfig(min_obs=30)

    prefs = compute_asset_regime_preferences(
        returns_wide=returns,
        regime_history=regime_history,
        regime="CALM_BULL",
        cfg=cfg,
    )

    assert set(prefs) == {"STRONG", "WEAK", "NOISY"}
    assert prefs["STRONG"].preference_score is not None
    assert prefs["WEAK"].preference_score is not None
    assert prefs["STRONG"].preference_score > prefs["WEAK"].preference_score
    assert prefs["STRONG"].rank < prefs["WEAK"].rank


def test_compute_asset_regime_preferences_marks_low_obs_unknown() -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")

    returns = pd.DataFrame(
        {
            "AAPL": [0.001] * 20,
        },
        index=dates,
    )

    regime_history = pd.DataFrame(
        {
            "date": dates,
            "regime": ["STRESS_BEAR"] * 20,
        }
    )

    cfg = RegimeAssetPreferenceConfig(min_obs=60)

    prefs = compute_asset_regime_preferences(
        returns_wide=returns,
        regime_history=regime_history,
        regime="STRESS_BEAR",
        cfg=cfg,
    )

    assert prefs["AAPL"].obs == 20
    assert prefs["AAPL"].preference_score is None
    assert prefs["AAPL"].bucket == "UNKNOWN"


def test_assess_portfolio_regime_fit_scores_known_and_unknown_weights() -> None:
    prefs = {
        "AAPL": RegimeAssetPreference(
            asset_id="AAPL",
            regime="CALM_BULL",
            obs=100,
            ann_return=0.20,
            ann_vol=0.15,
            downside_vol=0.08,
            max_drawdown=-0.10,
            sharpe_like=1.33,
            preference_score=0.90,
            rank=1,
            bucket="STRONG",
        ),
        "MSFT": RegimeAssetPreference(
            asset_id="MSFT",
            regime="CALM_BULL",
            obs=100,
            ann_return=0.05,
            ann_vol=0.20,
            downside_vol=0.12,
            max_drawdown=-0.25,
            sharpe_like=0.25,
            preference_score=0.20,
            rank=10,
            bucket="WEAK",
        ),
    }

    fit = assess_portfolio_regime_fit(
        weights={"AAPL": 0.50, "MSFT": 0.30, "UNKNOWN": 0.20},
        preferences=prefs,
        regime="CALM_BULL",
    )

    assert fit.weighted_preference_score == pytest.approx((0.50 * 0.90 + 0.30 * 0.20) / 0.80)
    assert fit.strong_asset_weight == pytest.approx(0.50)
    assert fit.weak_asset_weight == pytest.approx(0.30)
    assert fit.unknown_asset_weight == pytest.approx(0.20)
    assert fit.asset_count == 3
    assert fit.covered_asset_count == 2


def test_regime_fit_advantage_compares_candidate_to_current() -> None:
    current = assess_portfolio_regime_fit(
        weights={"AAPL": 1.0},
        preferences={
            "AAPL": RegimeAssetPreference(
                asset_id="AAPL",
                regime="CALM_BULL",
                obs=100,
                ann_return=0.05,
                ann_vol=0.20,
                downside_vol=0.12,
                max_drawdown=-0.20,
                sharpe_like=0.25,
                preference_score=0.20,
                rank=10,
                bucket="WEAK",
            )
        },
        regime="CALM_BULL",
    )

    candidate = assess_portfolio_regime_fit(
        weights={"MSFT": 1.0},
        preferences={
            "MSFT": RegimeAssetPreference(
                asset_id="MSFT",
                regime="CALM_BULL",
                obs=100,
                ann_return=0.20,
                ann_vol=0.15,
                downside_vol=0.08,
                max_drawdown=-0.10,
                sharpe_like=1.33,
                preference_score=0.90,
                rank=1,
                bucket="STRONG",
            )
        },
        regime="CALM_BULL",
    )

    comp = regime_fit_advantage(current_fit=current, candidate_fit=candidate)

    assert comp["preference_score_advantage"] == pytest.approx(0.70)
    assert comp["strong_asset_weight_advantage"] == pytest.approx(1.0)
    assert comp["weak_asset_weight_reduction"] == pytest.approx(1.0)


def test_build_portfolio_regime_fit_comparison() -> None:
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    returns = pd.DataFrame(
        {
            "AAPL": [-0.001] * 90,
            "MSFT": [0.002] * 90,
        },
        index=dates,
    )
    regime_history = pd.DataFrame({"date": dates, "regime": ["CALM_BULL"] * 90})

    out = build_portfolio_regime_fit_comparison(
        returns_wide=returns,
        regime_history=regime_history,
        regime="CALM_BULL",
        current_weights={"AAPL": 1.0},
        candidate_weights={"MSFT": 1.0},
        candidate_name="candidate",
        cfg=RegimeAssetPreferenceConfig(min_obs=30),
    )

    assert out["status"] == "success"
    assert out["regime"] == "CALM_BULL"
    assert out["comparison"]["preference_score_advantage"] > 0
