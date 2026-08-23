import numpy as np
import pandas as pd

from alpha_edge.portfolio.evaluation_service import (
    build_portfolio_behavior_regime,
    build_regime_alignment,
)


def test_regime_alignment_positive_divergence():
    out = build_regime_alignment(
        market_regime_label="STRESS_BEAR",
        portfolio_behavior_label="CALM_BULL",
    )
    assert out["status"] == "positive_divergence"
    assert out["market_regime_label"] == "STRESS_BEAR"
    assert out["portfolio_behavior_label"] == "CALM_BULL"


def test_portfolio_behavior_regime_returns_not_enough_observations_cleanly():
    r = pd.Series([0.01, -0.002, 0.003])
    out = build_portfolio_behavior_regime(
        portfolio_returns=r,
        market_regime_payload={"hmm": {"label_commit": "STRESS_BEAR"}},
        min_observations=252,
    )
    assert out["ok"] is False
    assert out["label"] is None
    assert out["market_regime"]["label"] == "STRESS_BEAR"
    assert out["regime_alignment"]["status"] == "unknown"
    assert "not_enough_observations" in out["reason"]


def test_portfolio_behavior_regime_fits_with_enough_observations():
    rng = np.random.default_rng(123)
    r = pd.Series(rng.normal(loc=0.0008, scale=0.01, size=320))
    out = build_portfolio_behavior_regime(
        portfolio_returns=r,
        market_regime_payload={"hmm": {"label_commit": "CHOPPY_BEAR"}},
        min_observations=100,
        seed=123,
    )
    assert out["ok"] is True
    assert out["label"] in {"CALM_BULL", "CHOPPY_BULL", "CHOPPY_BEAR", "STRESS_BEAR", "MIXED"}
    assert set(out["p_label_today"]) == {"CALM_BULL", "CHOPPY_BULL", "CHOPPY_BEAR", "STRESS_BEAR"}
    assert out["regime_alignment"]["market_regime_label"] == "CHOPPY_BEAR"
