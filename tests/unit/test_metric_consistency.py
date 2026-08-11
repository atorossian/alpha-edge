from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_edge.core.schemas import Position, ScoreConfig
from alpha_edge.portfolio.optimizer_engine import evaluate_portfolio
from alpha_edge.portfolio.report_engine import build_portfolio_report


def _make_synthetic_closes() -> pd.DataFrame:
    rng = np.random.default_rng(123)

    dates = pd.bdate_range("2020-01-01", periods=800)

    # synthetic daily returns
    r = pd.DataFrame(
        {
            "AAA": rng.normal(0.0004, 0.012, len(dates)),
            "BBB": rng.normal(0.0002, 0.010, len(dates)),
            "CCC": rng.normal(-0.0001, 0.015, len(dates)),
        },
        index=dates,
    )

    closes = 100.0 * (1.0 + r).cumprod()
    return closes


def test_optimizer_and_report_engine_agree_for_gross_signed_long_short():
    closes = _make_synthetic_closes()

    prices = closes.iloc[-1]

    equity0 = 10_000.0
    notional = 20_000.0

    # target signed gross weights:
    # gross = |0.50| + |0.30| + |-0.20| = 1.00
    weights = {
        "AAA": 0.50,
        "BBB": 0.30,
        "CCC": -0.20,
    }

    positions = {
        t: Position(
            ticker=t,
            quantity=(weights[t] * notional) / float(prices[t]),
            entry_price=None,
            currency="USD",
        )
        for t in weights
    }

    goals = [11_000.0, 12_000.0, 13_000.0]
    main_goal = 12_000.0
    score_cfg = ScoreConfig()

    returns = closes.pct_change().dropna(how="any")

    m1 = evaluate_portfolio(
        returns=returns,
        weights=weights,
        equity0=equity0,
        notional=notional,
        goals=goals,
        main_goal=main_goal,
        score_config=score_cfg,
        mc_seed=42,
        n_paths=2_000,
        weight_mode="gross_signed",
    )

    report = build_portfolio_report(
        closes=closes,
        positions=positions,
        equity=equity0,
        goals=goals,
        main_goal=main_goal,
        score_config=score_cfg,
        mc_seed=42,
        n_paths=2_000,
        prices_usd=prices,
    )

    m2 = report.eval

    assert np.isclose(m1.ann_return, m2.ann_return, atol=1e-10)
    assert np.isclose(m1.ann_vol, m2.ann_vol, atol=1e-10)
    assert np.isclose(m1.max_drawdown, m2.max_drawdown, atol=1e-10)
    assert np.isclose(m1.var_95, m2.var_95, atol=1e-10)
    assert np.isclose(m1.cvar_95, m2.cvar_95, atol=1e-10)

    # MC should also match because the same seed, same port returns,
    # same equity, same notional and same goals are used.
    assert np.isclose(m1.ruin_prob_1y, m2.ruin_prob_1y, atol=1e-8)
    assert np.isclose(m1.p_hit_goal_1_1y, m2.p_hit_goal_1_1y, atol=1e-8)
    assert np.isclose(m1.p_hit_goal_2_1y, m2.p_hit_goal_2_1y, atol=1e-8)
    assert np.isclose(m1.p_hit_goal_3_1y, m2.p_hit_goal_3_1y, atol=1e-8)
    assert np.isclose(m1.score, m2.score, atol=1e-8)


def test_long_short_mode_matches_gross_signed_mode():
    closes = _make_synthetic_closes()
    returns = closes.pct_change().dropna(how="any")

    weights = {
        "AAA": 0.50,
        "BBB": 0.30,
        "CCC": -0.20,
    }

    goals = [11_000.0, 12_000.0, 13_000.0]

    m_gross = evaluate_portfolio(
        returns=returns,
        weights=weights,
        equity0=10_000.0,
        notional=20_000.0,
        goals=goals,
        main_goal=12_000.0,
        score_config=ScoreConfig(),
        mc_seed=42,
        n_paths=2_000,
        weight_mode="gross_signed",
    )

    m_ls = evaluate_portfolio(
        returns=returns,
        weights=weights,
        equity0=10_000.0,
        notional=20_000.0,
        goals=goals,
        main_goal=12_000.0,
        score_config=ScoreConfig(),
        mc_seed=42,
        n_paths=2_000,
        weight_mode="long_short",
    )

    assert np.isclose(m_gross.ann_return, m_ls.ann_return, atol=1e-10)
    assert np.isclose(m_gross.ann_vol, m_ls.ann_vol, atol=1e-10)
    assert np.isclose(m_gross.max_drawdown, m_ls.max_drawdown, atol=1e-10)
    assert np.isclose(m_gross.ruin_prob_1y, m_ls.ruin_prob_1y, atol=1e-8)
    assert np.isclose(m_gross.score, m_ls.score, atol=1e-8)
