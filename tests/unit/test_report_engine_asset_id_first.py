from __future__ import annotations

import pandas as pd

from alpha_edge.core.schemas import EvalMetrics, Position
from alpha_edge.portfolio import report_engine


def test_build_portfolio_report_uses_asset_id_keys_for_evaluation(monkeypatch):
    captured = {}

    def fake_evaluate_portfolio(**kwargs):
        captured.update(kwargs)
        return EvalMetrics(
            weights=dict(kwargs["weights"]),
            goals=tuple(kwargs["goals"]),
            main_goal=float(kwargs["main_goal"]),
            ann_return=0.10,
            ann_vol=0.20,
            sharpe=0.50,
            sortino=0.60,
            max_drawdown=-0.10,
            ann_vol_lw=0.20,
            var_95=-0.02,
            cvar_95=-0.03,
            ruin_prob_1y=0.01,
            p_hit_goal_1_1y=0.70,
            p_hit_goal_2_1y=0.60,
            p_hit_goal_3_1y=0.50,
            med_t_goal_1_days=10.0,
            med_t_goal_2_days=20.0,
            med_t_goal_3_days=30.0,
            ending_equity_p5=900.0,
            ending_equity_p25=950.0,
            ending_equity_p50=1000.0,
            ending_equity_p75=1100.0,
            ending_equity_p95=1200.0,
            score=1.23,
        )

    monkeypatch.setattr(report_engine, "evaluate_portfolio", fake_evaluate_portfolio)

    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    closes = pd.DataFrame({"AAA": [10, 11, 12, 13, 14], "BBB": [20, 19, 18, 17, 16]}, index=idx)
    returns = pd.DataFrame(
        {
            "ASSET:AAA-US": [0.01, 0.02, 0.03, -0.01, 0.00],
            "ASSET:BBB-US": [-0.01, 0.00, 0.01, 0.02, -0.02],
        },
        index=idx,
    )

    report = report_engine.build_portfolio_report(
        closes=closes,
        positions={
            "AAA": Position(ticker="AAA", quantity=10.0),
            "BBB": Position(ticker="BBB", quantity=-5.0),
        },
        equity=1000.0,
        goals=[1100.0, 1200.0, 1300.0],
        main_goal=1200.0,
        prices_usd=pd.Series({"AAA": 14.0, "BBB": 16.0}),
        asset_returns=returns,
        asset_id_by_ticker={"AAA": "ASSET:AAA-US", "BBB": "ASSET:BBB-US"},
    )

    assert set(captured["weights"].keys()) == {"ASSET:AAA-US", "ASSET:BBB-US"}
    assert list(captured["returns"].columns) == ["ASSET:AAA-US", "ASSET:BBB-US"]
    rows = report.snapshot.positions_table
    assert rows[0]["asset_id"] == "ASSET:AAA-US"
    assert rows[0]["display_ticker"] == "AAA"
    assert rows[0]["evaluation_key"] == "ASSET:AAA-US"
