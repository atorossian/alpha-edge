from __future__ import annotations

from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

from alpha_edge.core.schemas import PortfolioReport, PortfolioSnapshot
from alpha_edge.jobs.run_daily_report import _build_live_augmented_returns_for_portfolio
from alpha_edge.portfolio.evaluation_service import build_portfolio_behavior_regime, build_regime_alignment
from alpha_edge.risk.actuarial import portfolio_search_output as actuarial_output


# The daily-report module imports yfinance for live price fallback. Unit tests in
# minimal CI environments do not need the real package for this pure helper.
sys.modules.setdefault("yfinance", SimpleNamespace())


def test_regime_alignment_normalizes_mixed_neutral_display_label():
    out = build_regime_alignment(
        market_regime_label="MIXED / NEUTRAL",
        portfolio_behavior_label="CHOPPY_BULL",
    )

    assert out["market_regime_label"] == "MIXED"
    assert out["portfolio_behavior_label"] == "CHOPPY_BULL"
    assert out["status"] == "positive_divergence"


def test_portfolio_behavior_uses_mixed_market_label_when_morning_regime_is_uncommitted():
    r = pd.Series(np.linspace(-0.01, 0.01, 80))
    out = build_portfolio_behavior_regime(
        portfolio_returns=r,
        market_regime_payload={
            "hmm": {
                "label_commit": None,
                "p_label_today": {"CHOPPY_BEAR": 0.57, "CHOPPY_BULL": 0.43},
            },
            "leverage_recommendation": {"label": "CHOPPY_BEAR"},
        },
        min_observations=252,
    )

    assert out["market_regime"]["label"] == "MIXED"
    assert out["regime_alignment"]["market_regime_label"] == "MIXED"


def test_live_augmented_returns_uses_logical_as_of_date_for_live_row():
    returns_wide = pd.DataFrame(
        {
            "CRYPTO:SOL-USD": [0.01, -0.02],
        },
        index=pd.to_datetime(["2026-08-08", "2026-08-09"]),
    )
    spot_rows = [{"asset_id": "CRYPTO:SOL-USD", "ticker": "SOL-USD"}]
    deriv_rows = []
    latest_close_prices = pd.Series({"SOL-USD": 100.0})
    prices_for_valuation = pd.Series({"SOL-USD": 103.0})

    out, meta = _build_live_augmented_returns_for_portfolio(
        returns_wide=returns_wide,
        spot_rows=spot_rows,
        deriv_rows=deriv_rows,
        latest_close_prices=latest_close_prices,
        prices_for_valuation=prices_for_valuation,
        as_of_run_date="2026-08-10",
    )

    assert str(out.index[-1].date()) == "2026-08-10"
    assert meta["live_return_date"] == "2026-08-10"
    assert abs(float(out.iloc[-1]["CRYPTO:SOL-USD"]) - 0.03) < 1e-12


def test_actuarial_daily_diagnostic_accepts_asset_id_keyed_returns(monkeypatch):
    captured = {}

    def fake_evaluate_portfolio_candidate_with_paths(**kwargs):
        captured["returns_columns"] = list(kwargs["returns"].columns)
        captured["weights"] = dict(kwargs["weights"])
        return SimpleNamespace(equity_paths=np.ones((3, 6)) * 100.0)

    monkeypatch.setattr(
        actuarial_output,
        "evaluate_portfolio_candidate_with_paths",
        fake_evaluate_portfolio_candidate_with_paths,
    )
    monkeypatch.setattr(actuarial_output, "evaluate_actuarial_risk", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        actuarial_output, "build_actuarial_diagnostic_report", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(actuarial_output, "format_actuarial_diagnostic_report", lambda *args, **kwargs: "ok")
    monkeypatch.setattr(actuarial_output, "diagnostic_report_to_json_block", lambda *args, **kwargs: {"ok": True})

    report = PortfolioReport(
        snapshot=PortfolioSnapshot(
            as_of=pd.Timestamp("2026-08-10"),
            total_notional=10000.0,
            equity=5000.0,
            leverage=2.0,
            positions_table=[],
        ),
        eval=SimpleNamespace(weights={"CRYPTO:SOL-USD": 1.0}),
    )
    closes = pd.DataFrame(
        {"SOL-USD": np.linspace(80.0, 100.0, 80)},
        index=pd.date_range("2026-01-01", periods=80),
    )
    asset_returns = pd.DataFrame(
        {"CRYPTO:SOL-USD": np.full(80, 0.001)},
        index=pd.date_range("2026-01-01", periods=80),
    )

    _report, text, block = actuarial_output.build_actuarial_diagnostic_from_portfolio_report(
        report=report,
        closes=closes,
        asset_returns=asset_returns,
        goals=[7500.0, 10000.0, 12500.0],
        main_goal=10000.0,
        days=5,
        n_paths=3,
    )

    assert text == "ok"
    assert block == {"ok": True}
    assert captured["returns_columns"] == ["CRYPTO:SOL-USD"]
    assert captured["weights"] == {"CRYPTO:SOL-USD": 1.0}
