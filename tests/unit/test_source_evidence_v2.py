"""Session endpoint and corporate adjustment evidence tests."""
import pandas as pd
from alpha_edge.market.source_evidence_v2 import (
    validate_adjustment_factors, validate_source_coverage,
)


def test_dividend_factor_is_mathematically_plausible():
    # Ratio step = 1/(1 - D/P_prior), not equality of dividend with price drop.
    before, after = 100., 100.
    step = 1 / (1 - 2 / before)
    bars = pd.DataFrame([
        {"asset_id": "A", "session_date": "2026-01-05",
         "close": before, "adj_close": before * .9},
        {"asset_id": "A", "session_date": "2026-01-06",
         "close": after, "adj_close": after * .9 * step},
    ])
    actions = pd.DataFrame([{"asset_id": "A", "effective_date": "2026-01-06",
                             "action_type": "DIVIDEND", "value": 2.}])
    out = validate_adjustment_factors(bars=bars, actions=actions)
    assert out.iloc[0]["status"] == "approx_factor_consistent"


def test_inconsistent_dividend_is_not_automatically_approved():
    bars = pd.DataFrame([
        {"asset_id": "A", "session_date": "2026-01-05",
         "close": 100., "adj_close": 90.},
        {"asset_id": "A", "session_date": "2026-01-06",
         "close": 101., "adj_close": 90.9},
    ])
    actions = pd.DataFrame([{"asset_id": "A", "effective_date": "2026-01-06",
                             "action_type": "DIVIDEND", "value": 2.}])
    out = validate_adjustment_factors(bars=bars, actions=actions)
    assert out.iloc[0]["status"] == "factor_requires_review"


def test_missing_endpoint_is_flagged_even_if_internal_sessions_are_contiguous():
    u = pd.DataFrame([{"asset_id": "A", "yahoo_ticker": "GLD", "calendar_id": "NYSE"}])
    bars = pd.DataFrame([{"asset_id": "A", "session_date": "2026-11-06"}])
    out = validate_source_coverage(
        bars=bars, universe=u, start="2026-11-05", end="2026-11-07",
    )
    assert out["session_date"].tolist() == ["2026-11-05"]
    assert out["issue"].tolist() == ["missing_expected_session"]
