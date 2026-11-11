"""Independent daily arithmetic checks cannot approve external source claims."""

import pandas as pd

from alpha_edge.market.independent_daily_quality_v2 import review
from alpha_edge.market.standardize_bronze_v2 import build_silver_gold_from_bronze


def _tables():
    bars = pd.DataFrame(
        {
            "asset_id": ["A", "A", "A"],
            "provider_symbol": ["SPY", "SPY", "SPY"],
            "currency_original": ["USD", "USD", "USD"],
            "session_date": ["2026-10-05", "2026-10-06", "2026-10-07"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "adj_close": [100.0, 101.0, 102.0],
            "volume": [1000, 1200, 900],
            "source_timestamp": [d + "T00:00:00+00:00" for d in ("2026-10-05", "2026-10-06", "2026-10-07")],
        }
    )
    quotes = pd.DataFrame(columns=["provider_symbol", "session_date", "quote_close"])
    return build_silver_gold_from_bronze(bars, quotes, calendars={"SPY": "NYSE"}, pairs={})


def test_daily_independent_arithmetic_passes_not_external_acceptance():
    data = _tables()
    result = review(data)
    assert result["internal_integrity_status"] == "passed"
    assert result["blocker_counts"]["integrity_findings"] == 0
    assert result["acceptance"] == "review_required"
    assert result["open_external_evidence_gates"]
    assert result["gold_rows"] == 3


def test_daily_independent_arithmetic_detects_corrupt_adjusted_returns():
    data = _tables()
    data["gold"].loc[data["gold"].index[-1], "ret_simple_close_adjusted_usd"] = 0.123
    result = review(data)
    assert result["internal_integrity_status"] == "failed"
    assert "return_arithmetic_ret_simple_close_adjusted_usd" in result["integrity_findings"]
    assert result["acceptance"] == "blocked"
