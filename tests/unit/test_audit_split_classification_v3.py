import pandas as pd
import pytest

from alpha_edge.market.audit_split_classification_v3 import classify, screen_asset


@pytest.mark.parametrize(
    ("ratio", "group"),
    [
        (2.0, "integer_forward"),
        (0.2, "integer_reverse"),
        (1.5, "simple_fraction"),
        (1.05, "simple_fraction"),
        (1.253, "irregular_ratio"),
        (0.9535, "irregular_ratio"),
        (1.0, "invalid_or_unit_ratio"),
    ],
)
def test_ratio_shape(ratio, group):
    assert classify(ratio) == group


def test_marginal_bounds_respect_other_blockers():
    f = pd.DataFrame(
        {
            "asset_id": "a",
            "provider_symbol": "A",
            "session_date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "adj_close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "stock_splits": [0.0, 2.0, 0.0, 1.253, 0.0],
            "dividends": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    q = pd.DataFrame(
        {
            "effective_date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "scope": "prefix",
            "reason": ["split_price_basis_unverified", "dividend_factor_mismatch", "split_price_basis_unverified"],
        }
    )
    events, report = screen_asset(f, q, asset_id="a")
    assert events[0]["continuity"] == "provider_normalized_signal"
    assert report["binding_group"] == "irregular_ratio"
    assert report["upper_bounds"]["integer_both"] == 0
    assert report["upper_bounds"]["all_reported_splits"] == 1
    q.loc[2, "effective_date"] = "2020-01-07"
    with pytest.raises(ValueError, match="audit/source mismatch"):
        screen_asset(f, q, asset_id="a")
