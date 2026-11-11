import pandas as pd
from alpha_edge.market.aea87_universe_screen import screen_one


def _prices(values):
    return pd.DataFrame(
        {
            "asset_id": ["A"] * len(values),
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"][: len(values)]),
            "close_adjusted_usd": values,
        }
    )


def _returns(values):
    return pd.DataFrame(
        {
            "asset_id": ["A"] * len(values),
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"][: len(values)]),
            "ret_close_adjusted_usd": values,
        }
    )


def test_coherent_prices_pass_numeric_screen_without_claiming_identity():
    result = screen_one("A", _prices([10, 11]), _returns([None, 0.1]))
    assert result["status"] == "PASSED_NUMERICAL_SCREEN_ONLY"
    assert result["identity_verified"] is False
    assert result["publication_approved"] is False


def test_large_move_alone_is_not_confirmed_corruption():
    result = screen_one("A", _prices([10, 20]), _returns([None, 1]))
    assert result["status"] == "INVESTIGATION_REQUIRED"
    assert result["issues"]["LARGE_MOVE_REVIEW"] == 1


def test_return_mismatch_triggers_review():
    result = screen_one("A", _prices([10, 11]), _returns([None, 0.5]))
    assert result["status"] == "REBUILD_OR_DATA_REPAIR_REVIEW"
    assert result["issues"]["RETURN_PRICE_MISMATCH"] == 1


def test_duplicate_dates_are_explicitly_classified():
    prices = _prices([10, 11]).assign(date=pd.Timestamp("2026-01-02"))
    result = screen_one("A", prices, _returns([None, 0.1]))
    assert result["issues"]["DUPLICATE_DATES"] == 1
    assert result["status"] == "REBUILD_OR_DATA_REPAIR_REVIEW"
