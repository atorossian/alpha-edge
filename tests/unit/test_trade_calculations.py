from __future__ import annotations

import pytest

from alpha_edge.operations.record_trade import (
    _calculate_pnl_fields,
    _calculate_quantity_price_value,
    _infer_position_effect,
)


def test_calculates_quantity_from_value_and_price() -> None:
    quantity, price, value, warnings = _calculate_quantity_price_value(
        quantity=None,
        price=0.2772,
        value=10035.59,
    )

    assert quantity == pytest.approx(36203.42712843)
    assert price == pytest.approx(0.2772)
    assert value == pytest.approx(10035.59)
    assert "quantity calculated from value / price" in warnings


def test_calculates_value_from_quantity_and_price() -> None:
    quantity, price, value, warnings = _calculate_quantity_price_value(
        quantity=301.53178145,
        price=78.01,
        value=None,
    )

    assert quantity == pytest.approx(301.53178145)
    assert price == pytest.approx(78.01)
    assert value == pytest.approx(23522.49)
    assert "value calculated from quantity * price" in warnings


def test_requires_two_economic_inputs() -> None:
    with pytest.raises(ValueError, match="Provide at least two"):
        _calculate_quantity_price_value(quantity=None, price=78.01, value=None)


def test_side_and_action_infer_long_and_short_closes() -> None:
    assert _infer_position_effect(side="SELL", action_tag="close") == "close_long"
    assert _infer_position_effect(side="BUY", action_tag="close") == "close_short"


def test_long_close_pnl_sign() -> None:
    result = _calculate_pnl_fields(
        position_effect="close_long",
        quantity=100.0,
        price=110.0,
        value=11000.0,
        reported_pnl=None,
        open_value=10000.0,
        entry_price=None,
    )

    assert result["calculated_pnl"] == 1000.0
    assert result["pnl_source"] == "calculated"


def test_short_close_pnl_sign_regression() -> None:
    result = _calculate_pnl_fields(
        position_effect="close_short",
        quantity=301.53178145,
        price=78.01,
        value=23522.49,
        reported_pnl=None,
        open_value=25000.0,
        entry_price=None,
    )

    assert result["calculated_pnl"] == 1477.51
    assert result["pnl_source"] == "calculated"


def test_reported_pnl_is_preserved_and_difference_is_calculated() -> None:
    result = _calculate_pnl_fields(
        position_effect="close_short",
        quantity=301.53178145,
        price=78.01,
        value=23522.49,
        reported_pnl=1475.0,
        open_value=25000.0,
        entry_price=None,
    )

    assert result["reported_pnl"] == 1475.0
    assert result["calculated_pnl"] == 1477.51
    assert result["pnl_diff"] == -2.51
    assert result["pnl_source"] == "broker_reported"
