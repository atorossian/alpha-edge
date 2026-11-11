"""Preflight uses explicit contract metadata and never infers currencies/calendars."""

import pandas as pd
import pytest

from alpha_edge.market.preflight_universe_v2 import (
    UniversePreflightError,
    make_cohorts,
    prepare_registry,
)


def registry():
    universe = pd.DataFrame(
        [
            {"asset_id": "A", "yahoo_ticker": "KESKOB.HE", "currency": "EUR", "calendar_id": "XHEL", "include": 1},
            {"asset_id": "B", "yahoo_ticker": "GLD", "currency": "USD", "calendar_id": "NYSE", "include": 1},
            {"asset_id": "C", "yahoo_ticker": "IGNORED", "currency": "USD", "calendar_id": "NYSE", "include": 0},
        ]
    )
    fx = pd.DataFrame([{"currency": "EUR", "provider_symbol": "EURUSD=X", "direction": "direct"}])
    return universe, fx


def test_explicit_registry_and_cohorts():
    u, fx = prepare_registry(*registry(), check_calendars=False)
    assert len(u) == 2
    assert len(fx) == 1
    assert [len(x) for x in make_cohorts(u, cohort_size=1)] == [1, 1]


def test_unknown_currency_must_not_silently_use_usd():
    u, fx = registry()
    fx = fx.iloc[:0]
    with pytest.raises(UniversePreflightError, match="Missing FX"):
        prepare_registry(u, fx, check_calendars=False)


def test_duplicate_symbol_fails_closed():
    u, fx = registry()
    u.loc[1, "yahoo_ticker"] = "KESKOB.HE"
    with pytest.raises(UniversePreflightError, match="repeated Yahoo"):
        prepare_registry(u, fx, check_calendars=False)


def test_unsafe_universe_fails_closed():
    u, fx = registry()
    u.loc[0, "asset_id"] = "../A"
    with pytest.raises(UniversePreflightError, match="Unsafe"):
        prepare_registry(u, fx, check_calendars=False)


def test_cohort_limit():
    u, _ = registry()
    with pytest.raises(UniversePreflightError, match="Cohort size"):
        make_cohorts(u, cohort_size=0)
