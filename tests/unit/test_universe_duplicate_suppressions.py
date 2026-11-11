"""Canonical duplicate review decisions must survive a full universe refresh."""

import pandas as pd
import pytest

from alpha_edge.jobs.run_universe_update import suppress_noncanonical_duplicate_rows


def test_suppresses_only_reviewed_duplicate_rows():
    universe = pd.DataFrame([
        {"row_id": "5E262727DC314688", "ticker": "IAC", "yahoo_ticker": "PPLI"},
        {"row_id": "670A9138B424DB13", "ticker": "PPLI", "yahoo_ticker": "PPLI"},
        {"row_id": "C73EB54227724B75", "ticker": "KMB", "yahoo_ticker": "KMB"},
        {"row_id": "DEA19FC8D266CFE2", "ticker": "KMB", "yahoo_ticker": "KMB"},
    ])
    mapping = pd.DataFrame([
        {"duplicate_row_id": "670A9138B424DB13", "canonical_row_id": "5E262727DC314688"},
        {"duplicate_row_id": "DEA19FC8D266CFE2", "canonical_row_id": "C73EB54227724B75"},
    ])
    kept = suppress_noncanonical_duplicate_rows(universe, mapping)
    assert set(kept["row_id"]) == {"5E262727DC314688", "C73EB54227724B75"}
    assert set(kept["yahoo_ticker"]) == {"PPLI", "KMB"}


def test_duplicate_suppression_refuses_to_drop_only_identity():
    universe = pd.DataFrame([{"row_id": "670A9138B424DB13", "ticker": "PPLI"}])
    mapping = pd.DataFrame([
        {"duplicate_row_id": "670A9138B424DB13", "canonical_row_id": "5E262727DC314688"}
    ])
    with pytest.raises(ValueError, match="missing keeper"):
        suppress_noncanonical_duplicate_rows(universe, mapping)


def test_duplicate_suppression_rejects_conflicting_mapping():
    universe = pd.DataFrame([{"row_id": "A", "ticker": "X"}])
    mapping = pd.DataFrame([
        {"duplicate_row_id": "A", "canonical_row_id": "B"},
        {"duplicate_row_id": "A", "canonical_row_id": "C"},
    ])
    with pytest.raises(ValueError, match="Ambiguous"):
        suppress_noncanonical_duplicate_rows(universe, mapping)
