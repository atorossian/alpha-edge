import json

import pytest

from alpha_edge.market.audit_deterministic_residual_v3 import pattern, validate_approval, market


def test_pattern_counts_unique_flags_and_cash_basis():
    attempt = {"status": "cash_amount_or_share_basis_disagreement",
               "yahoo_cash": 1., "eodhd_adjusted_cash": 1.0001, "pre_scale": 100., "post_scale": 100.}
    row = {"attempts": json.dumps([attempt, attempt]), "prior_factors": "[]"}
    reason, flags, statuses = pattern(row)
    assert reason == "cash_amount_basis_or_provider_definition"
    assert flags == ["adjusted_cash_matches_yahoo", "possible_decimal_quote_unit_scale", "stable_relative_quote_scale"]
    assert statuses == ["cash_amount_or_share_basis_disagreement"]
    assert market("AZM.MI") == "MI"


def test_approval_requires_exact_registry_and_evidence():
    chosen = {"status": "provider_normalized_candidate", "date_method": "reported_dates"}
    entry = {"asset_id": "asset", "date": "2020-01-01", "ratio": 2., "bronze_sha256": "y", "eodhd_sha256": "e",
             "chosen_hypothesis": chosen, "status": "reconciled_price_basis_candidate",
             "production_eligible": False, "quarantine_released": False, "economic_review_required": True}
    row = {**entry, "chosen_hypothesis": json.dumps(chosen)}
    registry = {"schema": "alpha_edge_deterministic_corporate_action_candidate_registry_v1",
                "inputs": {}, "entries": [entry], "production_eligible": False, "quarantine_released": False}
    manifest = {"schema": "alpha_edge_deterministic_corporate_action_reconciliation_v1", "inputs": {},
                "outputs": {"candidate_registry": {"sha256": "digest"}},
                "statistics": {"events": 1, "candidate_events": 1}}
    assert validate_approval(manifest, [row], registry, "digest", 1) == {("asset", "2020-01-01")}
    with pytest.raises(ValueError):
        validate_approval(manifest, [row], registry, "digest", 2)
    with pytest.raises(ValueError):
        validate_approval(manifest, [{**row, "bronze_sha256": "other"}], registry, "digest", 1)
