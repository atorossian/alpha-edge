import json

import pytest

from alpha_edge.market import approve_split_basis_v3 as approval
from alpha_edge.market.audit_matching_split_basis_v3 import CANDIDATE, POLICY


def test_approval_is_exact_and_price_basis_only(monkeypatch):
    registry = {"schema": "alpha_edge_split_basis_candidate_registry_v1", "policy": POLICY,
                "entries": [{"asset_id": "asset", "date": "2020-01-01", "status": CANDIDATE,
                             "group": "integer_forward", "policy": POLICY}]}
    monkeypatch.setattr(approval, "verified_bytes", lambda *args: json.dumps(registry).encode())
    writes = []
    monkeypatch.setattr(approval, "put_immutable", lambda *args, **kwargs: writes.append(args))
    result = approval.approve(None, "bucket", "a" * 64, expected_count=1, execute=True)
    assert result["approved_count"] == 1
    assert result["production_eligible"] is False
    assert len(writes) == 1
    with pytest.raises(ValueError):
        approval.approve(None, "bucket", "a" * 64, expected_count=2)
    registry["entries"][0]["status"] = "fractional_price_basis_candidate"
    with pytest.raises(ValueError):
        approval.approve(None, "bucket", "a" * 64, expected_count=1)


def test_fractional_approval_requires_explicit_mode(monkeypatch):
    from alpha_edge.market.audit_matching_split_basis_v3 import FRACTIONAL, FRACTIONAL_POLICY
    registry = {"schema": "alpha_edge_split_basis_candidate_registry_v1", "policy": FRACTIONAL_POLICY,
                "entries": [{"asset_id": "asset", "date": "2020-01-01", "status": FRACTIONAL,
                             "group": "simple_fraction", "policy": FRACTIONAL_POLICY}]}
    monkeypatch.setattr(approval, "verified_bytes", lambda *args: json.dumps(registry).encode())
    with pytest.raises(ValueError):
        approval.approve(None, "bucket", "a" * 64, expected_count=1)
    result = approval.approve(None, "bucket", "a" * 64, expected_count=1, fractional=True)
    assert result["economic_review_required"] is True
    assert result["quarantine_released"] is False
    registry["entries"][0]["status"] = "small_fraction_price_basis_candidate"
    with pytest.raises(ValueError):
        approval.approve(None, "bucket", "a" * 64, expected_count=1, fractional=True)
