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


def test_small_fraction_approval_requires_explicit_scope(monkeypatch):
    from alpha_edge.market.audit_matching_split_basis_v3 import SMALL_CANDIDATE, SMALL_POLICY
    registry = {"schema": "alpha_edge_split_basis_candidate_registry_v1", "policy": SMALL_POLICY,
                "entries": [{"asset_id": "asset", "date": "2020-01-01", "status": SMALL_CANDIDATE,
                             "group": "simple_fraction", "policy": SMALL_POLICY, "ratio": 1.05}]}
    monkeypatch.setattr(approval, "verified_bytes", lambda *args: json.dumps(registry).encode())
    with pytest.raises(ValueError):
        approval.approve(None, "bucket", "a" * 64, expected_count=1, fractional=True)
    result = approval.approve(None, "bucket", "a" * 64, expected_count=1, small_fraction=True)
    assert result["economic_review_required"] is True
    assert result["quarantine_released"] is False
    registry["entries"][0]["ratio"] = 1.5
    with pytest.raises(ValueError):
        approval.approve(None, "bucket", "a" * 64, expected_count=1, small_fraction=True)


def test_irregular_approval_requires_matching_policy(monkeypatch):
    from alpha_edge.market.audit_matching_split_basis_v3 import IRREGULAR_CANDIDATE, IRREGULAR_POLICY
    registry = {"schema": "alpha_edge_split_basis_candidate_registry_v1", "policy": IRREGULAR_POLICY,
                "entries": [{"asset_id": "asset", "date": "2020-01-01", "status": IRREGULAR_CANDIDATE,
                             "group": "irregular_ratio", "policy": IRREGULAR_POLICY, "ratio": 1.328}]}
    monkeypatch.setattr(approval, "verified_bytes", lambda *args: json.dumps(registry).encode())
    with pytest.raises(ValueError):
        approval.approve(None, "bucket", "a" * 64, expected_count=1, fractional=True)
    result = approval.approve(None, "bucket", "a" * 64, expected_count=1, irregular=True)
    assert result["economic_review_required"] is True
    assert result["quarantine_released"] is False
    registry["entries"][0]["ratio"] = 1.5
    with pytest.raises(ValueError):
        approval.approve(None, "bucket", "a" * 64, expected_count=1, irregular=True)
