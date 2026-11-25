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
