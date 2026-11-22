import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market import dividend_duplicate_registry_v3 as registry


def evidence(status="proposed"):
    return {
        "schema": registry.SCHEMA,
        "entries": [
            {
                "asset_id": "A",
                "provider_symbol": "AAA",
                "bronze_sha256": "a" * 64,
                "effective_date": "2008-05-09",
                "retained_date": "2008-06-11",
                "source_value": "1.36",
                "retained_source_value": "2.0",
                "source_currency": "EUR",
                "evidence_url": "https://example.com/issuer",
                "reason": "extra source event not reported by issuer",
                "status": status,
                "approved_by": "reviewer" if status == "approved" else "",
            }
        ],
    }


def actions():
    return pd.DataFrame(
        {
            "asset_id": ["A"] * 2,
            "provider_symbol": ["AAA"] * 2,
            "effective_date": ["2008-05-09", "2008-06-11"],
            "action_type": ["DIVIDEND"] * 2,
            "source_value": [1.36, 2.0],
            "source_currency": ["EUR"] * 2,
        }
    )


@pytest.mark.parametrize(("status", "expected"), [("proposed", 1.36), ("approved", 0.0)])
def test_only_approved_suppression_changes_candidate(status, expected):
    raw = actions()
    candidate, stats = registry.apply_suppressions(
        raw, evidence(status), asset_id="A", provider_symbol="AAA", bronze_sha256="a" * 64
    )
    assert candidate.normalized_value.tolist() == [expected, 2.0]
    assert candidate.source_value.tolist() == raw.source_value.tolist()
    assert "normalized_value" not in raw
    assert stats[status] == 1
    assert not stats["production_eligible"]


def test_stale_source_or_missing_retained_event_fails():
    with pytest.raises(ValueError, match="checksum"):
        registry.apply_suppressions(actions(), evidence(), asset_id="A", provider_symbol="AAA", bronze_sha256="b" * 64)
    with pytest.raises(ValueError, match="one dividend"):
        registry.apply_suppressions(
            actions().iloc[:1], evidence(), asset_id="A", provider_symbol="AAA", bronze_sha256="a" * 64
        )


def test_immutable_registry_proposal_s3_roundtrip():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        preview = registry.publish_registry(s3, "test-bucket", evidence())
        assert s3.list_objects_v2(Bucket="test-bucket")["KeyCount"] == 0
        result = registry.publish_registry(s3, "test-bucket", evidence(), execute=True)
        assert result["version"] == preview["version"]
        assert registry.load_registry(s3, "test-bucket", result["version"]) == evidence()
