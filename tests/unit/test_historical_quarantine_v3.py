from decimal import Decimal

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market import historical_quarantine_v3 as q
from alpha_edge.market.silver_financial_evidence_v3 import extract_source_actions


def bars():
    return pd.DataFrame(
        {
            "asset_id": ["A"] * 6,
            "provider_symbol": ["AAA"] * 6,
            "currency_original": ["EUR"] * 6,
            "session_date": pd.date_range("2001-01-01", periods=6).strftime("%Y-%m-%d"),
            "close": [100.0] * 6,
            "adj_close": [99.0, 99.0, 100.0, 100.0, 100.0, 100.0],
            "dividends": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "stock_splits": [0.0] * 6,
        }
    )


def assess(frame, registry=None, actions=None):
    return q.assess(
        frame,
        extract_source_actions(frame)[0] if actions is None else actions,
        asset_id="A",
        provider_symbol="AAA",
        bronze_sha256="a" * 64,
        registry=registry,
    )


def test_exact_share_basis_distinguishes_split_and_rights():
    assert q.normalize_declared_dividend(
        Decimal(8), split_ratios=(Decimal(2), Decimal(4)), price_basis_factors=()
    ) == Decimal(1)
    assert q.normalize_declared_dividend(
        Decimal(1), split_ratios=(Decimal("0.1"),), price_basis_factors=(Decimal("0.5"),)
    ) == Decimal(5)


def test_consistent_dividend_is_candidate_not_certified():
    sessions, findings, stats = assess(bars())
    assert findings.empty
    assert sessions.candidate_session.all()
    assert not sessions.production_eligible.any()
    assert stats["candidate_returns"] == 5


def test_sub_old_threshold_mismatch_quarantines_prefix_and_boundary_return():
    frame = bars()
    frame.loc[2, "dividends"] = 1.03
    sessions, findings, stats = assess(frame)
    assert findings.reason.tolist() == ["dividend_factor_mismatch"]
    assert stats["quarantine_through"] == "2001-01-03"
    assert sessions.candidate_session.tolist() == [False, False, False, True, True, True]
    assert sessions.candidate_return.tolist() == [False, False, False, False, True, True]


def test_approved_currency_correction_does_not_certify_price_basis():
    frame = bars()
    actions = extract_source_actions(frame)[0]
    actions["normalized_value"] = 0.5
    actions["currency_correction_status"] = "approved"
    _, findings, _ = assess(frame, actions=actions)
    assert "dividend_factor_mismatch" in findings.reason.values


def test_fractional_split_and_same_day_dividend_are_not_applied():
    frame = bars()
    frame.loc[2, "stock_splits"] = 7.7
    pristine = frame.copy()
    _, findings, stats = assess(frame)
    assert set(findings.reason) == {"split_price_basis_unverified", "dividend_factor_not_evaluable"}
    assert stats["quarantine_through"] == "2001-01-03"
    pd.testing.assert_frame_equal(frame, pristine)


def test_negative_adjusted_prices_and_missing_raw_close():
    frame = bars()
    frame.loc[0, "adj_close"] = -99
    frame.loc[4, "close"] = float("nan")
    sessions, _, stats = assess(frame)
    assert stats["quarantine_through"] == "2001-01-01"
    assert sessions.candidate_return.tolist() == [False, False, True, True, False, False]


def test_missing_reported_event_from_independent_evidence_is_pinned():
    registry = {
        "schema": q.SCHEMA,
        "entries": [
            {
                "asset_id": "A",
                "provider_symbol": "AAA",
                "bronze_sha256": "a" * 64,
                "effective_date": "2001-01-04",
                "reason": "missing_split",
                "evidence_url": "https://example.com/issuer",
            }
        ],
    }
    _, _, stats = assess(bars(), registry=registry)
    assert stats["quarantine_through"] == "2001-01-04"
    registry["entries"][0]["bronze_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="checksum"):
        assess(bars(), registry=registry)


def test_registry_publication_is_explicit_and_content_addressed():
    doc = {"schema": q.SCHEMA, "entries": []}
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = "quarantine-test"
        s3.create_bucket(Bucket=bucket)
        preview = q.publish_registry(s3, bucket, doc)
        assert not preview["executed"]
        assert s3.list_objects_v2(Bucket=bucket)["KeyCount"] == 0
        published = q.publish_registry(s3, bucket, doc, execute=True)
        assert published["version"] == preview["version"]
        assert q.load_registry(s3, bucket, published["version"]) == doc
