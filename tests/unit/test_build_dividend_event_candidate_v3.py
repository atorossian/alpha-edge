import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market import build_dividend_event_candidate_v3 as ev
from alpha_edge.market.lakehouse_v3.storage import digest, parquet_bytes


def source():
    return pd.DataFrame(
        {
            "asset_id": ["A"] * 5,
            "provider_symbol": ["AAA"] * 5,
            "currency_original": ["EUR"] * 5,
            "session_date": ["2008-05-09", "2008-05-14", "2008-05-20", "2008-05-21", "2022-06-16"],
            "dividends": [0.314028, 0.314028, 0.386496, 0.0, 0.0],
            "stock_splits": [0.0, 0.0, 0.0, 0.0, 4.0],
        }
    )


def entry(sha="a" * 64):
    return {
        "asset_id": "A",
        "provider_symbol": "AAA",
        "currency": "EUR",
        "source_key": "market/lake_v3/bronze/example.parquet",
        "source_sha256": sha,
        "window_start": "2008-05-09",
        "window_end": "2008-05-21",
        "issuer_ex_date": "2008-05-20",
        "issuer_amount": "1.60",
        "source_events": [
            {"date": "2008-05-09", "cash": "0.314028"},
            {"date": "2008-05-14", "cash": "0.314028"},
            {"date": "2008-05-20", "cash": "0.386496"},
        ],
        "basis_factors": [
            {
                "effective_date": "2010-09-30",
                "kind": "price_coefficient",
                "status": "provisional",
                "factor": "0.96624",
                "evidence_url": "https://example.com/rights",
            },
            {
                "effective_date": "2022-06-16",
                "kind": "share_ratio",
                "status": "verified",
                "factor": "4",
                "evidence_url": "https://example.com/split",
            },
        ],
        "basis_status": "provisional",
        "reason": "Single verified ex-date; rights basis provisional",
        "evidence_urls": ["https://example.com/issuer", "https://example.com/rights"],
    }


def test_candidate_keeps_source_and_one_event():
    raw = source()
    original = raw.copy(deep=True)
    out = ev.reconstruct(raw, entry(), source_sha256="a" * 64)
    assert out.source_cash.tolist() == [0.314028, 0.314028, 0.386496, 0.0]
    assert out.candidate_cash.tolist() == [0.0, 0.0, 0.386496, 0.0]
    assert out.basis_status.eq("provisional").all()
    assert not out.production_eligible.any()
    pd.testing.assert_frame_equal(raw, original)


def test_move_existing_amount_to_correct_session():
    e = entry()
    e["issuer_ex_date"] = "2008-05-21"
    out = ev.reconstruct(source(), e, source_sha256="a" * 64)
    assert out.candidate_cash.tolist() == [0.0, 0.0, 0.0, 0.386496]


def test_unexpected_cash_basis_or_source_fails_closed():
    raw = source()
    raw.loc[3, "dividends"] = 0.1
    with pytest.raises(ValueError, match="window differs"):
        ev.reconstruct(raw, entry(), source_sha256="a" * 64)
    raw = source()
    raw.loc[4, "stock_splits"] = 2
    with pytest.raises(ValueError, match="share ratios"):
        ev.reconstruct(raw, entry(), source_sha256="a" * 64)
    e = entry()
    e["issuer_amount"] = "2"
    with pytest.raises(ValueError, match="No correctly sized"):
        ev.reconstruct(source(), e, source_sha256="a" * 64)
    e = entry()
    e["basis_status"] = "verified"
    with pytest.raises(ValueError, match="Unverified factor"):
        ev.reconstruct(source(), e, source_sha256="a" * 64)


def test_immutable_candidate_roundtrip():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = "test-bucket"
        s3.create_bucket(Bucket=bucket)
        blob = parquet_bytes(source())
        e = entry(digest(blob))
        s3.put_object(Bucket=bucket, Key=e["source_key"], Body=blob)
        doc = {"schema": ev.SCHEMA, "status": "candidate_only", "entries": [e]}
        v = ev.publish_registry(s3, bucket, doc, execute=True)["version"]
        assert ev.build(s3, bucket, version=v, run_id="pilot")["status"] == "dry_run"
        a = ev.build(s3, bucket, version=v, run_id="pilot", execute=True)
        assert a == ev.build(s3, bucket, version=v, run_id="pilot", execute=True)
        assert a["source_positive_events"] == 3 and a["candidate_positive_events"] == 1
        assert s3.get_object(Bucket=bucket, Key=e["source_key"])["Body"].read() == blob
