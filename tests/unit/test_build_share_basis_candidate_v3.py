import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market import build_share_basis_candidate_v3 as sb
from alpha_edge.market.lakehouse_v3.storage import digest, parquet_bytes


def data():
    return pd.DataFrame(
        {
            "asset_id": ["A"] * 6,
            "provider_symbol": ["AAA"] * 6,
            "session_date": pd.date_range("2000-01-01", periods=6).strftime("%Y-%m-%d"),
            "open": [103.0, 100.0, 100.0, 100.0, 101.0, 102.0],
            "high": [103.0, 100.0, 100.0, 100.0, 101.0, 102.0],
            "low": [103.0, 100.0, 100.0, 100.0, 101.0, 102.0],
            "close": [103.0, 100.0, 100.0, 100.0, 101.0, 102.0],
            "adj_close": [50.0] * 6,
            "stock_splits": [0.0, 0.0, 1.03, 0.0, 0.0, 0.0],
            "dividends": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        }
    )


def registry(sha="a" * 64):
    return {
        "schema": sb.SCHEMA,
        "status": "candidate_only",
        "asset_id": "A",
        "provider_symbol": "AAA",
        "source_key": "market/lake_v3/bronze/example.parquet",
        "source_sha256": sha,
        "evidence_key": "market/lake_v3/config/evidence.json",
        "evidence_sha256": digest(b"evidence"),
        "scope_start": "2000-01-01",
        "events": [
            {
                "effective_date": "2000-01-02",
                "ratio": "1.03",
                "source_split": "0",
                "source_cash": "0",
                "date_status": "candidate",
                "evidence_url": "https://example.org/issuer",
            }
        ],
        "unresolved_cash_dates": ["2000-01-04"],
    }


def test_corrects_missing_factor_only_and_blocks_chain():
    source = data()
    original = source.copy(deep=True)
    c, stats = sb.reconstruct(source, registry(), bronze_sha256="a" * 64)
    assert c.candidate_close.tolist() == pytest.approx([100, 100, 100, 100, 101, 102])
    assert c.candidate_share_event.tolist() == [0, 1.03, 0, 0, 0, 0]
    assert c.candidate_price_return.iloc[1] == pytest.approx(0)
    assert pd.isna(c.candidate_total_return.iloc[3])
    assert c.candidate_adjusted_close.iloc[:3].isna().all()
    assert c.candidate_adjusted_close.iloc[3:].tolist() == pytest.approx([100, 101, 102])
    assert not stats["production_eligible"]
    pd.testing.assert_frame_equal(source, original)


def test_same_day_cash_is_not_assumed_to_be_post_share():
    source = data()
    source.loc[1, "dividends"] = 1.0
    doc = registry()
    doc["events"][0]["source_cash"] = "1"
    c, _ = sb.reconstruct(source, doc, bronze_sha256="a" * 64)
    assert c.unresolved_cash.iloc[1] and pd.isna(c.candidate_total_return.iloc[1])


def test_declared_cash_uses_complete_pre_event_share_basis():
    source = data()
    source["currency_original"] = "USD"
    source.loc[1, "dividends"] = 3.0
    doc = registry()
    doc["events"][0]["source_cash"] = "3"
    doc["cash_reconciliations"] = [
        {
            "effective_date": "2000-01-02",
            "source_cash": "3",
            "declared_cash": "1.03",
            "currency": "USD",
            "entitlement_basis": "pre_share_event",
            "status": "candidate",
            "reason": "Issuer declaration",
            "evidence_urls": ["https://example.org/a", "https://example.org/b"],
        }
    ]
    c, stats = sb.reconstruct(source, doc, bronze_sha256="a" * 64)
    assert c.candidate_cash_dividend.iloc[1] == pytest.approx(1 / 1.03)
    assert not c.unresolved_cash.iloc[1]
    assert stats["cash_reconciliations"] == 1
    assert c.dividends.iloc[1] == 3
    doc["cash_reconciliations"][0]["source_cash"] = "2"
    with pytest.raises(ValueError, match="source value"):
        sb.reconstruct(source, doc, bronze_sha256="a" * 64)


def test_existing_split_and_checksum_fail_closed():
    doc = registry()
    doc["events"][0]["effective_date"] = "2000-01-03"
    with pytest.raises(ValueError, match="already recorded"):
        sb.reconstruct(data(), doc, bronze_sha256="a" * 64)
    with pytest.raises(ValueError, match="checksum"):
        sb.reconstruct(data(), registry(), bronze_sha256="b" * 64)


def test_immutable_s3_candidate_roundtrip():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = "test-bucket"
        s3.create_bucket(Bucket=bucket)
        blob = parquet_bytes(data())
        doc = registry(digest(blob))
        s3.put_object(Bucket=bucket, Key=doc["source_key"], Body=blob)
        s3.put_object(Bucket=bucket, Key=doc["evidence_key"], Body=b"evidence")
        version = sb.publish_registry(s3, bucket, doc, execute=True)["version"]
        assert sb.build(s3, bucket, version=version, run_id="pilot")["status"] == "dry_run"
        result = sb.build(s3, bucket, version=version, run_id="pilot", execute=True)
        assert result["status"] == "candidate_written"
        assert result == sb.build(s3, bucket, version=version, run_id="pilot", execute=True)
        assert s3.get_object(Bucket=bucket, Key=doc["source_key"])["Body"].read() == blob
