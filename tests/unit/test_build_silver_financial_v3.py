"""S3-only Silver financial candidate tests using fake immutable Bronze runs."""

from __future__ import annotations

import io

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market import (
    audit_financial_universe_v3 as diagnostic,
    build_silver_financial_v3 as silver,
    ingest_bronze_full_history as bronze,
)
from alpha_edge.market.dividend_currency_registry_v3 import (
    RATE_REFERENCE,
    REGISTRY_ROOT,
    SCHEMA,
    UNITS_PER_EUR,
    registry_version,
)
from alpha_edge.market.lakehouse_v3.storage import put_json

BUCKET = "alpha-edge-algo"
BRONZE_RUN = "bronze-financial-test"
SILVER_RUN = "silver-financial-test"


def csv_bytes(rows):
    import csv

    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def source():
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
        universe = [
            {
                "asset_id": "EQH-SPY",
                "row_id": "one",
                "ticker": "SPY",
                "yahoo_ticker": "SPY",
                "include": "1",
                "currency": "USD",
            },
            {
                "asset_id": "EQH-JPY",
                "row_id": "two",
                "ticker": "USD-JPY",
                "yahoo_ticker": "JPY=X",
                "include": "1",
                "currency": "JPY",
            },
        ]
        sources = {
            "universe.csv": csv_bytes(universe),
            "universe_overrides.csv": csv_bytes([{"target_row_id": "", "ticker": "SPY"}]),
            "asset_excluded.csv": csv_bytes([{"target_row_id": "", "ticker": "OLD"}]),
            "ingest_force_refresh.csv": csv_bytes([{"ticker": "SPY"}]),
            "universe_manual_additions.csv": csv_bytes([{"ticker": "SPY"}]),
        }
        metadata = {name: {"bytes": len(data), "sha256": bronze._digest(data)} for name, data in sources.items()}
        fingerprint = bronze._digest(bronze._json_bytes(metadata))
        for name, data in sources.items():
            s3.put_object(
                Bucket=BUCKET,
                Key=f"{bronze.SNAPSHOT_ROOT}/{fingerprint}/{name}",
                Body=data,
                Metadata={"sha256": bronze._digest(data), "source-git-commit": "a" * 40},
            )
        run = {
            "schema": "alpha_edge_full_history_run_v1",
            "run_id": BRONZE_RUN,
            "source_fingerprint": fingerprint,
            "requested_asset_ids": ["EQH-SPY", "EQH-JPY"],
        }
        s3.put_object(
            Bucket=BUCKET, Key=f"{bronze.BRONZE_ROOT}/runs/{BRONZE_RUN}/run.json", Body=bronze._json_bytes(run)
        )
        for asset, ticker, closes, actions in [
            ("EQH-SPY", "SPY", [100.0, 101.0, 102.0], [0.0, 0.5, 0.0]),
            ("EQH-JPY", "JPY=X", [150.0, 151.0, 152.0], [0.0, 0.0, 0.0]),
        ]:
            bars = pd.DataFrame(
                {
                    "asset_id": [asset] * 3,
                    "provider_symbol": [ticker] * 3,
                    "currency_original": ["USD" if ticker == "SPY" else "JPY"] * 3,
                    "session_date": ["2026-11-12", "2026-11-13", "2026-11-16"],
                    "close": closes,
                    "adj_close": closes,
                    "dividends": actions,
                    "stock_splits": [0.0] * 3,
                }
            )
            buf = io.BytesIO()
            bars.to_parquet(buf, index=False)
            blob = buf.getvalue()
            prefix = bronze._target_prefix(BRONZE_RUN, asset)
            key = f"{prefix}/objects/{bronze._digest(blob)}.parquet"
            s3.put_object(Bucket=BUCKET, Key=key, Body=blob, Metadata={"sha256": bronze._digest(blob)})
            receipt = {
                "schema": "alpha_edge_full_history_asset_v1",
                "asset_id": asset,
                "provider_symbol": ticker,
                "run_id": BRONZE_RUN,
                "status": "committed",
                "data": {"key": key, "bytes": len(blob), "sha256": bronze._digest(blob)},
                "validation": {"rows": len(bars)},
            }
            s3.put_object(Bucket=BUCKET, Key=f"{prefix}/manifest.json", Body=bronze._json_bytes(receipt))
        yield s3, fingerprint


def test_spy_actions_dry_run_no_s3_write(source):
    s3, fingerprint = source
    result = silver.build(
        s3,
        BUCKET,
        fingerprint=fingerprint,
        bronze_run_id=BRONZE_RUN,
        silver_run_id=SILVER_RUN,
        selector="SPY",
    )
    assert result["status"] == "dry_run"
    assert result["statistics"]["action_statistics"]["reported_dividends"] == 1
    assert result["outputs"]["actions"] == 1
    assert not result["production_eligible"]
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix=silver.ROOT).get("KeyCount", 0) == 0


def test_optional_registry_is_pinned_in_dry_run_manifest(source):
    s3, fingerprint = source
    registry = {"schema": SCHEMA, "rate_reference": RATE_REFERENCE, "units_per_eur": UNITS_PER_EUR, "entries": []}
    version = registry_version(registry)
    put_json(s3, BUCKET, f"{REGISTRY_ROOT}/{version}/registry.json", registry)
    result = silver.build(
        s3,
        BUCKET,
        fingerprint=fingerprint,
        bronze_run_id=BRONZE_RUN,
        silver_run_id=SILVER_RUN,
        selector="SPY",
        dividend_currency_registry_version=version,
    )
    assert result["statistics"]["dividend_currency_correction"]["registry_version"] == version
    assert result["outputs"]["currency_corrected_actions"] == 1
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix=silver.ROOT).get("KeyCount", 0) == 0


def test_strict_quarantine_in_dry_run_manifest(source):
    s3, fingerprint = source
    result = silver.build(
        s3,
        BUCKET,
        fingerprint=fingerprint,
        bronze_run_id=BRONZE_RUN,
        silver_run_id=SILVER_RUN,
        selector="SPY",
        strict_historical_quarantine=True,
    )
    assert result["statistics"]["historical_quarantine"]["quarantine_through"] == "2026-11-13"
    assert result["statistics"]["historical_quarantine"]["candidate_returns"] == 0
    assert result["outputs"]["historical_quarantine_sessions"] == 3
    assert not result["production_eligible"]
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix=silver.ROOT).get("KeyCount", 0) == 0


def test_universe_diagnostic_persists_coverage_and_findings_only_on_execute(source):
    s3, fingerprint = source
    registry = {"schema": SCHEMA, "rate_reference": RATE_REFERENCE, "units_per_eur": UNITS_PER_EUR, "entries": []}
    version = registry_version(registry)
    put_json(s3, BUCKET, f"{REGISTRY_ROOT}/{version}/registry.json", registry)
    kwargs = {
        "fingerprint": fingerprint,
        "bronze_run_id": BRONZE_RUN,
        "run_id": "diagnostic-test",
        "currency_version": version,
    }
    dry = diagnostic.audit(s3, BUCKET, **kwargs)
    assert dry["statistics"]["assets"] == 2
    assert dry["statistics"]["dividend_reasons"] == {"dividend_factor_mismatch": 1}
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix=diagnostic.AUDIT_ROOT)["KeyCount"] == 0
    result = diagnostic.audit(s3, BUCKET, **kwargs, execute=True)
    assert result["outputs"] == dry["outputs"]
    assert not result["production_eligible"]
    for ref in result["outputs"].values():
        blob = s3.get_object(Bucket=BUCKET, Key=ref["key"])["Body"].read()
        assert bronze._digest(blob) == ref["sha256"]
    assert diagnostic.audit(s3, BUCKET, **kwargs, execute=True) == result
    json_run = diagnostic.audit(
        s3, BUCKET, **{**kwargs, "run_id": "diagnostic-json-test"}, execute=True, artifact_format="json"
    )
    for ref in json_run["outputs"].values():
        import json

        blob = s3.get_object(Bucket=BUCKET, Key=ref["key"])["Body"].read()
        assert bronze._digest(blob) == ref["sha256"]
        assert len(json.loads(blob)["rows"]) == ref["rows"]


def test_fx_inverse_rate_source_evidence_and_same_run_resume(source):
    s3, fingerprint = source
    kwargs = {
        "fingerprint": fingerprint,
        "bronze_run_id": BRONZE_RUN,
        "silver_run_id": SILVER_RUN,
        "selector": "USD-JPY",
        "fx_currency": "JPY",
        "fx_direction": "inverse",
    }
    dry = silver.build(s3, BUCKET, **kwargs)
    assert dry["statistics"]["fx_statistics"]["accepted_rates"] == 3
    assert dry["statistics"]["fx_statistics"]["conversion_to_underlying_asset_prices_applied"] is False
    published = silver.build(s3, BUCKET, **kwargs, execute=True)
    assert published["status"] == "candidate_written"
    assert len(published["output_objects"]) == 4
    rate_ref = published["output_objects"]["fx_rates"]
    blob = s3.get_object(Bucket=BUCKET, Key=rate_ref["key"])["Body"].read()
    assert bronze._digest(blob) == rate_ref["sha256"]
    rates = pd.read_parquet(io.BytesIO(blob))
    assert rates.iloc[0]["fx_to_usd"] == pytest.approx(1 / 150)
    assert silver.build(s3, BUCKET, **kwargs, execute=True)["status"] == "already_committed"
    with pytest.raises(silver.SilverEvidenceError, match="different inputs"):
        silver.build(s3, BUCKET, **{**kwargs, "fx_direction": "direct"}, execute=True)


def test_missing_hzn_source_does_not_synthesize(source):
    s3, fingerprint = source
    with pytest.raises(silver.SilverEvidenceError, match="Exactly one"):
        silver.build(
            s3,
            BUCKET,
            fingerprint=fingerprint,
            bronze_run_id=BRONZE_RUN,
            silver_run_id=SILVER_RUN,
            selector="HZN",
        )


def test_invalid_fx_config_fails_before_s3_writes(source):
    s3, fingerprint = source
    with pytest.raises(silver.SilverEvidenceError, match="together"):
        silver.build(
            s3,
            BUCKET,
            fingerprint=fingerprint,
            bronze_run_id=BRONZE_RUN,
            silver_run_id=SILVER_RUN,
            selector="USD-JPY",
            fx_currency="JPY",
        )
