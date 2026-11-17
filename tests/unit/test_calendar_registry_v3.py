"""Calendar registry and per-asset Silver integration: all AWS mocked."""

from __future__ import annotations

import io

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market import build_silver_sessions_v3 as silver
from alpha_edge.market import calendar_registry_v3 as calendars
from alpha_edge.market import ingest_bronze_full_history as bronze

BUCKET = "alpha-edge-algo"


def _csv(rows):
    import csv

    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


@pytest.fixture
def snapshot():
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
        rows = [
            dict(asset_id="EQH-SPY", row_id="a", ticker="SPY", yahoo_ticker="SPY", include="1",
                 currency="USD", exchange="PCX", quoteType="ETF", calendar_id="", asset_class="equity"),
            dict(asset_id="CRYPTO:BTC-USD", row_id="b", ticker="BTC-USD", yahoo_ticker="BTC-USD", include="1",
                 currency="USD", exchange="CCC", quoteType="CRYPTOCURRENCY", calendar_id="24/7", asset_class="crypto"),
            dict(asset_id="EQH-FX", row_id="c", ticker="USD-JPY", yahoo_ticker="JPY=X", include="1",
                 currency="JPY", exchange="", quoteType="", calendar_id="FOREX", asset_class="fx"),
            dict(asset_id="EQH-UNRESOLVED", row_id="d", ticker="UNKNOWN", yahoo_ticker="UNKNOWN", include="1",
                 currency="USD", exchange="TEX", quoteType="EQUITY", calendar_id="", asset_class="equity"),
            dict(asset_id="EQH-EXCLUDED", row_id="e", ticker="OLD", yahoo_ticker="OLD", include="0",
                 currency="USD", exchange="NYQ", quoteType="EQUITY", calendar_id="", asset_class="equity"),
        ]
        source = {
            "universe.csv": _csv(rows),
            "universe_overrides.csv": _csv([{"target_row_id": "", "ticker": "SPY"}]),
            "asset_excluded.csv": _csv([{"target_row_id": "e", "ticker": "OLD"}]),
            "ingest_force_refresh.csv": _csv([{"ticker": "SPY"}]),
            "universe_manual_additions.csv": _csv([{"ticker": "SPY"}]),
        }
        meta = {name: {"bytes": len(blob), "sha256": bronze._digest(blob)} for name, blob in source.items()}
        fingerprint = bronze._digest(bronze._json_bytes(meta))
        for name, blob in source.items():
            s3.put_object(
                Bucket=BUCKET,
                Key=f"{bronze.SNAPSHOT_ROOT}/{fingerprint}/{name}",
                Body=blob,
                Metadata={"sha256": bronze._digest(blob), "source-git-commit": "a" * 40},
            )
        yield s3, fingerprint


def test_registry_proposes_known_exchanges_and_preserves_unknown(snapshot):
    s3, fingerprint = snapshot
    registry, version = calendars.make_registry(s3, BUCKET, fingerprint)
    by_symbol = {r["ticker"]: r for r in registry["assets"]}
    assert registry["counts"] == {"active": 4, "proposed": 3, "unresolved": 1}
    assert by_symbol["SPY"]["calendar_id"] == "NYSE"
    assert by_symbol["BTC-USD"]["calendar_id"] == "CRYPTO_24_7"
    assert by_symbol["USD-JPY"]["calendar_id"] == "YAHOO_FX_WEEKDAYS"
    assert by_symbol["UNKNOWN"]["calendar_id"] is None
    assert all(not row["calendar_validation_approved"] for row in registry["assets"])
    assert len(version) == 64
    assert registry["approval"] == "candidate_not_certified"
    # Excluded security never enters Silver policy assignment.
    assert "OLD" not in by_symbol


def test_publish_registry_is_immutable_replayable_and_s3_only(snapshot):
    s3, fingerprint = snapshot
    registry, version = calendars.make_registry(s3, BUCKET, fingerprint)
    dry = calendars.publish(s3, BUCKET, registry=registry, version=version, execute=False)
    assert not dry["executed"]
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix=calendars.REGISTRY_ROOT).get("KeyCount", 0) == 0
    first = calendars.publish(s3, BUCKET, registry=registry, version=version, execute=True)
    assert first["readback_verified"]
    assert calendars.publish(s3, BUCKET, registry=registry, version=version, execute=True)["readback_verified"]
    loaded = calendars.load_registry(s3, BUCKET, version, fingerprint)
    assert loaded == registry
    with pytest.raises(calendars.CalendarRegistryError, match="pinned"):
        calendars.load_registry(s3, BUCKET, version, "b" * 64)


def test_registry_cannot_hide_conflicting_explicit_calendar():
    data = {
        "asset_id": "EQH", "ticker": "A", "yahoo_ticker": "A", "exchange": "PCX",
        "quoteType": "ETF", "calendar_id": "XMAD", "asset_class": "equity",
    }
    result = calendars._derive(data)
    assert result["calendar_id"] is None
    assert result["mapping_status"] == "unresolved"


def test_special_crypto_weekends_are_required():
    raw = pd.DataFrame({
        "asset_id": ["CRYPTO:BTC"] * 2,
        "provider_symbol": ["BTC-USD"] * 2,
        "session_date": ["2026-11-13", "2026-11-15"],
    })
    flags, issues = silver._special_session_audit(raw, "CRYPTO_24_7")
    assert flags["previous_expected_session_missing"].iloc[-1]
    assert issues["issue"].tolist() == ["missing_expected_session"]
    assert issues["session_date"].iloc[0] == pd.Timestamp("2026-11-14")


def test_special_fx_daily_weekday_convention():
    raw = pd.DataFrame({
        "asset_id": ["EQH-FX"] * 2,
        "provider_symbol": ["JPY=X"] * 2,
        "session_date": ["2026-11-13", "2026-11-16"],
    })
    flags, issues = silver._special_session_audit(raw, "YAHOO_FX_WEEKDAYS")
    assert len(issues) == 0
    assert not flags["previous_expected_session_missing"].iloc[-1]


def test_silver_reads_pinned_registry_mapping_and_keeps_proposal_status(snapshot, monkeypatch):
    s3, fingerprint = snapshot
    registry, version = calendars.make_registry(s3, BUCKET, fingerprint)
    calendars.publish(s3, BUCKET, registry=registry, version=version, execute=True)

    def fake_complete(s3, bucket, *, run_id, record, verify_hash):
        return None

    monkeypatch.setattr(silver, "_completed", fake_complete)
    run = {
        "source_fingerprint": fingerprint,
        "requested_asset_ids": [r["asset_id"] for r in registry["assets"]],
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{bronze.BRONZE_ROOT}/runs/bronze-test-20261117/run.json",
        Body=bronze._json_bytes(run),
    )
    result = silver.build(
        s3, BUCKET, source_fingerprint=fingerprint,
        bronze_run_id="bronze-test-20261117",
        silver_run_id="silver-test-20261117",
        only=["SPY", "BTC-USD", "UNKNOWN"],
        registry_version=version,
    )
    assert result["missing_bronze"] == 3
    assert result["processed"] == 0
    assert result["production_eligible"] is False
    with pytest.raises(silver.SilverSessionError, match="either"):
        silver.build(
            s3, BUCKET, source_fingerprint=fingerprint,
            bronze_run_id="bronze-test-20261117", silver_run_id="silver-test-20261117",
            only=["SPY"], registry_version=version, calendar_id="NYSE",
        )
