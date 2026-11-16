"""AEA-91 S3-only registry construction and immutable staging regressions."""

from __future__ import annotations

from unittest.mock import Mock

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market.lakehouse_v3.contracts import validate_pilot_registry_document
from alpha_edge.market.lakehouse_v3.registry_v3 import (
    ARCHIVE_ID,
    PILOT_POLICY,
    build_pilot_registry,
    stage_pilot_registry,
)
from alpha_edge.market.lakehouse_v3.storage import ROOT, LakehouseError, read_json


@pytest.fixture
def fake_sources(monkeypatch):
    """Source replicas are in memory and deliberately not approved."""
    ids = [row[0] for row in PILOT_POLICY]
    original = [
        {"asset_id": aid, "yahoo_ticker": ticker, "currency": currency, "calendar_id": calendar}
        for aid, ticker, currency, calendar, *_ in PILOT_POLICY
    ]
    old = next(x for x in original if x["yahoo_ticker"] == "EURUSD=X")
    old.update({"yahoo_ticker": "EUR-USD", "calendar_id": "24/7"})
    frames = pd.DataFrame(original)
    inventory = {
        aid: {
            "asset_id": aid,
            "current_eligible": True,
            "production_eligible": False,
            "external_evidence_approved": False,
            "candidate_state": "original_candidate_marker_unverified",
            "original_plan_present": True,
        }
        for aid in ids
    }
    inventory["EQHe874aca04f607837"]["reviewed_symbol"] = "EURUSD=X"
    refsha = "f" * 64
    monkeypatch.setattr(
        "alpha_edge.market.lakehouse_v3.registry_v3.load_archived_plan",
        lambda *args: (frames, {"plan_sha256": refsha}),
    )
    monkeypatch.setattr(
        "alpha_edge.market.lakehouse_v3.registry_v3._load_inventory",
        lambda *args: (
            inventory,
            {
                "original_plan_sha256": refsha,
                "inventory_sha256": "a" * 64,
                "current_universe_sha256": "b" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        "alpha_edge.market.lakehouse_v3.registry_v3._archived_fx",
        lambda *args: [
            {"currency": "BRL", "provider_symbol": "USDBRL=X", "direction": "inverse"},
            {"currency": "EUR", "provider_symbol": "EURUSD=X", "direction": "direct"},
            {"currency": "MXN", "provider_symbol": "USDMXN=X", "direction": "inverse"},
        ],
    )
    return frames, inventory


def test_eight_asset_candidate_preserves_eurusd_alias_without_fabricating_date(fake_sources):
    doc = build_pilot_registry(Mock(), "alpha-edge-algo", archive_id=ARCHIVE_ID)
    validate_pilot_registry_document(doc)
    assert doc["review_state"] == "review_required"
    assert len(doc["assets"]) == 8
    assert set(x["currency"] for x in doc["fx_pairs"]) == {"BRL", "MXN", "EUR"}
    fx = next(x for x in doc["assets"] if x["yahoo_ticker"] == "EURUSD=X")
    assert {a["provider_symbol"] for a in fx["source_aliases"]} == {"EUR-USD", "EURUSD=X"}
    assert all(a["effective_from"] is None and a["effective_to"] is None for a in fx["source_aliases"])
    assert "eurusd_effective_dated_alias_and_forex_calendar_review" in doc["release_blockers"]
    assert all(x["include"] is True and x["historical_retention_required"] is True for x in doc["assets"])


def test_exclusion_blocks_candidate_source(fake_sources):
    _, inv = fake_sources
    inv[PILOT_POLICY[0][0]]["current_eligible"] = False
    with pytest.raises(LakehouseError, match="excluded or historical-only"):
        build_pilot_registry(Mock(), "alpha-edge-algo")


def test_unreviewed_symbol_rename_blocks_candidate(fake_sources):
    frame, _ = fake_sources
    frame.loc[frame["yahoo_ticker"].eq("LOG.MC"), "yahoo_ticker"] = "LOG-OLD"
    with pytest.raises(LakehouseError, match="unreviewed provider"):
        build_pilot_registry(Mock(), "alpha-edge-algo")


def test_forex_alias_without_reviewed_symbol_blocks_candidate(fake_sources):
    _, inv = fake_sources
    inv["EQHe874aca04f607837"].pop("reviewed_symbol")
    with pytest.raises(LakehouseError, match="unreviewed provider"):
        build_pilot_registry(Mock(), "alpha-edge-algo")


def test_currency_conflict_blocks_candidate(fake_sources):
    frame, _ = fake_sources
    frame.loc[frame["yahoo_ticker"].eq("LOG.MC"), "currency"] = "GBP"
    with pytest.raises(LakehouseError, match="currency conflicts"):
        build_pilot_registry(Mock(), "alpha-edge-algo")


def test_review_only_registry_does_not_pass_legacy_ingestion_contract(fake_sources):
    from alpha_edge.market.lakehouse_v3.contracts import validate_registry_document

    doc = build_pilot_registry(Mock(), "alpha-edge-algo")
    with pytest.raises(LakehouseError, match="Unsupported registry schema"):
        validate_registry_document(doc)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(review_state="approved"),
        lambda d: d["assets"][0].update(quote_to_currency_factor=100.0),
        lambda d: d["assets"][0].update(include=False),
        lambda d: d["assets"][0]["source_aliases"][0].update(effective_dates_verified=True),
        lambda d: d["assets"][0]["source_aliases"][0].update(effective_from="2020-01-01"),
        lambda d: d["assets"][-1].update(source_aliases=[]),
        lambda d: d["assets"][0].update(exchange_timezone="invalid/timezone"),
        lambda d: d.update(release_blockers=[]),
    ],
)
def test_review_registry_rejects_unsafe_promotions_and_fabricated_aliases(fake_sources, change):
    doc = build_pilot_registry(Mock(), "alpha-edge-algo")
    change(doc)
    with pytest.raises(LakehouseError):
        validate_pilot_registry_document(doc)


def test_staging_dry_run_and_immutable_review_candidate_in_mock_s3(fake_sources):
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(
            Bucket="alpha-edge-algo",
            CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
        )
        dry = stage_pilot_registry(s3, "alpha-edge-algo")
        assert dry["writes"] == 0
        assert dry["state"] == "review_required"
        assert dry["registry_key"].startswith(f"{ROOT}/config/candidates/")
        assert s3.list_objects_v2(Bucket="alpha-edge-algo", Prefix=ROOT).get("KeyCount") == 0
        result = stage_pilot_registry(s3, "alpha-edge-algo", execute=True)
        assert result["registry_key"] == dry["registry_key"]
        assert result["sha256"] == dry["sha256"]
        assert result["writes"] == 1
        saved = read_json(s3, "alpha-edge-algo", result["registry_key"])
        validate_pilot_registry_document(saved)
        # Exact-byte retries are safe; a mismatched existing object fails.
        assert stage_pilot_registry(s3, "alpha-edge-algo", execute=True)["registry_key"] == result["registry_key"]
        s3.put_object(Bucket="alpha-edge-algo", Key=result["registry_key"], Body=b"corrupted")
        with pytest.raises(LakehouseError, match="collision"):
            stage_pilot_registry(s3, "alpha-edge-algo", execute=True)
