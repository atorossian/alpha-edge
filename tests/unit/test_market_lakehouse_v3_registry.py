"""AEA-91: current universe is selected from five and only five source files."""

from __future__ import annotations

import csv
import io

import boto3
import pytest
from moto import mock_aws

from alpha_edge.market.lakehouse_v3.registry_v3 import (
    SOURCE_FILES,
    activate_registry,
    build_registry,
    load_active,
    read_registry,
    registry_key,
    stage_registry,
    validate_registry,
)
from alpha_edge.market.lakehouse_v3.storage import ROOT, LakehouseError, put_json


def _csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def source_fixtures():
    universe = [
        {
            "row_id": "r-spy",
            "asset_id": "EQH001",
            "ticker": "SPY",
            "yahoo_ticker": "SPY",
            "include": "1",
            "currency": "USD",
            "market": "us_market",
            "calendar_id": "",
            "exchange": "PCX",
            "asset_class": "equity",
        },
        {
            "row_id": "r-eur",
            "asset_id": "EQH002",
            "ticker": "EUR-USD",
            "yahoo_ticker": "EURUSD=X",
            "include": "1",
            "currency": "USD",
            "market": "ccy_market",
            "calendar_id": "FOREX",
            "exchange": "CCY",
            "asset_class": "fx",
        },
        {
            "row_id": "r-log",
            "asset_id": "EQH003",
            "ticker": "LOG",
            "yahoo_ticker": "LOG.MC",
            "include": "1",
            "currency": "EUR",
            "market": "es_market",
            "calendar_id": "",
            "exchange": "MCE",
            "asset_class": "equity",
        },
        {
            "row_id": "r-bad",
            "asset_id": "EQH004",
            "ticker": "MISSING",
            "yahoo_ticker": "MISS",
            "include": "1",
            "currency": "",
            "market": "",
            "calendar_id": "",
            "exchange": "",
            "asset_class": "equity",
        },
        {
            "row_id": "r-old",
            "asset_id": "EQH005",
            "ticker": "OLD",
            "yahoo_ticker": "OLD",
            "include": "0",
            "currency": "USD",
            "market": "us_market",
            "calendar_id": "",
            "exchange": "NYQ",
            "asset_class": "equity",
        },
    ]
    overrides = [{"target_row_id": "r-eur", "ticker": "EUR-USD", "yahoo_ticker": "EURUSD=X", "lock_yahoo_ticker": "1"}]
    exclusions = [{"target_row_id": "r-old", "ticker": "OLD", "reason": "Not tradable"}]
    refresh = [{"ticker": "LOG"}]
    additions = [{"ticker": "EUR-USD", "yahoo_ticker": "EUR-USD", "lock_yahoo_ticker": "1"}]
    return dict(
        zip(
            SOURCE_FILES,
            [
                _csv(universe),
                _csv(overrides),
                _csv(exclusions),
                _csv(refresh),
                _csv(additions),
            ],
        )
    )


def test_five_file_selection_is_authoritative_and_retains_blocked():
    doc = build_registry(source_fixtures())
    assert doc["summary"]["universe_total"] == 5
    assert doc["summary"]["included"] == 4
    assert doc["summary"]["excluded"] == 1
    assert len(doc["assets"]) == 4
    assert {a["ticker"] for a in doc["assets"]} == {"SPY", "EUR-USD", "LOG", "MISSING"}
    assert doc["excluded"][0]["ticker"] == "OLD"
    assert next(a for a in doc["assets"] if a["ticker"] == "SPY")["calendar_id"] == "NYSE"
    assert next(a for a in doc["assets"] if a["ticker"] == "LOG")["force_refresh"]
    assert next(a for a in doc["assets"] if a["ticker"] == "EUR-USD")["blockers"] == [
        "manual_locked_yahoo_symbol_conflict"
    ]
    missing = next(a for a in doc["assets"] if a["ticker"] == "MISSING")
    assert missing["source_status"] == "blocked"
    assert "missing_or_invalid_quote_currency" in missing["blockers"]
    assert "unknown_market_calendar" in missing["blockers"]
    assert doc["summary"]["configured"] == 2
    assert doc["summary"]["blocked"] == 2
    assert doc["fx_pairs"][0]["currency"] == "EUR"
    assert doc["fx_pairs"][0]["provider_symbol"] == "EURUSD=X"
    assert doc["fx_pairs"][0]["verification"] == "unverified_candidate"


def test_only_five_files_allowed_without_any_old_data():
    data = source_fixtures()
    data["historical_price_export.csv"] = b"unapproved"
    with pytest.raises(LakehouseError, match="only these five"):
        build_registry(data)
    data.pop("historical_price_export.csv")
    data.pop("universe_overrides.csv")
    with pytest.raises(LakehouseError, match="only these five"):
        build_registry(data)


@pytest.mark.parametrize(
    "modify",
    [
        lambda x: x.update(asset_id=""),
        lambda x: x.update(include="maybe"),
        lambda x: x.update(row_id="r-old"),
    ],
)
def test_canonical_source_identity_and_include_fail_closed(modify):
    files = source_fixtures()
    lines = list(csv.DictReader(io.StringIO(files["universe.csv"].decode())))
    modify(lines[0])
    files["universe.csv"] = _csv(lines)
    with pytest.raises(LakehouseError):
        build_registry(files)


def test_exclusion_does_not_resurrect_or_silently_delete_asset():
    files = source_fixtures()
    lines = list(csv.DictReader(io.StringIO(files["universe.csv"].decode())))
    for row in lines:
        if row["ticker"] == "OLD":
            row["include"] = "1"
    files["universe.csv"] = _csv(lines)
    doc = build_registry(files)
    old = next(a for a in doc["assets"] if a["ticker"] == "OLD")
    assert "excluded_but_included_conflict" in old["blockers"]


def test_bad_locked_override_stops_that_asset_not_whole_registry():
    files = source_fixtures()
    lines = list(csv.DictReader(io.StringIO(files["universe_overrides.csv"].decode())))
    lines[0]["yahoo_ticker"] = "BAD=X"
    files["universe_overrides.csv"] = _csv(lines)
    doc = build_registry(files)
    assert "override_symbol_conflict" in next(a for a in doc["assets"] if a["ticker"] == "EUR-USD")["blockers"]
    assert len(doc["assets"]) == 4


def test_staging_versioning_retries_and_no_active_pointer():
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(Bucket="alpha-edge-algo", CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
        files = source_fixtures()
        dry = stage_registry(s3, "alpha-edge-algo", files)
        assert dry["writes"] == 0
        assert s3.list_objects_v2(Bucket="alpha-edge-algo", Prefix=ROOT).get("KeyCount") == 0
        written = stage_registry(s3, "alpha-edge-algo", files, execute=True)
        assert written["writes"] == 6
        assert written["registry_key"] == dry["registry_key"]
        assert (
            read_registry(s3, "alpha-edge-algo", written["registry_key"], written["sha256"])["summary"]["included"] == 4
        )
        with pytest.raises(LakehouseError):
            load_active(s3, "alpha-edge-algo")
        assert stage_registry(s3, "alpha-edge-algo", files, execute=True)["registry_key"] == dry["registry_key"]
        content = source_fixtures()
        content["universe_manual_additions.csv"] += b"\n"
        changed = stage_registry(s3, "alpha-edge-algo", content)
        assert changed["registry_key"] != written["registry_key"]


def test_runtime_rejects_unpinned_registry_bytes_and_invalid_pointer():
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(Bucket="alpha-edge-algo", CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
        staged = stage_registry(s3, "alpha-edge-algo", source_fixtures(), execute=True)
        with pytest.raises(LakehouseError, match="pinned"):
            read_registry(s3, "alpha-edge-algo", staged["registry_key"], "0" * 64)
        put_json(s3, "alpha-edge-algo", f"{ROOT}/config/active.json", {"schema": "not-authoritative"})
        with pytest.raises(LakehouseError, match="Unsupported"):
            load_active(s3, "alpha-edge-algo")


def test_registry_checksum_binding_and_determinism():
    files = source_fixtures()
    doc = build_registry(files)
    validate_registry(doc)
    assert registry_key(doc) == registry_key(build_registry(files))
    doc["assets"][0]["source_status"] = "blocked"
    with pytest.raises(LakehouseError):
        validate_registry(doc)


def test_configuration_activation_requires_complete_sources_and_never_approves_prices():
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(Bucket="alpha-edge-algo", CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
        candidate = stage_registry(s3, "alpha-edge-algo", source_fixtures(), execute=True)
        options = {
            "key": candidate["registry_key"],
            "sha256": candidate["sha256"],
        }
        preview = activate_registry(s3, "alpha-edge-algo", **options)
        assert preview["writes"] == 0
        assert preview["market_prices_approved"] is False
        with pytest.raises(LakehouseError):
            load_active(s3, "alpha-edge-algo")
        activated = activate_registry(s3, "alpha-edge-algo", **options, execute=True)
        assert activated["writes"] == 1
        data, pointer = load_active(s3, "alpha-edge-algo")
        assert data["summary"]["included"] == 4
        assert pointer["market_prices_approved"] is False
        assert activated["blocked"] == 2
        with pytest.raises(LakehouseError, match="expected previous SHA"):
            activate_registry(s3, "alpha-edge-algo", **options, execute=True)


def test_activation_rejects_tampered_configuration_source_bytes():
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(Bucket="alpha-edge-algo", CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
        candidate = stage_registry(s3, "alpha-edge-algo", source_fixtures(), execute=True)
        source = read_registry(s3, "alpha-edge-algo", candidate["registry_key"], candidate["sha256"])
        key = f"{ROOT}/config/source_snapshots/{source['source_fingerprint']}/universe.csv"
        s3.put_object(Bucket="alpha-edge-algo", Key=key, Body=b"corrupted")
        with pytest.raises(LakehouseError, match="checksum/size"):
            activate_registry(
                s3,
                "alpha-edge-algo",
                key=candidate["registry_key"],
                sha256=candidate["sha256"],
                execute=True,
            )
        with pytest.raises(LakehouseError):
            load_active(s3, "alpha-edge-algo")


def test_unknown_stable_override_row_id_fails_closed():
    files = source_fixtures()
    overrides = list(csv.DictReader(io.StringIO(files["universe_overrides.csv"].decode())))
    overrides[0]["target_row_id"] = "nonexistent-row"
    files["universe_overrides.csv"] = _csv(overrides)
    with pytest.raises(LakehouseError, match="unknown or ambiguous row target"):
        build_registry(files)
