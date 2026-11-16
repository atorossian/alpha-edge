"""AEA-90 pilot identity and mandatory contract regression tests.

These are synthetic manifests and static registry definitions, never real prices
or downloaded market data. S3 is mocked in memory.
"""

from __future__ import annotations

import copy

import boto3
import pytest
from moto import mock_aws

from alpha_edge.market.lakehouse_v3.contracts import (
    validate_bronze_manifest,
    validate_candidate_manifest,
    validate_registry_document,
)
from alpha_edge.market.lakehouse_v3.guardrails import lint_text, validate_source_tree
from alpha_edge.market.lakehouse_v3.ingest import registry_from_s3
from alpha_edge.market.lakehouse_v3.storage import ROOT, LakehouseError, put_json

BUCKET = "alpha-edge-algo"
RUN = "pilot-contract-20261116"
SHA = "a" * 64

# Exact asset IDs and source tickers verified against data/universe/universe.csv
# on integration/aea87-market-data-lake-v2. The proposed calendars for
# Spanish/Italian/Mexican equity are awaiting actual calendar smoke tests.
PILOT_ASSETS = [
    ("EQH7f9e4349dabafdd9", "SPY", "USD", "NYSE", "ETF"),
    ("EQH4561ad9433b2a662", "GLD", "USD", "NYSE", "ETF"),
    ("CRYPTO:BTC-USD", "BTC-USD", "USD", "24/7", "CRYPTO"),
    ("EQH0f232d452fb06a46", "RENT3.SA", "BRL", "BMF", "EQUITY"),
    ("EQH11f97bdc8d8dfe85", "TRAXIONA.MX", "MXN", "XMEX", "EQUITY"),
    ("EQH2325a2fe8044198c", "LOG.MC", "EUR", "XMAD", "EQUITY"),
    ("EQH2834655e74af1b24", "REY.MI", "EUR", "XMIL", "EQUITY"),
    ("EQHe874aca04f607837", "EURUSD=X", "USD", "FOREX", "FX"),
]


def pilot_registry():
    return {
        "schema": "alpha_edge_registry_v3_1",
        "review_state": "pilot_only_unverified",
        "assets": [
            {
                "asset_id": aid,
                "yahoo_ticker": ticker,
                "currency": currency,
                "calendar_id": calendar,
                "asset_class": kind,
            }
            for aid, ticker, currency, calendar, kind in PILOT_ASSETS
        ],
        "fx_pairs": [
            {"currency": "BRL", "provider_symbol": "USDBRL=X", "direction": "inverse"},
            {"currency": "MXN", "provider_symbol": "USDMXN=X", "direction": "inverse"},
            {"currency": "EUR", "provider_symbol": "EURUSD=X", "direction": "direct"},
        ],
    }


def _ref(prefix, filename):
    return {"key": f"{ROOT}/{prefix}/{filename}", "sha256": SHA, "bytes": 180}


def bronze():
    return {
        "schema": "alpha_edge_bronze_v3_1",
        "run_id": RUN,
        "status": "bronze_committed",
        "registry_key": f"{ROOT}/config/fixture.json",
        "registry_sha256": SHA,
        "start": "2025-01-01",
        "end_exclusive": "2026-11-16",
        "assets": 8,
        "rows": {"bars": 100, "actions": 5, "fx": 32},
        "artifacts": {
            k: _ref(f"bronze/runs/{RUN}", fn)
            for k, fn in {
                "bars": "source_ohlcv.parquet",
                "actions": "source_actions.parquet",
                "fx": "source_fx.parquet",
            }.items()
        },
        "provenance": [{"provider": "yahoo_finance"}],
        "completed_at_utc": "2026-11-16T10:30:00+01:00",
        "price_basis": "yahoo_unadjusted_auto_adjust_false_close_is_provider_split_adjusted",
    }


def candidate():
    return {
        "schema": "alpha_edge_gold_candidate_v3_1",
        "run_id": RUN,
        "status": "review_required",
        "approved": False,
        "bronze_manifest_key": f"{ROOT}/bronze/runs/{RUN}/manifest.json",
        "bronze_registry_sha256": SHA,
        "quality": {"assets": 8, "bronze_rows": 100, "silver_rows": 100, "gold_rows": 100},
        "artifacts": {
            k: _ref(f"candidates/{RUN}", fn)
            for k, fn in {
                "silver": "silver/data.parquet",
                "gold": "gold/data.parquet",
                "quality": "quality/report.json",
                "session_issues": "quality/session_issues.parquet",
            }.items()
        },
    }


def test_all_eight_assets_and_three_fx_source_roles():
    doc = pilot_registry()
    validate_registry_document(doc)
    assert len(doc["assets"]) == 8
    assert {x["currency"] for x in doc["assets"]} == {"USD", "BRL", "MXN", "EUR"}
    assert {x["currency"] for x in doc["fx_pairs"]} == {"BRL", "MXN", "EUR"}
    assert next(a for a in doc["assets"] if a["yahoo_ticker"] == "EURUSD=X")["currency"] == "USD"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["assets"][1].update(asset_id=d["assets"][0]["asset_id"]),
        lambda d: d["assets"][1].update(yahoo_ticker=d["assets"][0]["yahoo_ticker"]),
        lambda d: d["assets"][2].update(calendar_id="NYSE"),
        lambda d: d["assets"][7].update(calendar_id="24/7"),
        lambda d: d["assets"][4].update(currency="UNKNOWN"),
        lambda d: d["fx_pairs"].pop(),
        lambda d: d["fx_pairs"][0].update(direction="guess"),
        lambda d: d.update(schema="alpha_edge_registry_v999"),
    ],
)
def test_registry_fails_closed_on_identity_currency_calendar_and_version(mutation):
    doc = pilot_registry()
    mutation(doc)
    with pytest.raises(LakehouseError):
        validate_registry_document(doc)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema="old-version"),
        lambda d: d.update(status="uncommitted"),
        lambda d: d.update(run_id="other-run"),
        lambda d: d["artifacts"]["bars"].update(sha256="invalid"),
        lambda d: d["artifacts"]["actions"].update(key="market/returns_usd/v1/legacy.parquet"),
        lambda d: d["artifacts"]["fx"].update(bytes=0),
        lambda d: d["rows"].update(bars=0),
        lambda d: d.update(completed_at_utc="2026-11-16"),
        lambda d: d.update(start="2026-12-01", end_exclusive="2026-11-16"),
        lambda d: d.pop("provenance"),
    ],
)
def test_bronze_manifest_rejects_broken_lineage(mutation):
    doc = bronze()
    mutation(doc)
    with pytest.raises(LakehouseError):
        validate_bronze_manifest(doc, run_id=RUN)


def test_bronze_manifest_accepts_valid_pinned_provenance():
    validate_bronze_manifest(bronze(), run_id=RUN)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema="v999"),
        lambda d: d.update(status="approved"),
        lambda d: d.update(approved=True),
        lambda d: d.update(bronze_manifest_key="market/lake_v3/bronze/runs/wrong/manifest.json"),
        lambda d: d["artifacts"]["gold"].update(key="market/lake_v3/candidates/other/gold/data.parquet"),
        lambda d: d["quality"].update(gold_rows=99),
    ],
)
def test_candidate_manifest_cannot_bypass_lineage_or_approval(mutation):
    doc = candidate()
    mutation(doc)
    with pytest.raises(LakehouseError):
        validate_candidate_manifest(doc, run_id=RUN)


def test_candidate_manifest_accepts_pending_and_reconciled_state():
    validate_candidate_manifest(candidate(), run_id=RUN)


def test_pilot_config_roundtrip_uses_mocked_s3_not_local_files():
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
        )
        key = f"{ROOT}/config/pilot.json"
        put_json(s3, BUCKET, key, pilot_registry())
        assets, fx, source_sha = registry_from_s3(s3, BUCKET, key)
        assert set(assets["yahoo_ticker"]) == {x[1] for x in PILOT_ASSETS}
        assert len(fx) == 3
        assert len(source_sha) == 64
        corrupt = copy.deepcopy(pilot_registry())
        corrupt["assets"][0]["currency"] = "BRL"
        corrupt["fx_pairs"] = [x for x in corrupt["fx_pairs"] if x["currency"] != "BRL"]
        with pytest.raises(LakehouseError):
            validate_registry_document(corrupt)


@pytest.mark.parametrize(
    "source",
    [
        "from pathlib import Path\\nPath('market.parquet').write_bytes(b'data')",
        "import tempfile\\ntempfile.TemporaryDirectory()",
        "import pandas as pd\\npd.DataFrame().to_csv('data.csv')",
        "import pandas as pd\\npd.read_parquet('local.parquet')",
        "import pandas as pd\\npd.DataFrame().to_parquet('local.parquet')",
        "open('data.csv', 'wb')",
    ],
)
def test_static_architecture_guard_rejects_local_market_io(source):
    assert lint_text(source.replace("\\n", "\n"), "candidate.py")
    with pytest.raises(LakehouseError):
        validate_source_tree({"candidate.py": source.replace("\\n", "\n")})


def test_static_architecture_guard_allows_only_memory_bytesio_exceptions():
    safe_source = "import pandas as pd\\nimport io\\npd.read_parquet(io.BytesIO(b'data'))"
    unsafe_source = "import pandas as pd\\npd.read_parquet('data.parquet')"
    assert not lint_text(safe_source.replace("\\n", "\n"), "storage.py")
    assert lint_text(unsafe_source.replace("\\n", "\n"), "storage.py")


@pytest.mark.parametrize(
    ("symbol", "calendar_id"),
    [("LOG.MC", "XMAD"), ("REY.MI", "XMIL")],
)
def test_madrid_and_milan_calendar_identifiers_work_in_project_dependency(symbol, calendar_id):
    """Require an actual supported calendar and correct Easter exchange sessions."""
    import pandas_market_calendars as mcal

    calendar = mcal.get_calendar(calendar_id)
    sessions = {
        session.strftime("%Y-%m-%d")
        for session in calendar.valid_days(start_date="2026-04-02", end_date="2026-04-08")
    }
    # Both exchanges trade on Holy Thursday 2026, unlike Good Friday.
    assert "2026-04-02" in sessions, (symbol, calendar_id)
    assert "2026-04-03" not in sessions, (symbol, calendar_id)  # Good Friday
    assert "2026-04-06" not in sessions, (symbol, calendar_id)  # Easter Monday
    assert "2026-04-07" in sessions, (symbol, calendar_id)
    schedule = calendar.schedule(start_date="2026-04-07", end_date="2026-04-07")
    assert len(schedule) == 1
    assert schedule.iloc[0]["market_open"] < schedule.iloc[0]["market_close"]
