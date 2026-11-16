"""Full-history S3 Bronze ingestion tests (no real network or local market files)."""

from __future__ import annotations

import io
import json
from unittest.mock import Mock

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market import ingest_bronze_full_history as bronze

BUCKET = "alpha-edge-algo"
SOURCE_FINGERPRINT = "f" * 64


def _csv(rows):
    output = io.StringIO(newline="")
    writer = __import__("csv").DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _sources():
    rows = [
        {
            "asset_id": "EQH001", "row_id": "row-spy", "ticker": "SPY",
            "yahoo_ticker": "SPY", "currency": "USD", "include": "1",
        },
        {
            "asset_id": "CRYPTO:BTC-USD", "row_id": "row-btc", "ticker": "BTC-USD",
            "yahoo_ticker": "BTC-USD", "currency": "USD", "include": "1",
        },
        {
            "asset_id": "EQH002", "row_id": "row-eur", "ticker": "EUR-USD",
            "yahoo_ticker": "EURUSD=X", "currency": "USD", "include": "1",
        },
        {
            "asset_id": "EQH003", "row_id": "row-excluded", "ticker": "EXCLUDED",
            "yahoo_ticker": "EXCLUDED", "currency": "USD", "include": "0",
        },
        {
            "asset_id": "EQH004", "row_id": "row-unknown", "ticker": "UNKNOWN",
            "yahoo_ticker": "UNKNOWN", "currency": "", "include": "1",
        },
    ]
    return {
        "universe.csv": _csv(rows),
        "universe_overrides.csv": _csv([
            {"target_row_id": "row-eur", "ticker": "EUR-USD", "yahoo_ticker": "EURUSD=X", "lock_yahoo_ticker": "1"}
        ]),
        "asset_excluded.csv": _csv([
            {"target_row_id": "row-excluded", "ticker": "EXCLUDED", "reason": "not in broker"}
        ]),
        "ingest_force_refresh.csv": _csv([{"ticker": "SPY"}]),
        # The historical manual source mismatch is not a reason to
        # substitute the Yahoo ticker already resolved by universe.csv.
        "universe_manual_additions.csv": _csv([
            {"ticker": "EUR-USD", "yahoo_ticker": "EUR-USD", "lock_yahoo_ticker": "1"}
        ]),
    }


@pytest.fixture
def s3_with_snapshot():
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
        )
        files = _sources()
        metadata = {
            key: {"sha256": bronze._digest(blob), "bytes": len(blob)}
            for key, blob in files.items()
        }
        fingerprint = bronze._digest(bronze._json_bytes(metadata))
        for filename, blob in files.items():
            s3.put_object(
                Bucket=BUCKET,
                Key=f"{bronze.SNAPSHOT_ROOT}/{fingerprint}/{filename}",
                Body=blob,
                Metadata={
                    "sha256": bronze._digest(blob),
                    "source-git-commit": "a" * 40,
                },
            )
        yield s3, fingerprint


def _fake_bars(record, batch_id, n=3):
    timestamps = pd.date_range("2026-11-10", periods=n, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "asset_id": record["asset_id"],
            "provider_symbol": record["yahoo_ticker"],
            "session_date": timestamps.strftime("%Y-%m-%d"),
            "source_timestamp": timestamps.astype(str),
            "source_timezone": "UTC",
            "currency_original": record.get("currency_hint") or "USD",
            "open": [100.0 + j for j in range(n)],
            "high": [102.0 + j for j in range(n)],
            "low": [99.0 + j for j in range(n)],
            "close": [101.0 + j for j in range(n)],
            "adj_close": [100.0 + j for j in range(n)],
            "volume": [10.0] * n,
            "dividends": [0.0] * n,
            "stock_splits": [0.0] * n,
            "ingestion_batch_id": batch_id,
        }
    )
    return frame


def test_load_current_universe_only_from_five_s3_files(s3_with_snapshot):
    s3, fingerprint = s3_with_snapshot
    assets, evidence = bronze.load_universe(s3, BUCKET, fingerprint)
    assert evidence["universe_total"] == 5
    assert evidence["included"] == 4
    assert evidence["excluded"] == 1
    assert len(assets) == 4
    assert not any(a["ticker"] == "EXCLUDED" for a in assets)
    assert next(a for a in assets if a["ticker"] == "EUR-USD")["yahoo_ticker"] == "EURUSD=X"
    assert evidence["configuration_warnings"][0]["ticker"] == "EUR-USD"
    assert next(a for a in assets if a["ticker"] == "UNKNOWN")["currency_hint"] is None


def test_source_fingerprint_and_provenance_are_mandatory(s3_with_snapshot):
    s3, fingerprint = s3_with_snapshot
    key = f"{bronze.SNAPSHOT_ROOT}/{fingerprint}/universe.csv"
    s3.put_object(Bucket=BUCKET, Key=key, Body=b"corrupted", Metadata={"sha256": "0" * 64})
    with pytest.raises(bronze.BronzeIngestError, match="checksum"):
        bronze.load_universe(s3, BUCKET, fingerprint)


def test_dry_run_has_no_s3_writes_and_counts_every_included_asset(s3_with_snapshot):
    s3, fingerprint = s3_with_snapshot
    result = bronze.ingest(s3, BUCKET, fingerprint=fingerprint, run_id="all-history-test",
                           end="2026-11-14", execute=False)
    assert result["selected"] == 4
    assert result["writes"] == 0
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix=bronze.BRONZE_ROOT).get("KeyCount") == 0


def test_yahoo_period_max_arguments_and_immutable_per_asset_manifest(s3_with_snapshot, monkeypatch):
    s3, fingerprint = s3_with_snapshot
    calls = []

    def acquire(asset_id, symbol, *, start, end, batch_id,
                ticker_factory, currency_hint, full_history):
        calls.append((asset_id, symbol, start, end, full_history, currency_hint))
        record = {"asset_id": asset_id, "yahoo_ticker": symbol, "currency_hint": currency_hint}
        return _fake_bars(record, batch_id), {"source": "Yahoo mock", "rows": 3}

    monkeypatch.setattr(bronze, "acquire_yahoo_bars", acquire)
    result = bronze.ingest(
        s3, BUCKET, fingerprint=fingerprint, run_id="pilot-history-test",
        end="2026-11-14", execute=True, only=["SPY", "EUR-USD"],
        retry_count=0, pause_seconds=0,
    )
    assert result["selected"] == result["succeeded"] == 2
    assert all(call[2] is None and call[4] is True for call in calls)
    assert sorted(call[1] for call in calls) == ["EURUSD=X", "SPY"]
    assert bronze.verify(
        s3, BUCKET, fingerprint=fingerprint, run_id="pilot-history-test"
    )["state"] == "pass"
    for aid in ("EQH001", "EQH002"):
        key = bronze._target_prefix("pilot-history-test", aid) + "/manifest.json"
        m = json.loads(bronze._read(s3, BUCKET, key))
        assert m["status"] == "committed"
        assert m["period_requested"] == "max"
        assert m["data"]["key"].startswith(bronze.BRONZE_ROOT)
        saved = pd.read_parquet(io.BytesIO(bronze._read(s3, BUCKET, m["data"]["key"])))
        assert saved["close"].tolist() == [101.0, 102.0, 103.0]


def test_same_run_resumes_without_refetching_successful_assets(s3_with_snapshot, monkeypatch):
    s3, fingerprint = s3_with_snapshot
    calls = []

    def acquire(asset_id, symbol, *, start, end, batch_id, **_kwargs):
        calls.append(symbol)
        return _fake_bars({"asset_id": asset_id, "yahoo_ticker": symbol}, batch_id), {}

    monkeypatch.setattr(bronze, "acquire_yahoo_bars", acquire)
    options = dict(
        fingerprint=fingerprint, run_id="resume-history-test", end="2026-11-14",
        execute=True, only=["SPY", "BTC-USD"], retry_count=0, pause_seconds=0,
    )
    assert bronze.ingest(s3, BUCKET, **options)["succeeded"] == 2
    second = bronze.ingest(s3, BUCKET, **options)
    assert second["already_committed"] == 2
    assert second["succeeded"] == 0
    assert calls == ["BTC-USD", "SPY"] or calls == ["SPY", "BTC-USD"]
    assert bronze.verify(s3, BUCKET, fingerprint=fingerprint, run_id=options["run_id"])["committed"] == 2


def test_partial_failures_are_recorded_and_retry_is_possible(s3_with_snapshot, monkeypatch):
    s3, fingerprint = s3_with_snapshot
    failures = {"BTC-USD"}

    def acquire(asset_id, symbol, *, batch_id, **_kwargs):
        if symbol in failures:
            raise RuntimeError("Yahoo transient outage")
        return _fake_bars({"asset_id": asset_id, "yahoo_ticker": symbol}, batch_id), {}

    monkeypatch.setattr(bronze, "acquire_yahoo_bars", acquire)
    options = dict(
        fingerprint=fingerprint, run_id="recover-history-test", end="2026-11-14",
        execute=True, only=["SPY", "BTC-USD"], retry_count=0, pause_seconds=0,
    )
    first = bronze.ingest(s3, BUCKET, **options)
    assert first["failed"] == 1
    assert first["succeeded"] == 1
    assert bronze.verify(s3, BUCKET, fingerprint=fingerprint, run_id=options["run_id"])["state"] == "incomplete"
    failures.clear()
    second = bronze.ingest(s3, BUCKET, **options)
    assert second["already_committed"] == 1
    assert second["succeeded"] == 1
    assert bronze.verify(s3, BUCKET, fingerprint=fingerprint, run_id=options["run_id"])["state"] == "pass"


def test_ambiguous_or_excluded_selection_never_silently_ingested(s3_with_snapshot):
    s3, fingerprint = s3_with_snapshot
    with pytest.raises(bronze.BronzeIngestError, match="Unknown or excluded"):
        bronze.ingest(s3, BUCKET, fingerprint=fingerprint, run_id="bad-excluded",
                      end="2026-11-14", only=["EXCLUDED"])
    with pytest.raises(bronze.BronzeIngestError, match="different selection"):
        bronze._write_run(
            s3, BUCKET, run_id="immutable-config", run_config={"run_id": "immutable-config"}, execute=True
        )
        bronze._write_run(
            s3, BUCKET, run_id="immutable-config", run_config={"run_id": "immutable-config", "extra": 1}, execute=True
        )


def test_wrong_source_commit_or_bad_source_checksum_blocks_bronze(s3_with_snapshot):
    s3, fingerprint = s3_with_snapshot
    filename = "universe_overrides.csv"
    key = f"{bronze.SNAPSHOT_ROOT}/{fingerprint}/{filename}"
    original = bronze._read(s3, BUCKET, key)
    s3.put_object(
        Bucket=BUCKET, Key=key, Body=original,
        Metadata={"sha256": bronze._digest(original), "source-git-commit": "b" * 40},
    )
    with pytest.raises(bronze.BronzeIngestError, match="one exact Git commit"):
        bronze.load_universe(s3, BUCKET, fingerprint)


def test_raw_bronze_keeps_invalid_adjusted_values_for_later_quality_review():
    record = {"asset_id": "EQH001", "yahoo_ticker": "SPY", "currency_hint": "USD"}
    bars = _fake_bars(record, "run")
    bars.loc[1, "adj_close"] = float("nan")
    bars.loc[2, "dividends"] = 0.75
    inspected = bronze._check_bars(bars, record)
    assert inspected["rows"] == 3
    assert inspected["missing_adjusted_close_rows"] == 1
    assert bars.loc[2, "dividends"] == 0.75
    assert pd.isna(bars.loc[1, "adj_close"])


def test_verify_detects_corrupt_source_object(s3_with_snapshot, monkeypatch):
    s3, fingerprint = s3_with_snapshot

    def acquire(asset_id, symbol, *, batch_id, **_kwargs):
        return _fake_bars({"asset_id": asset_id, "yahoo_ticker": symbol}, batch_id), {}

    monkeypatch.setattr(bronze, "acquire_yahoo_bars", acquire)
    options = dict(fingerprint=fingerprint, run_id="corrupt-history-test",
                   end="2026-11-14", execute=True, only=["SPY"],
                   retry_count=0, pause_seconds=0)
    bronze.ingest(s3, BUCKET, **options)
    manifest = json.loads(
        bronze._read(s3, BUCKET, bronze._target_prefix("corrupt-history-test", "EQH001") + "/manifest.json")
    )
    s3.put_object(Bucket=BUCKET, Key=manifest["data"]["key"], Body=b"corrupt")
    report = bronze.verify(s3, BUCKET, fingerprint=fingerprint, run_id="corrupt-history-test")
    assert report["corrupt"] == 1
    assert report["state"] == "incomplete"
