"""No real Yahoo/AWS. Prove incremental Bronze reads S3 state and NEVER asks period=max."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market import ingest_bronze_full_history as full
from alpha_edge.market import ingest_bronze_incremental as inc
from alpha_edge.market.ingest_bronze_full_history import BronzeIngestError


@pytest.fixture(autouse=True)
def frozen_utc_clock(monkeypatch):
    """Future end-exclusive test dates must not depend on CI's actual date."""
    class TestClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 20, tzinfo=timezone.utc).astimezone(tz or timezone.utc)

    monkeypatch.setattr(inc, "datetime", TestClock)


BUCKET = "alpha-edge-algo"
FINGERPRINT = "a" * 64
BASE_RUN = "bronze-base-20261117"
DAILY_RUN = "bronze-day-20261118"


def _raw_csv(records):
    import csv

    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def s3():
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"})
        yield s3


@pytest.fixture
def prepared(s3):
    rows = [
        {"asset_id": "EQH-SPY", "row_id": "id-spy", "ticker": "SPY", "yahoo_ticker": "SPY", "currency": "USD", "include": "1"},
        {"asset_id": "EQH-HZN", "row_id": "id-hzn", "ticker": "BKH", "yahoo_ticker": "HZN", "currency": "USD", "include": "1"},
        {"asset_id": "EQH-OLD", "row_id": "id-old", "ticker": "OLD", "yahoo_ticker": "OLD", "currency": "USD", "include": "0"},
    ]
    sources = {
        "universe.csv": _raw_csv(rows),
        "universe_overrides.csv": _raw_csv([{"target_row_id": "id-hzn", "ticker": "BKH", "yahoo_ticker": "HZN", "lock_yahoo_ticker": "1"}]),
        "asset_excluded.csv": _raw_csv([{"target_row_id": "id-old", "ticker": "OLD", "reason": "excluded"}]),
        "ingest_force_refresh.csv": _raw_csv([{"ticker": "SPY"}]),
        "universe_manual_additions.csv": _raw_csv([{"ticker": "SPY"}]),
    }
    metadata = {k: {"bytes": len(v), "sha256": full._digest(v)} for k, v in sources.items()}
    fingerprint = full._digest(full._json_bytes(metadata))
    for filename, blob in sources.items():
        s3.put_object(Bucket=BUCKET, Key=f"{full.SNAPSHOT_ROOT}/{fingerprint}/{filename}", Body=blob,
                      Metadata={"sha256": full._digest(blob), "source-git-commit": "a" * 40})
    return s3, fingerprint, rows


def _bars(asset_id, symbol, batch_id, dates):
    timestamps = pd.to_datetime(dates, utc=True)
    return pd.DataFrame({
        "asset_id": [asset_id] * len(dates),
        "provider_symbol": [symbol] * len(dates),
        "session_date": list(dates),
        "source_timestamp": timestamps.astype(str),
        "source_timezone": ["UTC"] * len(dates),
        "currency_original": ["USD"] * len(dates),
        "open": [101.0] * len(dates),
        "high": [102.0] * len(dates),
        "low": [99.0] * len(dates),
        "close": [101.0] * len(dates),
        "adj_close": [100.0] * len(dates),
        "volume": [1.0] * len(dates),
        "dividends": [0.0] * len(dates),
        "stock_splits": [0.0] * len(dates),
        "ingestion_batch_id": [batch_id] * len(dates),
    })


def _baseline(s3, fingerprint, monkeypatch, *, assets=("SPY", "BKH")):
    def fake_history(asset_id, symbol, *, full_history, end, start, batch_id, **_):
        assert full_history is True
        assert start is None
        return _bars(asset_id, symbol, batch_id, ["2026-11-12", "2026-11-13", "2026-11-16"]), {}
    monkeypatch.setattr(full, "acquire_yahoo_bars", fake_history)
    result = full.ingest(
        s3, BUCKET, fingerprint=fingerprint, run_id=BASE_RUN,
        end="2026-11-17", execute=True, only=list(assets),
        retry_count=0, pause_seconds=0,
    )
    assert result["failed"] == 0


def test_incremental_acquires_only_missing_dates_plus_overlap(prepared, monkeypatch, capsys):
    s3, fingerprint, rows = prepared
    _baseline(s3, fingerprint, monkeypatch)
    calls = []

    def fetch(asset_id, symbol, *, start, end, full_history, batch_id, **_kwargs):
        calls.append((asset_id, symbol, start, end, full_history))
        assert full_history is False
        assert start is not None
        assert start == "2026-11-11"
        assert end == "2026-11-18"
        return _bars(asset_id, symbol, batch_id, ["2026-11-12", "2026-11-13", "2026-11-16", "2026-11-17"]), {
            "start": start, "full_history_requested": False,
        }

    monkeypatch.setattr(inc, "acquire_yahoo_bars", fetch)
    result = inc.ingest(
        s3, BUCKET, fingerprint=fingerprint, run_id=DAILY_RUN,
        parent_run_id=BASE_RUN, end="2026-11-18",
        overlap_days=5, max_staleness_days=60,
        execute=True, only=["SPY", "BKH"], retry_count=0,
        pause_seconds=0, progress_every=1,
    )
    assert result["succeeded"] == 2
    assert result["failed"] == 0
    assert len(calls) == 2
    assert all(call[4] is False for call in calls)
    assert inc.verify(s3, BUCKET, fingerprint=fingerprint, run_id=DAILY_RUN)["state"] == "pass"
    assert "date-window ONLY" in capsys.readouterr().out
    sample = next(a for a in rows if a["ticker"] == "SPY")
    manifest = inc._incremental_commit(s3, BUCKET, DAILY_RUN, {
        "asset_id": sample["asset_id"], "yahoo_ticker": "SPY",
    }, verify_hash=True)
    assert manifest["start_inclusive"] == "2026-11-11"
    assert manifest["baseline_run_id"] == BASE_RUN
    assert manifest["validation"]["new_rows_since_baseline"] == 1
    assert manifest["validation"]["overlap_rows"] == 3


def test_resume_skips_committed_assets_without_yahoo_call(prepared, monkeypatch):
    s3, fingerprint, _ = prepared
    _baseline(s3, fingerprint, monkeypatch, assets=("SPY",))
    calls = []

    def fetch(asset_id, symbol, *, batch_id, **_kwargs):
        calls.append(symbol)
        return _bars(asset_id, symbol, batch_id, ["2026-11-16", "2026-11-17"]), {}

    monkeypatch.setattr(inc, "acquire_yahoo_bars", fetch)
    kwargs = dict(fingerprint=fingerprint, run_id=DAILY_RUN, parent_run_id=BASE_RUN,
                  end="2026-11-18", only=["SPY"], execute=True, retry_count=0, pause_seconds=0)
    first = inc.ingest(s3, BUCKET, **kwargs)
    second = inc.ingest(s3, BUCKET, **kwargs)
    assert first["succeeded"] == 1
    assert second["succeeded"] == 0
    assert second["already_committed"] == 1
    assert calls == ["SPY"]


def test_missing_parent_history_fails_without_falling_back_to_full_history(prepared, monkeypatch):
    s3, fingerprint, _ = prepared
    _baseline(s3, fingerprint, monkeypatch, assets=("SPY",))
    calls = []
    monkeypatch.setattr(inc, "acquire_yahoo_bars", lambda *a, **k: calls.append(a))
    result = inc.ingest(
        s3, BUCKET, fingerprint=fingerprint, run_id=DAILY_RUN,
        parent_run_id=BASE_RUN, end="2026-11-18", only=["BKH"],
        execute=True, pause_seconds=0, retry_count=0,
    )
    assert result["failed"] == 1
    assert "No baseline" in result["failure_details"][0]["error"]
    assert calls == []


def test_next_day_finds_last_committed_delta_in_parent_lineage(prepared, monkeypatch):
    s3, fingerprint, _ = prepared
    _baseline(s3, fingerprint, monkeypatch, assets=("SPY",))
    def fetch(asset_id, symbol, *, batch_id, start, end, **_kwargs):
        assert start is not None
        if batch_id == DAILY_RUN:
            assert start == "2026-11-06"
            dates = ["2026-11-13", "2026-11-16", "2026-11-17"]
        else:
            assert start == "2026-11-07"
            dates = ["2026-11-16", "2026-11-17", "2026-11-18"]
        return _bars(asset_id, symbol, batch_id, dates), {}
    monkeypatch.setattr(inc, "acquire_yahoo_bars", fetch)
    first = inc.ingest(
        s3, BUCKET, fingerprint=fingerprint, run_id=DAILY_RUN,
        parent_run_id=BASE_RUN, end="2026-11-18", only=["SPY"],
        overlap_days=10, execute=True, retry_count=0, pause_seconds=0,
    )
    assert first["failed"] == 0
    second = inc.ingest(
        s3, BUCKET, fingerprint=fingerprint, run_id="bronze-day-20261119",
        parent_run_id=DAILY_RUN, end="2026-11-19", only=["SPY"],
        overlap_days=10, execute=True, retry_count=0, pause_seconds=0,
    )
    assert second["failed"] == 0


def test_stale_baseline_refuses_unbounded_query(prepared, monkeypatch):
    s3, fingerprint, _ = prepared
    _baseline(s3, fingerprint, monkeypatch, assets=("SPY",))
    called = []
    monkeypatch.setattr(inc, "acquire_yahoo_bars", lambda *a, **k: called.append(a))
    result = inc.ingest(
        s3, BUCKET, fingerprint=fingerprint, run_id="bronze-stale-run",
        parent_run_id=BASE_RUN, end="2026-11-30", only=["SPY"],
        max_staleness_days=10, overlap_days=5,
        execute=True, retry_count=0, pause_seconds=0,
    )
    assert result["failed"] == 1
    assert "Stale baseline" in result["failure_details"][0]["error"]
    assert called == []


def test_no_historical_period_parameter_on_any_incremental_request(prepared, monkeypatch):
    s3, fingerprint, _ = prepared
    _baseline(s3, fingerprint, monkeypatch, assets=("SPY",))
    def spy(*args, **kwargs):
        assert kwargs.get("full_history") is False
        assert kwargs.get("start") is not None
        assert "period" not in kwargs
        return _bars(args[0], args[1], kwargs["batch_id"], ["2026-11-16", "2026-11-17"]), {}
    monkeypatch.setattr(inc, "acquire_yahoo_bars", spy)
    result = inc.ingest(
        s3, BUCKET, fingerprint=fingerprint, run_id=DAILY_RUN, parent_run_id=BASE_RUN,
        end="2026-11-18", only=["SPY"], execute=True, retry_count=0, pause_seconds=0,
    )
    assert result["succeeded"] == 1


def test_incremental_run_config_is_immutable(prepared, monkeypatch):
    s3, fingerprint, _ = prepared
    _baseline(s3, fingerprint, monkeypatch, assets=("SPY",))
    kwargs = dict(fingerprint=fingerprint, run_id=DAILY_RUN, parent_run_id=BASE_RUN,
                  end="2026-11-18", only=["SPY"], execute=True, retry_count=0, pause_seconds=0)
    monkeypatch.setattr(
        inc, "acquire_yahoo_bars",
        lambda asset_id, symbol, *, batch_id, **_: (_bars(asset_id, symbol, batch_id, ["2026-11-16", "2026-11-17"]), {}),
    )
    assert inc.ingest(s3, BUCKET, **kwargs)["failed"] == 0
    with pytest.raises(BronzeIngestError, match="different immutable configuration"):
        inc.ingest(s3, BUCKET, **{**kwargs, "overlap_days": 4})


def test_older_parent_full_history_symbol_collision_fails_closed(prepared, monkeypatch):
    s3, fingerprint, _ = prepared
    _baseline(s3, fingerprint, monkeypatch, assets=("BKH",))
    wrong = {"asset_id": "EQH-HZN", "yahoo_ticker": "OLD_SYMBOL"}
    with pytest.raises(BronzeIngestError, match="Incompatible|incompatible"):
        inc._find_parent(s3, BUCKET, BASE_RUN, wrong)


def test_bounded_request_respects_end_exclusive():
    prior = {"validation": {"last_session": "2026-11-16"}}
    start, baseline = inc._window(prior, end="2026-11-18", overlap_days=7, max_staleness_days=60)
    assert start == "2026-11-09"
    assert baseline == "2026-11-16"
    with pytest.raises(BronzeIngestError, match="not before"):
        inc._window(prior, end="2026-11-16", overlap_days=7, max_staleness_days=60)
