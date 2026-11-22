"""Silver v3 stage-1 contract tests with no live Yahoo or AWS writes."""

from __future__ import annotations

import io
import json

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from alpha_edge.market import build_silver_sessions_v3 as silver, ingest_bronze_full_history as bronze

BUCKET = "alpha-edge-algo"
BRONZE_RUN = "bronze-test-20261117"
SILVER_RUN = "silver-pilot-20261117"


def raw_frame(
    sessions: tuple[str, ...] = ("2026-10-05", "2026-10-06", "2026-10-07"),
    *,
    asset_id: str = "EQH-SPY",
    symbol: str = "SPY",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": [asset_id] * len(sessions),
            "provider_symbol": [symbol] * len(sessions),
            "session_date": list(sessions),
            "source_timestamp": [day + "T00:00:00+09:00" for day in sessions],
            "source_timezone": ["Asia/Tokyo"] * len(sessions),
            "currency_original": ["USD"] * len(sessions),
            "open": [100.0] * len(sessions),
            "high": [101.0] * len(sessions),
            "low": [99.0] * len(sessions),
            "close": [100.5] * len(sessions),
            "adj_close": [100.2] * len(sessions),
            "volume": [1000] * len(sessions),
            "dividends": [0.0] * len(sessions),
            "stock_splits": [0.0] * len(sessions),
        }
    )


def _csv(rows: list[dict]) -> bytes:
    import csv

    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


@pytest.fixture
def s3():
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-1")
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
        )
        yield s3


def setup_bronze(s3):
    rows = [
        {
            "asset_id": "EQH-SPY",
            "row_id": "one",
            "ticker": "SPY",
            "yahoo_ticker": "SPY",
            "include": "1",
            "currency": "USD",
        },
        {
            "asset_id": "EQH-HZN",
            "row_id": "two",
            "ticker": "BKH",
            "yahoo_ticker": "HZN",
            "include": "1",
            "currency": "USD",
        },
    ]
    source = {
        "universe.csv": _csv(rows),
        "universe_overrides.csv": _csv(
            [{"target_row_id": "two", "ticker": "BKH", "yahoo_ticker": "HZN", "lock_yahoo_ticker": "1"}]
        ),
        "asset_excluded.csv": _csv([{"target_row_id": "", "ticker": "OLD", "reason": "historical"}]),
        "ingest_force_refresh.csv": _csv([{"ticker": "SPY"}]),
        "universe_manual_additions.csv": _csv([{"ticker": "SPY"}]),
    }
    # No active-asset exclusion is implied by this support-file entry.
    entries = {name: {"bytes": len(blob), "sha256": bronze._digest(blob)} for name, blob in source.items()}
    fingerprint = bronze._digest(bronze._json_bytes(entries))
    for name, blob in source.items():
        s3.put_object(
            Bucket=BUCKET,
            Key=f"{bronze.SNAPSHOT_ROOT}/{fingerprint}/{name}",
            Body=blob,
            Metadata={"sha256": bronze._digest(blob), "source-git-commit": "a" * 40},
        )
    config = {
        "schema": "alpha_edge_full_history_run_v1",
        "run_id": BRONZE_RUN,
        "source_fingerprint": fingerprint,
        "requested_asset_ids": ["EQH-SPY", "EQH-HZN"],
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{bronze.BRONZE_ROOT}/runs/{BRONZE_RUN}/run.json",
        Body=bronze._json_bytes(config),
    )
    bars = raw_frame()
    buf = io.BytesIO()
    bars.to_parquet(buf, index=False)
    data = buf.getvalue()
    prefix = bronze._target_prefix(BRONZE_RUN, "EQH-SPY")
    data_key = f"{prefix}/objects/{bronze._digest(data)}.parquet"
    s3.put_object(Bucket=BUCKET, Key=data_key, Body=data, Metadata={"sha256": bronze._digest(data)})
    manifest = {
        "schema": "alpha_edge_full_history_asset_v1",
        "asset_id": "EQH-SPY",
        "provider_symbol": "SPY",
        "run_id": BRONZE_RUN,
        "status": "committed",
        "data": {"key": data_key, "bytes": len(data), "sha256": bronze._digest(data)},
        "validation": {"rows": len(bars)},
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{prefix}/manifest.json",
        Body=bronze._json_bytes(manifest),
    )
    return fingerprint, data_key


def test_calendar_reuses_existing_v2_audit_and_preserves_source_values():
    bars = raw_frame(("2026-10-05", "2026-10-07"))
    copy = bars.copy(deep=True)
    candidate, issues, report = silver.prepare_session_candidate(
        bars, asset_id="EQH-SPY", symbol="SPY", calendar_id="NYSE"
    )
    pd.testing.assert_frame_equal(bars, copy)
    assert candidate["exchange_session_date"].tolist() == ["2026-10-05", "2026-10-07"]
    assert candidate["source_timestamp_utc"].iloc[0] == "2026-10-04T15:00:00+00:00"
    assert candidate["source_timestamp_utc"].iloc[1] == "2026-10-06T15:00:00+00:00"
    assert report["missing_prior_expected_session_rows"] == 1
    assert report["calendar_issue_rows"] == 1
    assert "missing_expected_session" in set(issues["issue"])
    assert candidate["close"].tolist() == bars["close"].tolist()
    assert not report["production_eligible"]


def test_missing_calendar_is_review_not_silent_weekdays():
    frame, findings, report = silver.prepare_session_candidate(raw_frame(), asset_id="EQH-SPY", symbol="SPY")
    assert not report["calendar_mapped"]
    assert "calendar_mapping_not_configured" in set(findings["issue"])
    assert frame["calendar_session_status"].eq("not_checked").all()
    assert frame["previous_expected_session_missing"].isna().all()
    assert frame["session_quality_status"].eq("review_required").all()


def test_duplicates_and_malformed_dates_are_preserved_with_evidence():
    frame = raw_frame(("2026-10-05", "2026-10-05", "bad-date"))
    candidate, issues, report = silver.prepare_session_candidate(
        frame, asset_id="EQH-SPY", symbol="SPY", calendar_id="NYSE"
    )
    assert report["rows"] == 3
    assert report["duplicate_session_rows"] == 2
    assert report["invalid_session_dates"] == 1
    assert candidate["session_quality_status"].eq("quarantined").all()
    assert "duplicate_source_session" in set(issues["issue"])
    assert "invalid_session_date" in set(issues["issue"])
    assert candidate["calendar_session_status"].eq("blocked_invalid_source_dates").all()


def test_naive_source_timestamp_does_not_fabricate_utc():
    frame = raw_frame()
    frame["source_timestamp"] = ["2026-10-05", "2026-10-06", "2026-10-07"]
    candidate, issues, report = silver.prepare_session_candidate(
        frame, asset_id="EQH-SPY", symbol="SPY", calendar_id="NYSE"
    )
    assert candidate["source_timestamp_utc"].isna().all()
    assert report["missing_offset_aware_timestamps"] == 3
    assert "missing_offset_aware_timestamp" in set(issues["issue"])


def test_unusual_ohlc_is_flagged_not_rewritten():
    frame = raw_frame()
    frame.loc[1, "high"] = 1.0
    result, issues, report = silver.prepare_session_candidate(
        frame, asset_id="EQH-SPY", symbol="SPY", calendar_id="NYSE"
    )
    assert report["invalid_ohlc_rows"] == 1
    assert result.loc[1, "high"] == 1.0
    assert result.loc[1, "session_quality_status"] == "quarantined"
    assert "inconsistent_raw_ohlc" in set(issues["issue"])
    assert result["close_eligible"].tolist() == [True, False, True]
    assert result["daily_return_eligible"].tolist() == [False, False, False]


def test_expected_sessions_do_not_bridge_gaps_or_unexpected_bars():
    frame = raw_frame(("2026-10-02", "2026-10-03", "2026-10-05", "2026-10-07"))
    result, _, _ = silver.prepare_session_candidate(frame, asset_id="EQH-SPY", symbol="SPY", calendar_id="NYSE")
    assert result["close_eligible"].tolist() == [True, False, True, True]
    assert result["daily_return_eligible"].tolist() == [False, False, False, False]
    assert result["available_at_utc"].isna().all()


def test_calendar_policy_preserves_review_and_applies_approved_session():
    frame = raw_frame(("2026-10-02", "2026-10-03", "2026-10-05", "2026-10-07"))
    exception = {"calendar_id": "NYSE", "session_date": "2026-10-03", "decision": "review"}
    policy = {"exceptions": [exception]}
    result, _, _ = silver.prepare_session_candidate(
        frame, asset_id="EQH-SPY", symbol="SPY", calendar_id="NYSE", calendar_policy=policy
    )
    assert result.loc[1, "calendar_exception_review"]
    assert result.loc[1, "session_quality_status"] == "review_required"
    assert not result.loc[1, "close_eligible"]
    assert not result.loc[2, "daily_return_eligible"]
    exception["decision"] = "expected"
    result, _, _ = silver.prepare_session_candidate(
        frame, asset_id="EQH-SPY", symbol="SPY", calendar_id="NYSE", calendar_policy=policy
    )
    assert result["close_eligible"].tolist() == [True, True, True, True]
    assert result["daily_return_eligible"].tolist() == [False, True, True, False]
    assert pd.isna(result.loc[1, "scheduled_market_close_utc"])


@pytest.mark.parametrize("calendar", ["NYSE", "XETR", "XLON", "JPX"])
def test_exchange_placeholder_rule_is_calendar_independent(calendar):
    frame = raw_frame(("2026-10-02", "2026-10-03", "2026-10-05", "2026-10-06"))
    frame.loc[1, ["open", "high", "low", "close"]] = frame.loc[0, "close"]
    frame.loc[1, "volume"] = 0
    result, findings, report = silver.prepare_session_candidate(
        frame, asset_id="EQH-SPY", symbol="SPY", calendar_id=calendar
    )
    assert result["non_trading_placeholder"].tolist() == [False, True, False, False]
    assert result["daily_return_eligible"].tolist() == [False, False, True, True]
    assert report["non_trading_placeholder_rows"] == 1
    assert "non_trading_placeholder" in set(findings["issue"])
    pd.testing.assert_series_equal(result["close"], frame["close"])
    frame.loc[1, "volume"] = 1
    result, _, _ = silver.prepare_session_candidate(frame, asset_id="EQH-SPY", symbol="SPY", calendar_id=calendar)
    assert result.loc[1, "calendar_exception_review"]
    assert not result.loc[2, "daily_return_eligible"]


@pytest.mark.parametrize("calendar,weekend_expected", [("YAHOO_FX_WEEKDAYS", False), ("CRYPTO_24_7", True)])
def test_fx_and_crypto_zero_volume_is_not_exchange_placeholder(calendar, weekend_expected):
    frame = raw_frame(("2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05"))
    frame.loc[1, ["open", "high", "low", "close"]] = frame.loc[0, "close"]
    frame.loc[1, "volume"] = 0
    result, _, _ = silver.prepare_session_candidate(frame, asset_id="EQH-SPY", symbol="SPY", calendar_id=calendar)
    assert not result["non_trading_placeholder"].any()
    assert bool(result.loc[1, "close_eligible"]) == weekend_expected
    assert bool(result.loc[1, "calendar_exception_review"]) != weekend_expected


def test_close_correction_retains_original_and_restores_candidate_eligibility():
    frame = raw_frame()
    frame.loc[1, "close"] = 200.0
    correction = {
        "asset_id": "EQH-SPY",
        "provider_symbol": "SPY",
        "currency_original": "USD",
        "session_date": "2026-10-06",
        "original_close": 200.0,
        "corrected_close": float(frame.loc[0, "close"]),
        "correction_id": "test-correction",
    }
    result, _, _ = silver.prepare_session_candidate(
        frame, asset_id="EQH-SPY", symbol="SPY", calendar_id="NYSE", close_correction=correction
    )
    assert result.loc[1, "close"] == 200.0
    assert result.loc[1, "effective_close"] == frame.loc[0, "close"]
    assert result.loc[1, "close_correction_id"] == "test-correction"
    assert result["daily_return_eligible"].tolist() == [False, True, True]
    correction.update(kind="close_corroboration", corrected_close=200.0)
    result, _, _ = silver.prepare_session_candidate(
        frame, asset_id="EQH-SPY", symbol="SPY", calendar_id="NYSE", close_correction=correction
    )
    assert result.loc[1, "close_eligible"]
    assert not result.loc[1, "ohlc_eligible"]


def test_pinned_bronze_s3_dry_run_and_candidate_publication(s3):
    fingerprint, data_key = setup_bronze(s3)
    kwargs = {
        "source_fingerprint": fingerprint,
        "bronze_run_id": BRONZE_RUN,
        "silver_run_id": SILVER_RUN,
        "only": ["SPY"],
        "calendar_id": "NYSE",
    }
    dry = silver.build(s3, BUCKET, **kwargs)
    assert dry["processed"] == 1
    assert dry["missing_bronze"] == 0
    assert dry["assets"][0]["source"] == data_key
    assert s3.list_objects_v2(Bucket=BUCKET, Prefix=silver.SILVER_ROOT).get("KeyCount", 0) == 0

    committed = silver.build(s3, BUCKET, **kwargs, execute=True)
    assert committed["processed"] == 1
    assert committed["assets"][0]["status"] == "candidate_written"
    assert not committed["production_eligible"]
    same = silver.build(s3, BUCKET, **kwargs, execute=True)
    assert same["assets"][0]["status"] == "already_committed"
    assert silver.verify_candidate_run(s3, BUCKET, SILVER_RUN)["state"] == "pass"


def test_hzn_missing_baseline_is_a_reported_exception_not_fabricated(s3):
    fingerprint, _ = setup_bronze(s3)
    report = silver.build(
        s3,
        BUCKET,
        source_fingerprint=fingerprint,
        bronze_run_id=BRONZE_RUN,
        silver_run_id=SILVER_RUN,
        only=["BKH"],
        calendar_id="NYSE",
    )
    assert report["missing_bronze"] == 1
    assert report["assets"][0]["status"] == "source_unavailable"
    assert report["processed"] == 0


def test_bronze_provenance_mismatch_blocks_silver(s3):
    fingerprint, _ = setup_bronze(s3)
    run_key = f"{bronze.BRONZE_ROOT}/runs/{BRONZE_RUN}/run.json"
    cfg = json.loads(s3.get_object(Bucket=BUCKET, Key=run_key)["Body"].read())
    cfg["source_fingerprint"] = "b" * 64
    s3.put_object(Bucket=BUCKET, Key=run_key, Body=bronze._json_bytes(cfg))
    with pytest.raises(silver.SilverSessionError, match="differs"):
        silver.build(
            s3,
            BUCKET,
            source_fingerprint=fingerprint,
            bronze_run_id=BRONZE_RUN,
            silver_run_id=SILVER_RUN,
            only=["SPY"],
            calendar_id="NYSE",
        )


def test_full_universe_reuses_compatible_fallback_and_reports_missing(s3, monkeypatch):
    historical_fingerprint, _ = setup_bronze(s3)
    current_fingerprint = "b" * 64
    records, metadata = bronze.load_universe(s3, BUCKET, historical_fingerprint)
    primary_run = "bronze-new-snapshot"
    config = {
        "source_fingerprint": current_fingerprint,
        "requested_asset_ids": ["EQH-HZN"],
        "end_exclusive": "2026-10-09",
    }
    s3.put_object(
        Bucket=BUCKET, Key=f"{bronze.BRONZE_ROOT}/runs/{primary_run}/run.json", Body=bronze._json_bytes(config)
    )
    monkeypatch.setattr(silver, "load_universe", lambda *args: (records, metadata))
    monkeypatch.setattr(
        silver,
        "load_registry",
        lambda *args: {
            "assets": [
                {
                    "asset_id": a["asset_id"],
                    "provider_symbol": a["yahoo_ticker"],
                    "calendar_id": "NYSE",
                    "mapping_status": "proposed_review_required",
                }
                for a in records
            ]
        },
    )
    result = silver.build(
        s3,
        BUCKET,
        source_fingerprint=current_fingerprint,
        bronze_run_id=primary_run,
        silver_run_id="silver-full-test",
        only=[],
        all_assets=True,
        registry_version="c" * 64,
        fallback_bronze_run_ids=[BRONZE_RUN],
        execute=True,
    )
    assert result["selected"] == 2
    assert result["processed"] == 1
    assert result["missing_bronze"] == 1
    assert result["summary"]["quality_totals"]["missing_endpoint_sessions"] == 1
    spy = next(row for row in result["assets"] if row["ticker"] == "SPY")
    assert spy["bronze_run_id"] == BRONZE_RUN
    manifest_key = f"{silver.SILVER_ROOT}/runs/silver-full-test/assets/{bronze._digest(b'EQH-SPY')}/manifest.json"
    manifest = json.loads(s3.get_object(Bucket=BUCKET, Key=manifest_key)["Body"].read())
    assert manifest["bronze_source_fingerprint"] == historical_fingerprint
    assert result["production_eligible"] is False
    verification = silver.verify_candidate_run(s3, BUCKET, "silver-full-test")
    assert verification["verified"] == 1
    assert verification["missing"] == 1


def test_endpoint_completeness_distinguishes_weekend_from_missing_session():
    frame = raw_frame(("2026-10-01", "2026-10-02"))
    _, findings, report = silver.prepare_session_candidate(
        frame, asset_id="EQH-SPY", symbol="SPY", calendar_id="NYSE", end_exclusive="2026-10-06"
    )
    assert report["missing_endpoint_sessions"] == 1
    assert findings.loc[findings["issue"].eq("missing_endpoint_session"), "session_date"].tolist() == ["2026-10-05"]


def test_effective_exchange_transfer_gap_and_early_close():
    from alpha_edge.market.effective_exchange_calendars import TXSE_POLICY, transfer_history

    aid = "EQH272fff4855629062"
    history = transfer_history(aid, "DDS")
    frame = raw_frame(("2026-10-02", "2026-10-06"), asset_id=aid, symbol="DDS")
    candidate, issues, quality = silver.prepare_session_candidate(
        frame, asset_id=aid, symbol="DDS", calendar_id="TXSE", calendar_history=history, txse_policy=TXSE_POLICY
    )
    assert candidate["effective_calendar_id"].tolist() == ["NYSE", "TXSE"]
    assert not candidate["daily_return_eligible"].any()
    assert issues.loc[issues["issue"].eq("missing_expected_session"), "session_date"].tolist() == ["2026-10-05"]
    assert quality["source_rows_preserved"]
    frame = raw_frame(("2026-11-25", "2026-11-27"), asset_id=aid, symbol="DDS")
    candidate, _, _ = silver.prepare_session_candidate(
        frame, asset_id=aid, symbol="DDS", calendar_id="TXSE", calendar_history=history, txse_policy=TXSE_POLICY
    )
    assert candidate["daily_return_eligible"].tolist() == [False, True]
    assert candidate["scheduled_market_close_utc"].iloc[-1].startswith("2026-11-27T18:00")
