"""Regression tests for the non-production v2 recovery migration stage."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

from alpha_edge.market import run_v2_migration_stage as stage
from alpha_edge.market.standardize_bronze_v2 import audit_asset_sessions


def test_fx_daily_calendar_does_not_expect_sunday(monkeypatch):
    class Forex:
        def valid_days(self, start_date, end_date):
            # Original FOREX calendar includes Sundays.
            return pd.date_range(start_date, end_date, freq="D")

    fake = types.SimpleNamespace(get_calendar=lambda identifier: Forex())
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", fake)
    bars = pd.DataFrame([
        {"asset_id": "FX:A", "provider_symbol": "EURUSD=X", "session_date": "2026-11-06"},
        {"asset_id": "FX:A", "provider_symbol": "EURUSD=X", "session_date": "2026-11-09"},
    ])
    flags, missing = audit_asset_sessions(bars, calendars={"EURUSD=X": "FOREX"})
    assert missing.empty
    assert flags["is_expected_session"].all()


def test_fx_daily_calendar_still_reports_missing_weekday(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "pandas_market_calendars",
        types.SimpleNamespace(get_calendar=lambda identifier: object()),
    )
    bars = pd.DataFrame([
        {"asset_id": "FX:A", "provider_symbol": "EURUSD=X", "session_date": "2026-11-06"},
        {"asset_id": "FX:A", "provider_symbol": "EURUSD=X", "session_date": "2026-11-10"},
    ])
    _, missing = audit_asset_sessions(bars, calendars={"EURUSD=X": "FOREX"})
    assert missing["issue"].tolist() == ["missing_expected_session"]
    assert missing["session_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-09"]


def test_stage_executes_five_independent_recovery_selections(monkeypatch):
    calls = []

    def recover(s3, bucket, archive_id, **kwargs):
        calls.append(kwargs["start_index"])
        return {"results": [
            {"asset_id": f"asset-{kwargs['start_index']}-{i}", "status": "candidate_committed"}
            for i in range(kwargs["max_assets"])
        ]}

    monkeypatch.setattr(stage, "recover_candidates", recover)
    monkeypatch.setattr(
        stage, "reprocess_committed",
        lambda *args, **kwargs: {"status": "reprocessed_candidate_committed"},
    )
    monkeypatch.setattr(
        stage, "retain_legacy",
        lambda *args, **kwargs: {"results": [
            {"asset_id": "excluded", "status": "historical_bytes_retained_in_v2"}
        ]},
    )
    monkeypatch.setattr(
        stage, "release_inventory",
        lambda *args, **kwargs: {"status": "published_blocked_release_inventory",
                                "production_eligible": False},
    )
    output = stage.run(
        object(), "bucket", universe_csv=Path("universe.csv"),
        fx_config=Path("fx.csv"), execute=True,
    )
    assert calls == [0, 25, 50, 75, 100]
    assert output["recovered_assets"]["outcomes"] == {"candidate_committed": 125}
    assert output["production_pointer_changed"] is False
    assert output["legacy_v1_modified"] is False
    assert output["yahoo_downloads"] == 0
    assert output["production_eligible"] is False
