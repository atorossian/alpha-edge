"""End-to-end reviewed history acquisition reuses existing preflight/backfill."""

import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_edge.market import acquire_reviewed_history_v2 as acquisition


@pytest.fixture
def inputs(tmp_path):
    universe = tmp_path / "universe.csv"
    pd.DataFrame([
        {
            "asset_id": "EQH4a3a0282d0f4690f",
            "row_id": "C73EB54227724B75",
            "yahoo_ticker": "KMB",
            "include": 1,
            "currency": "USD",
            "exchange": "NMS",
            "calendar_id": "",
        },
        {
            "asset_id": "EQH1e4a2ab4c61b0284",
            "row_id": "DEA19FC8D266CFE2",
            "yahoo_ticker": "KMB",
            "include": 0,
            "currency": "USD",
            "exchange": "NMS",
            "calendar_id": "",
        },
    ]).to_csv(universe, index=False)
    exchange_map = tmp_path / "exchange.csv"
    pd.DataFrame([{"exchange": "NMS", "calendar_id": "NYSE"}]).to_csv(exchange_map, index=False)
    fx = tmp_path / "fx.csv"
    pd.DataFrame(
        [{"currency": "EUR", "provider_symbol": "EURUSD=X", "direction": "direct"}]
    ).to_csv(fx, index=False)
    return universe, exchange_map, fx


def test_selects_retained_kmb_and_explicit_calendar(inputs):
    universe, exchange, _ = inputs
    asset, source = acquisition.reviewed_asset(
        universe, exchange, asset_id="EQH4a3a0282d0f4690f", expected_symbol="KMB"
    )
    assert asset.iloc[0]["calendar_id"] == "NYSE"
    assert asset.iloc[0]["currency"] == "USD"
    assert source["row_id"] == "C73EB54227724B75"


def test_rejects_duplicate_active_kmb(inputs):
    universe, exchange, _ = inputs
    rows = pd.read_csv(universe)
    rows.loc[rows["asset_id"].eq("EQH1e4a2ab4c61b0284"), "include"] = 1
    rows.to_csv(universe, index=False)
    with pytest.raises(acquisition.HistoricalAcquisitionError, match="Another included"):
        acquisition.reviewed_asset(
            universe, exchange, asset_id="EQH4a3a0282d0f4690f", expected_symbol="KMB"
        )


def test_dry_run_does_not_execute_or_write_s3(monkeypatch, inputs):
    universe, exchange, fx = inputs

    def forbidden(**kwargs):
        raise AssertionError("Backfill must never run during planning")

    monkeypatch.setattr(acquisition, "run_backfill", forbidden)
    result = acquisition.run(
        s3=object(), bucket="bucket", universe_csv=universe, exchange_map_csv=exchange,
        fx_csv=fx, asset_id="EQH4a3a0282d0f4690f", expected_symbol="KMB",
        end="2026-11-10", minimum_expected_first_date="1985-01-31",
    )
    assert result["s3_writes"] == 0
    assert result["status"] == "ready_for_full_history_acquisition"
    assert result["preflight"] == "configuration_pass_not_data_acceptance"


def test_existing_pipeline_execution_reports_history_shortfall(monkeypatch, inputs):
    universe, exchange, fx = inputs
    seen = {}

    def execute_backfill(**kwargs):
        seen["plan_path"] = kwargs["plan_dir"]
        assert (kwargs["plan_dir"] / "cohort_0001.csv").is_file()
        assert kwargs["execute"] is True
        assert kwargs["max_cohorts"] == 1
        return {"s3_verified_cohorts": 1}

    monkeypatch.setattr(acquisition, "run_backfill", execute_backfill)
    monkeypatch.setattr(acquisition, "verify_commit", lambda *args: {"files": []})

    class S3:
        def get_object(self, Bucket, Key):
            import io

            manifest = {
                "sources": [{
                    "asset_id": "EQH4a3a0282d0f4690f",
                    "provider_symbol": "KMB",
                    "first_observed_session": "2010-01-04",
                    "last_observed_session": "2026-11-09",
                    "row_count": 4100,
                }]
            }
            return {"Body": io.BytesIO(json.dumps(manifest).encode())}

    result = acquisition.run(
        s3=S3(), bucket="bucket", universe_csv=universe, exchange_map_csv=exchange,
        fx_csv=fx, asset_id="EQH4a3a0282d0f4690f", expected_symbol="KMB",
        end="2026-11-10", minimum_expected_first_date="1985-01-31", execute=True,
    )
    assert result["status"] == "published_research_candidate_with_coverage_shortfall"
    assert result["earlier_history_shortfall"] is True
    assert result["first_observed_session"] == "2010-01-04"
