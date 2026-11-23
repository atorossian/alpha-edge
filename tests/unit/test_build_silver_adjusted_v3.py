"""Candidate integration preserves raw evidence and never releases quarantine."""

import numpy as np
import pandas as pd
import pytest

from alpha_edge.market import (
    build_silver_adjusted_v3 as adjusted,
    dividend_currency_registry_v3 as currency,
    dividend_duplicate_registry_v3 as duplicates,
    historical_quarantine_v3 as quarantine,
)


def fixture():
    dates = ["2025-03-31", "2025-04-01", "2025-04-02", "2025-04-04"]
    bars = pd.DataFrame(
        {
            "asset_id": "a",
            "provider_symbol": "A",
            "currency_original": "MXN",
            "session_date": dates,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": 1000,
            "dividends": [0.0, 2.0, 0.0, 0.0],
            "stock_splits": 0.0,
            "adj_close": [98.0, 101.0, 102.0, 103.0],
        }
    )
    cal = pd.DataFrame(
        {
            "session_date": dates,
            "is_expected_session": True,
            "ohlc_eligible": True,
            "calendar_exception_review": False,
            "close_correction_id": "",
            "effective_close": bars.close,
        }
    )
    findings = pd.DataFrame({"issue": ["missing_expected_session"], "session_date": ["2025-04-03"]})
    bundle = {
        "currency": {
            "schema": currency.SCHEMA,
            "entries": [],
            "units_per_eur": currency.UNITS_PER_EUR,
            "rate_reference": currency.RATE_REFERENCE,
        },
        "duplicates": {"schema": duplicates.SCHEMA, "entries": []},
        "quarantine": {"schema": quarantine.SCHEMA, "entries": []},
        "share": {"asset_id": "other"},
        "reconciliation": {"entries": [], "additional_currency_candidates": [], "event_reconstruction_candidates": []},
        "events": [],
        "suppressions": [],
        "missing": {"entries": []},
        "complex": {"entries": []},
        "spin": {"asset_id": "other"},
    }
    return bars, cal, findings, bundle


def run(bars, cal, findings, bundle):
    return adjusted.reconstruct(bars, cal, findings, bundle, asset_id="a", symbol="A", source_key="key", sha="a" * 64)


def pin():
    return {"asset_id": "a", "provider_symbol": "A", "source_key": "key", "source_sha256": "a" * 64}


def test_rebuild_with_cash_adjustment_and_missing_session():
    bars, cal, findings, bundle = fixture()
    original = bars.copy(deep=True)
    bundle["missing"]["entries"] = [
        {
            **pin(),
            "source_dividend_date": "2025-04-02",
            "source_cash": 0.0,
            "session_date": "2025-04-03",
            "prior_observation": bars.iloc[2].to_dict(),
            "next_observation": bars.iloc[3].to_dict(),
            "currency_original": "MXN",
            "open": 102.0,
            "high": 103.0,
            "low": 101.0,
            "close": 102.5,
            "volume_estimate": 1000,
            "candidate_cash": 1.0,
        }
    ]
    f, seen = run(bars, cal, findings, bundle)
    pd.testing.assert_frame_equal(bars, original)
    assert len(f) == 5 and seen == {("a", "2025-04-02")}
    inserted = f.loc[f.session_date.eq("2025-04-03")].iloc[0]
    assert pd.isna(inserted.close) and inserted.candidate_volume_estimated
    assert inserted.candidate_total_return == pytest.approx(103.5 / 102 - 1)
    assert f.candidate_adjusted_close.notna().all()
    assert not f.production_eligible.any()


def test_unknown_spin_close_breaks_actual_chain_not_reference_return():
    bars, cal, findings, bundle = fixture()
    cash_date = "2025-04-01"
    bundle["spin"] = {
        **pin(),
        "rows": [],
        "cash_event": {"ex_date": cash_date, "source_cash": 2.0, "normalized_cash": 1.0},
        "spin_off_event": {"effective_date": "2025-04-02", "reference_value": 3.0, "actual_distribution_close": None},
    }
    f, seen = run(bars, cal, findings, bundle)
    assert seen == {("a", cash_date)}
    assert f.loc[1, "candidate_total_return"] == pytest.approx(102 / 100 - 1)
    assert np.isnan(f.loc[2, "candidate_total_return"])
    assert f.loc[2, "candidate_reference_return"] == pytest.approx(105 / 101 - 1)
    assert np.isnan(f.loc[3, "candidate_total_return"])
    assert np.isnan(f.loc[1, "candidate_adjusted_close"])
    assert f.loc[0, "candidate_segment_id"] == f.loc[1, "candidate_segment_id"]
    assert f.loc[1, "candidate_segment_id"] != f.loc[2, "candidate_segment_id"]
    assert f.loc[1, "candidate_segment_adjusted_close"] == 101


def test_source_cash_mismatch_fails_closed():
    bars, cal, findings, bundle = fixture()
    bundle["reconciliation"]["entries"] = [
        {**pin(), "effective_date": "2025-04-01", "source_cash": 99, "verified_candidate_cash": 1}
    ]
    with pytest.raises(ValueError, match="Source dividend changed"):
        run(bars, cal, findings, bundle)


def test_cash_and_factor_recomputed_and_source_preserved():
    bars, cal, findings, bundle = fixture()
    bundle["reconciliation"]["entries"] = [
        {
            **pin(),
            "effective_date": "2025-04-01",
            "price_currency": "MXN",
            "source_cash": 2.0,
            "assessed_cash": 2.0,
            "verified_candidate_cash": 1.0,
            "previous_close": 100.0,
            "candidate_factor": 0.99,
        }
    ]
    f, seen = run(bars, cal, findings, bundle)
    assert seen == {("a", "2025-04-01")}
    assert f.loc[1, "dividends"] == 2
    assert f.loc[1, "candidate_cash"] == 1
    assert f.loc[1, "candidate_adjustment_factor"] == pytest.approx(0.99)
    assert f.loc[1, "candidate_total_return"] == pytest.approx(0.02)
    assert f.loc[1, "candidate_provisional"]
    bundle["reconciliation"]["entries"][0]["candidate_factor"] = 0.98
    with pytest.raises(ValueError, match="Pinned cash and factor"):
        run(bars, cal, findings, bundle)


def test_existing_split_not_applied_twice():
    bars, cal, findings, bundle = fixture()
    bars.loc[1, "stock_splits"] = 2.0
    f, _ = run(bars, cal, findings, bundle)
    assert f.loc[1, "candidate_close"] == 101
    assert f.loc[1, "candidate_volume"] == 1000
    assert f.loc[1, "candidate_share_multiplier"] == 1
    assert f.loc[1, "candidate_total_return"] == pytest.approx(103 / 100 - 1)
    assert f.loc[1, "historical_quarantine"]
