"""P1 calendar, identity, and bounded targeting tests (no AWS writes)."""
import json
import pandas as pd
import pytest
from alpha_edge.market.aea87_publication_validation import (
    calendar_for_provider, verify_candidate_against_calendar,
    validate_old_only_against_calendar,
)
from alpha_edge.market.aea87_rebuild_publication_prepare import (
    identity_conflict_diagnostic, align_verified_metadata,
)
from alpha_edge.market.aea87_rebuild import select_p1_assets, P1_DATE_REPAIR_ASSETS


def test_known_exchange_mappings_and_unknown_fails_closed():
    assert calendar_for_provider("GER", "EQUITY", "EQH1") == "XETR"
    assert calendar_for_provider("PAR", "EQUITY", "EQH1") == "XPAR"
    assert calendar_for_provider("MCE", "EQUITY", "EQH1") == "XMAD"
    assert calendar_for_provider("NMS", "EQUITY", "EQH1") == "NASDAQ"
    assert calendar_for_provider("UNKNOWN", "EQUITY", "EQH1") is None
    assert calendar_for_provider("CCC", "CRYPTOCURRENCY", "CRYPTO:BTC-USD") is None


def test_exchange_calendar_excludes_sunday_source():
    result = verify_candidate_against_calendar(
        pd.to_datetime(["2026-06-01", "2026-05-31"]), "XPAR"
    )
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["dates"] == ["2026-05-31"]


def test_nyse_calendar_distinguishes_real_missing_day_from_holiday():
    old = pd.to_datetime(["2026-01-01", "2026-01-02"])
    fresh = pd.to_datetime(["2026-01-02"])
    assert validate_old_only_against_calendar(old, fresh, "NYSE")["status"] == "PASSED"
    old = pd.to_datetime(["2026-01-05", "2026-01-06"])
    fresh = pd.to_datetime(["2026-01-06"])
    result = validate_old_only_against_calendar(old, fresh, "NYSE")
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["sessions"] == ["2026-01-05"]


def test_unknown_calendar_blocks_without_guessing():
    result = verify_candidate_against_calendar(
        pd.to_datetime(["2026-06-01"]), "NOT_AN_EXCHANGE"
    )
    assert result["status"] == "REVIEW_REQUIRED"


def test_ambiguous_yahoo_ticker_never_autoresolves():
    active = pd.DataFrame({
        "asset_id": ["EQH1", "EQH1"],
        "ticker": ["B", "B"], "yahoo_ticker": ["ONE", "TWO"],
        "currency": ["USD", "USD"],
    })
    candidate = pd.DataFrame({"asset_id": ["EQH1"], "date": pd.to_datetime(["2026-01-05"])})
    with pytest.raises(ValueError, match="Ambiguous active identity field: yahoo_ticker"):
        align_verified_metadata(candidate, active, "ohlcv_usd",
                                provider_symbol="ONE", provider_currency="USD",
                                broker_ticker="B")
    diag = identity_conflict_diagnostic(
        "EQH1", {"ohlcv_usd": active},
        {"symbol": "ONE", "currency": "USD", "exchange": "NYQ"}, "B",
        "Ambiguous active identity field: yahoo_ticker"
    )
    assert diag["status"] == "BLOCKED_IDENTITY"
    assert diag["operations"] == []
    assert diag["identity_evidence"]["active_observations"]["ohlcv_usd"]["yahoo_ticker"] == ["ONE", "TWO"]


class FakeS3:
    def __init__(self, payloads):
        self.payloads = payloads
    def get_object(self, Bucket, Key):
        import io
        from botocore.exceptions import ClientError
        if Key not in self.payloads:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.payloads[Key])}


def test_p1_selection_only_original_287_plus_two_date_corruption():
    from alpha_edge.market.aea87_rebuild import ROOT, VALIDATION_LATEST, P1_SOURCE_RELEASE
    exceptions = [{"asset_id": f"EQH{i:04d}", "reason": "BLOCKED_COVERAGE"} for i in range(256)]
    exceptions += [{"asset_id": f"EQHid{i:04d}", "reason": "ValueError"} for i in range(31)]
    exceptions += [{"asset_id": f"EQHsource{i:04d}", "reason": "ERROR"} for i in range(116)]
    exceptions += [{"asset_id": f"EQHbad{i:04d}", "reason": "INVALID_SOURCE_PRICES"} for i in range(2)]
    unresolved = {x["asset_id"] for x in exceptions if x["reason"] in ("BLOCKED_COVERAGE", "ValueError")}
    report_key = "validation/test/report.json"
    pointer = {"report_key": report_key}
    validation = {"complete_universe": True, "scanned": len(unresolved) + 2,
                  "universe_count": len(unresolved) + 2,
                  "assets": [{"asset_id": x} for x in sorted(unresolved | P1_DATE_REPAIR_ASSETS)]}
    release = {"release_id": P1_SOURCE_RELEASE, "status": "COMPLETED_WITH_EXCEPTIONS",
               "exceptions": exceptions}
    payloads = {
        VALIDATION_LATEST: json.dumps(pointer).encode(),
        report_key: json.dumps(validation).encode(),
        f"{ROOT}/{P1_SOURCE_RELEASE}/result.json": json.dumps(release).encode(),
    }
    selected = select_p1_assets(FakeS3(payloads), "test", sorted(unresolved))
    assert set(selected) == unresolved | P1_DATE_REPAIR_ASSETS
    assert len(selected) == 289
