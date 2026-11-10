import pandas as pd

from alpha_edge.market.aea87_full_history_rebuild import normalize_dates
from alpha_edge.market.aea87_session_date_audit import audit_asset, _dates_from_ops


def test_berlin_monday_preserved_instead_of_utc_sunday():
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2026-06-01 00:00", tz="Europe/Berlin"),
                 pd.Timestamp("2026-06-02 00:00", tz="Europe/Berlin")],
        "Close": [10, 11],
    })
    result = normalize_dates(frame)
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-06-01", "2026-06-02"]


def test_mixed_dst_offsets_preserve_local_sessions():
    frame = pd.DataFrame({
        "date": [
            "2026-01-12T00:00:00+01:00",
            "2026-06-01T00:00:00+02:00",
        ],
        "Close": [10, 11],
    })
    result = normalize_dates(frame)
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-12", "2026-06-01"]


def test_naive_dates_unchanged_and_duplicates_deduplicated():
    frame = pd.DataFrame({"date": ["2026-06-01", "2026-06-01", "2026-06-02"],
                          "Close": [10, 11, 12]})
    result = normalize_dates(frame)
    assert result["Close"].tolist() == [11, 12]


def test_market_date_audit_flags_noncrypto_weekends():
    source = {"yahoo_ticker": "MRK.DE",
              "provider_metadata": {"exchange": "GER", "quote_type": "EQUITY"}}
    tables = {"ohlcv_usd": ["2026-05-31", "2026-06-02"],
              "returns_usd": ["2026-05-31", "2026-06-02"]}
    result = audit_asset("EQHxxx", tables, source)
    assert result["weekend_count"] == 1
    assert result["weekend_dates_sample"] == ["2026-05-31"]
    assert "WEEKEND_DATES_NONCRYPTO" in result["issues"]


def test_market_date_audit_allows_crypto_weekends():
    tables = {"ohlcv_usd": ["2026-05-31", "2026-06-01"],
              "returns_usd": ["2026-05-31", "2026-06-01"]}
    result = audit_asset("CRYPTO:BTC-USD", tables, {"provider_metadata": {"exchange": "CCC"}})
    assert result["weekend_count"] == 0
    assert result["status"] == "PASSED_WEEKEND_AND_DATE_CONSISTENCY"


def test_market_date_audit_flags_mismatched_price_return_dates():
    source = {"provider_metadata": {"exchange": "NYQ", "quote_type": "EQUITY"}}
    tables = {"ohlcv_usd": ["2026-06-01", "2026-06-02"],
              "returns_usd": ["2026-06-01"]}
    result = audit_asset("EQHxxx", tables, source)
    assert "PRICE_RETURN_DATE_SET_MISMATCH" in result["issues"]


def test_publication_evidence_aggregates_annual_manifest_dates():
    import json
    p = {"asset_id": "EQHxx", "dates": ["2025-12-31"]}
    r = {"asset_id": "EQHxx", "dates": ["2025-12-31"]}
    ops = [
        {"asset_id": "EQHxx", "manifest_key": "market/manifests/v1/ohlcv_usd/asset_id=EQHxx/year=2025/manifest.json", "after": json.dumps(p)},
        {"asset_id": "EQHxx", "manifest_key": "market/manifests/v1/returns_usd/asset_id=EQHxx/year=2025/manifest.json", "after": json.dumps(r)},
    ]
    result = _dates_from_ops(ops, {"EQHxx"})
    assert result["EQHxx"]["ohlcv_usd"] == ["2025-12-31"]
    assert result["EQHxx"]["returns_usd"] == ["2025-12-31"]
