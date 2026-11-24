"""EODHD endpoint routing preserves credentials and old Bronze artifacts."""

import json
from types import SimpleNamespace

import pytest

from alpha_edge.market import eodhd_source_v3 as source


def record(symbol="PRY.MI", base=source.MILAN_API_BASE):
    return {
        "asset_id": "test-asset",
        "provider_symbol": symbol,
        "mapping_version": "a" * 64,
        "api_base": base,
    }


def test_milan_routes_all_endpoints_with_certificate_verification(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        rows = (
            [{"date": "2026-11-23", "open": 1, "high": 2, "low": 1,
              "close": 2, "adjusted_close": 2, "volume": 100}]
            if "/eod/" in url else []
        )
        return SimpleNamespace(status_code=200, content=json.dumps(rows).encode())

    monkeypatch.setattr(source.requests, "get", get)
    payload, stats, provenance = source.acquire_eodhd(record(), "2026-11-24", "test-token")
    assert stats["rows"] == 1
    assert provenance["api_base"] == source.MILAN_API_BASE
    assert [url for url, _ in calls] == [
        f"{source.MILAN_API_BASE}/{endpoint}/PRY.MI" for endpoint in source.ENDPOINTS
    ]
    assert all(options["verify"] == source.certifi.where() and not options["allow_redirects"]
               for _, options in calls)
    assert b"test-token" not in payload
    assert source.verify_bundle(payload, record(), "2026-11-24") == stats


@pytest.mark.parametrize("symbol,base", [
    ("AAPL.US", source.MILAN_API_BASE),
    ("PRY.MI", "https://unapproved.example/api"),
    ("PRY.MI", "http://cboe.eodhd.com/api"),
])
def test_unapproved_destination_rejected_before_transport(monkeypatch, symbol, base):
    def fail(*args, **kwargs):
        pytest.fail("Unexpected network call")

    monkeypatch.setattr(source.requests, "get", fail)
    with pytest.raises(source.EodhdSourceError, match="Unapproved"):
        source.acquire_eodhd(record(symbol, base), "2026-11-24", "test-token")


def test_redirect_is_rejected_and_error_omits_token(monkeypatch):
    monkeypatch.setattr(source.requests, "get", lambda *a, **k: SimpleNamespace(status_code=302))
    with pytest.raises(source.EodhdSourceError, match="HTTP 302") as error:
        source.acquire_eodhd(record(), "2026-11-24", "test-token")
    assert "test-token" not in str(error.value)


def test_ampersand_symbol_mapping_and_encoded_path(monkeypatch):
    version = "a" * 64
    entry = {"asset_id": "test-asset", "yahoo_symbol": "PE&OLES.MX", "eodhd_symbol": "PE&OLES.MX"}
    document = {"schema": "alpha_edge_eodhd_symbol_mapping_v1", "assets": [entry]}
    monkeypatch.setattr(source, "verified_bytes", lambda *a: json.dumps(document).encode())
    mapped = source.load_mapping(None, "bucket", [{"asset_id": "test-asset", "yahoo_ticker": "PE&OLES.MX"}], version)[0]
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        rows = (
            [{"date": "2026-11-23", "open": 1, "high": 2, "low": 1,
              "close": 2, "adjusted_close": 2, "volume": 100}]
            if "/eod/" in url else []
        )
        return SimpleNamespace(status_code=200, content=json.dumps(rows).encode())

    monkeypatch.setattr(source.requests, "get", get)
    payload, _, _ = source.acquire_eodhd(mapped, "2026-11-24", "test-token")
    assert json.loads(payload)["provider_symbol"] == "PE&OLES.MX"
    assert [url for url, _ in calls] == [
        f"{source.STANDARD_API_BASE}/{endpoint}/PE%26OLES.MX" for endpoint in source.ENDPOINTS
    ]
    assert all(options["params"]["api_token"] == "test-token" for _, options in calls)


@pytest.mark.parametrize("symbol", ["A/B.MX", "../ABC.MX", "ABC?x.MX", "ABC#x.MX", "ABC%2F.MX", ".MX"])
def test_invalid_path_symbols_rejected(symbol):
    with pytest.raises(source.EodhdSourceError, match="Invalid EODHD symbol"):
        source._validate_symbol(symbol)
