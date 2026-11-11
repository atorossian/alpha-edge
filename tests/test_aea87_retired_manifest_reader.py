from alpha_edge.core.market_store import MarketStore


def test_explicit_retirement_does_not_fallback_to_orphan_parquet(monkeypatch):
    store = MarketStore(bucket="test")
    listed = []
    monkeypatch.setattr(store, "read_asset_year_manifest", lambda **kw: {"status": "RETIRED", "parts": []})

    def listing(prefix):
        listed.append(prefix)
        return ["legacy.parquet"]

    monkeypatch.setattr(store, "_list_keys", listing)
    out = store.read_ohlcv_usd(["TEST"], start="2024-01-01", end="2024-12-31")
    assert out.empty
    assert not listed


def test_missing_manifest_preserves_legacy_fallback(monkeypatch):
    store = MarketStore(bucket="test")
    monkeypatch.setattr(store, "read_asset_year_manifest", lambda **kw: {})
    seen = []
    monkeypatch.setattr(store, "_list_keys", lambda prefix: (seen.append(prefix) or []))
    store.read_ohlcv_usd(["TEST"], start="2024-01-01", end="2024-12-31")
    assert len(seen) == 1
