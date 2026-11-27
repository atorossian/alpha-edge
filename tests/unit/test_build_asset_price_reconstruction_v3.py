import pandas as pd
import pytest

from alpha_edge.market.build_asset_price_reconstruction_v3 import reconstruct


def fixture():
    days = pd.date_range("2020-01-01", periods=11).strftime("%Y-%m-%d")
    bars = pd.DataFrame({"session_date": days, "asset_id": "a", "provider_symbol": "TM", "open": 100., "high": 101., "low": 99., "close": 100., "volume": 10., "dividends": 0., "stock_splits": 0.})
    bars.loc[5, "stock_splits"] = 2.
    e = [{"date": d, "open": 200. if i < 5 else 100., "high": 202. if i < 5 else 101., "low": 198. if i < 5 else 99., "close": 200. if i < 5 else 100., "adjusted_close": 100.} for i, d in enumerate(days)]
    doc = {"schema": "alpha_edge_asset_price_reconstruction_approval_v1", "approval": "approved_candidate_reconstruction", "production_eligible": False, "quarantine_released": False, "asset_id": "a", "symbol": "TM", "entries": [{"date": days[5], "ratio": 2., "method": "eodhd_price_basis_reconstruct", "effective_trading_date": days[5]}]}
    return bars, {"eod": e, "div": [{"date": days[4], "value": 1.}]}, doc


def test_anchored_window_no_double_split_and_cash_is_held():
    bars, responses, doc = fixture()
    out, stats = reconstruct(bars, responses, doc)
    assert out.candidate_close.eq(100).all()
    assert out.close.equals(bars.close)
    assert out.loc[4, "candidate_cash_review"]
    assert pd.isna(out.loc[4, "candidate_total_return"])
    assert out.loc[5, "candidate_price_return"] == 0
    assert pd.isna(out.loc[5, "candidate_volume"])
    assert stats["touched_rows"] == 9
    assert not out.production_eligible.any()


def test_local_patch_preserves_source_and_checks_split():
    bars, responses, doc = fixture()
    day = bars.loc[5, "session_date"]
    doc["entries"] = [{"date": day, "ratio": 2., "method": "local_ohlc_scale", "sessions": [day], "multiplier": .5}]
    out, _ = reconstruct(bars, responses, doc)
    assert out.loc[5, "candidate_close"] == 50
    assert out.loc[5, "candidate_high"] == 50.5
    assert out.loc[5, "close"] == 100
    doc["entries"][0]["ratio"] = 3.
    with pytest.raises(ValueError, match="Source split mismatch"):
        reconstruct(bars, responses, doc)


def test_issuer_precision_only_applies_incremental_factor_and_keeps_cash_entitlement():
    bars, responses, doc = fixture()
    day = bars.loc[5, "session_date"]
    rounded, exact = 1.005, 1.0056
    bars.loc[5, "stock_splits"] = rounded
    bars.loc[5, "dividends"] = .60 / rounded
    for i, row in enumerate(responses["eod"]):
        for c in ("open", "high", "low", "close"):
            row[c] = bars.loc[i, c] * (rounded / exact if i < 5 else 1)
    doc["entries"] = [{"date": day, "ratio": rounded, "issuer_ratio": exact,
                       "declared_cash": .60, "method": "issuer_ratio_precision"}]
    out, stats = reconstruct(bars, responses, doc)
    assert out.loc[4, "candidate_close"] == pytest.approx(100 * rounded / exact)
    assert out.loc[5, "candidate_close"] == 100
    assert out.loc[5, "candidate_split_event"] == exact
    assert out.loc[5, "stock_splits"] == rounded
    assert out.loc[5, "candidate_cash_dividend"] == pytest.approx(.60 / exact)
    assert not out.loc[5, "candidate_cash_review"]
    assert stats["corrected_ratios"] == 1
    assert out.loc[5, "dividends"] == pytest.approx(.60 / rounded)
    responses["eod"][0]["close"] = 50
    for row in responses["eod"][1:5]:
        row["close"] = 50
    with pytest.raises(ValueError, match="Independent precision step mismatch"):
        reconstruct(bars, responses, doc)
