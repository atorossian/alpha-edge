"""Source revision comparison never uses v1 or modifies source batches."""
import json
import hashlib
import pandas as pd
import pytest

from alpha_edge.market.compare_bronze_revisions_v2 import (
    RevisionContractError, compare_batches, compare_table, generate,
)


def _batch(tmp_path, batch_id, *, close=25.18, fx=1.12, missing_day=False,
           adjusted=None, date_shift=False):
    root = tmp_path / batch_id
    root.mkdir()
    dates = ["2026-10-27", "2026-10-28"]
    if missing_day:
        dates = dates[:1]
    bars = pd.DataFrame([{
        "asset_id": "A", "provider_symbol": "KESKOB.HE",
        "currency_original": "EUR",
        "session_date": day,
        "source_timestamp": f"{day}T00:00:00+02:00",
        "source_timezone": "Europe/Helsinki",
        "open": close, "high": close + 1,
        "low": close - 1, "close": close,
        "adj_close": adjusted if adjusted is not None else close,
        "volume": 1000., "dividends": 0., "stock_splits": 0.,
    } for day in dates])
    if date_shift:
        bars.loc[0, "source_timestamp"] = "2026-10-27T00:00:00+03:00"
    quotes = pd.DataFrame([{
        "provider_symbol": "EURUSD=X", "session_date": day,
        "quote_close": fx, "source_timestamp": day + "T00:00:00+00:00",
        "source_timezone": "UTC",
    } for day in dates])
    files = {}
    for name, frame in [("source_ohlcv.parquet", bars),
                        ("source_fx_quotes.parquet", quotes)]:
        target = root / name
        frame.to_parquet(target, index=False)
        files[name] = {"sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    manifest = {
        "schema_version": "bronze-v2-1", "batch_id": batch_id,
        "files": files,
        "sources": [
            {"asset_id": "A", "provider_symbol": "KESKOB.HE",
             "start": "2026-10-27", "end_exclusive": "2026-10-29",
             "interval": "1d", "auto_adjust": False, "repair": False},
            {"provider_symbol": "EURUSD=X",
             "start": "2026-10-27", "end_exclusive": "2026-10-29",
             "interval": "1d", "auto_adjust": False, "repair": False},
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_detect_price_scaling_even_when_close_equals_adjusted(tmp_path):
    first = _batch(tmp_path, "first", close=25.18)
    later = _batch(tmp_path, "later", close=24.503658)
    changes, manifest = compare_batches(old_dir=first, new_dir=later)
    assert manifest["legacy_v1_used_as_reference"] is False
    assert manifest["comparisons"]["ohlcv"]["changed_keys"] == 2
    assert {"close", "adj_close"}.issubset(set(changes.field))
    assert manifest["source_equivalence"] == "differences_require_review"


def test_no_revision_for_same_observations(tmp_path):
    a = _batch(tmp_path, "one")
    b = _batch(tmp_path, "two")
    changes, report = compare_batches(old_dir=a, new_dir=b)
    assert changes.empty
    assert report["comparisons"]["fx"]["changed_fields"] == 0


def test_missing_session_is_flagged(tmp_path):
    a = _batch(tmp_path, "original")
    b = _batch(tmp_path, "revised", missing_day=True)
    details, report = compare_batches(old_dir=a, new_dir=b)
    assert "left_only" in set(details["change_type"])
    assert report["comparisons"]["ohlcv"]["old_only"] == 1


def test_fx_revision_is_separate_from_price(tmp_path):
    a = _batch(tmp_path, "first", fx=1.12)
    b = _batch(tmp_path, "second", fx=1.17)
    changes, report = compare_batches(old_dir=a, new_dir=b)
    assert report["comparisons"]["ohlcv"]["changed_fields"] == 0
    assert report["comparisons"]["fx"]["changed_fields"] == 2
    assert set(changes.loc[changes.table.eq("fx"), "field"]) == {"quote_close"}


def test_reject_same_batch_or_checksum_tampering(tmp_path):
    a = _batch(tmp_path, "first")
    with pytest.raises(RevisionContractError, match="DISTINCT"):
        compare_batches(old_dir=a, new_dir=a)
    b = _batch(tmp_path, "second")
    with (b / "source_ohlcv.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        compare_batches(old_dir=a, new_dir=b)


def test_read_only_output_is_immutable(tmp_path):
    a = _batch(tmp_path, "first")
    b = _batch(tmp_path, "second", close=25.3)
    output = tmp_path / "report"
    report = generate(old_dir=a, new_dir=b, output=output)
    assert report["s3_writes"] == 0
    assert (output / "source_revisions.csv").is_file()
    with pytest.raises(FileExistsError):
        generate(old_dir=a, new_dir=b, output=output)
