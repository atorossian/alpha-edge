"""Data Quality Explorer: deterministic read-only candidate visibility tests."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from alpha_edge.market.data_quality_explorer_v2 import (
    build_explorer_model, generate_explorer, _safe_json,
)
from alpha_edge.market.publish_lake_v2 import PublicationError


def candidate(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    gold = pd.DataFrame([
        {
            "asset_id": "A_MX", "provider_symbol": "ALSEA.MX",
            "currency_original": "MXN", "session_date": pd.Timestamp("2026-04-21"),
            "provider": "yahoo_finance", "close": 51.10, "adj_close": 50.9,
            "close_adjusted_usd": 2.90, "fx_to_usd": 0.057,
            "quality_status": "accepted", "quality_reason": "",
            "return_quality_status": "accepted", "ret_simple_close_adjusted_usd": -0.01,
            "observation_origin": "yahoo_finance",
        },
        {
            "asset_id": "A_MX", "provider_symbol": "ALSEA.MX",
            "currency_original": "MXN", "session_date": pd.Timestamp("2026-04-22"),
            "provider": "manual_correction", "close": 51.80, "adj_close": float("nan"),
            "close_adjusted_usd": float("nan"), "fx_to_usd": 0.058,
            "quality_status": "quarantined", "quality_reason": "invalid_price",
            "return_quality_status": "unavailable",
            "ret_simple_close_adjusted_usd": float("nan"),
            "observation_origin": "user_supplied_manual",
            "correction_id": "manual-alsea-20260422-v1",
            "verification_status": "pending_independent_verification",
            "source_reference": "<unsafe>embedded string</unsafe>",
        },
        {
            "asset_id": "A_US", "provider_symbol": "SPY",
            "currency_original": "USD", "session_date": pd.Timestamp("2026-04-21"),
            "provider": "yahoo_finance", "close": 100.0, "adj_close": 100.0,
            "close_adjusted_usd": 100.0, "fx_to_usd": 1.0,
            "quality_status": "accepted", "quality_reason": "",
            "return_quality_status": "unavailable",
            "ret_simple_close_adjusted_usd": float("nan"),
            "observation_origin": "yahoo_finance",
        },
    ])
    sessions = pd.DataFrame(columns=[
        "asset_id", "provider_symbol", "session_date", "issue", "calendar_id"
    ])
    fx = pd.DataFrame([{
        "currency_original": "MXN", "fx_date": pd.Timestamp("2026-04-22"),
        "fx_quote_symbol": "USDMXN=X", "quote_close": 17.25,
        "fx_validation_status": "large_fx_move",
    }])
    tables = {
        "gold.parquet": gold, "session_issues.parquet": sessions,
        "fx_evidence.parquet": fx, "silver.parquet": gold,
        "bronze_validated.parquet": gold, "fx_normalized.parquet": fx,
    }
    for name, frame in tables.items():
        frame.to_parquet(root / name, index=False)
    (root / "asset_quality_summary.csv").write_text("asset_id,rows\nA_MX,2\nA_US,1\n")
    manifest = {
        "schema": "market_data_lake_v2_silver_gold_candidate_1",
        "publication_state": "candidate_only",
        "legacy_production_unchanged": True,
        "bronze_manifest_sha256": "a" * 64,
        "source_rows": 2, "manual_correction_rows": 1,
        "candidate_input_rows": 3, "silver_rows": 3, "gold_rows": 3,
        "acceptance": "review_required",
        "manual_correction_approval": "pending_independent_verification",
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.iterdir() if path.is_file()
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_model_preserves_manual_observation_and_null_adjusted_price(tmp_path):
    model = build_explorer_model(candidate(tmp_path))
    assert model["candidate_acceptance"] == "review_required"
    assert len(model["assets"]) == 2
    manual = next(x for x in model["observations"] if x["observation_origin"] == "user_supplied_manual")
    assert manual["close"] == 51.8
    assert manual["adj_close"] is None
    assert manual["close_adjusted_usd"] is None
    assert manual["quality_status"] == "quarantined"
    assert manual["verification_status"] == "pending_independent_verification"
    assert len(model["fx_issues"]) == 1


def test_html_is_self_contained_without_network_or_html_script_injection(tmp_path):
    source = candidate(tmp_path)
    target = tmp_path / "explorer" / "index.html"
    result = generate_explorer(source, target)
    page = target.read_text()
    assert result["observations"] == 3
    assert "https://" not in page
    assert "<script src=" not in page
    assert "<unsafe>embedded string</unsafe>" not in page
    assert "\\u003cunsafe" in page
    assert 'type="application/json"' in page
    with pytest.raises(FileExistsError):
        generate_explorer(source, target)


def test_checksum_tampering_is_refused(tmp_path):
    root = candidate(tmp_path)
    (root / "gold.parquet").write_bytes(b"replaced")
    with pytest.raises(PublicationError, match="SHA-256"):
        build_explorer_model(root)


def test_safe_json_escapes_closing_script_tags():
    payload = _safe_json({"label": "</script><script>alert(1)</script>"})
    assert "</script>" not in payload
    assert "\\u003c/script>" in payload
