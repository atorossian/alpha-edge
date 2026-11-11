"""Historical reconciliation tests: no silent date, price or return mismatch."""

import hashlib
import json

import pandas as pd
import pytest

from alpha_edge.market.reconcile_aea87_lake_v2 import (
    ReconciliationError,
    generate,
    reconcile,
)


def fixture(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    gold = pd.DataFrame(
        [
            {
                "asset_id": "A_US",
                "session_date": day,
                "close_adjusted_usd": close,
                "ret_simple_close_adjusted_usd": ret,
                "ret_log_close_adjusted_usd": logret,
                "quality_status": "accepted",
                "return_quality_status": "accepted",
                "observation_origin": "yahoo_finance",
            }
            for day, close, ret, logret in [
                ("2026-04-21", 100.0, None, None),
                ("2026-04-22", 101.0, 0.01, 0.009950330853168),
                ("2026-04-23", 102.0, 102.0 / 101 - 1, 0.009852296443),
            ]
        ]
    )
    gold.to_parquet(root / "gold.parquet", index=False)
    manifest = {
        "schema": "market_data_lake_v2_silver_gold_candidate_1",
        "publication_state": "candidate_only",
        "legacy_production_unchanged": True,
        "bronze_manifest_sha256": "a" * 64,
        "source_rows": 3,
        "gold_rows": 3,
        "silver_rows": 3,
        "acceptance": "initial_contract_pass",
        "files_sha256": {},
    }
    for filename in (
        "silver.parquet",
        "bronze_validated.parquet",
        "fx_normalized.parquet",
        "fx_evidence.parquet",
        "session_issues.parquet",
    ):
        (root / filename).write_bytes(b"placeholder")
    (root / "asset_quality_summary.csv").write_text("asset_id,rows\nA_US,3\n")
    for f in root.iterdir():
        manifest["files_sha256"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def legacy(tmp_path):
    prices = tmp_path / "legacy_prices.csv"
    returns = tmp_path / "legacy_returns.csv"
    pd.DataFrame(
        [
            {"asset_id": "A_US", "date": "2026-04-21", "close_adjusted_usd": 100.0},
            {"asset_id": "A_US", "date": "2026-04-22", "close_adjusted_usd": 101.5},
            {"asset_id": "A_US", "date": "2026-04-24", "close_adjusted_usd": 104.0},
        ]
    ).to_csv(prices, index=False)
    pd.DataFrame(
        [
            {"asset_id": "A_US", "date": "2026-04-21", "ret_close_adjusted_usd": None},
            {"asset_id": "A_US", "date": "2026-04-22", "ret_close_adjusted_usd": 0.015},
            {"asset_id": "A_US", "date": "2026-04-24", "ret_close_adjusted_usd": 0.023},
        ]
    ).to_csv(returns, index=False)
    return prices, returns


def test_reconcile_outer_join_catches_missing_sessions_and_price_mismatch(tmp_path):
    root = fixture(tmp_path)
    prices, returns = legacy(tmp_path)
    detail, summary, manifest = reconcile(root, prices, returns, return_kind="simple")
    a = detail.set_index(detail["session_date"].dt.strftime("%Y-%m-%d"))
    assert a.loc["2026-04-22", "price_comparison"] == "mismatch"
    assert a.loc["2026-04-22", "return_comparison"] == "mismatch"
    assert a.loc["2026-04-23", "historical_presence"] == "right_only"
    assert a.loc["2026-04-24", "historical_presence"] == "left_only"
    assert summary.iloc[0]["aea87_only_sessions"] == 1
    assert summary.iloc[0]["v2_only_sessions"] == 1
    assert manifest["comparison_status"] == "review_required"


def test_unknown_return_semantics_never_infer_simple(tmp_path):
    root = fixture(tmp_path)
    prices, returns = legacy(tmp_path)
    detail, _, manifest = reconcile(root, prices, returns)
    assert detail["return_comparison"].eq("not_comparable_semantics_unknown").all()
    assert detail["return_diff"].isna().all()
    assert manifest["legacy_return_kind"] == "unknown"


def test_log_semantics_are_explicit(tmp_path):
    root = fixture(tmp_path)
    prices, returns = legacy(tmp_path)
    ret = pd.read_csv(returns)
    ret.loc[ret["date"] == "2026-04-22", "ret_close_adjusted_usd"] = 0.009950330853168
    ret.to_csv(returns, index=False)
    detail, _, _ = reconcile(root, prices, returns, return_kind="log")
    row = detail.loc[detail["session_date"].eq(pd.Timestamp("2026-04-22"))].iloc[0]
    assert row["return_comparison"] == "match"
    assert row["price_comparison"] == "mismatch"


def test_duplicate_legacy_keys_fail_closed(tmp_path):
    root = fixture(tmp_path)
    prices, returns = legacy(tmp_path)
    duplicated = pd.read_csv(prices)
    duplicated = pd.concat([duplicated, duplicated.iloc[[0]]], ignore_index=True)
    duplicated.to_csv(prices, index=False)
    with pytest.raises(ReconciliationError, match="duplicate"):
        reconcile(root, prices, returns)


def test_generate_has_source_hashes_and_does_not_overwrite(tmp_path):
    root = fixture(tmp_path)
    prices, returns = legacy(tmp_path)
    out = tmp_path / "reconcile"
    result = generate(root, prices, returns, output=out)
    assert result["legacy_prices_sha256"] == hashlib.sha256(prices.read_bytes()).hexdigest()
    assert (out / "session_differences.csv").exists()
    assert (out / "asset_reconciliation_summary.csv").exists()
    with pytest.raises(FileExistsError):
        generate(root, prices, returns, output=out)


def test_corrected_missing_adjusted_price_stays_explicitly_unavailable(tmp_path):
    root = fixture(tmp_path)
    gold = pd.read_parquet(root / "gold.parquet")
    gold.loc[gold["session_date"].eq("2026-04-22"), "close_adjusted_usd"] = float("nan")
    gold.loc[gold["session_date"].eq("2026-04-22"), "quality_status"] = "quarantined"
    gold.loc[gold["session_date"].eq("2026-04-22"), "observation_origin"] = "user_supplied_manual"
    gold.to_parquet(root / "gold.parquet", index=False)
    manifest_file = root / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["files_sha256"]["gold.parquet"] = hashlib.sha256((root / "gold.parquet").read_bytes()).hexdigest()
    manifest_file.write_text(json.dumps(manifest))
    prices, returns = legacy(tmp_path)
    detail, _, report = reconcile(root, prices, returns, return_kind="simple")
    row = detail.loc[detail["session_date"].eq(pd.Timestamp("2026-04-22"))].iloc[0]
    assert row["price_comparison"] == "unavailable_price"
    assert "manual_correction" in row["review_reason"]
    assert report["comparison_status"] == "review_required"
