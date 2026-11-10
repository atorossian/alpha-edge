"""Read an immutable Bronze pilot and generate audited Silver/Gold candidates.

Standalone local-only stage. No Yahoo fetches, no AWS writes, and no changes
to legacy MarketStore manifests, snapshots or production returns cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from alpha_edge.market.standardize_bronze_v2 import build_silver_gold_from_bronze
from alpha_edge.market.manual_observations_v2 import append_manual_observations
from alpha_edge.market.preflight_universe_v2 import prepare_registry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_inputs(batch_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    required = ["source_ohlcv.parquet", "source_fx_quotes.parquet"]
    for filename in required:
        path = batch_dir / filename
        checksum = manifest.get("files", {}).get(filename, {}).get("sha256")
        if not checksum or _sha256(path) != checksum:
            raise ValueError(f"Bronze source hash mismatch: {filename}")
    return (
        pd.read_parquet(batch_dir / required[0]),
        pd.read_parquet(batch_dir / required[1]),
        manifest,
    )


def generate(
    batch_dir: Path, *, output: Path, corrections_csv: Path | None = None,
    asset_registry_csv: Path | None = None, fx_registry_csv: Path | None = None,
) -> dict:
    bars, quotes, source = _source_inputs(batch_dir)
    source_row_count = len(bars)
    correction_audit = None
    correction_sha256 = None
    # Never silently omit registered manual observations during cohort runs.
    # Filter the global registry by cohort asset, preserving original file hash.
    registry_path = (
        Path(__file__).resolve().parents[3]
        / "data/quality/market_lake_v2/manual_ohlcv_corrections.csv"
    )
    if corrections_csv is None:
        if not registry_path.is_file():
            raise ValueError("Mandatory manual-correction registry is missing")
        corrections_csv = registry_path
    if corrections_csv is not None:
        corrections = pd.read_csv(corrections_csv)
        if "asset_id" not in corrections:
            raise ValueError("Manual correction registry missing asset_id")
        corrections = corrections.loc[
            corrections["asset_id"].astype(str).isin(set(bars["asset_id"].astype(str)))
        ].copy()
        if len(corrections):
            correction_sha256 = _sha256(corrections_csv)
            bars, correction_audit = append_manual_observations(bars, corrections)
    if (asset_registry_csv is None) != (fx_registry_csv is None):
        raise ValueError("Both asset and FX registries must be supplied together")
    config_hashes = None
    if asset_registry_csv is not None:
        u, fx_registry = prepare_registry(
            pd.read_csv(asset_registry_csv), pd.read_csv(fx_registry_csv)
        )
        source_identity = bars.groupby("asset_id")[["provider_symbol", "currency_original"]].first()
        expected_identity = u.set_index("asset_id")
        if set(source_identity.index) != set(expected_identity.index):
            raise ValueError("Bronze cohort does not match explicit asset registry")
        for aid in expected_identity.index:
            if (source_identity.loc[aid, "provider_symbol"] != expected_identity.loc[aid, "yahoo_ticker"]
                    or source_identity.loc[aid, "currency_original"] != expected_identity.loc[aid, "currency"]):
                raise ValueError(f"{aid}: Bronze identity or native currency inconsistent with registry")
        calendars = dict(zip(u["yahoo_ticker"], u["calendar_id"]))
        pairs = {r.currency: (r.provider_symbol, r.direction)
                 for r in fx_registry.itertuples(index=False)}
        config_hashes = {
            "asset_registry_sha256": _sha256(asset_registry_csv),
            "fx_registry_sha256": _sha256(fx_registry_csv),
        }
    else:
        calendars, pairs = None, None
    outputs = build_silver_gold_from_bronze(
        bars, quotes, calendars=calendars, pairs=pairs
    )
    if len(outputs["bronze_validated"]) != len(bars) or len(outputs["silver"]) != len(bars):
        raise ValueError("Bronze-to-Silver row count mismatch")
    if len(outputs["gold"]) != len(bars):
        raise ValueError("Silver-to-Gold row count mismatch")
    if output.exists():
        raise FileExistsError(f"Candidate run already exists: {output}")
    # Work in a local candidate folder; manifest last.
    output.mkdir(parents=True)
    artifacts = {}
    if correction_audit is not None:
        target = output / "manual_corrections_audit.parquet"
        correction_audit.to_parquet(target, index=False)
        artifacts[target.name] = _sha256(target)
    for name, table in outputs.items():
        filename = name + ".parquet"
        target = output / filename
        table.to_parquet(target, index=False)
        artifacts[filename] = _sha256(target)

    summary = []
    traces = output / "asset_traces"
    traces.mkdir()
    for (aid, symbol), g in outputs["gold"].groupby(
        ["asset_id", "provider_symbol"], sort=True
    ):
        quality = g["quality_status"]
        return_status = g["return_quality_status"]
        summary.append({
            "asset_id": aid,
            "symbol": symbol,
            "currency": g["currency_original"].iloc[0],
            "rows": len(g),
            "accepted_prices": int(quality.eq("accepted").sum()),
            "quarantined_prices": int(quality.eq("quarantined").sum()),
            "missing_fx": int(g["quality_reason"].str.contains("fx_missing").sum()),
            "stale_fx": int(g["quality_reason"].str.contains("fx_stale").sum()),
            "session_gap_affected_rows": int(
                g["quality_reason"].str.contains("prior_expected_session_missing").sum()
            ),
            "valid_returns": int(return_status.eq("accepted").sum()),
            "review_returns": int(return_status.eq("review").sum()),
            "unavailable_returns": int(return_status.eq("unavailable").sum()),
        })
        # Asset-id is treated as data, never as a filesystem path.
        filename = "asset_" + hashlib.sha256(str(aid).encode()).hexdigest()[:16] + ".csv"
        target = traces / filename
        g.sort_values("session_date").to_csv(target, index=False)
        artifacts[target.relative_to(output).as_posix()] = _sha256(target)

    pd.DataFrame(summary).to_csv(output / "asset_quality_summary.csv", index=False)
    artifacts["asset_quality_summary.csv"] = _sha256(output / "asset_quality_summary.csv")
    report = {
        "schema": "market_data_lake_v2_silver_gold_candidate_1",
        "bronze_batch_id": source.get("batch_id"),
        "bronze_manifest_sha256": _sha256(batch_dir / "manifest.json"),
        "source_rows": source_row_count,
        "manual_correction_rows": len(bars) - source_row_count,
        "candidate_input_rows": len(bars),
        "manual_corrections_csv_sha256": correction_sha256,
        "manual_correction_approval": (
            "pending_independent_verification"
            if correction_audit is not None else "not_applicable"
        ),
        "silver_rows": len(outputs["silver"]),
        "gold_rows": len(outputs["gold"]),
        "fx_quote_rows": len(quotes),
        "fx_accepted_rows": len(outputs["fx_normalized"]),
        "fx_rejected_rows": int(
            outputs["fx_evidence"]["fx_validation_status"].ne("accepted").sum()
        ),
        "missing_expected_sessions": int(
            outputs["session_issues"]["issue"].eq("missing_expected_session").sum()
        ),
        "unexpected_sessions": int(
            outputs["session_issues"]["issue"].eq("unexpected_session").sum()
        ),
        "assets": summary,
        "source_config_hashes": config_hashes,
        "files_sha256": artifacts,
        "publication_state": "candidate_only",
        "legacy_production_unchanged": True,
        "acceptance": "review_required" if (
            correction_audit is not None
            or len(outputs["session_issues"]) > 0
            or any(a["quarantined_prices"] or a["review_returns"] for a in summary)
        ) else "initial_contract_pass",
        "warning": "Calendar audit covers only first-to-last observed sessions; "
                   "no official-close-time or corporate-action verification yet.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bronze-batch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corrections-csv", type=Path, default=None)
    parser.add_argument("--asset-registry-csv", type=Path, default=None)
    parser.add_argument("--fx-registry-csv", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(generate(
        args.bronze_batch_dir, output=args.output, corrections_csv=args.corrections_csv,
        asset_registry_csv=args.asset_registry_csv, fx_registry_csv=args.fx_registry_csv,
    ), indent=2))


if __name__ == "__main__":
    main()
