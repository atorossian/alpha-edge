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


def generate(batch_dir: Path, *, output: Path) -> dict:
    bars, quotes, source = _source_inputs(batch_dir)
    outputs = build_silver_gold_from_bronze(bars, quotes)
    if len(outputs["bronze_validated"]) != len(bars) or len(outputs["silver"]) != len(bars):
        raise ValueError("Bronze-to-Silver row count mismatch")
    if len(outputs["gold"]) != len(bars):
        raise ValueError("Silver-to-Gold row count mismatch")
    if output.exists():
        raise FileExistsError(f"Candidate run already exists: {output}")
    # Work in a local candidate folder; manifest last.
    output.mkdir(parents=True)
    artifacts = {}
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
        artifacts[str(target.relative_to(output))] = _sha256(target)

    pd.DataFrame(summary).to_csv(output / "asset_quality_summary.csv", index=False)
    artifacts["asset_quality_summary.csv"] = _sha256(output / "asset_quality_summary.csv")
    report = {
        "schema": "market_data_lake_v2_silver_gold_candidate_1",
        "bronze_batch_id": source.get("batch_id"),
        "bronze_manifest_sha256": _sha256(batch_dir / "manifest.json"),
        "source_rows": len(bars),
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
        "files_sha256": artifacts,
        "publication_state": "candidate_only",
        "legacy_production_unchanged": True,
        "acceptance": "review_required" if any(
            a["quarantined_prices"] or a["review_returns"] for a in summary
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
    args = parser.parse_args()
    print(json.dumps(generate(args.bronze_batch_dir, output=args.output), indent=2))


if __name__ == "__main__":
    main()
