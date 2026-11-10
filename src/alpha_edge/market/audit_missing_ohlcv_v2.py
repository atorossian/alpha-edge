"""Report missing/invalid OHLCV and exchange-session gaps across an acquired cohort.

Every missing source bar and invalid field remains evidence; never forward-fill.
Expected sessions are evaluated between first and last observed source dates.
Requested endpoint coverage is reported separately; inception cannot be
inferred from an arbitrary earliest requested date.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_edge.market.preflight_universe_v2 import prepare_registry
from alpha_edge.market.run_standardization_pilot_v2 import _source_inputs
from alpha_edge.market.standardize_bronze_v2 import audit_asset_sessions

FIELDS = ("open", "high", "low", "close", "adj_close", "volume")


def report_missing(bars: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendars = dict(zip(universe["yahoo_ticker"], universe["calendar_id"]))
    if set(bars["asset_id"].astype(str)) != set(universe["asset_id"]):
        raise ValueError("Bronze assets differ from requested cohort")
    _, sessions = audit_asset_sessions(bars, calendars=calendars)
    findings = []
    for row in bars.itertuples(index=False):
        for field in FIELDS:
            value = getattr(row, field, None)
            if pd.isna(value) or not np.isfinite(float(value)):
                findings.append({
                    "asset_id": row.asset_id, "provider_symbol": row.provider_symbol,
                    "session_date": str(row.session_date)[:10],
                    "field": field, "issue": "missing_or_nonfinite",
                })
            elif field in ("open", "high", "low", "close", "adj_close") and float(value) <= 0:
                findings.append({
                    "asset_id": row.asset_id, "provider_symbol": row.provider_symbol,
                    "session_date": str(row.session_date)[:10],
                    "field": field, "issue": "nonpositive_price",
                })
            elif field == "volume" and float(value) < 0:
                findings.append({
                    "asset_id": row.asset_id, "provider_symbol": row.provider_symbol,
                    "session_date": str(row.session_date)[:10],
                    "field": field, "issue": "negative_volume",
                })
    frame = pd.DataFrame(findings, columns=[
        "asset_id", "provider_symbol", "session_date", "field", "issue"
    ])
    return frame, sessions


def generate(bronze_dir: Path, asset_registry: Path, fx_registry: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Refusing to overwrite missing-price report")
    bars, _, manifest = _source_inputs(bronze_dir)
    universe, _ = prepare_registry(pd.read_csv(asset_registry), pd.read_csv(fx_registry))
    fields, sessions = report_missing(bars, universe)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, frame in (("missing_price_fields.csv", fields),
                        ("missing_exchange_sessions.csv", sessions)):
        path = output / name
        frame.to_csv(path, index=False)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    by_asset = []
    for asset in universe.itertuples(index=False):
        f = fields.loc[fields["asset_id"].eq(asset.asset_id)]
        s = sessions.loc[sessions["asset_id"].eq(asset.asset_id)]
        a = bars.loc[bars["asset_id"].eq(asset.asset_id)]
        by_asset.append({
            "asset_id": asset.asset_id, "provider_symbol": asset.yahoo_ticker,
            "observations": len(a), "first_observed": str(a["session_date"].min())[:10],
            "last_observed": str(a["session_date"].max())[:10],
            "missing_field_count": len(f), "missing_session_count": int(s["issue"].eq("missing_expected_session").sum()),
            "unexpected_session_count": int(s["issue"].eq("unexpected_session").sum()),
        })
    summary_path = output / "asset_coverage.csv"
    pd.DataFrame(by_asset).to_csv(summary_path, index=False)
    hashes["asset_coverage.csv"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    result = {
        "schema": "alpha_edge_lake_v2_missing_ohlcv_audit_1",
        "bronze_batch_id": manifest["batch_id"],
        "bronze_manifest_sha256": hashlib.sha256((bronze_dir / "manifest.json").read_bytes()).hexdigest(),
        "assets": len(by_asset), "missing_price_fields": len(fields),
        "missing_exchange_sessions": int(sessions["issue"].eq("missing_expected_session").sum()),
        "unexpected_exchange_sessions": int(sessions["issue"].eq("unexpected_session").sum()),
        "acceptance": "review_required" if len(fields) or len(sessions) else "internally_clean",
        "coverage_scope": "observed_history_first_to_last_excludes_unverified_inception_and_endpoints",
        "files_sha256": hashes, "s3_writes": 0,
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bronze-dir", type=Path, required=True)
    p.add_argument("--asset-registry-csv", type=Path, required=True)
    p.add_argument("--fx-registry-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(generate(a.bronze_dir, a.asset_registry_csv,
                              a.fx_registry_csv, a.output), indent=2))


if __name__ == "__main__":
    main()
