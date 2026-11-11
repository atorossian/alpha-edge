"""Summarize a real-data Yahoo Bronze pilot without altering its observations.

Usage:
 python -m alpha_edge.market.validate_bronze_pilot_v2 --batch-dir outputs/market_bronze_v2/BATCH_ID
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def validate(batch_dir: Path, expected_universe: pd.DataFrame) -> dict:
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing committed manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bars = pd.read_parquet(batch_dir / "source_ohlcv.parquet")
    fx = pd.read_parquet(batch_dir / "source_fx_quotes.parquet")
    if bars.empty:
        raise ValueError("No source bars downloaded")
    expected = expected_universe[["asset_id", "yahoo_ticker", "currency"]].copy()
    if expected.duplicated("asset_id").any():
        raise ValueError("Pilot universe has duplicate asset IDs")
    expected["asset_id"] = expected["asset_id"].astype(str)
    records = []
    failed = False
    for row in expected.itertuples(index=False):
        asset = bars.loc[bars["asset_id"].eq(row.asset_id)].copy()
        session = pd.to_datetime(asset["session_date"], errors="coerce")
        close = pd.to_numeric(asset["close"], errors="coerce")
        adj_close = pd.to_numeric(asset["adj_close"], errors="coerce")
        duplicate_count = int(session.duplicated(keep=False).sum())
        nonpositive = int(((close <= 0) | (adj_close <= 0)).sum())
        missing = int((close.isna() | adj_close.isna()).sum())
        currency_mismatch = not asset.empty and set(asset["currency_original"]) != {row.currency}
        symbol_mismatch = not asset.empty and set(asset["provider_symbol"]) != {row.yahoo_ticker}
        bad_dates = int(session.isna().sum())
        status = (
            "review"
            if (
                asset.empty
                or missing
                or nonpositive
                or duplicate_count
                or currency_mismatch
                or symbol_mismatch
                or bad_dates
            )
            else "ok"
        )
        failed |= status != "ok"
        records.append(
            {
                "asset_id": row.asset_id,
                "symbol": row.yahoo_ticker,
                "expected_currency": row.currency,
                "rows": int(len(asset)),
                "first_session": str(session.min().date()) if session.notna().any() else None,
                "last_session": str(session.max().date()) if session.notna().any() else None,
                "missing_close_rows": missing,
                "nonpositive_prices": nonpositive,
                "duplicate_session_rows": duplicate_count,
                "invalid_session_dates": bad_dates,
                "currency_mismatch": bool(currency_mismatch),
                "symbol_mismatch": bool(symbol_mismatch),
                "status": status,
            }
        )
    fx_records = []
    for symbol, group in fx.groupby("provider_symbol", dropna=False):
        dates = pd.to_datetime(group["session_date"], errors="coerce")
        rate = pd.to_numeric(group["quote_close"], errors="coerce")
        fx_records.append(
            {
                "provider_symbol": str(symbol),
                "rows": int(len(group)),
                "first_session": str(dates.min().date()) if dates.notna().any() else None,
                "last_session": str(dates.max().date()) if dates.notna().any() else None,
                "missing_or_nonpositive_rates": int((rate.isna() | (rate <= 0)).sum()),
            }
        )
    return {
        "batch_id": manifest["batch_id"],
        "stage": "bronze",
        "assets": records,
        "fx_quotes": fx_records,
        "source_manifest_files": manifest.get("files", {}),
        "assets_for_review": int(sum(x["status"] != "ok" for x in records)),
        "pilot_status": "review" if failed else "raw_contract_pass",
        "note": "This does not certify exchange-calendar completeness, FX correctness, or AEA-87 historical equivalence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--universe-csv", type=Path, default=Path("data/quality/market_lake_v2/pilot_six_assets.csv"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = validate(args.batch_dir, pd.read_csv(args.universe_csv))
    output = args.output or args.batch_dir / "validation_summary.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["pilot_status"] == "review":
        raise SystemExit("One or more assets require review; see validation_summary.json")


if __name__ == "__main__":
    main()
