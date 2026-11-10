"""Build a full-population Lake v2 registry from the canonical universe.

Never fill missing currencies, exchange calendars or FX quote directions
by guessing from ticker suffixes. Audit all included assets, preserving
unresolved entries as explicit work items before broad acquisition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def build(source: pd.DataFrame, exchange_map: pd.DataFrame,
          fx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"asset_id", "yahoo_ticker", "include", "currency", "exchange"}
    if missing := required - set(source.columns):
        raise ValueError(f"Canonical universe lacks {sorted(missing)}")
    if {"exchange", "calendar_id"} - set(exchange_map.columns):
        raise ValueError("Exchange calendar mapping requires exchange,calendar_id")
    if {"currency", "provider_symbol", "direction"} - set(fx.columns):
        raise ValueError("FX mapping requires currency,provider_symbol,direction")
    if exchange_map["exchange"].duplicated().any() or fx["currency"].duplicated().any():
        raise ValueError("Duplicate exchange or FX mappings")
    all_included = pd.to_numeric(source["include"], errors="coerce")
    if all_included.isna().any() or not all_included.isin([0, 1]).all():
        raise ValueError("Canonical universe include must be 0 or 1")
    included = source.loc[all_included.eq(1)].copy()
    for col in ("asset_id", "yahoo_ticker", "currency", "exchange"):
        included[col] = included[col].astype("string").str.strip()
    included["currency"] = included["currency"].str.upper()
    xmap = exchange_map[["exchange", "calendar_id"]].copy()
    xmap["exchange"] = xmap["exchange"].astype("string").str.strip()
    joined = included.merge(xmap, on="exchange", how="left", validate="many_to_one")
    fx_ccy = set(fx["currency"].astype(str).str.upper())
    duplicates = joined["asset_id"].duplicated(keep=False)
    duplicated_symbols = joined["yahoo_ticker"].duplicated(keep=False)
    reasons = []
    for row, dup, dupticker in zip(joined.itertuples(index=False), duplicates, duplicated_symbols):
        problems = []
        if dup:
            problems.append("duplicate_asset_id")
        if dupticker:
            problems.append("duplicate_provider_symbol")
        for col in ("asset_id", "yahoo_ticker", "currency", "exchange", "calendar_id"):
            value = getattr(row, col)
            if pd.isna(value) or str(value).strip() in ("", "<NA>", "nan", "None"):
                problems.append(f"missing_{col}")
        if not pd.isna(row.currency) and row.currency != "USD" and row.currency not in fx_ccy:
            problems.append("unmapped_fx_currency")
        reasons.append(";".join(problems))
    joined["review_reason"] = reasons
    fields = ["asset_id", "yahoo_ticker", "currency", "calendar_id", "include"]
    approved = joined.loc[joined["review_reason"].eq(""), fields].copy()
    unresolved = joined.loc[joined["review_reason"].ne("")].copy()
    return approved, unresolved, joined


def generate(*, universe_csv: Path, exchange_map_csv: Path,
             fx_csv: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing overwrite: {output}")
    canonical = pd.read_csv(universe_csv, low_memory=False)
    exchange_map = pd.read_csv(exchange_map_csv)
    fx = pd.read_csv(fx_csv)
    ready, unresolved, all_included = build(canonical, exchange_map, fx)
    output.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for name, frame in (("ready_registry.csv", ready),
                        ("requires_review.csv", unresolved),
                        ("all_included_audit.csv", all_included)):
        p = output / name
        frame.to_csv(p, index=False)
        checksums[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    report = {
        "schema": "alpha_edge_full_universe_registry_review_1",
        "canonical_source_sha256": hashlib.sha256(universe_csv.read_bytes()).hexdigest(),
        "included_assets": len(all_included),
        "configuration_ready_assets": len(ready),
        "unresolved_assets": len(unresolved),
        "full_population_ready": len(unresolved) == 0,
        "files_sha256": checksums, "s3_writes": 0,
        "limitations": ("Ready means explicit metadata configuration, not validated source "
                        "history, inception, completeness or provider reliability. "
                        "Do not treat unresolved instruments as silently excluded."),
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe-csv", type=Path, default=Path("data/universe/universe.csv"))
    p.add_argument("--exchange-map-csv", type=Path, required=True)
    p.add_argument("--fx-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(generate(universe_csv=a.universe_csv,
                              exchange_map_csv=a.exchange_map_csv,
                              fx_csv=a.fx_csv, output=a.output), indent=2))


if __name__ == "__main__":
    main()
