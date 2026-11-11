"""Preflight contracts for a staged market universe, without Yahoo or S3 writes.

Each included asset must have explicit source identity, quote currency and
exchange calendar; each non-USD currency must have an approved Yahoo FX pair
and quote direction. No inference from suffixes or legacy metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


class UniversePreflightError(ValueError):
    pass


REQUIRED = ("asset_id", "yahoo_ticker", "currency", "calendar_id")
FX_REQUIRED = ("currency", "provider_symbol", "direction")
FX_DIRECTIONS = {"direct", "inverse"}


def _nonblank(table: pd.DataFrame, cols: tuple[str, ...], label: str):
    if missing := set(cols) - set(table.columns):
        raise UniversePreflightError(f"{label} missing columns: {sorted(missing)}")
    for col in cols:
        if table[col].isna().any() or table[col].astype(str).str.strip().isin({"", "nan", "None", "NULL"}).any():
            raise UniversePreflightError(f"{label}: empty {col}")


def prepare_registry(
    universe: pd.DataFrame, fx_config: pd.DataFrame, *, check_calendars: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = universe.copy()
    if "include" in u:
        included = pd.to_numeric(u["include"], errors="coerce")
        if included.isna().any() or not included.isin([0, 1]).all():
            raise UniversePreflightError("Include must be explicitly 0 or 1")
        u = u.loc[included.eq(1)].copy()
    if u.empty:
        raise UniversePreflightError("No included assets")
    _nonblank(u, REQUIRED, "Asset universe")
    for col in REQUIRED:
        u[col] = u[col].astype(str).str.strip()
    u["currency"] = u["currency"].str.upper()
    if u["asset_id"].duplicated().any():
        raise UniversePreflightError("Duplicate asset IDs")
    if u["yahoo_ticker"].duplicated().any():
        raise UniversePreflightError("Ambiguous repeated Yahoo symbol")
    if u[["asset_id", "yahoo_ticker"]].apply(lambda s: s.str.contains(r"[/\\]|\.\.", regex=True)).any().any():
        raise UniversePreflightError("Unsafe asset or provider symbol")
    fx = fx_config.copy()
    _nonblank(fx, FX_REQUIRED, "FX reference")
    for col in FX_REQUIRED:
        fx[col] = fx[col].astype(str).str.strip()
    fx["currency"] = fx["currency"].str.upper()
    fx["direction"] = fx["direction"].str.lower()
    if fx["currency"].duplicated().any() or not fx["direction"].isin(FX_DIRECTIONS).all():
        raise UniversePreflightError("Duplicate FX currency or invalid quote direction")
    if fx["currency"].eq("USD").any():
        raise UniversePreflightError("USD conversion must be implicit identity, never quoted")
    if not fx["provider_symbol"].str.endswith("=X").all():
        raise UniversePreflightError("FX symbols must be explicit Yahoo =X pairs")
    required_ccy = set(u["currency"]) - {"USD"}
    if missing := required_ccy - set(fx["currency"]):
        raise UniversePreflightError(f"Missing FX mapping for {sorted(missing)}")
    if check_calendars:
        import pandas_market_calendars as mcal

        for cid in u["calendar_id"].unique():
            try:
                mcal.get_calendar(cid)
            except (ValueError, KeyError, RuntimeError) as exc:
                raise UniversePreflightError(f"Unknown exchange calendar: {cid}") from exc
    return u.sort_values("asset_id").reset_index(drop=True), fx.loc[fx["currency"].isin(required_ccy)].sort_values(
        "currency"
    ).reset_index(drop=True)


def make_cohorts(u: pd.DataFrame, *, cohort_size: int) -> list[pd.DataFrame]:
    if cohort_size < 1 or cohort_size > 100:
        raise UniversePreflightError("Cohort size must be between 1 and 100")
    return [u.iloc[i : i + cohort_size].copy() for i in range(0, len(u), cohort_size)]


def prepare(*, universe_csv: Path, fx_csv: Path, output: Path, cohort_size: int = 25) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight output: {output}")
    universe = pd.read_csv(universe_csv, dtype={"asset_id": "string"})
    fx_config = pd.read_csv(fx_csv)
    u, fx = prepare_registry(universe, fx_config)
    cohorts = make_cohorts(u, cohort_size=cohort_size)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for i, cohort in enumerate(cohorts, start=1):
        name = f"cohort_{i:04d}.csv"
        cohort.to_csv(output / name, index=False)
        files[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    fx.to_csv(output / "fx_config.csv", index=False)
    files["fx_config.csv"] = hashlib.sha256((output / "fx_config.csv").read_bytes()).hexdigest()
    manifest = {
        "schema": "alpha_edge_lake_v2_universe_preflight_1",
        "universe_csv_sha256": hashlib.sha256(universe_csv.read_bytes()).hexdigest(),
        "fx_source_csv_sha256": hashlib.sha256(fx_csv.read_bytes()).hexdigest(),
        "assets_included": len(u),
        "cohorts": len(cohorts),
        "cohort_size": cohort_size,
        "currencies": sorted(u["currency"].unique().tolist()),
        "fx_symbols": fx["provider_symbol"].tolist(),
        "files_sha256": files,
        "status": "configuration_pass_not_data_acceptance",
        "s3_writes": 0,
        "production_eligible": False,
        "limitations": (
            "Valid configuration does not establish FX observation timing, "
            "historical price accuracy, corporate-action factors or source "
            "provider capacity. Cohorts are plans, not ingestion runs."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-csv", type=Path, required=True)
    parser.add_argument("--fx-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-size", type=int, default=25)
    a = parser.parse_args()
    print(
        json.dumps(
            prepare(universe_csv=a.universe_csv, fx_csv=a.fx_csv, output=a.output, cohort_size=a.cohort_size), indent=2
        )
    )


if __name__ == "__main__":
    main()
