"""Read-only AEA-87 triage over three user-provided files. Run under project Poetry environment.

python analyze_root_causes.py --prices price_history_extracted.parquet --cells returns_quality_cells.csv --fx fx_history_all.parquet --out DIR
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def normalize_dates(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    return df


def main():
    ap = argparse.ArgumentParser()
    for x in ("prices", "cells", "fx", "out"):
        ap.add_argument("--" + x, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = normalize_dates(pd.read_parquet(args.prices))
    c = normalize_dates(pd.read_csv(args.cells))
    f = normalize_dates(pd.read_parquet(args.fx))
    for x in (p, c):
        x["asset_id"] = x["asset_id"].astype(str).str.strip()
    p = p.sort_values(["asset_id", "date"]).copy()
    if p.duplicated(["asset_id", "date"]).any():
        raise ValueError("Duplicate asset_id/date prices")
    f["currency"] = f["currency"].astype(str).str.upper().str.strip()
    if f.duplicated(["currency", "date"]).any():
        raise ValueError("Duplicate currency/date FX")
    # Restricted as-of alignment to prevent forward filling over known long gaps.
    f = f.sort_values(["currency", "date"])
    f["fx_date"] = f["date"]
    f["fx_valid"] = f["validation_status"].isin(["VALID", "VALID_AFTER_GAP"])
    rates = f[["currency", "date", "fx_to_usd", "fx_date", "fx_valid", "validation_status"]].rename(
        columns={"fx_to_usd": "new_fx", "validation_status": "fx_validation_status"}
    )
    for col in ("close", "adj_close", "fx_to_usd", "close_adjusted_usd"):
        if col not in p.columns:
            raise ValueError(f"Price extract missing {col}")
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p["currency"] = (
        p.get("currency", pd.Series(index=p.index, dtype="object")).fillna("USD").astype(str).str.upper().str.strip()
    )
    nonusd = p[p.currency.ne("USD")].copy()
    usd = p[p.currency.eq("USD")].copy()
    if len(nonusd):
        # exact-date joins first; forex date may differ across market holiday schedules.
        nonusd = nonusd.merge(
            rates[["currency", "date", "new_fx", "fx_validation_status"]],
            on=["currency", "date"],
            how="left",
            validate="many_to_one",
        )
        nonusd["fx_source_date"] = nonusd["date"].where(nonusd.new_fx.notna())
        # No blind ffill; missing dates are unrepairable until explicit freshness policy.
    else:
        nonusd["new_fx"] = pd.Series(dtype=float)
        nonusd["fx_validation_status"] = pd.Series(dtype=str)
    usd["new_fx"] = 1.0
    usd["fx_validation_status"] = "USD_NOT_APPLICABLE"
    usd["fx_source_date"] = usd.date
    allp = pd.concat([nonusd, usd], ignore_index=True).sort_values(["asset_id", "date"])
    allp["stored_usd"] = allp["close_adjusted_usd"]
    allp["recomputed_usd"] = allp["adj_close"] * allp["new_fx"]
    allp["stored_fx_ratio"] = allp["fx_to_usd"] / allp["new_fx"]
    allp["bad_fx_level"] = (
        allp.currency.ne("USD") & allp["new_fx"].notna() & (allp.stored_fx_ratio.lt(0.9) | allp.stored_fx_ratio.gt(1.1))
    )
    for pricecol, prefix in [("stored_usd", "stored"), ("recomputed_usd", "recomputed"), ("adj_close", "local")]:
        prev = allp.groupby("asset_id")[pricecol].shift(1)
        allp[prefix + "_simple"] = allp[pricecol] / prev - 1
        allp[prefix + "_log"] = np.log(allp[pricecol] / prev)
    allp["large_local_move"] = allp["local_simple"].abs() > 0.5
    base_cols = [
        "asset_id",
        "date",
        "currency",
        "yahoo_ticker",
        "stored_usd",
        "recomputed_usd",
        "fx_to_usd",
        "new_fx",
        "fx_validation_status",
        "stored_fx_ratio",
        "bad_fx_level",
        "large_local_move",
        "stored_simple",
        "stored_log",
        "recomputed_simple",
        "recomputed_log",
        "local_simple",
    ]
    cols = [x for x in base_cols if x in allp.columns]
    merged = c.merge(allp[cols], on=["asset_id", "date"], how="left", validate="many_to_one", indicator=True)
    rv = pd.to_numeric(merged["return"], errors="coerce")
    ismatch = lambda a, b: np.isclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=False)
    merged["matches_stored_log"] = ismatch(rv, merged["stored_log"])
    merged["matches_stored_simple"] = ismatch(rv, merged["stored_simple"])
    merged["matches_recomputed_log"] = ismatch(rv, merged["recomputed_log"])
    merged["matches_recomputed_simple"] = ismatch(rv, merged["recomputed_simple"])
    merged["repaired_simple_still_extreme"] = merged.recomputed_simple.abs() > 0.5
    merged["classification"] = np.select(
        [
            merged["_merge"].eq("left_only"),
            merged["bad_fx_level"].fillna(False)
            & ~merged["large_local_move"].fillna(False)
            & ~merged.repaired_simple_still_extreme.fillna(False),
            merged["bad_fx_level"].fillna(False),
            merged["large_local_move"].fillna(False),
            merged.matches_stored_log & ~merged.matches_stored_simple,
            merged.matches_stored_simple,
        ],
        [
            "OUTSIDE_EXTRACT_OR_MISSING_DATE",
            "FX_CORRUPTION_CORRECTS_EXTREME_RETURN",
            "FX_CORRUPTION_PLUS_OTHER_ISSUE",
            "LOCAL_SOURCE_PRICE_EXTREME",
            "LOG_IN_SIMPLE_CONVENTION",
            "STORED_SIMPLE_MOVE",
        ],
        default="UNRESOLVED_OR_PRICE_GAP",
    )
    merged.to_csv(out / "anomaly_diagnosis.csv", index=False)
    allp.loc[allp.bad_fx_level | allp.large_local_move].to_csv(out / "suspicious_source_rows.csv", index=False)
    summary = merged.classification.value_counts().rename_axis("classification").reset_index(name="count")
    summary.to_csv(out / "classification_summary.csv", index=False)
    print("\nTotal bad cells:", len(c), "unique assets:", c.asset_id.nunique())
    print("Extracted asset count:", p.asset_id.nunique())
    print("Mapped anomalous dates:", (merged["_merge"] == "both").sum())
    print(summary.to_string(index=False))
    print("\nBy currency (matched only):")
    print(merged[merged["_merge"].eq("both")].groupby("currency")["classification"].value_counts().to_string())
    print("\nAffected FX rows in full extracted prices:", int(allp.bad_fx_level.sum()))
    print("Source-local price moves >50%:", int(allp.large_local_move.sum()))
    print("\nSaved reports in", out)


if __name__ == "__main__":
    main()
