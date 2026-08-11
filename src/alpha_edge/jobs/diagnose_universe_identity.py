from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alpha_edge import paths


def _norm_str(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def _norm_ticker(x: object) -> str:
    return _norm_str(x).upper()


def diagnose_universe_identity(
    *,
    universe_csv: str | None,
    output_csv: str | None,
    active_only: bool,
) -> pd.DataFrame:
    path = Path(universe_csv) if universe_csv else paths.universe_dir() / "universe.csv"

    df = pd.read_csv(path)
    if df is None or df.empty:
        raise RuntimeError(f"Universe is empty: {path}")

    required = ["asset_id", "ticker"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Universe missing required column {col!r}: {path}")

    out = df.copy()
    out["asset_id_norm"] = out["asset_id"].map(_norm_str)
    out["ticker_norm"] = out["ticker"].map(_norm_ticker)

    if "yahoo_ticker" not in out.columns:
        out["yahoo_ticker"] = out["ticker_norm"]
    out["yahoo_ticker_norm"] = out["yahoo_ticker"].map(_norm_str)

    if "broker_ticker" not in out.columns:
        out["broker_ticker"] = out["ticker_norm"]
    out["broker_ticker_norm"] = out["broker_ticker"].map(_norm_str)

    if "name" not in out.columns:
        out["name"] = out["ticker_norm"]
    out["name_norm"] = out["name"].map(_norm_str)

    if "asset_class" not in out.columns:
        out["asset_class"] = "unknown"

    if "region" not in out.columns:
        out["region"] = "unknown"

    if "include" in out.columns:
        out["include_norm"] = pd.to_numeric(out["include"], errors="coerce").fillna(1).astype(int)
    else:
        out["include_norm"] = 1

    if active_only:
        scope = out[out["include_norm"] == 1].copy()
    else:
        scope = out.copy()

    duplicate_asset = scope[scope["asset_id_norm"].duplicated(keep=False)].copy()
    duplicate_ticker = scope[scope["ticker_norm"].duplicated(keep=False)].copy()
    duplicate_yahoo = scope[scope["yahoo_ticker_norm"].duplicated(keep=False)].copy()

    cols = [
        "ticker_norm",
        "asset_id_norm",
        "yahoo_ticker_norm",
        "broker_ticker_norm",
        "name_norm",
        "asset_class",
        "region",
        "include_norm",
    ]

    duplicate_ticker = duplicate_ticker.sort_values(["ticker_norm", "yahoo_ticker_norm", "asset_id_norm"])

    print("\n=== UNIVERSE IDENTITY DIAGNOSTIC ===")
    print(f"universe_csv:              {path}")
    print(f"active_only:               {active_only}")
    print(f"rows_checked:              {len(scope)}")
    print(f"duplicate_asset_id_groups: {duplicate_asset['asset_id_norm'].nunique() if not duplicate_asset.empty else 0}")
    print(f"duplicate_ticker_groups:   {duplicate_ticker['ticker_norm'].nunique() if not duplicate_ticker.empty else 0}")
    print(f"duplicate_yahoo_groups:    {duplicate_yahoo['yahoo_ticker_norm'].nunique() if not duplicate_yahoo.empty else 0}")

    if not duplicate_ticker.empty:
        print("\nDuplicate active ticker rows:")
        print(duplicate_ticker[cols].to_string(index=False))
    else:
        print("\nNo duplicate active ticker rows found.")

    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duplicate_ticker[cols].to_csv(output_path, index=False)
        print(f"\n[OK] wrote duplicate ticker report: {output_path}")

    return duplicate_ticker[cols]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diagnose duplicate identifiers in Alpha Edge universe.csv.")
    ap.add_argument("--universe-csv", default=None)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--all", action="store_true", help="Check all rows, not just include=1.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    diagnose_universe_identity(
        universe_csv=args.universe_csv,
        output_csv=args.output_csv,
        active_only=(not bool(args.all)),
    )


if __name__ == "__main__":
    main()