"""Inspect an asset across Bronze, Silver and Gold in one lake-v2 pilot run.

Example:
  python -m alpha_edge.market.inspect_data_lake_v2 \
    --run-dir outputs/market_data_lake_v2/RUN_ID --asset-id EQ:EXAMPLE
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def inspect(run_dir: Path, asset_id: str, *, export_csv: str | None = None) -> pd.DataFrame:
    paths = {
        "bronze": run_dir / "bronze" / "validated_ohlcv.parquet",
        "silver": run_dir / "silver" / "ohlcv_usd.parquet",
        "gold": run_dir / "gold" / "returns.parquet",
    }
    tables: dict[str, pd.DataFrame] = {}
    for layer, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        data = pd.read_parquet(path)
        data = data.loc[data["asset_id"].astype(str).eq(asset_id)].copy()
        tables[layer] = data
    bronze = tables["bronze"]
    silver = tables["silver"]
    gold = tables["gold"]
    if bronze.empty:
        raise ValueError(f"Asset not found: {asset_id}")
    # Use row ordinals for duplicate sessions; do not hide source duplicates.
    def stage(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
        out = df[["session_date", *[c for c in cols if c in df]]].copy()
        out["observation_sequence"] = out.groupby("session_date").cumcount()
        return out.rename(columns={
            c: f"{prefix}_{c}" for c in out.columns
            if c not in {"session_date", "observation_sequence"}
        })
    a = stage(bronze, ["close", "adj_close", "quality_status", "quality_reason"], "raw")
    b = stage(silver, ["fx_date", "fx_to_usd", "fx_age_days",
                       "close_raw_usd", "close_adjusted_usd", "quality_status",
                       "quality_reason"], "silver")
    c = stage(gold, ["ret_simple_close_adjusted_usd", "ret_log_close_adjusted_usd",
                     "return_quality_status"], "gold")
    keys = ["session_date", "observation_sequence"]
    joined = a.merge(b, on=keys, how="outer", validate="one_to_one")
    joined = joined.merge(c, on=keys, how="outer", validate="one_to_one")
    joined = joined.sort_values(keys).reset_index(drop=True)
    if export_csv:
        output = Path(export_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(output, index=False)
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--export-csv", default=None)
    args = parser.parse_args()
    report = inspect(args.run_dir, args.asset_id, export_csv=args.export_csv)
    with pd.option_context("display.max_rows", 150, "display.max_columns", None,
                           "display.width", 240):
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
