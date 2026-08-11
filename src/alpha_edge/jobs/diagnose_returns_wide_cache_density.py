from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from alpha_edge import paths
from alpha_edge.core.runtime import load_runtime_config


def _norm_str(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def _load_universe(universe_csv: str | None) -> pd.DataFrame:
    path = universe_csv or str(paths.universe_dir() / "universe.csv")
    u = pd.read_csv(path)
    u = u.copy()

    if "asset_id" not in u.columns:
        raise RuntimeError(f"Universe missing asset_id: {path}")

    if "include" in u.columns:
        u["include"] = pd.to_numeric(u["include"], errors="coerce").fillna(1).astype(int)
    else:
        u["include"] = 1

    for col in ["asset_id", "ticker", "yahoo_ticker", "yahoo_ticker_norm", "name", "asset_class", "region"]:
        if col not in u.columns:
            u[col] = ""

    u["asset_id"] = u["asset_id"].map(_norm_str)
    u["ticker"] = u["ticker"].map(_norm_str)
    u["yahoo_ticker"] = u["yahoo_ticker"].map(_norm_str)
    u["yahoo_ticker_norm"] = u["yahoo_ticker_norm"].map(_norm_str)
    u["name"] = u["name"].map(_norm_str)
    u["asset_class"] = u["asset_class"].map(_norm_str)
    u["region"] = u["region"].map(_norm_str)

    return u[u["include"] == 1].copy()


def _asset_display_map(u: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for _, r in u.iterrows():
        aid = str(r.get("asset_id", "")).strip()
        if not aid:
            continue

        y_norm = str(r.get("yahoo_ticker_norm", "") or "").strip()
        y = str(r.get("yahoo_ticker", "") or "").strip()
        t = str(r.get("ticker", "") or "").strip()
        name = str(r.get("name", "") or "").strip()

        out[aid] = {
            "ticker": t,
            "yahoo_ticker": y,
            "yahoo_ticker_norm": y_norm,
            "display_symbol": y_norm or y or t or aid,
            "name": name,
            "asset_class": str(r.get("asset_class", "") or "").strip(),
            "region": str(r.get("region", "") or "").strip(),
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose returns_wide cache density and cleaning behavior.")
    ap.add_argument("--env", default=None)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--start", default=None, help="Optional explicit start date. If omitted, uses --years.")
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--cache-uri", default=None)
    ap.add_argument("--universe-csv", default=None)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    cfg = load_runtime_config(args.env)
    bucket = str(cfg.bucket)
    market_root = str(cfg.market_root).strip("/")

    cache_uri = args.cache_uri or f"s3://{bucket}/{market_root}/cache/v1/returns_wide_min5y.parquet"

    as_of_ts = pd.Timestamp(args.as_of).tz_localize(None).normalize()
    if args.start:
        start_ts = pd.Timestamp(args.start).tz_localize(None).normalize()
    else:
        start_ts = as_of_ts - pd.Timedelta(days=int(float(args.years) * 365.25))

    print("\n=== RETURNS WIDE CACHE DENSITY DIAGNOSTIC ===")
    print(f"env:       {args.env}")
    print(f"cache:     {cache_uri}")
    print(f"window:    {start_ts.date()}..{as_of_ts.date()}")

    rw = pd.read_parquet(cache_uri, engine="pyarrow").sort_index()
    rw.index = pd.to_datetime(rw.index, errors="coerce", utc=True).tz_convert(None).normalize()
    rw = rw.loc[~rw.index.isna()].copy()
    rw.columns = [str(c).strip() for c in rw.columns]
    rw = rw.loc[:, ~pd.Index(rw.columns).duplicated(keep="last")]
    rw = rw.loc[(rw.index >= start_ts) & (rw.index <= as_of_ts)].copy()
    rw = rw.sort_index()

    if rw.empty:
        raise RuntimeError("No returns_wide rows in requested window.")

    u = _load_universe(args.universe_csv)
    meta = _asset_display_map(u)

    total_rows = int(len(rw))
    weekend_rows = int((pd.Series(rw.index).dt.weekday >= 5).sum())
    weekday_rows = int(total_rows - weekend_rows)
    all_nan_rows = int(rw.isna().all(axis=1).sum())
    any_non_null_rows = int((~rw.isna().all(axis=1)).sum())

    print("\nIndex shape")
    print(f"  rows_total:       {total_rows}")
    print(f"  rows_weekday:     {weekday_rows}")
    print(f"  rows_weekend:     {weekend_rows}")
    print(f"  rows_all_nan:     {all_nan_rows}")
    print(f"  rows_any_nonnull: {any_non_null_rows}")
    print(f"  assets_total:     {rw.shape[1]}")

    nn_total = rw.notna().sum(axis=0).astype(int)
    nn_weekday = rw.loc[pd.Series(rw.index, index=rw.index).dt.weekday < 5].notna().sum(axis=0).astype(int)
    nn_weekend = rw.loc[pd.Series(rw.index, index=rw.index).dt.weekday >= 5].notna().sum(axis=0).astype(int)

    rows = []
    for aid in rw.columns:
        aid_s = str(aid).strip()
        m = meta.get(aid_s, {})
        n_total = int(nn_total.get(aid_s, 0))
        n_weekday = int(nn_weekday.get(aid_s, 0))
        n_weekend = int(nn_weekend.get(aid_s, 0))

        nan_frac_total = 1.0 - (n_total / total_rows) if total_rows else np.nan
        nan_frac_weekday = 1.0 - (n_weekday / weekday_rows) if weekday_rows else np.nan

        max_abs = rw[aid_s].abs().max(skipna=True)
        max_abs = float(max_abs) if pd.notna(max_abs) and np.isfinite(max_abs) else np.nan

        rows.append(
            {
                "asset_id": aid_s,
                "display_symbol": m.get("display_symbol", aid_s),
                "ticker": m.get("ticker", ""),
                "yahoo_ticker_norm": m.get("yahoo_ticker_norm", ""),
                "name": m.get("name", ""),
                "asset_class": m.get("asset_class", ""),
                "region": m.get("region", ""),
                "n_total": n_total,
                "n_weekday": n_weekday,
                "n_weekend": n_weekend,
                "nan_frac_total_calendar": float(nan_frac_total),
                "nan_frac_weekday_only": float(nan_frac_weekday),
                "max_abs_return": max_abs,
                "passes_current_25pct_calendar_nan": bool(nan_frac_total <= 0.25 and n_total >= 504),
                "passes_40pct_calendar_nan": bool(nan_frac_total <= 0.40 and n_total >= 756),
                "passes_weekday_25pct_nan": bool(nan_frac_weekday <= 0.25 and n_weekday >= 756),
                "passes_min_1000_obs": bool(n_total >= 1000),
                "passes_min_1260_obs": bool(n_total >= 1260),
            }
        )

    out = pd.DataFrame(rows)

    print("\nEligibility counts")
    for col in [
        "passes_current_25pct_calendar_nan",
        "passes_40pct_calendar_nan",
        "passes_weekday_25pct_nan",
        "passes_min_1000_obs",
        "passes_min_1260_obs",
    ]:
        print(f"  {col:<36} {int(out[col].sum())}")

    print("\nBy asset_class")
    if "asset_class" in out.columns:
        summary = (
            out.groupby("asset_class", dropna=False)
            .agg(
                assets=("asset_id", "count"),
                median_n_total=("n_total", "median"),
                median_nan_frac_calendar=("nan_frac_total_calendar", "median"),
                median_nan_frac_weekday=("nan_frac_weekday_only", "median"),
                pass_current=("passes_current_25pct_calendar_nan", "sum"),
                pass_weekday=("passes_weekday_25pct_nan", "sum"),
            )
            .sort_values("assets", ascending=False)
        )
        print(summary.to_string())

    print(f"\nWorst {int(args.top)} by calendar nan fraction")
    cols = [
        "display_symbol",
        "asset_id",
        "asset_class",
        "n_total",
        "n_weekday",
        "n_weekend",
        "nan_frac_total_calendar",
        "nan_frac_weekday_only",
        "max_abs_return",
    ]
    print(out.sort_values("nan_frac_total_calendar", ascending=False)[cols].head(int(args.top)).to_string(index=False))

    print(f"\nBest {int(args.top)} by calendar nan fraction")
    print(out.sort_values("nan_frac_total_calendar", ascending=True)[cols].head(int(args.top)).to_string(index=False))

    if args.output_csv:
        out.to_csv(args.output_csv, index=False)
        print(f"\n[OK] wrote: {args.output_csv}")


if __name__ == "__main__":
    main()