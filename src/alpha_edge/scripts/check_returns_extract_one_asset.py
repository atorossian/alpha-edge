# scripts/check_returns_extract_one_asset.py
from __future__ import annotations

import pandas as pd

from alpha_edge.core.runtime import load_runtime_config
from alpha_edge.core.market_store import MarketStore
from alpha_edge.market.build_returns_wide_cache import _extract_return_series, _qualifies_full_history

asset_id = "EQH285da8d0d2be0514"

cfg = load_runtime_config("prod")
store = MarketStore(
    bucket=cfg.bucket,
    region=cfg.region,
    base_prefix=cfg.market_root,
)

df = store.read_returns_usd(
    asset_ids=[asset_id],
    start="2010-01-01",
    end="2026-06-21",
    columns=[
        "date",
        "asset_id",
        "ticker",
        "ret_log_close_adjusted_usd",
        "ret_close_adjusted_usd",
        "ret_adj_close_usd",
    ],
)

print("\n=== RAW READ_RETURNS_USD ===")
print("shape:", None if df is None else df.shape)
print("columns:", [] if df is None else list(df.columns))

if df is None or df.empty:
    raise SystemExit("EMPTY read_returns_usd result")

df = df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()

print("date min:", df["date"].min())
print("date max:", df["date"].max())
print("rows by year:")
print(df.groupby(df["date"].dt.year).size().to_string())

print("\n=== COLUMN COVERAGE ===")
for c in ["ret_log_close_adjusted_usd", "ret_close_adjusted_usd", "ret_adj_close_usd"]:
    if c not in df.columns:
        print(c, "MISSING")
        continue

    s = pd.to_numeric(df[c], errors="coerce")
    valid_dates = df.loc[s.notna(), "date"]

    print()
    print(c)
    print("non_null:", int(s.notna().sum()))
    print("first:", valid_dates.min())
    print("last: ", valid_dates.max())

print("\n=== DUPLICATE DATE CHECK ===")
dups = df[df.duplicated(subset=["date"], keep=False)].copy()
print("duplicate rows:", len(dups))
if not dups.empty:
    print("duplicate dates:", dups["date"].nunique())
    print(dups.sort_values("date").tail(30).to_string(index=False))

print("\n=== EXTRACTED SERIES ===")
series = _extract_return_series(
    df=df,
    asset_id=asset_id,
    start_ts=pd.Timestamp("2010-01-01"),
    end_ts=pd.Timestamp("2026-06-21"),
    dtype="float32",
    strict_window=True,
)

print("nobs:", int(series.notna().sum()))
print("first:", series.index.min())
print("last: ", series.index.max())
print("span_days:", int((series.index.max() - series.index.min()).days))
print("attrs:", dict(series.attrs))

print("\n=== QUALIFICATION ===")
print(
    _qualifies_full_history(
        series=series,
        min_years=5.0,
        min_obs=252 * 5,
    )
)