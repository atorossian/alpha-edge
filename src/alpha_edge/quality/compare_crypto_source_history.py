"""AEA-87 read-only comparison of selected crypto S3 OHLCV against fresh Yahoo closes.

Run from src/: poetry run python -m alpha_edge.quality.compare_crypto_source_history
Never writes to S3. A matching Yahoo price is not proof that a source price is correct.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

from alpha_edge.core.market_store import MarketStore


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="../data/universe/universe.csv")
    ap.add_argument("--tickers", default="DOGE-USD,HYPE-USD,FIL-USD,SOL-USD,ADA-USD")
    ap.add_argument("--start", default="2026-07-01")
    ap.add_argument("--end", default="2026-09-15", help="Exclusive end for Yahoo")
    ap.add_argument("--bucket", default="alpha-edge-algo")
    ap.add_argument("--region", default="eu-west-1")
    ap.add_argument("--out", default="../data/quality/returns/crypto_source_comparison")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(args.universe).fillna("")
    if "asset_id" not in universe or "yahoo_ticker" not in universe:
        raise ValueError("Universe must include asset_id and yahoo_ticker")
    symbol = universe["yahoo_ticker"].astype(str).str.strip().str.upper()
    ticker = universe["ticker"].astype(str).str.strip().str.upper() if "ticker" in universe else symbol
    wanted = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    store = MarketStore(bucket=args.bucket, region=args.region)
    details = []
    summaries = []
    staged_bars = []
    for requested in wanted:
        aliases = {requested, requested.replace("-", ""), requested.replace("/", "-")}
        rows = universe.loc[symbol.isin(aliases) | ticker.isin(aliases)].copy()
        if len(rows) != 1:
            summaries.append({"requested": requested, "status": "AMBIGUOUS_OR_MISSING_UNIVERSE_MATCH", "matches": len(rows)})
            continue
        row = rows.iloc[0]
        aid = str(row["asset_id"]).strip()
        sym = str(row["yahoo_ticker"]).strip().upper().replace("/", "-")
        stored = store.read_ohlcv_usd([aid], start=args.start, end=args.end)
        if stored.empty:
            summaries.append({"requested": requested, "asset_id": aid, "status": "NO_STORED_ROWS"})
            continue
        stored = stored.copy()
        stored["date"] = pd.to_datetime(stored["date"], utc=True).dt.tz_convert(None).dt.normalize()
        stored["stored_close"] = pd.to_numeric(stored["close"], errors="coerce")
        stored["stored_adjusted_close"] = pd.to_numeric(stored.get("adj_close"), errors="coerce")
        stored = stored.sort_values("date")
        duplicate_count = int(stored.duplicated("date").sum())
        stored = stored.drop_duplicates("date", keep="last")
        raw = yf.download(sym, start=args.start, end=args.end, auto_adjust=False, progress=False, threads=False, interval="1d")
        if raw.empty:
            summaries.append({"requested": requested, "asset_id": aid, "yahoo_ticker": sym, "status": "YAHOO_EMPTY"})
            continue
        fresh = raw["Close"]
        if isinstance(fresh, pd.DataFrame):
            fresh = fresh[sym] if sym in fresh.columns else fresh.iloc[:, 0]
        fresh = fresh.rename("fresh_close").to_frame()
        fresh.index = pd.to_datetime(fresh.index, utc=True).tz_convert(None).normalize()
        fresh.index.name = "date"
        fresh = fresh.reset_index()
        joined = stored.merge(fresh, on="date", how="outer", indicator=True)
        joined["asset_id"] = aid
        joined["requested"] = requested
        joined["yahoo_ticker_expected"] = sym
        joined["ratio_stored_to_yahoo"] = joined["stored_close"] / joined["fresh_close"]
        joined["relative_error"] = (joined["ratio_stored_to_yahoo"] - 1).abs()
        both = joined["_merge"].eq("both")
        different = both & joined["relative_error"].gt(0.01)
        details.append(joined)
        # Stage complete replacement OHLCV bars locally, never just a close
        # price patch. Rebuilding from entire bars preserves OHLC invariants.
        if different.any():
            fields = {"Open": "open", "High": "high", "Low": "low",
                      "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
            raw_bars = pd.DataFrame(index=raw.index)
            for yf_field, target in fields.items():
                source = raw[yf_field] if yf_field in raw.columns.get_level_values(0) else None
                if isinstance(source, pd.DataFrame):
                    if sym not in source.columns:
                        raise ValueError(f"{sym}: downloaded OHLCV symbol mismatch")
                    source = source[sym]
                if source is None:
                    raise ValueError(f"{sym}: missing {yf_field} for staged repair")
                raw_bars[target] = source
            raw_bars.index = pd.to_datetime(raw_bars.index, utc=True).tz_convert(None).normalize()
            raw_bars.index.name = "date"
            raw_bars = raw_bars.reset_index()
            repair_dates = set(joined.loc[different, "date"])
            patches = raw_bars.loc[raw_bars["date"].isin(repair_dates)].copy()
            if len(patches) != int(different.sum()):
                raise ValueError(f"{sym}: staged OHLCV row count does not match discrepancy dates")
            numeric = patches[["open", "high", "low", "close", "adj_close"]].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(numeric.to_numpy(dtype=float)).all() or (numeric <= 0).any().any():
                raise ValueError(f"{sym}: invalid staged Yahoo OHLCV bars")
            if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any() or (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
                raise ValueError(f"{sym}: inconsistent staged OHLC values")
            patches["asset_id"] = aid
            patches["yahoo_ticker"] = sym
            patches["currency"] = "USD"
            patches["fx_to_usd"] = 1.0
            patches["close_raw_usd"] = patches["close"]
            patches["close_adjusted_usd"] = patches["adj_close"]
            patches["close_usd"] = patches["close"]
            patches["adj_close_usd"] = patches["adj_close"]
            staged_bars.append(patches)
        summaries.append({
            "requested": requested, "asset_id": aid, "yahoo_ticker": sym, "status": "COMPARED",
            "s3_rows": int(len(stored)), "fresh_rows": int(len(fresh)),
            "overlap": int(both.sum()), "diff_over_1pct": int(different.sum()),
            "s3_only": int(joined["_merge"].eq("left_only").sum()),
            "yahoo_only": int(joined["_merge"].eq("right_only").sum()),
            "duplicate_s3_dates": duplicate_count,
            "max_relative_error": float(joined.loc[both, "relative_error"].max()) if both.any() else None,
        })
        print(summaries[-1])
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    if details:
        all_dates = pd.concat(details, ignore_index=True)
        all_dates.to_csv(out / "date_comparison.csv", index=False)
        # Stage price-level replacements as a LOCAL REVIEW PLAN, never as a write.
        # Only overlapping observations with a >1% discrepancy are candidates.
        bad = all_dates.loc[
            all_dates["_merge"].eq("both")
            & all_dates["relative_error"].gt(0.01)
            & all_dates["fresh_close"].notna()
            & np.isfinite(all_dates["fresh_close"])
            & all_dates["fresh_close"].gt(0),
            ["asset_id", "requested", "date", "stored_close", "fresh_close", "relative_error"]
        ].copy()
        bad = bad.rename(columns={"stored_close": "old_close", "fresh_close": "proposed_close"})
        bad["action"] = "REVIEW_ONLY_NO_S3_WRITE"
        bad.to_csv(out / "replacement_candidates.csv", index=False)
        print(f"Staged {len(bad)} read-only source-price replacement candidates")
    if staged_bars:
        patch = pd.concat(staged_bars, ignore_index=True)
        if patch.duplicated(["asset_id", "date"]).any():
            raise ValueError("Duplicate asset/date in local repair patches")
        patch.to_parquet(out / "replacement_ohlcv_bars_STAGING_ONLY.parquet", index=False)
        print(f"Staged {len(patch)} complete replacement OHLCV bars LOCALLY; no S3 writes")
        # Build full asset-year replacements rather than writing individual
        # affected dates: the canonical store writes one Parquet per asset-year.
        import hashlib
        import json
        year_reports = []
        for (asset_id, year), group in patch.groupby(["asset_id", patch["date"].dt.year]):
            year_start = f"{int(year)}-01-01"
            year_end = f"{int(year)}-12-31"
            original = store.read_ohlcv_usd([asset_id], start=year_start, end=year_end).copy()
            if original.empty:
                raise ValueError(f"{asset_id} {year}: S3 base year is empty")
            original["date"] = pd.to_datetime(original["date"], utc=True).dt.tz_convert(None).dt.normalize()
            if original.duplicated(["asset_id", "date"]).any():
                raise ValueError(f"{asset_id} {year}: duplicate historical dates; cannot safely patch")
            if not set(group["date"]).issubset(set(original["date"])):
                raise ValueError(f"{asset_id} {year}: patch contains dates absent from historical year")
            patched = original.set_index("date", drop=False)
            for _, bar in group.iterrows():
                date = bar["date"]
                old = patched.loc[date]
                if str(old["asset_id"]) != asset_id or str(old["currency"]).upper() != "USD":
                    raise ValueError(f"{asset_id} {date}: asset or currency mismatch")
                for col in ["open", "high", "low", "close", "adj_close", "volume",
                            "fx_to_usd", "close_raw_usd", "close_adjusted_usd",
                            "close_usd", "adj_close_usd"]:
                    if col in patched.columns:
                        patched.at[date, col] = bar[col]
            repaired = patched.reset_index(drop=True).sort_values("date")
            if len(repaired) != len(original) or repaired.duplicated(["asset_id", "date"]).any():
                raise ValueError(f"{asset_id} {year}: full-year preservation failed")
            unchanged = ~repaired["date"].isin(set(group["date"]))
            before = original.set_index("date").sort_index()
            after = repaired.set_index("date").sort_index()
            if not before.loc[~before.index.isin(group["date"])].equals(
                after.loc[~after.index.isin(group["date"])]
            ):
                raise ValueError(f"{asset_id} {year}: unrelated rows were changed")
            local_name = asset_id.replace(":", "_").replace("/", "_") + f"_{int(year)}.parquet"
            local_file = out / "full_year_replacements" / local_name
            local_file.parent.mkdir(parents=True, exist_ok=True)
            repaired.to_parquet(local_file, index=False)
            digest = hashlib.sha256(local_file.read_bytes()).hexdigest()
            year_reports.append({"asset_id": asset_id, "year": int(year),
                                 "source_rows": len(original), "output_rows": len(repaired),
                                 "patched_dates": len(group), "filename": str(local_file),
                                 "sha256": digest, "status": "LOCAL_STAGING_ONLY"})
        (out / "full_year_replacement_manifest.json").write_text(
            json.dumps(year_reports, indent=2, default=str), encoding="utf-8"
        )
        print(f"Prepared {len(year_reports)} complete asset-year replacement files locally")

    print("READ-ONLY comparison saved to", out.resolve())


if __name__ == "__main__":
    main()
