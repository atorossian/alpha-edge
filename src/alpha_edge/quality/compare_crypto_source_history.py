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
        pd.concat(details, ignore_index=True).to_csv(out / "date_comparison.csv", index=False)
    print("READ-ONLY comparison saved to", out.resolve())


if __name__ == "__main__":
    main()
