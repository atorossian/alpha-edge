from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alpha_edge.core.market_store import MarketStore
from engine_market_loader import load_ohlcv_usd_long, load_returns_usd_long, returns_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="alpha-edge-algo")
    ap.add_argument("--tickers", required=True, help="Comma-separated tickers (e.g. AAPL,MSFT,SPY)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--out", default="data_exports", help="output folder")
    ap.add_argument("--wide-returns", action="store_true", help="also export a wide returns matrix parquet")
    args = ap.parse_args()

    store = MarketStore(bucket=args.bucket)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    ohlcv = load_ohlcv_usd_long(store, tickers, start=args.start, end=args.end)
    rets = load_returns_usd_long(store, tickers, start=args.start, end=args.end)

    ohlcv_path = outdir / "ohlcv_usd_long.parquet"
    rets_path = outdir / "returns_usd_long.parquet"

    ohlcv.to_parquet(ohlcv_path, index=False)
    rets.to_parquet(rets_path, index=False)

    print(f"[OK] wrote {ohlcv_path} rows={len(ohlcv)}")
    print(f"[OK] wrote {rets_path} rows={len(rets)}")

    if args.wide_returns:
        wide = returns_matrix(rets)
        wide_path = outdir / "returns_usd_wide.parquet"
        wide.to_parquet(wide_path)
        print(f"[OK] wrote {wide_path} shape={wide.shape}")


if __name__ == "__main__":
    main()
