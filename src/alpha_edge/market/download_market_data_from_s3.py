from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.runtime import load_runtime_config
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.market.engine_market_loader import load_ohlcv_usd_long, load_returns_usd_long, returns_matrix


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--tickers", required=True, help="Comma-separated tickers (e.g. AAPL,MSFT,SPY)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("--out", default="data_exports", help="output folder")
    ap.add_argument("--wide-returns", action="store_true", help="also export a wide returns matrix parquet")
    return ap.parse_args()


def _main_impl(args: argparse.Namespace) -> None:
    cfg = load_runtime_config(args.env)
    bucket = args.bucket or cfg.bucket

    store = MarketStore(bucket=bucket)
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


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(args.env)

    with capture_script_run(
        cfg=cfg,
        script_name="market/download_market_data_from_s3.py",
        input_args=vars(args),
        dry_run=False,
    ):
        _main_impl(args)


if __name__ == "__main__":
    main()
