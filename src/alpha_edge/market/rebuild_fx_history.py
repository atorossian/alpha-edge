"""AEA-87 read-only Yahoo FX history refresh to local staging; no S3 writes.

Usage: poetry run python alpha_edge/market/rebuild_fx_history.py --currencies MXN,EUR,GBP --start 2010-01-01 --out-dir ../data/quality/fx/staging
Optional: --currencies-csv FILE (CSV with currency / ccy / quote_currency column).
"""
from __future__ import annotations
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf


def download(ticker: str, start: str, end: str | None) -> pd.Series:
    kwargs = dict(start=start, interval="1d", progress=False, threads=False, auto_adjust=False)
    if end: kwargs["end"] = end
    frame = yf.download(ticker, **kwargs)
    if frame is None or frame.empty:
        return pd.Series(dtype="float64", name=ticker)
    series = frame["Close"]
    if isinstance(series, pd.DataFrame):
        if ticker in series.columns:
            series = series[ticker]
        elif series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            raise ValueError(f"Ambiguous Close columns for {ticker}: {list(series.columns)}")
    series.index = pd.to_datetime(series.index, utc=True).tz_convert(None).normalize()
    series = pd.to_numeric(series, errors="coerce").sort_index()
    if series.index.has_duplicates:
        raise ValueError(f"Duplicate FX dates: {ticker}")
    return series.rename(ticker)


def refresh(currency: str, start: str, end: str | None, tolerance: float, max_daily_change: float) -> pd.DataFrame:
    direct_symbol, inverse_symbol = f"{currency}USD=X", f"USD{currency}=X"
    direct = download(direct_symbol, start, end)
    inverse_raw = download(inverse_symbol, start, end)
    inverse = 1.0 / inverse_raw.where(inverse_raw > 0)
    dates = direct.index.union(inverse.index).sort_values()
    result = pd.DataFrame(index=dates)
    result["direct_quote"] = direct.reindex(dates)
    result["inverse_quote"] = inverse.reindex(dates)
    d, i = result["direct_quote"], result["inverse_quote"]
    d_ok = np.isfinite(d) & (d > 0)
    i_ok = np.isfinite(i) & (i > 0)
    both = d_ok & i_ok
    result["relative_difference"] = (d - i).abs() / ((d + i) / 2)
    # Only accept observations with two independent quote directions.
    agreement = both & (result["relative_difference"] <= tolerance)
    result["fx_to_usd"] = ((d + i) / 2).where(agreement)
    result["validation_status"] = np.where(~both, "MISSING_QUOTE", np.where(agreement, "VALID", "QUOTE_DISAGREEMENT"))
    valid = result["fx_to_usd"].dropna()
    # Check adjacent FX quote dates, not the gap between non-trading days.
    jumps = valid.pct_change().abs() > max_daily_change
    if jumps.any():
        result.loc[valid.index[jumps], "validation_status"] = "LARGE_FX_CHANGE"
        result.loc[valid.index[jumps], "fx_to_usd"] = np.nan
    result.insert(0, "date", result.index)
    result.insert(1, "currency", currency)
    result["direct_symbol"] = direct_symbol
    result["inverse_symbol"] = inverse_symbol
    result["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result.reset_index(drop=True)


def parse_currencies(args: argparse.Namespace) -> list[str]:
    parts = [p.strip().upper() for p in (args.currencies or "").split(",") if p.strip()]
    if args.currencies_csv:
        frame = pd.read_csv(args.currencies_csv)
        column = next((c for c in ("currency", "ccy", "quote_currency") if c in frame.columns), None)
        if column is None:
            raise ValueError("Currency CSV needs column currency, ccy, or quote_currency")
        parts.extend(frame[column].dropna().astype(str).str.strip().str.upper().tolist())
    currencies = sorted({x for x in parts if x != "USD"})
    if not currencies or any(not re.fullmatch(r"[A-Z]{3}", x) for x in currencies):
        raise ValueError("Provide 3-letter ISO currencies with --currencies or --currencies-csv")
    return currencies


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--currencies", default="", help="Comma-separated currencies, e.g. MXN,EUR,GBP")
    ap.add_argument("--currencies-csv", help="CSV with currency/ccy/quote_currency column")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=None, help="Exclusive end date (YYYY-MM-DD)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reciprocal-tolerance", type=float, default=0.02)
    ap.add_argument("--max-daily-change", type=float, default=0.20)
    args = ap.parse_args()
    currencies = parse_currencies(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs, reports = [], []
    for ccy in currencies:
        print(f"Downloading {ccy}...")
        try:
            df = refresh(ccy, args.start, args.end, args.reciprocal_tolerance, args.max_daily_change)
            df.to_parquet(out / f"{ccy}_fx_history.parquet", index=False)
            counts = df["validation_status"].value_counts().to_dict()
            reports.append({"currency": ccy, "rows": len(df), "statuses": counts})
            outputs.append(df)
            print(ccy, counts)
        except Exception as e:
            reports.append({"currency": ccy, "error": str(e)})
            print(f"ERROR {ccy}: {e}")
    (out / "validation_report.json").write_text(json.dumps({"start": args.start, "end": args.end, "currencies": reports}, indent=2))
    if outputs:
        pd.concat(outputs, ignore_index=True).to_parquet(out / "fx_history_all.parquet", index=False)
    failed = any("error" in r or r.get("statuses", {}).get("QUOTE_DISAGREEMENT", 0) or r.get("statuses", {}).get("LARGE_FX_CHANGE", 0) or r.get("rows",0)==0 for r in reports)
    print("OUTPUT:", out.resolve())
    if failed:
        raise SystemExit("FX staging completed with validation exceptions; review validation_report.json before any promotion")


if __name__ == "__main__":
    main()
