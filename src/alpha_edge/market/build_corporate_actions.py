from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Any

import pandas as pd
import yfinance as yf

from alpha_edge import paths
from alpha_edge.core.market_store import MarketStore


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"


@dataclass(frozen=True)
class CorporateActionRow:
    asset_id: str
    ticker: str
    yahoo_ticker: str
    effective_date: str
    action_type: str
    split_factor: float
    source: str
    source_action_id: str | None
    detected_at_utc: str
    notes: str | None


def _now_utc_iso() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_yahoo_symbol(sym: str) -> str:
    return str(sym or "").strip().upper()


def _load_universe(universe_path: str) -> pd.DataFrame:
    u = pd.read_csv(universe_path)

    if "asset_id" not in u.columns:
        raise RuntimeError("Universe CSV must include 'asset_id'.")
    if "ticker" not in u.columns and "broker_ticker" not in u.columns:
        raise RuntimeError("Universe CSV must include 'ticker' or 'broker_ticker'.")

    u = u.copy()
    u["asset_id"] = u["asset_id"].astype(str).str.strip()

    if "ticker" in u.columns:
        u["ticker"] = u["ticker"].astype(str).str.upper().str.strip()
    else:
        u["ticker"] = u["broker_ticker"].astype(str).str.upper().str.strip()

    if "yahoo_ticker" in u.columns:
        u["yahoo_ticker"] = u["yahoo_ticker"].astype(str).str.strip()
        u["yahoo_ticker"] = u["yahoo_ticker"].replace({"": None, "NAN": None, "None": None})
    else:
        u["yahoo_ticker"] = None

    u["yahoo_ticker_norm"] = u["yahoo_ticker"].fillna(u["ticker"]).astype(str).map(_normalize_yahoo_symbol)

    if "include" in u.columns:
        u["include"] = pd.to_numeric(u["include"], errors="coerce").fillna(0).astype(int)
        u = u[u["include"] == 1].copy()

    u = u.drop_duplicates(subset=["asset_id"], keep="last").reset_index(drop=True)
    return u


def _fetch_splits_for_symbol(yahoo_ticker: str) -> pd.Series:
    t = yf.Ticker(str(yahoo_ticker).strip())
    s = t.splits

    if s is None:
        return pd.Series(dtype="float64")

    if not isinstance(s, pd.Series):
        try:
            s = pd.Series(s)
        except Exception:
            return pd.Series(dtype="float64")

    if s.empty:
        return pd.Series(dtype="float64")

    s = s.dropna().astype("float64")
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()
    return s


def _build_rows_for_asset(asset_id: str, ticker: str, yahoo_ticker: str) -> list[dict]:
    detected_at = _now_utc_iso()
    rows: list[dict] = []

    try:
        splits = _fetch_splits_for_symbol(yahoo_ticker)
    except Exception as e:
        return [
            {
                "asset_id": str(asset_id).strip(),
                "ticker": str(ticker).upper().strip(),
                "yahoo_ticker": str(yahoo_ticker).strip(),
                "effective_date": None,
                "action_type": "ERROR",
                "split_factor": None,
                "source": "yfinance",
                "source_action_id": None,
                "detected_at_utc": detected_at,
                "notes": f"split_fetch_error: {type(e).__name__}: {e}",
            }
        ]

    for idx, factor in splits.items():
        eff = pd.Timestamp(idx).date().isoformat()
        factor_f = float(factor)

        # yfinance split ratios are suitable for the same convention you are already
        # using in rebuild_ledger: if trade_date < effective_date, qty *= factor, px /= factor
        rows.append(
            asdict_safe(
                CorporateActionRow(
                    asset_id=str(asset_id).strip(),
                    ticker=str(ticker).upper().strip(),
                    yahoo_ticker=str(yahoo_ticker).strip(),
                    effective_date=eff,
                    action_type="SPLIT",
                    split_factor=factor_f,
                    source="yfinance",
                    source_action_id=f"{str(yahoo_ticker).strip()}::{eff}::SPLIT",
                    detected_at_utc=detected_at,
                    notes=None,
                )
            )
        )

    return rows


def asdict_safe(obj: Any) -> dict:
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    raise TypeError(f"Cannot convert object to dict: {type(obj)}")


def build_corporate_actions_df(
    *,
    universe_path: str,
    asset_id: str | None = None,
    ticker: str | None = None,
    yahoo_ticker: str | None = None,
) -> pd.DataFrame:
    u = _load_universe(universe_path)

    if asset_id:
        u = u[u["asset_id"] == str(asset_id).strip()].copy()

    if ticker:
        t = str(ticker).upper().strip()
        u = u[u["ticker"] == t].copy()

    if yahoo_ticker:
        yt = str(yahoo_ticker).strip().upper()
        u = u[u["yahoo_ticker_norm"] == yt].copy()

    rows: list[dict] = []
    for _, r in u.iterrows():
        rows.extend(
            _build_rows_for_asset(
                asset_id=str(r["asset_id"]).strip(),
                ticker=str(r["ticker"]).upper().strip(),
                yahoo_ticker=str(r["yahoo_ticker_norm"]).strip(),
            )
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "asset_id",
                "ticker",
                "yahoo_ticker",
                "effective_date",
                "action_type",
                "split_factor",
                "source",
                "source_action_id",
                "detected_at_utc",
                "notes",
            ]
        )

    # Keep only real corporate actions in the canonical table
    df = df[df["action_type"] == "SPLIT"].copy()

    if not df.empty:
        df["asset_id"] = df["asset_id"].astype(str).str.strip()
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["yahoo_ticker"] = df["yahoo_ticker"].astype(str).str.strip()
        df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce").dt.date
        df["split_factor"] = pd.to_numeric(df["split_factor"], errors="coerce")
        df = df.dropna(subset=["asset_id", "ticker", "effective_date", "split_factor"])
        df = df.drop_duplicates(subset=["asset_id", "effective_date", "action_type"], keep="last")
        df = df.sort_values(["asset_id", "effective_date"], kind="stable").reset_index(drop=True)

    return df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build corporate actions parquet storage from Yahoo splits.")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--universe-path", default=str(paths.universe_dir() / "universe.csv"))

    ap.add_argument("--asset-id", default=None)
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--yahoo-ticker", default=None)

    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    store = MarketStore(bucket=str(args.bucket), region=str(args.region))

    df = build_corporate_actions_df(
        universe_path=str(args.universe_path),
        asset_id=args.asset_id,
        ticker=args.ticker,
        yahoo_ticker=args.yahoo_ticker,
    )

    print("\n=== BUILD CORPORATE ACTIONS ===")
    print(f"rows={len(df)}")
    print(f"bucket={args.bucket}")
    print(f"prefix={store.corporate_actions_prefix}")

    if df.empty:
        print("[OK] no corporate actions found.")
        return

    if args.dry_run:
        sample = df.head(20).copy()
        print(sample.to_string(index=False))
        return

    written = store.write_corporate_actions_partitioned(df)
    print(f"[OK] wrote partitions={len(written)}")
    for k in written[:20]:
        print(f"  s3://{args.bucket}/{k}")
    if len(written) > 20:
        print(f"  ... ({len(written) - 20} more)")


if __name__ == "__main__":
    main()