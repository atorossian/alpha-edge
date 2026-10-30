# ingest_market_data.py
from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run

import argparse
import threading
from typing import Optional, Any
from pathlib import Path
import hashlib
import os

from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import RuntimeConfig

from alpha_edge import paths

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from alpha_edge.core.market_store import MarketStore
from alpha_edge.jobs.run_universe_triage import run_post_ingest_triage


# ============================================================
# MARKET DATA SEMANTICS (PHASE 0)
#
# This ingestion writes a combined OHLCV dataset with both:
#   - raw / execution-consistent price fields
#   - adjusted / analytics-consistent price fields
#
# Canonical usage rules:
#   - close_raw_usd:
#       ledger, holdings valuation, account pnl, trade reconciliation
#   - close_adjusted_usd:
#       returns, optimizer, research, analytics, risk stats
#   - ret_close_adjusted_usd:
#       canonical return field for analytics
#
# Backward-compatibility aliases are still written temporarily:
#   - close_usd == close_raw_usd
#   - adj_close_usd == close_adjusted_usd
#   - ret_adj_close_usd == ret_close_adjusted_usd
#
# NOTE:
#   This Phase 0 version preserves:
#     - incremental ingestion
#     - yearly manifests
#     - snapshots
#     - retries / caching / FX handling
#     - parallelism
#
#   It does NOT yet add corporate actions or split-aware repair.
# ============================================================

UTC = "UTC"

# Local cache dir (safe even in containers: helps within-run; if persistent volume, helps across runs)
_CACHE_DIR: Path = paths.ensure_dir(paths.local_outputs_dir() / "yf_result_cache")
_CACHE_LOCK = threading.Lock()

DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_MARKET_ROOT = "market"


def _cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET)).strip()


def _cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION)).strip()


def _cfg_market_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "market_root", DEFAULT_MARKET_ROOT)).strip("/")


def _make_market_store(*, bucket: str, region: str, market_root: str) -> MarketStore:
    """
    Build a runtime-aware MarketStore.

    Important:
      - dev/staging MUST write under cfg.market_root.
      - if MarketStore does not support base_prefix yet, fail loudly for non-prod roots
        instead of silently writing to prod market/.
    """
    try:
        return MarketStore(bucket=bucket, region=region, base_prefix=market_root)
    except TypeError:
        if str(market_root).strip("/") != DEFAULT_MARKET_ROOT:
            raise RuntimeError(
                "MarketStore does not accept base_prefix yet, but this ingest run requires "
                f"market_root={market_root!r}. Patch MarketStore before running dev/staging ingest."
            )
        return MarketStore(bucket=bucket, region=region)

# -------------------------
# timezone helpers (STANDARD: tz-aware UTC everywhere)
# -------------------------
def _to_utc_ts(x) -> pd.Timestamp:
    """Parse anything into a tz-aware UTC Timestamp (or NaT)."""
    t = pd.to_datetime(x, errors="coerce", utc=True)
    return pd.Timestamp(t) if pd.notna(t) else pd.NaT

def _valid_latest_price_rows(ohlcv_usd: pd.DataFrame) -> pd.DataFrame:
    """
    Rows eligible for latest executable/analytics snapshots.

    Yahoo can return an incomplete current daily candle, especially for FX,
    where open/high/low exist but close/adj_close are NaN. Those rows may be
    useful to inspect, but they must never become latest_prices snapshots.
    """
    if ohlcv_usd is None or ohlcv_usd.empty:
        return pd.DataFrame()

    out = ohlcv_usd.copy()

    required = ["close_raw_usd", "close_adjusted_usd", "close_usd", "adj_close_usd"]
    for c in required:
        if c not in out.columns:
            return pd.DataFrame()
        out[c] = pd.to_numeric(out[c], errors="coerce")

    mask = (
        out["close_raw_usd"].notna()
        & out["close_adjusted_usd"].notna()
        & out["close_usd"].notna()
        & out["adj_close_usd"].notna()
        & np.isfinite(out["close_raw_usd"])
        & np.isfinite(out["close_adjusted_usd"])
        & (out["close_raw_usd"] > 0)
        & (out["close_adjusted_usd"] > 0)
    )

    return out.loc[mask].copy()


def _to_utc_series(x) -> pd.Series:
    """Parse anything into tz-aware UTC timestamps (Series)."""
    return pd.to_datetime(x, errors="coerce", utc=True)


def _normalize_day_index(idx: pd.Index) -> pd.DatetimeIndex:
    """
    Make an index:
      - datetime
      - tz-aware UTC
      - normalized to day boundary (00:00:00 UTC)
    """
    dti = pd.to_datetime(idx, errors="coerce", utc=True)
    dti = pd.DatetimeIndex(dti).normalize()
    return dti


def _clean_ccy(x: Any) -> str | None:
    """
    Normalize currency inputs into either:
      - clean uppercase currency code (e.g. 'USD', 'EUR', 'GBP')
      - None when missing/invalid

    Also maps common Yahoo oddities:
      - 'GBp', 'GBX' -> 'GBP'  (UK pence vs pounds)
      - 'ZAc'       -> 'ZAR'  (South Africa cents)
    """
    if x is None:
        return None

    try:
        if isinstance(x, float) and np.isnan(x):
            return None
    except Exception:
        pass

    s = str(x).strip()
    if not s:
        return None

    s_up = s.upper()

    if s_up in {"NAN", "NONE", "NULL"}:
        return None

    if s.lower() in {"gbp", "gbp ", "gbx"}:
        return "GBP"
    if s.lower() == "zac":
        return "ZAR"

    if len(s_up) < 3 or len(s_up) > 10:
        return None

    return s_up


def _hash_key(*parts: Any) -> str:
    s = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


def _df_cache_path(kind: str, key: str) -> Path:
    return _CACHE_DIR / kind / f"{key}.parquet"


def _series_cache_path(kind: str, key: str) -> Path:
    return _CACHE_DIR / kind / f"{key}.parquet"


def _cache_read_df(kind: str, key: str) -> pd.DataFrame | None:
    p = _df_cache_path(kind, key)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p, engine="pyarrow")
    except Exception:
        return None


def _cache_write_df(kind: str, key: str, df: pd.DataFrame) -> None:
    p = _df_cache_path(kind, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        df.to_parquet(tmp, engine="pyarrow", index=False)
        os.replace(tmp, p)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _cache_read_series(kind: str, key: str) -> pd.Series | None:
    p = _series_cache_path(kind, key)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, engine="pyarrow")
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        if "date" not in df.columns or "value" not in df.columns:
            return None
        s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"], errors="coerce", utc=True))
        s = s.dropna()
        s.index = _normalize_day_index(s.index)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s
    except Exception:
        return None


def _cache_write_series(kind: str, key: str, s: pd.Series) -> None:
    p = _series_cache_path(kind, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        ss = s.copy()
        ss.index = _normalize_day_index(ss.index)
        ss = ss[~ss.index.isna()].sort_index()
        out = pd.DataFrame({"date": ss.index, "value": pd.to_numeric(ss.values, errors="coerce")})
        out = out.dropna(subset=["date", "value"])
        out.to_parquet(tmp, engine="pyarrow", index=False)
        os.replace(tmp, p)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _is_fx_pair_like_symbol(sym: str) -> bool:
    """
    Identify Yahoo FX symbols or Quantfury-style pairs.
    """
    s = str(sym or "").strip().upper()
    if not s:
        return False
    if s.endswith("=X"):
        return True
    if "-" in s and len(s.split("-", 1)[0]) == 3 and len(s.split("-", 1)[1]) == 3:
        return True
    return False


def _normalize_fx_symbol_to_yahoo(sym: str) -> str:
    """
    Convert pair formats to Yahoo FX tickers.
    Examples:
      USD-CNY -> CNY=X   (Yahoo convention for USD/CCY)
      USD-JPY -> JPY=X
      EUR-USD -> EURUSD=X
      EUR-GBP -> EURGBP=X
      Already Yahoo: EURUSD=X stays as-is.
    """
    s = str(sym or "").strip().upper()
    if not s:
        return s

    if s.endswith("=X"):
        return s

    if "-" not in s:
        return s

    a, b = s.split("-", 1)
    a = a.strip().upper()
    b = b.strip().upper()

    if len(a) == 3 and len(b) == 3:
        if a == "USD":
            return f"{b}=X"
        return f"{a}{b}=X"

    return s


def _expected_last_closed_day_utc() -> pd.Timestamp:
    """
    Best-effort expected last fully-closed daily bar date (UTC, tz-aware),
    without exchange calendars:
      - take today's UTC midnight
      - subtract 1 day
      - roll back Sat/Sun to Friday
    """
    d = pd.Timestamp.now(tz=UTC).normalize() - pd.Timedelta(days=1)
    if d.weekday() == 5:
        d = d - pd.Timedelta(days=1)
    elif d.weekday() == 6:
        d = d - pd.Timedelta(days=2)
    return d.normalize()


def _yf_pop_error_for(symbol: str) -> str | None:
    """
    yfinance records request failures in yfinance.shared._ERRORS instead of raising.
    We pop and return the error string (if any) so callers can trigger retries.
    """
    try:
        import yfinance.shared as yfs  # type: ignore
        if not hasattr(yfs, "_ERRORS"):
            return None

        candidates = [symbol, str(symbol), str(symbol).upper(), str(symbol).lower()]
        for k in candidates:
            if k in yfs._ERRORS:  # type: ignore[attr-defined]
                err = yfs._ERRORS.pop(k, None)  # type: ignore[attr-defined]
                if err:
                    return str(err)
    except Exception:
        return None
    return None


def _is_retryable_yf_error(err: str | None) -> bool:
    """
    Retry only errors that are likely transient.

    Yahoo's "User is unable to access this feature" is usually not fixed by
    hammering retries, so treating it as retryable slows full-universe ingest.
    """
    if not err:
        return False

    e = str(err).lower()

    if "unable to access this feature" in e:
        return False

    if "invalid crumb" in e:
        return True

    if "unauthorized" in e and "unable to access this feature" not in e:
        return True

    return False


def _is_up_to_date_for_run(*, start: str, end: str | None) -> bool:
    """
    yfinance uses [start, end) semantics (end is exclusive).
    If start >= end, there is nothing new to fetch for this run.

    Standard: compare dates in UTC (tz-aware).
    """
    if not end:
        return False
    try:
        s = pd.to_datetime(start, errors="coerce", utc=True)
        e = pd.to_datetime(end, errors="coerce", utc=True)
        if pd.isna(s) or pd.isna(e):
            return False
        s = pd.Timestamp(s).normalize()
        e = pd.Timestamp(e).normalize()
        return s >= e
    except Exception:
        return False


def safe_end_date_for_interval(interval: str) -> str | None:
    """
    For daily bars, exclude the current day to avoid partial candles while markets are open
    (and to avoid today's still-forming crypto daily bar too).
    Returns an ISO date string suitable for yfinance end=...
    """
    interval = str(interval).strip().lower()
    if interval in {"1d", "1wk", "1mo"}:
        end = pd.Timestamp.now(tz=UTC).normalize()
        return end.strftime("%Y-%m-%d")
    return None


def fetch_yahoo_currency(ticker: str, session=None) -> dict:
    """
    Yahoo metadata fetch (currency/exchange/quoteType).
    session ignored intentionally.
    Returns cleaned currency (or None).
    """
    t = yf.Ticker(ticker)
    out = {"ticker": ticker, "currency": None, "exchange": None, "quoteType": None}

    try:
        fi = getattr(t, "fast_info", None) or {}
        if isinstance(fi, dict):
            out["currency"] = fi.get("currency")
            out["exchange"] = fi.get("exchange")
    except Exception:
        pass

    if not _clean_ccy(out.get("currency")):
        try:
            info = t.info or {}
            out["currency"] = info.get("currency")
            out["exchange"] = out.get("exchange") or info.get("exchange")
            out["quoteType"] = info.get("quoteType")
        except Exception:
            pass

    out["currency"] = _clean_ccy(out.get("currency"))

    if out.get("exchange") is not None:
        out["exchange"] = str(out["exchange"]).strip()

    if out.get("quoteType") is not None:
        out["quoteType"] = str(out["quoteType"]).strip()

    return out

def _as_1d_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Return a single Series for a column name.

    Defensive fix for yfinance outputs where column normalization can create
    duplicated names, causing df[col] to return a DataFrame instead of Series.
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, name=col)

    x = df[col]

    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(np.nan, index=df.index, name=col)
        x = x.iloc[:, 0]

    return pd.Series(x, index=df.index, name=col)


def _to_numeric_1d(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(_as_1d_series(df, col), errors="coerce").astype("float64")


def download_ohlcv(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
    session=None,
    allow_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV via yfinance with:
      - local result caching
      - retry-triggering on crumb/401
      - error surfacing on other Yahoo errors
      - fallback to Ticker().history() when download() returns empty
    NOTE: session is ignored intentionally (curl_cffi backend).
    """
    cache_key = _hash_key("ohlcv", ticker, start, end, interval, "auto_adjust_false_v4_utc")

    if allow_cache:
        with _CACHE_LOCK:
            cached = _cache_read_df("ohlcv", cache_key)
        if cached is not None and not cached.empty:
            return cached

    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        # yfinance can return either:
        #   columns = ["Open", "High", ...]
        # or MultiIndex columns:
        #   level 0 = price fields, level 1 = ticker
        # or occasionally the reverse.
        if isinstance(df.columns, pd.MultiIndex):
            field_names = {"open", "high", "low", "close", "adj_close", "adjclose", "volume"}

            best_level = 0
            best_score = -1

            for level in range(df.columns.nlevels):
                vals = [
                    str(x).strip().replace(" ", "_").lower()
                    for x in df.columns.get_level_values(level)
                ]
                score = sum(v in field_names for v in vals)
                if score > best_score:
                    best_score = score
                    best_level = level

            # Do not silently discard the symbol level. A Yahoo response for
            # another instrument must never be attributed to this asset_id.
            if best_score <= 0:
                raise ValueError(f"OHLCV[{ticker}]: unrecognized MultiIndex price columns")
            for level in range(df.columns.nlevels):
                if level == best_level:
                    continue
                symbols = {str(x).strip().upper() for x in df.columns.get_level_values(level)}
                if symbols != {str(ticker).strip().upper()}:
                    raise ValueError(
                        f"OHLCV[{ticker}]: unexpected symbol level {level}: {sorted(symbols)}"
                    )
            df.columns = df.columns.get_level_values(best_level)

        df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]

        # Normalize common aliases.
        rename_map = {
            "adjclose": "adj_close",
            "adj_close_": "adj_close",
            "datetime": "date",
            "index": "date",
        }
        df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

        # Remove duplicated columns defensively. Duplicates make df["close"]
        # return a DataFrame, which later breaks pd.to_numeric.
        df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")].copy()

        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
        df = df.reset_index()

        df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]

        if "date" not in df.columns:
            if "index" in df.columns:
                df = df.rename(columns={"index": "date"})
            elif "datetime" in df.columns:
                df = df.rename(columns={"datetime": "date"})

        if "adj_close" not in df.columns and "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "adj_close"})

        df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")].copy()

        return df

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    err = _yf_pop_error_for(ticker)

    if err and (df is None or df.empty):
        if _is_retryable_yf_error(err):
            raise RuntimeError(err)
        raise RuntimeError(f"yfinance_error[{ticker}] {err}")

    df = _normalize_df(df)

    if df.empty:
        try:
            t = yf.Ticker(ticker)
            h = t.history(start=start, end=end, interval=interval, auto_adjust=False)
            err2 = _yf_pop_error_for(ticker)

            if err2 and (h is None or h.empty):
                if _is_retryable_yf_error(err2):
                    raise RuntimeError(err2)
                raise RuntimeError(f"yfinance_error[{ticker}] {err2}")

            df = _normalize_df(h)
        except Exception:
            return pd.DataFrame()

    if not df.empty and allow_cache:
        with _CACHE_LOCK:
            _cache_write_df("ohlcv", cache_key, df)

    return df


def validate_mapping_continuity(
    *,
    store: MarketStore,
    ticker: str,
    overlap_start: str,
    overlap_end: str | None,
    new_ohlcv_usd: pd.DataFrame,
) -> dict:
    """
    Compare old stored series vs newly downloaded series on overlap window.
    Returns dict of metrics + suggested classification.
    """
    old = store.read_ohlcv_usd(
        [ticker],
        start=overlap_start,
        end=overlap_end,
        columns=["date", "ticker", "close_adjusted_usd"],
    )
    if old is None or old.empty:
        return {
            "ticker": ticker,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "n_overlap": 0,
            "ret_corr": None,
            "median_abs_pct_diff": None,
            "suggested": "NO_BASELINE",
            "why": "no existing stored data in overlap window",
        }

    new = new_ohlcv_usd[["date", "ticker", "close_adjusted_usd"]].copy()
    new["date"] = pd.to_datetime(new["date"], errors="coerce", utc=True)
    old["date"] = pd.to_datetime(old["date"], errors="coerce", utc=True)

    new = new.dropna(subset=["date"]).sort_values("date")
    old = old.dropna(subset=["date"]).sort_values("date")

    a = old.set_index("date")[["close_adjusted_usd"]].rename(columns={"close_adjusted_usd": "old"})
    b = new.set_index("date")[["close_adjusted_usd"]].rename(columns={"close_adjusted_usd": "new"})
    m = a.join(b, how="inner").dropna()

    if m.empty or m.shape[0] < 10:
        return {
            "ticker": ticker,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "n_overlap": int(m.shape[0]),
            "ret_corr": None,
            "median_abs_pct_diff": None,
            "suggested": "NO_OVERLAP",
            "why": "too few overlapping points",
        }

    r_old = m["old"].pct_change()
    r_new = m["new"].pct_change()
    rr = pd.concat([r_old, r_new], axis=1).dropna()
    ret_corr = float(rr.corr().iloc[0, 1]) if rr.shape[0] >= 5 else None

    pct_diff = (m["new"] / m["old"] - 1.0).abs()
    median_abs_pct_diff = float(pct_diff.median())

    if ret_corr is None:
        suggested = "INVESTIGATE"
        why = "not enough overlap for returns correlation"
    elif ret_corr >= 0.90 and median_abs_pct_diff <= 0.15:
        suggested = "LIKELY_SAME"
        why = "high return correlation and low price-level divergence"
    elif ret_corr <= 0.50 or median_abs_pct_diff >= 0.50:
        suggested = "LIKELY_DIFFERENT"
        why = "low return correlation or large price divergence"
    else:
        suggested = "INVESTIGATE"
        why = "mixed signals"

    return {
        "ticker": ticker,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "n_overlap": int(m.shape[0]),
        "ret_corr": ret_corr,
        "median_abs_pct_diff": median_abs_pct_diff,
        "suggested": suggested,
        "why": why,
    }


_FIAT_CCY = {
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK",
    "CNY", "CNH", "HKD", "SGD", "KRW", "TWD", "INR", "BRL", "MXN", "ZAR", "RUB",
    "PLN", "CZK", "HUF", "TRY", "ILS", "SAR", "AED", "QAR", "KWD", "BHD", "OMR",
    "THB", "MYR", "IDR", "PHP", "VND", "CLP", "COP", "PEN", "ARS",
}


def _normalize_crypto_symbol_to_yahoo(sym: str) -> str:
    """
    Fix common "crypto mis-labeled as FX" Yahoo symbols.
    Examples:
      BTCUSD=X -> BTC-USD
      ADAUSD=X -> ADA-USD
      BCHUSD=X -> BCH-USD
    Keep real FX:
      EURUSD=X stays EURUSD=X
      JPY=X stays JPY=X
    """
    s = str(sym or "").strip().upper()
    if not s:
        return s

    if s.endswith("USD=X") and len(s) > len("USD=X"):
        base = s[: -len("USD=X")].strip()
        if base in _FIAT_CCY:
            return s
        if base and 2 <= len(base) <= 10:
            return f"{base}-USD"

    return s


def download_fx_to_usd_series(
    ccy: str,
    start: str,
    end: str | None = None,
    session=None,
) -> pd.Series:
    """
    Returns series: FX rate to convert 1 unit of CCY into USD.
    - Uses yfinance
    - Caches final CCY->USD series locally
    - Includes 'end' in cache key so FX updates day-to-day

    NOTE: session is ignored intentionally.
    """
    ccy = str(ccy).upper().strip()
    if ccy == "USD":
        raise ValueError("USD has no FX series")

    cache_key = _hash_key("fx_to_usd", ccy, start, end, "v3_utc")
    with _CACHE_LOCK:
        cached = _cache_read_series("fx_to_usd", cache_key)
    if cached is not None and not cached.empty:
        cached.name = ccy
        return cached

    candidates = [
        (f"{ccy}USD=X", False),
        (f"USD{ccy}=X", True),
    ]

    last_err = None
    for fx_ticker, invert in candidates:
        try:
            fx = yf.download(
                fx_ticker,
                start=start,
                end=end,
                interval="1d",
                progress=False,
                threads=False,
                auto_adjust=False,
            )

            err = _yf_pop_error_for(fx_ticker)
            if _is_retryable_yf_error(err):
                raise RuntimeError(err)

            if fx is None or fx.empty:
                continue

            if "Adj Close" in fx.columns:
                s = fx["Adj Close"]
            elif "Close" in fx.columns:
                s = fx["Close"]
            else:
                continue

            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]

            s = s.dropna().astype("float64")
            s.index = _normalize_day_index(s.index)
            s = s[~s.index.isna()].sort_index()
            s = s[~s.index.duplicated(keep="last")]

            if s.empty:
                continue

            if invert:
                s = 1.0 / s

            s.name = ccy

            with _CACHE_LOCK:
                _cache_write_series("fx_to_usd", cache_key, s)

            return s

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not fetch FX for {ccy}. last={last_err}")


def get_fx_to_usd_for_dates(
    *,
    ccy: str,
    dates: pd.DatetimeIndex,
    start_base: str,
    fx_cache: dict[str, pd.Series],
    end: str | None = None,
    session=None,
) -> pd.Series:
    """
    Returns FX rate series aligned to `dates`, using an in-memory cache.
    - downloads once per currency per run
    - forward-fills to requested dates
    - will forward-fill beyond last FX date (weekend/lag) via union+ffill.

    Standard: dates/index are tz-aware UTC normalized.
    """
    ccy = str(ccy).upper().strip()
    dates_norm = _normalize_day_index(dates)

    if ccy == "USD":
        return pd.Series(1.0, index=dates_norm, name="USD")

    if ccy not in fx_cache:
        fx_cache[ccy] = download_fx_to_usd_series(ccy, start=start_base, end=end, session=None)

    s = fx_cache[ccy].copy()
    s.index = _normalize_day_index(s.index)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]

    full_idx = s.index.union(dates_norm)
    filled = s.reindex(full_idx).sort_index().ffill()
    out = filled.reindex(dates_norm)
    out.name = ccy
    return out


def compute_returns_per_ticker(ohlcv_usd: pd.DataFrame) -> pd.DataFrame:
    """
    Compute canonical USD log returns per ticker.

    Canonical returns are computed on adjusted USD closes:

        ret_log = log(close_adjusted_usd_t / close_adjusted_usd_{t-1})

    Compatibility:
      - ret_close_adjusted_usd remains the canonical return column.
      - ret_adj_close_usd remains a backward-compatible alias.
      - both now contain log returns.
    """
    cols = [
        "date",
        "ticker",
        "ret_close_adjusted_usd",
        "ret_adj_close_usd",
        "ret_log_close_adjusted_usd",
    ]

    if ohlcv_usd.empty:
        return pd.DataFrame(columns=cols)

    df = ohlcv_usd[["date", "ticker", "close_adjusted_usd"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["close_adjusted_usd"] = pd.to_numeric(df["close_adjusted_usd"], errors="coerce")

    df = (
        df.dropna(subset=["date", "ticker", "close_adjusted_usd"])
        .sort_values(["ticker", "date"], kind="stable")
        .drop_duplicates(subset=["ticker", "date"], keep="last")
    )

    prev = df.groupby("ticker")["close_adjusted_usd"].shift(1)

    ret_log = np.log(df["close_adjusted_usd"] / prev.replace(0.0, np.nan))
    ret_log = ret_log.replace([np.inf, -np.inf], np.nan)

    df["ret_log_close_adjusted_usd"] = ret_log
    df["ret_close_adjusted_usd"] = np.expm1(df["ret_log_close_adjusted_usd"])
    df["ret_adj_close_usd"] = df["ret_close_adjusted_usd"]

    df = df.dropna(subset=["ret_log_close_adjusted_usd"])

    return df[cols]


class RateLimiter:
    """
    Global limiter across all threads:
    allow at most `rate_per_sec` calls.
    """

    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / max(float(rate_per_sec), 1e-9)
        self._lock = threading.Lock()
        self._next_ts = 0.0

    def wait(self) -> None:
        import time

        with self._lock:
            now = time.time()
            if now < self._next_ts:
                time.sleep(self._next_ts - now)
            self._next_ts = max(self._next_ts, now) + self.min_interval


def call_yf_with_retries(
    fn,
    *,
    sem: threading.Semaphore,
    limiter: RateLimiter,
    attempts: int = 2,
    base_sleep: float = 0.4
):
    """
    Wrap any yfinance call with:
      - semaphore (concurrency)
      - global rate limiter (rate)
      - retry with exp backoff + jitter
    """
    import time
    import random

    last_err = None
    for k in range(int(attempts)):
        try:
            with sem:
                limiter.wait()
                return fn()
        except Exception as e:
            last_err = e
            sleep_s = float(base_sleep) * (2**k) * (0.7 + 0.6 * random.random())
            time.sleep(sleep_s)
    raise last_err


def ingest(
    *,
    bucket: str = DEFAULT_BUCKET,
    region: str = DEFAULT_REGION,
    market_root: str = DEFAULT_MARKET_ROOT,
    universe_csv: str | Path = paths.universe_dir() / "universe.csv",
    start_base: str = "2010-01-01",
    end_date: str | None = None,
    interval: str = "1d",
    max_assets: Optional[int] = None,
    force_refresh_csv: str | Path | None = paths.universe_dir() / "ingest_force_refresh.csv",
    max_workers: int = 4,
    yahoo_max_concurrency: int = 2,
    yahoo_rate_per_sec: float = 1.5,
    print_first_failures: int = 25,
    flush_failures_every: int = 50,
    flush_failures_min_seconds: float = 30.0,
    ignore_existing_state: bool = False,
    run_triage: bool = True,
    env_name: str = "dev",
    allow_large_dev_universe: bool = False,
    repair_bad_cells: str | None = None,
    repair_fx_parquet: str | None = None,
) -> None:
    import time

    t_start = time.time()

    fx_cache: dict[str, pd.Series] = {}
    fx_lock = threading.Lock()

    yf_sem = threading.Semaphore(int(yahoo_max_concurrency))
    limiter = RateLimiter(rate_per_sec=float(yahoo_rate_per_sec))

    universe_csv = Path(universe_csv)
    market_root = str(market_root).strip("/")

    store = _make_market_store(bucket=bucket, region=region, market_root=market_root)

    # yfinance end is exclusive. If caller passes --end 2024-02-05, we pass that through.
    end = str(end_date).strip() if end_date else safe_end_date_for_interval(interval)
    expected_last = (
        pd.Timestamp(end).tz_localize("UTC").normalize() - pd.Timedelta(days=1)
        if end
        else _expected_last_closed_day_utc()
    )

    if not universe_csv.exists():
        raise FileNotFoundError(f"Universe CSV not found: {universe_csv}")

    u_raw = pd.read_csv(universe_csv)
    u = u_raw[u_raw.get("include", 1).fillna(1).astype(int) == 1].copy()

    print("\n=== INGEST MARKET DATA ===")
    print(f"env:           {env_name}")
    print(f"bucket:        {bucket}")
    print(f"region:        {region}")
    print(f"market_root:   {market_root}")
    print(f"universe_csv:  {universe_csv}")
    print(f"universe_rows: {len(u_raw)}")
    print(f"included_rows: {len(u)}")
    print(f"start:         {start_base}")
    print(f"end:           {end}")
    print(f"interval:      {interval}")
    print(f"ignore_state:  {bool(ignore_existing_state)}")
    print("")

    if str(env_name).lower() == "dev" and len(u) > 100 and not bool(allow_large_dev_universe):
        raise RuntimeError(
            f"Refusing dev ingest with included_rows={len(u)}. "
            "This looks like a full universe run. Pass --allow-large-dev-universe only if intentional."
        )

    if "asset_id" not in u.columns:
        raise RuntimeError("Universe CSV must include 'asset_id' column (partition key).")

    u["asset_id"] = u["asset_id"].astype(str).str.strip()
    u["ticker"] = u.get("ticker", u["asset_id"]).astype(str).str.strip()
    u["yahoo_ticker"] = u.get("yahoo_ticker", u["ticker"]).astype(str).str.strip()

    u["yahoo_ticker_norm"] = (
        u["yahoo_ticker"]
        .apply(_normalize_fx_symbol_to_yahoo)
        .apply(_normalize_crypto_symbol_to_yahoo)
    )

    if "currency" in u.columns:
        u["currency"] = u["currency"].apply(_clean_ccy)
    else:
        u["currency"] = None

    repair_assets: set[str] = set()
    validated_fx = None
    if repair_bad_cells:
        bad = pd.read_csv(repair_bad_cells, usecols=["asset_id"])
        repair_assets = set(bad["asset_id"].dropna().astype(str).str.strip())
        missing = repair_assets - set(u["asset_id"])
        if missing:
            print(f"[repair] {len(missing)} anomaly asset IDs absent from included universe; NOT repaired")
        if not repair_fx_parquet:
            raise ValueError("--repair-fx-parquet required for full-history repair")
        validated_fx = pd.read_parquet(repair_fx_parquet)
        needed = {"currency", "date", "fx_to_usd", "validation_status"}
        if not needed.issubset(validated_fx.columns):
            raise ValueError("Validated FX table missing required columns")
        validated_fx = validated_fx.loc[
            validated_fx["validation_status"].isin(["VALID", "VALID_AFTER_GAP"])
        ].copy()
        validated_fx["date"] = pd.to_datetime(validated_fx["date"], utc=True).dt.normalize()
        validated_fx["currency"] = validated_fx["currency"].astype(str).str.upper()
        if validated_fx.duplicated(["currency", "date"]).any():
            raise ValueError("Duplicate validated FX date/currency")
        print(f"[repair] selected {len(repair_assets)} anomalous asset IDs for full-history replacement")
        u = u.loc[u["asset_id"].isin(repair_assets)].copy()
        if u.empty:
            raise ValueError("No eligible repair assets in universe")
    triples = list(
        zip(
            u["asset_id"].tolist(),
            u["ticker"].tolist(),
            u["yahoo_ticker_norm"].tolist(),
            u["currency"].tolist(),
        )
    )
    if max_assets:
        triples = triples[:max_assets]
    n_total = len(triples)

    force_refresh: set[str] = set()
    if force_refresh_csv:
        try:
            fr = pd.read_csv(force_refresh_csv)
            cols = set(c.strip().lower() for c in fr.columns)
            if "asset_id" in cols:
                col = [c for c in fr.columns if c.strip().lower() == "asset_id"][0]
                force_refresh = set(fr[col].astype(str).str.strip().tolist())
            elif "ticker" in cols:
                col = [c for c in fr.columns if c.strip().lower() == "ticker"][0]
                tickers = set(fr[col].astype(str).str.strip().str.upper().tolist())
                u_map = u.copy()
                u_map["ticker_u"] = u_map["ticker"].astype(str).str.upper().str.strip()
                force_refresh = set(
                    u_map.loc[u_map["ticker_u"].isin(tickers), "asset_id"].astype(str).str.strip().tolist()
                )
        except Exception:
            force_refresh = set()

    if repair_bad_cells:
        force_refresh = repair_assets
    if ignore_existing_state or repair_bad_cells:
        last_state = {}
        provider_state = {}
    else:
        last_state = store.read_last_date_state() or {}
        provider_state = store.read_provider_symbol_state() or {}

    # Seed map for adjusted-return continuity only.
    prev_adjusted_px_map: dict[str, float] = {}
    try:
        if hasattr(store, "read_latest_prices_adjusted_snapshot"):
            snap = store.read_latest_prices_adjusted_snapshot()
        else:
            snap = store.read_latest_prices_snapshot()

        if snap is not None and not snap.empty:
            if "asset_id" in snap.columns:
                snap["asset_id"] = snap["asset_id"].astype(str).str.strip()

                if "close_adjusted_usd" in snap.columns:
                    px_col = "close_adjusted_usd"
                elif "adj_close_usd" in snap.columns:
                    px_col = "adj_close_usd"
                else:
                    px_col = None

                if px_col is not None:
                    snap[px_col] = pd.to_numeric(snap[px_col], errors="coerce")
                    snap = snap.dropna(subset=["asset_id", px_col])
                    prev_adjusted_px_map = snap.set_index("asset_id")[px_col].astype(float).to_dict()
    except Exception:
        prev_adjusted_px_map = {}

    latest_prices_rows: list[dict] = []             # compatibility combined snapshot
    latest_prices_raw_rows: list[dict] = []
    latest_prices_adjusted_rows: list[dict] = []
    latest_returns_rows: list[dict] = []
    fail_rows: list[dict] = []

    last_flush_ts = time.time()
    n_fail_printed = 0

    as_of = pd.Timestamp.now(tz=UTC).strftime("%Y-%m-%d")

    max_written_return_date: pd.Timestamp | None = None
    total_returns_written = 0

    def _flush_failures_live(force: bool = False) -> None:
        nonlocal last_flush_ts
        if not fail_rows:
            return

        now = time.time()
        if (not force) and (done % int(flush_failures_every) != 0) and (
            (now - last_flush_ts) < float(flush_failures_min_seconds)
        ):
            return

        try:
            df_fail = pd.DataFrame(fail_rows)
            if not df_fail.empty:
                store.write_ingest_failures(df_fail)
                out_dir = paths.ensure_dir(paths.local_outputs_dir() / "ingest_failures")
                out_csv = out_dir / f"failures_live_{as_of}.csv"
                df_fail.to_csv(out_csv, index=False)
                print(f"[fails][live] rows={len(df_fail)} -> {out_csv}")
        except Exception as e:
            print(f"[fails][live][warn] flush failed: {e}")

        last_flush_ts = now

    def get_fx_locked_local(ccy: str, dates: pd.DatetimeIndex) -> pd.Series:
        if validated_fx is not None:
            data = validated_fx.loc[validated_fx["currency"].eq(ccy), ["date", "fx_to_usd"]]
            rates = data.set_index("date")["fx_to_usd"]
            normalized = pd.to_datetime(dates, utc=True).normalize()
            return pd.Series(rates.reindex(normalized).to_numpy(), index=dates)
        with fx_lock:
            return get_fx_to_usd_for_dates(
                ccy=ccy,
                dates=dates,
                start_base=start_base,
                end=end,
                fx_cache=fx_cache,
                session=None,
            )

    def _append_failure_row(*, res: dict, reason: str, error: str | None) -> None:
        fail_rows.append(
            {
                "as_of": as_of,
                "asset_id": res.get("asset_id"),
                "ticker": res.get("ticker"),
                "yahoo_ticker": res.get("yahoo_ticker"),
                "start": res.get("start"),
                "interval": interval,
                "reason": reason,
                "error": (error[:800] if isinstance(error, str) else error),
            }
        )

    def _build_raw_snapshot_row(last_price_row: dict) -> dict:
        return {
            "date": last_price_row.get("date"),
            "asset_id": last_price_row.get("asset_id"),
            "ticker": last_price_row.get("ticker"),
            "yahoo_ticker": last_price_row.get("yahoo_ticker"),
            "currency": last_price_row.get("currency"),
            "fx_to_usd": last_price_row.get("fx_to_usd"),
            "close": last_price_row.get("close"),
            "close_raw_usd": last_price_row.get("close_raw_usd"),
            "volume": last_price_row.get("volume"),
        }

    def _build_adjusted_snapshot_row(last_price_row: dict) -> dict:
        return {
            "date": last_price_row.get("date"),
            "asset_id": last_price_row.get("asset_id"),
            "ticker": last_price_row.get("ticker"),
            "yahoo_ticker": last_price_row.get("yahoo_ticker"),
            "currency": last_price_row.get("currency"),
            "fx_to_usd": last_price_row.get("fx_to_usd"),
            "adj_close": last_price_row.get("adj_close"),
            "close_adjusted_usd": last_price_row.get("close_adjusted_usd"),
            "volume": last_price_row.get("volume"),
        }

    def _process_one_local(asset_id: str, ticker: str, yahoo_sym: str, currency_hint: str | None) -> dict:
        asset_id = str(asset_id).strip()
        ticker = str(ticker).strip().upper()
        yahoo_sym = (str(yahoo_sym).strip() or ticker).upper()
        yahoo_sym = _normalize_crypto_symbol_to_yahoo(yahoo_sym)
        ccy_hint = _clean_ccy(currency_hint)

        is_force = asset_id in force_refresh
        if is_force:
            start = start_base
        else:
            if asset_id in last_state:
                s = pd.to_datetime(last_state[asset_id], errors="coerce", utc=True) + pd.Timedelta(days=1)
                start = pd.Timestamp(s).strftime("%Y-%m-%d") if pd.notna(s) else start_base
            else:
                start = start_base

        if _is_up_to_date_for_run(start=start, end=end):
            return {"status": "skip_up_to_date", "asset_id": asset_id}

        is_fx_asset = _is_fx_pair_like_symbol(yahoo_sym)

        try:
            df = call_yf_with_retries(
                lambda: download_ohlcv(yahoo_sym, start=start, end=end, interval=interval, session=None, allow_cache=not bool(repair_bad_cells)),
                sem=yf_sem,
                limiter=limiter,
            )

            if df is None or df.empty:
                meta = None
                try:
                    meta = fetch_yahoo_currency(yahoo_sym, session=None)
                except Exception:
                    meta = None

                meta_s = ""
                if isinstance(meta, dict) and meta:
                    meta_s = f" meta={ {k: meta.get(k) for k in ['currency', 'exchange', 'quoteType']} }"

                return {
                    "status": "empty",
                    "asset_id": asset_id,
                    "ticker": ticker,
                    "yahoo_ticker": yahoo_sym,
                    "start": start,
                    "error": f"no_ohlcv_from_yahoo.{meta_s}".strip(),
                }

            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col not in df.columns:
                    df[col] = np.nan
            df["date"] = df["date"].dt.normalize()
            df = df.dropna(subset=["date"]).sort_values("date")

            last_bar = _to_utc_ts(df["date"].max()).normalize() if not df.empty else None
            lag_days = int((expected_last - last_bar).days) if (last_bar is not None and pd.notna(last_bar)) else None
            freshness_note = None
            if last_bar is not None and lag_days is not None and lag_days > 5:
                freshness_note = (
                    f"stale: last_bar={last_bar.strftime('%Y-%m-%d')} "
                    f"expected~{expected_last.strftime('%Y-%m-%d')} lag_days={lag_days}"
                )

            if is_fx_asset:
                ccy = "USD"
                df["fx_to_usd"] = 1.0
            else:
                ccy = ccy_hint
                if not ccy:
                    meta = call_yf_with_retries(
                        lambda: fetch_yahoo_currency(yahoo_sym, session=None),
                        sem=yf_sem,
                        limiter=limiter,
                    )
                    ccy = _clean_ccy(meta.get("currency")) or "USD"

                if ccy == "USD":
                    df["fx_to_usd"] = 1.0
                else:
                    fx_s = get_fx_locked_local(ccy, pd.DatetimeIndex(df["date"]))
                    if fx_s is None or fx_s.empty or fx_s.isna().all():
                        return {
                            "status": "no_fx",
                            "asset_id": asset_id,
                            "ticker": ticker,
                            "yahoo_ticker": yahoo_sym,
                            "start": start,
                            "error": f"no_fx_for_ccy={ccy}",
                        }
                    df["fx_to_usd"] = fx_s.values
                    if df["fx_to_usd"].isna().any():
                        if repair_bad_cells:
                            return {"status":"repair_missing_validated_fx", "asset_id":asset_id, "error":f"{ccy}: no validated rate for historical dates"}
                        df = df.dropna(subset=["fx_to_usd"])
                        if df.empty:
                            return {
                                "status": "no_fx_aligned",
                                "asset_id": asset_id,
                                "ticker": ticker,
                                "yahoo_ticker": yahoo_sym,
                                "start": start,
                                "error": f"no_fx_aligned_for_ccy={ccy}",
                            }

            df["asset_id"] = asset_id
            df["ticker"] = ticker
            df["yahoo_ticker"] = yahoo_sym
            df["currency"] = ("USD" if is_fx_asset else ccy)

            # Phase 0 explicit semantics
            close_s = _to_numeric_1d(df, "close")
            adj_close_s = _to_numeric_1d(df, "adj_close")
            fx_s = _to_numeric_1d(df, "fx_to_usd")

            df["close_raw_usd"] = close_s * fx_s
            df["close_adjusted_usd"] = adj_close_s * fx_s

            # Fail closed before persisting OHLCV: derived USD prices must be
            # finite, positive, and consistent with local price * approved FX.
            for local_col, usd_col in (("close", "close_raw_usd"), ("adj_close", "close_adjusted_usd")):
                local = pd.to_numeric(df[local_col], errors="coerce")
                usd = pd.to_numeric(df[usd_col], errors="coerce")
                rates = pd.to_numeric(df["fx_to_usd"], errors="coerce")
                expected = local * rates
                ok = (
                    np.isfinite(local) & (local > 0)
                    & np.isfinite(rates) & (rates > 0)
                    & np.isfinite(usd) & (usd > 0)
                    & np.isclose(usd, expected, rtol=1e-10, atol=1e-10)
                )
                if not bool(ok.all()):
                    examples = df.loc[~ok, ["date", "asset_id", local_col, "fx_to_usd", usd_col]].head(5)
                    raise ValueError(
                        f"OHLCV derived USD integrity failure: {usd_col}; "
                        f"invalid_rows={int((~ok).sum())}; examples={examples.to_dict('records')}"
                    )

            # Backward-compatibility aliases
            df["close_usd"] = df["close_raw_usd"]
            df["adj_close_usd"] = df["close_adjusted_usd"]

            ohlcv_usd = df[
                [
                    "date",
                    "asset_id",
                    "ticker",
                    "yahoo_ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                    "currency",
                    "fx_to_usd",
                    "close_raw_usd",
                    "close_adjusted_usd",
                    "close_usd",
                    "adj_close_usd",
                ]
            ].copy()

            ohlcv_usd = ohlcv_usd.sort_values("date").drop_duplicates(subset=["asset_id", "date"], keep="last")
            ohlcv_usd["year"] = ohlcv_usd["date"].dt.year.astype(int)

            rows_written = 0
            newly_written_dates: set[str] = set()

            for year, g in ohlcv_usd.groupby("year", sort=False):
                g = g.drop(columns=["year"]).copy()
                g["date_str"] = g["date"].dt.strftime("%Y-%m-%d")

                man = store.read_asset_year_manifest(table="ohlcv_usd", asset_id=asset_id, year=int(year)) or {}
                have_dates = set(man.get("dates", []))
                have_parts = set(man.get("parts", []))

                to_write = (g.copy() if repair_bad_cells else g[~g["date_str"].isin(have_dates)].copy())
                if to_write.empty:
                    continue
                if repair_bad_cells and have_dates and not have_dates.issubset(set(to_write["date_str"])):
                    return {"status":"repair_incomplete_source", "asset_id":asset_id,
                            "error":f"{year}: fresh source missing {len(have_dates - set(to_write['date_str']))} existing dates; original preserved"}
                if repair_bad_cells:
                    # Preserve old objects. Publish new immutable part first, then switch
                    # active manifest references. Back up the old manifest for rollback.
                    import json as _json
                    backup_key = f"{store.base_prefix}/repairs/aea87/full_history/asset_id={asset_id}/year={year}/ohlcv_manifest.json"
                    manifest_key = store._manifest_key(table="ohlcv_usd", asset_id=asset_id, year=int(year))
                    try:
                        original_manifest = store._get_bytes(manifest_key)
                        if store._key_exists(backup_key):
                            return {"status":"repair_backup_exists", "asset_id":asset_id, "error":backup_key}
                        store._put_bytes(backup_key, original_manifest, "application/json")
                    except Exception as exc:
                        return {"status":"repair_backup_failed", "asset_id":asset_id, "error":str(exc)}
                written_parts = store.write_ohlcv_usd_partitioned(to_write.drop(columns=["date_str"]))
                rows_written += int(to_write.shape[0])
                new_dates = sorted(set(to_write["date_str"].tolist()))
                newly_written_dates.update(new_dates)
                store.write_asset_year_manifest(
                    table="ohlcv_usd", asset_id=asset_id, year=int(year),
                    dates=(new_dates if repair_bad_cells else list(have_dates.union(new_dates))),
                    parts=(written_parts if repair_bad_cells else list(have_parts.union(written_parts))),
                )

            if rows_written == 0:
                return {"status": "skip_already_ingested", "asset_id": asset_id}

            # RETURNS (canonical on adjusted closes)
            px = ohlcv_usd[["date", "close_adjusted_usd"]].copy()
            px["date"] = pd.to_datetime(px["date"], errors="coerce", utc=True).dt.normalize()
            px["close_adjusted_usd"] = pd.to_numeric(px["close_adjusted_usd"], errors="coerce")
            px = px.dropna(subset=["date", "close_adjusted_usd"]).sort_values("date").drop_duplicates(
                subset=["date"], keep="last"
            )

            if (not is_force) and (asset_id in prev_adjusted_px_map) and (not px.empty):
                first_date = pd.Timestamp(px["date"].iloc[0]).normalize()
                seed_date = first_date - pd.Timedelta(days=1)
                seed_px_adjusted = float(prev_adjusted_px_map[asset_id])
                if np.isfinite(seed_px_adjusted) and seed_px_adjusted > 0:
                    px = pd.concat(
                        [pd.DataFrame([{"date": seed_date, "close_adjusted_usd": seed_px_adjusted}]), px],
                        ignore_index=True,
                    ).sort_values("date")

            prev_close = px["close_adjusted_usd"].shift(1).replace(0.0, np.nan)

            px["ret_log_close_adjusted_usd"] = np.log(px["close_adjusted_usd"] / prev_close)
            px["ret_log_close_adjusted_usd"] = px["ret_log_close_adjusted_usd"].replace([np.inf, -np.inf], np.nan)

            # Backward-compatible aliases. These are now log returns.
            px["ret_close_adjusted_usd"] = np.expm1(px["ret_log_close_adjusted_usd"])
            px["ret_adj_close_usd"] = px["ret_close_adjusted_usd"]

            px = px.dropna(subset=["ret_log_close_adjusted_usd"])

            returns_written = 0
            last_return_row = None

            if not px.empty and newly_written_dates:
                px["date_str"] = px["date"].dt.strftime("%Y-%m-%d")
                ret_new = px[px["date_str"].isin(newly_written_dates)].copy()

                if not ret_new.empty:
                    returns = pd.DataFrame(
                        {
                            "date": ret_new["date"].values,
                            "asset_id": asset_id,
                            "ticker": ticker,
                            "ret_log_close_adjusted_usd": ret_new["ret_log_close_adjusted_usd"].astype("float64").values,
                        }
                    )

                    # Backward-compatible aliases. These are now log returns.
                    returns["ret_close_adjusted_usd"] = np.expm1(returns["ret_log_close_adjusted_usd"])
                    returns["ret_adj_close_usd"] = returns["ret_close_adjusted_usd"]
                    returns["date"] = pd.to_datetime(returns["date"], errors="coerce", utc=True).dt.normalize()
                    returns["year"] = returns["date"].dt.year.astype(int)

                    for year, rg in returns.groupby("year", sort=False):
                        rg = rg.drop(columns=["year"]).copy()
                        rg["date_str"] = rg["date"].dt.strftime("%Y-%m-%d")

                        rman = store.read_asset_year_manifest(table="returns_usd", asset_id=asset_id, year=int(year)) or {}
                        have_dates = set(rman.get("dates", []))
                        have_parts = set(rman.get("parts", []))

                        r_to_write = (rg.copy() if repair_bad_cells else rg[~rg["date_str"].isin(have_dates)].copy())
                        if r_to_write.empty:
                            continue

                        if repair_bad_cells:
                            import json as _json
                            rkey = store._manifest_key(table="returns_usd", asset_id=asset_id, year=int(year))
                            backup = f"{store.base_prefix}/repairs/aea87/full_history/asset_id={asset_id}/year={year}/returns_manifest.json"
                            try:
                                prior = store._get_bytes(rkey)
                                if store._key_exists(backup):
                                    return {"status":"repair_backup_exists", "asset_id":asset_id, "error":backup}
                                store._put_bytes(backup, prior, "application/json")
                            except Exception as exc:
                                return {"status":"repair_returns_backup_failed", "asset_id":asset_id, "error":str(exc)}
                        written_parts = store.write_returns_usd_partitioned(r_to_write.drop(columns=["date_str"]))
                        returns_written += int(r_to_write.shape[0])

                        new_dates = sorted(set(r_to_write["date_str"].tolist()))
                        store.write_asset_year_manifest(
                            table="returns_usd",
                            asset_id=asset_id,
                            year=int(year),
                            dates=(new_dates if repair_bad_cells else list(have_dates.union(new_dates))),
                            parts=(written_parts if repair_bad_cells else list(have_parts.union(written_parts))),
                        )

                    if returns_written > 0:
                        last_return_row = returns.sort_values("date").iloc[-1].to_dict()

            ohlcv_snapshot_valid = _valid_latest_price_rows(ohlcv_usd)

            if ohlcv_snapshot_valid.empty:
                return {
                    "status": "empty_valid_price",
                    "asset_id": asset_id,
                    "ticker": ticker,
                    "yahoo_ticker": yahoo_sym,
                    "start": start,
                    "error": (
                        "downloaded rows exist but no valid close_raw_usd/close_adjusted_usd; "
                        "likely incomplete current-day candle"
                    ),
                }

            last_price_row = ohlcv_snapshot_valid.sort_values("date").iloc[-1].to_dict()
            last_date = pd.Timestamp(last_price_row["date"]).date().isoformat()
            last_raw = last_price_row.get("close_raw_usd")
            last_adj = last_price_row.get("close_adjusted_usd")

            raw_latest_downloaded = ohlcv_usd.sort_values("date").iloc[-1].to_dict()
            raw_latest_date = pd.Timestamp(raw_latest_downloaded["date"]).date().isoformat()

            if raw_latest_date != last_date:
                last_price_row["_freshness"] = (
                    f"ignored_incomplete_latest_bar: raw_latest_date={raw_latest_date} "
                    f"snapshot_date={last_date}"
                )
            elif freshness_note:
                last_price_row["_freshness"] = freshness_note

            raw_snapshot_row = _build_raw_snapshot_row(last_price_row)
            adjusted_snapshot_row = _build_adjusted_snapshot_row(last_price_row)

            return {
                "status": "ok",
                "asset_id": asset_id,
                "ticker": ticker,
                "yahoo_ticker": yahoo_sym,
                "start": start,
                "ohlcv_rows_written": rows_written,
                "returns_written": returns_written,
                "last_date": last_date,
                "last_price_row": last_price_row,
                "last_price_raw_row": raw_snapshot_row,
                "last_price_adjusted_row": adjusted_snapshot_row,
                "last_return_row": last_return_row,
                "last_close_raw_usd": float(last_raw) if last_raw is not None else None,
                "last_close_adjusted_usd": float(last_adj) if last_adj is not None else None,
                "freshness_note": freshness_note,
            }

        except Exception as e:
            return {
                "status": "fail",
                "asset_id": asset_id,
                "ticker": ticker,
                "yahoo_ticker": yahoo_sym,
                "start": start,
                "error": str(e)[:800],
            }

    done = ok = skipped = failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        futs = [ex.submit(_process_one_local, a, t, y, c) for (a, t, y, c) in triples]

        for fut in as_completed(futs):
            res = fut.result()
            done += 1

            st = res.get("status")

            if st == "ok":
                ok += 1

                latest_prices_rows.append(res["last_price_row"])  # compatibility combined snapshot
                latest_prices_raw_rows.append(res["last_price_raw_row"])
                latest_prices_adjusted_rows.append(res["last_price_adjusted_row"])

                if res.get("last_return_row") is not None:
                    latest_returns_rows.append(res["last_return_row"])
                    d = pd.to_datetime(res["last_return_row"].get("date"), errors="coerce", utc=True)
                    if pd.notna(d):
                        d = pd.Timestamp(d).normalize()
                        if (max_written_return_date is None) or (d > max_written_return_date):
                            max_written_return_date = d

                total_returns_written += int(res.get("returns_written") or 0)

                aid = res["asset_id"]
                last_state[aid] = res["last_date"]
                provider_state[aid] = res["yahoo_ticker"]

                if res.get("last_close_adjusted_usd") is not None:
                    prev_adjusted_px_map[aid] = float(res["last_close_adjusted_usd"])

                if res.get("freshness_note"):
                    print(f"[freshness][warn] {res.get('ticker')} {res.get('yahoo_ticker')} {res.get('freshness_note')}")

            elif st in {"skip_up_to_date", "skip_already_ingested"}:
                skipped += 1
            else:
                failed += 1
                err = res.get("error")
                if res.get("status") == "empty" and not err:
                    err = "empty_ohlcv"
                _append_failure_row(res=res, reason=st, error=err)

                if n_fail_printed < int(print_first_failures):
                    n_fail_printed += 1
                    print(
                        f"[fail][{n_fail_printed}] status={st} "
                        f"asset_id={res.get('asset_id')} ticker={res.get('ticker')} yahoo={res.get('yahoo_ticker')} "
                        f"start={res.get('start')} err={err}"
                    )

                _flush_failures_live(force=False)

            if done % 50 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                print(
                    f"[ingest] done={done}/{n_total} ok={ok} skipped={skipped} failed={failed} "
                    f"rets_written={total_returns_written} fx_cached={len(fx_cache)} "
                    f"rate={rate:.2f} assets/s elapsed={elapsed/60:.1f}m"
                )

    # ---------- SNAPSHOTS / STATE WRITE ----------

    # Combined latest prices snapshot (compatibility)
    latest_prices_new = pd.DataFrame(latest_prices_rows)
    try:
        old_prices = store.read_latest_prices_snapshot()
        latest_prices = pd.concat([old_prices, latest_prices_new], ignore_index=True)
    except Exception:
        latest_prices = latest_prices_new

    if not latest_prices.empty:
        latest_prices["date"] = pd.to_datetime(latest_prices["date"], errors="coerce", utc=True).dt.normalize()
        latest_prices = latest_prices.dropna(subset=["date"])
        latest_prices["asset_id"] = latest_prices["asset_id"].astype(str).str.strip()

        latest_prices = _valid_latest_price_rows(latest_prices)
        latest_prices = latest_prices.sort_values("date").drop_duplicates(subset=["asset_id"], keep="last")

        store.write_latest_prices_snapshot(latest_prices.reset_index(drop=True))

    # Raw latest prices snapshot
    latest_prices_raw_new = pd.DataFrame(latest_prices_raw_rows)
    if not latest_prices_raw_new.empty:
        try:
            if hasattr(store, "read_latest_prices_raw_snapshot"):
                old_prices_raw = store.read_latest_prices_raw_snapshot()
            else:
                old_prices_raw = pd.DataFrame()
            latest_prices_raw = pd.concat([old_prices_raw, latest_prices_raw_new], ignore_index=True)
        except Exception:
            latest_prices_raw = latest_prices_raw_new

        latest_prices_raw["date"] = pd.to_datetime(latest_prices_raw["date"], errors="coerce", utc=True).dt.normalize()
        latest_prices_raw = latest_prices_raw.dropna(subset=["date"])
        latest_prices_raw["asset_id"] = latest_prices_raw["asset_id"].astype(str).str.strip()
        latest_prices_raw["close_raw_usd"] = pd.to_numeric(latest_prices_raw["close_raw_usd"], errors="coerce")
        latest_prices_raw = latest_prices_raw[
            latest_prices_raw["close_raw_usd"].notna()
            & np.isfinite(latest_prices_raw["close_raw_usd"])
            & (latest_prices_raw["close_raw_usd"] > 0)
        ].copy()
        latest_prices_raw = latest_prices_raw.sort_values("date").drop_duplicates(subset=["asset_id"], keep="last")

        if hasattr(store, "write_latest_prices_raw_snapshot"):
            store.write_latest_prices_raw_snapshot(latest_prices_raw.reset_index(drop=True))

    # Adjusted latest prices snapshot
    latest_prices_adjusted_new = pd.DataFrame(latest_prices_adjusted_rows)
    if not latest_prices_adjusted_new.empty:
        try:
            if hasattr(store, "read_latest_prices_adjusted_snapshot"):
                old_prices_adjusted = store.read_latest_prices_adjusted_snapshot()
            else:
                old_prices_adjusted = pd.DataFrame()
            latest_prices_adjusted = pd.concat(
                [old_prices_adjusted, latest_prices_adjusted_new],
                ignore_index=True,
            )
        except Exception:
            latest_prices_adjusted = latest_prices_adjusted_new

        latest_prices_adjusted["date"] = pd.to_datetime(
            latest_prices_adjusted["date"], errors="coerce", utc=True
        ).dt.normalize()
        latest_prices_adjusted = latest_prices_adjusted.dropna(subset=["date"])
        latest_prices_adjusted["asset_id"] = latest_prices_adjusted["asset_id"].astype(str).str.strip()
        latest_prices_adjusted["close_adjusted_usd"] = pd.to_numeric(
            latest_prices_adjusted["close_adjusted_usd"],
            errors="coerce",
        )
        latest_prices_adjusted = latest_prices_adjusted[
            latest_prices_adjusted["close_adjusted_usd"].notna()
            & np.isfinite(latest_prices_adjusted["close_adjusted_usd"])
            & (latest_prices_adjusted["close_adjusted_usd"] > 0)
        ].copy()
        latest_prices_adjusted = latest_prices_adjusted.sort_values("date").drop_duplicates(
            subset=["asset_id"], keep="last"
        )

        if hasattr(store, "write_latest_prices_adjusted_snapshot"):
            store.write_latest_prices_adjusted_snapshot(latest_prices_adjusted.reset_index(drop=True))

    # Latest returns snapshot (canonical adjusted-return semantics)
    latest_returns_new = pd.DataFrame(latest_returns_rows)
    try:
        old_rets = store.read_latest_returns_snapshot()
        latest_returns = pd.concat([old_rets, latest_returns_new], ignore_index=True)
    except Exception:
        latest_returns = latest_returns_new

    if latest_returns is not None and not latest_returns.empty:
        latest_returns["date"] = pd.to_datetime(latest_returns["date"], errors="coerce", utc=True).dt.normalize()
        latest_returns = latest_returns.dropna(subset=["date"])
        latest_returns["asset_id"] = latest_returns["asset_id"].astype(str).str.strip()
        latest_returns = latest_returns.sort_values("date").drop_duplicates(subset=["asset_id"], keep="last")
        store.write_latest_returns_snapshot(latest_returns.reset_index(drop=True))

    _flush_failures_live(force=True)

    fails = pd.DataFrame(fail_rows)
    if not fails.empty:
        store.write_ingest_failures(fails)

    try:
        out_dir = paths.ensure_dir(paths.local_outputs_dir() / "ingest_failures")
        out_csv = out_dir / f"failures_{as_of}.csv"
        fails.to_csv(out_csv, index=False)
        print(f"[fails][local] wrote -> {out_csv}")
    except Exception as e:
        print(f"[fails][local][warn] could not write local failures csv: {e}")

    store.write_last_date_state(last_state)
    store.write_provider_symbol_state(provider_state)

    if max_written_return_date is not None:
        store.write_returns_latest_state(
            {
                "as_of_utc": pd.Timestamp.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last_date": pd.Timestamp(max_written_return_date).strftime("%Y-%m-%d"),
                "n_returns_written": int(total_returns_written),
                "n_assets_total": int(n_total),
                "interval": str(interval),
                "job": "ingest_market_data.py",
            }
        )

    if run_triage:
        run_post_ingest_triage(
            store=store,
            as_of=as_of,
            universe_csv=universe_csv,
            overrides_csv=paths.universe_dir() / "universe_overrides.csv",
            excluded_csv=paths.universe_dir() / "asset_excluded.csv",
            mapping_changes=pd.DataFrame(),
            mapping_validation=pd.DataFrame(),
            verbose=True,
            sample_n=15,
            local_out_dir=paths.universe_dir() / "triage_outputs",
        )

    print("\n[DONE]")
    print(f"assets_total={n_total}")
    print(f"ok={ok} skipped={skipped} failed={failed}")
    print(f"latest_prices_rows_written={len(latest_prices_new)}")
    print(f"latest_prices_raw_rows_written={len(latest_prices_raw_new)}")
    print(f"latest_prices_adjusted_rows_written={len(latest_prices_adjusted_new)}")
    print(f"latest_returns_rows_written={len(latest_returns_new)}")
    print(f"returns_written_total={total_returns_written}")
    print(f"[FX] currencies_downloaded={len(fx_cache)} -> {sorted(fx_cache.keys())}")
    print(f"state_entries={len(last_state)}")
    print(f"expected_last_closed_day_utc={expected_last.strftime('%Y-%m-%d')}")
    print(f"elapsed_s={time.time()-t_start:.1f}")
    print(f"[cache] dir={_CACHE_DIR}")



def publish_validated_fx_from_local(*, input_dir: str, run_id: str, store: MarketStore, upload: bool) -> str:
    """AEA-87: reuse MarketStore; reject partial historical FX downloads."""
    import json
    from pathlib import Path
    import pandas as pd
    root = Path(input_dir)
    report_path = root / "validation_report.json"
    data_path = root / "fx_history_all.parquet"
    if not report_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("Expected validation_report.json and fx_history_all.parquet in --fx-input-dir")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    entries = report.get("currencies", [])
    if not entries or any("error" in e or int(e.get("rows", 0)) == 0 for e in entries):
        raise ValueError("FX refresh incomplete: one or more currency downloads failed")
    if any(e.get("statuses", {}).get(k, 0) for e in entries for k in ("QUOTE_DISAGREEMENT", "LARGE_FX_CHANGE")):
        raise ValueError("FX refresh has unresolved critical quote disagreements or large changes")
    frame = pd.read_parquet(data_path)
    wanted = {e["currency"] for e in entries}
    if set(frame["currency"].dropna().unique()) != wanted:
        raise ValueError("Currency coverage mismatch between report and combined parquet")
    if len(frame) != sum(int(e["rows"]) for e in entries):
        raise ValueError("FX report rows differ from combined parquet")
    for entry in entries:
        ccy = entry["currency"]
        part = frame[frame["currency"] == ccy]
        if part["validation_status"].value_counts().to_dict() != entry.get("statuses", {}):
            raise ValueError(f"FX validation status counts differ: {ccy}")
        individual = root / f"{ccy}_fx_history.parquet"
        if not individual.is_file():
            raise FileNotFoundError(f"Missing per-currency FX file: {individual}")
        other = pd.read_parquet(individual)
        cols = ["date", "currency", "fx_to_usd", "validation_status", "direct_quote", "inverse_quote"]
        if not set(cols).issubset(other.columns):
            raise ValueError(f"Missing FX columns in {individual}")
        left = part[cols].sort_values(["date"]).reset_index(drop=True)
        right = other[cols].sort_values(["date"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    dest = store.write_fx_snapshot_staging(frame, report, run_id=run_id, upload=upload)
    print(("UPLOADED " if upload else "DRY RUN verified; destination would be ") + dest)
    return dest



def repair_full_histories(*, cfg, bad_cells: str, fx_parquet: str, universe_path: str,
                          start: str, end: str | None, max_assets: int | None,
                          confirm_prod_write: bool) -> None:
    """AEA-87 direct full-history replacement, with per-asset manifest rollback.

    This is intentionally separate from the ordinary append-only ingestion loop.
    No missing Yahoo dates or missing FX dates are invented. Successful assets
    are committed individually; failures are listed in a local CSV.
    """
    import hashlib
    import json
    import io
    import uuid
    from datetime import datetime, timezone

    require_prod_confirmation(cfg, bool(confirm_prod_write))
    if not confirm_prod_write:
        raise ValueError("Historical repair requires --confirm-prod-write")
    if str(getattr(cfg, "env", "")).lower() != "prod":
        raise ValueError("Historical repair requires --env prod")

    store = _make_market_store(
        bucket=_cfg_bucket(cfg), region=_cfg_region(cfg),
        market_root=_cfg_market_root(cfg),
    )
    bad = pd.read_csv(bad_cells)
    if "asset_id" not in bad.columns:
        raise ValueError("Anomaly CSV missing asset_id")
    bad["asset_id"] = bad["asset_id"].astype(str).str.strip()
    if "date" in bad.columns:
        bad["date"] = pd.to_datetime(bad["date"], errors="coerce", utc=True).dt.normalize()
    requested = set(bad["asset_id"].dropna().astype(str).str.strip())
    universe = pd.read_csv(universe_path)
    if "include" in universe:
        universe = universe.loc[universe["include"].fillna(1).astype(int).eq(1)]
    universe["asset_id"] = universe["asset_id"].astype(str).str.strip()
    universe = universe.loc[universe["asset_id"].isin(requested)].copy()
    if universe["asset_id"].duplicated().any():
        raise ValueError("Multiple universe rows for the same asset_id")
    missing = requested - set(universe["asset_id"])
    if missing:
        print(f"[repair] excluded/unmapped assets: {len(missing)}")
    if max_assets:
        universe = universe.head(max_assets)
    if universe.empty:
        raise ValueError("No assets selected for historical repair")
    fx = pd.read_parquet(fx_parquet)
    fx["date"] = pd.to_datetime(fx["date"], utc=True).dt.normalize()
    fx["currency"] = fx["currency"].astype(str).str.upper()
    fx = fx.loc[fx["validation_status"].isin(["VALID", "VALID_AFTER_GAP"])].copy()
    if fx.duplicated(["currency", "date"]).any():
        raise ValueError("Duplicate validated FX dates")
    fx_lookup = {c: g.set_index("date")["fx_to_usd"] for c, g in fx.groupby("currency")}

    report = []
    # Preserve all active historical years even when Yahoo no longer serves them.
    def existing_history(aid: str) -> pd.DataFrame:
        prefix = f"{store.manifests_prefix}/ohlcv_usd/asset_id={aid}/"
        frames = []
        for key in store._list_keys(prefix):
            if not key.endswith("/manifest.json"):
                continue
            man = json.loads(store._get_bytes(key))
            parts = man.get("parts") or []
            if not parts:
                raise ValueError(f"{aid}: active year manifest has no part references")
            for part in parts:
                frames.append(pd.read_parquet(io.BytesIO(store._get_bytes(part))))
        if not frames:
            return pd.DataFrame()
        old = pd.concat(frames, ignore_index=True)
        old["date"] = pd.to_datetime(old["date"], utc=True).dt.normalize()
        if old.duplicated(["asset_id", "date"]).any():
            raise ValueError(f"{aid}: duplicate active historical dates")
        return old

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    report_file = paths.ensure_dir(paths.local_outputs_dir() / "aea87_repair") / f"full_history_{run_id}.csv"
    total = len(universe)
    for seq, (_, asset) in enumerate(universe.iterrows(), 1):
        aid = str(asset["asset_id"]).strip()
        yahoo = _normalize_crypto_symbol_to_yahoo(_normalize_fx_symbol_to_yahoo(
            str(asset.get("yahoo_ticker") or asset.get("ticker")).strip().upper()
        ))
        ticker = str(asset.get("ticker") or yahoo)
        currency = _clean_ccy(asset.get("currency"))
        if aid.startswith("CRYPTO:") or _is_fx_pair_like_symbol(yahoo):
            currency = "USD"
        if not currency:
            report.append({"asset_id": aid, "status": "SKIPPED_UNRESOLVED_CURRENCY"})
            continue
        print(f"[repair] {seq}/{total} {aid} ({yahoo})")
        updated = []
        try:
            downloaded = download_ohlcv(yahoo, start=start, end=end,
                                        interval="1d", allow_cache=False)
            if downloaded.empty:
                raise ValueError("Fresh Yahoo history is empty")
            downloaded["date"] = pd.to_datetime(downloaded["date"], utc=True).dt.normalize()
            downloaded = downloaded.dropna(subset=["date"]).sort_values("date")
            if downloaded["date"].duplicated().any():
                raise ValueError("Duplicate dates in fresh Yahoo history")
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col not in downloaded:
                    raise ValueError(f"Missing downloaded column {col}")
            historical = existing_history(aid)
            if currency == "USD":
                downloaded["fx_to_usd"] = 1.0
            else:
                if currency not in fx_lookup:
                    raise ValueError(f"Validated FX missing currency {currency}")
                new_dates = pd.DatetimeIndex(downloaded["date"])
                rate = fx_lookup[currency].reindex(new_dates)
                downloaded["fx_to_usd"] = rate.to_numpy()
                # Missing FX does not authorize an invented conversion rate.
                # Prefer the original bar at these dates and flag it as retained.
                if rate.isna().any():
                    print(f"[repair][coverage] {aid}: {int(rate.isna().sum())} downloaded rows lack validated FX; retaining old where present")
            for col in ["open", "high", "low", "close", "adj_close", "volume", "fx_to_usd"]:
                downloaded[col] = pd.to_numeric(downloaded[col], errors="coerce")
            if downloaded["close"].isna().all() or downloaded["adj_close"].isna().all():
                raise ValueError("Downloaded source closes missing")
            downloaded["asset_id"] = aid
            downloaded["ticker"] = ticker
            downloaded["yahoo_ticker"] = yahoo
            downloaded["currency"] = currency
            downloaded["close_raw_usd"] = downloaded["close"] * downloaded["fx_to_usd"]
            downloaded["close_adjusted_usd"] = downloaded["adj_close"] * downloaded["fx_to_usd"]
            downloaded["close_usd"] = downloaded["close_raw_usd"]
            downloaded["adj_close_usd"] = downloaded["close_adjusted_usd"]
            cols = ["date", "asset_id", "ticker", "yahoo_ticker", "open", "high",
                    "low", "close", "adj_close", "volume", "currency", "fx_to_usd",
                    "close_raw_usd", "close_adjusted_usd", "close_usd", "adj_close_usd"]
            prices = downloaded[cols].copy()
            prices = prices.loc[prices["close"].notna() & prices["adj_close"].notna()
                                & prices["fx_to_usd"].notna()
                                & prices["close"].gt(0) & prices["adj_close"].gt(0)]
            replaced_count = len(prices)
            if not historical.empty:
                historical = historical.copy()
                historical["date"] = pd.to_datetime(historical["date"], utc=True).dt.normalize()
                original_dates = set(historical["date"])
                refreshed_dates = set(prices["date"])
                retained = historical.loc[~historical["date"].isin(refreshed_dates)].copy()
                for col in cols:
                    if col not in retained.columns:
                        retained[col] = pd.NA
                retained = retained[cols]
                # Reconvert retained non-USD observations only where a validated
                # currency/date rate exists. Otherwise keep and report old FX.
                if currency != "USD" and not retained.empty and currency in fx_lookup:
                    historical_rate = fx_lookup[currency].reindex(pd.DatetimeIndex(retained["date"]))
                    reconvert = historical_rate.notna().to_numpy()
                    retained.loc[reconvert, "fx_to_usd"] = historical_rate.to_numpy()[reconvert]
                    for raw_col, usd_col in (("close", "close_raw_usd"), ("adj_close", "close_adjusted_usd"),
                                             ("close", "close_usd"), ("adj_close", "adj_close_usd")):
                        retained.loc[reconvert, usd_col] = (
                            pd.to_numeric(retained.loc[reconvert, raw_col], errors="coerce")
                            * pd.to_numeric(retained.loc[reconvert, "fx_to_usd"], errors="coerce")
                        )
                prices = pd.concat([retained, prices], ignore_index=True)
            else:
                original_dates = set()
                refreshed_dates = set(prices["date"])
            prices = prices.sort_values("date")
            if prices.duplicated(["asset_id", "date"]).any():
                raise ValueError("Merged history contains duplicated dates")
            preserved_count = len(prices) - replaced_count
            if replaced_count == 0:
                raise ValueError("No valid refreshed observations, refusing replacement")
            px = prices[["date", "close_adjusted_usd"]].dropna()
            px = px.loc[px["close_adjusted_usd"].gt(0)].copy()
            px["ret_log_close_adjusted_usd"] = np.log(
                px["close_adjusted_usd"] / px["close_adjusted_usd"].shift(1)
            )
            px = px.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_log_close_adjusted_usd"])
            returns = pd.DataFrame({"date": px["date"], "asset_id": aid, "ticker": ticker,
                                    "ret_log_close_adjusted_usd": px["ret_log_close_adjusted_usd"]})
            returns["ret_close_adjusted_usd"] = np.expm1(returns["ret_log_close_adjusted_usd"])
            returns["ret_adj_close_usd"] = returns["ret_close_adjusted_usd"]

            # Prepare all historical partitions before modifying active manifests.
            pending = []
            for table, frame in (("ohlcv_usd", prices), ("returns_usd", returns)):
                for year, g in frame.groupby(frame["date"].dt.year):
                    y = int(year)
                    key = store._manifest_key(table=table, asset_id=aid, year=y)
                    old = None
                    try:
                        old = store._get_bytes(key)
                    except store._client().exceptions.NoSuchKey:
                        pass
                    if old is not None:
                        prior = json.loads(old)
                        old_dates = set(prior.get("dates") or [])
                        fresh_dates = set(g["date"].dt.strftime("%Y-%m-%d"))
                        if table == "ohlcv_usd" and not old_dates.issubset(fresh_dates):
                            raise ValueError(f"{table} {y}: merged history missing previously active dates")
                    data = io.BytesIO()
                    g.to_parquet(data, index=False)
                    pending.append((table, y, key, old, g, data.getvalue()))

            # A merged rebuild must also cover each existing OHLCV manifest year.
            covered_years = set(prices["date"].dt.year.astype(int))
            for key in store._list_keys(f"{store.manifests_prefix}/ohlcv_usd/asset_id={aid}/"):
                if key.endswith("/manifest.json"):
                    y = int(key.split("/year=")[1].split("/")[0])
                    if y not in covered_years:
                        raise ValueError(f"Active year {y} absent from merged history")

            # Back up all manifests before switching any references.
            for table, year, key, old, g, payload in pending:
                if old is not None:
                    backup = f"{store.base_prefix}/repairs/aea87/{run_id}/asset_id={aid}/{table}/{year}/old_manifest.json"
                    store._put_bytes(backup, old, "application/json")
                    if store._get_bytes(backup) != old:
                        raise IOError("Manifest backup verification failed")

            staged = []
            for table, year, key, old, g, payload in pending:
                prefix = store.ohlcv_prefix if table == "ohlcv_usd" else store.returns_prefix
                part = store._part_key(prefix, aid, year)
                store._put_bytes(part, payload)
                if hashlib.sha256(store._get_bytes(part)).digest() != hashlib.sha256(payload).digest():
                    raise IOError("Staged partition checksum mismatch")
                manifest = {
                    "asset_id": aid, "year": year,
                    "dates": sorted(g["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
                    "parts": [part], "repair_run_id": run_id,
                }
                staged.append((key, old, manifest))

            for key, old, manifest in staged:
                if old is not None and store._get_bytes(key) != old:
                    raise RuntimeError("Manifest was changed concurrently")
                store._put_bytes(key, json.dumps(manifest, indent=2).encode(), "application/json")
                updated.append((key, old))

            report.append({"asset_id": aid, "status": "REPLACED", "price_rows": len(prices),
                           "refreshed_rows": replaced_count, "preserved_rows": preserved_count,
                           "unrefreshed_flagged_dates": len(set(pd.to_datetime(
                               bad.loc[bad["asset_id"].eq(aid), "date"], utc=True
                           ).dt.normalize()) & (original_dates - refreshed_dates)) if "date" in bad else None,
                           "return_rows": len(returns), "manifests": len(staged)})
        except Exception as exc:
            rollback_failed = False
            for key, old in reversed(updated):
                try:
                    if old is None:
                        store._client().delete_object(Bucket=store.bucket, Key=key)
                    else:
                        store._put_bytes(key, old, "application/json")
                except Exception:
                    rollback_failed = True
            report.append({"asset_id": aid,
                           "status": "ROLLBACK_FAILED" if rollback_failed else "FAILED_OR_ROLLED_BACK",
                           "error": str(exc)})
            print(f"[repair][error] {aid}: {exc}")
            if rollback_failed:
                pd.DataFrame(report).to_csv(report_file, index=False)
                raise RuntimeError(f"Repair rollback failed for {aid}; manual recovery needed") from exc
        finally:
            pd.DataFrame(report).to_csv(report_file, index=False)
    print(f"[repair] campaign complete; report={report_file}")
    counts = pd.Series([r["status"] for r in report]).value_counts().to_dict()
    print("[repair] results:", counts)
    unresolved_count = sum(int(r.get("unrefreshed_flagged_dates") or 0) for r in report)
    if unresolved_count:
        print(f"[repair][warning] {unresolved_count} previously flagged dates were preserved from old S3; NOT considered repaired")
    if any(r["status"] != "REPLACED" for r in report) or missing or unresolved_count:
        raise RuntimeError("Repair incomplete: inspect campaign report; wide cache not rebuilt")
    # Rebuild derived wide returns cache only after all targeted assets are committed.
    import subprocess
    import sys
    cmd = [sys.executable, "-m", "alpha_edge.market.build_returns_wide_cache",
           "--env", "prod", "--confirm-prod-write", "--force"]
    print("[repair] rebuilding wide returns cache:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[repair] OHLCV, returns and wide-cache repair completed")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ingest Alpha Edge market data into S3.")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--fx-input-dir", default=None, help="AEA-87 validated local FX staging directory; FX-only mode")
    ap.add_argument("--fx-run-id", default=None, help="Immutable S3 FX staging snapshot ID")
    ap.add_argument("--fx-upload", action="store_true", help="Explicitly upload FX to versioned S3 staging")
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument("--universe-path", default=str(paths.universe_dir() / "universe.csv"))
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--interval", default="1d")

    ap.add_argument("--max-assets", type=int, default=None)
    ap.add_argument("--force-refresh-csv", default=str(paths.universe_dir() / "ingest_force_refresh.csv"))

    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--yahoo-max-concurrency", type=int, default=2)
    ap.add_argument("--yahoo-rate-per-sec", type=float, default=1.5)

    ap.add_argument("--repair-mode", choices=["prepare", "promote"], default="prepare")
    ap.add_argument("--repair-run-id", default=None)
    ap.add_argument("--repair-asset-ids", default=None, help="Comma-separated canonical asset_ids; prepare only these anomalous assets")
    ap.add_argument("--repair-reset-asset-ids", default=None, help="Comma-separated IDs to replace with fresh history only (must be selected)")
    ap.add_argument("--repair-bad-cells", help="AEA-87: CSV of anomalous asset IDs for full-history replacement")
    ap.add_argument("--repair-fx-parquet", help="AEA-87: validated local eight-currency FX Parquet")
    ap.add_argument("--ignore-existing-state", action="store_true")
    ap.add_argument("--no-triage", action="store_true")
    ap.add_argument("--allow-large-dev-universe", action="store_true")

    return ap.parse_args()


def _main_impl() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)
    if args.repair_reset_asset_ids and (not args.repair_asset_ids or args.repair_mode != "prepare"):
        raise ValueError("--repair-reset-asset-ids requires --repair-asset-ids and prepare mode")
    if args.repair_asset_ids and (not args.repair_bad_cells or args.repair_mode != "prepare"):
        raise ValueError("--repair-asset-ids requires --repair-bad-cells and --repair-mode prepare")
    if args.repair_bad_cells:
        from alpha_edge.market.historical_repair import run as run_historical_repair
        run_historical_repair(cfg=cfg, bad_cells=args.repair_bad_cells,
                              fx_parquet=args.repair_fx_parquet,
                              universe_path=args.universe_path,
                              start=args.start, end=args.end,
                              max_assets=args.max_assets,
                              confirm_prod_write=args.confirm_prod_write,
                              mode=args.repair_mode, run_id=args.repair_run_id,
                              repair_asset_ids=args.repair_asset_ids,
                              reset_asset_ids=args.repair_reset_asset_ids)
        return
    if args.fx_input_dir:
        if args.fx_upload:
            require_prod_confirmation(cfg, bool(args.confirm_prod_write))
        if not args.fx_run_id:
            raise ValueError("--fx-run-id required with --fx-input-dir")
        store = _make_market_store(
            bucket=_cfg_bucket(cfg),
            region=_cfg_region(cfg),
            market_root=_cfg_market_root(cfg),
        )
        publish_validated_fx_from_local(
            input_dir=args.fx_input_dir,
            run_id=args.fx_run_id,
            store=store,
            upload=bool(args.fx_upload),
        )
        return
    if args.fx_upload:
        raise ValueError("--fx-upload requires --fx-input-dir")
    require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    ingest(
        bucket=_cfg_bucket(cfg),
        region=_cfg_region(cfg),
        market_root=_cfg_market_root(cfg),
        universe_csv=args.universe_path,
        start_base=str(args.start),
        end_date=args.end,
        interval=str(args.interval),
        max_assets=args.max_assets,
        force_refresh_csv=args.force_refresh_csv,
        max_workers=int(args.max_workers),
        yahoo_max_concurrency=int(args.yahoo_max_concurrency),
        yahoo_rate_per_sec=float(args.yahoo_rate_per_sec),
        ignore_existing_state=bool(args.ignore_existing_state),
        repair_bad_cells=args.repair_bad_cells,
        repair_fx_parquet=args.repair_fx_parquet,
        run_triage=(not bool(args.no_triage)),
        env_name=str(getattr(cfg, "env", args.env or "dev")),
        allow_large_dev_universe=bool(args.allow_large_dev_universe),
    )


# ----------------------------
# Audit/logging entrypoint wrapper
# ----------------------------
def _tier1_audit_is_dry_run(args: argparse.Namespace) -> bool:
    if getattr(args, "repair_bad_cells", None):
        return getattr(args, "repair_mode", "prepare") != "promote"
    if getattr(args, "fx_input_dir", None):
        return not bool(getattr(args, "fx_upload", False))
    return bool(getattr(args, "dry_run", False) or getattr(args, "no_write", False))


def main_with_audit() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = _tier1_audit_is_dry_run(args)

    with capture_script_run(
        cfg=cfg,
        script_name="ingest_market_data.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            _main_impl()

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="ingest_market_data",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="ingest_market_data.py",
                source_mode="market_data_ingestion",
                status=("dry_run" if is_dry_run else "success"),
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                    "note": "Tier 1 audit event is entrypoint-level. Detailed output keys/row counts are available in the script log stdout and script-specific metadata where emitted by the script.",
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
        except Exception as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="ingest_market_data",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="ingest_market_data.py",
                source_mode="market_data_ingestion",
                status="failed",
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise


def main() -> None:
    main_with_audit()


if __name__ == "__main__":
    main()
