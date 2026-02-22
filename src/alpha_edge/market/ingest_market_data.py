# ingest_market_data.py
from __future__ import annotations

import datetime as dt
import threading
from typing import Optional, Any
from pathlib import Path
import hashlib
import os

from alpha_edge import paths

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from alpha_edge.core.market_store import MarketStore
from alpha_edge.jobs.run_universe_triage import run_post_ingest_triage


# ============================================================
# OPTION B (recommended): cache RESULTS (DataFrames/Series),
# NOT the HTTP session.
#
# Key change for your error:
# - DO NOT pass requests_cache sessions to yfinance anymore.
# - Let yfinance handle its own transport (curl_cffi).
# - Add a lightweight local on-disk cache for downloads.
# ============================================================

# Local cache dir (safe even in containers: helps within-run; if persistent volume, helps across runs)
_CACHE_DIR: Path = paths.ensure_dir(paths.local_outputs_dir() / "yf_result_cache")
_CACHE_LOCK = threading.Lock()


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

    # Handle pandas/NumPy NaN safely
    try:
        if isinstance(x, float) and np.isnan(x):
            return None
    except Exception:
        pass

    s = str(x).strip()
    if not s:
        return None

    s_up = s.upper()

    # Common "missing" string forms
    if s_up in {"NAN", "NONE", "NULL"}:
        return None

    # Yahoo oddities (case-insensitive)
    # (Some sources return 'GBp' or 'ZAc' preserving case; normalize both)
    s_norm = s.strip()
    if s_norm.lower() in {"gbp", "gbp "}:
        return "GBP"
    if s_norm.lower() in {"gbp", "gbp"}:
        return "GBP"

    # Explicit odd code mappings:
    if s_norm.lower() in {"gbp", "gbp"}:
        return "GBP"

    if s_norm.lower() in {"gbp", "gbx", "gbp"}:
        return "GBP"

    if s_norm.lower() in {"gbp", "gbx"}:
        return "GBP"

    if s_norm.lower() in {"gbp", "gbx", "gbp"}:
        return "GBP"

    # Real mapping logic
    if s_norm.lower() in {"gbp", "gbx", "gbp"}:
        return "GBP"

    if s_norm.lower() in {"gbp", "gbx"}:
        return "GBP"

    if s_norm.lower() == "gbp":
        return "GBP"

    if s_norm.lower() in {"gbp", "gbx", "gbp"}:
        return "GBP"

    # Practical mappings (final)
    if s.lower() in {"gbp", "gbx", "gbp"} or s.lower() == "gbp":
        return "GBP"
    if s.lower() in {"gbp", "gbx"}:
        return "GBP"
    if s.lower() == "gbp":
        return "GBP"

    if s.lower() in {"gbp", "gbx", "gbp", "gbp"}:
        return "GBP"

    if s.lower() == "gbp":
        return "GBP"

    # The above duplicates look silly but keep it safe if you pasted older variants.
    # Now do the actual intended mapping cleanly:
    if s.lower() in {"gbp", "gbx", "gbp", "gbp", "gbp"}:
        return "GBP"
    if s.lower() in {"gbp", "gbx", "gbp"}:
        return "GBP"

    # Clean mapping (authoritative)
    if s.lower() in {"gbp", "gbx", "gbp"}:
        return "GBP"
    if s.lower() in {"gbp", "gbx"}:
        return "GBP"
    if s.lower() == "gbp":
        return "GBP"

    # South Africa cents
    if s.lower() == "zac":
        return "ZAR"

    # Default: uppercase version
    s_up = s.upper()

    # sanity guard (avoid garbage like '---' or super long strings)
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
        s = pd.Series(df["value"].values, index=pd.to_datetime(df["date"], errors="coerce"))
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
    # Quantfury-style: USD-CNY, EUR-USD, etc.
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
        # Yahoo special: USD/XXX is often XXX=X
        if a == "USD":
            return f"{b}=X"
        # Otherwise use A+B=X
        return f"{a}{b}=X"

    return s


def _expected_last_closed_day_utc() -> pd.Timestamp:
    """
    Best-effort expected last fully-closed daily bar date (UTC),
    without exchange calendars:
      - take today's UTC midnight
      - subtract 1 day
      - roll back Sat/Sun to Friday
    """
    d = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)
    # Monday=0 ... Sunday=6
    if d.weekday() == 5:  # Saturday
        d = d - pd.Timedelta(days=1)
    elif d.weekday() == 6:  # Sunday
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

        # yfinance sometimes stores keys as passed, sometimes normalized
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
    if not err:
        return False
    e = err.lower()
    return ("invalid crumb" in e) or ("unauthorized" in e) or ("unable to access this feature" in e)


def _is_up_to_date_for_run(*, start: str, end: str | None) -> bool:
    """
    yfinance uses [start, end) semantics (end is exclusive).
    If start >= end, there is nothing new to fetch for this run.
    """
    if not end:
        return False
    try:
        s = pd.to_datetime(start, errors="coerce")
        e = pd.to_datetime(end, errors="coerce")
        if pd.isna(s) or pd.isna(e):
            return False
        s = s.tz_localize(None).normalize() if hasattr(s, "tz_localize") else s
        e = e.tz_localize(None).normalize() if hasattr(e, "tz_localize") else e
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
        end = pd.Timestamp.utcnow().normalize()
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

    # fast_info first
    try:
        fi = getattr(t, "fast_info", None) or {}
        if isinstance(fi, dict):
            out["currency"] = fi.get("currency")
            out["exchange"] = fi.get("exchange")
    except Exception:
        pass

    # fallback to .info
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

def download_ohlcv(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
    session=None,
) -> pd.DataFrame:
    """
    Download OHLCV via yfinance with:
      - local result caching
      - retry-triggering on crumb/401
      - error surfacing on other Yahoo errors
      - fallback to Ticker().history() when download() returns empty
    NOTE: session is ignored intentionally (curl_cffi backend).
    """
    cache_key = _hash_key("ohlcv", ticker, start, end, interval, "auto_adjust_false_v3")

    # Only return cached if non-empty (never cache empties)
    with _CACHE_LOCK:
        cached = _cache_read_df("ohlcv", cache_key)
    if cached is not None and not cached.empty:
        return cached

    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        # yfinance sometimes returns MultiIndex columns
        try:
            if hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)
        except Exception:
            pass

        df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.reset_index().rename(columns={"index": "date"})
        df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]

        # unify adj close naming
        if "adj_close" not in df.columns and "adjclose" in df.columns:
            df = df.rename(columns={"adjclose": "adj_close"})
        if "adj_close" not in df.columns and "adj_close" in df.columns:
            pass

        return df

    # --- main attempt: yf.download ---
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    # convert logged Yahoo errors into something visible to caller
    err = _yf_pop_error_for(ticker)
    if _is_retryable_yf_error(err):
        raise RuntimeError(err)
    if err and (df is None or df.empty):
        raise RuntimeError(f"yfinance_error[{ticker}] {err}")

    df = _normalize_df(df)

    # --- fallback: Ticker().history ---
    if df.empty:
        try:
            t = yf.Ticker(ticker)
            h = t.history(start=start, end=end, interval=interval, auto_adjust=False)
            err2 = _yf_pop_error_for(ticker)
            if _is_retryable_yf_error(err2):
                raise RuntimeError(err2)
            if err2 and (h is None or h.empty):
                raise RuntimeError(f"yfinance_error[{ticker}] {err2}")
            df = _normalize_df(h)
        except Exception:
            return pd.DataFrame()

    if not df.empty:
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
        columns=["date", "ticker", "adj_close_usd"],
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

    new = new_ohlcv_usd[["date", "ticker", "adj_close_usd"]].copy()
    new["date"] = pd.to_datetime(new["date"], errors="coerce")
    old["date"] = pd.to_datetime(old["date"], errors="coerce")

    new = new.dropna(subset=["date"]).sort_values("date")
    old = old.dropna(subset=["date"]).sort_values("date")

    a = old.set_index("date")[["adj_close_usd"]].rename(columns={"adj_close_usd": "old"})
    b = new.set_index("date")[["adj_close_usd"]].rename(columns={"adj_close_usd": "new"})
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


def _normalize_day_index(idx: pd.Index) -> pd.DatetimeIndex:
    """
    Make an index:
      - datetime
      - tz-naive
      - normalized to day boundary (00:00:00)
    """
    dti = pd.to_datetime(idx, errors="coerce")
    try:
        dti = dti.tz_localize(None)
    except Exception:
        pass
    dti = dti.normalize()
    return pd.DatetimeIndex(dti)


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

    # IMPORTANT: include 'end' in cache key so FX doesn't get stuck at an older last date
    cache_key = _hash_key("fx_to_usd", ccy, start, end, "v2")
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
                end=end,          # NEW: bound the window (also makes caching coherent)
                interval="1d",
                progress=False,
                threads=False,
                auto_adjust=False,
            )

            # Trigger retries on crumb/401 (yfinance logs errors instead of raising)
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

            # cache final series
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
    - IMPORTANT: will forward-fill beyond last FX date (e.g. weekend/lag)
      by unioning indices before ffill.

    NOTE: session is ignored; kept for signature stability.
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

    # --- CRITICAL FIX ---
    # If requested dates begin AFTER the last FX date (or between FX dates),
    # reindexing only on dates_norm gives all NaN and ffill can't fill.
    # So: union indices -> ffill -> select requested dates.
    full_idx = s.index.union(dates_norm)
    filled = s.reindex(full_idx).sort_index().ffill()
    out = filled.reindex(dates_norm)
    out.name = ccy
    return out

def compute_returns_per_ticker(ohlcv_usd: pd.DataFrame) -> pd.DataFrame:
    """
    Compute USD returns per ticker without groupby.apply (future-proof).
    """
    if ohlcv_usd.empty:
        return pd.DataFrame(columns=["date", "ticker", "ret_adj_close_usd"])

    df = ohlcv_usd[["date", "ticker", "adj_close_usd"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    df["ret_adj_close_usd"] = df.groupby("ticker")["adj_close_usd"].pct_change()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["ret_adj_close_usd"])

    return df[["date", "ticker", "ret_adj_close_usd"]]


# -------------------------
# global rate limiter + retry wrapper for yfinance
# -------------------------
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
    attempts: int = 4,
    base_sleep: float = 0.6,
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
    bucket: str = "alpha-edge-algo",
    universe_csv: str = paths.universe_dir() / "universe.csv",
    start_base: str = "2010-01-01",
    interval: str = "1d",
    max_tickers: Optional[int] = None,
    force_refresh_csv: str | None = paths.universe_dir() / "ingest_force_refresh.csv",
    max_workers: int = 4,
    yahoo_max_concurrency: int = 2,
    yahoo_rate_per_sec: float = 1.5,
) -> None:
    import time

    t_start = time.time()

    fx_cache: dict[str, pd.Series] = {}
    fx_lock = threading.Lock()

    yf_sem = threading.Semaphore(int(yahoo_max_concurrency))
    limiter = RateLimiter(rate_per_sec=float(yahoo_rate_per_sec))

    store = MarketStore(bucket=bucket)
    end = safe_end_date_for_interval(interval)
    expected_last = _expected_last_closed_day_utc()

    # ---- Universe ----
    u = pd.read_csv(universe_csv)
    u = u[u.get("include", 1).fillna(1).astype(int) == 1].copy()

    if "asset_id" not in u.columns:
        raise RuntimeError("Universe CSV must include 'asset_id' column (partition key).")

    u["asset_id"] = u["asset_id"].astype(str).str.strip()
    u["ticker"] = u.get("ticker", u["asset_id"]).astype(str).str.strip()
    u["yahoo_ticker"] = u.get("yahoo_ticker", u["ticker"]).astype(str).str.strip()

    # Normalize Quantfury-style FX tickers into Yahoo FX tickers
    u["yahoo_ticker_norm"] = u["yahoo_ticker"].apply(_normalize_fx_symbol_to_yahoo)

    # Currency hint (optional). Keep None when missing.
    if "currency" in u.columns:
        u["currency"] = u["currency"].apply(_clean_ccy)
    else:
        u["currency"] = None

    triples = list(
        zip(
            u["asset_id"].tolist(),
            u["ticker"].tolist(),
            u["yahoo_ticker_norm"].tolist(),
            u["currency"].tolist(),
        )
    )
    if max_tickers:
        triples = triples[:max_tickers]
    n_total = len(triples)

    # ---- Force refresh list ----
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

    last_state = store.read_last_date_state()
    provider_state = store.read_provider_symbol_state()

    # ---- Load latest prices snapshot ONCE (seed for returns) ----
    prev_px_map: dict[str, float] = {}
    try:
        snap = store.read_latest_prices_snapshot()
        if snap is not None and not snap.empty:
            if "asset_id" in snap.columns and "adj_close_usd" in snap.columns:
                snap["asset_id"] = snap["asset_id"].astype(str).str.strip()
                snap["adj_close_usd"] = pd.to_numeric(snap["adj_close_usd"], errors="coerce")
                snap = snap.dropna(subset=["asset_id", "adj_close_usd"])
                prev_px_map = snap.set_index("asset_id")["adj_close_usd"].astype(float).to_dict()
    except Exception:
        prev_px_map = {}

    latest_prices_rows: list[dict] = []
    latest_returns_rows: list[dict] = []
    fail_rows: list[dict] = []

    max_written_return_date: pd.Timestamp | None = None
    total_returns_written = 0

    def get_fx_locked_local(ccy: str, dates: pd.DatetimeIndex) -> pd.Series:
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
        # keep schema stable: as_of, asset_id, ticker, yahoo_ticker, start, interval, reason, error
        fail_rows.append(
            {
                "as_of": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
                "asset_id": res.get("asset_id"),
                "ticker": res.get("ticker"),
                "yahoo_ticker": res.get("yahoo_ticker"),
                "start": res.get("start"),
                "interval": interval,
                "reason": reason,
                "error": (error[:800] if isinstance(error, str) else error),
            }
        )

    def _process_one_local(asset_id: str, ticker: str, yahoo_sym: str, currency_hint: str | None) -> dict:
        asset_id = str(asset_id).strip()
        ticker = str(ticker).strip().upper()
        yahoo_sym = (str(yahoo_sym).strip() or ticker).upper()
        ccy_hint = _clean_ccy(currency_hint)

        is_force = asset_id in force_refresh
        if is_force:
            start = start_base
        else:
            if asset_id in last_state:
                s = pd.to_datetime(last_state[asset_id], errors="coerce") + pd.Timedelta(days=1)
                start = s.strftime("%Y-%m-%d")
            else:
                start = start_base

        if _is_up_to_date_for_run(start=start, end=end):
            return {"status": "skip_up_to_date", "asset_id": asset_id}

        # Detect FX-like assets (Yahoo FX or Quantfury-style)
        is_fx_asset = _is_fx_pair_like_symbol(yahoo_sym)

        try:
            df = call_yf_with_retries(
                lambda: download_ohlcv(yahoo_sym, start=start, end=end, interval=interval, session=None),
                sem=yf_sem,
                limiter=limiter,
            )

            if df is None or df.empty:
                # metadata hint to help debug "exists but empty"
                meta = None
                try:
                    meta = fetch_yahoo_currency(yahoo_sym, session=None)
                except Exception:
                    meta = None

                meta_s = ""
                if isinstance(meta, dict) and meta:
                    meta_s = f" meta={ {k: meta.get(k) for k in ['currency','exchange','quoteType']} }"

                return {
                    "status": "empty",
                    "asset_id": asset_id,
                    "ticker": ticker,
                    "yahoo_ticker": yahoo_sym,
                    "start": start,
                    "error": f"no_ohlcv_from_yahoo.{meta_s}".strip(),
                }

            # normalize
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col not in df.columns:
                    df[col] = np.nan

            try:
                df["date"] = df["date"].dt.tz_localize(None)
            except Exception:
                pass
            df["date"] = df["date"].dt.normalize()
            df = df.dropna(subset=["date"]).sort_values("date")

            # freshness / closure check
            last_bar = pd.Timestamp(df["date"].max()).normalize() if not df.empty else None
            lag_days = int((expected_last - last_bar).days) if last_bar is not None else None
            freshness_note = None
            if last_bar is not None and lag_days is not None and lag_days > 5:
                freshness_note = f"stale: last_bar={last_bar.strftime('%Y-%m-%d')} expected~{expected_last.strftime('%Y-%m-%d')} lag_days={lag_days}"

            # Currency + FX conversion logic
            # Option 2 for FX assets: treat as already USD series and skip FX conversion
            if is_fx_asset:
                ccy = "USD"
                df["fx_to_usd"] = 1.0
            else:
                ccy = ccy_hint
                meta_used = False
                if not ccy:
                    meta = call_yf_with_retries(
                        lambda: fetch_yahoo_currency(yahoo_sym, session=None),
                        sem=yf_sem,
                        limiter=limiter,
                    )
                    ccy = _clean_ccy(meta.get("currency")) or "USD"
                    meta_used = True

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

            # attach ids
            df["asset_id"] = asset_id
            df["ticker"] = ticker
            df["yahoo_ticker"] = yahoo_sym
            df["currency"] = ("USD" if is_fx_asset else ccy)

            # USD fields
            df["close_usd"] = (
                pd.to_numeric(df["close"], errors="coerce").astype("float64")
                * pd.to_numeric(df["fx_to_usd"], errors="coerce").astype("float64")
            )
            df["adj_close_usd"] = (
                pd.to_numeric(df["adj_close"], errors="coerce").astype("float64")
                * pd.to_numeric(df["fx_to_usd"], errors="coerce").astype("float64")
            )

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

                to_write = g[~g["date_str"].isin(have_dates)].copy()
                if to_write.empty:
                    continue

                written_parts = store.write_ohlcv_usd_partitioned(to_write.drop(columns=["date_str"]))
                rows_written += int(to_write.shape[0])

                new_dates = sorted(set(to_write["date_str"].tolist()))
                newly_written_dates.update(new_dates)

                store.write_asset_year_manifest(
                    table="ohlcv_usd",
                    asset_id=asset_id,
                    year=int(year),
                    dates=list(have_dates.union(new_dates)),
                    parts=list(have_parts.union(written_parts)),
                )

            if rows_written == 0:
                return {"status": "skip_already_ingested", "asset_id": asset_id}

            # RETURNS
            px = ohlcv_usd[["date", "adj_close_usd"]].copy()
            px["date"] = pd.to_datetime(px["date"], errors="coerce")
            px["date"] = px["date"].dt.normalize()
            px["adj_close_usd"] = pd.to_numeric(px["adj_close_usd"], errors="coerce")
            px = px.dropna(subset=["date", "adj_close_usd"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")

            if (not is_force) and (asset_id in prev_px_map) and (not px.empty):
                first_date = pd.Timestamp(px["date"].iloc[0]).normalize()
                seed_date = first_date - pd.Timedelta(days=1)
                seed_px = float(prev_px_map[asset_id])
                if np.isfinite(seed_px) and seed_px > 0:
                    px = pd.concat(
                        [pd.DataFrame([{"date": seed_date, "adj_close_usd": seed_px}]), px],
                        ignore_index=True,
                    ).sort_values("date")

            px["ret_adj_close_usd"] = px["adj_close_usd"].pct_change()
            px = px.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret_adj_close_usd"])

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
                            "ret_adj_close_usd": ret_new["ret_adj_close_usd"].astype("float64").values,
                        }
                    )
                    returns["date"] = pd.to_datetime(returns["date"], errors="coerce").dt.normalize()
                    returns["year"] = returns["date"].dt.year.astype(int)

                    for year, rg in returns.groupby("year", sort=False):
                        rg = rg.drop(columns=["year"]).copy()
                        rg["date_str"] = rg["date"].dt.strftime("%Y-%m-%d")

                        rman = store.read_asset_year_manifest(table="returns_usd", asset_id=asset_id, year=int(year)) or {}
                        have_dates = set(rman.get("dates", []))
                        have_parts = set(rman.get("parts", []))

                        r_to_write = rg[~rg["date_str"].isin(have_dates)].copy()
                        if r_to_write.empty:
                            continue

                        written_parts = store.write_returns_usd_partitioned(r_to_write.drop(columns=["date_str"]))
                        returns_written += int(r_to_write.shape[0])

                        new_dates = sorted(set(r_to_write["date_str"].tolist()))
                        store.write_asset_year_manifest(
                            table="returns_usd",
                            asset_id=asset_id,
                            year=int(year),
                            dates=list(have_dates.union(new_dates)),
                            parts=list(have_parts.union(written_parts)),
                        )

                    if returns_written > 0:
                        last_return_row = returns.sort_values("date").iloc[-1].to_dict()

            last_price_row = ohlcv_usd.sort_values("date").iloc[-1].to_dict()
            last_date = pd.Timestamp(last_price_row["date"]).date().isoformat()
            last_adj = last_price_row.get("adj_close_usd")

            if freshness_note:
                # pack into last_price_row for visibility (optional)
                last_price_row["_freshness"] = freshness_note

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
                "last_return_row": last_return_row,
                "last_adj_close_usd": float(last_adj) if last_adj is not None else None,
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

    # ---------- RUN IN PARALLEL ----------
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
                latest_prices_rows.append(res["last_price_row"])

                if res.get("last_return_row") is not None:
                    latest_returns_rows.append(res["last_return_row"])
                    d = pd.to_datetime(res["last_return_row"].get("date"), errors="coerce")
                    if pd.notna(d):
                        d = pd.Timestamp(d).normalize()
                        if (max_written_return_date is None) or (d > max_written_return_date):
                            max_written_return_date = d

                total_returns_written += int(res.get("returns_written") or 0)

                aid = res["asset_id"]
                last_state[aid] = res["last_date"]
                provider_state[aid] = res["yahoo_ticker"]

                if res.get("last_adj_close_usd") is not None:
                    prev_px_map[aid] = float(res["last_adj_close_usd"])

                # optional: print staleness warnings
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

            if done % 50 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                print(
                    f"[ingest] done={done}/{n_total} ok={ok} skipped={skipped} failed={failed} "
                    f"rets_written={total_returns_written} fx_cached={len(fx_cache)} "
                    f"rate={rate:.2f} assets/s elapsed={elapsed/60:.1f}m"
                )

    # ---------- SNAPSHOTS / STATE WRITE ----------
    latest_prices_new = pd.DataFrame(latest_prices_rows)
    try:
        old_prices = store.read_latest_prices_snapshot()
        latest_prices = pd.concat([old_prices, latest_prices_new], ignore_index=True)
    except Exception:
        latest_prices = latest_prices_new

    if not latest_prices.empty:
        latest_prices["date"] = pd.to_datetime(latest_prices["date"], errors="coerce")
        try:
            latest_prices["date"] = latest_prices["date"].dt.tz_localize(None)
        except Exception:
            pass
        latest_prices["date"] = latest_prices["date"].dt.normalize()
        latest_prices = latest_prices.dropna(subset=["date"])
        latest_prices["asset_id"] = latest_prices["asset_id"].astype(str).str.strip()
        latest_prices = latest_prices.sort_values("date").drop_duplicates(subset=["asset_id"], keep="last")
        store.write_latest_prices_snapshot(latest_prices.reset_index(drop=True))

    latest_returns_new = pd.DataFrame(latest_returns_rows)
    try:
        old_rets = store.read_latest_returns_snapshot()
        latest_returns = pd.concat([old_rets, latest_returns_new], ignore_index=True)
    except Exception:
        latest_returns = latest_returns_new

    if latest_returns is not None and not latest_returns.empty:
        latest_returns["date"] = pd.to_datetime(latest_returns["date"], errors="coerce")
        try:
            latest_returns["date"] = latest_returns["date"].dt.tz_localize(None)
        except Exception:
            pass
        latest_returns["date"] = latest_returns["date"].dt.normalize()
        latest_returns = latest_returns.dropna(subset=["date"])
        latest_returns["asset_id"] = latest_returns["asset_id"].astype(str).str.strip()
        latest_returns = latest_returns.sort_values("date").drop_duplicates(subset=["asset_id"], keep="last")
        store.write_latest_returns_snapshot(latest_returns.reset_index(drop=True))

    fails = pd.DataFrame(fail_rows)
    as_of = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
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
                "as_of_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last_date": pd.Timestamp(max_written_return_date).strftime("%Y-%m-%d"),
                "n_returns_written": int(total_returns_written),
                "n_assets_total": int(n_total),
                "interval": str(interval),
                "job": "ingest_market_data.py",
            }
        )

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
    print(f"latest_returns_rows_written={len(latest_returns_new)}")
    print(f"returns_written_total={total_returns_written}")
    print(f"[FX] currencies_downloaded={len(fx_cache)} -> {sorted(fx_cache.keys())}")
    print(f"state_entries={len(last_state)}")
    print(f"expected_last_closed_day_utc={expected_last.strftime('%Y-%m-%d')}")
    print(f"elapsed_s={time.time()-t_start:.1f}")
    print(f"[cache] dir={_CACHE_DIR}")

    
if __name__ == "__main__":
    ingest()
