# ingest_market_data.py
from __future__ import annotations

import datetime as dt
import threading
from typing import Optional
from alpha_edge import paths

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from alpha_edge.core.market_store import MarketStore
from alpha_edge.jobs.run_universe_triage import run_post_ingest_triage


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
        # Normalize to day boundary to match how we store 'date'
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
        # Use UTC day boundary to be consistent for everyone
        end = pd.Timestamp.utcnow().normalize()
        return end.strftime("%Y-%m-%d")
    return None


def fetch_yahoo_currency(ticker: str) -> dict:
    """
    Lightweight-ish metadata fetch. Still a network call.
    """
    t = yf.Ticker(ticker)
    out = {"ticker": ticker, "currency": None}

    try:
        fi = getattr(t, "fast_info", None) or {}
        if isinstance(fi, dict):
            out["currency"] = fi.get("currency")
            out["exchange"] = fi.get("exchange")
    except Exception:
        pass

    if not out.get("currency"):
        try:
            info = t.info or {}
            out["currency"] = info.get("currency")
            out["exchange"] = info.get("exchange")
            out["quoteType"] = info.get("quoteType")
        except Exception:
            pass

    if out.get("currency"):
        out["currency"] = str(out["currency"]).upper().strip()

    return out


def download_ohlcv(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # normalize
    df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    df.index = pd.to_datetime(df.index)
    df = df.reset_index().rename(columns={"index": "date"})
    df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]

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


def download_fx_to_usd_series(ccy: str, start: str) -> pd.Series:
    """
    Returns series: FX rate to convert 1 unit of CCY into USD.
    """
    ccy = str(ccy).upper().strip()
    if ccy == "USD":
        raise ValueError("USD has no FX series")

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
                interval="1d",
                progress=False,
                threads=False,
                auto_adjust=False,
            )
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
) -> pd.Series:
    """
    Returns FX rate series aligned to `dates`, using an in-memory cache.
    - downloads once per currency per run
    - forward-fills to requested dates
    """
    ccy = str(ccy).upper().strip()
    dates_norm = _normalize_day_index(dates)

    if ccy == "USD":
        return pd.Series(1.0, index=dates_norm, name="USD")

    if ccy not in fx_cache:
        fx_cache[ccy] = download_fx_to_usd_series(ccy, start=start_base)

    s = fx_cache[ccy].copy()
    s.index = _normalize_day_index(s.index)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]

    out = s.reindex(dates_norm).ffill()
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
# NEW: global rate limiter + retry wrapper for yfinance
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
            sleep_s = float(base_sleep) * (2 ** k) * (0.7 + 0.6 * random.random())
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
    yahoo_max_concurrency: int = 2,   # concurrency cap
    yahoo_rate_per_sec: float = 1.5,  # NEW: global rate cap (across all threads)
) -> None:
    import time

    t_start = time.time()

    # Shared caches/state (thread-safe access enforced)
    fx_cache: dict[str, pd.Series] = {}
    fx_lock = threading.Lock()

    yf_sem = threading.Semaphore(int(yahoo_max_concurrency))
    limiter = RateLimiter(rate_per_sec=float(yahoo_rate_per_sec))

    store = MarketStore(bucket=bucket)
    end = safe_end_date_for_interval(interval)

    # ---- Universe ----
    u = pd.read_csv(universe_csv)
    u = u[u.get("include", 1).fillna(1).astype(int) == 1].copy()

    if "asset_id" not in u.columns:
        raise RuntimeError("Universe CSV must include 'asset_id' column (partition key).")

    u["asset_id"] = u["asset_id"].astype(str).str.strip()
    u["ticker"] = u.get("ticker", u["asset_id"]).astype(str).str.strip()
    u["yahoo_ticker"] = u.get("yahoo_ticker", u["ticker"]).astype(str).str.strip()

    # NEW: allow universe to carry currency so we can skip metadata calls
    u["currency"] = u.get("currency", "").astype(str).str.strip().str.upper()

    # Pass currency into worker so we avoid an extra Yahoo call per asset when possible
    triples = list(zip(
        u["asset_id"].tolist(),
        u["ticker"].tolist(),
        u["yahoo_ticker"].tolist(),
        u["currency"].tolist(),
    ))
    if max_tickers:
        triples = triples[:max_tickers]
    n_total = len(triples)

    # ---- Force refresh list (by asset_id) ----
    force_refresh: set[str] = set()
    if force_refresh_csv:
        try:
            fr = pd.read_csv(force_refresh_csv)
            if "asset_id" in fr.columns:
                force_refresh = set(fr["asset_id"].astype(str).str.strip().tolist())
        except Exception:
            force_refresh = set()

    # ---- State (by asset_id) ----
    last_state = store.read_last_date_state()              # asset_id -> "YYYY-MM-DD"
    provider_state = store.read_provider_symbol_state()    # asset_id -> yahoo_sym used last run

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

    # Results accumulators (ONLY mutated in main thread)
    latest_prices_rows: list[dict] = []
    latest_returns_rows: list[dict] = []
    fail_rows: list[dict] = []

    max_written_return_date: pd.Timestamp | None = None
    total_returns_written = 0

    # ---------- helpers captured by closure ----------
    def get_fx_locked_local(ccy: str, dates: pd.DatetimeIndex) -> pd.Series:
        with fx_lock:
            return get_fx_to_usd_for_dates(
                ccy=ccy,
                dates=dates,
                start_base=start_base,
                fx_cache=fx_cache,
            )

    def _process_one_local(asset_id: str, ticker: str, yahoo_sym: str, currency_hint: str) -> dict:
        asset_id = str(asset_id).strip()
        ticker = str(ticker).strip().upper()
        yahoo_sym = (str(yahoo_sym).strip() or ticker).upper()
        currency_hint = (str(currency_hint).strip() or "").upper()

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

        try:
            # 1) Download OHLCV (required)
            df = call_yf_with_retries(
                lambda: download_ohlcv(yahoo_sym, start=start, end=end, interval=interval),
                sem=yf_sem,
                limiter=limiter,
            )

            if df is None or df.empty:
                return {
                    "status": "empty",
                    "asset_id": asset_id,
                    "ticker": ticker,
                    "yahoo_ticker": yahoo_sym,
                    "start": start,
                }

            # 2) Currency: use universe hint first to avoid an extra Yahoo metadata call
            ccy = currency_hint
            if (not ccy) or (ccy.lower() == "nan"):
                meta = call_yf_with_retries(
                    lambda: fetch_yahoo_currency(yahoo_sym),
                    sem=yf_sem,
                    limiter=limiter,
                )
                ccy = (meta.get("currency") or "USD").upper().strip()

            # normalize
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col not in df.columns:
                    df[col] = np.nan

            df["asset_id"] = asset_id
            df["ticker"] = ticker
            df["yahoo_ticker"] = yahoo_sym
            df["currency"] = ccy

            try:
                df["date"] = df["date"].dt.tz_localize(None)
            except Exception:
                pass
            df["date"] = df["date"].dt.normalize()
            df = df.dropna(subset=["date"]).sort_values("date")

            # FX
            if ccy == "USD":
                df["fx_to_usd"] = 1.0
            else:
                fx_s = get_fx_locked_local(ccy, pd.DatetimeIndex(df["date"]))
                if fx_s.isna().all():
                    return {
                        "status": "no_fx",
                        "asset_id": asset_id,
                        "ticker": ticker,
                        "yahoo_ticker": yahoo_sym,
                        "ccy": ccy,
                        "start": start,
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
                            "ccy": ccy,
                            "start": start,
                        }

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
                    "date", "asset_id", "ticker", "yahoo_ticker",
                    "open", "high", "low", "close", "adj_close", "volume",
                    "currency", "fx_to_usd", "close_usd", "adj_close_usd",
                ]
            ].copy()

            ohlcv_usd = ohlcv_usd.sort_values("date").drop_duplicates(subset=["asset_id", "date"], keep="last")
            ohlcv_usd["year"] = ohlcv_usd["date"].dt.year.astype(int)

            rows_written = 0
            newly_written_dates: set[str] = set()

            # ---------- OHLCV manifest gating (read ONCE per year) ----------
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

            # ---------- RETURNS for newly written dates only ----------
            px = ohlcv_usd[["date", "adj_close_usd"]].copy()
            px["date"] = pd.to_datetime(px["date"], errors="coerce")
            try:
                px["date"] = px["date"].dt.tz_localize(None)
            except Exception:
                pass
            px["date"] = px["date"].dt.normalize()
            px["adj_close_usd"] = pd.to_numeric(px["adj_close_usd"], errors="coerce")
            px = px.dropna(subset=["date", "adj_close_usd"]).sort_values("date").drop_duplicates(
                subset=["date"], keep="last"
            )

            # seed previous close for incremental runs
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

                    # manifest-gate returns (read ONCE per year)
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

            return {
                "status": "ok",
                "asset_id": asset_id,
                "ticker": ticker,
                "yahoo_ticker": yahoo_sym,
                "ccy": ccy,
                "start": start,
                "ohlcv_rows_written": rows_written,
                "returns_written": returns_written,
                "last_date": last_date,
                "last_price_row": last_price_row,
                "last_return_row": last_return_row,
                "last_adj_close_usd": float(last_adj) if last_adj is not None else None,
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

                # update state maps (main thread)
                aid = res["asset_id"]
                last_state[aid] = res["last_date"]
                provider_state[aid] = res["yahoo_ticker"]

                if res.get("last_adj_close_usd") is not None:
                    prev_px_map[aid] = float(res["last_adj_close_usd"])

            elif st in {"skip_up_to_date", "skip_already_ingested"}:
                skipped += 1
            else:
                failed += 1
                fail_rows.append(
                    {
                        "as_of": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
                        "asset_id": res.get("asset_id"),
                        "ticker": res.get("ticker"),
                        "yahoo_ticker": res.get("yahoo_ticker"),
                        "start": res.get("start"),
                        "interval": interval,
                        "reason": st,
                        "error": res.get("error"),
                    }
                )

            if done % 50 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                print(
                    f"[ingest] done={done}/{n_total} ok={ok} skipped={skipped} failed={failed} "
                    f"rets_written={total_returns_written} fx_cached={len(fx_cache)} "
                    f"rate={rate:.2f} assets/s elapsed={elapsed/60:.1f}m"
                )

    # ---------- SNAPSHOTS / STATE WRITE (single-threaded) ----------
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

    # Local debug copy (easy to inspect)
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
    print(f"elapsed_s={time.time()-t_start:.1f}")


if __name__ == "__main__":
    ingest()
