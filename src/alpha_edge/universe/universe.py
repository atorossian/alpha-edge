# universe.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import re
import pandas as pd
import yfinance as yf
from alpha_edge import paths


# =========================
# Core schema
# =========================

@dataclass
class Asset:
    ticker: str
    yahoo_ticker: str
    name: str
    asset_class: str
    role: str
    region: str
    max_weight: float
    min_weight: float
    include: bool


def load_universe(csv_path: str | Path) -> Dict[str, Asset]:
    df = pd.read_csv(csv_path)
    assets: Dict[str, Asset] = {}
    for _, row in df.iterrows():
        if not bool(row.get("include", 1)):
            continue
        asset = Asset(
            ticker=str(row["ticker"]),
            yahoo_ticker=str(row.get("yahoo_ticker") or row["ticker"]),
            name=str(row.get("name", row["ticker"])),
            asset_class=str(row.get("asset_class", "unknown")),
            role=str(row.get("role", "unknown")),
            region=str(row.get("region", "unknown")),
            max_weight=float(row.get("max_weight", 1.0)),
            min_weight=float(row.get("min_weight", 0.0)),
            include=bool(row.get("include", 1)),
        )
        assets[asset.ticker] = asset

    return assets


# =========================
# Yahoo meta enrichment
# =========================

FIELDS = [
    "symbol",
    "shortName",
    "longName",
    "quoteType",
    "exchange",
    "fullExchangeName",
    "currency",
    "market",
    "country",
]

_BAD_QUOTETYPES = {"NONE", "MUTUALFUND", "ECNQUOTE"}
_GOOD_QUOTETYPES = {"EQUITY", "ETF", "ADR"}


def _tokenize(s: str) -> set[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = {t for t in s.split() if len(t) >= 3}
    toks -= {"sa", "ag", "se", "plc", "inc", "corp", "ltd", "sme", "nv", "spa", "holdings"}
    return toks


def name_match_score(expected: str, got: str) -> float:
    a = _tokenize(expected)
    b = _tokenize(got)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a))


def validate_yahoo_meta_for_row(
    meta: dict,
    *,
    expected_name: str,
    expected_region: str,
    asset_class: str,
    role: str,
) -> tuple[bool, str, float]:
    """
    Returns: (validated_ok, reason, name_score)
    """
    qt = str(meta.get("quoteType") or "").upper().strip()
    exch = str(meta.get("exchange") or "").upper().strip()
    full = str(meta.get("fullExchangeName") or "").upper().strip()
    country = str(meta.get("country") or "").upper().strip()
    market = str(meta.get("market") or "").lower().strip()

    if not meta.get("yahoo_ok", False):
        return False, "yahoo_ok=False", 0.0

    if qt in _BAD_QUOTETYPES:
        return False, f"bad_quoteType={qt}", 0.0

    # For equities, we only accept equity-like types
    if asset_class == "equity" and role in {"stock", "etf"}:
        if qt and qt not in _GOOD_QUOTETYPES:
            return False, f"unexpected_quoteType={qt}", 0.0

    reg = (expected_region or "").lower()
    looks_us = (
        (market == "us_market")
        or (country == "UNITED STATES")
        or (exch in {"NMS", "NYQ", "NCM"})
        or ("NASDAQ" in full)
        or ("NYSE" in full)
    )

    if (
        "mexico" in reg
        or "bmv" in reg
        or "mexican" in reg
        or "bolsa mexicana" in reg
    ):
        if looks_us:
            return False, "region_mismatch_looks_us_for_mexico", 0.0

    if (
        "brazil" in reg
        or "b3" in reg
        or "bovespa" in reg
        or "sao paulo" in reg
        or "são paulo" in reg
    ):
        if looks_us:
            return False, "region_mismatch_looks_us_for_brazil", 0.0

    if (
        "europe" in reg
        or "euronext" in reg
        or "cboe europe" in reg
    ):
        if looks_us:
            return False, "region_mismatch_looks_us", 0.0

    got_name = str(meta.get("longName") or meta.get("shortName") or "")
    score = float(name_match_score(expected_name, got_name))
    if score < 0.35:
        return False, f"name_mismatch score={score:.2f}", score

    return True, "ok", score


def fetch_yahoo_meta(ticker: str) -> dict:
    """
    Low-level fetch. Does NOT decide final correctness.
    """
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    meta = {k: info.get(k) for k in FIELDS}
    meta["yahoo_ok"] = bool(meta.get("exchange") or meta.get("quoteType") or meta.get("shortName") or meta.get("longName"))
    return meta


def _assert_unique_key_or_raise(df: pd.DataFrame, key: str, *, context: str) -> None:
    """
    Enrichment MUST NOT deduplicate by itself (it can destroy asset_id integrity).
    If duplicates exist in key, dump them to audit and raise hard.
    """
    if df is None or df.empty:
        return
    if key not in df.columns:
        return

    dup_mask = df[key].duplicated(keep=False)
    if not bool(dup_mask.any()):
        return

    dup = df.loc[dup_mask].copy()

    audit_path = None
    try:
        audit_dir = paths.local_outputs_dir() / "universe_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"enrich_duplicates_{context}_{key}.csv"
        dup.to_csv(audit_path, index=False)
    except Exception:
        audit_path = None

    cols = [key] + [c for c in ["ticker", "yahoo_ticker", "asset_id", "include"] if c in dup.columns]
    sample = dup[cols].head(12)

    msg = (
        f"[enrich] Duplicate key detected: {key} (context={context}). "
        f"rows_involved={len(dup)} unique_keys_duplicated={dup[key].nunique()}."
    )
    if audit_path is not None:
        msg += f" Audit written: {audit_path}."
    msg += "\nSample:\n" + sample.to_string(index=False)

    raise ValueError(msg)


def enrich_universe_csv(
    universe_csv: str,
    out_csv: str,
    sleep_s: float = 0.25,
    *,
    ticker_col: str = "ticker",   # internal id (join key)
    yahoo_col: str = "symbol",    # provider symbol to query
    tickers_subset: list[str] | set[str] | None = None,
) -> pd.DataFrame:
    """
    Enriches universe.csv with Yahoo meta columns.

    CONTRACT:
      - NEVER deduplicates or drops rows by ticker_col.
      - If ticker_col has duplicates, raises hard (and writes audit).
    """
    df = pd.read_csv(universe_csv)

    if ticker_col not in df.columns:
        raise ValueError(f"Missing ticker_col={ticker_col!r} in {universe_csv}")

    df[ticker_col] = df[ticker_col].astype(str).fillna("").str.strip().str.upper()
    _assert_unique_key_or_raise(df, ticker_col, context="input")

    if yahoo_col not in df.columns:
        df[yahoo_col] = df[ticker_col]

    if tickers_subset is not None:
        subset = {str(t).strip().upper() for t in tickers_subset if str(t).strip() and str(t).strip().lower() != "nan"}
        df_sub = df[df[ticker_col].isin(subset)].copy()
    else:
        df_sub = df.copy()

    metas = []
    for _, row in df_sub.iterrows():
        internal = str(row[ticker_col]).strip().upper()
        yahoo_sym = str(row.get(yahoo_col, "")).strip()
        if yahoo_sym == "" or yahoo_sym.lower() == "nan":
            yahoo_sym = internal

        meta = fetch_yahoo_meta(yahoo_sym)

        # avoid collision with universe df columns
        if "symbol" in meta:
            meta["yahoo_symbol_returned"] = meta.pop("symbol")

        meta[ticker_col] = internal
        meta["yahoo_symbol_used"] = yahoo_sym
        metas.append(meta)

        print(
            f"[meta] {internal} (yahoo={yahoo_sym}) -> ok={meta.get('yahoo_ok')} "
            f"quoteType={meta.get('quoteType')} exchange={meta.get('exchange')} country={meta.get('country')}"
        )
        time.sleep(float(sleep_s))

    meta_df = pd.DataFrame(metas)
    if meta_df.empty:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved (no enrichment performed): {out_csv}")
        return df

    meta_df[ticker_col] = meta_df[ticker_col].astype(str).fillna("").str.strip().str.upper()
    _assert_unique_key_or_raise(meta_df, ticker_col, context="meta_df")
    meta_df = meta_df.set_index(ticker_col)

    overlap = [c for c in meta_df.columns if c in df.columns]
    if overlap:
        df = df.drop(columns=overlap)

    out = df.set_index(ticker_col).join(meta_df, how="left").reset_index()
    _assert_unique_key_or_raise(out, ticker_col, context="output")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    return out


# =========================
# Robust ticker resolution
# =========================

EU_SUFFIXES = [
    ".DE", ".MU", ".F", ".DU", ".HM",
    ".MC", ".PA", ".MI", ".AS",
    ".BR", ".SW", ".ST", ".L",
    ".SA", ".MX", ".HE",
]

PREF_SUFFIX_ORDER = [
    ".DE", ".MC", ".PA", ".MI", ".AS", ".L", ".SW",
    ".BR", ".ST",
    ".SA", ".MX",
    ".MU", ".F", ".DU", ".HM",
    ".HE",
    "",
]


def _venue_rank(sym: str) -> int:
    sym = str(sym).upper()
    for i, suf in enumerate(PREF_SUFFIX_ORDER):
        if suf == "":
            if "." not in sym:
                return i
        elif sym.endswith(suf):
            return i
    return 999


REGION_SUFFIX_HINTS = {
    "cboe europe": [".DE", ".MC", ".PA", ".MI", ".AS", ".L", ".SW", ".BR", ".ST", ".MU", ".F", ".DU", ".HM"],
    "spain": [".MC"],
    "madrid": [".MC"],
    "bme": [".MC"],
    "germany": [".DE", ".MU", ".F", ".DU", ".HM"],
    "xetra": [".DE"],
    "france": [".PA"],
    "paris": [".PA"],
    "italy": [".MI"],
    "milan": [".MI"],
    "netherlands": [".AS"],
    "amsterdam": [".AS"],
    "london": [".L"],
    "uk": [".L"],
    "switzerland": [".SW"],
    "six": [".SW"],
    "sweden": [".ST"],
    "stockholm": [".ST"],
    "belgium": [".BR"],
    "brussels": [".BR"],
    "brazil": [".SA"],
    "b3": [".SA"],
    "bovespa": [".SA"],
    "sao paulo": [".SA"],
    "são paulo": [".SA"],
    "bm&fbovespa": [".SA"],
    "bvmf": [".SA"],
    "b3 -": [".SA"],
    "mexico": [".MX"],
    "bmv": [".MX"],
    "bolsa mexicana": [".MX"],
    "mexican": [".MX"],
    "mexico city": [".MX"],
    "finland": [".HE"],
    "helsinki": [".HE"],
    "nasdaq helsinki": [".HE"],
}


def _has_price_history(ticker: str) -> bool:
    try:
        df = yf.download(ticker, period="10d", progress=False, auto_adjust=False, threads=False)
        return (df is not None) and (not df.empty) and ("Close" in df.columns) and (df["Close"].dropna().shape[0] >= 1)
    except Exception:
        return False


def _candidate_symbols(raw: str, region: str | None) -> list[str]:
    raw = str(raw).strip().upper()
    reg = (region or "").strip().lower()

    candidates: list[str] = [raw]

    for key, sufs in REGION_SUFFIX_HINTS.items():
        if key in reg:
            for suf in sufs:
                candidates.append(raw + suf)

    if re.match(r"^[A-Z]{4}\d{1,2}$", raw):
        candidates.append(raw + ".SA")

    for suf in EU_SUFFIXES:
        candidates.append(raw + suf)

    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def resolve_ticker_strict(
    raw: str,
    *,
    expected_name: str,
    expected_region: str,
    asset_class: str,
    role: str,
) -> tuple[Optional[str], dict]:
    best = None
    best_score = -1.0
    best_rank = 999
    best_meta = None
    best_reason = None

    for cand in _candidate_symbols(raw, expected_region):
        meta = fetch_yahoo_meta(cand)

        ok, reason, score = validate_yahoo_meta_for_row(
            meta,
            expected_name=expected_name,
            expected_region=expected_region,
            asset_class=asset_class,
            role=role,
        )

        px_ok = _has_price_history(cand)
        if not px_ok:
            ok = False
            reason = f"{reason}; no_price_history"

        if ok:
            r = _venue_rank(cand)
            if (score > best_score) or (abs(score - best_score) < 1e-12 and r < best_rank):
                best = cand
                best_score = score
                best_rank = r
                best_meta = meta
                best_reason = reason

            if score >= 0.75:
                break

    debug = {
        "resolved": best,
        "best_score": best_score if best is not None else None,
        "best_rank": best_rank if best is not None else None,
        "best_reason": best_reason,
        "best_meta_quoteType": None if best_meta is None else best_meta.get("quoteType"),
        "best_meta_exchange": None if best_meta is None else best_meta.get("exchange"),
        "best_meta_country": None if best_meta is None else best_meta.get("country"),
    }
    return best, debug
