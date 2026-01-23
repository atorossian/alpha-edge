from __future__ import annotations

import argparse
import hashlib
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import numpy as np

from alpha_edge.universe.universe import resolve_ticker_strict, enrich_universe_csv, enrich_universe_csv_patch


UNIVERSE_CSV = "data/universe/universe.csv"

QF_PAGES = {
    "etf": "https://help.quantfury.com/en/articles/5448760-etfs",
    "crypto": "https://help.quantfury.com/en/articles/5448756-cryptocurrency-pairs",
    "stock": "https://help.quantfury.com/en/articles/5448752-stocks",
}

STABLE_QUOTES = {"USD", "USDT", "USDC"}

SAFE_SCRAPE_UPDATE_COLS = ["name", "asset_class", "role", "region", "include", "max_weight", "min_weight"]
OVERRIDES_CSV = "data/universe/universe_overrides.csv"
ASSET_EXCLUDED_CSV = "data/universe/asset_excluded.csv"

SNAP_DIR = Path("data/universe/.snapshots")
SNAP_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def _read_csv_or_empty(path: str, cols: list[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=cols or [])
    df = pd.read_csv(p)
    if df is None or df.empty:
        return pd.DataFrame(columns=cols or list(df.columns))
    return df


def _norm_ticker(s: object) -> str:
    x = "" if s is None else str(s)
    x = x.strip().upper()
    if x.lower() == "nan":
        return ""
    return x


def fetch_html(url: str, timeout: int = 20) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; alpha-edge-universe/1.0)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def parse_first_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody")
    if tbody is None:
        raise ValueError("No <tbody> found on page (page layout may have changed)")

    rows: list[list[str]] = []
    for tr in tbody.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cols:
            rows.append(cols)

    if not rows:
        raise ValueError("No table rows found")

    header = rows[0]
    data_rows = rows[1:]
    return pd.DataFrame(data_rows, columns=header)


def canonical_crypto_pair(pair: str) -> str:
    s = str(pair).strip().upper().replace("/", "-")
    if "-" not in s:
        return s
    base, quote = s.split("-", 1)
    base = base.strip()
    quote = quote.strip()
    if quote in STABLE_QUOTES:
        quote = "USD"
    return f"{base}-{quote}"


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = {
        "asset_id": "",
        "ticker": None,
        "yahoo_ticker": None,
        "name": None,
        "asset_class": None,
        "role": None,
        "region": None,
        "max_weight": 0.25,
        "min_weight": 0.0,
        "include": 1,
    }
    for c, default in base_cols.items():
        if c not in df.columns:
            df[c] = default

    df["ticker"] = df["ticker"].astype(str)
    df["yahoo_ticker"] = df["yahoo_ticker"].astype(str)
    df["name"] = df["name"].astype(str)
    df["include"] = df["include"].fillna(1).astype(int)

    return df


def load_existing_universe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ----------------------------
# Builders (full mode only)
# ----------------------------
def build_etfs() -> pd.DataFrame:
    df = parse_first_table(fetch_html(QF_PAGES["etf"]))
    out = pd.DataFrame({
        "ticker": df["Ticker"].astype(str).str.upper().str.strip(),
        "yahoo_ticker": df["Ticker"].astype(str).str.upper().str.strip(),
        "name": df["ETF"].astype(str),
        "asset_class": "equity",
        "role": "etf",
        "region": df["Exchange"].astype(str),
        "max_weight": 0.25,
        "min_weight": 0.0,
        "include": 1,
    })
    return ensure_columns(out)


def build_stocks() -> pd.DataFrame:
    df = parse_first_table(fetch_html(QF_PAGES["stock"]))
    out = pd.DataFrame({
        "ticker": df["Ticker"].astype(str).str.upper().str.strip(),
        "yahoo_ticker": df["Ticker"].astype(str).str.upper().str.strip(),
        "name": df["Company"].astype(str),
        "asset_class": "equity",
        "role": "stock",
        "region": df["Exchange"].astype(str),
        "max_weight": 0.25,
        "min_weight": 0.0,
        "include": 1,
    })
    return ensure_columns(out)


def build_crypto_pairs() -> pd.DataFrame:
    df = parse_first_table(fetch_html(QF_PAGES["crypto"]))
    pair_col = "Pair" if "Pair" in df.columns else df.columns[0]
    internal = df[pair_col].astype(str).apply(canonical_crypto_pair)

    out = pd.DataFrame({
        "ticker": internal,
        "yahoo_ticker": internal,
        "name": internal,
        "asset_class": "crypto",
        "role": "crypto",
        "region": df["Exchange"].astype(str) if "Exchange" in df.columns else "crypto",
        "max_weight": 0.25,
        "min_weight": 0.0,
        "include": 1,
    })
    return ensure_columns(out)


# ----------------------------
# Governance (patch-friendly)
# ----------------------------
def load_overrides(path: str) -> pd.DataFrame:
    base_cols = ["ticker","yahoo_ticker","lock_yahoo_ticker","exclude","exclude_reason","expected_name","note"]
    df = _read_csv_or_empty(path, cols=base_cols)
    for c in base_cols:
        if c not in df.columns:
            df[c] = "" if c in {"ticker","yahoo_ticker","exclude_reason","expected_name","note"} else 0

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["yahoo_ticker"] = df["yahoo_ticker"].astype(str).str.strip()
    df["lock_yahoo_ticker"] = pd.to_numeric(df["lock_yahoo_ticker"], errors="coerce").fillna(1).astype(int)
    df["exclude"] = pd.to_numeric(df["exclude"], errors="coerce").fillna(0).astype(int)
    return df[base_cols]


def load_exclusions(path: str) -> pd.DataFrame:
    df = _read_csv_or_empty(path, cols=["ticker", "asset_class", "reason"])
    if df.empty:
        return df
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["asset_class"] = df.get("asset_class", "").astype(str).str.strip().str.lower()
    df["reason"] = df.get("reason", "").astype(str)
    return df


def apply_asset_exclusions_patch(universe: pd.DataFrame, excluded_df: pd.DataFrame) -> pd.DataFrame:
    """
    PATCH-FRIENDLY: do not drop rows. Instead set include=0 for excluded.
    If not excluded, leave include as-is (do not force include=1).
    """
    u = universe.copy()
    if "include" not in u.columns:
        u["include"] = 1
    u["ticker"] = u.get("ticker", "").astype(str).str.strip().str.upper()
    u["asset_class"] = u.get("asset_class", "").astype(str).str.strip().str.lower()
    u["include"] = u["include"].fillna(1).astype(int)

    if excluded_df is None or excluded_df.empty:
        return u

    ex = excluded_df.copy()
    ex["ticker"] = ex["ticker"].astype(str).str.strip().str.upper()
    ex["asset_class"] = ex.get("asset_class", "").astype(str).str.strip().str.lower()

    # two types:
    # - global ticker excludes (asset_class empty)
    # - scoped excludes (ticker+asset_class)
    ex_any = ex[ex["asset_class"].isin(["", "nan"])]
    ex_scoped = ex[~ex["asset_class"].isin(["", "nan"])]

    if not ex_any.empty:
        u.loc[u["ticker"].isin(set(ex_any["ticker"])), "include"] = 0

    if not ex_scoped.empty:
        key_u = u["ticker"].astype(str) + "||" + u["asset_class"].astype(str)
        key_ex = ex_scoped["ticker"].astype(str) + "||" + ex_scoped["asset_class"].astype(str)
        u.loc[key_u.isin(set(key_ex)), "include"] = 0

    return u.reset_index(drop=True)


def apply_overrides_patch(universe: pd.DataFrame, overrides: pd.DataFrame, *, affected: set[str]) -> pd.DataFrame:
    """
    Apply overrides ONLY for affected tickers:
      - exclude => include=0
      - yahoo_ticker override when lock_yahoo_ticker==1
      - do not drop rows
    """
    if overrides is None or overrides.empty or not affected:
        return universe

    u = universe.copy()
    u["ticker"] = u.get("ticker", "").astype(str).str.strip().str.upper()
    if "yahoo_ticker" not in u.columns:
        u["yahoo_ticker"] = u["ticker"]
    if "include" not in u.columns:
        u["include"] = 1
    u["include"] = u["include"].fillna(1).astype(int)

    o = overrides.copy()
    o["ticker"] = o["ticker"].astype(str).str.strip().str.upper()

    o = o[o["ticker"].isin(affected)].copy()
    if o.empty:
        return u

    o_map = o.set_index("ticker")

    for t in affected:
        if t not in set(u["ticker"]):
            continue
        if t not in o_map.index:
            continue

        row = o_map.loc[t]
        # exclude => include=0
        ex = int(row.get("exclude", 0) or 0)
        if ex == 1:
            u.loc[u["ticker"] == t, "include"] = 0

        # yahoo_ticker override if locked
        lock = int(row.get("lock_yahoo_ticker", 1) or 0)
        y = str(row.get("yahoo_ticker", "")).strip()
        if lock == 1 and y and y.lower() != "nan":
            u.loc[u["ticker"] == t, "yahoo_ticker"] = y

        # keep lock column visible in universe
        u.loc[u["ticker"] == t, "lock_yahoo_ticker"] = lock

    return u.reset_index(drop=True)


# ----------------------------
# Resolver (patch-friendly)
# ----------------------------
def resolve_bad_yahoo_symbols(universe_csv: str, *, tickers_subset: list[str] | None = None) -> list[str]:
    """
    If tickers_subset provided, only attempt resolve on those tickers.
    Returns list of tickers whose yahoo_ticker changed.
    """
    df = pd.read_csv(universe_csv)
    if df.empty:
        return []

    df["ticker"] = df.get("ticker", "").astype(str).str.strip().str.upper()
    df["yahoo_ticker"] = df.get("yahoo_ticker", df["ticker"]).astype(str).str.strip()

    if tickers_subset:
        subset = set([_norm_ticker(t) for t in tickers_subset if _norm_ticker(t)])
        df = df[df["ticker"].isin(subset)].copy()
        if df.empty:
            return []
        # We'll write back by merging later
        full = pd.read_csv(universe_csv)
        full["ticker"] = full.get("ticker", "").astype(str).str.strip().str.upper()
    else:
        full = None

    locked = df.get("lock_yahoo_ticker", 0).fillna(0).astype(int) == 1
    bad = (df.get("yahoo_ok").fillna(False) == False) & (~locked)

    qt = df.get("quoteType").fillna("").astype(str).str.upper()
    bad |= qt.isin(["NONE", "MUTUALFUND", "ECNQUOTE"])

    fixed = []
    if "symbol" not in df.columns:
        df["symbol"] = df["yahoo_ticker"]
    if "resolver_debug" not in df.columns:
        df["resolver_debug"] = None

    for i, row in df[bad].iterrows():
        raw = str(row.get("yahoo_ticker") or row.get("ticker") or "").strip()
        if not raw:
            continue

        resolved, dbg = resolve_ticker_strict(
            raw,
            expected_name=str(row.get("name") or ""),
            expected_region=str(row.get("region") or ""),
            asset_class=str(row.get("asset_class") or ""),
            role=str(row.get("role") or ""),
        )

        if resolved and resolved != raw:
            df.at[i, "yahoo_ticker"] = resolved
            df.at[i, "symbol"] = resolved
            fixed.append(str(row.get("ticker")).strip().upper())

        df.at[i, "resolver_debug"] = json.dumps(dbg, ensure_ascii=False)

    # write back
    if full is None:
        # overwrite whole csv
        df_all = pd.read_csv(universe_csv)
        df_all["ticker"] = df_all.get("ticker", "").astype(str).str.strip().str.upper()
        patch = df[["ticker", "yahoo_ticker", "symbol", "resolver_debug"]].copy()
        df_all = df_all.merge(patch, on="ticker", how="left", suffixes=("", "_p"))

        for c in ["yahoo_ticker", "symbol", "resolver_debug"]:
            cp = f"{c}_p"
            if cp in df_all.columns:
                df_all[c] = df_all[c].where(df_all[cp].isna(), df_all[cp])
                df_all.drop(columns=[cp], inplace=True)

        df_all.to_csv(universe_csv, index=False)
    else:
        patch = df[["ticker", "yahoo_ticker", "symbol", "resolver_debug"]].copy()
        out = full.merge(patch, on="ticker", how="left", suffixes=("", "_p"))
        for c in ["yahoo_ticker", "symbol", "resolver_debug"]:
            cp = f"{c}_p"
            if cp in out.columns:
                out[c] = out[c].where(out[cp].isna(), out[cp])
                out.drop(columns=[cp], inplace=True)
        out.to_csv(universe_csv, index=False)

    return sorted(set(fixed))


# ----------------------------
# Asset ID assignment (same logic as you already adopted)
# ----------------------------
def _clean_str(x: object) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    if s.lower() == "nan":
        return ""
    return s


def _is_valid_isin(x: str) -> bool:
    x = _clean_str(x).upper()
    return len(x) == 12 and x.isalnum()


def _stable_hash(parts: list[str], *, prefix: str, n: int = 16) -> str:
    key = "|".join([_clean_str(p).upper() for p in parts if _clean_str(p)])
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}{h}"


def assign_asset_ids(universe: pd.DataFrame, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    u = universe.copy()
    if "asset_id" not in u.columns:
        u["asset_id"] = ""

    u["ticker"] = u.get("ticker", "").astype(str).str.strip().str.upper()
    u["asset_class"] = u.get("asset_class", "").astype(str).str.strip().str.lower()

    # preserve by ticker
    if existing is not None and not existing.empty and "asset_id" in existing.columns:
        ex = existing.copy()
        ex["ticker"] = ex.get("ticker", "").astype(str).str.strip().str.upper()
        ex["asset_id"] = ex.get("asset_id", "").astype(str).str.strip()
        ex_map = ex.set_index("ticker")["asset_id"].to_dict()

        cur = u["asset_id"].astype(str).str.strip()
        empty = cur.isna() | (cur == "") | (cur.str.lower() == "nan")
        u.loc[empty, "asset_id"] = u.loc[empty, "ticker"].map(ex_map).fillna("")

    # ISIN
    if "isin" in u.columns:
        isin = u["isin"].astype(str)
        isin_ok = isin.apply(_is_valid_isin)
        empty = u["asset_id"].astype(str).str.strip().replace({"nan": ""}) == ""
        u.loc[empty & isin_ok, "asset_id"] = isin[empty & isin_ok].astype(str).str.upper().str.strip()

    # Crypto
    empty = u["asset_id"].astype(str).str.strip().replace({"nan": ""}) == ""
    is_crypto = u["asset_class"].astype(str).str.lower() == "crypto"
    u.loc[empty & is_crypto, "asset_id"] = "CRYPTO:" + u.loc[empty & is_crypto, "ticker"].astype(str).str.upper()

    # fallback hash
    empty = u["asset_id"].astype(str).str.strip().replace({"nan": ""}) == ""
    if empty.any():
        name = u.get("name", "")
        exchange = u.get("exchange", u.get("fullExchangeName", ""))
        country = u.get("country", "")
        currency = u.get("currency", "")
        role = u.get("role", "")

        for idx in u[empty].index:
            parts = [
                u.at[idx, "ticker"],
                name[idx] if hasattr(name, "__getitem__") else "",
                exchange[idx] if hasattr(exchange, "__getitem__") else "",
                country[idx] if hasattr(country, "__getitem__") else "",
                currency[idx] if hasattr(currency, "__getitem__") else "",
                role[idx] if hasattr(role, "__getitem__") else "",
            ]
            u.at[idx, "asset_id"] = _stable_hash(parts, prefix="EQH")

    u["asset_id"] = u["asset_id"].astype(str).str.strip()
    return u


# ----------------------------
# Patch diff logic
# ----------------------------
def _snapshot_path(name: str) -> Path:
    return SNAP_DIR / name


def _diff_overrides(prev: pd.DataFrame, curr: pd.DataFrame) -> set[str]:
    cols = ["ticker", "yahoo_ticker", "lock_yahoo_ticker", "exclude"]
    for c in cols:
        if c not in prev.columns:
            prev[c] = ""
        if c not in curr.columns:
            curr[c] = ""

    p = prev.copy()
    c = curr.copy()
    p["ticker"] = p["ticker"].astype(str).str.strip().str.upper()
    c["ticker"] = c["ticker"].astype(str).str.strip().str.upper()

    p = p[cols].fillna("")
    c = c[cols].fillna("")

    p_map = p.set_index("ticker").to_dict(orient="index")
    c_map = c.set_index("ticker").to_dict(orient="index")

    all_t = set(p_map.keys()) | set(c_map.keys())
    affected = set()
    for t in all_t:
        if not t or t.lower() == "nan":
            continue
        if t not in p_map or t not in c_map:
            affected.add(t)
            continue
        if p_map[t] != c_map[t]:
            affected.add(t)

    return affected


def _diff_exclusions(prev: pd.DataFrame, curr: pd.DataFrame) -> set[str]:
    cols = ["ticker", "asset_class"]
    for c in cols:
        if c not in prev.columns:
            prev[c] = ""
        if c not in curr.columns:
            curr[c] = ""

    p = prev.copy()
    c = curr.copy()
    p["ticker"] = p["ticker"].astype(str).str.strip().str.upper()
    c["ticker"] = c["ticker"].astype(str).str.strip().str.upper()
    p["asset_class"] = p["asset_class"].astype(str).str.strip().str.lower()
    c["asset_class"] = c["asset_class"].astype(str).str.strip().str.lower()

    p = p[cols].fillna("")
    c = c[cols].fillna("")

    # compare sets of (ticker, asset_class)
    sp = set((r["ticker"], r["asset_class"]) for _, r in p.iterrows() if r["ticker"])
    sc = set((r["ticker"], r["asset_class"]) for _, r in c.iterrows() if r["ticker"])

    changed = sp.symmetric_difference(sc)
    return set(t for (t, _) in changed if t)


# ----------------------------
# MAIN
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "patch"], default="full")
    ap.add_argument("--sleep_s", type=float, default=0.25)
    args = ap.parse_args()

    mode = args.mode
    sleep_s = float(args.sleep_s)

    existing = load_existing_universe(UNIVERSE_CSV)

    overrides = load_overrides(OVERRIDES_CSV)
    exclusions = load_exclusions(ASSET_EXCLUDED_CSV)

    if mode == "full":
        etfs = build_etfs()
        stocks = build_stocks()
        crypto = build_crypto_pairs()
        scraped = pd.concat([etfs, stocks, crypto], ignore_index=True)
        scraped = scraped.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)

        universe = existing.merge(
            scraped[["ticker"] + SAFE_SCRAPE_UPDATE_COLS + ["yahoo_ticker"]],
            on="ticker",
            how="outer",
            suffixes=("", "_scr"),
        )

        for c in SAFE_SCRAPE_UPDATE_COLS:
            sc = f"{c}_scr"
            if sc in universe.columns:
                universe[c] = universe[c].where(universe[sc].isna(), universe[sc])
                universe.drop(columns=[sc], inplace=True)

        if "yahoo_ticker_scr" in universe.columns:
            ex = universe["yahoo_ticker"].astype(str)
            ex_empty = ex.isna() | (ex.str.strip() == "") | (ex.str.lower() == "nan")
            universe.loc[ex_empty, "yahoo_ticker"] = universe.loc[ex_empty, "yahoo_ticker_scr"]
            universe.drop(columns=["yahoo_ticker_scr"], inplace=True)

        yt = universe["yahoo_ticker"].astype(str)
        yt_empty = yt.isna() | (yt.str.strip() == "") | (yt.str.lower() == "nan")
        universe.loc[yt_empty, "yahoo_ticker"] = universe.loc[yt_empty, "ticker"]

        universe = ensure_columns(universe)

        # exclusions + overrides first
        universe = apply_asset_exclusions_patch(universe, exclusions)

        affected_all = set(universe["ticker"].astype(str).str.strip().str.upper().tolist())
        universe = apply_overrides_patch(universe, overrides, affected=affected_all)

        # normalize before selecting live tickers
        universe["ticker"] = universe["ticker"].astype(str).str.strip().str.upper()
        universe["include"] = universe["include"].fillna(1).astype(int)

        universe.to_csv(UNIVERSE_CSV, index=False)
        print(f"[ok] wrote merged {len(universe)} rows -> {UNIVERSE_CSV}")

        tickers_live = sorted(universe.loc[universe["include"] == 1, "ticker"].dropna().unique().tolist())

        enrich_universe_csv(
            universe_csv=UNIVERSE_CSV,
            out_csv=UNIVERSE_CSV,
            sleep_s=sleep_s,
            ticker_col="ticker",
            yahoo_col="yahoo_ticker",
            tickers_subset=tickers_live,
        )

        resolve_bad_yahoo_symbols(UNIVERSE_CSV, tickers_subset=tickers_live)

        enrich_universe_csv(
            universe_csv=UNIVERSE_CSV,
            out_csv=UNIVERSE_CSV,
            sleep_s=sleep_s,
            ticker_col="ticker",
            yahoo_col="yahoo_ticker",
            tickers_subset=tickers_live,
        )

        u2 = pd.read_csv(UNIVERSE_CSV)
        u2 = assign_asset_ids(u2, existing=existing)
        u2.to_csv(UNIVERSE_CSV, index=False)
        print("[ok] asset_id assigned.")


    else:
        # --- patch mode ---
        if existing is None or existing.empty:
            raise RuntimeError("Patch mode requires an existing universe.csv. Run --mode full first.")

        prev_over = _read_csv_or_empty(str(_snapshot_path("universe_overrides.prev.csv")))
        prev_exc = _read_csv_or_empty(str(_snapshot_path("asset_excluded.prev.csv")))

        affected = set()
        affected |= _diff_overrides(prev_over, overrides)
        affected |= _diff_exclusions(prev_exc, exclusions)

        affected = set([t for t in affected if t and t.lower() != "nan"])
        print(f"[patch] affected_tickers={len(affected)}")

        if not affected:
            print("[patch] no changes detected; nothing to do.")
            return

        universe = existing.copy()
        universe = ensure_columns(universe)

        # apply exclusions + overrides only for affected
        universe = apply_asset_exclusions_patch(universe, exclusions)
        universe = apply_overrides_patch(universe, overrides, affected=affected)

        # write universe first (so patch-enrich reads the updated rows)
        universe.to_csv(UNIVERSE_CSV, index=False)

        # patch enrich only affected tickers
        enrich_universe_csv_patch(
            universe_csv=UNIVERSE_CSV,
            out_csv=UNIVERSE_CSV,
            tickers_subset=sorted(affected),
            sleep_s=sleep_s,
            ticker_col="ticker",
            yahoo_col="yahoo_ticker",
        )

        # resolver only affected tickers (may change yahoo_ticker)
        changed = resolve_bad_yahoo_symbols(UNIVERSE_CSV, tickers_subset=sorted(affected))

        # if resolver changed any yahoo_ticker, re-enrich those tickers only
        if changed:
            enrich_universe_csv_patch(
                universe_csv=UNIVERSE_CSV,
                out_csv=UNIVERSE_CSV,
                tickers_subset=sorted(set(changed)),
                sleep_s=sleep_s,
                ticker_col="ticker",
                yahoo_col="yahoo_ticker",
            )

        # asset_id assignment (preserve existing, only changes for new rows)
        u2 = pd.read_csv(UNIVERSE_CSV)
        u2 = assign_asset_ids(u2, existing=existing)
        u2.to_csv(UNIVERSE_CSV, index=False)

        print("[patch][ok] universe patched + enriched subset + asset_id preserved.")

    # snapshots for next patch diff
    try:
        Path(OVERRIDES_CSV).exists() and pd.read_csv(OVERRIDES_CSV).to_csv(_snapshot_path("universe_overrides.prev.csv"), index=False)
    except Exception:
        pass

    try:
        Path(ASSET_EXCLUDED_CSV).exists() and pd.read_csv(ASSET_EXCLUDED_CSV).to_csv(_snapshot_path("asset_excluded.prev.csv"), index=False)
    except Exception:
        pass

    print("[ok] snapshots updated.")


if __name__ == "__main__":
    main()
