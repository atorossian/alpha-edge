# run_universe_update.py
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from alpha_edge import paths
from alpha_edge.universe.universe import enrich_universe_csv, resolve_ticker_strict


UNIVERSE_CSV = paths.universe_dir() / "universe.csv"
MANUAL_CSV = paths.universe_dir() / "universe_manual_additions.csv"
OVERRIDES_CSV = paths.universe_dir() / "universe_overrides.csv"
ASSET_EXCLUDED_CSV = paths.universe_dir() / "asset_excluded.csv"

QF_PAGES = {
    "etf": "https://help.quantfury.com/en/articles/5448760-etfs",
    "crypto": "https://help.quantfury.com/en/articles/5448756-cryptocurrency-pairs",
    "stock": "https://help.quantfury.com/en/articles/5448752-stocks",
}

STABLE_QUOTES = {"USD", "USDT", "USDC"}

SAFE_SCRAPE_UPDATE_COLS = ["name", "asset_class", "role", "region", "include", "max_weight", "min_weight"]

SNAP_DIR = paths.universe_dir() / "snapshots"
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


def _clean_str(x: object) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    if s.lower() == "nan":
        return ""
    return s


def _stable_hash(parts: list[str], *, prefix: str, n: int = 16) -> str:
    key = "|".join([_clean_str(p).upper() for p in parts if _clean_str(p)])
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}{h}"


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
        "row_id": "",
        "asset_id": "",
        "ticker": "",
        "broker_ticker": "",
        "yahoo_ticker": "",
        "name": "",
        "asset_class": "",
        "role": "",
        "region": "",
        "max_weight": 0.25,
        "min_weight": 0.0,
        "include": 1,
        "lock_yahoo_ticker": 0,
    }
    for c, default in base_cols.items():
        if c not in df.columns:
            df[c] = default

    # fill NaNs before string ops
    for c in ["row_id", "asset_id", "ticker", "broker_ticker", "yahoo_ticker", "name", "asset_class", "role", "region"]:
        df[c] = df[c].fillna("")

    df["include"] = pd.to_numeric(df["include"], errors="coerce").fillna(1).astype(int)
    df["lock_yahoo_ticker"] = pd.to_numeric(df["lock_yahoo_ticker"], errors="coerce").fillna(0).astype(int)

    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["ticker"] = df["ticker"].where(~df["ticker"].str.lower().isin(["nan", "none", ""]), "")
    df["ticker"] = df["ticker"].str.upper()

    df["broker_ticker"] = df["broker_ticker"].astype(str).fillna("").str.strip()
    df["broker_ticker"] = df["broker_ticker"].where(~df["broker_ticker"].str.lower().isin(["nan", "none"]), "")

    df["yahoo_ticker"] = df["yahoo_ticker"].astype(str).fillna("").str.strip()
    df["yahoo_ticker"] = df["yahoo_ticker"].where(~df["yahoo_ticker"].str.lower().isin(["nan", "none"]), "")

    df["asset_class"] = df["asset_class"].astype(str).fillna("").str.strip().str.lower()
    df["role"] = df["role"].astype(str).fillna("").str.strip().str.lower()
    df["region"] = df["region"].astype(str).fillna("").str.strip()
    df["name"] = df["name"].astype(str).fillna("").str.strip()

    # defaults
    m = (df["yahoo_ticker"] == "") & (df["ticker"] != "")
    df.loc[m, "yahoo_ticker"] = df.loc[m, "ticker"]

    m = (df["broker_ticker"] == "") & (df["ticker"] != "")
    df.loc[m, "broker_ticker"] = df.loc[m, "ticker"]

    return df


def compute_row_id(df: pd.DataFrame) -> pd.Series:
    tmp = ensure_columns(df.copy())
    key = (
        tmp["ticker"] + "|" +
        tmp["name"] + "|" +
        tmp["region"] + "|" +
        tmp["role"] + "|" +
        tmp["asset_class"]
    )
    return key.apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:16])


def drop_scr_columns(df: pd.DataFrame) -> pd.DataFrame:
    # drop any columns created by repeated merges: *_scr, *_scr.1, *_scr.2, ...
    cols = [c for c in df.columns if re.search(r"_scr(\.\d+)?$", c)]
    return df.drop(columns=cols, errors="ignore")


def load_existing_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df is None or df.empty:
        return pd.DataFrame()
    df = drop_scr_columns(df)  # CRITICAL: never carry _scr into next runs
    df = ensure_columns(df)
    if "row_id" not in df.columns or df["row_id"].astype(str).str.strip().eq("").all():
        df["row_id"] = compute_row_id(df)
    return df


def _is_empty_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str)
    return s.isna() | s2.str.strip().eq("") | s2.str.lower().eq("nan")


def backfill_from_scr(universe: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    If merge produced col_scr, fill universe[col] where empty from universe[col_scr],
    then drop col_scr.
    """
    sc = f"{col}_scr"
    if sc not in universe.columns:
        return universe
    if col not in universe.columns:
        universe[col] = ""
    empty = _is_empty_series(universe[col])
    universe.loc[empty, col] = universe.loc[empty, sc]
    universe = universe.drop(columns=[sc], errors="ignore")
    return universe


# ----------------------------
# Builders
# ----------------------------
def build_etfs() -> pd.DataFrame:
    df = parse_first_table(fetch_html(QF_PAGES["etf"]))
    raw = df["Ticker"].astype(str).str.upper().str.strip()
    out = pd.DataFrame({
        "broker_ticker": raw,
        "ticker": raw,
        "yahoo_ticker": raw,
        "name": df["ETF"].astype(str),
        "asset_class": "equity",
        "role": "etf",
        "region": df["Exchange"].astype(str),
        "max_weight": 0.25,
        "min_weight": 0.0,
        "include": 1,
    })
    out = ensure_columns(out)
    out["row_id"] = compute_row_id(out)
    return out


def build_stocks() -> pd.DataFrame:
    df = parse_first_table(fetch_html(QF_PAGES["stock"]))
    raw = df["Ticker"].astype(str).str.upper().str.strip()
    out = pd.DataFrame({
        "broker_ticker": raw,
        "ticker": raw,
        "yahoo_ticker": raw,
        "name": df["Company"].astype(str),
        "asset_class": "equity",
        "role": "stock",
        "region": df["Exchange"].astype(str),
        "max_weight": 0.25,
        "min_weight": 0.0,
        "include": 1,
    })
    out = ensure_columns(out)
    out["row_id"] = compute_row_id(out)
    return out


def build_crypto_pairs() -> pd.DataFrame:
    df = parse_first_table(fetch_html(QF_PAGES["crypto"]))
    pair_col = "Pair" if "Pair" in df.columns else df.columns[0]
    internal = df[pair_col].astype(str).apply(canonical_crypto_pair)
    out = pd.DataFrame({
        "broker_ticker": internal,
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
    out = ensure_columns(out)
    out["row_id"] = compute_row_id(out)
    return out


def load_manual_additions(manual_path: Path) -> pd.DataFrame:
    manual = _read_csv_or_empty(str(manual_path))
    if manual.empty:
        return manual

    manual.columns = [c.strip() for c in manual.columns]

    # Your manual header uses expected_name; pipeline uses name
    if "expected_name" in manual.columns and "name" not in manual.columns:
        manual = manual.rename(columns={"expected_name": "name"})

    manual = ensure_columns(manual)

    # Manual rows typically don't carry broker_ticker; enforce it
    bt_empty = manual["broker_ticker"].astype(str).str.strip().eq("")
    manual.loc[bt_empty, "broker_ticker"] = manual.loc[bt_empty, "ticker"]

    # include default if not provided
    manual["include"] = pd.to_numeric(manual.get("include", 1), errors="coerce").fillna(1).astype(int)

    # must have ticker
    bad = manual[manual["ticker"].astype(str).str.strip().eq("")]
    if not bad.empty:
        raise RuntimeError(f"Manual additions have empty ticker. Fix: {manual_path}. Columns={list(manual.columns)}")

    # compute row_id
    rid_empty = manual["row_id"].astype(str).str.strip().eq("")
    if rid_empty.any():
        manual.loc[rid_empty, "row_id"] = compute_row_id(manual.loc[rid_empty].copy())

    return manual


# ----------------------------
# Overrides / Exclusions (your current format)
# ----------------------------
def load_overrides(path: Path) -> pd.DataFrame:
    base_cols = ["target_row_id", "ticker", "yahoo_ticker", "lock_yahoo_ticker", "exclude", "exclude_reason", "expected_name", "note"]
    df = _read_csv_or_empty(str(path), cols=base_cols)
    for c in base_cols:
        if c not in df.columns:
            df[c] = "" if c in {"target_row_id", "ticker", "yahoo_ticker", "exclude_reason", "expected_name", "note"} else 0

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["target_row_id"] = df["target_row_id"].astype(str).fillna("").str.strip().str.upper()
    df["yahoo_ticker"] = df["yahoo_ticker"].astype(str).fillna("").str.strip()
    df["lock_yahoo_ticker"] = pd.to_numeric(df["lock_yahoo_ticker"], errors="coerce").fillna(1).astype(int)
    df["exclude"] = pd.to_numeric(df["exclude"], errors="coerce").fillna(0).astype(int)
    df["expected_name"] = df.get("expected_name", "").astype(str).fillna("").str.strip()
    return df[base_cols]


def load_exclusions(path: Path) -> pd.DataFrame:
    df = _read_csv_or_empty(str(path), cols=["target_row_id", "ticker", "asset_class", "reason"])
    if df.empty:
        return df
    if "target_row_id" not in df.columns:
        df["target_row_id"] = ""
    df["target_row_id"] = df["target_row_id"].astype(str).fillna("").str.strip().str.upper()
    df["ticker"] = df["ticker"].astype(str).fillna("").str.strip().str.upper()
    df["asset_class"] = df.get("asset_class", "").astype(str).fillna("").str.strip().str.lower()
    df["reason"] = df.get("reason", "").astype(str).fillna("")
    return df[["target_row_id", "ticker", "asset_class", "reason"]]


def apply_asset_exclusions_patch(universe: pd.DataFrame, excluded_df: pd.DataFrame) -> pd.DataFrame:
    u = universe.copy()
    u = ensure_columns(u)

    u["ticker"] = u["ticker"].astype(str).str.strip().str.upper()
    u["asset_class"] = u["asset_class"].astype(str).fillna("").str.strip().str.lower()
    u["row_id"] = u["row_id"].astype(str).fillna("").str.strip().str.upper()
    u["include"] = pd.to_numeric(u["include"], errors="coerce").fillna(1).astype(int)

    if excluded_df is None or excluded_df.empty:
        return u.reset_index(drop=True)

    ex = excluded_df.copy()
    ex["target_row_id"] = ex.get("target_row_id", "").astype(str).fillna("").str.strip().str.upper()
    ex["ticker"] = ex.get("ticker", "").astype(str).fillna("").str.strip().str.upper()
    ex["asset_class"] = ex.get("asset_class", "").astype(str).fillna("").str.strip().str.lower()

    # 1) row_id-targeted exclusions
    ex_rid = ex[ex["target_row_id"] != ""]
    if not ex_rid.empty:
        u.loc[u["row_id"].isin(set(ex_rid["target_row_id"])), "include"] = 0

    # 2) legacy ticker exclusions
    ex_legacy = ex[ex["target_row_id"] == ""]
    if ex_legacy.empty:
        return u.reset_index(drop=True)

    ex_any = ex_legacy[ex_legacy["asset_class"].isin(["", "nan"])]
    ex_scoped = ex_legacy[~ex_legacy["asset_class"].isin(["", "nan"])]

    if not ex_any.empty:
        u.loc[u["ticker"].isin(set(ex_any["ticker"])), "include"] = 0

    if not ex_scoped.empty:
        key_u = u["ticker"].astype(str) + "||" + u["asset_class"].astype(str)
        key_ex = ex_scoped["ticker"].astype(str) + "||" + ex_scoped["asset_class"].astype(str)
        u.loc[key_u.isin(set(key_ex)), "include"] = 0

    return u.reset_index(drop=True)


def apply_overrides_patch(universe: pd.DataFrame, overrides: pd.DataFrame, *, affected_tickers: set[str]) -> pd.DataFrame:
    if overrides is None or overrides.empty or not affected_tickers:
        return universe.reset_index(drop=True)

    u = ensure_columns(universe.copy())
    u["ticker"] = u["ticker"].astype(str).str.strip().str.upper()
    u["row_id"] = u["row_id"].astype(str).fillna("").str.strip().str.upper()
    u["yahoo_ticker"] = u["yahoo_ticker"].astype(str).fillna("").str.strip()
    u["include"] = pd.to_numeric(u["include"], errors="coerce").fillna(1).astype(int)
    u["lock_yahoo_ticker"] = pd.to_numeric(u["lock_yahoo_ticker"], errors="coerce").fillna(0).astype(int)

    o = overrides.copy()
    o["ticker"] = o.get("ticker", "").astype(str).str.strip().str.upper()
    o["target_row_id"] = o.get("target_row_id", "").astype(str).fillna("").str.strip().str.upper()

    o = o[o["ticker"].isin(affected_tickers)].copy()
    if o.empty:
        return u.reset_index(drop=True)

    # last wins per (ticker, target_row_id)
    o = o.drop_duplicates(subset=["ticker", "target_row_id"], keep="last")

    for _, r in o.iterrows():
        t = str(r.get("ticker", "")).strip().upper()
        rid = str(r.get("target_row_id", "")).strip().upper()
        if not t or t.lower() == "nan":
            continue

        sel = (u["row_id"] == rid) if rid else (u["ticker"] == t)
        if not bool(sel.any()):
            continue

        ex = int(pd.to_numeric(r.get("exclude", 0), errors="coerce") or 0)
        if ex == 1:
            u.loc[sel, "include"] = 0

        lock = int(pd.to_numeric(r.get("lock_yahoo_ticker", 1), errors="coerce") or 0)
        y = str(r.get("yahoo_ticker", "")).strip()
        if lock == 1 and y and y.lower() != "nan":
            u.loc[sel, "yahoo_ticker"] = y

        u.loc[sel, "lock_yahoo_ticker"] = lock

    return u.reset_index(drop=True)


# ----------------------------
# Resolver (ROW_ID SAFE)
# ----------------------------
def resolve_bad_yahoo_symbols(universe_csv: Path, *, row_ids_subset: list[str] | None = None) -> list[str]:
    full = pd.read_csv(universe_csv)
    if full.empty:
        return []

    full = ensure_columns(full)
    if "row_id" not in full.columns or full["row_id"].astype(str).str.strip().eq("").all():
        full["row_id"] = compute_row_id(full)

    if "symbol" not in full.columns:
        full["symbol"] = full["yahoo_ticker"]
    if "resolver_debug" not in full.columns:
        full["resolver_debug"] = None

    if row_ids_subset:
        subset = {str(x).strip() for x in row_ids_subset if str(x).strip()}
        mask = full["row_id"].astype(str).isin(subset)
    else:
        mask = pd.Series([True] * len(full), index=full.index)

    work = full.loc[mask].copy()
    if work.empty:
        return []

    locked = work.get("lock_yahoo_ticker", 0).fillna(0).astype(int) == 1
    bad = (work.get("yahoo_ok").fillna(False) == False) & (~locked)

    qt = work.get("quoteType").fillna("").astype(str).str.upper()
    bad |= qt.isin(["NONE", "MUTUALFUND", "ECNQUOTE"])

    changed_row_ids: list[str] = []

    for i, row in work[bad].iterrows():
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
            work.at[i, "yahoo_ticker"] = resolved
            work.at[i, "symbol"] = resolved
            changed_row_ids.append(str(row.get("row_id")))

        work.at[i, "resolver_debug"] = json.dumps(dbg, ensure_ascii=False)

    cols = ["yahoo_ticker", "symbol", "resolver_debug"]
    full.loc[work.index, cols] = work[cols]
    full.to_csv(universe_csv, index=False)

    return sorted(set([x for x in changed_row_ids if x]))


# ----------------------------
# Asset ID assignment
# ----------------------------
def assign_asset_ids(universe: pd.DataFrame, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    u = ensure_columns(universe.copy())

    if "row_id" not in u.columns or u["row_id"].astype(str).str.strip().eq("").all():
        u["row_id"] = compute_row_id(u)

    # preserve asset_id by row_id
    if existing is not None and not existing.empty and "asset_id" in existing.columns:
        ex = ensure_columns(existing.copy())
        if "row_id" not in ex.columns or ex["row_id"].astype(str).str.strip().eq("").all():
            ex["row_id"] = compute_row_id(ex)

        ex["asset_id"] = ex.get("asset_id", "").astype(str).fillna("").str.strip()
        ex_map = ex.set_index("row_id")["asset_id"].to_dict()

        cur = u["asset_id"].astype(str).fillna("").str.strip()
        empty = (cur == "") | (cur.str.lower() == "nan")
        u.loc[empty, "asset_id"] = u.loc[empty, "row_id"].map(ex_map).fillna("")

    # crypto deterministic
    cur = u["asset_id"].astype(str).fillna("").str.strip()
    empty = (cur == "") | (cur.str.lower() == "nan")
    is_crypto = u["asset_class"].astype(str).str.lower() == "crypto"
    u.loc[empty & is_crypto, "asset_id"] = "CRYPTO:" + u.loc[empty & is_crypto, "ticker"].astype(str).str.upper()

    # equity fallback hash
    cur = u["asset_id"].astype(str).fillna("").str.strip()
    empty = (cur == "") | (cur.str.lower() == "nan")
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


def dedup_by_asset_id(universe: pd.DataFrame, *, out_path: Path) -> pd.DataFrame:
    u = universe.copy()
    if "asset_id" not in u.columns:
        return u.reset_index(drop=True)

    u["asset_id"] = u["asset_id"].astype(str).fillna("").str.strip()
    u = u[u["asset_id"] != ""].copy()

    dup = u[u["asset_id"].duplicated(keep=False)].copy()
    if not dup.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dup.to_csv(out_path, index=False)

    u["include"] = pd.to_numeric(u.get("include", 1), errors="coerce").fillna(1).astype(int)
    yok = u.get("yahoo_ok")
    u["_yok"] = 0 if yok is None else yok.fillna(False).astype(bool).astype(int)
    u["_keep_score"] = (u["include"] * 10) + u["_yok"]

    u = u.sort_values(["asset_id", "_keep_score"], ascending=[True, False])
    out = u.drop_duplicates(subset=["asset_id"], keep="first").drop(columns=["_yok", "_keep_score"], errors="ignore")
    return out.reset_index(drop=True)


def _snapshot_path(name: str) -> Path:
    return SNAP_DIR / name


def _diff_overrides(prev: pd.DataFrame, curr: pd.DataFrame) -> set[str]:
    cols = ["target_row_id", "ticker", "yahoo_ticker", "lock_yahoo_ticker", "exclude"]
    for c in cols:
        if c not in prev.columns:
            prev[c] = ""
        if c not in curr.columns:
            curr[c] = ""

    p = prev.copy()
    c = curr.copy()
    p["ticker"] = p["ticker"].astype(str).str.strip().str.upper()
    c["ticker"] = c["ticker"].astype(str).str.strip().str.upper()
    p["target_row_id"] = p["target_row_id"].astype(str).fillna("").str.strip().str.upper()
    c["target_row_id"] = c["target_row_id"].astype(str).fillna("").str.strip().str.upper()

    p["lock_yahoo_ticker"] = pd.to_numeric(p["lock_yahoo_ticker"], errors="coerce").fillna(1).astype(int)
    c["lock_yahoo_ticker"] = pd.to_numeric(c["lock_yahoo_ticker"], errors="coerce").fillna(1).astype(int)
    p["exclude"] = pd.to_numeric(p["exclude"], errors="coerce").fillna(0).astype(int)
    c["exclude"] = pd.to_numeric(c["exclude"], errors="coerce").fillna(0).astype(int)

    p = p[cols].fillna("")
    c = c[cols].fillna("")

    def to_map(df: pd.DataFrame) -> dict[tuple[str, str], dict]:
        out = {}
        for _, r in df.iterrows():
            t = str(r["ticker"]).strip().upper()
            rid = str(r["target_row_id"]).strip().upper()
            if not t or t.lower() == "nan":
                continue
            out[(t, rid)] = {
                "yahoo_ticker": str(r.get("yahoo_ticker", "")).strip(),
                "lock_yahoo_ticker": int(r.get("lock_yahoo_ticker", 1) or 1),
                "exclude": int(r.get("exclude", 0) or 0),
            }
        return out

    p_map = to_map(p)
    c_map = to_map(c)
    all_keys = set(p_map.keys()) | set(c_map.keys())

    affected = set()
    for k in all_keys:
        if k not in p_map or k not in c_map or p_map[k] != c_map[k]:
            affected.add(k[0])

    return affected


def _diff_exclusions(prev: pd.DataFrame, curr: pd.DataFrame) -> set[str]:
    cols = ["target_row_id", "ticker", "asset_class"]
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
    ap.add_argument("--assert_paths", action="store_true")
    args = ap.parse_args()

    if args.assert_paths:
        print(f"[paths] project_root={paths.project_root()}")
        print(f"[paths] universe_dir={paths.universe_dir()}")
        print(f"[paths] UNIVERSE_CSV={UNIVERSE_CSV}")
        print(f"[paths] MANUAL_CSV={MANUAL_CSV} exists={MANUAL_CSV.exists()}")

    sleep_s = float(args.sleep_s)
    mode = args.mode

    existing = load_existing_universe(UNIVERSE_CSV)

    overrides = load_overrides(OVERRIDES_CSV)
    exclusions = load_exclusions(ASSET_EXCLUDED_CSV)

    if mode == "full":
        etfs = build_etfs()
        stocks = build_stocks()
        crypto = build_crypto_pairs()
        scraped = pd.concat([etfs, stocks, crypto], ignore_index=True)
        scraped = ensure_columns(scraped)

        manual = load_manual_additions(MANUAL_CSV) if MANUAL_CSV.exists() else pd.DataFrame()
        if not manual.empty:
            scraped = pd.concat([scraped, manual], ignore_index=True)
            scraped = ensure_columns(scraped)
            print(f"[manual] appended rows={len(manual)} from {MANUAL_CSV}")
        else:
            print(f"[manual] no manual additions or empty at {MANUAL_CSV}")

        # guarantee row_id in scraped
        scraped["row_id"] = scraped.get("row_id", "").astype(str)
        rid_empty = scraped["row_id"].astype(str).str.strip().eq("")
        if rid_empty.any():
            scraped.loc[rid_empty, "row_id"] = compute_row_id(scraped.loc[rid_empty].copy())

        is_bootstrap = existing.empty

        if is_bootstrap:
            universe = scraped.copy()
        else:
            # IMPORTANT: keep _scr columns until AFTER backfill
            universe = existing.merge(
                scraped[["row_id", "ticker", "broker_ticker"] + SAFE_SCRAPE_UPDATE_COLS + ["yahoo_ticker", "lock_yahoo_ticker"]],
                on="row_id",
                how="outer",
                suffixes=("", "_scr"),
            )

            # backfill base columns from _scr where base is empty (this preserves new rows!)
            for col in ["ticker", "broker_ticker", "yahoo_ticker", "lock_yahoo_ticker"] + SAFE_SCRAPE_UPDATE_COLS:
                universe = backfill_from_scr(universe, col)

            # NOW safe to drop any remaining _scr columns
            universe = drop_scr_columns(universe)

        universe = ensure_columns(universe)

        # drop invalid key rows (ticker empty) AFTER backfill
        universe = universe[universe["ticker"].astype(str).str.strip().ne("")].copy()

        # apply exclusions + overrides
        universe = apply_asset_exclusions_patch(universe, exclusions)
        affected_tickers = set(universe["ticker"].astype(str).str.strip().str.upper().tolist())
        universe = apply_overrides_patch(universe, overrides, affected_tickers=affected_tickers)

        universe.to_csv(UNIVERSE_CSV, index=False)
        print(f"[ok] wrote universe rows={len(universe)} -> {UNIVERSE_CSV}")

        # Enrich only included rows by row_id
        u_live = universe.copy()
        u_live["include"] = pd.to_numeric(u_live.get("include", 1), errors="coerce").fillna(1).astype(int)
        row_ids_live = sorted(u_live.loc[u_live["include"] == 1, "row_id"].dropna().astype(str).unique().tolist())

        enrich_universe_csv(
            universe_csv=str(UNIVERSE_CSV),
            out_csv=str(UNIVERSE_CSV),
            sleep_s=sleep_s,
            ticker_col="row_id",
            yahoo_col="yahoo_ticker",
            tickers_subset=row_ids_live,
        )

        resolve_bad_yahoo_symbols(UNIVERSE_CSV, row_ids_subset=row_ids_live)

        enrich_universe_csv(
            universe_csv=str(UNIVERSE_CSV),
            out_csv=str(UNIVERSE_CSV),
            sleep_s=sleep_s,
            ticker_col="row_id",
            yahoo_col="yahoo_ticker",
            tickers_subset=row_ids_live,
        )

        u2 = pd.read_csv(UNIVERSE_CSV)
        u2 = drop_scr_columns(u2)
        u2 = assign_asset_ids(u2, existing=existing)

        dedup_report = paths.local_outputs_dir() / "universe_audit" / "dedup_asset_id_removed.csv"
        u2 = dedup_by_asset_id(u2, out_path=dedup_report)

        u2.to_csv(UNIVERSE_CSV, index=False)
        print("[ok] asset_id assigned + dedup_by_asset_id done.")

    else:
        # patch mode
        if existing is None or existing.empty:
            raise RuntimeError("Patch mode requires an existing universe.csv. Run --mode full first.")

        prev_over = _read_csv_or_empty(str(_snapshot_path("universe_overrides.prev.csv")))
        prev_exc = _read_csv_or_empty(str(_snapshot_path("asset_excluded.prev.csv")))

        affected_tickers = set()
        affected_tickers |= _diff_overrides(prev_over, overrides)
        affected_tickers |= _diff_exclusions(prev_exc, exclusions)
        affected_tickers = {t for t in affected_tickers if t and t.lower() != "nan"}

        print(f"[patch] affected_tickers={len(affected_tickers)}")
        if not affected_tickers:
            print("[patch] no changes detected; nothing to do.")
            return

        universe = pd.read_csv(UNIVERSE_CSV)
        universe = drop_scr_columns(universe)
        universe = ensure_columns(universe)
        if "row_id" not in universe.columns or universe["row_id"].astype(str).str.strip().eq("").all():
            universe["row_id"] = compute_row_id(universe)

        universe = apply_asset_exclusions_patch(universe, exclusions)
        universe = apply_overrides_patch(universe, overrides, affected_tickers=affected_tickers)
        universe.to_csv(UNIVERSE_CSV, index=False)

        mask = universe["ticker"].astype(str).str.upper().isin(affected_tickers)
        row_ids_affected = sorted(universe.loc[mask, "row_id"].astype(str).unique().tolist())

        enrich_universe_csv(
            universe_csv=str(UNIVERSE_CSV),
            out_csv=str(UNIVERSE_CSV),
            sleep_s=sleep_s,
            ticker_col="row_id",
            yahoo_col="yahoo_ticker",
            tickers_subset=row_ids_affected,
        )

        changed = resolve_bad_yahoo_symbols(UNIVERSE_CSV, row_ids_subset=row_ids_affected)

        if changed:
            enrich_universe_csv(
                universe_csv=str(UNIVERSE_CSV),
                out_csv=str(UNIVERSE_CSV),
                sleep_s=sleep_s,
                ticker_col="row_id",
                yahoo_col="yahoo_ticker",
                tickers_subset=changed,
            )

        u2 = pd.read_csv(UNIVERSE_CSV)
        u2 = assign_asset_ids(u2, existing=existing)
        u2.to_csv(UNIVERSE_CSV, index=False)

        print("[patch][ok] universe patched + enriched by row_id.")

    # snapshots for patch diff
    try:
        if OVERRIDES_CSV.exists():
            pd.read_csv(OVERRIDES_CSV).to_csv(_snapshot_path("universe_overrides.prev.csv"), index=False)
    except Exception:
        pass

    try:
        if ASSET_EXCLUDED_CSV.exists():
            pd.read_csv(ASSET_EXCLUDED_CSV).to_csv(_snapshot_path("asset_excluded.prev.csv"), index=False)
    except Exception:
        pass

    print("[ok] snapshots updated.")


if __name__ == "__main__":
    main()
