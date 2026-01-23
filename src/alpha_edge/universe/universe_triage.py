# universe_triage.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


REGION_SUFFIX_HINTS = {
    "spain": [".MC"], "madrid": [".MC"], "bme": [".MC"],
    "germany": [".DE", ".MU", ".F", ".DU", ".HM"], "xetra": [".DE"],
    "france": [".PA"], "paris": [".PA"],
    "italy": [".MI"], "milan": [".MI"],
    "netherlands": [".AS"], "amsterdam": [".AS"],
    "london": [".L"], "uk": [".L"],
    "switzerland": [".SW"], "six": [".SW"],
    "sweden": [".ST"], "stockholm": [".ST"],
    "belgium": [".BR"], "brussels": [".BR"],
    "brazil": [".SA"], "b3": [".SA"], "bovespa": [".SA"], "sao paulo": [".SA"], "são paulo": [".SA"], "bvmf": [".SA"],
    "mexico": [".MX"], "bmv": [".MX"], "bolsa mexicana": [".MX"], "mexican": [".MX"],
    "finland": [".HE"], "helsinki": [".HE"], "nasdaq helsinki": [".HE"],
    "cboe europe": [".DE", ".MC", ".PA", ".MI", ".AS", ".L", ".SW", ".BR", ".ST", ".MU", ".F", ".DU", ".HM"],
}

EU_FALLBACK_SUFFIXES = [".DE",".MC",".PA",".MI",".AS",".L",".SW",".BR",".ST",".SA",".MX",".HE",".MU",".F",".DU",".HM"]
_BAD_QUOTETYPES = {"NONE", "MUTUALFUND", "ECNQUOTE"}
STABLE_QUOTES = {"USD", "USDT", "USDC"}


@dataclass
class Decision:
    classification: str
    suggested_action: str
    suggested_yahoo_ticker: Optional[str] = None
    confidence: str = "LOW"
    why: str = ""


def _is_empty_symbol(s: str) -> bool:
    s = "" if s is None else str(s)
    return (s.strip() == "") or (s.strip().lower() == "nan")


def _normalize_str(s: object) -> str:
    return "" if s is None else str(s).strip()


def _build_exclusion_set(asset_excluded: pd.DataFrame) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    if asset_excluded is None or asset_excluded.empty:
        return out

    t = asset_excluded.copy()
    t["ticker"] = t.get("ticker", "").astype(str).str.strip()
    t["asset_class"] = t.get("asset_class", "").astype(str).str.strip().str.lower()

    for _, r in t.iterrows():
        tick = _normalize_str(r.get("ticker"))
        ac = _normalize_str(r.get("asset_class")).lower()
        if not tick:
            continue
        if (ac == "") or (ac == "nan"):
            out.add((tick, "*"))
        else:
            out.add((tick, ac))
    return out


def _is_excluded(ticker: str, asset_class: str, ex_set: set[tuple[str, str]]) -> bool:
    key = (ticker, (asset_class or "").lower().strip())
    return (key in ex_set) or ((ticker, "*") in ex_set)


def _overrides_lookup(overrides: pd.DataFrame) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if overrides is None or overrides.empty:
        return out

    o = overrides.copy()
    if "ticker" not in o.columns:
        return out

    if "yahoo_ticker" not in o.columns:
        o["yahoo_ticker"] = ""
    if "exclude" not in o.columns:
        o["exclude"] = 0
    if "lock_yahoo_ticker" not in o.columns:
        o["lock_yahoo_ticker"] = 1
    if "exclude_reason" not in o.columns:
        o["exclude_reason"] = ""
    if "note" not in o.columns:
        o["note"] = ""
    if "expected_name" not in o.columns:
        o["expected_name"] = ""

    o["ticker"] = o["ticker"].astype(str).str.strip()
    o["yahoo_ticker"] = o["yahoo_ticker"].astype(str).str.strip()
    o["exclude"] = pd.to_numeric(o["exclude"], errors="coerce").fillna(0).astype(int)
    o["lock_yahoo_ticker"] = pd.to_numeric(o["lock_yahoo_ticker"], errors="coerce").fillna(1).astype(int)

    for _, r in o.iterrows():
        t = _normalize_str(r.get("ticker"))
        if not t:
            continue
        out[t] = {
            "yahoo_ticker": _normalize_str(r.get("yahoo_ticker")),
            "exclude": int(r.get("exclude", 0)),
            "lock_yahoo_ticker": int(r.get("lock_yahoo_ticker", 0)),
            "exclude_reason": r.get("exclude_reason", None),
            "note": r.get("note", None),
            "expected_name": r.get("expected_name", None),
        }
    return out


def _guess_suffix_candidates(raw: str, region: str) -> list[str]:
    raw = _normalize_str(raw).upper()
    reg = _normalize_str(region).lower()
    if not raw:
        return []
    if "." in raw:
        return [raw]

    cands: list[str] = []

    for key, sufs in REGION_SUFFIX_HINTS.items():
        if key in reg:
            cands += [raw + suf for suf in sufs]

    if re.match(r"^[A-Z]{4}\d{1,2}$", raw):
        cands.insert(0, raw + ".SA")

    cands += [raw + suf for suf in EU_FALLBACK_SUFFIXES]

    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _maybe_exclude_crypto_pair(ticker: str, asset_class: str, role: str) -> bool:
    if (asset_class or "").lower() != "crypto" and (role or "").lower() != "crypto":
        return False
    s = _normalize_str(ticker).upper()
    if "-" not in s:
        return False
    base, quote = s.split("-", 1)
    quote = quote.strip()
    if quote in STABLE_QUOTES:
        quote = "USD"
    return quote != "USD"


def triage_failures(
    *,
    fails: pd.DataFrame,
    universe: pd.DataFrame,
    overrides: pd.DataFrame,
    excluded: pd.DataFrame,
    verbose: bool = False,     # NEW
    sample_n: int = 10,        # NEW
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if fails is None or fails.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(columns=["ticker","yahoo_ticker","lock_yahoo_ticker","exclude","exclude_reason","expected_name","note"]),
            pd.DataFrame(columns=["ticker","asset_class","reason"]),
        )

    f = fails.copy()
    f["ticker"] = f.get("ticker", "").astype(str).str.strip()
    f["yahoo_ticker"] = f.get("yahoo_ticker", "").astype(str).str.strip()
    f["reason"] = f.get("reason", "").astype(str).str.strip()
    f["error"] = f.get("error", None)

    # Universe context
    u = universe.copy()
    u["ticker"] = u.get("ticker", "").astype(str).str.strip()

    keep_cols = [c for c in [
        "ticker","name","asset_class","role","region","currency","yahoo_ok","quoteType",
        "exchange","fullExchangeName","market","country","yahoo_ticker"
    ] if c in u.columns]

    u_ctx = u[keep_cols].drop_duplicates(subset=["ticker"], keep="last")
    df = f.merge(u_ctx, on="ticker", how="left", suffixes=("", "_u"))

    # governance
    ex_set = _build_exclusion_set(excluded)
    ov_map = _overrides_lookup(overrides)

    if verbose:
        print(f"[triage_core] fails={len(f)} universe={len(u)} overrides={len(overrides) if overrides is not None else 0} excluded={len(excluded) if excluded is not None else 0}")
        if "reason" in f.columns:
            print("[triage_core] reason_counts:")
            print(f["reason"].value_counts(dropna=False).head(20).to_string())

    triage_rows: list[dict] = []
    sug_over_rows: list[dict] = []
    sug_ex_rows: list[dict] = []

    printed = 0

    for _, r in df.iterrows():
        ticker = _normalize_str(r.get("ticker"))
        yahoo_sym = _normalize_str(r.get("yahoo_ticker"))
        reason = _normalize_str(r.get("reason"))
        err = _normalize_str(r.get("error"))

        asset_class = _normalize_str(r.get("asset_class")).lower()
        role = _normalize_str(r.get("role")).lower()
        region = _normalize_str(r.get("region"))
        name = _normalize_str(r.get("name"))

        currency = _normalize_str(r.get("currency")).upper()
        qt = _normalize_str(r.get("quoteType")).upper()
        yahoo_ok = r.get("yahoo_ok", None)

        # Already excluded?
        if _is_excluded(ticker, asset_class, ex_set):
            d = Decision(
                classification="EXCLUDE",
                suggested_action="no-op (already excluded)",
                confidence="HIGH",
                why="ticker is in asset_excluded.csv",
            )

        # Already overridden (locked yahoo_ticker)?
        elif ticker in ov_map and ov_map[ticker].get("lock_yahoo_ticker", 0) == 1 and not _is_empty_symbol(ov_map[ticker].get("yahoo_ticker")):
            d = Decision(
                classification="OVERRIDE",
                suggested_action=f"no-op (already overridden to {ov_map[ticker]['yahoo_ticker']})",
                suggested_yahoo_ticker=ov_map[ticker]["yahoo_ticker"],
                confidence="HIGH",
                why="ticker has locked yahoo_ticker in universe_overrides.csv",
            )

        else:
            if reason == "no_fx":
                looks_brazil = ("brazil" in region.lower()) or currency == "BRL"
                if looks_brazil:
                    d = Decision(
                        classification="FX_ALIGN_BUG",
                        suggested_action="fix FX date normalization (not an asset override); then re-run ingest",
                        confidence="HIGH",
                        why=f"no_fx concentrated in BRL/Brazil (currency={currency or 'n/a'}, region={region or 'n/a'})",
                    )
                else:
                    d = Decision(
                        classification="INVESTIGATE",
                        suggested_action="check currency / fx symbol availability; consider exclusion if exotic",
                        confidence="MED",
                        why=f"no_fx but not clearly BRL/Brazil (currency={currency or 'n/a'})",
                    )

            elif reason == "empty":
                if _maybe_exclude_crypto_pair(ticker, asset_class, role):
                    d = Decision(
                        classification="EXCLUDE",
                        suggested_action="exclude non-USD crypto quote pair",
                        confidence="HIGH",
                        why=f"crypto pair quote != USD ({ticker})",
                    )
                    sug_ex_rows.append({
                        "ticker": ticker,
                        "asset_class": asset_class if asset_class else "",
                        "reason": "crypto quote != USD (yfinance often unsupported)",
                    })

                elif qt in _BAD_QUOTETYPES:
                    d = Decision(
                        classification="EXCLUDE",
                        suggested_action="exclude (bad quoteType from enrichment)",
                        confidence="HIGH",
                        why=f"quoteType={qt}",
                    )
                    sug_ex_rows.append({
                        "ticker": ticker,
                        "asset_class": asset_class if asset_class else "",
                        "reason": f"bad quoteType={qt}",
                    })

                else:
                    base_sym = yahoo_sym if yahoo_sym else ticker
                    cands = _guess_suffix_candidates(base_sym, region)
                    suggested = cands[0] if cands else None

                    high_conf = False
                    if suggested:
                        reg = region.lower()
                        if "." not in base_sym:
                            if any(k in reg for k in ["brazil","b3","bovespa","sao paulo","são paulo","bvmf"]):
                                high_conf = suggested.endswith(".SA")
                            elif any(k in reg for k in ["mexico","bmv","bolsa mexicana","mexican"]):
                                high_conf = suggested.endswith(".MX")
                            elif any(k in reg for k in ["spain","madrid","bme"]):
                                high_conf = suggested.endswith(".MC")

                    if suggested and suggested != base_sym:
                        d = Decision(
                            classification="OVERRIDABLE",
                            suggested_action=f"add override: yahoo_ticker={suggested} (then re-run ingest)",
                            suggested_yahoo_ticker=suggested,
                            confidence="HIGH" if high_conf else "LOW",
                            why=f"empty OHLCV; region-based suffix guess from region={region or 'n/a'}",
                        )
                        sug_over_rows.append({
                            "ticker": ticker,
                            "yahoo_ticker": suggested,
                            "lock_yahoo_ticker": 1,
                            "exclude": 0,
                            "exclude_reason": "",
                            "expected_name": name,
                            "note": f"auto-suggest from failures(empty); base={base_sym}; region={region}",
                        })
                    else:
                        d = Decision(
                            classification="RETRY",
                            suggested_action="retry later; if repeats, investigate or exclude",
                            confidence="LOW",
                            why=f"empty OHLCV but no safe deterministic override candidate (region={region or 'n/a'})",
                        )

            elif reason == "exception":
                transients = ["timeout", "temporarily", "rate limit", "too many requests", "connection", "503", "502", "504"]
                if any(t in err.lower() for t in transients):
                    d = Decision(
                        classification="RETRY",
                        suggested_action="retry (transient provider/network error)",
                        confidence="MED",
                        why=err[:160],
                    )
                else:
                    d = Decision(
                        classification="INVESTIGATE",
                        suggested_action="inspect error; consider override/exclusion after checking root cause",
                        confidence="LOW",
                        why=err[:160],
                    )
            else:
                d = Decision(
                    classification="INVESTIGATE",
                    suggested_action="inspect reason/error; decide override or exclusion",
                    confidence="LOW",
                    why=f"reason={reason}; error={err[:160]}",
                )

        triage_rows.append({
            "as_of": _normalize_str(r.get("as_of")),
            "ticker": ticker,
            "yahoo_ticker": yahoo_sym,
            "reason": reason,
            "error": err if err else None,
            "asset_class": asset_class or None,
            "role": role or None,
            "region": region or None,
            "currency": currency or None,
            "quoteType": qt or None,
            "yahoo_ok": yahoo_ok,
            "classification": d.classification,
            "suggested_action": d.suggested_action,
            "suggested_yahoo_ticker": d.suggested_yahoo_ticker,
            "confidence": d.confidence,
            "why": d.why,
        })

        # optional per-row logging (limited)
        if verbose and printed < int(sample_n):
            printed += 1
            print(
                f"[triage_row] ticker={ticker} yahoo={yahoo_sym} reason={reason} "
                f"class={d.classification} conf={d.confidence} sugg={d.suggested_yahoo_ticker or ''} why={d.why}"
            )

    triage_report = pd.DataFrame(triage_rows)

    # suggested overrides: remove ones already present/locked to same mapping
    sug_over = pd.DataFrame(sug_over_rows)
    if not sug_over.empty:
        sug_over["ticker"] = sug_over["ticker"].astype(str).str.strip()
        sug_over["yahoo_ticker"] = sug_over["yahoo_ticker"].astype(str).str.strip()

        drop_idx = []
        for i, row in sug_over.iterrows():
            t = row["ticker"]
            y = row["yahoo_ticker"]
            if t in ov_map and ov_map[t].get("lock_yahoo_ticker", 0) == 1 and _normalize_str(ov_map[t].get("yahoo_ticker")) == y:
                drop_idx.append(i)
        if drop_idx:
            sug_over = sug_over.drop(index=drop_idx)

        sug_over = sug_over.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    else:
        sug_over = pd.DataFrame(columns=["ticker","yahoo_ticker","lock_yahoo_ticker","exclude","exclude_reason","expected_name","note"])

    # suggested exclusions: remove already excluded
    sug_ex = pd.DataFrame(sug_ex_rows)
    if not sug_ex.empty:
        sug_ex["ticker"] = sug_ex["ticker"].astype(str).str.strip()
        sug_ex["asset_class"] = sug_ex.get("asset_class", "").astype(str).str.strip().str.lower()
        sug_ex = sug_ex.drop_duplicates(subset=["ticker","asset_class","reason"], keep="first")

        drop_idx = []
        for i, row in sug_ex.iterrows():
            if _is_excluded(row["ticker"], row.get("asset_class",""), ex_set):
                drop_idx.append(i)
        if drop_idx:
            sug_ex = sug_ex.drop(index=drop_idx)

        sug_ex = sug_ex.reset_index(drop=True)
    else:
        sug_ex = pd.DataFrame(columns=["ticker","asset_class","reason"])

    if verbose:
        print(f"[triage_core] DONE triage_rows={len(triage_report)} sug_overrides={len(sug_over)} sug_exclusions={len(sug_ex)}")

    return triage_report, sug_over, sug_ex


def write_triage_outputs_local(
    *,
    out_dir: str,
    triage_report: pd.DataFrame,
    suggested_overrides: pd.DataFrame,
    suggested_exclusions: pd.DataFrame,
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    triage_report.to_csv(str(Path(out_dir) / "triage_report.csv"), index=False)
    suggested_overrides.to_csv(str(Path(out_dir) / "suggested_overrides.csv"), index=False)
    suggested_exclusions.to_csv(str(Path(out_dir) / "suggested_exclusions.csv"), index=False)
