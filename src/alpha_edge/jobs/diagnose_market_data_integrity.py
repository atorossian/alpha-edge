from __future__ import annotations

import argparse
import io
import json
from typing import Any

import boto3
import numpy as np
import pandas as pd

from alpha_edge import paths
from alpha_edge.core.data_loader import (
    s3_init,
    s3_write_json_event,
    s3_write_parquet_partition,
)
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.market.build_returns_wide_cache import RETURN_COLUMNS
from concurrent.futures import ThreadPoolExecutor, as_completed

DIAG_TABLE = "diagnostics/data_quality/v1"


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _safe_date_str(x: Any) -> str | None:
    try:
        ts = pd.Timestamp(x)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return str(ts.date())


def _clean_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def _norm_asset_id(x: Any) -> str:
    return _clean_str(x)


def _norm_ticker(x: Any) -> str:
    return _clean_str(x).upper()


def _make_market_store(*, bucket: str, region: str, market_root: str) -> MarketStore:
    try:
        return MarketStore(
            bucket=bucket,
            region=region,
            base_prefix=market_root,
        )
    except TypeError:
        if market_root != "market":
            raise RuntimeError(
                "MarketStore does not support base_prefix, while "
                f"market_root={market_root!r}. Refusing to fall back to default root."
            )
        return MarketStore(bucket=bucket, region=region)


def _list_s3_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) or []:
            key = obj.get("Key")
            if key:
                keys.append(key)

        if not resp.get("IsTruncated"):
            break

        token = resp.get("NextContinuationToken")

    return keys


def _read_parquet_s3(s3, *, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    raw = obj["Body"].read()
    return pd.read_parquet(io.BytesIO(raw), engine="pyarrow")


# ---------------------------------------------------------------------
# Universe integrity
# ---------------------------------------------------------------------

def _load_universe(*, universe_csv: str | None = None) -> pd.DataFrame:
    path = universe_csv or str(paths.universe_dir() / "universe.csv")
    df = pd.read_csv(path)

    required = ["asset_id", "ticker"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Universe missing required column: {col}")

    df = df.copy()
    df["asset_id"] = df["asset_id"].map(_norm_asset_id)
    df["ticker"] = df["ticker"].map(_norm_ticker)

    if "yahoo_ticker" not in df.columns:
        df["yahoo_ticker"] = df["ticker"]
    df["yahoo_ticker"] = df["yahoo_ticker"].map(_clean_str)

    if "broker_ticker" not in df.columns:
        df["broker_ticker"] = df["ticker"]
    df["broker_ticker"] = df["broker_ticker"].map(_clean_str)

    if "name" not in df.columns:
        df["name"] = df["ticker"]
    df["name"] = df["name"].map(_clean_str)

    if "asset_class" not in df.columns:
        df["asset_class"] = "unknown"
    df["asset_class"] = df["asset_class"].map(lambda x: _clean_str(x).lower() or "unknown")

    if "role" not in df.columns:
        df["role"] = "unknown"
    df["role"] = df["role"].map(lambda x: _clean_str(x).lower() or "unknown")

    if "region" not in df.columns:
        df["region"] = "unknown"
    df["region"] = df["region"].map(lambda x: _clean_str(x) or "unknown")

    if "include" in df.columns:
        df["include"] = pd.to_numeric(df["include"], errors="coerce").fillna(1).astype(int)
    else:
        df["include"] = 1

    return df


def _universe_duplicate_diagnostics(active: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}

    dup_asset = active[active["asset_id"].duplicated(keep=False)].sort_values("asset_id")
    dup_ticker = active[active["ticker"].duplicated(keep=False)].sort_values("ticker")
    dup_yahoo = active[active["yahoo_ticker"].duplicated(keep=False)].sort_values("yahoo_ticker")

    cols = [
        "asset_id",
        "ticker",
        "yahoo_ticker",
        "broker_ticker",
        "name",
        "asset_class",
        "role",
        "region",
        "include",
    ]
    cols = [c for c in cols if c in active.columns]

    out["duplicate_asset_id_count"] = int(dup_asset["asset_id"].nunique()) if not dup_asset.empty else 0
    out["duplicate_ticker_count"] = int(dup_ticker["ticker"].nunique()) if not dup_ticker.empty else 0
    out["duplicate_yahoo_ticker_count"] = int(dup_yahoo["yahoo_ticker"].nunique()) if not dup_yahoo.empty else 0

    out["duplicate_asset_id_rows"] = dup_asset[cols].head(100).to_dict("records") if not dup_asset.empty else []
    out["duplicate_ticker_rows"] = dup_ticker[cols].head(200).to_dict("records") if not dup_ticker.empty else []
    out["duplicate_yahoo_ticker_rows"] = dup_yahoo[cols].head(200).to_dict("records") if not dup_yahoo.empty else []

    out["identity_status"] = (
        "FAIL"
        if out["duplicate_asset_id_count"] > 0
        else ("WARN" if out["duplicate_ticker_count"] > 0 or out["duplicate_yahoo_ticker_count"] > 0 else "PASS")
    )

    return out


# ---------------------------------------------------------------------
# Data loading by asset_id
# ---------------------------------------------------------------------

def _load_ohlcv_for_asset_id(
    *,
    s3,
    bucket: str,
    market_root: str,
    asset_id: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    years = range(int(start_ts.year), int(end_ts.year) + 1)

    frames: list[pd.DataFrame] = []
    for y in years:
        prefix = f"{market_root.strip('/')}/ohlcv_usd/v1/asset_id={asset_id}/year={y}/"
        keys = [k for k in _list_s3_keys(s3, bucket=bucket, prefix=prefix) if k.endswith(".parquet")]

        for key in keys:
            df = _read_parquet_s3(s3, bucket=bucket, key=key)
            if df is None or df.empty:
                continue

            cols_lower = {str(c).lower(): c for c in df.columns}
            date_col = cols_lower.get("date")
            if date_col is None:
                continue

            px_col = (
                cols_lower.get("adj_close_usd")
                or cols_lower.get("close_usd")
                or cols_lower.get("adj_close")
                or cols_lower.get("close")
            )
            if px_col is None:
                continue

            out = df.copy()
            out["__s3_key"] = key
            out["__price_col_used"] = px_col
            out["asset_id"] = asset_id
            out["date"] = pd.to_datetime(out[date_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
            out["price"] = pd.to_numeric(out[px_col], errors="coerce")

            keep_cols = ["date", "asset_id", "price", "__price_col_used", "__s3_key"]
            for optional in ["adj_close_usd", "close_usd", "adj_close", "close", "volume"]:
                if optional in out.columns:
                    keep_cols.append(optional)

            frames.append(out[keep_cols])

    if not frames:
        return pd.DataFrame(columns=["date", "asset_id", "price"])

    long = pd.concat(frames, ignore_index=True)
    long = long.dropna(subset=["date"])
    long = long[(long["date"] >= start_ts) & (long["date"] <= end_ts)].copy()
    long["price"] = pd.to_numeric(long["price"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    long = long.sort_values(["date", "__s3_key"], kind="stable")
    long = long.drop_duplicates(subset=["date"], keep="last")
    long = long.sort_values("date")

    return long


def _load_returns_usd_for_asset_id(
    *,
    store: MarketStore,
    asset_id: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    try:
        df = store.read_returns_usd(
            asset_ids=[asset_id],
            start=start,
            end=end,
            columns=RETURN_COLUMNS,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    else:
        return pd.DataFrame()

    out["asset_id"] = asset_id
    return out.dropna(subset=["date"]).sort_values("date")


def _load_returns_wide_cache(
    *,
    bucket: str,
    market_root: str,
    min_years: float,
) -> pd.DataFrame:
    path = f"s3://{bucket}/{market_root.strip('/')}/cache/v1/returns_wide_min{int(float(min_years))}y.parquet"
    try:
        df = pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
        out = out.dropna(subset=["date"]).set_index("date")
    else:
        idx = pd.to_datetime(out.index, errors="coerce", utc=True)
        mask = ~idx.isna()
        out = out.loc[mask].copy()
        out.index = idx[mask].tz_convert(None).normalize()

    out.index.name = "date"
    out.columns = [str(c).strip() for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="last")]
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _max_drawdown_from_simple_returns(r: pd.Series) -> float | None:
    r = pd.to_numeric(r, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return None
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return _safe_float(dd.min())


def _largest_abs_rows(s: pd.Series, *, n: int = 10) -> list[dict]:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return []

    ranked = s.abs().sort_values(ascending=False).head(n)
    return [
        {
            "date": str(pd.Timestamp(idx).date()),
            "value": _safe_float(s.loc[idx]),
            "abs_value": _safe_float(value),
        }
        for idx, value in ranked.items()
    ]


def _threshold_for_asset(row: pd.Series) -> float:
    asset_class = str(row.get("asset_class", "unknown")).lower()
    role = str(row.get("role", "unknown")).lower()
    ticker = str(row.get("ticker", "")).upper()

    if asset_class in {"crypto", "digital_asset"}:
        return 0.80

    if asset_class in {"fx", "forex", "currency"} or "-" in ticker:
        return 0.15

    if "bond" in asset_class or "bond" in role or role in {"fixed_income", "treasury"}:
        return 0.20

    if asset_class in {"equity", "stock", "etf"} or role in {"stock", "etf"}:
        return 0.50

    return 0.50

def _mean_abs_diff_pair(df: pd.DataFrame, left: str, right: str, *, min_obs: int = 20) -> float | None:
    if df is None or df.empty:
        return None

    if left not in df.columns or right not in df.columns:
        return None

    pair = df[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < int(min_obs):
        return None

    value = float((pair[left].astype(float) - pair[right].astype(float)).abs().mean())
    return value if np.isfinite(value) else None


def _classify_returns_wide_basis(cmp: pd.DataFrame) -> dict:
    """
    Classify whether returns_wide looks like:
      - simple returns
      - raw log returns
      - log converted to simple
      - unknown

    Important:
    simple and log returns are very close for normal daily moves, so we only call
    it "log" when raw-log distance is near zero AND simple distance is meaningfully worse.
    """
    out = {
        "basis": None,
        "diff_vs_returns_usd_simple_adj": None,
        "diff_vs_returns_usd_simple_close_adjusted": None,
        "diff_vs_returns_usd_log": None,
        "diff_vs_returns_usd_log_as_simple": None,
        "confidence": "none",
    }

    if cmp is None or cmp.empty or "returns_wide" not in cmp.columns:
        return out

    if cmp["returns_wide"].notna().sum() < 20:
        return out

    d_simple_adj = _mean_abs_diff_pair(cmp, "returns_wide", "returns_usd_simple_adj")
    d_simple_close = _mean_abs_diff_pair(cmp, "returns_wide", "returns_usd_simple_close_adjusted")
    d_log = _mean_abs_diff_pair(cmp, "returns_wide", "returns_usd_log")
    d_log_as_simple = _mean_abs_diff_pair(cmp, "returns_wide", "returns_usd_log_as_simple")

    out["diff_vs_returns_usd_simple_adj"] = d_simple_adj
    out["diff_vs_returns_usd_simple_close_adjusted"] = d_simple_close
    out["diff_vs_returns_usd_log"] = d_log
    out["diff_vs_returns_usd_log_as_simple"] = d_log_as_simple

    candidates = {
        "returns_usd_simple_adj": d_simple_adj,
        "returns_usd_simple_close_adjusted": d_simple_close,
        "returns_usd_log": d_log,
        "returns_usd_log_as_simple": d_log_as_simple,
    }
    candidates = {k: v for k, v in candidates.items() if v is not None and np.isfinite(v)}

    if not candidates:
        return out

    best_name, best_diff = min(candidates.items(), key=lambda kv: kv[1])

    # Tolerance for "effectively identical".
    # float32 cache + parquet roundtrips should be far below this for normal returns.
    near_zero = 1e-7

    simple_diffs = [
        v for k, v in candidates.items()
        if k in {"returns_usd_simple_adj", "returns_usd_simple_close_adjusted", "returns_usd_log_as_simple"}
    ]
    best_simple = min(simple_diffs) if simple_diffs else None

    # Strong simple classification.
    if best_name in {
        "returns_usd_simple_adj",
        "returns_usd_simple_close_adjusted",
        "returns_usd_log_as_simple",
    }:
        out["basis"] = (
            "simple"
            if best_name != "returns_usd_log_as_simple"
            else "log_converted_to_simple"
        )
        out["confidence"] = "high" if best_diff <= near_zero else "medium"
        return out

    # Only classify as raw log if raw log is effectively identical AND simple is materially worse.
    if best_name == "returns_usd_log":
        if best_diff <= near_zero and best_simple is not None and best_simple > max(1e-5, best_diff * 100.0):
            out["basis"] = "raw_log"
            out["confidence"] = "high"
        else:
            # Ambiguous case: log is slightly closer, but not enough to prove raw-log basis.
            out["basis"] = "ambiguous_simple_vs_log"
            out["confidence"] = "low"
        return out

    out["basis"] = "unknown"
    out["confidence"] = "low"
    return out

def _summarize_asset_quality(
    *,
    asset_row: pd.Series,
    ohlcv: pd.DataFrame,
    returns_usd: pd.DataFrame,
    returns_wide: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> dict:
    asset_id = str(asset_row["asset_id"])
    ticker = str(asset_row.get("ticker", ""))
    yahoo_ticker = str(asset_row.get("yahoo_ticker", ""))
    name = str(asset_row.get("name", ""))
    asset_class = str(asset_row.get("asset_class", "unknown"))
    role = str(asset_row.get("role", "unknown"))
    region = str(asset_row.get("region", "unknown"))

    threshold = _threshold_for_asset(asset_row)

    # OHLCV prices and returns
    price = pd.Series(dtype="float64")
    ohlcv_simple = pd.Series(dtype="float64")
    ohlcv_log = pd.Series(dtype="float64")

    duplicate_dates = 0
    non_positive_prices = 0
    price_col_used = None

    if ohlcv is not None and not ohlcv.empty:
        duplicate_dates = int(ohlcv["date"].duplicated(keep=False).sum())
        non_positive_prices = int((pd.to_numeric(ohlcv["price"], errors="coerce") <= 0).sum())
        price_col_used = str(ohlcv["__price_col_used"].dropna().iloc[-1]) if "__price_col_used" in ohlcv.columns and not ohlcv["__price_col_used"].dropna().empty else None

        price = (
            ohlcv.set_index("date")["price"]
            .sort_index()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        price = price[price > 0]

        ohlcv_simple = price.pct_change().replace([np.inf, -np.inf], np.nan)
        ohlcv_log = np.log(price / price.shift(1)).replace([np.inf, -np.inf], np.nan)

    # returns_usd columns
    ru_simple_adj = pd.Series(dtype="float64")
    ru_simple_close_adjusted = pd.Series(dtype="float64")
    ru_log = pd.Series(dtype="float64")

    if returns_usd is not None and not returns_usd.empty:
        ru = returns_usd.copy()
        ru = ru.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        ru = ru.set_index("date")

        if "ret_adj_close_usd" in ru.columns:
            ru_simple_adj = pd.to_numeric(ru["ret_adj_close_usd"], errors="coerce").replace([np.inf, -np.inf], np.nan)

        if "ret_close_adjusted_usd" in ru.columns:
            ru_simple_close_adjusted = pd.to_numeric(ru["ret_close_adjusted_usd"], errors="coerce").replace([np.inf, -np.inf], np.nan)

        if "ret_log_close_adjusted_usd" in ru.columns:
            ru_log = pd.to_numeric(ru["ret_log_close_adjusted_usd"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # returns_wide by asset_id
    rw = pd.Series(dtype="float64")
    if returns_wide is not None and not returns_wide.empty and asset_id in returns_wide.columns:
        rw = pd.to_numeric(returns_wide[asset_id], errors="coerce").replace([np.inf, -np.inf], np.nan)
        rw = rw.loc[(rw.index >= start_ts) & (rw.index <= end_ts)]

    # Compare series
    cmp = pd.DataFrame(
        {
            "ohlcv_simple": ohlcv_simple,
            "ohlcv_log": ohlcv_log,
            "returns_usd_simple_adj": ru_simple_adj,
            "returns_usd_simple_close_adjusted": ru_simple_close_adjusted,
            "returns_usd_log": ru_log,
            "returns_usd_log_as_simple": np.expm1(ru_log) if not ru_log.empty else pd.Series(dtype="float64"),
            "returns_wide": rw,
        }
    ).replace([np.inf, -np.inf], np.nan)

    rw_basis_diag = _classify_returns_wide_basis(cmp)
    rw_looks_like = rw_basis_diag.get("basis")

    max_abs_ohlcv_simple = _safe_float(ohlcv_simple.abs().max()) if not ohlcv_simple.empty else None
    max_abs_ru_simple = _safe_float(ru_simple_adj.abs().max()) if not ru_simple_adj.empty else None
    max_abs_ru_log = _safe_float(ru_log.abs().max()) if not ru_log.empty else None
    max_abs_rw = _safe_float(rw.abs().max()) if not rw.empty else None

    # Main failure criteria
    flags: list[str] = []

    if price.empty:
        flags.append("missing_ohlcv_prices")

    if duplicate_dates > 0:
        flags.append("duplicate_ohlcv_dates")

    if non_positive_prices > 0:
        flags.append("non_positive_ohlcv_prices")

    if max_abs_ohlcv_simple is not None and max_abs_ohlcv_simple > threshold:
        flags.append(f"ohlcv_simple_extreme_return>{threshold}")

    if max_abs_ru_simple is not None and max_abs_ru_simple > threshold:
        flags.append(f"returns_usd_simple_extreme_return>{threshold}")

    # Log returns are not thresholded against the same simple-return threshold,
    # but very large log values are still suspicious.
    if max_abs_ru_log is not None and max_abs_ru_log > np.log1p(threshold):
        flags.append(f"returns_usd_log_extreme_return>log1p({threshold})")

    if rw.empty:
        flags.append("missing_returns_wide_asset_id")

    if rw_looks_like == "raw_log":
        flags.append("returns_wide_appears_to_be_raw_log_returns")

    # Check consistency between OHLCV-derived simple returns and returns_usd simple returns.
    pair_oh_ru = cmp[["ohlcv_simple", "returns_usd_simple_adj"]].dropna()
    oh_ru_corr = None
    oh_ru_max_abs_diff = None
    if len(pair_oh_ru) >= 20:
        oh_ru_corr = _safe_float(pair_oh_ru["ohlcv_simple"].corr(pair_oh_ru["returns_usd_simple_adj"]))
        oh_ru_max_abs_diff = _safe_float((pair_oh_ru["ohlcv_simple"] - pair_oh_ru["returns_usd_simple_adj"]).abs().max())
        if oh_ru_corr is not None and oh_ru_corr < 0.98:
            flags.append("ohlcv_vs_returns_usd_simple_low_corr")
        if oh_ru_max_abs_diff is not None and oh_ru_max_abs_diff > 0.02:
            flags.append("ohlcv_vs_returns_usd_simple_large_diff")

    status = "PASS"
    if flags:
        severe = [
            f for f in flags
            if (
                "missing_ohlcv_prices" in f
                or "non_positive" in f
                or "extreme_return" in f
                or "missing_returns_wide" in f
                or "appears_to_be_raw_log" in f
                or "raw_log" in f
            )
        ]
        status = "FAIL" if severe else "WARN"

    return {
        "asset_id": asset_id,
        "ticker": ticker,
        "yahoo_ticker": yahoo_ticker,
        "display_symbol": yahoo_ticker or ticker or asset_id,
        "name": name,
        "asset_class": asset_class,
        "role": role,
        "region": region,
        "threshold_max_abs_simple_return": float(threshold),
        "status": status,
        "flags": flags,

        "ohlcv": {
            "n_rows": int(len(ohlcv)) if ohlcv is not None else 0,
            "first_date": None if price.empty else str(price.index.min().date()),
            "last_date": None if price.empty else str(price.index.max().date()),
            "duplicate_dates": int(duplicate_dates),
            "non_positive_prices": int(non_positive_prices),
            "price_col_used": price_col_used,
            "max_abs_simple_return": max_abs_ohlcv_simple,
            "max_abs_log_return": _safe_float(ohlcv_log.abs().max()) if not ohlcv_log.empty else None,
            "mdd_from_simple_returns": _max_drawdown_from_simple_returns(ohlcv_simple),
            "largest_simple_returns": _largest_abs_rows(ohlcv_simple, n=10),
        },

        "returns_usd": {
            "n_rows": int(len(returns_usd)) if returns_usd is not None else 0,
            "first_date": _safe_date_str(returns_usd["date"].min()) if returns_usd is not None and not returns_usd.empty and "date" in returns_usd.columns else None,
            "last_date": _safe_date_str(returns_usd["date"].max()) if returns_usd is not None and not returns_usd.empty and "date" in returns_usd.columns else None,
            "max_abs_ret_adj_close_usd": max_abs_ru_simple,
            "max_abs_ret_close_adjusted_usd": _safe_float(ru_simple_close_adjusted.abs().max()) if not ru_simple_close_adjusted.empty else None,
            "max_abs_ret_log_close_adjusted_usd": max_abs_ru_log,
            "largest_ret_adj_close_usd": _largest_abs_rows(ru_simple_adj, n=10),
            "largest_ret_log_close_adjusted_usd": _largest_abs_rows(ru_log, n=10),
        },

        "returns_wide": {
            "exists": bool(not rw.empty),
            "n_rows": int(rw.notna().sum()) if not rw.empty else 0,
            "first_date": None if rw.dropna().empty else str(rw.dropna().index.min().date()),
            "last_date": None if rw.dropna().empty else str(rw.dropna().index.max().date()),
            "max_abs_return": max_abs_rw,
            "looks_like": rw_looks_like,
            "basis_diagnostic": rw_basis_diag,
            "largest_returns": _largest_abs_rows(rw, n=10),
        },

        "comparisons": {
            "ohlcv_simple_vs_returns_usd_simple_corr": oh_ru_corr,
            "ohlcv_simple_vs_returns_usd_simple_max_abs_diff": oh_ru_max_abs_diff,
        },
    }


# ---------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------

def run_diagnostic(
    *,
    env: str | None,
    as_of: str,
    start: str,
    universe_csv: str | None,
    cache_min_years: float,
    asset_ids: list[str] | None,
    max_assets: int | None,
    workers: int,
    write_outputs: bool,
    confirm_prod_write: bool,
) -> dict:
    cfg = load_runtime_config(env)

    if write_outputs:
        require_prod_confirmation(cfg, bool(confirm_prod_write))

    bucket = cfg.bucket
    region = cfg.region
    market_root = cfg.market_root.strip("/")

    s3 = s3_init(region)
    store = _make_market_store(bucket=bucket, region=region, market_root=market_root)

    start_ts = pd.Timestamp(start).tz_localize(None).normalize()
    end_ts = pd.Timestamp(as_of).tz_localize(None).normalize()

    universe = _load_universe(universe_csv=universe_csv)
    active = universe[universe["include"].astype(int) == 1].copy()
    active = active[active["asset_id"] != ""].copy()

    identity_diag = _universe_duplicate_diagnostics(active)

    if asset_ids:
        selected_set = {_norm_asset_id(x) for x in asset_ids if _norm_asset_id(x)}
        active = active[active["asset_id"].isin(selected_set)].copy()

    active = active.sort_values(["asset_class", "ticker", "asset_id"])

    if max_assets is not None and int(max_assets) > 0:
        active = active.head(int(max_assets)).copy()

    returns_wide = _load_returns_wide_cache(
        bucket=bucket,
        market_root=market_root,
        min_years=cache_min_years,
    )

    rows: list[dict] = []
    details: list[dict] = []

    total = len(active)
    print("\n=== MARKET DATA INTEGRITY DIAGNOSTIC ===")
    print(f"env:          {cfg.env}")
    print(f"bucket:       {bucket}")
    print(f"market_root:  {market_root}")
    print(f"window:       {start_ts.date()}..{end_ts.date()}")
    print(f"assets:       {total}")
    print(f"write:        {write_outputs}")
    print("")

    def _diagnose_one_asset(row_dict: dict) -> dict:
        row = pd.Series(row_dict)
        asset_id = str(row["asset_id"])
        display = str(row.get("yahoo_ticker") or row.get("ticker") or asset_id)

        try:
            ohlcv = _load_ohlcv_for_asset_id(
                s3=s3,
                bucket=bucket,
                market_root=market_root,
                asset_id=asset_id,
                start_ts=start_ts,
                end_ts=end_ts,
            )

            returns_usd = _load_returns_usd_for_asset_id(
                store=store,
                asset_id=asset_id,
                start=str(start_ts.date()),
                end=str(end_ts.date()),
            )

            detail = _summarize_asset_quality(
                asset_row=row,
                ohlcv=ohlcv,
                returns_usd=returns_usd,
                returns_wide=returns_wide,
                start_ts=start_ts,
                end_ts=end_ts,
            )

        except Exception as exc:
            detail = {
                "asset_id": asset_id,
                "ticker": str(row.get("ticker", "")),
                "yahoo_ticker": str(row.get("yahoo_ticker", "")),
                "display_symbol": display,
                "name": str(row.get("name", "")),
                "asset_class": str(row.get("asset_class", "")),
                "role": str(row.get("role", "")),
                "region": str(row.get("region", "")),
                "status": "FAIL",
                "flags": [f"diagnostic_exception:{type(exc).__name__}"],
                "error": str(exc),
            }

        return detail


    def _summary_row(detail: dict) -> dict:
        return {
            "asset_id": detail.get("asset_id"),
            "ticker": detail.get("ticker"),
            "yahoo_ticker": detail.get("yahoo_ticker"),
            "display_symbol": detail.get("display_symbol"),
            "name": detail.get("name"),
            "asset_class": detail.get("asset_class"),
            "role": detail.get("role"),
            "region": detail.get("region"),
            "status": detail.get("status"),
            "flags": ",".join(detail.get("flags") or []),
            "threshold_max_abs_simple_return": detail.get("threshold_max_abs_simple_return"),
            "ohlcv_n_rows": (detail.get("ohlcv") or {}).get("n_rows"),
            "ohlcv_first_date": (detail.get("ohlcv") or {}).get("first_date"),
            "ohlcv_last_date": (detail.get("ohlcv") or {}).get("last_date"),
            "ohlcv_price_col_used": (detail.get("ohlcv") or {}).get("price_col_used"),
            "ohlcv_max_abs_simple_return": (detail.get("ohlcv") or {}).get("max_abs_simple_return"),
            "ohlcv_max_abs_log_return": (detail.get("ohlcv") or {}).get("max_abs_log_return"),
            "ohlcv_mdd_from_simple_returns": (detail.get("ohlcv") or {}).get("mdd_from_simple_returns"),
            "returns_usd_n_rows": (detail.get("returns_usd") or {}).get("n_rows"),
            "returns_usd_max_abs_ret_adj_close_usd": (detail.get("returns_usd") or {}).get("max_abs_ret_adj_close_usd"),
            "returns_usd_max_abs_ret_log_close_adjusted_usd": (detail.get("returns_usd") or {}).get("max_abs_ret_log_close_adjusted_usd"),
            "returns_wide_exists": (detail.get("returns_wide") or {}).get("exists"),
            "returns_wide_n_rows": (detail.get("returns_wide") or {}).get("n_rows"),
            "returns_wide_max_abs_return": (detail.get("returns_wide") or {}).get("max_abs_return"),
            "returns_wide_looks_like": (detail.get("returns_wide") or {}).get("looks_like"),
            "returns_wide_basis_confidence": ((detail.get("returns_wide") or {}).get("basis_diagnostic") or {}).get("confidence"),
            "returns_wide_diff_vs_simple_adj": ((detail.get("returns_wide") or {}).get("basis_diagnostic") or {}).get("diff_vs_returns_usd_simple_adj"),
            "returns_wide_diff_vs_log": ((detail.get("returns_wide") or {}).get("basis_diagnostic") or {}).get("diff_vs_returns_usd_log"),
            "ohlcv_vs_returns_usd_simple_corr": (detail.get("comparisons") or {}).get("ohlcv_simple_vs_returns_usd_simple_corr"),
            "ohlcv_vs_returns_usd_simple_max_abs_diff": (detail.get("comparisons") or {}).get("ohlcv_simple_vs_returns_usd_simple_max_abs_diff"),
        }


    active_records = active.to_dict("records")
    max_workers = max(1, int(workers or 1))

    print(f"[parallel] workers={max_workers}")

    details = []
    rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_diagnose_one_asset, row_dict): row_dict.get("asset_id")
            for row_dict in active_records
        }

        for i, future in enumerate(as_completed(futures), start=1):
            detail = future.result()
            details.append(detail)
            rows.append(_summary_row(detail))

            if i % 25 == 0 or i == total:
                print(f"[progress] {i}/{total}")

    summary_df = pd.DataFrame(rows)

    counts = summary_df["status"].value_counts(dropna=False).to_dict() if not summary_df.empty else {}
    fail_assets = summary_df[summary_df["status"] == "FAIL"].copy() if not summary_df.empty else pd.DataFrame()
    warn_assets = summary_df[summary_df["status"] == "WARN"].copy() if not summary_df.empty else pd.DataFrame()

    payload = {
        "schema_version": "market_data_integrity_v1",
        "as_of": str(end_ts.date()),
        "start": str(start_ts.date()),
        "runtime": {
            "env": cfg.env,
            "bucket": bucket,
            "region": region,
            "market_root": market_root,
        },
        "identity": identity_diag,
        "summary": {
            "n_assets_checked": int(len(summary_df)),
            "status_counts": {str(k): int(v) for k, v in counts.items()},
            "n_fail": int(len(fail_assets)),
            "n_warn": int(len(warn_assets)),
            "returns_wide_columns": int(returns_wide.shape[1]) if returns_wide is not None and not returns_wide.empty else 0,
            "returns_wide_rows": int(returns_wide.shape[0]) if returns_wide is not None and not returns_wide.empty else 0,
        },
        "fail_assets_sample": fail_assets.head(100).to_dict("records") if not fail_assets.empty else [],
        "warn_assets_sample": warn_assets.head(100).to_dict("records") if not warn_assets.empty else [],
        "details": details,
    }

    print("\n=== DATA QUALITY SUMMARY ===")
    print(f"checked: {len(summary_df)}")
    print(f"counts:  {counts}")
    print(f"identity_status: {identity_diag.get('identity_status')}")
    print(f"duplicate_tickers: {identity_diag.get('duplicate_ticker_count')}")
    print(f"duplicate_asset_ids: {identity_diag.get('duplicate_asset_id_count')}")

    if not fail_assets.empty:
        print("\nTop FAIL assets:")
        cols = [
            "display_symbol",
            "asset_id",
            "name",
            "asset_class",
            "flags",
            "ohlcv_max_abs_simple_return",
            "returns_usd_max_abs_ret_adj_close_usd",
            "returns_wide_max_abs_return",
            "returns_wide_looks_like",
        ]
        cols = [c for c in cols if c in fail_assets.columns]
        print(fail_assets[cols].head(20).to_string(index=False))

    if write_outputs:
        dt = end_ts.normalize()

        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=market_root,
            table=DIAG_TABLE,
            dt=dt,
            filename="market_data_integrity.json",
            payload=payload,
            update_latest=True,
        )

        s3_write_parquet_partition(
            s3,
            bucket=bucket,
            root_prefix=market_root,
            table=DIAG_TABLE,
            dt=dt,
            filename="market_data_integrity_summary.parquet",
            df=summary_df,
        )

        print(
            f"\n[S3] Wrote diagnostics under "
            f"s3://{bucket}/{market_root}/{DIAG_TABLE}/dt={end_ts.date()}/"
        )

    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diagnose Alpha Edge market data integrity by asset_id.")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    ap.add_argument("--start", default="2015-01-01", help="YYYY-MM-DD")
    ap.add_argument("--universe-csv", default=None)
    ap.add_argument("--cache-min-years", type=float, default=5.0)

    ap.add_argument(
        "--asset-id",
        action="append",
        default=None,
        help="Optional asset_id to diagnose. Can be passed multiple times.",
    )
    ap.add_argument("--max-assets", type=int, default=None)

    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--confirm-prod-write", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel asset diagnostics to run.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    run_diagnostic(
        env=args.env,
        as_of=args.as_of,
        start=args.start,
        universe_csv=args.universe_csv,
        cache_min_years=float(args.cache_min_years),
        asset_ids=args.asset_id,
        max_assets=args.max_assets,
        workers=int(args.workers),
        write_outputs=(not bool(args.no_write)),
        confirm_prod_write=bool(args.confirm_prod_write),
    )


if __name__ == "__main__":
    main()