from __future__ import annotations

import argparse
import json
import io
from dataclasses import asdict
from typing import Any

import boto3
import numpy as np
import pandas as pd

from alpha_edge import paths
from alpha_edge.core.data_loader import (
    clean_returns_matrix,
    s3_get_json,
    s3_init,
    s3_write_json_event,
)
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.jobs.run_daily_report import _load_closes_usd_from_ohlcv


PORTFOLIO_RUNS_TABLE = "portfolio_search/runs"
DIAG_TABLE = "diagnostics/returns_source_compare"


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _load_universe() -> pd.DataFrame:
    u = pd.read_csv(paths.universe_dir() / "universe.csv")
    u = u.copy()
    u["asset_id"] = u["asset_id"].astype(str).str.strip()
    u["ticker"] = u["ticker"].astype(str).str.upper().str.strip()

    if "include" in u.columns:
        u["include"] = pd.to_numeric(u["include"], errors="coerce").fillna(1).astype(int)
    else:
        u["include"] = 1

    return u


def _portfolio_run_key(root_prefix: str, run_dt: str, run_id: str) -> str:
    return (
        f"{root_prefix.strip('/')}/"
        f"{PORTFOLIO_RUNS_TABLE}/"
        f"dt={run_dt}/"
        f"run_{run_id}.json"
    )


def _load_returns_wide(
    *,
    bucket: str,
    market_root: str,
    min_years: float,
    as_of: str,
    universe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    path = (
        f"s3://{bucket}/"
        f"{market_root.strip('/')}/cache/v1/"
        f"returns_wide_min{int(float(min_years))}y.parquet"
    )

    returns_wide = pd.read_parquet(path, engine="pyarrow").sort_index()
    returns_wide, diag = clean_returns_matrix(returns_wide)

    as_of_ts = pd.Timestamp(as_of).tz_localize(None).normalize()
    returns_wide.index = pd.to_datetime(returns_wide.index, errors="coerce").tz_localize(None).normalize()
    returns_wide = returns_wide.loc[returns_wide.index <= as_of_ts]

    asset_to_ticker = dict(zip(universe["asset_id"], universe["ticker"]))
    returns_wide = returns_wide.rename(columns=lambda c: asset_to_ticker.get(str(c).strip(), str(c).strip()))
    returns_wide.columns = [str(c).upper().strip() for c in returns_wide.columns]

    return returns_wide, diag


def _derive_ohlcv_returns(
    *,
    tickers: list[str],
    start: str,
    as_of: str,
    bucket: str,
    market_root: str,
    region: str,
) -> dict[str, pd.DataFrame]:
    closes = _load_closes_usd_from_ohlcv(
        tickers=tickers,
        start=start,
        end=as_of,
        s3_bucket=bucket,
        s3_root_prefix=f"{market_root.strip('/')}/ohlcv_usd/v1",
        s3_region=region,
    )

    closes.index = pd.to_datetime(closes.index, errors="coerce").tz_localize(None).normalize()
    closes = closes.sort_index()

    simple = closes.pct_change()
    logret = np.log(closes / closes.shift(1))

    simple = simple.replace([np.inf, -np.inf], np.nan)
    logret = logret.replace([np.inf, -np.inf], np.nan)

    return {
        "closes": closes,
        "simple": simple,
        "log": logret,
    }


def _max_drawdown_from_returns(r: pd.Series) -> float | None:
    r = pd.to_numeric(r, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return None
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return _safe_float(dd.min())


def _summarize_asset_compare(
    *,
    ticker: str,
    rw: pd.Series,
    oh_simple: pd.Series,
    oh_log: pd.Series,
) -> dict:
    df = pd.DataFrame(
        {
            "returns_wide": rw,
            "ohlcv_simple": oh_simple,
            "ohlcv_log": oh_log,
        }
    ).replace([np.inf, -np.inf], np.nan)

    aligned = df.dropna(how="any")

    out = {
        "ticker": ticker,
        "rw_first": None if rw.dropna().empty else str(rw.dropna().index.min().date()),
        "rw_last": None if rw.dropna().empty else str(rw.dropna().index.max().date()),
        "ohlcv_first": None if oh_simple.dropna().empty else str(oh_simple.dropna().index.min().date()),
        "ohlcv_last": None if oh_simple.dropna().empty else str(oh_simple.dropna().index.max().date()),
        "rw_n": int(rw.notna().sum()),
        "ohlcv_simple_n": int(oh_simple.notna().sum()),
        "ohlcv_log_n": int(oh_log.notna().sum()),
        "aligned_n": int(len(aligned)),
    }

    if len(aligned) >= 20:
        out.update(
            {
                "corr_rw_vs_ohlcv_simple": _safe_float(aligned["returns_wide"].corr(aligned["ohlcv_simple"])),
                "corr_rw_vs_ohlcv_log": _safe_float(aligned["returns_wide"].corr(aligned["ohlcv_log"])),
                "corr_rw_vs_negative_ohlcv_simple": _safe_float(aligned["returns_wide"].corr(-aligned["ohlcv_simple"])),
                "mean_abs_diff_simple": _safe_float((aligned["returns_wide"] - aligned["ohlcv_simple"]).abs().mean()),
                "max_abs_diff_simple": _safe_float((aligned["returns_wide"] - aligned["ohlcv_simple"]).abs().max()),
                "max_abs_rw_return": _safe_float(aligned["returns_wide"].abs().max()),
                "max_abs_ohlcv_simple_return": _safe_float(aligned["ohlcv_simple"].abs().max()),
                "rw_mdd": _max_drawdown_from_returns(aligned["returns_wide"]),
                "ohlcv_simple_mdd": _max_drawdown_from_returns(aligned["ohlcv_simple"]),
            }
        )

        diff = (aligned["returns_wide"] - aligned["ohlcv_simple"]).abs().sort_values(ascending=False)
        out["largest_diffs"] = [
            {
                "date": str(pd.Timestamp(idx).date()),
                "returns_wide": _safe_float(aligned.loc[idx, "returns_wide"]),
                "ohlcv_simple": _safe_float(aligned.loc[idx, "ohlcv_simple"]),
                "ohlcv_log": _safe_float(aligned.loc[idx, "ohlcv_log"]),
                "abs_diff_simple": _safe_float(value),
            }
            for idx, value in diff.head(10).items()
        ]
    else:
        out.update(
            {
                "corr_rw_vs_ohlcv_simple": None,
                "corr_rw_vs_ohlcv_log": None,
                "corr_rw_vs_negative_ohlcv_simple": None,
                "mean_abs_diff_simple": None,
                "max_abs_diff_simple": None,
                "largest_diffs": [],
            }
        )

    return out


def _portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    cols = [t for t in weights if t in returns.columns]
    if not cols:
        return pd.Series(dtype="float64")

    w = pd.Series({t: float(weights[t]) for t in cols}, dtype="float64")
    gross = float(w.abs().sum())
    if not np.isfinite(gross) or gross <= 0:
        return pd.Series(dtype="float64")

    w = w / gross
    r = returns[cols].copy()
    return r.mul(w, axis=1).sum(axis=1).replace([np.inf, -np.inf], np.nan).dropna()


def diagnose(
    *,
    run_id: str,
    run_dt: str,
    as_of: str,
    env: str | None,
    cache_min_years: float,
    write_outputs: bool,
    confirm_prod_write: bool,
) -> dict:
    cfg = load_runtime_config(env)
    if write_outputs:
        require_prod_confirmation(cfg, confirm_prod_write)

    bucket = cfg.bucket
    region = cfg.region
    engine_root = cfg.engine_root.strip("/")
    market_root = cfg.market_root.strip("/")

    s3 = s3_init(region)

    run_key = _portfolio_run_key(engine_root, run_dt, run_id)
    payload = s3_get_json(s3, bucket=bucket, key=run_key)

    outputs = payload.get("outputs") or {}
    final_exec = outputs.get("final_executable") or {}
    disc = outputs.get("discrete_allocation") or {}
    candidate_context = outputs.get("candidate_context") or {}

    weights = final_exec.get("weights") or {}
    shares = disc.get("shares") or {}

    weights = {str(k).upper().strip(): float(v) for k, v in weights.items() if _safe_float(v) is not None}
    shares = {str(k).upper().strip(): float(v) for k, v in shares.items() if _safe_float(v) is not None}
    tickers = sorted(set(weights) | set(shares))

    if not tickers:
        raise RuntimeError("Could not extract tickers from final_executable.weights or discrete_allocation.shares")

    universe = _load_universe()

    ticker_to_asset_rows = []
    for t in tickers:
        matches = universe[universe["ticker"] == t]
        ticker_to_asset_rows.append(
            {
                "ticker": t,
                "n_universe_matches": int(len(matches)),
                "asset_ids": matches["asset_id"].tolist(),
                "asset_classes": matches["asset_class"].astype(str).tolist() if "asset_class" in matches.columns else [],
                "roles": matches["role"].astype(str).tolist() if "role" in matches.columns else [],
                "regions": matches["region"].astype(str).tolist() if "region" in matches.columns else [],
            }
        )

    returns_wide, clean_diag = _load_returns_wide(
        bucket=bucket,
        market_root=market_root,
        min_years=cache_min_years,
        as_of=as_of,
        universe=universe,
    )

    ohlcv = _derive_ohlcv_returns(
        tickers=tickers,
        start="2015-01-01",
        as_of=as_of,
        bucket=bucket,
        market_root=market_root,
        region=region,
    )

    rw_sub = returns_wide[[t for t in tickers if t in returns_wide.columns]].copy()
    oh_simple = ohlcv["simple"][[t for t in tickers if t in ohlcv["simple"].columns]].copy()
    oh_log = ohlcv["log"][[t for t in tickers if t in ohlcv["log"].columns]].copy()

    asset_rows = []
    for t in tickers:
        asset_rows.append(
            _summarize_asset_compare(
                ticker=t,
                rw=rw_sub[t] if t in rw_sub.columns else pd.Series(dtype="float64"),
                oh_simple=oh_simple[t] if t in oh_simple.columns else pd.Series(dtype="float64"),
                oh_log=oh_log[t] if t in oh_log.columns else pd.Series(dtype="float64"),
            )
        )

    common_cols = [t for t in tickers if t in rw_sub.columns and t in oh_simple.columns]
    common_idx = rw_sub.index.intersection(oh_simple.index)

    rw_aligned = rw_sub.reindex(common_idx)[common_cols]
    oh_aligned = oh_simple.reindex(common_idx)[common_cols]

    rw_port = _portfolio_returns(rw_aligned, weights)
    oh_port = _portfolio_returns(oh_aligned, weights)

    port_cmp = pd.DataFrame(
        {
            "returns_wide_port": rw_port,
            "ohlcv_simple_port": oh_port,
        }
    ).dropna(how="any")

    fx_like = [
        t for t in tickers
        if "-" in t and any(ccy in t for ccy in ["USD", "EUR", "JPY", "CNY", "BRL", "GBP", "CHF"])
    ]
    fx_gross_weight = float(sum(abs(float(weights.get(t, 0.0))) for t in fx_like))

    portfolio_summary = {
        "common_tickers": common_cols,
        "n_common_tickers": int(len(common_cols)),
        "common_start": None if port_cmp.empty else str(port_cmp.index.min().date()),
        "common_end": None if port_cmp.empty else str(port_cmp.index.max().date()),
        "common_rows": int(len(port_cmp)),
        "portfolio_corr": None if len(port_cmp) < 20 else _safe_float(port_cmp["returns_wide_port"].corr(port_cmp["ohlcv_simple_port"])),
        "portfolio_corr_negative": None if len(port_cmp) < 20 else _safe_float(port_cmp["returns_wide_port"].corr(-port_cmp["ohlcv_simple_port"])),
        "returns_wide_port_mdd": _max_drawdown_from_returns(port_cmp["returns_wide_port"]) if not port_cmp.empty else None,
        "ohlcv_simple_port_mdd": _max_drawdown_from_returns(port_cmp["ohlcv_simple_port"]) if not port_cmp.empty else None,
        "returns_wide_port_max_abs_return": None if port_cmp.empty else _safe_float(port_cmp["returns_wide_port"].abs().max()),
        "ohlcv_simple_port_max_abs_return": None if port_cmp.empty else _safe_float(port_cmp["ohlcv_simple_port"].abs().max()),
        "fx_like_tickers": fx_like,
        "fx_gross_weight": fx_gross_weight,
    }

    if not port_cmp.empty:
        d = (port_cmp["returns_wide_port"] - port_cmp["ohlcv_simple_port"]).abs().sort_values(ascending=False)
        portfolio_summary["largest_portfolio_diffs"] = [
            {
                "date": str(pd.Timestamp(idx).date()),
                "returns_wide_port": _safe_float(port_cmp.loc[idx, "returns_wide_port"]),
                "ohlcv_simple_port": _safe_float(port_cmp.loc[idx, "ohlcv_simple_port"]),
                "abs_diff": _safe_float(value),
            }
            for idx, value in d.head(20).items()
        ]
    else:
        portfolio_summary["largest_portfolio_diffs"] = []

    out = {
        "schema_version": "returns_source_compare_v1",
        "run_id": run_id,
        "run_dt": run_dt,
        "as_of": as_of,
        "runtime": {
            "env": cfg.env,
            "bucket": bucket,
            "region": region,
            "engine_root": engine_root,
            "market_root": market_root,
        },
        "source_run_key": run_key,
        "candidate_context": candidate_context,
        "search_baseline": {
            "health_score": final_exec.get("health_score"),
            "health_grade": final_exec.get("health_grade"),
            "status": final_exec.get("status"),
            "selected_candidate_label": final_exec.get("selected_candidate_label"),
            "gross_notional": final_exec.get("gross_notional"),
        },
        "mapping": ticker_to_asset_rows,
        "weights": weights,
        "shares": shares,
        "returns_wide_clean_diag": clean_diag,
        "asset_comparison": asset_rows,
        "portfolio_comparison": portfolio_summary,
    }

    print("\n=== RETURNS SOURCE MISMATCH DIAGNOSTIC ===")
    print(f"run_id:              {run_id}")
    print(f"as_of:               {as_of}")
    print(f"tickers:             {len(tickers)}")
    print(f"common_tickers:      {len(common_cols)}")
    print(f"portfolio_corr:      {portfolio_summary.get('portfolio_corr')}")
    print(f"rw_port_mdd:         {portfolio_summary.get('returns_wide_port_mdd')}")
    print(f"ohlcv_port_mdd:      {portfolio_summary.get('ohlcv_simple_port_mdd')}")
    print(f"fx_gross_weight:     {fx_gross_weight:.2%}")

    print("\nWorst asset return mismatches:")
    ranked = sorted(
        asset_rows,
        key=lambda r: -float(r.get("max_abs_diff_simple") or 0.0),
    )
    for r in ranked[:10]:
        print(
            f"  {r['ticker']:<12} "
            f"corr={r.get('corr_rw_vs_ohlcv_simple')} "
            f"corr_neg={r.get('corr_rw_vs_negative_ohlcv_simple')} "
            f"max_abs_diff={r.get('max_abs_diff_simple')} "
            f"rw_max_abs={r.get('max_abs_rw_return')} "
            f"oh_max_abs={r.get('max_abs_ohlcv_simple_return')}"
        )

    if write_outputs:
        dt = pd.Timestamp(as_of).normalize()
        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=engine_root,
            table=DIAG_TABLE,
            dt=dt,
            filename=f"returns_compare_{run_id}.json",
            payload=out,
            update_latest=True,
        )

    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Diagnose returns_wide vs OHLCV-derived returns mismatch.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-dt", required=True, help="Portfolio search run partition date, YYYY-MM-DD.")
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--cache-min-years", type=float, default=5.0)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--confirm-prod-write", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    diagnose(
        run_id=args.run_id,
        run_dt=args.run_dt,
        as_of=args.as_of,
        env=args.env,
        cache_min_years=float(args.cache_min_years),
        write_outputs=(not bool(args.no_write)),
        confirm_prod_write=bool(args.confirm_prod_write),
    )


if __name__ == "__main__":
    main()