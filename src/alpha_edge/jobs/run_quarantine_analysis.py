# src/alpha_edge/jobs/run_quarantine_analysis.py
from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run

import argparse
import datetime as dt
import json
import io
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.core.data_loader import (
    s3_get_json,
    s3_init,
    s3_load_latest_json,
    s3_write_json_event,
)

from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import Position, ScoreConfig
from alpha_edge.portfolio.report_engine import build_portfolio_report

from alpha_edge.risk.actuarial.portfolio_search_output import (
    build_actuarial_diagnostic_from_portfolio_report,
)


DEFAULT_ENGINE_BUCKET = "alpha-edge-algo"
DEFAULT_ENGINE_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT_PREFIX = "engine/v1"

QUAR_EVALS_TABLE = "quarantine/evals"
QUAR_SUMMARY_TABLE = "quarantine/summary"
QUAR_CAND_TABLE = "quarantine/candidates"
QUAR_REPORTS_TABLE = "quarantine/reports"

PORTFOLIO_RUNS_TABLE = "portfolio_search/runs"


# ----------------------------
# Runtime helpers
# ----------------------------
def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_ENGINE_BUCKET))


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_ENGINE_REGION))


def cfg_engine_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "engine_root", DEFAULT_ENGINE_ROOT_PREFIX)).strip("/")


def cfg_env(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "env", "dev"))


def cfg_market_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "market_root", "market")).strip("/")


def _resolve_root_prefix(*, engine_root: str, backtest_run_id: str | None) -> str:
    root = str(engine_root).strip("/")
    if backtest_run_id:
        return f"{root}/backtests/{backtest_run_id}"
    return root


# ----------------------------
# S3 / small helpers
# ----------------------------
def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _s3_list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None

    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)

        for obj in resp.get("Contents", []) or []:
            k = obj.get("Key", "")
            if k:
                keys.append(k)

        if not resp.get("IsTruncated"):
            break

        token = resp.get("NextContinuationToken")

    return keys




def _read_parquet_s3_bytes(s3, *, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return pd.read_parquet(io.BytesIO(body), engine="pyarrow")


def _norm_key(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _load_active_universe_resolution_maps() -> dict:
    """
    Build asset-id-first resolution maps for quarantine.

    Quarantine must calculate by asset_id. Legacy states may still contain
    tickers or uppercased asset_ids, so this resolver supports:
      - exact/case-insensitive asset_id -> canonical asset_id
      - unambiguous ticker/yahoo_ticker/yahoo_ticker_norm -> canonical asset_id
      - ambiguous symbols are intentionally not mapped
    """
    from alpha_edge import paths

    u = pd.read_csv(paths.universe_dir() / "universe.csv").copy()

    if "asset_id" not in u.columns:
        raise RuntimeError("Universe missing required column asset_id.")

    if "include" in u.columns:
        u["include"] = pd.to_numeric(u["include"], errors="coerce").fillna(1).astype(int)
    else:
        u["include"] = 1

    for col in ["ticker", "yahoo_ticker", "yahoo_ticker_norm", "name"]:
        if col not in u.columns:
            u[col] = ""

    u["asset_id"] = u["asset_id"].map(_norm_key)
    u = u[(u["include"] == 1) & (u["asset_id"] != "")].copy()

    dup_asset = u[u["asset_id"].duplicated(keep=False)].sort_values("asset_id")
    if not dup_asset.empty:
        cols = [c for c in ["asset_id", "ticker", "yahoo_ticker_norm", "yahoo_ticker", "name"] if c in dup_asset.columns]
        raise RuntimeError(
            "Duplicate active asset_id values found in universe. asset_id must be unique.\n"
            + dup_asset[cols].head(50).to_string(index=False)
        )

    asset_id_casefold: dict[str, str] = {}
    display: dict[str, dict] = {}
    symbol_to_assets: dict[str, set[str]] = {}

    for _, row in u.iterrows():
        aid = _norm_key(row.get("asset_id"))
        if not aid:
            continue

        asset_id_casefold[aid.casefold()] = aid
        asset_id_casefold[aid.upper().casefold()] = aid
        asset_id_casefold[aid.lower().casefold()] = aid

        ticker = _norm_key(row.get("ticker"))
        yahoo = _norm_key(row.get("yahoo_ticker"))
        ynorm = _norm_key(row.get("yahoo_ticker_norm"))
        name = _norm_key(row.get("name"))
        disp = ynorm or yahoo or ticker or aid

        display[aid] = {
            "asset_id": aid,
            "ticker": ticker,
            "yahoo_ticker": yahoo,
            "yahoo_ticker_norm": ynorm,
            "display_symbol": disp,
            "name": name,
        }

        for sym in [ticker, yahoo, ynorm]:
            sym = _norm_key(sym)
            if sym:
                symbol_to_assets.setdefault(sym.upper(), set()).add(aid)

    symbol_unique: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    for sym, aids in symbol_to_assets.items():
        if len(aids) == 1:
            symbol_unique[sym] = next(iter(aids))
        else:
            ambiguous[sym] = sorted(aids)

    if ambiguous:
        print(
            "[universe][warn] duplicate active tickers/symbols exist and ambiguous symbols "
            "will not be used as unique lookup keys. "
            f"ambiguous_count={len(ambiguous)} sample={sorted(ambiguous)[:10]}"
        )

    return {
        "asset_id_casefold": asset_id_casefold,
        "symbol_unique": symbol_unique,
        "ambiguous_symbols": ambiguous,
        "display": display,
    }


def _resolve_key_to_asset_id(key: object, maps: dict) -> tuple[str | None, str | None]:
    raw = _norm_key(key)
    if not raw:
        return None, "empty"

    asset_id_casefold = maps.get("asset_id_casefold") or {}
    symbol_unique = maps.get("symbol_unique") or {}
    ambiguous = maps.get("ambiguous_symbols") or {}

    aid = asset_id_casefold.get(raw.casefold())
    if aid:
        return aid, None

    sym = raw.upper()
    if sym in symbol_unique:
        return symbol_unique[sym], None

    if sym in ambiguous:
        return None, "ambiguous_symbol"

    return None, "unresolved"


def _canonicalize_shares_to_asset_id(shares: dict[str, float], *, maps: dict) -> tuple[dict[str, float], dict]:
    out: dict[str, float] = {}
    unresolved: list[dict] = []

    for raw_key, raw_qty in (shares or {}).items():
        q = _safe_float(raw_qty)
        if q is None or abs(q) <= 0:
            continue

        aid, reason = _resolve_key_to_asset_id(raw_key, maps)
        if not aid:
            unresolved.append({"key": str(raw_key), "reason": reason})
            continue

        out[aid] = float(out.get(aid, 0.0) + float(q))

    return {
        k: float(v)
        for k, v in out.items()
        if _safe_float(v) is not None and abs(float(v)) > 0
    }, {
        "input_count": int(len(shares or {})),
        "resolved_count": int(len(out)),
        "unresolved_count": int(len(unresolved)),
        "unresolved_sample": unresolved[:20],
    }


def _canonicalize_weight_dict_to_asset_id(weights: dict, *, maps: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw_key, raw_value in (weights or {}).items():
        v = _safe_float(raw_value)
        if v is None:
            continue
        aid, _reason = _resolve_key_to_asset_id(raw_key, maps)
        if aid:
            out[aid] = float(out.get(aid, 0.0) + float(v))
    return out


def _load_closes_usd_from_ohlcv_asset_ids(
    *,
    asset_ids: list[str],
    start: str,
    end: str,
    s3_bucket: str,
    s3_root_prefix: str,
    s3_region: str,
) -> pd.DataFrame:
    """Load OHLCV closes keyed by canonical asset_id, not ticker."""
    aids: list[str] = []
    seen: set[str] = set()
    for x in asset_ids:
        aid = _norm_key(x)
        if aid and aid not in seen:
            aids.append(aid)
            seen.add(aid)

    if not aids:
        raise RuntimeError("No asset_ids provided to _load_closes_usd_from_ohlcv_asset_ids().")

    start_ts = pd.Timestamp(start).tz_localize(None).normalize()
    end_ts = pd.Timestamp(end).tz_localize(None).normalize()
    years = list(range(int(start_ts.year), int(end_ts.year) + 1))

    import boto3

    s3 = boto3.client("s3", region_name=s3_region)
    all_keys: list[tuple[str, str]] = []
    total_prefixes = len(aids) * len(years)
    seen_prefixes = 0

    print(f"[ohlcv] listing parquet keys asset_ids={len(aids)} years={years[0]}..{years[-1]}")

    for aid in aids:
        for y in years:
            seen_prefixes += 1
            prefix = f"{s3_root_prefix.strip('/')}/asset_id={aid}/year={y}/"
            keys = [
                k for k in _s3_list_keys(s3, bucket=s3_bucket, prefix=prefix)
                if k.lower().endswith(".parquet")
            ]
            for key in keys:
                all_keys.append((aid, key))

            if seen_prefixes % 100 == 0:
                print(f"[ohlcv] listed prefixes={seen_prefixes}/{total_prefixes} keys_so_far={len(all_keys)}")

    if not all_keys:
        raise RuntimeError(
            f"No parquet files found under s3://{s3_bucket}/{s3_root_prefix} "
            f"for asset_ids={aids[:5]}... years={years}"
        )

    frames: list[pd.DataFrame] = []
    for aid, key in all_keys:
        df = _read_parquet_s3_bytes(s3, bucket=s3_bucket, key=key)
        if df is None or df.empty:
            continue

        cols = {str(c).lower(): c for c in df.columns}
        date_col = cols.get("date")
        px_col = cols.get("adj_close_usd") or cols.get("close_usd") or cols.get("adj_close") or cols.get("close")
        if date_col is None or px_col is None:
            raise RuntimeError(
                f"Unexpected OHLCV parquet schema in s3://{s3_bucket}/{key}. Columns={list(df.columns)}"
            )

        out = df[[date_col, px_col]].copy()
        out.columns = ["date", "adj_close_usd"]
        out["asset_id"] = aid
        frames.append(out)

    if not frames:
        raise RuntimeError("Parquet keys were found but all read as empty frames.")

    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    long["adj_close_usd"] = pd.to_numeric(long["adj_close_usd"], errors="coerce")
    long = long.dropna(subset=["date", "asset_id", "adj_close_usd"])
    long = long[(long["date"] >= start_ts) & (long["date"] <= end_ts)].copy()
    long = long.sort_values(["date", "asset_id"])
    long = long.drop_duplicates(subset=["date", "asset_id"], keep="last")

    wide = long.pivot(index="date", columns="asset_id", values="adj_close_usd").sort_index()
    wide = wide.reindex(columns=aids)
    wide = wide.ffill()

    missing_cols = [aid for aid in aids if aid not in wide.columns or wide[aid].dropna().empty]
    if missing_cols:
        raise RuntimeError("Some asset_ids have no OHLCV close history: " + ", ".join(missing_cols[:20]))

    return wide

def _s3_put_text(s3, *, bucket: str, key: str, text: str) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=(text or "").encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )


def _s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _dt_prefix(root_prefix: str, table: str, dt_str: str) -> str:
    return f"{root_prefix.strip('/')}/{table.strip('/')}/dt={dt_str}/"


def _candidate_latest_key(root_prefix: str, table: str, cid: str) -> str:
    return f"{root_prefix.strip('/')}/{table.strip('/')}/candidate_id={cid}/latest.json"


def _metric(eval_dict: dict, *keys: str) -> float | None:
    for k in keys:
        if k in eval_dict:
            v = _safe_float(eval_dict.get(k))
            if v is not None:
                return v
    return None



def _main_goal_probability(eval_dict: dict, goals: list[float] | tuple[float, ...], main_goal: float) -> float | None:
    """Resolve the probability matching main_goal from an EvalMetrics-like dict."""
    if not isinstance(eval_dict, dict):
        return None
    try:
        goals_l = [float(x) for x in goals]
        mg = float(main_goal)
    except Exception:
        goals_l = []
        mg = float("nan")

    idx = None
    for i, g in enumerate(goals_l[:3], start=1):
        if np.isfinite(mg) and abs(float(g) - mg) <= 1e-9:
            idx = i
            break
    if idx is None:
        idx = 1

    return _metric(eval_dict, f"p_hit_goal_{idx}_1y", f"p_hit_goal_{idx}", "p_main", "main_goal_probability")


def _clamp01(x: Any) -> float:
    v = _safe_float(x)
    if v is None:
        return 0.0
    return float(min(1.0, max(0.0, v)))


def _safe_ratio_good(value: Any, cap: Any, *, lower_is_better: bool = True) -> float:
    v0 = _safe_float(value)
    c0 = _safe_float(cap)
    if v0 is None or c0 is None or abs(float(c0)) <= 0:
        return 0.0
    v = abs(float(v0))
    c = abs(float(c0))
    if lower_is_better:
        return _clamp01(1.0 - (v / c))
    return _clamp01(v / c)


def _compute_current_execution_quality(
    *,
    shares: dict[str, float],
    prices_usd: pd.Series,
    target_notional: float | None,
    baseline_weights: dict[str, float] | None,
    baseline_execution_quality: dict | None,
) -> dict:
    gross = 0.0
    for t, qv in shares.items():
        if t not in prices_usd.index:
            continue
        p = _safe_float(prices_usd.loc[t])
        if p is None or p <= 0:
            continue
        gross += abs(float(p) * float(qv))

    target = _safe_float(target_notional)
    if target is None or target <= 0:
        target = gross if gross > 0 else None

    if target is None or target <= 0:
        deployment_ratio = None
        cash_left = None
        cash_weight = None
    else:
        deployment_ratio = float(gross / target)
        cash_left = float(max(0.0, target - gross))
        cash_weight = float(cash_left / target)

    w_today = _signed_gross_weights_from_shares(shares=shares, prices_usd=prices_usd)
    w_base = baseline_weights if isinstance(baseline_weights, dict) and baseline_weights else w_today
    drift = _l1_drift(w_today, {_norm_key(k): float(v) for k, v in w_base.items() if _norm_key(k)}) if w_today else 0.0

    base_eq = baseline_execution_quality if isinstance(baseline_execution_quality, dict) else {}
    dropped = _safe_float(base_eq.get("dropped_theoretical_weight"))
    if dropped is None:
        dropped = 0.0

    return {
        "target_notional": None if target is None else float(target),
        "executable_gross_notional": float(gross),
        "deployment_ratio": None if deployment_ratio is None else float(deployment_ratio),
        "cash_left": None if cash_left is None else float(cash_left),
        "cash_weight": None if cash_weight is None else float(cash_weight),
        "weight_drift_l1": float(drift),
        "dropped_theoretical_weight": float(dropped),
    }


def _compute_quarantine_health_score(
    *,
    eval_dict: dict,
    execution_quality: dict,
    score_cfg: ScoreConfig,
    goals: list[float] | tuple[float, ...],
    main_goal: float,
    max_cash_weight: float,
    min_deployment_ratio: float,
    max_executable_mdd: float,
    max_executable_cdar_95: float,
    max_stability_energy: float,
    max_dropped_weight: float,
    max_weight_drift_l1: float,
) -> dict:
    p_main = _main_goal_probability(eval_dict, goals, main_goal)
    if p_main is None:
        p_main = 0.0

    ruin_cap = float(getattr(score_cfg, "ruin_cap", 0.10))
    cvar_cap = float(getattr(score_cfg, "cvar_cap", 0.03))
    path_mdd_cap = float(getattr(score_cfg, "path_mdd_mean_cap", 0.30))
    p_dd_breach_cap = float(getattr(score_cfg, "p_dd_breach_cap", 0.25))
    underwater_cap = float(getattr(score_cfg, "underwater_mean_cap", 1.00))
    ttr_cap = float(getattr(score_cfg, "ttr_cap_days", 252.0))

    components = {
        "goal_probability": _clamp01(p_main),
        "ruin": _safe_ratio_good(_metric(eval_dict, "ruin_prob_1y", "ruin_probability", "ruin_prob"), ruin_cap),
        "max_drawdown": _safe_ratio_good(_metric(eval_dict, "max_drawdown", "mdd"), max_executable_mdd),
        "cvar_95": _safe_ratio_good(_metric(eval_dict, "cvar_95", "cvar"), cvar_cap),
        "stability_energy": _safe_ratio_good(_metric(eval_dict, "stability_energy"), max_stability_energy),
        "path_mdd_mean": _safe_ratio_good(_metric(eval_dict, "path_mdd_mean"), path_mdd_cap),
        "cdar_95": _safe_ratio_good(_metric(eval_dict, "cdar_95"), max_executable_cdar_95),
        "p_dd_breach": _safe_ratio_good(_metric(eval_dict, "p_dd_breach"), p_dd_breach_cap),
        "underwater_mean": _safe_ratio_good(_metric(eval_dict, "underwater_mean"), underwater_cap),
        "ttr_mean_days": _safe_ratio_good(_metric(eval_dict, "ttr_mean_days"), ttr_cap),
        "deployment": _clamp01(
            float(execution_quality.get("deployment_ratio") or 0.0) / float(min_deployment_ratio)
            if float(min_deployment_ratio) > 0
            else 0.0
        ),
        "cash": _safe_ratio_good(execution_quality.get("cash_weight"), max_cash_weight),
        "weight_drift": _safe_ratio_good(execution_quality.get("weight_drift_l1"), max_weight_drift_l1),
        "dropped_weight": _safe_ratio_good(execution_quality.get("dropped_theoretical_weight"), max_dropped_weight),
    }

    risk_component = float(np.mean([components["ruin"], components["max_drawdown"], components["cvar_95"]]))
    stability_component = float(np.mean([
        components["stability_energy"],
        components["path_mdd_mean"],
        components["cdar_95"],
        components["p_dd_breach"],
        components["underwater_mean"],
        components["ttr_mean_days"],
    ]))
    execution_component = float(np.mean([
        components["deployment"],
        components["cash"],
        components["weight_drift"],
        components["dropped_weight"],
    ]))

    weights = {"goal_probability": 0.30, "risk": 0.30, "stability": 0.30, "execution": 0.10}
    health_score = 100.0 * (
        weights["goal_probability"] * components["goal_probability"]
        + weights["risk"] * risk_component
        + weights["stability"] * stability_component
        + weights["execution"] * execution_component
    )

    if health_score >= 80:
        grade = "A"
    elif health_score >= 70:
        grade = "B"
    elif health_score >= 60:
        grade = "C"
    elif health_score >= 50:
        grade = "D"
    else:
        grade = "F"

    return {
        "schema_version": "portfolio_health_score_v1",
        "health_score": float(health_score),
        "health_grade": grade,
        "raw_optimizer_score": _metric(eval_dict, "score"),
        "components": {
            "goal_probability": float(components["goal_probability"]),
            "risk": risk_component,
            "stability": stability_component,
            "execution": execution_component,
        },
        "component_details": {k: float(v) for k, v in components.items()},
        "weights": {k: float(v) for k, v in weights.items()},
        "note": "quarantine uses health_score for degradation/entry rules; raw optimizer score is retained only for diagnostics.",
    }


def _baseline_health_score(eval_dict: dict) -> float | None:
    if not isinstance(eval_dict, dict):
        return None
    v = _safe_float(eval_dict.get("health_score"))
    if v is not None:
        return v
    h = eval_dict.get("health")
    if isinstance(h, dict):
        return _safe_float(h.get("health_score"))
    return None

HEALTH_CONSISTENCY_METRIC_KEYS = [
    "ann_return",
    "ann_vol",
    "ann_vol_lw",
    "sharpe",
    "sortino",
    "max_drawdown",
    "var_95",
    "cvar_95",
    "ruin_prob_1y",
    "p_hit_goal_1_1y",
    "p_hit_goal_2_1y",
    "p_hit_goal_3_1y",
    "ending_equity_p5",
    "ending_equity_p25",
    "ending_equity_p50",
    "ending_equity_p75",
    "ending_equity_p95",
    "score",
    "stability_energy",
    "path_mdd_mean",
    "cdar_95",
    "p_dd_breach",
    "underwater_mean",
    "ttr_mean_days",
]


def _metric_value(d: dict, key: str) -> float | None:
    if not isinstance(d, dict):
        return None
    if key not in d:
        return None
    return _safe_float(d.get(key))


def _metric_delta(today: float | None, baseline: float | None) -> float | None:
    if today is None or baseline is None:
        return None
    return float(today - baseline)


def _build_metric_consistency_diagnostic(
    *,
    baseline_eval: dict,
    eval_today: dict,
    health_today: float | None,
    execution_quality_today: dict,
    candidate_context: dict,
    baseline_execution_quality: dict,
    baseline_weights: dict,
    shares: dict[str, float],
    prices_usd: pd.Series,
    age_days: int,
) -> dict:
    baseline_health = _baseline_health_score(baseline_eval)

    metric_comparison: dict[str, dict] = {}
    missing_today: list[str] = []
    missing_baseline: list[str] = []

    for key in HEALTH_CONSISTENCY_METRIC_KEYS:
        b = _metric_value(baseline_eval, key)
        t = _metric_value(eval_today, key)

        if b is None:
            missing_baseline.append(key)
        if t is None:
            missing_today.append(key)

        metric_comparison[key] = {
            "baseline": b,
            "today": t,
            "delta": _metric_delta(t, b),
        }

    # Compare reconstructed current weights vs persisted search weights.
    current_weights = _signed_gross_weights_from_shares(
        shares=shares,
        prices_usd=prices_usd,
    )

    baseline_weights_n = {}
    if isinstance(baseline_weights, dict):
        for k, v in baseline_weights.items():
            vf = _safe_float(v)
            if vf is not None:
                baseline_weights_n[_norm_key(k)] = float(vf)

    weight_l1_vs_search = (
        _l1_drift(current_weights, baseline_weights_n)
        if current_weights and baseline_weights_n
        else None
    )
    weight_cosine_vs_search = (
        _cosine_sim(current_weights, baseline_weights_n)
        if current_weights and baseline_weights_n
        else None
    )

    baseline_target_notional = _safe_float(candidate_context.get("target_notional"))
    today_gross = _safe_float(execution_quality_today.get("executable_gross_notional"))

    baseline_gross = None
    if isinstance(baseline_execution_quality, dict):
        baseline_gross = _safe_float(baseline_execution_quality.get("executable_gross_notional"))

    return {
        "schema_version": "portfolio_metric_consistency_v1",
        "age_days": int(age_days),
        "baseline_source": "portfolio_search.final_executable",
        "today_source": "quarantine.report_engine.build_portfolio_report",
        "baseline_metric_engine": (
            candidate_context.get("metric_engine")
            if isinstance(candidate_context, dict)
            else None
        ),
        "baseline_return_basis": (
            candidate_context.get("return_basis")
            if isinstance(candidate_context, dict)
            else None
        ),
        "baseline_mc_basis": (
            candidate_context.get("mc_basis")
            if isinstance(candidate_context, dict)
            else None
        ),
        "baseline_health_score": baseline_health,
        "quarantine_health_score": health_today,
        "health_delta": _metric_delta(health_today, baseline_health),
        "missing_today_metrics": missing_today,
        "missing_baseline_metrics": missing_baseline,
        "metric_comparison": metric_comparison,
        "execution_comparison": {
            "baseline_target_notional": baseline_target_notional,
            "baseline_executable_gross_notional": baseline_gross,
            "today_executable_gross_notional": today_gross,
            "gross_notional_delta": _metric_delta(today_gross, baseline_gross),
            "baseline_cash_weight": (
                _safe_float(baseline_execution_quality.get("cash_weight"))
                if isinstance(baseline_execution_quality, dict)
                else None
            ),
            "today_cash_weight": _safe_float(execution_quality_today.get("cash_weight")),
            "baseline_deployment_ratio": (
                _safe_float(baseline_execution_quality.get("deployment_ratio"))
                if isinstance(baseline_execution_quality, dict)
                else None
            ),
            "today_deployment_ratio": _safe_float(execution_quality_today.get("deployment_ratio")),
        },
        "weight_comparison": {
            "baseline_weight_count": len(baseline_weights_n),
            "today_weight_count": len(current_weights),
            "l1_vs_search_weights": weight_l1_vs_search,
            "cosine_vs_search_weights": weight_cosine_vs_search,
            "baseline_weights": baseline_weights_n,
            "today_reconstructed_weights": current_weights,
        },
        "decision_note": (
            "For age_days=0, quarantine should not reject an accepted portfolio-search "
            "candidate solely because quarantine recomputation differs from the persisted "
            "portfolio-search baseline. The mismatch must be diagnosed first."
        ),
    }

def _s3_get_json_or_none(s3, *, bucket: str, key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise

    try:
        out = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None

    return out if isinstance(out, dict) else None


# ----------------------------
# Candidate discovery
# ----------------------------
def _discover_candidates_from_portfolio_runs(
    *,
    s3,
    bucket: str,
    root_prefix: str,
    as_of_ts: pd.Timestamp,
    lookback_days: int = 10,
) -> list[dict]:
    out: list[dict] = []
    start_ts = (as_of_ts - pd.Timedelta(days=int(lookback_days))).normalize()
    days = pd.date_range(start_ts, as_of_ts, freq="D")

    for d in days:
        dt_str = d.strftime("%Y-%m-%d")
        prefix = _dt_prefix(root_prefix, PORTFOLIO_RUNS_TABLE, dt_str)
        keys = [k for k in _s3_list_keys(s3, bucket=bucket, prefix=prefix) if k.lower().endswith(".json")]

        for k in sorted(keys):
            try:
                payload = s3_get_json(s3, bucket=bucket, key=k) or {}
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            run_id = str(payload.get("run_id") or "").strip()
            if not run_id:
                run_id = (k.split("/")[-1] or "").replace(".json", "").strip()
                if not run_id:
                    continue

            outputs = payload.get("outputs") or {}
            if not isinstance(outputs, dict):
                continue
            inputs = payload.get("inputs") or {}
            if not isinstance(inputs, dict):
                inputs = {}

            candidate_context = outputs.get("candidate_context") or {}
            if not isinstance(candidate_context, dict):
                candidate_context = {}

            # Fallback for older search runs that do not yet have outputs.candidate_context
            candidate_context = {
                "equity0": candidate_context.get("equity0", inputs.get("equity0")),
                "target_leverage": candidate_context.get("target_leverage", inputs.get("target_leverage")),
                "target_notional": candidate_context.get("target_notional", inputs.get("target_notional")),
                "goals": candidate_context.get("goals", inputs.get("goals")),
                "main_goal": candidate_context.get("main_goal", inputs.get("main_goal")),
                "score_config": candidate_context.get("score_config", inputs.get("score_config")),
                "weight_mode": candidate_context.get("weight_mode", "long_short"),
                "metric_engine": candidate_context.get("metric_engine", "optimizer_engine.evaluate_portfolio"),
                "return_basis": candidate_context.get("return_basis", "gross_notional_signed_weights"),
                "mc_basis": candidate_context.get("mc_basis", "fixed_gross_notional_on_equity"),
            }

            final_exec = outputs.get("final_executable") or {}
            if not isinstance(final_exec, dict):
                final_exec = {}

            # New Milestone 11+ payloads can explicitly reject a final executable
            # candidate. Quarantine should only track accepted final executable
            # portfolios; rejected search outputs are not valid baselines.
            final_status = str(final_exec.get("status") or "").strip().lower()
            if final_status and final_status != "accepted":
                continue

            baseline_search_eval: dict = {}
            baseline_weights: dict = {}
            baseline_execution_quality: dict = {}

            final_metrics = final_exec.get("metrics") if isinstance(final_exec, dict) else None
            if isinstance(final_metrics, dict) and final_metrics:
                baseline_search_eval = dict(final_metrics)
                baseline_search_eval["health_score"] = final_exec.get("health_score")
                baseline_search_eval["health_grade"] = final_exec.get("health_grade")
                baseline_search_eval["health"] = final_exec.get("health") or {}
                baseline_search_eval["raw_optimizer_score"] = baseline_search_eval.get("score")
                baseline_search_eval["score_basis"] = "health_score"
                baseline_weights = final_exec.get("weights") if isinstance(final_exec.get("weights"), dict) else {}
                baseline_execution_quality = (
                    final_exec.get("execution_quality")
                    if isinstance(final_exec.get("execution_quality"), dict)
                    else {}
                )
            else:
                # Legacy fallback for older search runs. New runs should come
                # through outputs.final_executable.
                baseline_search_eval = outputs.get("best_refined") or outputs.get("best_refined_theoretical") or {}
                if not isinstance(baseline_search_eval, dict):
                    baseline_search_eval = {}
                baseline_search_eval.setdefault("score_basis", "legacy_optimizer_score")

            disc = outputs.get("discrete_allocation") or {}
            if not isinstance(disc, dict):
                continue

            shares = disc.get("shares")
            if not isinstance(shares, dict) or not shares:
                continue

            shares_n: dict[str, float] = {}
            for t, q in shares.items():
                tt = _norm_key(t)
                qf = _safe_float(q)
                if tt and qf is not None and abs(qf) > 0:
                    shares_n[tt] = float(qf)

            if len(shares_n) < 2:
                continue

            out.append(
                {
                    "candidate_id": run_id,
                    "run_key": k,
                    "shares": shares_n,
                    "candidate_context": candidate_context,
                    "baseline_search_eval": baseline_search_eval,
                    "baseline_weights": baseline_weights,
                    "baseline_execution_quality": baseline_execution_quality,
                    "source": {
                        "table": PORTFOLIO_RUNS_TABLE,
                        "run_key": k,
                        "run_id": run_id,
                        "run_as_of": payload.get("as_of"),
                        "portfolio_output": (
                            "final_executable"
                            if isinstance(final_metrics, dict) and final_metrics
                            else "legacy_best_refined"
                        ),
                        "final_executable_status": (
                            final_status
                            if final_status
                            else None
                        ),
                    },
                }
            )

    dedup: dict[str, dict] = {}
    for rec in out:
        cid = str(rec["candidate_id"])
        if cid not in dedup or str(rec.get("run_key", "")) > str(dedup[cid].get("run_key", "")):
            dedup[cid] = rec

    return list(dedup.values())


def _discover_pending_candidates_from_state(
    *,
    s3,
    bucket: str,
    root_prefix: str,
) -> list[dict]:
    prefix = f"{root_prefix.strip('/')}/{QUAR_CAND_TABLE.strip('/')}/candidate_id="
    keys = _s3_list_keys(s3, bucket=bucket, prefix=prefix)
    latest_keys = [k for k in keys if k.endswith("/latest.json")]

    out: list[dict] = []
    for k in latest_keys:
        cand_state = _s3_get_json_or_none(s3, bucket=bucket, key=k) or {}
        if not isinstance(cand_state, dict):
            continue

        cid = str(cand_state.get("candidate_id") or "").strip()
        if not cid:
            try:
                cid = k.split("candidate_id=")[-1].split("/")[0].strip()
            except Exception:
                cid = ""

        if not cid:
            continue

        q = cand_state.get("quarantine") or {}
        if not isinstance(q, dict):
            q = {}

        status = str(q.get("status") or "PENDING").upper()
        if status in ("APPROVED", "REJECTED", "EXPIRED"):
            continue

        shares = cand_state.get("shares") or {}
        if not isinstance(shares, dict) or len(shares) < 2:
            continue

        shares_n: dict[str, float] = {}
        for t, qv in shares.items():
            tt = _norm_key(t)
            qf = _safe_float(qv)
            if tt and qf is not None and abs(qf) > 0:
                shares_n[tt] = float(qf)

        if len(shares_n) < 2:
            continue

        out.append(
            {
                "candidate_id": cid,
                "run_key": None,
                "shares": shares_n,
                "source": cand_state.get("source") or {"table": QUAR_CAND_TABLE, "candidate_state_key": k},
                "from_state_key": k,
            }
        )

    return out


# ----------------------------
# Metrics
# ----------------------------
def _signed_gross_weights_from_shares(*, shares: dict[str, float], prices_usd: pd.Series) -> dict[str, float]:
    px = pd.to_numeric(prices_usd, errors="coerce").replace([np.inf, -np.inf], np.nan)

    exp: dict[str, float] = {}
    gross = 0.0

    for t, q in shares.items():
        if t not in px.index:
            continue

        p = _safe_float(px.loc[t])
        if p is None or p <= 0:
            continue

        e = float(p) * float(q)
        exp[t] = e
        gross += abs(e)

    if gross <= 0:
        return {}

    return {t: float(exp[t] / gross) for t in exp}


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float | None:
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return None

    va = np.array([float(a.get(k, 0.0)) for k in keys], dtype=np.float64)
    vb = np.array([float(b.get(k, 0.0)) for k in keys], dtype=np.float64)

    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))

    if na <= 0 or nb <= 0:
        return None

    return float(np.dot(va, vb) / (na * nb))


def _l1_drift(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a.keys()) | set(b.keys())
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def _severity_rules(
    *,
    health_today: float | None,
    health_ref: float | None,
    decay_rate: float | None,
    metric_flags: list[str],
    age_days: int,
    consecutive_amber: int,
    consecutive_red: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if health_today is None or health_ref is None:
        return "AMBER", ["missing_health_ref_or_today"]

    health_decay = float(health_today - health_ref)

    # Health score is 0-100, so these are percentage-point drops.
    if health_decay <= -15.0:
        reasons.append("health_drop_gt_15pts")
        return "RED", reasons + metric_flags

    if decay_rate is not None and decay_rate <= -1.0 and age_days >= 5:
        reasons.append("health_decay_rate_lt_-1pt_per_day")
        return "RED", reasons + metric_flags

    if any(f.startswith("ruin") or f.startswith("mdd") for f in metric_flags):
        reasons.append("risk_metric_gap")
        return "AMBER", reasons + metric_flags

    if health_decay <= -8.0:
        reasons.append("health_drop_gt_8pts")
        return "AMBER", reasons + metric_flags

    if decay_rate is not None and decay_rate <= -0.5:
        reasons.append("health_decay_rate_lt_-0.5pt_per_day")
        return "AMBER", reasons + metric_flags

    if consecutive_red >= 2:
        return "RED", ["red_streak"] + metric_flags

    if consecutive_amber >= 3:
        return "AMBER", ["amber_streak"] + metric_flags

    return "GREEN", metric_flags


def _print_quarantine_candidate_report(cid: str, rec: dict) -> None:
    d = rec.get("degradation") or {}
    sev = d.get("severity")
    age = d.get("age_days")
    s_today = d.get("health_today")
    s_ref = d.get("health_ref")
    decay = d.get("health_decay")
    rate = d.get("health_decay_rate")
    reg_ch = d.get("regime_changed")
    reasons = d.get("reasons") or []

    print("\n" + "─" * 60)
    print(f"Quarantine Candidate: {cid}")
    print("─" * 60)
    print(f"Status: {rec.get('status')} | Severity: {sev} | Age: {age}d")

    if s_today is not None and s_ref is not None and decay is not None and rate is not None:
        print(f"Health: today={s_today:.1f}  ref={s_ref:.1f}  decay={decay:+.1f}  rate={rate:+.2f}/day")
    else:
        print(f"Health: today={s_today} ref={s_ref} decay={decay} rate={rate}")

    print(f"Regime: {d.get('baseline_regime')} -> {d.get('current_regime')}  changed={reg_ch}")

    actuarial = rec.get("actuarial_diagnostics") or {}
    if isinstance(actuarial, dict) and actuarial:
        print(
            "Actuarial: "
            f"verdict={str(actuarial.get('verdict', actuarial.get('status', 'n/a'))).upper()} "
            f"grade={actuarial.get('risk_grade', 'n/a')} "
            f"flags={len(actuarial.get('risk_flags') or [])}"
        )

    print("Reasons:", ", ".join(reasons) if reasons else "none")


# ----------------------------
# Main execution
# ----------------------------
def run_quarantine_analysis_asof(
    *,
    as_of: str,
    bucket: str = DEFAULT_ENGINE_BUCKET,
    region: str = DEFAULT_ENGINE_REGION,
    engine_root: str = DEFAULT_ENGINE_ROOT_PREFIX,
    market_root: str = "market",
    backtest_run_id: str | None = None,
    write_outputs: bool = True,
    update_latest: bool = True,
    min_quarantine_days: int = 5,
    approve_requires_green_days: int = 5,
    ttl_days: int = 12,
    lookback_days: int = 10,
    min_entry_health_score: float = 60.0,
    min_entry_score: float | None = None,
    max_cash_weight: float = 0.05,
    min_deployment_ratio: float = 0.95,
    max_executable_mdd: float = 0.40,
    max_executable_cdar_95: float = 0.60,
    max_stability_energy: float = 2.00,
    max_dropped_weight: float = 0.04,
    max_weight_drift_l1: float = 0.15,
    actuarial_max_allowed_leverage: float = 2.0,
    actuarial_n_paths: int = 5_000,
) -> dict:
    root_prefix = _resolve_root_prefix(engine_root=engine_root, backtest_run_id=backtest_run_id)
    market_root = str(market_root).strip("/")
    mode = "backtest" if backtest_run_id else "live"

    as_of_ts = pd.Timestamp(as_of).tz_localize(None).normalize()
    as_of_date = as_of_ts.strftime("%Y-%m-%d")
    run_dt = pd.Timestamp(dt.date.today()).normalize() if mode == "live" else as_of_ts

    s3 = s3_init(region)
    try:
        _market = MarketStore(bucket=bucket, region=region, base_prefix=market_root)
    except TypeError:
        _market = MarketStore(bucket=bucket, region=region)

    raw_score_cfg = s3_load_latest_json(
        s3,
        bucket=bucket,
        root_prefix=root_prefix,
        table="configs/score_config",
    )
    if not raw_score_cfg:
        raise RuntimeError(f"Missing S3 latest score_config under s3://{bucket}/{root_prefix}/configs/score_config/latest.json")

    score_cfg = ScoreConfig(**raw_score_cfg)

    recent = _discover_candidates_from_portfolio_runs(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
        as_of_ts=as_of_ts,
        lookback_days=int(lookback_days),
    )

    pending_from_state = _discover_pending_candidates_from_state(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
    )

    union: dict[str, dict] = {}
    for rec in pending_from_state:
        cid = str(rec.get("candidate_id") or "").strip()
        if cid:
            union[cid] = rec

    for rec in recent:
        cid = str(rec.get("candidate_id") or "").strip()
        if cid:
            union[cid] = rec

    discovered = list(union.values())

    if not discovered:
        out = {
            "as_of": as_of_date,
            "status": "no_candidates",
            "approved": [],
            "pending": [],
            "rejected": [],
            "expired": [],
            "counts": {"approved": 0, "pending": 0, "rejected": 0, "expired": 0},
            "meta": {
                "mode": mode,
                "bucket": bucket,
                "region": region,
                "root_prefix": root_prefix,
                "lookback_days": int(lookback_days),
                "ttl_days": int(ttl_days),
                "skipped_resolution": skipped_resolution[:50] if 'skipped_resolution' in locals() else [],
            },
        }

        if write_outputs:
            s3_write_json_event(
                s3,
                bucket=bucket,
                root_prefix=root_prefix,
                table=QUAR_SUMMARY_TABLE,
                dt=run_dt,
                filename="summary.json",
                payload=out,
                update_latest=update_latest,
            )

        return out

    market_hmm = s3_load_latest_json(
        s3,
        bucket=bucket,
        root_prefix=str(engine_root).strip("/"),
        table="regimes/market_hmm",
    ) or {}

    cur_regime_label = str(market_hmm.get("label_commit") or market_hmm.get("label") or "UNKNOWN")

    asset_maps = _load_active_universe_resolution_maps()

    resolved: list[dict] = []
    asset_ids_union: set[str] = set()
    skipped_resolution: list[dict] = []

    for rec in discovered:
        cid = str(rec.get("candidate_id") or "").strip()
        shares = rec.get("shares") or {}

        if not cid or not isinstance(shares, dict) or len(shares) < 2:
            continue

        shares_raw = {_norm_key(t): float(q) for t, q in shares.items() if _norm_key(t)}
        shares_n, resolution_meta = _canonicalize_shares_to_asset_id(shares_raw, maps=asset_maps)
        if len(shares_n) < 2:
            skipped_resolution.append(
                {
                    "candidate_id": cid,
                    "reason": "could_not_resolve_enough_asset_ids",
                    "resolution": resolution_meta,
                    "source": rec.get("source") or {},
                }
            )
            continue

        asset_ids_union |= set(shares_n.keys())

        baseline_weights = rec.get("baseline_weights") or {}
        if isinstance(baseline_weights, dict):
            baseline_weights = _canonicalize_weight_dict_to_asset_id(baseline_weights, maps=asset_maps)
        else:
            baseline_weights = {}

        resolved.append(
            {
                "candidate_id": cid,
                "shares": shares_n,
                "asset_resolution": resolution_meta,
                "source": rec.get("source") or {},
                "candidate_context": rec.get("candidate_context") or {},
                "baseline_search_eval": rec.get("baseline_search_eval") or {},
                "baseline_weights": baseline_weights,
                "baseline_execution_quality": rec.get("baseline_execution_quality") or {},
            }
        )

    if not resolved:
        out = {
            "as_of": as_of_date,
            "status": "no_valid_candidates",
            "approved": [],
            "pending": [],
            "rejected": [],
            "expired": [],
            "counts": {"approved": 0, "pending": 0, "rejected": 0, "expired": 0},
            "meta": {
                "mode": mode,
                "bucket": bucket,
                "region": region,
                "root_prefix": root_prefix,
                "lookback_days": int(lookback_days),
                "ttl_days": int(ttl_days),
                "skipped_resolution": skipped_resolution[:50] if 'skipped_resolution' in locals() else [],
            },
        }

        if write_outputs:
            s3_write_json_event(
                s3,
                bucket=bucket,
                root_prefix=root_prefix,
                table=QUAR_SUMMARY_TABLE,
                dt=run_dt,
                filename="summary.json",
                payload=out,
                update_latest=update_latest,
            )

        return out

    start_history = "2015-01-01"

    closes_all = _load_closes_usd_from_ohlcv_asset_ids(
        asset_ids=sorted(asset_ids_union),
        start=start_history,
        end=as_of_date,
        s3_bucket=bucket,
        s3_root_prefix=f"{market_root}/ohlcv_usd/v1",
        s3_region=region,
    )

    if skipped_resolution:
        print(
            f"[quarantine][warn] skipped_unresolved_candidates={len(skipped_resolution)} "
            f"sample={skipped_resolution[:3]}"
        )

    prices_usd = pd.to_numeric(closes_all.iloc[-1], errors="coerce").replace([np.inf, -np.inf], np.nan)

    approved: list[str] = []
    pending: list[str] = []
    rejected: list[str] = []
    expired: list[str] = []

    for item in resolved:
        cid = item["candidate_id"]
        shares = item["shares"]

        cand_state_key = _candidate_latest_key(root_prefix, QUAR_CAND_TABLE, cid)
        cand_state = _s3_get_json_or_none(s3, bucket=bucket, key=cand_state_key) or {}

        q = cand_state.get("quarantine") or {}
        if not isinstance(q, dict):
            q = {}

        status_existing = str(q.get("status") or "").upper()
        if status_existing in ("APPROVED", "REJECTED", "EXPIRED"):
            continue

        is_new = not isinstance(q.get("baseline_eval"), dict)

        cand_state.setdefault("candidate_id", cid)
        cand_state.setdefault("source", item.get("source") or {})
        cand_state["shares"] = {_norm_key(t): float(qv) for t, qv in shares.items() if _norm_key(t)}

        candidate_context = cand_state.get("candidate_context")
        if not isinstance(candidate_context, dict) or not candidate_context:
            candidate_context = item.get("candidate_context") or {}
            if not isinstance(candidate_context, dict):
                candidate_context = {}

        baseline_search_eval = cand_state.get("baseline_search_eval")
        if not isinstance(baseline_search_eval, dict) or not baseline_search_eval:
            baseline_search_eval = item.get("baseline_search_eval") or {}
            if not isinstance(baseline_search_eval, dict):
                baseline_search_eval = {}

        cand_state["candidate_context"] = candidate_context
        cand_state["baseline_search_eval"] = baseline_search_eval
        if item.get("baseline_weights") and not isinstance(cand_state.get("baseline_weights"), dict):
            cand_state["baseline_weights"] = item.get("baseline_weights")
        elif item.get("baseline_weights"):
            cand_state["baseline_weights"] = item.get("baseline_weights")
        if item.get("baseline_execution_quality") and not isinstance(cand_state.get("baseline_execution_quality"), dict):
            cand_state["baseline_execution_quality"] = item.get("baseline_execution_quality")
        elif item.get("baseline_execution_quality"):
            cand_state["baseline_execution_quality"] = item.get("baseline_execution_quality")

        start_as_of = str(q.get("start_as_of") or "").strip() or None
        if start_as_of is None:
            start_as_of = str(cand_state.get("last_seen_as_of") or "").strip() or None
        if start_as_of is None:
            start_as_of = as_of_date

        age_days = int((as_of_ts - pd.Timestamp(start_as_of).tz_localize(None).normalize()).days)
        age_days = max(0, age_days)

        # ---------------------------------------------------------
        # Candidate capital context
        # ---------------------------------------------------------
        candidate_context = cand_state.get("candidate_context") or {}
        if not isinstance(candidate_context, dict):
            candidate_context = {}

        equity_ref = _safe_float(cand_state.get("equity_ref"))
        if equity_ref is None:
            equity_ref = _safe_float(candidate_context.get("equity0"))

        gross_tmp = 0.0
        for t, qv in shares.items():
            if t not in prices_usd.index:
                continue

            p = _safe_float(prices_usd.loc[t])
            if p is None or p <= 0:
                continue

            gross_tmp += abs(float(p) * float(qv))

        # Fallback only. Do NOT assume hardcoded 5x unless nothing else exists.
        if equity_ref is None:
            target_leverage_ctx = _safe_float(candidate_context.get("target_leverage"))
            if target_leverage_ctx is not None and target_leverage_ctx > 0 and gross_tmp > 0:
                equity_ref = float(gross_tmp / target_leverage_ctx)

        if equity_ref is None:
            equity_ref = 10000.0

        goals = cand_state.get("goals") or candidate_context.get("goals") or [7500.0, 10000.0, 12500.0]
        if not isinstance(goals, list) or len(goals) != 3:
            goals = [7500.0, 10000.0, 12500.0]
        goals = [float(x) for x in goals]

        main_goal = (
            _safe_float(cand_state.get("main_goal"))
            or _safe_float(candidate_context.get("main_goal"))
            or 10000.0
        )

        score_cfg_for_candidate = score_cfg
        score_cfg_payload = cand_state.get("score_config") or candidate_context.get("score_config")
        if isinstance(score_cfg_payload, dict):
            try:
                score_cfg_for_candidate = ScoreConfig(**score_cfg_payload)
            except Exception:
                score_cfg_for_candidate = score_cfg

        actuarial_diagnostics = None
        try:
            positions = {
                t: Position(ticker=t, quantity=float(qv), entry_price=None, currency="USD")
                for t, qv in shares.items()
            }
            closes = closes_all[list(positions.keys())].copy()

            report = build_portfolio_report(
                closes=closes,
                positions=positions,
                equity=float(equity_ref),
                goals=list(goals),
                main_goal=float(main_goal),
                score_config=score_cfg_for_candidate,
                prices_usd=prices_usd,
            )
            eval_today = asdict(report.eval)

            try:
                _actuarial_report, _actuarial_text, actuarial_diagnostics = (
                    build_actuarial_diagnostic_from_portfolio_report(
                        report=report,
                        closes=closes,
                        goals=list(goals),
                        main_goal=float(main_goal),
                        score_config=score_cfg_for_candidate,
                        portfolio_id=str(cid),
                        run_id=str(cid),
                        source="quarantine_analysis",
                        terminal_title="ACTUARIAL RISK DIAGNOSTICS - QUARANTINE",
                        current_leverage=float(report.snapshot.leverage),
                        max_allowed_leverage=float(actuarial_max_allowed_leverage),
                        days=252,
                        n_paths=int(actuarial_n_paths),
                        mc_seed=86420,
                        path_source="bootstrap",
                        pca_k=5,
                        block_size=(8, 12),
                        metadata={
                            "as_of": as_of_date,
                            "candidate_id": str(cid),
                            "mode": mode,
                            "root_prefix": root_prefix,
                        },
                    )
                )
            except Exception as e:
                actuarial_diagnostics = {
                    "status": "failed",
                    "source": "quarantine_analysis",
                    "candidate_id": str(cid),
                    "error_type": type(e).__name__,
                    "error": str(e),
                }

        except Exception as e:
            eval_today = {"error": f"{type(e).__name__}: {e}"}
            actuarial_diagnostics = {
                "status": "failed",
                "source": "quarantine_analysis",
                "candidate_id": str(cid),
                "error_type": type(e).__name__,
                "error": str(e),
            }

        if isinstance(cand_state, dict):
            cand_state["actuarial_diagnostics"] = actuarial_diagnostics

        score_today = _metric(eval_today, "score")

        baseline_weights = cand_state.get("baseline_weights") or item.get("baseline_weights") or {}
        if not isinstance(baseline_weights, dict):
            baseline_weights = {}
        baseline_execution_quality = cand_state.get("baseline_execution_quality") or item.get("baseline_execution_quality") or {}
        if not isinstance(baseline_execution_quality, dict):
            baseline_execution_quality = {}

        execution_quality_today = _compute_current_execution_quality(
            shares=shares,
            prices_usd=prices_usd,
            target_notional=_safe_float(candidate_context.get("target_notional")),
            baseline_weights=baseline_weights,
            baseline_execution_quality=baseline_execution_quality,
        )

        health_today_payload = _compute_quarantine_health_score(
            eval_dict=eval_today if isinstance(eval_today, dict) else {},
            execution_quality=execution_quality_today,
            score_cfg=score_cfg_for_candidate,
            goals=goals,
            main_goal=float(main_goal),
            max_cash_weight=float(max_cash_weight),
            min_deployment_ratio=float(min_deployment_ratio),
            max_executable_mdd=float(max_executable_mdd),
            max_executable_cdar_95=float(max_executable_cdar_95),
            max_stability_energy=float(max_stability_energy),
            max_dropped_weight=float(max_dropped_weight),
            max_weight_drift_l1=float(max_weight_drift_l1),
        )
        health_today = _safe_float(health_today_payload.get("health_score"))

        baseline_entry_health = _baseline_health_score(baseline_search_eval)

        metric_consistency = _build_metric_consistency_diagnostic(
            baseline_eval=baseline_search_eval if isinstance(baseline_search_eval, dict) else {},
            eval_today=eval_today if isinstance(eval_today, dict) else {},
            health_today=health_today,
            execution_quality_today=execution_quality_today,
            candidate_context=candidate_context if isinstance(candidate_context, dict) else {},
            baseline_execution_quality=baseline_execution_quality if isinstance(baseline_execution_quality, dict) else {},
            baseline_weights=baseline_weights if isinstance(baseline_weights, dict) else {},
            shares=shares,
            prices_usd=prices_usd,
            age_days=int(age_days),
        )

        if isinstance(eval_today, dict):
            eval_today["health_score"] = health_today
            eval_today["health_grade"] = health_today_payload.get("health_grade")
            eval_today["health"] = health_today_payload
            eval_today["raw_optimizer_score"] = score_today
            eval_today["score_basis"] = "health_score"
            eval_today["execution_quality"] = execution_quality_today
            eval_today["actuarial_diagnostics"] = actuarial_diagnostics
            eval_today["metric_consistency"] = metric_consistency

        entry_health = baseline_entry_health
        entry_health_source = "portfolio_search_baseline"

        if entry_health is None:
            entry_health = health_today
            entry_health_source = "quarantine_recomputed"

        if is_new and (entry_health is None or float(entry_health) < float(min_entry_health_score)):
            status = "REJECTED"
            sev = "RED"
            reasons = [f"below_entry_health_score({entry_health}->{min_entry_health_score})"]

            q = {
                "start_as_of": as_of_date,
                "baseline_eval": (
                    dict(baseline_search_eval)
                    if isinstance(baseline_search_eval, dict)
                    else {}
                ),
                "baseline_regime_label": cur_regime_label,
                "streak_amber": 0,
                "streak_red": 0,
                "streak_green": 0,
                "status": status,
                "last_eval_as_of": as_of_date,
                "last_regime_label": cur_regime_label,
                "degradation": {
                    "as_of": as_of_date,
                    "start_as_of": as_of_date,
                    "age_days": 0,
                    "entry_health_source": entry_health_source,
                    "metric_consistency": metric_consistency,
                    "health_today": health_today,
                    "health_ref": entry_health,
                    "health_decay": (
                        None
                        if health_today is None or entry_health is None
                        else float(health_today - entry_health)
                    ),
                    "health_decay_rate": None,
                    "score_today": score_today,
                    "score_ref": None,
                    "score_decay": None,
                    "score_decay_rate": None,
                    "half_life_days": None,
                    "w_drift_l1": 0.0,
                    "cosine": None,
                    "regime_changed": False,
                    "baseline_regime": cur_regime_label,
                    "current_regime": cur_regime_label,
                    "severity": sev,
                    "reasons": reasons,
                },
            }

            cand_state["quarantine"] = q
            cand_state["last_seen_as_of"] = as_of_date

            eval_rec = {
                "as_of": as_of_date,
                "candidate_id": cid,
                "status": status,
                "severity": sev,
                "degradation": q["degradation"],
                "eval": eval_today,
                "baseline_eval": q.get("baseline_eval"),
                "actuarial_diagnostics": actuarial_diagnostics,
                "source": item.get("source") or {},
            }

            rejected.append(cid)
            _print_quarantine_candidate_report(cid=cid, rec=eval_rec)

            if write_outputs:
                latest_eval_key = _candidate_latest_key(root_prefix, QUAR_EVALS_TABLE, cid)
                _s3_put_json(s3, bucket=bucket, key=latest_eval_key, payload=eval_rec)

                s3_write_json_event(
                    s3,
                    bucket=bucket,
                    root_prefix=root_prefix,
                    table=QUAR_EVALS_TABLE,
                    dt=run_dt,
                    filename=f"eval_{cid}.json",
                    payload=eval_rec,
                    update_latest=False,
                )

                _s3_put_json(s3, bucket=bucket, key=cand_state_key, payload=cand_state)

            continue

        if "baseline_eval" not in q or not isinstance(q.get("baseline_eval"), dict):
            q["start_as_of"] = start_as_of
            if isinstance(baseline_search_eval, dict) and baseline_search_eval:
                q["baseline_eval"] = dict(baseline_search_eval)
            else:
                q["baseline_eval"] = dict(eval_today) if isinstance(eval_today, dict) else {}
            q["baseline_regime_label"] = cur_regime_label
            q["streak_amber"] = 0
            q["streak_red"] = 0
            q["streak_green"] = 0
            q["status"] = "PENDING"

        base_eval = q.get("baseline_eval") or {}
        if not isinstance(base_eval, dict):
            base_eval = {}

        health_ref = _baseline_health_score(base_eval)
        if health_ref is None:
            # Legacy fallback only. New candidates should carry health_score.
            legacy_score_ref = _metric(base_eval, "score")
            legacy_score_today = score_today
            if legacy_score_ref is not None and legacy_score_today is not None:
                health_ref = 100.0 * float(legacy_score_ref)
                health_today = 100.0 * float(legacy_score_today)
        if health_ref is None:
            health_ref = health_today

        score_ref = _metric(base_eval, "score")
        if score_ref is None:
            score_ref = score_today

        health_decay = None if (health_today is None or health_ref is None) else float(health_today - health_ref)
        decay_rate = None if health_decay is None else float(health_decay / max(1, age_days))
        score_decay = None if (score_today is None or score_ref is None) else float(score_today - score_ref)

        half_life_days = None
        if health_today is not None and health_ref is not None and decay_rate is not None and decay_rate < 0:
            target = 0.5 * float(health_ref)
            t_h = (float(target) - float(health_today)) / float(decay_rate)
            if np.isfinite(t_h) and t_h > 0:
                half_life_days = float(t_h)

        metric_flags: list[str] = []

        ruin_today = _metric(eval_today, "ruin_prob_1y", "ruin_probability", "ruin_prob")
        ruin_ref = _metric(base_eval, "ruin_prob_1y", "ruin_probability", "ruin_prob")
        if ruin_today is not None and ruin_ref is not None and ruin_today > ruin_ref + 0.01:
            metric_flags.append(f"ruin_up({ruin_ref:.3f}->{ruin_today:.3f})")

        mdd_today = _metric(eval_today, "max_drawdown", "mdd")
        mdd_ref = _metric(base_eval, "max_drawdown", "mdd")
        if mdd_today is not None and mdd_ref is not None and float(mdd_today) < float(mdd_ref) - 0.05:
            metric_flags.append(f"mdd_worse({mdd_ref:.3f}->{mdd_today:.3f})")

        ann_today = _metric(eval_today, "ann_return", "annual_return")
        ann_ref = _metric(base_eval, "ann_return", "annual_return")
        if ann_today is not None and ann_ref is not None and ann_today < ann_ref - 0.08:
            metric_flags.append(f"ann_down({ann_ref:.3f}->{ann_today:.3f})")

        w_target = _signed_gross_weights_from_shares(shares=shares, prices_usd=prices_usd)
        w_today = w_target
        w_drift_l1 = _l1_drift(w_today, w_target)
        cosine = _cosine_sim(w_today, w_target)

        base_reg = str(q.get("baseline_regime_label") or "UNKNOWN")
        regime_changed = base_reg != cur_regime_label

        sev, reasons = _severity_rules(
            health_today=health_today,
            health_ref=health_ref,
            decay_rate=decay_rate,
            metric_flags=metric_flags,
            age_days=age_days,
            consecutive_amber=int(q.get("streak_amber", 0) or 0),
            consecutive_red=int(q.get("streak_red", 0) or 0),
        )

        if regime_changed and age_days >= 3 and sev != "GREEN":
            reasons = (reasons or []) + ["regime_changed"]

        if sev == "GREEN":
            q["streak_green"] = int(q.get("streak_green", 0) or 0) + 1
            q["streak_amber"] = 0
            q["streak_red"] = 0
        elif sev == "AMBER":
            q["streak_green"] = 0
            q["streak_amber"] = int(q.get("streak_amber", 0) or 0) + 1
            q["streak_red"] = 0
        else:
            q["streak_green"] = 0
            q["streak_amber"] = 0
            q["streak_red"] = int(q.get("streak_red", 0) or 0) + 1

        status = str(q.get("status") or "PENDING").upper()
        if status not in ("PENDING", "APPROVED", "REJECTED", "EXPIRED"):
            status = "PENDING"

        health_for_status = health_today
        health_for_status_source = "quarantine_recomputed"

        # On day 0, a candidate that was just accepted by portfolio_search should not
        # be rejected only because quarantine recomputation disagrees. Use the persisted
        # search baseline for the entry gate, and store the quarantine recomputation as
        # diagnostic evidence.
        if int(age_days) == 0 and baseline_entry_health is not None:
            health_for_status = baseline_entry_health
            health_for_status_source = "portfolio_search_baseline"

        if status == "PENDING" and (
            health_for_status is None
            or float(health_for_status) < float(min_entry_health_score)
        ):
            status = "REJECTED"
            sev = "RED"
            reasons = (reasons or []) + [
                f"fell_below_entry_health_score({health_for_status}->{min_entry_health_score}; source={health_for_status_source})"
            ]

        if status == "PENDING" and age_days > int(ttl_days):
            status = "EXPIRED"

        if status == "PENDING" and sev == "RED" and int(q.get("streak_red", 0) or 0) >= 2:
            status = "REJECTED"

        if status == "PENDING":
            if age_days >= int(min_quarantine_days) and int(q.get("streak_green", 0) or 0) >= int(approve_requires_green_days):
                status = "APPROVED"

        q["status"] = status
        q["approved_for_shadow"] = bool(status == "APPROVED")
        q["last_eval_as_of"] = as_of_date
        q["last_regime_label"] = cur_regime_label
        q["degradation"] = {
            "as_of": as_of_date,
            "start_as_of": str(q.get("start_as_of")),
            "age_days": int(age_days),
            "health_today": health_today,
            "health_ref": health_ref,
            "health_decay": health_decay,
            "health_decay_rate": decay_rate,
            "score_today": score_today,
            "score_ref": score_ref,
            "score_decay": score_decay,
            "score_decay_rate": None if score_decay is None else float(score_decay / max(1, age_days)),
            "half_life_days": half_life_days,
            "w_drift_l1": float(w_drift_l1),
            "cosine": cosine,
            "regime_changed": bool(regime_changed),
            "baseline_regime": base_reg,
            "current_regime": cur_regime_label,
            "severity": sev,
            "reasons": reasons,
            "health_for_status": health_for_status,
            "health_for_status_source": health_for_status_source,
            "metric_consistency": metric_consistency,
        }

        cand_state["quarantine"] = q
        cand_state["quarantine_status"] = status
        cand_state["approved_for_shadow"] = bool(status == "APPROVED")
        cand_state["last_seen_as_of"] = as_of_date

        eval_rec = {
            "as_of": as_of_date,
            "candidate_id": cid,
            "status": status,
            "severity": sev,
            "degradation": q["degradation"],
            "eval": eval_today,
            "baseline_eval": q.get("baseline_eval"),
            "actuarial_diagnostics": actuarial_diagnostics,
            "source": item.get("source") or {},
        }

        if status == "APPROVED":
            approved.append(cid)
        elif status == "REJECTED":
            rejected.append(cid)
        elif status == "EXPIRED":
            expired.append(cid)
        else:
            pending.append(cid)

        _print_quarantine_candidate_report(cid=cid, rec=eval_rec)

        if write_outputs:
            lines: list[str] = []
            d = eval_rec.get("degradation") or {}

            lines.append(f"Quarantine Candidate: {cid}")
            lines.append(f"as_of: {as_of_date}")
            lines.append(f"status: {status} | severity: {sev} | age_days: {d.get('age_days')}")
            lines.append("")
            lines.append(f"health_today: {d.get('health_today')}")
            lines.append(f"health_ref:   {d.get('health_ref')}")
            lines.append(f"health_decay: {d.get('health_decay')}")
            lines.append(f"health_decay_rate: {d.get('health_decay_rate')}")
            lines.append("")
            lines.append(f"raw_optimizer_score_today: {d.get('score_today')}")
            lines.append(f"raw_optimizer_score_ref:   {d.get('score_ref')}")
            lines.append(f"raw_optimizer_score_decay: {d.get('score_decay')}")
            lines.append("")
            lines.append(
                f"baseline_regime: {d.get('baseline_regime')}  "
                f"current_regime: {d.get('current_regime')}  "
                f"changed={d.get('regime_changed')}"
            )
            lines.append("reasons: " + (", ".join(d.get("reasons") or []) or "none"))
            lines.append("")
            lines.append("actuarial_diagnostics:")
            if isinstance(actuarial_diagnostics, dict) and actuarial_diagnostics:
                lines.append(f"  verdict: {actuarial_diagnostics.get('verdict', actuarial_diagnostics.get('status'))}")
                lines.append(f"  risk_grade: {actuarial_diagnostics.get('risk_grade')}")
                hm = actuarial_diagnostics.get("headline_metrics") or {}
                if isinstance(hm, dict):
                    for k, v in sorted(hm.items()):
                        lines.append(f"  {k}: {v}")
                flags = actuarial_diagnostics.get("risk_flags") or []
                warnings = actuarial_diagnostics.get("warnings") or []
                lines.append("  risk_flags: " + (", ".join(flags) if flags else "none"))
                lines.append("  warnings: " + (", ".join(warnings) if warnings else "none"))
            else:
                lines.append("  none")
            lines.append("")
            lines.append("eval_today:")
            for k, v in sorted((eval_today or {}).items()):
                lines.append(f"  {k}: {v}")

            lines.append("")
            lines.append("baseline_eval:")
            base_eval2 = q.get("baseline_eval") or {}
            if isinstance(base_eval2, dict):
                for k, v in sorted(base_eval2.items()):
                    lines.append(f"  {k}: {v}")

            report_key = (
                f"{root_prefix.strip('/')}/{QUAR_REPORTS_TABLE}/"
                f"dt={run_dt.strftime('%Y-%m-%d')}/report_{cid}.txt"
            )
            _s3_put_text(s3, bucket=bucket, key=report_key, text="\n".join(lines))

            latest_eval_key = _candidate_latest_key(root_prefix, QUAR_EVALS_TABLE, cid)
            _s3_put_json(s3, bucket=bucket, key=latest_eval_key, payload=eval_rec)

            s3_write_json_event(
                s3,
                bucket=bucket,
                root_prefix=root_prefix,
                table=QUAR_EVALS_TABLE,
                dt=run_dt,
                filename=f"eval_{cid}.json",
                payload=eval_rec,
                update_latest=False,
            )

            _s3_put_json(s3, bucket=bucket, key=cand_state_key, payload=cand_state)

    summary = {
        "as_of": as_of_date,
        "status": "ok",
        "approved": sorted(set(approved)),
        "approved_for_shadow": sorted(set(approved)),
        "pending": sorted(set(pending)),
        "rejected": sorted(set(rejected)),
        "expired": sorted(set(expired)),
        "counts": {
            "approved": int(len(set(approved))),
            "pending": int(len(set(pending))),
            "rejected": int(len(set(rejected))),
            "expired": int(len(set(expired))),
        },
        "meta": {
            "mode": mode,
            "bucket": bucket,
            "region": region,
            "root_prefix": root_prefix,
            "market_regime_label": cur_regime_label,
            "n_union": int(len(discovered)),
            "n_resolved": int(len(resolved)),
            "n_skipped_resolution": int(len(skipped_resolution)) if 'skipped_resolution' in locals() else 0,
            "skipped_resolution_sample": skipped_resolution[:20] if 'skipped_resolution' in locals() else [],
            "source_table": PORTFOLIO_RUNS_TABLE,
            "lookback_days": int(lookback_days),
            "min_quarantine_days": int(min_quarantine_days),
            "approve_requires_green_days": int(approve_requires_green_days),
            "ttl_days": int(ttl_days),
            "score_basis": "health_score",
            "shadow_gate_policy": "only quarantine.status=APPROVED and approved_for_shadow=true candidates are eligible for shadow_portfolio_assessment",
            "actuarial_diagnostics": {
                "enabled": True,
                "max_allowed_leverage": float(actuarial_max_allowed_leverage),
                "n_paths": int(actuarial_n_paths),
                "decision_policy": "diagnostic_only_no_status_change",
            },
            "min_entry_health_score": float(min_entry_health_score),
            "legacy_min_entry_score": None if min_entry_score is None else float(min_entry_score),
        },
    }

    if write_outputs:
        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table=QUAR_SUMMARY_TABLE,
            dt=run_dt,
            filename="summary.json",
            payload=summary,
            update_latest=update_latest,
        )

    return summary


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Alpha Edge quarantine analysis.")

    ap.add_argument("--as-of", type=str, default=None)
    ap.add_argument("--backtest-run-id", type=str, default=None)

    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-latest", action="store_true")

    ap.add_argument("--min-days", type=int, default=5)
    ap.add_argument("--approve-green-days", type=int, default=5)
    ap.add_argument("--ttl-days", type=int, default=12)
    ap.add_argument("--lookback-days", type=int, default=10)
    ap.add_argument("--min-entry-health-score", type=float, default=60.0)
    ap.add_argument(
        "--min-entry-score",
        type=float,
        default=None,
        help="Deprecated legacy optimizer-score entry threshold. Health score is used by default.",
    )

    ap.add_argument("--actuarial-max-allowed-leverage", type=float, default=2.0)
    ap.add_argument("--actuarial-n-paths", type=int, default=5000)

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    return ap.parse_args()


def _main_impl() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)

    bucket = cfg_bucket(cfg)
    region = cfg_region(cfg)
    engine_root = cfg_engine_root(cfg)
    market_root = cfg_market_root(cfg)

    write_outputs = not bool(args.no_write)
    if write_outputs:
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    as_of = args.as_of or pd.Timestamp(dt.date.today()).strftime("%Y-%m-%d")

    out = run_quarantine_analysis_asof(
        as_of=as_of,
        bucket=bucket,
        region=region,
        engine_root=engine_root,
        market_root=market_root,
        backtest_run_id=args.backtest_run_id,
        write_outputs=write_outputs,
        update_latest=(not bool(args.no_latest)),
        min_quarantine_days=int(args.min_days),
        approve_requires_green_days=int(args.approve_green_days),
        ttl_days=int(args.ttl_days),
        lookback_days=int(args.lookback_days),
        min_entry_health_score=float(args.min_entry_health_score),
        min_entry_score=None if args.min_entry_score is None else float(args.min_entry_score),
        actuarial_max_allowed_leverage=float(args.actuarial_max_allowed_leverage),
        actuarial_n_paths=int(args.actuarial_n_paths),
    )

    print("\n=== QUARANTINE SUMMARY ===")
    print(f"env:       {cfg_env(cfg)}")
    print(f"bucket:    {bucket}")
    print(f"region:    {region}")
    print(f"root:      {engine_root}")
    print(f"market:    {market_root}")
    print(f"as_of:     {out.get('as_of')}")
    print(f"approved:  {out.get('counts', {}).get('approved', 0)}")
    print(f"pending:   {out.get('counts', {}).get('pending', 0)}")
    print(f"rejected:  {out.get('counts', {}).get('rejected', 0)}")
    print(f"expired:   {out.get('counts', {}).get('expired', 0)}")

    if out.get("approved"):
        print("approved_ids:", ", ".join(out["approved"][:20]))


# ----------------------------
# Audit/logging entrypoint wrapper
# ----------------------------
def _tier1_audit_is_dry_run(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "dry_run", False) or getattr(args, "no_write", False))


def main_with_audit() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = _tier1_audit_is_dry_run(args)

    with capture_script_run(
        cfg=cfg,
        script_name="run_quarantine_analysis.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            _main_impl()

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="quarantine_analysis",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="run_quarantine_analysis.py",
                source_mode="quarantine_analysis",
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
                entity_type="quarantine_analysis",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="run_quarantine_analysis.py",
                source_mode="quarantine_analysis",
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
