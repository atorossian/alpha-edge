# run_portfolio_search.py  (S3-only I/O)
from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from multiprocessing import freeze_support
import argparse
import datetime as dt
from dataclasses import asdict

import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge import paths
from alpha_edge.core.data_loader import (
    clean_returns_matrix,
    parse_positions_obj,
    s3_init,
    s3_load_latest_json,
    s3_load_latest_report_score,
    s3_write_json_event,
    s3_write_parquet_partition,
)
from alpha_edge.core.schemas import (
    ActuarialRiskConfig,
    CapitalAdequacyConfig,
    DrawdownBreachConfig,
    GoalConfig,
    RecoveryConfig,
    RuinConfig,
    SurvivalConfig,
)
from alpha_edge.risk.actuarial.portfolio_search_output import (
    attach_actuarial_diagnostic_to_output_payload,
    build_portfolio_search_actuarial_diagnostic_section,
    maybe_print_actuarial_diagnostic_section,
)
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import RuntimeConfig, ScoreConfig, StabilityEnergyConfig, StabilityReport
from alpha_edge.market.build_returns_wide_cache import CacheConfig, build_returns_wide_cache
from alpha_edge.market.regime_filter import RegimeFilterState
from alpha_edge.market.regime_leverage import leverage_from_hmm
from alpha_edge.portfolio.execution_engine import weights_to_discrete_shares
from alpha_edge.portfolio.optimizer_engine import (
    compute_stability_for_candidate,
    evaluate_portfolio_candidate,
    evaluate_portfolio_candidate_with_paths,
)
from alpha_edge.portfolio.portfolio_search import evolve_portfolios_ga, refine_portfolio_annealing
from alpha_edge.universe.universe import Asset


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT = "engine/v1"
DEFAULT_MARKET_ROOT = "market"

DATA_QUALITY_TABLE = "diagnostics/data_quality/v1"

DATA_QUALITY_SEVERE_FLAG_PREFIXES = (
    "ohlcv_simple_extreme_return",
    "returns_usd_simple_extreme_return",
    "ohlcv_vs_returns_usd_simple_low_corr",
    "ohlcv_vs_returns_usd_simple_large_diff",
    "non_positive_ohlcv_prices",
    "missing_ohlcv_prices",
    "diagnostic_exception",
)

DATA_QUALITY_ALLOWED_MISSING_RETURNS_WIDE_FLAGS = (
    "missing_returns_wide_asset_id",
)


def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET)).strip()


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION)).strip()


def cfg_engine_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "engine_root", DEFAULT_ENGINE_ROOT)).strip("/")


def cfg_market_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "market_root", DEFAULT_MARKET_ROOT)).strip("/")


def make_market_store(cfg: RuntimeConfig) -> MarketStore:
    """
    Runtime-aware MarketStore constructor.

    If the current MarketStore does not accept base_prefix, this falls back to the
    old constructor. In that case, MarketStore itself should eventually be made
    runtime-aware too.
    """
    try:
        return MarketStore(
            bucket=cfg_bucket(cfg),
            region=cfg_region(cfg),
            base_prefix=cfg_market_root(cfg),
        )
    except TypeError:
        return MarketStore(
            bucket=cfg_bucket(cfg),
            region=cfg_region(cfg),
        )


def returns_cache_prefix(cfg: RuntimeConfig) -> str:
    return f"{cfg_market_root(cfg)}/cache/v1"


def returns_cache_uri(cfg: RuntimeConfig, *, min_years: float) -> str:
    return (
        f"s3://{cfg_bucket(cfg)}/"
        f"{returns_cache_prefix(cfg)}/"
        f"returns_wide_min{int(float(min_years))}y.parquet"
    )


def latest_prices_snapshot_ref(cfg: RuntimeConfig) -> str:
    return f"{cfg_market_root(cfg)}/snapshots/v1/latest_prices.parquet"


def make_cache_config(cfg: RuntimeConfig, *, min_years: float) -> CacheConfig:
    """
    Build CacheConfig in a way that works with both older and newer versions of
    alpha_edge.market.build_returns_wide_cache.CacheConfig.

    Preferred shape:
      CacheConfig(bucket=..., region=..., market_root=..., min_years=...)

    Older shape:
      CacheConfig(bucket=..., cache_prefix=..., min_years=...)

    Legacy shape:
      CacheConfig(bucket=..., min_years=...)
    """
    bucket = cfg_bucket(cfg)
    region = cfg_region(cfg)
    market_root = cfg_market_root(cfg)
    cache_prefix = returns_cache_prefix(cfg)

    try:
        return CacheConfig(
            bucket=bucket,
            region=region,
            market_root=market_root,
            min_years=float(min_years),
        )
    except TypeError:
        pass

    try:
        return CacheConfig(
            bucket=bucket,
            cache_prefix=cache_prefix,
            min_years=float(min_years),
        )
    except TypeError:
        pass

    return CacheConfig(
        bucket=bucket,
        min_years=float(min_years),
    )

def s3_key_exists(s3, *, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def returns_cache_key(cfg: RuntimeConfig, *, min_years: float) -> str:
    return (
        f"{returns_cache_prefix(cfg)}/"
        f"returns_wide_min{int(float(min_years))}y.parquet"
    )

def _split_flags(value: object) -> set[str]:
    if value is None:
        return set()

    if isinstance(value, list):
        return {str(x).strip() for x in value if str(x).strip()}

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return set()

    return {part.strip() for part in text.split(",") if part.strip()}


def _has_severe_data_quality_flag(flags: set[str]) -> bool:
    for flag in flags:
        for prefix in DATA_QUALITY_SEVERE_FLAG_PREFIXES:
            if flag.startswith(prefix):
                return True
    return False


def _load_latest_market_data_quality_summary(
    *,
    bucket: str,
    market_root: str,
) -> pd.DataFrame:
    """
    Load latest market data-quality summary from S3.

    Preferred path:
      s3://<bucket>/<market_root>/diagnostics/data_quality/v1/latest/market_data_integrity_summary.parquet

    Fallback:
      discover the newest dated partition:
      s3://<bucket>/<market_root>/diagnostics/data_quality/v1/dt=YYYY-MM-DD/market_data_integrity_summary.parquet
    """
    market_root = str(market_root).strip("/")
    table_prefix = f"{market_root}/{DATA_QUALITY_TABLE}".strip("/")

    latest_path = (
        f"s3://{bucket}/{table_prefix}/latest/market_data_integrity_summary.parquet"
    )

    attempted_paths = [latest_path]

    try:
        df = pd.read_parquet(latest_path, engine="pyarrow")
        source_path = latest_path
    except Exception:
        import boto3

        s3 = boto3.client("s3")

        prefix = f"{table_prefix}/"
        keys: list[str] = []
        token = None

        while True:
            kwargs = {
                "Bucket": bucket,
                "Prefix": prefix,
                "MaxKeys": 1000,
            }
            if token:
                kwargs["ContinuationToken"] = token

            resp = s3.list_objects_v2(**kwargs)

            for obj in resp.get("Contents", []) or []:
                key = str(obj.get("Key", ""))
                if (
                    "/dt=" in key
                    and key.endswith("market_data_integrity_summary.parquet")
                ):
                    keys.append(key)

            if not resp.get("IsTruncated"):
                break

            token = resp.get("NextContinuationToken")

        if not keys:
            raise RuntimeError(
                "Could not load market data-quality summary. "
                "Run diagnose_market_data_integrity.py with --confirm-prod-write first. "
                f"Tried latest path: {latest_path}. "
                f"No dated summary parquet found under s3://{bucket}/{prefix}"
            )

        def _extract_dt_from_key(key: str) -> pd.Timestamp:
            try:
                value = key.split("/dt=", 1)[1].split("/", 1)[0]
                return pd.Timestamp(value)
            except Exception:
                return pd.Timestamp.min

        best_key = max(keys, key=_extract_dt_from_key)
        fallback_path = f"s3://{bucket}/{best_key}"
        attempted_paths.append(fallback_path)

        df = pd.read_parquet(fallback_path, engine="pyarrow")
        source_path = fallback_path

    if df is None or df.empty:
        raise RuntimeError(
            "Market data-quality summary is empty. "
            f"Attempted paths: {attempted_paths}"
        )

    if "asset_id" not in df.columns:
        raise RuntimeError(
            "Market data-quality summary missing 'asset_id'. "
            f"Loaded from: {source_path}. "
            f"Columns={list(df.columns)}"
        )

    out = df.copy()
    out["asset_id"] = out["asset_id"].astype(str).str.strip()
    out = out[out["asset_id"] != ""].copy()

    if "status" not in out.columns:
        out["status"] = "UNKNOWN"

    if "flags" not in out.columns:
        out["flags"] = ""

    print(f"[data_quality_gate] loaded_summary={source_path} rows={len(out)}")

    return out


def _eligible_asset_ids_from_market_data_quality(
    dq: pd.DataFrame,
    *,
    allow_warn: bool = True,
) -> tuple[set[str], dict]:
    """
    Build an asset_id allowlist from market-data diagnostics.

    Rules:
      - PASS is allowed.
      - WARN is allowed only if allow_warn=True and has no severe flags.
      - FAIL is normally rejected.
      - missing_returns_wide_asset_id is not allowed for portfolio search because
        portfolio search requires returns_wide presence.
      - raw-log returns_wide is rejected.
    """
    eligible: set[str] = set()

    counters = {
        "rows": int(len(dq)),
        "eligible": 0,
        "rejected_status": 0,
        "rejected_severe_flags": 0,
        "rejected_missing_returns_wide": 0,
        "rejected_raw_log": 0,
    }

    rejected_sample: list[dict] = []

    for _, row in dq.iterrows():
        asset_id = str(row.get("asset_id", "")).strip()
        if not asset_id:
            continue

        status = str(row.get("status", "UNKNOWN")).upper().strip()
        flags = _split_flags(row.get("flags"))

        returns_wide_exists = row.get("returns_wide_exists", True)
        try:
            returns_wide_exists_bool = bool(returns_wide_exists)
        except Exception:
            returns_wide_exists_bool = False

        rw_basis = str(row.get("returns_wide_looks_like", "")).strip()

        reject_reason = None

        if status == "PASS":
            pass
        elif status == "WARN" and allow_warn:
            pass
        else:
            reject_reason = f"status={status}"
            counters["rejected_status"] += 1

        if reject_reason is None and not returns_wide_exists_bool:
            reject_reason = "missing_returns_wide"
            counters["rejected_missing_returns_wide"] += 1

        if reject_reason is None and "missing_returns_wide_asset_id" in flags:
            reject_reason = "missing_returns_wide_asset_id"
            counters["rejected_missing_returns_wide"] += 1

        if reject_reason is None and rw_basis == "raw_log":
            reject_reason = "returns_wide_raw_log"
            counters["rejected_raw_log"] += 1

        if reject_reason is None and _has_severe_data_quality_flag(flags):
            reject_reason = "severe_data_quality_flags"
            counters["rejected_severe_flags"] += 1

        if reject_reason is None:
            eligible.add(asset_id)
        else:
            if len(rejected_sample) < 50:
                rejected_sample.append(
                    {
                        "asset_id": asset_id,
                        "display_symbol": row.get("display_symbol"),
                        "ticker": row.get("ticker"),
                        "yahoo_ticker": row.get("yahoo_ticker"),
                        "status": status,
                        "flags": sorted(flags),
                        "reject_reason": reject_reason,
                    }
                )

    counters["eligible"] = int(len(eligible))
    counters["rejected_sample"] = rejected_sample

    return eligible, counters


def _fmt_float(x, nd: int = 2, default: str = "n/a") -> str:
    try:
        xf = float(x)
    except Exception:
        return default
    if not np.isfinite(xf):
        return default
    return f"{xf:.{nd}f}"


def _fmt_money(x, default: str = "n/a") -> str:
    try:
        xf = float(x)
    except Exception:
        return default
    if not np.isfinite(xf):
        return default
    return f"{xf:,.2f}"


def _fmt_pct(x, default: str = "n/a") -> str:
    try:
        xf = float(x)
    except Exception:
        return default
    if not np.isfinite(xf):
        return default
    return f"{xf:.2%}"


def print_weights_table(
    weights: dict[str, float],
    *,
    title: str,
    include_zero: bool = False,
    asset_display: dict[str, dict] | None = None,
) -> None:
    print(f"\n=== {title} ===")

    clean = []
    for asset_id, w in (weights or {}).items():
        aid = _norm_asset_key(asset_id)
        if not aid:
            continue
        try:
            wf = float(w)
        except Exception:
            continue
        if not include_zero and abs(wf) < 1e-8:
            continue
        clean.append((aid, _display_symbol(aid, asset_display), wf))

    if not clean:
        print("(empty)")
        return

    clean = sorted(clean, key=lambda x: -x[2])

    print(f"{'Symbol':<16} {'Asset ID':<22} {'Weight':>12}")
    print("-" * 54)
    for aid, sym, w in clean:
        print(f"{sym:<16} {aid:<22} {w:>11.2%}")

def print_shares_table(
    shares: dict[str, float],
    *,
    title: str = "Final executable shares / units",
    asset_display: dict[str, dict] | None = None,
) -> None:
    print(f"\n=== {title} ===")

    clean = []
    for asset_id, q in (shares or {}).items():
        aid = _norm_asset_key(asset_id)
        if not aid:
            continue
        try:
            qf = float(q)
        except Exception:
            continue
        if abs(qf) < 1e-8:
            continue
        clean.append((aid, _display_symbol(aid, asset_display), qf))

    if not clean:
        print("(empty)")
        return

    clean = sorted(clean, key=lambda x: -abs(x[2]))

    print(f"{'Symbol':<16} {'Asset ID':<22} {'Shares / Units':>18}")
    print("-" * 60)
    for aid, sym, q in clean:
        print(f"{sym:<16} {aid:<22} {q:>18,.4f}")


def print_portfolio_metrics(
    m,
    goals=None,
    *,
    title: str = "Portfolio Metrics",
    show_weights: bool = False,
) -> None:
    if goals is None:
        goals = getattr(m, "goals", (800.0, 1200.0, 2000.0))

    g1, g2, g3 = [float(g) for g in goals]

    print(f"\n=== {title} ===")
    print(f"{'Score':<28} {_fmt_float(m.score, 4):>14}")
    print(f"{'Ruin probability 1y':<28} {_fmt_pct(m.ruin_prob_1y):>14}")
    print(f"{'Annual return':<28} {_fmt_pct(m.ann_return):>14}")
    print(f"{'Annual vol sample':<28} {_fmt_pct(m.ann_vol):>14}")
    print(f"{'Annual vol LW':<28} {_fmt_pct(m.ann_vol_lw):>14}")
    print(f"{'Sharpe':<28} {_fmt_float(m.sharpe, 2):>14}")
    print(f"{'Sortino':<28} {_fmt_float(m.sortino, 2):>14}")
    print(f"{'Max drawdown':<28} {_fmt_pct(m.max_drawdown):>14}")
    print(f"{'VaR 95':<28} {_fmt_pct(m.var_95):>14}")
    print(f"{'CVaR 95':<28} {_fmt_pct(m.cvar_95):>14}")

    print("\nGoal probabilities")
    print("-" * 46)
    print(f"{f'P(hit {g1:.0f}, 1y)':<28} {_fmt_pct(m.p_hit_goal_1_1y):>14}")
    print(f"{f'P(hit {g2:.0f}, 1y)':<28} {_fmt_pct(m.p_hit_goal_2_1y):>14}")
    print(f"{f'P(hit {g3:.0f}, 1y)':<28} {_fmt_pct(m.p_hit_goal_3_1y):>14}")

    print("\nMedian time to goal")
    print("-" * 46)
    print(f"{f'{g1:.0f} target':<28} {str(m.med_t_goal_1_days):>14}")
    print(f"{f'{g2:.0f} target':<28} {str(m.med_t_goal_2_days):>14}")
    print(f"{f'{g3:.0f} target':<28} {str(m.med_t_goal_3_days):>14}")

    print("\nEnding equity percentiles")
    print("-" * 46)
    print(f"{'P5':<28} {_fmt_money(m.ending_equity_p5):>14}")
    print(f"{'P25':<28} {_fmt_money(m.ending_equity_p25):>14}")
    print(f"{'P50':<28} {_fmt_money(m.ending_equity_p50):>14}")
    print(f"{'P75':<28} {_fmt_money(m.ending_equity_p75):>14}")
    print(f"{'P95':<28} {_fmt_money(m.ending_equity_p95):>14}")

    if hasattr(m, "stability_energy"):
        print("\nPath stability")
        print("-" * 46)
        print(f"{'Stability energy':<28} {_fmt_float(getattr(m, 'stability_energy', None), 4):>14}")
        print(f"{'Path avg MDD':<28} {_fmt_pct(getattr(m, 'path_mdd_mean', None)):>14}")
        print(f"{'CDaR 95':<28} {_fmt_pct(getattr(m, 'cdar_95', None)):>14}")
        print(f"{'P(DD breach)':<28} {_fmt_pct(getattr(m, 'p_dd_breach', None)):>14}")
        print(f"{'Underwater mean':<28} {_fmt_pct(getattr(m, 'underwater_mean', None)):>14}")
        print(f"{'TTR mean days':<28} {_fmt_float(getattr(m, 'ttr_mean_days', None), 1):>14}")

    if show_weights:
        print_weights_table(m.weights, title=f"{title} weights")


def format_stability_report(
    rep: StabilityReport,
    *,
    days: int = 252,
    cfg: StabilityEnergyConfig | None = None,
) -> str:
    if cfg is None:
        cfg = StabilityEnergyConfig()

    mdd = rep.mdd_mean * 100.0
    cdar = rep.cdar_alpha * 100.0
    ttr_days = rep.ttr_mean_norm * float(days)
    uw = rep.underwater_mean * 100.0
    breach = rep.p_breach * 100.0

    return (
        "Stability (lower is better)\n"
        f"  Energy:            {rep.energy:.4f}\n"
        f"  Avg MDD:           {mdd:.1f}%\n"
        f"  CDaR@{int(cfg.alpha_cdar * 100)}:        {cdar:.1f}%\n"
        f"  Avg TTR:           {ttr_days:.0f} days\n"
        f"  P(MDD ≥ {cfg.breach_dd:.0%}):   {breach:.1f}%\n"
        f"  Underwater time:   {uw:.1f}% of days\n"
    )


def evalmetrics_to_row(m) -> dict:
    goals = getattr(m, "goals", (800.0, 1200.0, 2000.0))
    g1, g2, g3 = [float(g) for g in goals]

    return dict(
        score=float(m.score),
        ann_return=float(m.ann_return),
        ann_vol=float(m.ann_vol),
        ann_vol_lw=float(m.ann_vol_lw),
        sharpe=float(m.sharpe),
        sortino=float(m.sortino),
        max_drawdown=float(m.max_drawdown),
        var_95=float(m.var_95),
        cvar_95=float(m.cvar_95),
        ruin_prob_1y=float(m.ruin_prob_1y),
        p_hit_goal_1_1y=float(m.p_hit_goal_1_1y),
        p_hit_goal_2_1y=float(m.p_hit_goal_2_1y),
        p_hit_goal_3_1y=float(m.p_hit_goal_3_1y),
        med_t_goal_1_days=m.med_t_goal_1_days,
        med_t_goal_2_days=m.med_t_goal_2_days,
        med_t_goal_3_days=m.med_t_goal_3_days,
        end_p5=float(m.ending_equity_p5),
        end_p25=float(m.ending_equity_p25),
        end_p50=float(m.ending_equity_p50),
        end_p75=float(m.ending_equity_p75),
        end_p95=float(m.ending_equity_p95),
        goal_1=g1,
        goal_2=g2,
        goal_3=g3,
    )


def weights_to_rows(weights: dict[str, float], *, tag: str) -> list[dict]:
    # The calculation key is now asset_id. Keep ticker as a backward-compatible
    # column name for older parquet consumers, but store the same value in asset_id.
    return [
        {"tag": tag, "asset_id": str(t), "ticker": str(t), "weight": float(w)}
        for t, w in weights.items()
    ]


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _parse_goals(s: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in str(s).split(",") if str(x).strip()]
    if len(parts) != 3:
        raise ValueError(f"--goals must contain exactly 3 comma-separated numbers. Got {s!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))



def _norm_asset_key(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text

def _build_asset_key_canonical_map(*key_sets: object) -> dict[str, str]:
    """
    Build case-insensitive canonical mapping for asset_id keys.

    The optimizer and returns_wide use case-sensitive asset_id values.
    Some legacy execution code may uppercase keys. This function maps those
    uppercase/lowercase variants back to the canonical asset_id.
    """
    out: dict[str, str] = {}

    for keys in key_sets:
        if keys is None:
            continue

        if isinstance(keys, dict):
            iterable = keys.keys()
        elif isinstance(keys, pd.DataFrame):
            iterable = keys.columns
        elif isinstance(keys, pd.Index):
            iterable = keys.tolist()
        else:
            iterable = keys

        for key in iterable:
            canonical = _norm_asset_key(key)
            if not canonical:
                continue

            out[canonical] = canonical
            out[canonical.upper()] = canonical
            out[canonical.lower()] = canonical
            out[canonical.casefold()] = canonical

    return out


def _canonicalize_asset_keyed_dict(
    values: dict[str, float],
    *,
    canonical_map: dict[str, str],
    keep_cash: bool = True,
    aggregate: bool = True,
) -> dict[str, float]:
    """
    Convert dict keys back to canonical asset_id casing.

    This is required after discrete allocation because legacy ticker-oriented
    code may uppercase equity asset IDs.
    """
    out: dict[str, float] = {}

    for raw_key, raw_value in (values or {}).items():
        key = _norm_asset_key(raw_key)
        if not key:
            continue

        if keep_cash and key.upper() == "CASH":
            canonical = "CASH"
        else:
            canonical = (
                canonical_map.get(key)
                or canonical_map.get(key.upper())
                or canonical_map.get(key.lower())
                or canonical_map.get(key.casefold())
                or key
            )

        try:
            value = float(raw_value)
        except Exception:
            continue

        if aggregate:
            out[canonical] = float(out.get(canonical, 0.0) + value)
        else:
            out[canonical] = float(value)

    return out

def _realized_weights_from_shares(
    shares: dict[str, float],
    prices: dict[str, float],
) -> tuple[dict[str, float], float, dict[str, float]]:
    """
    Convert executable shares/units into signed gross-normalized weights.

    IMPORTANT:
    Keys are asset_id values, not tickers. Do not uppercase them.
    Asset IDs are case-sensitive across the engine.
    """
    exposures: dict[str, float] = {}
    gross = 0.0

    for asset_id, q in (shares or {}).items():
        aid = _norm_asset_key(asset_id)
        if not aid or aid == "CASH":
            continue

        px = prices.get(aid)
        if px is None:
            continue

        try:
            qf = float(q)
            pxf = float(px)
        except Exception:
            continue

        if not np.isfinite(qf) or not np.isfinite(pxf) or pxf <= 0 or abs(qf) <= 1e-12:
            continue

        exp = qf * pxf
        if abs(exp) <= 1e-12:
            continue

        exposures[aid] = float(exp)
        gross += abs(float(exp))

    if gross <= 0 or not np.isfinite(gross):
        raise RuntimeError(
            "Executable allocation produced zero gross notional; cannot re-evaluate final portfolio."
        )

    weights = {asset_id: float(v / gross) for asset_id, v in exposures.items()}
    return weights, float(gross), exposures


def _metric_p_main(m, goals: tuple[float, float, float], main_goal: float) -> float:
    goal_values = [float(x) for x in goals]
    probs = [
        getattr(m, "p_hit_goal_1_1y", np.nan),
        getattr(m, "p_hit_goal_2_1y", np.nan),
        getattr(m, "p_hit_goal_3_1y", np.nan),
    ]

    try:
        mg = float(main_goal)
    except Exception:
        return float(probs[0])

    idx = int(np.argmin([abs(g - mg) for g in goal_values]))
    return float(probs[idx])


def _weights_l1_drift(
    theoretical_weights: dict[str, float],
    executable_weights: dict[str, float],
) -> float:
    """
    L1 drift between theoretical and executable weights.

    Keys are asset_id values. Do not uppercase.
    """
    tw = {_norm_asset_key(k): float(v) for k, v in (theoretical_weights or {}).items() if _norm_asset_key(k)}
    ew = {_norm_asset_key(k): float(v) for k, v in (executable_weights or {}).items() if _norm_asset_key(k)}

    keys = set(tw.keys()) | set(ew.keys())
    keys.discard("CASH")

    drift = 0.0
    for k in keys:
        drift += abs(float(tw.get(k, 0.0) or 0.0) - float(ew.get(k, 0.0) or 0.0))

    return float(drift)

def build_rounding_impact_rows(
    theoretical_weights: dict[str, float],
    executable_weights: dict[str, float],
    shares: dict[str, float],
    *,
    drop_eps: float = 1e-8,
) -> list[dict]:
    """
    Build rounding-impact diagnostics.

    Keys are asset_id values. Do not uppercase.
    """
    tw = {_norm_asset_key(k): float(v) for k, v in (theoretical_weights or {}).items() if _norm_asset_key(k)}
    ew = {_norm_asset_key(k): float(v) for k, v in (executable_weights or {}).items() if _norm_asset_key(k)}
    sh = {_norm_asset_key(k): float(v) for k, v in (shares or {}).items() if _norm_asset_key(k)}

    keys = set(tw.keys()) | set(ew.keys()) | set(sh.keys())
    keys.discard("CASH")

    rows: list[dict] = []
    for asset_id in sorted(keys):
        theoretical_weight = float(tw.get(asset_id, 0.0) or 0.0)
        executable_weight = float(ew.get(asset_id, 0.0) or 0.0)
        quantity = float(sh.get(asset_id, 0.0) or 0.0)

        if abs(theoretical_weight) <= drop_eps and abs(executable_weight) <= drop_eps and abs(quantity) <= drop_eps:
            continue

        if abs(theoretical_weight) > drop_eps and abs(executable_weight) <= drop_eps:
            status = "DROPPED"
        elif abs(executable_weight - theoretical_weight) >= 0.01:
            status = "DRIFT"
        else:
            status = "OK"

        rows.append(
            {
                "asset_id": asset_id,
                "ticker": asset_id,  # backward-compatible column name for old consumers
                "theoretical_weight": float(theoretical_weight),
                "executable_weight": float(executable_weight),
                "delta_weight": float(executable_weight - theoretical_weight),
                "shares": float(quantity),
                "status": status,
            }
        )

    rows.sort(key=lambda r: (-abs(float(r["delta_weight"])), str(r["asset_id"])))
    return rows

def compute_execution_quality(
    *,
    theoretical_metrics,
    final_metrics,
    theoretical_weights: dict[str, float],
    executable_weights: dict[str, float],
    realized_weights_with_cash: dict[str, float],
    shares: dict[str, float],
    target_notional: float,
    executable_gross_notional: float,
    cash_left: float,
    goals: tuple[float, float, float],
    main_goal: float,
) -> dict:
    rounding_rows = build_rounding_impact_rows(
        theoretical_weights=theoretical_weights,
        executable_weights=executable_weights,
        shares=shares,
    )

    dropped_theoretical_weight = float(
        sum(abs(float(r["theoretical_weight"])) for r in rounding_rows if r["status"] == "DROPPED")
    )

    target = float(target_notional)
    executable_gross = float(executable_gross_notional)
    cash = float(cash_left)

    deployment_ratio = float(executable_gross / target) if target > 0 else float("nan")
    cash_weight = float(cash / target) if target > 0 else float("nan")

    theoretical_p_main = _metric_p_main(theoretical_metrics, goals, main_goal)
    final_p_main = _metric_p_main(final_metrics, goals, main_goal)

    return {
        "target_notional": target,
        "executable_gross_notional": executable_gross,
        "deployment_ratio": deployment_ratio,
        "cash_left": cash,
        "cash_weight": cash_weight,
        "weight_drift_l1": _weights_l1_drift(theoretical_weights, executable_weights),
        "dropped_theoretical_weight": dropped_theoretical_weight,
        "score_drop": float(float(theoretical_metrics.score) - float(final_metrics.score)),
        "ruin_increase": float(float(final_metrics.ruin_prob_1y) - float(theoretical_metrics.ruin_prob_1y)),
        "p_main_drop": float(theoretical_p_main - final_p_main),
        "mdd_worsening": float(abs(float(final_metrics.max_drawdown)) - abs(float(theoretical_metrics.max_drawdown))),
        "rounding_impact": rounding_rows,
    }



def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return float(min(1.0, max(0.0, v)))


def _safe_ratio_good(value: float, cap: float, *, lower_is_better: bool = True) -> float:
    """Return 0..1 where 1 is good and 0 is at/above cap for lower-is-better metrics."""
    try:
        v = abs(float(value))
        c = abs(float(cap))
    except Exception:
        return 0.0
    if not np.isfinite(v) or not np.isfinite(c) or c <= 0:
        return 0.0
    if lower_is_better:
        return _clamp01(1.0 - (v / c))
    return _clamp01(v / c)


def compute_portfolio_health_score(
    *,
    final_metrics,
    execution_quality: dict,
    score_cfg: ScoreConfig,
    goals: tuple[float, float, float],
    main_goal: float,
    max_cash_weight: float,
    min_deployment_ratio: float,
    max_executable_mdd: float,
    max_executable_cdar_95: float,
    max_stability_energy: float,
    max_dropped_weight: float,
    max_weight_drift_l1: float,
) -> dict:
    """
    Human-facing 0-100 portfolio health score.

    This is intentionally separate from final_metrics.score:
      - final_metrics.score is the raw optimizer objective and may be negative
        after lambda tuning.
      - health_score is normalized for validation/reporting and should be
        comparable across score-config scale changes.
    """
    p_main = _metric_p_main(final_metrics, goals, main_goal)

    ruin_cap = float(getattr(score_cfg, "ruin_cap", 0.10))
    cvar_cap = float(getattr(score_cfg, "cvar_cap", 0.03))
    path_mdd_cap = float(getattr(score_cfg, "path_mdd_mean_cap", 0.30))
    p_dd_breach_cap = float(getattr(score_cfg, "p_dd_breach_cap", 0.25))
    underwater_cap = float(getattr(score_cfg, "underwater_mean_cap", 1.00))
    ttr_cap = float(getattr(score_cfg, "ttr_cap_days", 252.0))

    final_ruin = float(getattr(final_metrics, "ruin_prob_1y", np.nan))
    final_mdd = abs(float(getattr(final_metrics, "max_drawdown", np.nan)))
    final_cvar = abs(float(getattr(final_metrics, "cvar_95", np.nan)))
    final_stability = float(getattr(final_metrics, "stability_energy", np.nan))
    final_path_mdd = float(getattr(final_metrics, "path_mdd_mean", np.nan))
    final_cdar = float(getattr(final_metrics, "cdar_95", np.nan))
    final_p_breach = float(getattr(final_metrics, "p_dd_breach", np.nan))
    final_underwater = float(getattr(final_metrics, "underwater_mean", np.nan))
    final_ttr = float(getattr(final_metrics, "ttr_mean_days", np.nan))

    components = {
        "goal_probability": _clamp01(p_main),
        "ruin": _safe_ratio_good(final_ruin, ruin_cap),
        "max_drawdown": _safe_ratio_good(final_mdd, max_executable_mdd),
        "cvar_95": _safe_ratio_good(final_cvar, cvar_cap),
        "stability_energy": _safe_ratio_good(final_stability, max_stability_energy),
        "path_mdd_mean": _safe_ratio_good(final_path_mdd, path_mdd_cap),
        "cdar_95": _safe_ratio_good(final_cdar, max_executable_cdar_95),
        "p_dd_breach": _safe_ratio_good(final_p_breach, p_dd_breach_cap),
        "underwater_mean": _safe_ratio_good(final_underwater, underwater_cap),
        "ttr_mean_days": _safe_ratio_good(final_ttr, ttr_cap),
        "deployment": _clamp01(
            float(execution_quality.get("deployment_ratio", np.nan)) / float(min_deployment_ratio)
            if float(min_deployment_ratio) > 0
            else 0.0
        ),
        "cash": _safe_ratio_good(float(execution_quality.get("cash_weight", np.nan)), max_cash_weight),
        "weight_drift": _safe_ratio_good(float(execution_quality.get("weight_drift_l1", np.nan)), max_weight_drift_l1),
        "dropped_weight": _safe_ratio_good(float(execution_quality.get("dropped_theoretical_weight", np.nan)), max_dropped_weight),
    }

    risk_component = float(np.mean([components["ruin"], components["max_drawdown"], components["cvar_95"]]))
    stability_component = float(
        np.mean(
            [
                components["stability_energy"],
                components["path_mdd_mean"],
                components["cdar_95"],
                components["p_dd_breach"],
                components["underwater_mean"],
                components["ttr_mean_days"],
            ]
        )
    )
    execution_component = float(
        np.mean([components["deployment"], components["cash"], components["weight_drift"], components["dropped_weight"]])
    )

    weights = {
        "goal_probability": 0.30,
        "risk": 0.30,
        "stability": 0.30,
        "execution": 0.10,
    }
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
        "raw_optimizer_score": float(getattr(final_metrics, "score", np.nan)),
        "components": {
            "goal_probability": float(components["goal_probability"]),
            "risk": risk_component,
            "stability": stability_component,
            "execution": execution_component,
        },
        "component_details": {k: float(v) for k, v in components.items()},
        "weights": {k: float(v) for k, v in weights.items()},
        "note": "health_score is for validation/reporting; raw_optimizer_score is for candidate ranking only.",
    }


def validate_final_executable(
    *,
    theoretical_metrics,
    final_metrics,
    execution_quality: dict,
    score_cfg: ScoreConfig,
    goals: tuple[float, float, float],
    main_goal: float,
    min_health_score: float,
    min_executable_score: float | None,
    max_score_drop: float | None,
    max_ruin_increase: float,
    max_p_main_drop: float,
    max_cash_weight: float,
    min_deployment_ratio: float,
    max_executable_mdd: float,
    max_executable_cdar_95: float,
    max_stability_energy: float,
    max_dropped_weight: float,
    max_weight_drift_l1: float,
) -> dict:
    reasons: list[str] = []

    final_score = float(final_metrics.score)
    final_ruin = float(final_metrics.ruin_prob_1y)
    final_mdd_abs = abs(float(final_metrics.max_drawdown))
    final_cvar_abs = abs(float(final_metrics.cvar_95))
    final_cdar = float(getattr(final_metrics, "cdar_95", np.nan))
    final_stability = float(getattr(final_metrics, "stability_energy", np.nan))
    final_p_main = _metric_p_main(final_metrics, goals, main_goal)

    ruin_cap = float(getattr(score_cfg, "ruin_cap", 0.10))
    cvar_cap = float(getattr(score_cfg, "cvar_cap", 0.03))

    health = compute_portfolio_health_score(
        final_metrics=final_metrics,
        execution_quality=execution_quality,
        score_cfg=score_cfg,
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

    if float(health["health_score"]) < float(min_health_score):
        reasons.append(
            f"health score {float(health['health_score']):.1f} below min_health_score {float(min_health_score):.1f}"
        )

    if min_executable_score is not None and final_score < float(min_executable_score):
        reasons.append(
            f"optimizer score {final_score:.4f} below explicit min_executable_score {float(min_executable_score):.4f}"
        )

    if final_ruin > ruin_cap:
        reasons.append(f"ruin {final_ruin:.2%} above ruin_cap {ruin_cap:.2%}")

    if final_mdd_abs > float(max_executable_mdd):
        reasons.append(f"max drawdown {final_mdd_abs:.2%} above executable cap {float(max_executable_mdd):.2%}")

    if final_cvar_abs > cvar_cap:
        reasons.append(f"CVaR95 {final_cvar_abs:.2%} above cvar_cap {cvar_cap:.2%}")

    if np.isfinite(final_cdar) and final_cdar > float(max_executable_cdar_95):
        reasons.append(f"CDaR95 {final_cdar:.2%} above executable cap {float(max_executable_cdar_95):.2%}")

    if np.isfinite(final_stability) and final_stability > float(max_stability_energy):
        reasons.append(
            f"stability energy {final_stability:.4f} above executable cap {float(max_stability_energy):.4f}"
        )

    if float(execution_quality["cash_weight"]) > float(max_cash_weight):
        reasons.append(
            f"cash weight {float(execution_quality['cash_weight']):.2%} above max_cash_weight {float(max_cash_weight):.2%}"
        )

    if float(execution_quality["deployment_ratio"]) < float(min_deployment_ratio):
        reasons.append(
            f"deployment ratio {float(execution_quality['deployment_ratio']):.2%} below min_deployment_ratio {float(min_deployment_ratio):.2%}"
        )

    if float(execution_quality["dropped_theoretical_weight"]) > float(max_dropped_weight):
        reasons.append(
            f"dropped theoretical weight {float(execution_quality['dropped_theoretical_weight']):.2%} above max_dropped_weight {float(max_dropped_weight):.2%}"
        )

    if float(execution_quality["weight_drift_l1"]) > float(max_weight_drift_l1):
        reasons.append(
            f"weight drift L1 {float(execution_quality['weight_drift_l1']):.2%} above max_weight_drift_l1 {float(max_weight_drift_l1):.2%}"
        )

    if max_score_drop is not None and float(execution_quality["score_drop"]) > float(max_score_drop):
        reasons.append(
            f"optimizer score drop {float(execution_quality['score_drop']):.4f} above explicit max_score_drop {float(max_score_drop):.4f}"
        )

    if float(execution_quality["ruin_increase"]) > float(max_ruin_increase):
        reasons.append(
            f"ruin increase {float(execution_quality['ruin_increase']):.2%} above max_ruin_increase {float(max_ruin_increase):.2%}"
        )

    if float(execution_quality["p_main_drop"]) > float(max_p_main_drop):
        reasons.append(
            f"P(main) drop {float(execution_quality['p_main_drop']):.2%} above max_p_main_drop {float(max_p_main_drop):.2%}"
        )

    return {
        "status": "accepted" if not reasons else "rejected",
        "passed": not bool(reasons),
        "reasons": reasons,
        "thresholds": {
            "min_health_score": float(min_health_score),
            "min_executable_score": None if min_executable_score is None else float(min_executable_score),
            "max_score_drop": None if max_score_drop is None else float(max_score_drop),
            "max_ruin_increase": float(max_ruin_increase),
            "max_p_main_drop": float(max_p_main_drop),
            "max_cash_weight": float(max_cash_weight),
            "min_deployment_ratio": float(min_deployment_ratio),
            "max_executable_mdd": float(max_executable_mdd),
            "max_executable_cdar_95": float(max_executable_cdar_95),
            "max_stability_energy": float(max_stability_energy),
            "max_dropped_weight": float(max_dropped_weight),
            "max_weight_drift_l1": float(max_weight_drift_l1),
        },
        "final_p_main": float(final_p_main),
        "health": health,
        "health_score": float(health["health_score"]),
        "health_grade": str(health["health_grade"]),
    }


def _clean_display_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def build_asset_display_map(universe_df: pd.DataFrame) -> dict[str, dict]:
    """
    Build display metadata keyed by asset_id.

    Internal calculations use asset_id.
    User-facing tables should display yahoo_ticker_norm / yahoo_ticker / ticker.
    """
    if universe_df is None or universe_df.empty or "asset_id" not in universe_df.columns:
        return {}

    out: dict[str, dict] = {}

    for _, row in universe_df.iterrows():
        asset_id = _clean_display_value(row.get("asset_id"))
        if not asset_id:
            continue

        ticker = _clean_display_value(row.get("ticker"))
        yahoo_ticker = _clean_display_value(row.get("yahoo_ticker"))
        yahoo_ticker_norm = _clean_display_value(row.get("yahoo_ticker_norm"))
        name = _clean_display_value(row.get("name"))
        asset_class = _clean_display_value(row.get("asset_class"))
        region = _clean_display_value(row.get("region"))

        display_symbol = yahoo_ticker_norm or yahoo_ticker or ticker or asset_id

        out[asset_id] = {
            "asset_id": asset_id,
            "ticker": ticker,
            "yahoo_ticker": yahoo_ticker,
            "yahoo_ticker_norm": yahoo_ticker_norm,
            "display_symbol": display_symbol,
            "name": name,
            "asset_class": asset_class,
            "region": region,
        }

    return out


def _display_symbol(asset_id: object, asset_display: dict[str, dict] | None = None) -> str:
    aid = _norm_asset_key(asset_id)
    if not aid:
        return ""

    if asset_display:
        meta = asset_display.get(aid) or {}
        return (
            _clean_display_value(meta.get("display_symbol"))
            or _clean_display_value(meta.get("yahoo_ticker_norm"))
            or _clean_display_value(meta.get("yahoo_ticker"))
            or _clean_display_value(meta.get("ticker"))
            or aid
        )

    return aid

def print_execution_quality(execution_quality: dict) -> None:
    print("\n=== Execution quality ===")
    print(f"{'Target notional':<32} {_fmt_money(execution_quality.get('target_notional')):>16} USD")
    print(f"{'Executable gross notional':<32} {_fmt_money(execution_quality.get('executable_gross_notional')):>16} USD")
    print(f"{'Deployment ratio':<32} {_fmt_pct(execution_quality.get('deployment_ratio')):>16}")
    print(f"{'Cash left':<32} {_fmt_money(execution_quality.get('cash_left')):>16} USD")
    print(f"{'Cash weight':<32} {_fmt_pct(execution_quality.get('cash_weight')):>16}")
    print(f"{'Weight drift L1':<32} {_fmt_pct(execution_quality.get('weight_drift_l1')):>16}")
    print(f"{'Dropped theoretical weight':<32} {_fmt_pct(execution_quality.get('dropped_theoretical_weight')):>16}")
    print(f"{'Score drop':<32} {_fmt_float(execution_quality.get('score_drop'), 4):>16}")
    print(f"{'Ruin increase':<32} {_fmt_pct(execution_quality.get('ruin_increase')):>16}")
    print(f"{'P(main) drop':<32} {_fmt_pct(execution_quality.get('p_main_drop')):>16}")
    print(f"{'MDD worsening':<32} {_fmt_pct(execution_quality.get('mdd_worsening')):>16}")


def print_rounding_impact_table(
    rows: list[dict],
    *,
    asset_display: dict[str, dict] | None = None,
) -> None:
    print("\n=== Rounding impact ===")

    if not rows:
        print("(empty)")
        return

    print(
        f"{'Symbol':<16} {'Asset ID':<22} "
        f"{'Theory W':>10} {'Exec W':>10} {'Delta':>10} "
        f"{'Shares':>14} {'Status':>10}"
    )
    print("-" * 98)

    for r in rows:
        aid = _norm_asset_key(r.get("asset_id") or r.get("ticker"))
        sym = _display_symbol(aid, asset_display)
        print(
            f"{sym:<16} {aid:<22} "
            f"{_fmt_pct(r.get('theoretical_weight')):>10} "
            f"{_fmt_pct(r.get('executable_weight')):>10} "
            f"{_fmt_pct(r.get('delta_weight')):>10} "
            f"{_fmt_float(r.get('shares'), 4):>14} "
            f"{str(r.get('status', '')):>10}"
        )

def print_executable_validation(validation: dict) -> None:
    status = str(validation.get("status", "unknown")).upper()
    print("\n=== Executable validation ===")
    print(f"Status: {status}")
    if "health_score" in validation:
        print(f"Health score: {_fmt_float(validation.get('health_score'), 1)} / 100 ({validation.get('health_grade', 'n/a')})")
    print("Optimizer score floor: " + ("disabled" if validation.get("thresholds", {}).get("min_executable_score") is None else str(validation.get("thresholds", {}).get("min_executable_score"))))

    reasons = validation.get("reasons") or []
    if not reasons:
        print("All validation checks passed.")
        return

    print("Reasons:")
    for reason in reasons:
        print(f"  - {reason}")



def _candidate_fingerprint(weights: dict[str, float], *, ndigits: int = 8) -> tuple[tuple[str, float], ...]:
    items = []
    for k, v in (weights or {}).items():
        try:
            vf = float(v)
        except Exception:
            continue
        if not np.isfinite(vf) or abs(vf) < 1e-10:
            continue
        items.append((str(k).strip(), round(vf, int(ndigits))))
    return tuple(sorted(items))


def _candidate_summary(
    *,
    label: str,
    theoretical_metrics,
    final_metrics,
    validation: dict,
    execution_quality: dict,
) -> dict:
    return {
        "label": str(label),
        "status": str(validation.get("status", "unknown")),
        "passed": bool(validation.get("passed", False)),
        "optimizer_score": float(final_metrics.score),
        "score": float(final_metrics.score),  # backward-compatible alias for raw optimizer score
        "health_score": float(validation.get("health_score", np.nan)),
        "health_grade": str(validation.get("health_grade", "n/a")),
        "theoretical_score": float(theoretical_metrics.score),
        "p_main": float(validation.get("final_p_main", np.nan)),
        "ruin_prob_1y": float(final_metrics.ruin_prob_1y),
        "max_drawdown": float(final_metrics.max_drawdown),
        "stability_energy": float(getattr(final_metrics, "stability_energy", np.nan)),
        "path_mdd_mean": float(getattr(final_metrics, "path_mdd_mean", np.nan)),
        "cdar_95": float(getattr(final_metrics, "cdar_95", np.nan)),
        "deployment_ratio": float(execution_quality.get("deployment_ratio", np.nan)),
        "cash_weight": float(execution_quality.get("cash_weight", np.nan)),
        "weight_drift_l1": float(execution_quality.get("weight_drift_l1", np.nan)),
        "dropped_theoretical_weight": float(execution_quality.get("dropped_theoretical_weight", np.nan)),
        "score_drop": float(execution_quality.get("score_drop", np.nan)),
        "ruin_increase": float(execution_quality.get("ruin_increase", np.nan)),
        "p_main_drop": float(execution_quality.get("p_main_drop", np.nan)),
        "mdd_worsening": float(execution_quality.get("mdd_worsening", np.nan)),
        "reasons": list(validation.get("reasons") or []),
    }


def _build_executable_selection_payload(
    *,
    executable_candidate_summaries: list[dict],
    selected_candidate_label: str,
    final_validation: dict,
    candidate_errors: list[dict],
    executable_selection_top_k: int,
) -> dict:
    accepted = [c for c in executable_candidate_summaries if bool(c.get("passed", False))]
    rejected = [c for c in executable_candidate_summaries if not bool(c.get("passed", False))]

    return {
        "schema_version": "executable_selection_v1",
        "candidate_count": int(len(executable_candidate_summaries)),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "error_count": int(len(candidate_errors or [])),
        "selected_label": str(selected_candidate_label),
        "selected_status": str(final_validation.get("status", "unknown")),
        "selected_passed": bool(final_validation.get("passed", False)),
        "selection_rule": (
            "highest final executable optimizer score among accepted candidates; "
            "if none accepted, highest final executable optimizer score retained for diagnostics only"
        ),
        "executable_selection_top_k": int(executable_selection_top_k),
        "candidates": list(executable_candidate_summaries),
        "candidate_errors": list(candidate_errors or []),
    }


def print_executable_candidate_summary(candidates: list[dict], *, max_rows: int = 15) -> None:
    print("\n=== Executable candidate selection ===")
    if not candidates:
        print("No executable candidates evaluated.")
        return

    accepted = [c for c in candidates if bool(c.get("validation", {}).get("passed", False))]
    print(f"Candidates evaluated: {len(candidates)} | Accepted: {len(accepted)} | Rejected: {len(candidates) - len(accepted)}")
    print("-" * 128)
    print(
        f"{'Rank':>4} {'Label':<30} {'Status':<9} {'Health':>8} {'OptScore':>9} {'P(main)':>9} "
        f"{'Ruin':>9} {'MDD':>9} {'Stab':>9} {'Drift':>9} {'Dropped':>9}"
    )
    print("-" * 128)

    ordered = sorted(
        candidates,
        key=lambda c: (
            not bool(c.get("validation", {}).get("passed", False)),
            -float(getattr(c.get("final_metrics"), "score", -1e9)),
        ),
    )

    for i, c in enumerate(ordered[: int(max_rows)], start=1):
        v = c.get("validation", {}) or {}
        eq = c.get("execution_quality", {}) or {}
        fm = c.get("final_metrics")
        status = str(v.get("status", "unknown")).upper()
        print(
            f"{i:>4} {str(c.get('label', 'candidate')):<30.30} {status:<9} "
            f"{_fmt_float(v.get('health_score', np.nan), 1):>8} "
            f"{_fmt_float(getattr(fm, 'score', np.nan), 4):>9} "
            f"{_fmt_pct(v.get('final_p_main', np.nan)):>9} "
            f"{_fmt_pct(getattr(fm, 'ruin_prob_1y', np.nan)):>9} "
            f"{_fmt_pct(getattr(fm, 'max_drawdown', np.nan)):>9} "
            f"{_fmt_float(getattr(fm, 'stability_energy', np.nan), 3):>9} "
            f"{_fmt_pct(eq.get('weight_drift_l1', np.nan)):>9} "
            f"{_fmt_pct(eq.get('dropped_theoretical_weight', np.nan)):>9}"
        )

    if len(ordered) > int(max_rows):
        print(f"... {len(ordered) - int(max_rows)} more candidates")


def _evaluate_executable_candidate(
    *,
    label: str,
    theoretical_metrics,
    returns: pd.DataFrame,
    prices_asset_id: dict[str, float],
    equity0: float,
    target_notional: float,
    goals: tuple[float, float, float],
    main_goal: float,
    score_cfg: ScoreConfig,
    n_paths: int,
    pca_k: int,
    block_size: tuple[int, int],
    mc_seed: int,
    min_health_score: float,
    min_executable_score: float | None,
    max_score_drop: float | None,
    max_ruin_increase: float,
    max_p_main_drop: float,
    max_cash_weight: float,
    min_deployment_ratio: float,
    max_executable_mdd: float,
    max_executable_cdar_95: float,
    max_stability_energy: float,
    max_dropped_weight: float,
    max_weight_drift_l1: float,
) -> dict:
    alloc = weights_to_discrete_shares(
        weights=dict(theoretical_metrics.weights),
        prices=prices_asset_id,
        notional=float(target_notional),
        min_weight=0.01,
        min_units_equity=1.0,
        min_units_crypto=0.0,
        min_units_weight_thr=0.03,
        crypto_decimals=8,
        nearest_step_remaining_frac=0.10,
    )

    canonical_map = _build_asset_key_canonical_map(
        theoretical_metrics.weights,
        prices_asset_id,
        returns.columns,
    )

    alloc_shares = _canonicalize_asset_keyed_dict(
        {k: float(v) for k, v in (alloc.shares or {}).items()},
        canonical_map=canonical_map,
        keep_cash=False,
    )

    alloc_realized_weights = _canonicalize_asset_keyed_dict(
        {k: float(v) for k, v in (alloc.realized_weights or {}).items()},
        canonical_map=canonical_map,
        keep_cash=True,
    )

    final_weights, final_gross_notional, final_exposures = _realized_weights_from_shares(
        alloc_shares,
        prices_asset_id,
    )

    final_metrics = evaluate_portfolio_candidate(
        returns=returns,
        weights=final_weights,
        equity0=float(equity0),
        notional=float(final_gross_notional),
        goals=list(goals),
        main_goal=float(main_goal),
        lw_cov=None,
        days=252,
        n_paths=int(n_paths),
        score_config=score_cfg,
        mc_seed=int(mc_seed),
        path_source="bootstrap",
        pca_k=int(pca_k),
        block_size=block_size,
        weight_mode="gross_signed",
    )

    execution_quality = compute_execution_quality(
        theoretical_metrics=theoretical_metrics,
        final_metrics=final_metrics,
        theoretical_weights=dict(theoretical_metrics.weights),
        executable_weights=final_weights,
        realized_weights_with_cash=alloc_realized_weights,
        shares=alloc_shares,
        target_notional=float(target_notional),
        executable_gross_notional=float(final_gross_notional),
        cash_left=float(alloc.cash_left),
        goals=goals,
        main_goal=float(main_goal),
    )

    raw_share_keys = {_norm_asset_key(k) for k in (alloc.shares or {}).keys()}
    canonical_share_keys = {_norm_asset_key(k) for k in (alloc_shares or {}).keys()}

    unmapped_share_keys = sorted(
        k for k in raw_share_keys
        if k
        and k.upper() != "CASH"
        and (
            k not in canonical_map
            and k.upper() not in canonical_map
            and k.lower() not in canonical_map
            and k.casefold() not in canonical_map
        )
    )

    if unmapped_share_keys:
        raise RuntimeError(
            "Discrete allocation produced share keys that could not be mapped "
            f"back to canonical asset_id values: {unmapped_share_keys[:20]}"
        )

    if final_gross_notional < float(target_notional) * 0.50:
        raise RuntimeError(
            "Executable allocation deployed less than 50% of target notional after "
            "canonicalization. This is likely a key-mapping or price-coverage bug. "
            f"gross={final_gross_notional:.2f} target={float(target_notional):.2f} "
            f"raw_share_keys_sample={sorted(list(raw_share_keys))[:20]} "
            f"canonical_share_keys_sample={sorted(list(canonical_share_keys))[:20]}"
        )

    validation = validate_final_executable(
        theoretical_metrics=theoretical_metrics,
        final_metrics=final_metrics,
        execution_quality=execution_quality,
        score_cfg=score_cfg,
        goals=goals,
        main_goal=float(main_goal),
        min_health_score=float(min_health_score),
        min_executable_score=min_executable_score,
        max_score_drop=max_score_drop,
        max_ruin_increase=float(max_ruin_increase),
        max_p_main_drop=float(max_p_main_drop),
        max_cash_weight=float(max_cash_weight),
        min_deployment_ratio=float(min_deployment_ratio),
        max_executable_mdd=float(max_executable_mdd),
        max_executable_cdar_95=float(max_executable_cdar_95),
        max_stability_energy=float(max_stability_energy),
        max_dropped_weight=float(max_dropped_weight),
        max_weight_drift_l1=float(max_weight_drift_l1),
    )

    return {
        "label": str(label),
        "theoretical_metrics": theoretical_metrics,
        "final_metrics": final_metrics,
        "final_weights": final_weights,
        "final_gross_notional": float(final_gross_notional),
        "final_exposures": final_exposures,
        "alloc": alloc,
        "alloc_shares": alloc_shares,
        "alloc_realized_weights": alloc_realized_weights,
        "execution_quality": execution_quality,
        "validation": validation,
        "health": validation.get("health", {}),
        "health_score": float(validation.get("health_score", np.nan)),
    }


def _build_executable_candidate_pool(
    *,
    best_ga,
    best_by_stability,
    best_refined,
    ga_archive: list,
    stability_ranked: list[tuple] | None,
    executable_selection_top_k: int,
) -> list[tuple[str, object]]:
    pool: list[tuple[str, object]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()

    def add(label: str, m) -> None:
        if m is None:
            return
        fp = _candidate_fingerprint(getattr(m, "weights", {}) or {})
        if not fp or fp in seen:
            return
        seen.add(fp)
        pool.append((label, m))

    add("ga_best", best_ga)
    add("stability_best", best_by_stability)
    add("annealed_theoretical", best_refined)

    max_extra = max(0, int(executable_selection_top_k))
    if stability_ranked:
        for i, item in enumerate(stability_ranked[:max_extra], start=1):
            try:
                m = item[0]
            except Exception:
                continue
            add(f"stability_rank_{i}", m)
    else:
        for i, m in enumerate((ga_archive or [])[:max_extra], start=1):
            add(f"ga_archive_{i}", m)

    return pool


def run_portfolio_search_asof(
    *,
    as_of: str,
    equity0: float,
    goals: tuple[float, float, float],
    main_goal: float,
    universe_csv: str | None = None,
    use_market_hmm: bool = True,
    override_target_leverage: float | None = None,
    write_outputs: bool = True,
    run_dt: str | pd.Timestamp | None = None,
    cache_min_years: float = 5.0,
    cfg: RuntimeConfig | None = None,
    env: str | None = None,
    confirm_prod_write: bool = False,
    pop_size: int = 80,
    generations: int = 50,
    elite_frac: float = 0.10,
    max_assets: int = 10,
    min_assets: int = 5,
    n_paths_init: int = 5000,
    n_paths_final: int = 20000,
    pca_k: int = 3,
    block_min: int = 8,
    block_max: int = 12,
    skip_stability_rerank: bool = False,
    stability_top_k: int = 200,
    stability_n_paths: int = 20000,
    executable_selection_top_k: int = 25,
    anneal_steps: int = 200,
    anneal_n_paths_init: int = 3000,
    anneal_n_paths_final: int = 20000,
    min_universe_size: int = 10,
    rebuild_returns_cache: bool = False,
    min_health_score: float = 60.0,
    min_executable_score: float | None = None,
    max_score_drop: float | None = None,
    max_ruin_increase: float = 0.03,
    max_p_main_drop: float = 0.15,
    max_cash_weight: float = 0.05,
    min_deployment_ratio: float = 0.95,
    max_executable_mdd: float = 0.40,
    max_executable_cdar_95: float = 0.60,
    max_stability_energy: float = 2.00,
    max_dropped_weight: float = 0.04,
    max_weight_drift_l1: float = 0.15,
    actuarial_max_allowed_leverage: float = 2.0,
    mutation_sigma_start: float = 0.30,
    mutation_sigma_end: float = 0.05,
    replace_prob_start: float = 0.40,
    replace_prob_end: float = 0.05,
    immigrant_rate_start: float = 0.20,
    immigrant_rate_end: float = 0.03,
    exploration_power: float = 1.5,
    archive_diversity_min_l1: float = 0.15,
    archive_diversity_check_top_k: int = 250,
) -> dict:
    """
    Backtest-friendly portfolio search.

    Runtime behavior:
      - cfg controls bucket, region, engine root, and market root.
      - dev/staging write under cfg.engine_root, not prod engine/v1.
      - prod writes require confirm_prod_write=True when write_outputs=True.

    Smoke-test behavior:
      - use CLI knobs to run smaller GA/stability/annealing workloads.
      - use --universe-csv to pass a small dev universe sample.
    """
    if cfg is None:
        cfg = load_runtime_config(env)

    if write_outputs:
        require_prod_confirmation(cfg, bool(confirm_prod_write))

    bucket = cfg_bucket(cfg)
    region = cfg_region(cfg)
    engine_root = cfg_engine_root(cfg)
    market_root = cfg_market_root(cfg)

    as_of_ts = pd.Timestamp(as_of).tz_localize(None).normalize()
    as_of_market_date = as_of_ts.strftime("%Y-%m-%d")

    if run_dt is None:
        run_dt_ts = pd.Timestamp(dt.date.today()).normalize()
    else:
        run_dt_ts = pd.Timestamp(run_dt).tz_localize(None).normalize()
    as_of_run_date = run_dt_ts.strftime("%Y-%m-%d")

    s3 = s3_init(region)
    market = make_market_store(cfg)

    GOALS = goals
    MAIN_GOAL = float(main_goal)

    print("\n=== PORTFOLIO SEARCH RUNTIME ===")
    print(f"env:           {getattr(cfg, 'env', 'unknown')}")
    print(f"bucket:        {bucket}")
    print(f"region:        {region}")
    print(f"engine_root:   {engine_root}")
    print(f"market_root:   {market_root}")
    print(f"write_outputs: {bool(write_outputs)}")
    print("")

    # ---------- Load latest inputs ----------
    raw_positions = s3_load_latest_json(
        s3,
        bucket=bucket,
        root_prefix=engine_root,
        table="inputs/positions",
    )
    if not raw_positions:
        raise RuntimeError(
            f"Missing S3 latest positions. Expected s3://{bucket}/{engine_root}/inputs/positions/latest.json"
        )
    positions = parse_positions_obj(raw_positions)

    raw_score_cfg = s3_load_latest_json(
        s3,
        bucket=bucket,
        root_prefix=engine_root,
        table="configs/score_config",
    )
    if not raw_score_cfg:
        raise RuntimeError(
            f"Missing S3 latest score_config. Expected s3://{bucket}/{engine_root}/configs/score_config/latest.json"
        )
    score_cfg = ScoreConfig(**raw_score_cfg)

    # ---------- Universe ----------
    if universe_csv is None:
        u = pd.read_csv(paths.universe_dir() / "universe.csv")
    else:
        u = pd.read_csv(universe_csv)

    asset_display = build_asset_display_map(u)

    u = u[u.get("include", 1).fillna(1).astype(int) == 1].copy()
    u["asset_id"] = u["asset_id"].astype(str).str.strip()
    u["ticker"] = u.get("ticker", u["asset_id"]).astype(str).str.upper().str.strip()

    if "yahoo_ticker" not in u.columns:
        u["yahoo_ticker"] = u["ticker"]
    u["yahoo_ticker"] = u["yahoo_ticker"].astype(str).str.strip()

    # Provider/display symbol. universe_override.csv should already have resolved
    # this upstream where applicable. Fall back safely if the normalized column
    # is not present in older universe files.
    if "yahoo_ticker_norm" not in u.columns:
        u["yahoo_ticker_norm"] = u["yahoo_ticker"]
    u["yahoo_ticker_norm"] = u["yahoo_ticker_norm"].astype(str).str.strip()

    if "name" not in u.columns:
        u["name"] = u["ticker"]
    u["name"] = u["name"].astype(str).str.strip()

    dup_asset = u[u["asset_id"].duplicated(keep=False)].sort_values("asset_id")
    if not dup_asset.empty:
        cols = [c for c in ["asset_id", "ticker", "yahoo_ticker_norm", "yahoo_ticker", "name"] if c in dup_asset.columns]
        raise RuntimeError(
            "Duplicate active asset_id values found in universe. asset_id must be unique.\n"
            + dup_asset[cols].head(50).to_string(index=False)
        )

    # Ticker is no longer a primary key. Keep this only for reference/current
    # position notional estimation, and only when the ticker is unambiguous.
    ticker_groups = u.groupby("ticker")["asset_id"].apply(list).to_dict()
    ticker_to_asset = {t: ids[0] for t, ids in ticker_groups.items() if len(ids) == 1}

    asset_display = {}
    for _, _r in u.iterrows():
        _aid = str(_r["asset_id"]).strip()
        _ticker = str(_r.get("ticker", _aid)).strip()
        _yahoo = str(_r.get("yahoo_ticker", _ticker)).strip()
        _ynorm = str(_r.get("yahoo_ticker_norm", _yahoo)).strip()
        _name = str(_r.get("name", _ynorm or _ticker or _aid)).strip()
        asset_display[_aid] = {
            "asset_id": _aid,
            "ticker": _ticker,
            "yahoo_ticker": _yahoo,
            "yahoo_ticker_norm": _ynorm,
            "name": _name,
            "display_symbol": _ynorm or _yahoo or _ticker or _aid,
        }

    # ---------- Latest prices ----------
    latest_prices_df = market.read_latest_prices_snapshot()
    if latest_prices_df.empty:
        raise RuntimeError(
            f"Missing latest_prices snapshot in S3. Expected under s3://{bucket}/{latest_prices_snapshot_ref(cfg)}"
        )

    latest_prices_df["asset_id"] = latest_prices_df["asset_id"].astype(str).str.strip()

    if "adj_close_usd" in latest_prices_df.columns:
        px_col = "adj_close_usd"
    elif "close_raw_usd" in latest_prices_df.columns:
        px_col = "close_raw_usd"
    elif "close_usd" in latest_prices_df.columns:
        px_col = "close_usd"
    else:
        raise RuntimeError(f"latest_prices snapshot missing price column. Columns={list(latest_prices_df.columns)}")

    latest_prices_df[px_col] = pd.to_numeric(latest_prices_df[px_col], errors="coerce")
    px_map = latest_prices_df.set_index("asset_id")[px_col].dropna().to_dict()

    # ---------- Regime / leverage ----------
    lev_rec: dict = {}
    target_leverage: float

    if override_target_leverage is not None:
        target_leverage = float(override_target_leverage)
        lev_rec = {
            "mode": "override",
            "chosen_label": None,
            "confidence": np.nan,
            "leverage": target_leverage,
        }
    else:
        hmm_payload_wrapped = {}
        if use_market_hmm:
            hmm_payload_wrapped = s3_load_latest_json(
                s3,
                bucket=bucket,
                root_prefix=engine_root,
                table="regimes/market_hmm",
            ) or {}

        lr = (
            (hmm_payload_wrapped.get("leverage_recommendation") or {})
            if isinstance(hmm_payload_wrapped, dict)
            else {}
        )

        if isinstance(lr, dict) and lr.get("leverage") is not None:
            lev_rec = dict(lr)
            target_leverage = float(lev_rec["leverage"])
            lev_rec.setdefault("mode", "stored")
            lev_rec.setdefault("confidence", np.nan)
            lev_rec.setdefault(
                "chosen_label",
                (lev_rec.get("chosen_label") or lev_rec.get("label") or lev_rec.get("label_commit")),
            )
        else:
            hmm_res = hmm_payload_wrapped.get("hmm") if isinstance(hmm_payload_wrapped, dict) else None
            if hmm_res is None:
                hmm_res = hmm_payload_wrapped if isinstance(hmm_payload_wrapped, dict) else {}

            st_raw = market.read_regime_filter_state() or {}
            st = RegimeFilterState(
                last_date=st_raw.get("last_date"),
                chosen_label=st_raw.get("chosen_label"),
                days_in_regime=int(st_raw.get("days_in_regime", 0) or 0),
                probs_smoothed=st_raw.get("probs_smoothed"),
            )

            lev_rec = leverage_from_hmm(
                hmm_res or {},
                default=1.0,
                risk_appetite=0.6,
                low_confidence_floor=0.2,
                hard_cap=12.0,
                filter_state=st,
                as_of=as_of_market_date,
                filter_alpha=0.20,
                min_hold_days=3,
                min_prob_to_switch=0.60,
                min_margin_to_switch=0.12,
            )

            if write_outputs and isinstance(lev_rec.get("filter_state"), dict):
                market.write_regime_filter_state(lev_rec["filter_state"])

            target_leverage = float(lev_rec.get("leverage", 1.0))

    # ---------- Target notional ----------
    notional = float(equity0) * float(target_leverage)
    if not np.isfinite(notional) or notional <= 0:
        raise RuntimeError(f"Invalid target notional={notional} from equity0={equity0} and lev={target_leverage}")

    current_gross_notional = 0.0
    missing_px = []
    for p in positions.values():
        t = str(p.ticker).upper().strip()
        aid = ticker_to_asset.get(t)
        if not aid or aid not in px_map:
            missing_px.append(t)
            continue

        px = float(px_map[aid])
        if not np.isfinite(px) or px <= 0:
            missing_px.append(t)
            continue

        current_gross_notional += abs(float(p.quantity) * px)

    conf = _safe_float(lev_rec.get("confidence"), default=np.nan)
    conf_s = "n/a" if not np.isfinite(conf) else f"{conf:.2f}"

    print(
        f"[dates] as_of_market_date={as_of_market_date} | as_of_run_date={as_of_run_date}\n"
        f"[capital] equity0={equity0:.2f} USD | "
        f"regime={lev_rec.get('chosen_label')} ({lev_rec.get('mode')}, conf={conf_s}) | "
        f"target_leverage={target_leverage:.2f}x -> target_notional={notional:.2f} USD"
    )

    if current_gross_notional > 0:
        print(f"[capital] current_positions_notional≈{current_gross_notional:.2f} USD (reference only)")
    if missing_px:
        print(f"[capital][warn] missing prices for {len(missing_px)} tickers (sample: {missing_px[:10]})")

    # ---------- Load returns wide cache ----------
    cache_cfg = make_cache_config(cfg, min_years=float(cache_min_years))
    cache_key = returns_cache_key(cfg, min_years=float(cache_min_years))
    returns_path = returns_cache_uri(cfg, min_years=float(cache_min_years))

    if rebuild_returns_cache:
        print(f"[returns_cache] rebuild requested -> {returns_path}")
        build_returns_wide_cache(cache_cfg)
    else:
        if not s3_key_exists(s3, bucket=bucket, key=cache_key):
            raise RuntimeError(
                "Missing runtime returns cache.\n"
                f"Expected: s3://{bucket}/{cache_key}\n"
                "Do not let portfolio search silently build this from default/prod roots.\n"
                "Build the dev cache first, or rerun with --rebuild-returns-cache only after "
                "alpha_edge.market.build_returns_wide_cache is confirmed runtime-aware."
            )
        print(f"[returns_cache] using existing -> {returns_path}")

    returns_wide = pd.read_parquet(returns_path, engine="pyarrow").sort_index()

    # Normalize index BEFORE cleaning. The cache can start in 2010, while many assets
    # only have 5y of valid history. Cleaning the full 2010-now matrix would drop
    # newer but valid assets because of pre-inception NaNs.
    returns_wide.index = pd.to_datetime(
        returns_wide.index,
        errors="coerce",
        utc=True,
    ).tz_convert(None).normalize()

    returns_wide = returns_wide.loc[~returns_wide.index.isna()].copy()
    returns_wide = returns_wide.loc[returns_wide.index <= as_of_ts].copy()
    returns_wide.columns = [str(c).strip() for c in returns_wide.columns]
    returns_wide = returns_wide.loc[:, ~pd.Index(returns_wide.columns).duplicated(keep="last")]
    returns_wide = returns_wide.sort_index()

    if returns_wide.shape[0] < 252:
        raise RuntimeError(
            f"Not enough returns history up to as_of={as_of_market_date}: rows={returns_wide.shape[0]}"
        )

    # Portfolio search should evaluate on the same effective horizon required by the
    # cache, not on the full 2010-now sparse matrix.
    search_start_ts = max(
        pd.Timestamp(returns_wide.index.min()).normalize(),
        as_of_ts - pd.Timedelta(days=int(float(cache_min_years) * 365.25)),
    )

    returns_wide = returns_wide.loc[returns_wide.index >= search_start_ts].copy()

    print(
        "[returns_cache] effective_search_window="
        f"{search_start_ts.date()}..{as_of_ts.date()} "
        f"rows={returns_wide.shape[0]} "
        f"assets_before_gate={returns_wide.shape[1]}"
    )

    if returns_wide.shape[0] < 252:
        raise RuntimeError(
            f"Not enough returns history in effective search window: "
            f"start={search_start_ts.date()} end={as_of_market_date} rows={returns_wide.shape[0]}"
        )

    # ---------- Market data-quality gate ----------
    dq = _load_latest_market_data_quality_summary(
        bucket=bucket,
        market_root=market_root,
    )

    eligible_asset_ids, dq_gate_meta = _eligible_asset_ids_from_market_data_quality(
        dq,
        allow_warn=True,
    )

    before_cols = int(returns_wide.shape[1])
    returns_wide.columns = [str(c).strip() for c in returns_wide.columns]

    returns_wide = returns_wide[
        [c for c in returns_wide.columns if str(c).strip() in eligible_asset_ids]
    ].copy()

    after_gate_cols = int(returns_wide.shape[1])

    print(
        "[data_quality_gate] "
        f"returns_wide_assets_before={before_cols} "
        f"after_gate={after_gate_cols} "
        f"eligible_asset_ids={len(eligible_asset_ids)}"
    )

    if after_gate_cols == 0:
        raise RuntimeError(
            "Data-quality gate removed all returns_wide assets. "
            f"Gate meta: {dq_gate_meta}"
        )

    # Clean only after:
    #   1. date slicing to the effective portfolio-search window
    #   2. asset_id data-quality filtering
    #
    # This avoids dropping valid assets because they had NaNs before inception in the
    # full 2010-now cache.
    returns_wide, diag = clean_returns_matrix(returns_wide)

    print(
        "[returns_clean] "
        f"assets_after_clean={returns_wide.shape[1]} "
        f"rows_after_clean={returns_wide.shape[0]}"
    )

    if returns_wide.shape[1] == 0:
        raise RuntimeError(
            "No returns_wide assets remain after data-quality gate and cleaning. "
            f"Gate meta: {dq_gate_meta} diag={diag}"
        )

    if "asset_id" in u.columns:
        before_u = int(len(u))
        u["asset_id"] = u["asset_id"].astype(str).str.strip()
        u = u[u["asset_id"].isin(eligible_asset_ids)].copy()
        after_u = int(len(u))

        print(
            "[data_quality_gate] "
            f"universe_rows_before={before_u} "
            f"after={after_u}"
        )

        if after_u == 0:
            raise RuntimeError(
                "Data-quality gate removed all universe rows. "
                f"Gate meta: {dq_gate_meta}"
            )
    else:
        raise RuntimeError("Universe dataframe missing 'asset_id'; cannot apply data-quality gate.")

    # Keep returns_wide keyed by immutable asset_id. Do not rename to ticker.
    returns_wide.columns = [str(c).strip() for c in returns_wide.columns]

    # ---------- Build universe keyed by asset_id ----------
    universe = {}
    for _, row in u.iterrows():
        aid = str(row["asset_id"]).strip()
        if aid in returns_wide.columns:
            display = asset_display.get(aid, {})
            universe[aid] = Asset(
                ticker=aid,  # calculation key used by optimizer = asset_id
                yahoo_ticker=display.get("yahoo_ticker_norm") or display.get("yahoo_ticker") or aid,
                name=display.get("name") or display.get("display_symbol") or aid,
                asset_class=row.get("asset_class", ""),
                role=row.get("role", ""),
                region=row.get("region", ""),
                max_weight=float(row.get("max_weight", 1.0)),
                min_weight=float(row.get("min_weight", 0.0)),
                include=True,
            )

    if len(universe) < int(min_universe_size):
        raise RuntimeError(
            f"Universe after returns slicing/cleaning too small: {len(universe)}. "
            f"Required min_universe_size={int(min_universe_size)}. diag={diag}"
        )

    # ---------- Search ----------
    ga_params = dict(
        pop_size=int(pop_size),
        generations=int(generations),
        elite_frac=float(elite_frac),
        max_assets=int(max_assets),
        min_assets=int(min_assets),
        n_paths_init=int(n_paths_init),
        n_paths_final=int(n_paths_final),
        path_source="bootstrap",
        pca_k=int(pca_k),
        block_size=(int(block_min), int(block_max)),
        mutation_sigma_start=float(mutation_sigma_start),
        mutation_sigma_end=float(mutation_sigma_end),
        replace_prob_start=float(replace_prob_start),
        replace_prob_end=float(replace_prob_end),
        immigrant_rate_start=float(immigrant_rate_start),
        immigrant_rate_end=float(immigrant_rate_end),
        exploration_power=float(exploration_power),
        archive_diversity_min_l1=float(archive_diversity_min_l1),
        archive_diversity_check_top_k=int(archive_diversity_check_top_k),
    )

    print("\n=== Running Genetic Algorithm Portfolio Search ===")
    print(f"[ga] params={ga_params}")

    ga_pop, ga_archive, ga_exploration_diagnostics = evolve_portfolios_ga(
        returns=returns_wide,
        universe=universe,
        lw_cov=None,
        equity0=float(equity0),
        notional=float(notional),
        goals=GOALS,
        main_goal=MAIN_GOAL,
        score_config=score_cfg,
        return_archive=True,
        return_diagnostics=True,
        weight_mode="long_short",
        **ga_params,
    )

    if not ga_pop:
        raise RuntimeError("GA returned empty population.")

    best_ga = ga_pop[0]
    st_ranked = []

    # ---------- Stability rerank ----------
    if skip_stability_rerank:
        print("\n=== Stability rerank skipped ===")
        best_by_stability = best_ga
        best_st_rep = None
    else:
        topK = ga_archive[: max(1, int(stability_top_k))]
        if not topK:
            topK = ga_pop[:1]

        st_cfg = StabilityEnergyConfig(
            alpha_cdar=0.95,
            breach_dd=0.25,
            lambda_mdd=1.0,
            lambda_cdar=1.2,
            lambda_ttr=0.7,
            lambda_breach=1.5,
            lambda_underwater=0.5,
        )

        rng = np.random.default_rng(123)
        st_ranked = []

        for m in topK:
            rep = compute_stability_for_candidate(
                returns=returns_wide,
                weights=m.weights,
                equity0=float(equity0),
                notional=float(notional),
                goals=GOALS,
                days=252,
                n_paths=int(stability_n_paths),
                mc_seed=int(rng.integers(0, 2**31 - 1)),
                path_source="bootstrap",
                pca_k=5,
                block_size=(int(block_min), int(block_max)),
                stability_cfg=st_cfg,
                weight_mode="long_short",
            )
            st_ranked.append((m, rep))

        st_ranked.sort(key=lambda x: x[1].energy)
        best_by_stability, best_st_rep = st_ranked[0]

        print(f"\n=== Stability rerank top={len(topK)} ===")
        print(format_stability_report(best_st_rep, days=252, cfg=st_cfg))

    # ---------- Annealing refine ----------
    anneal_params = dict(
        max_assets=int(max_assets),
        min_assets=int(min_assets),
        n_steps=int(anneal_steps),
        temp_start=1.0,
        temp_end=0.05,
        n_paths_init=int(anneal_n_paths_init),
        n_paths_final=int(anneal_n_paths_final),
        path_source="pca",
        pca_k=5,
        block_size=(int(block_min), int(block_max)),
    )

    print("\n=== Running annealing refinement ===")
    print(f"[anneal] params={anneal_params}")

    best_refined = refine_portfolio_annealing(
        base_metrics=best_by_stability,
        returns=returns_wide,
        universe=universe,
        lw_cov=None,
        equity0=float(equity0),
        notional=float(notional),
        goals=GOALS,
        main_goal=MAIN_GOAL,
        score_config=score_cfg,
        weight_mode="long_short",
        **anneal_params,
    )

    # ---------- Discretize into shares ----------
    # Price map is asset_id -> latest USD price.
    p2 = latest_prices_df[["asset_id", px_col]].copy()
    p2["asset_id"] = p2["asset_id"].astype(str).str.strip()
    p2[px_col] = pd.to_numeric(p2[px_col], errors="coerce")

    p2 = p2.dropna(subset=["asset_id", px_col])
    p2 = p2[p2["asset_id"] != ""].copy()
    p2 = p2[p2[px_col] > 0].copy()
    p2 = p2.drop_duplicates(subset=["asset_id"], keep="last")

    prices_asset_id = dict(zip(p2["asset_id"], p2[px_col]))

    w = dict(best_refined.weights)
    missing = [aid for aid, wt in w.items() if abs(float(wt)) >= 0.02 and aid not in prices_asset_id]
    if missing:
        raise RuntimeError(
            f"Missing prices for {len(missing)} asset_ids with abs(weight)>=2% "
            f"(sample={missing[:10]}). Price-map coverage bug upstream."
        )

    # ---------- Execution-aware final selection ----------
    # Batch 2B: do not select only the final annealed theoretical portfolio.
    # Evaluate multiple promising theoretical candidates after discrete rounding,
    # validate each executable portfolio, and choose the best accepted executable.
    missing_by_candidate = []
    candidate_pool = _build_executable_candidate_pool(
        best_ga=best_ga,
        best_by_stability=best_by_stability,
        best_refined=best_refined,
        ga_archive=ga_archive,
        stability_ranked=st_ranked,
        executable_selection_top_k=int(executable_selection_top_k),
    )

    print("\n=== Execution-aware candidate selection ===")
    print(f"[executable] candidate_pool={len(candidate_pool)} top_k={int(executable_selection_top_k)}")

    executable_candidates: list[dict] = []
    rng_exec = np.random.default_rng(987654)

    for label, theoretical_candidate in candidate_pool:
        w_cand = dict(theoretical_candidate.weights)
        missing = [aid for aid, wt in w_cand.items() if abs(float(wt)) >= 0.02 and aid not in prices_asset_id]
        if missing:
            missing_by_candidate.append({"label": str(label), "missing": missing[:10], "n_missing": len(missing)})
            continue

        try:
            executable_candidates.append(
                _evaluate_executable_candidate(
                    label=str(label),
                    theoretical_metrics=theoretical_candidate,
                    returns=returns_wide,
                    prices_asset_id=prices_asset_id,
                    equity0=float(equity0),
                    target_notional=float(notional),
                    goals=GOALS,
                    main_goal=float(MAIN_GOAL),
                    score_cfg=score_cfg,
                    n_paths=int(n_paths_final),
                    pca_k=int(pca_k),
                    block_size=(int(block_min), int(block_max)),
                    mc_seed=int(rng_exec.integers(0, 2**31 - 1)),
                    min_health_score=float(min_health_score),
                    min_executable_score=min_executable_score,
                    max_score_drop=max_score_drop,
                    max_ruin_increase=float(max_ruin_increase),
                    max_p_main_drop=float(max_p_main_drop),
                    max_cash_weight=float(max_cash_weight),
                    min_deployment_ratio=float(min_deployment_ratio),
                    max_executable_mdd=float(max_executable_mdd),
                    max_executable_cdar_95=float(max_executable_cdar_95),
                    max_stability_energy=float(max_stability_energy),
                    max_dropped_weight=float(max_dropped_weight),
                    max_weight_drift_l1=float(max_weight_drift_l1),
                )
            )
        except Exception as exc:
            missing_by_candidate.append(
                {
                    "label": str(label),
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_missing": None,
                    "missing": [],
                }
            )

    if not executable_candidates:
        raise RuntimeError(
            "No executable candidates could be evaluated after rounding. "
            f"missing_by_candidate={missing_by_candidate[:5]}"
        )

    accepted_candidates = [c for c in executable_candidates if bool(c["validation"].get("passed", False))]
    if accepted_candidates:
        selected_exec = max(accepted_candidates, key=lambda c: float(c["final_metrics"].score))
    else:
        # No official selection, but keep the best rejected candidate for diagnostics.
        selected_exec = max(executable_candidates, key=lambda c: float(c["final_metrics"].score))

    selected_theoretical = selected_exec["theoretical_metrics"]
    final_executable = selected_exec["final_metrics"]
    final_weights = selected_exec["final_weights"]
    final_gross_notional = float(selected_exec["final_gross_notional"])
    final_exposures = selected_exec["final_exposures"]
    alloc = selected_exec["alloc"]
    alloc_shares = selected_exec.get("alloc_shares")
    if alloc_shares is None:
        alloc_shares = {k: float(v) for k, v in (alloc.shares or {}).items()}

    alloc_realized_weights = selected_exec.get("alloc_realized_weights")
    if alloc_realized_weights is None:
        alloc_realized_weights = {k: float(v) for k, v in (alloc.realized_weights or {}).items()}
    execution_quality = selected_exec["execution_quality"]
    final_validation = selected_exec["validation"]
    selected_candidate_label = str(selected_exec["label"])
    run_id = f"{run_dt_ts.strftime('%Y%m%d')}-{pd.Timestamp.utcnow().strftime('%H%M%S')}"

    allocation_asset_ids = (
        set(str(k).strip() for k in (final_weights or {}).keys())
        | set(str(k).strip() for k in (alloc_shares or {}).keys())
        | set(str(k).strip() for k in (final_exposures or {}).keys())
    )
    allocation_asset_ids = {k for k in allocation_asset_ids if k and k.upper() != "CASH"}

    final_holdings = []
    for aid in sorted(allocation_asset_ids):
        disp = asset_display.get(aid, {
            "asset_id": aid,
            "ticker": aid,
            "yahoo_ticker": aid,
            "yahoo_ticker_norm": aid,
            "name": aid,
            "display_symbol": aid,
        })
        final_holdings.append({
            **disp,
            "weight": float((final_weights or {}).get(aid, 0.0) or 0.0),
            "shares": float((alloc_shares or {}).get(aid, 0.0) or 0.0),
            "exposure_usd": float((final_exposures or {}).get(aid, 0.0) or 0.0),
            "price_usd": float(prices_asset_id.get(aid, np.nan)),
        })

    # ---------- Actuarial diagnostics for selected executable ----------
    actuarial_diagnostic_block = None
    actuarial_text = None

    try:
        actuarial_eval = evaluate_portfolio_candidate_with_paths(
            returns=returns_wide,
            weights=final_weights,
            equity0=float(equity0),
            notional=float(final_gross_notional),
            goals=list(GOALS),
            main_goal=float(MAIN_GOAL),
            lw_cov=None,
            days=252,
            n_paths=int(n_paths_final),
            score_config=score_cfg,
            mc_seed=24681357,
            path_source="bootstrap",
            pca_k=int(pca_k),
            block_size=(int(block_min), int(block_max)),
            weight_mode="gross_signed",
        )

        final_actuarial_paths = actuarial_eval.equity_paths
        if final_actuarial_paths is None:
            raise RuntimeError("Diagnostic evaluator did not return equity_paths.")

        actuarial_horizon_days = int(np.asarray(final_actuarial_paths).shape[1] - 1)

        actuarial_config = ActuarialRiskConfig(
            initial_value=float(equity0),
            horizon_days=int(actuarial_horizon_days),
            ruin=RuinConfig(
                threshold_mode="fraction_of_initial",
                threshold_value=0.50,
            ),
            drawdown=DrawdownBreachConfig(
                drawdown_limit_pct=0.30,
            ),
            goal=GoalConfig(
                enabled=True,
                goal_value=float(MAIN_GOAL),
            ),
            recovery=RecoveryConfig(
                enabled=True,
                recovery_level=1.0,
            ),
            survival=SurvivalConfig(
                horizons_days=[
                    h for h in [21, 63, 126, 252, 756]
                    if h <= int(actuarial_horizon_days)
                ]
                or [int(actuarial_horizon_days)],
            ),
            capital_adequacy=CapitalAdequacyConfig(
                enabled=True,
                target_ruin_probability=0.05,
                target_drawdown_breach_probability=0.20,
                current_leverage=float(target_leverage),
                max_allowed_leverage=float(actuarial_max_allowed_leverage),
            ),
            metadata={
                "source": "run_portfolio_search",
                "run_id": run_id,
                "selected_candidate_label": selected_candidate_label,
                "actuarial_max_allowed_leverage": float(actuarial_max_allowed_leverage),
                "diagnostic_mc_seed": 24681357,
                "diagnostic_note": (
                    "Actuarial diagnostics are informational only and do not affect "
                    "portfolio-search scoring or executable validation."
                ),
            },
        )

        selected_portfolio_actuarial_input = {
            "portfolio_id": selected_candidate_label,
            "run_id": run_id,
            "equity_paths": final_actuarial_paths,
        }

        _actuarial_report, actuarial_text, actuarial_diagnostic_block = (
            build_portfolio_search_actuarial_diagnostic_section(
                selected_portfolio_actuarial_input,
                config=actuarial_config,
                equity_paths_key="equity_paths",
                portfolio_id=str(selected_candidate_label),
                run_id=str(run_id),
            )
        )

    except Exception as e:
        actuarial_diagnostic_block = {
            "status": "failed",
            "error": str(e),
            "source": "portfolio_search",
            "run_id": str(run_id),
            "selected_candidate_label": str(selected_candidate_label),
        }
        actuarial_text = (
            "ACTUARIAL RISK DIAGNOSTICS\n"
            "--------------------------\n"
            "Status: FAILED\n"
            f"Reason: {e}"
        )



    executable_candidate_summaries = [
        _candidate_summary(
            label=str(c["label"]),
            theoretical_metrics=c["theoretical_metrics"],
            final_metrics=c["final_metrics"],
            validation=c["validation"],
            execution_quality=c["execution_quality"],
        )
        for c in executable_candidates
    ]
    executable_selection_payload = _build_executable_selection_payload(
        executable_candidate_summaries=executable_candidate_summaries,
        selected_candidate_label=selected_candidate_label,
        final_validation=final_validation,
        candidate_errors=missing_by_candidate,
        executable_selection_top_k=int(executable_selection_top_k),
    )

    # ---------- Persist to S3 ----------
    last_score = s3_load_latest_report_score(s3, bucket=bucket, root_prefix=engine_root)

    if write_outputs:
        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=engine_root,
            table="portfolio_search/runs",
            dt=run_dt_ts,
            filename=f"run_{run_id}.json",
            payload={
                "run_id": run_id,
                "as_of": as_of_market_date,
                "meta": {
                    "env": getattr(cfg, "env", None),
                    "as_of_market_date": as_of_market_date,
                    "as_of_run_date": as_of_run_date,
                    "hmm_snapshot_as_of": as_of_market_date,
                },
                "inputs": {
                    "equity0": float(equity0),
                    "target_leverage": float(target_leverage),
                    "target_notional": float(notional),
                    "current_positions_notional": float(current_gross_notional),
                    "current_leverage_real": (
                        float(current_gross_notional / float(equity0))
                        if float(equity0) > 0
                        else float("inf")
                    ),
                    "goals": list(GOALS),
                    "main_goal": MAIN_GOAL,
                    "last_daily_report_score": last_score,
                    "positions": {t: asdict(p) for t, p in positions.items()},
                    "score_config": asdict(score_cfg),
                    "universe_size": len(universe),
                    "universe_key": "asset_id",
                    "display_key": "yahoo_ticker_norm",
                    "returns_clean_diag": diag,
                    "regime": lev_rec,
                    "market_data": {
                        "bucket": bucket,
                        "market_root": market_root,
                        "returns_cache": (
                            f"{returns_cache_prefix(cfg)}/"
                            f"returns_wide_min{int(float(cache_min_years))}y.parquet"
                        ),
                        "latest_prices_snapshot": latest_prices_snapshot_ref(cfg),
                    },
                },
                "params": {
                    "ga": ga_params,
                    "anneal": anneal_params,
                    "skip_stability_rerank": bool(skip_stability_rerank),
                    "stability_top_k": int(stability_top_k),
                    "stability_n_paths": int(stability_n_paths),
                    "executable_selection_top_k": int(executable_selection_top_k),
                },
                "outputs": {
                    "candidate_context": {
                        "equity0": float(equity0),
                        "target_leverage": float(target_leverage),
                        "target_notional": float(notional),
                        "goals": list(GOALS),
                        "main_goal": float(MAIN_GOAL),
                        "score_config": asdict(score_cfg),
                        "weight_mode": "long_short",
                        "composition_key": "asset_id",
                        "display_key": "yahoo_ticker_norm",
                        "metric_engine": "optimizer_engine.evaluate_portfolio",
                        "return_basis": "gross_notional_signed_weights",
                        "mc_basis": "fixed_gross_notional_on_equity",
                        "data_quality_gate": dq_gate_meta,
                    },
                    "best_ga": asdict(best_ga),
                    "ga_exploration_diagnostics": ga_exploration_diagnostics,
                    "best_by_stability": None if best_st_rep is None else asdict(best_by_stability),
                    "best_refined_theoretical": asdict(best_refined),
                    "selected_theoretical": {
                        "label": selected_candidate_label,
                        "metrics": asdict(selected_theoretical),
                    },
                    "executable_selection": executable_selection_payload,
                    # Backward-compatible aliases for early Batch 2B consumers.
                    "executable_candidates": executable_candidate_summaries,
                    "executable_candidate_errors": missing_by_candidate,
                    "final_executable": {
                        "selected_candidate_label": selected_candidate_label,
                        "status": str(final_validation["status"]),
                        "validation": final_validation,
                        "health": final_validation.get("health", {}),
                        "health_score": float(final_validation.get("health_score", np.nan)),
                        "health_grade": str(final_validation.get("health_grade", "n/a")),
                        "execution_quality": execution_quality,
                        "metrics": asdict(final_executable),
                        "weights": {k: float(v) for k, v in final_weights.items()},
                        "gross_notional": float(final_gross_notional),
                        "exposures": {k: float(v) for k, v in final_exposures.items()},
                        "holdings": final_holdings,
                        "composition_key": "asset_id",
                        "display_key": "yahoo_ticker_norm",
                    },
                    "actuarial_diagnostics": actuarial_diagnostic_block,
                    "discrete_allocation": {
                        "cash_left": float(alloc.cash_left),
                        "shares": {k: float(v) for k, v in alloc_shares.items()},
                        "realized_weights": {k: float(v) for k, v in alloc_realized_weights.items()},
                        "holdings": final_holdings,
                    },
                },
            },
        )

        top_n = min(50, len(ga_pop))
        df_top = pd.DataFrame([evalmetrics_to_row(m) for m in ga_pop[:top_n]])
        df_top.insert(0, "run_id", run_id)

        s3_write_parquet_partition(
            s3,
            bucket=bucket,
            root_prefix=engine_root,
            table="portfolio_search/candidates",
            dt=run_dt_ts,
            filename=f"top_{top_n}_{run_id}.parquet",
            df=df_top,
        )

        rows = []
        rows += weights_to_rows(best_refined.weights, tag="best_refined_theoretical_weights")
        rows += weights_to_rows(selected_theoretical.weights, tag="selected_theoretical_weights")
        rows += weights_to_rows(final_weights, tag="final_executable_weights")
        rows += weights_to_rows(alloc_realized_weights, tag="realized_weights_with_cash")
        df_w = pd.DataFrame(rows)
        df_w.insert(0, "run_id", run_id)

        s3_write_parquet_partition(
            s3,
            bucket=bucket,
            root_prefix=engine_root,
            table="portfolio_search/weights",
            dt=run_dt_ts,
            filename=f"weights_{run_id}.parquet",
            df=df_w,
        )

    # ---------- Print ----------
    print("\n" + "=" * 72)
    print("PORTFOLIO SEARCH RESULT")
    print("=" * 72)

    print_executable_candidate_summary(executable_candidates)
    if missing_by_candidate:
        print(f"\n[executable][warn] skipped_or_failed_candidates={len(missing_by_candidate)} sample={missing_by_candidate[:3]}")

    print("\nSelection chain")
    print("-" * 72)
    print(f"{'Stage':<32} {'Score':>10} {'P(main)':>10} {'Ruin':>10}")
    print("-" * 72)
    print(
        f"{'GA best':<32} "
        f"{'n/a':>8} "
        f"{_fmt_float(best_ga.score, 4):>10} "
        f"{_fmt_pct(_metric_p_main(best_ga, GOALS, MAIN_GOAL)):>10} "
        f"{_fmt_pct(best_ga.ruin_prob_1y):>10}"
    )
    print(
        f"{'After annealing theoretical':<32} "
        f"{'n/a':>8} "
        f"{_fmt_float(best_refined.score, 4):>10} "
        f"{_fmt_pct(_metric_p_main(best_refined, GOALS, MAIN_GOAL)):>10} "
        f"{_fmt_pct(best_refined.ruin_prob_1y):>10}"
    )
    print(
        f"{('Selected theoretical: ' + selected_candidate_label):<32.32} "
        f"{'n/a':>8} "
        f"{_fmt_float(selected_theoretical.score, 4):>10} "
        f"{_fmt_pct(_metric_p_main(selected_theoretical, GOALS, MAIN_GOAL)):>10} "
        f"{_fmt_pct(selected_theoretical.ruin_prob_1y):>10}"
    )
    print(
        f"{'Selected executable rounded':<32} "
        f"{_fmt_float(final_validation.get('health_score', np.nan), 1):>8} "
        f"{_fmt_float(final_executable.score, 4):>10} "
        f"{_fmt_pct(_metric_p_main(final_executable, GOALS, MAIN_GOAL)):>10} "
        f"{_fmt_pct(final_executable.ruin_prob_1y):>10}"
    )

    print_weights_table(best_ga.weights, title="GA best weights", asset_display=asset_display)
    print_weights_table(best_refined.weights, title="Annealing theoretical weights", asset_display=asset_display)
    print_weights_table(
        selected_theoretical.weights,
        title=f"Selected theoretical weights ({selected_candidate_label})",
        asset_display=asset_display,
    )
    print_weights_table(
        final_weights,
        title="Selected executable weights after rounding",
        asset_display=asset_display,
    )

    print_execution_quality(execution_quality)
    print_rounding_impact_table(
        execution_quality.get("rounding_impact", []),
        asset_display=asset_display,
    )

    print("\n=== Final executable allocation ===")
    print(f"{'Target notional':<32} {_fmt_money(notional):>16} USD")
    print(f"{'Executable gross notional':<32} {_fmt_money(final_gross_notional):>16} USD")
    print(f"{'Cash left':<32} {_fmt_money(alloc.cash_left):>16} USD")
    print(f"{'Deployment ratio':<32} {_fmt_pct(final_gross_notional / notional if notional else np.nan):>16}")

    print_shares_table(
        {k: float(v) for k, v in alloc_shares.items()},
        title="Final executable shares / units",
        asset_display=asset_display,
    )

    print_weights_table(
        {k: float(v) for k, v in alloc_realized_weights.items()},
        title="Realized weights including cash",
        include_zero=False,
        asset_display=asset_display,
    )

    print("\n=== Portfolio Health Score ===")
    print(f"Health score: {_fmt_float(final_validation.get('health_score'), 1)} / 100 ({final_validation.get('health_grade', 'n/a')})")
    print(f"Raw optimizer score: {_fmt_float(final_executable.score, 4)}")
    print("Note: raw optimizer score is used for ranking only; health score is used for validation/reporting.")

    print_portfolio_metrics(
        final_executable,
        goals=GOALS,
        title="Selected Executable Portfolio Metrics",
        show_weights=False,
    )

    print_executable_validation(final_validation)

    if actuarial_text:
        maybe_print_actuarial_diagnostic_section(
            actuarial_text,
            enabled=True,
        )
    
    print("\n" + "=" * 72)
    
    if final_validation.get("passed"):
        print(f"Official selected portfolio: ACCEPTED FINAL EXECUTABLE ({selected_candidate_label})")
    else:
        print("Official selected portfolio: NONE — FINAL EXECUTABLE FAILED VALIDATION")
    print("=" * 72)

    if write_outputs:
        print(f"\n[S3] Saved portfolio search run_id={run_id}")

    return {
        "run_id": run_id,
        "as_of": as_of_market_date,
        "as_of_run_date": as_of_run_date,
        "equity0": float(equity0),
        "target_leverage": float(target_leverage),
        "target_notional": float(notional),
        "regime": lev_rec,
        "returns_diag": diag,
        "universe_size": int(len(universe)),
        "best_ga": asdict(best_ga),
        "ga_exploration_diagnostics": ga_exploration_diagnostics,
        "best_by_stability": None if best_st_rep is None else asdict(best_by_stability),
        "best_refined_theoretical": asdict(best_refined),
        "selected_theoretical": {
            "label": selected_candidate_label,
            "metrics": asdict(selected_theoretical),
        },
        "executable_selection": executable_selection_payload,
        # Backward-compatible aliases for early Batch 2B consumers.
        "executable_candidates": executable_candidate_summaries,
        "executable_candidate_errors": missing_by_candidate,
        "final_executable": {
            "selected_candidate_label": selected_candidate_label,
            "status": str(final_validation["status"]),
            "validation": final_validation,
            "health": final_validation.get("health", {}),
            "health_score": float(final_validation.get("health_score", np.nan)),
            "health_grade": str(final_validation.get("health_grade", "n/a")),
            "execution_quality": execution_quality,
            "metrics": asdict(final_executable),
            "weights": {k: float(v) for k, v in final_weights.items()},
            "gross_notional": float(final_gross_notional),
            "exposures": {k: float(v) for k, v in final_exposures.items()},
            "holdings": final_holdings,
            "composition_key": "asset_id",
            "display_key": "yahoo_ticker_norm",
        },
        "actuarial_diagnostics": actuarial_diagnostic_block,
        "discrete_allocation": {
            "cash_left": float(alloc.cash_left),
            "shares": {k: float(v) for k, v in alloc_shares.items()},
            "realized_weights": {k: float(v) for k, v in alloc_realized_weights.items()},
            "holdings": final_holdings,
        },
        "params": {
            "ga": ga_params,
            "anneal": anneal_params,
            "skip_stability_rerank": bool(skip_stability_rerank),
            "stability_top_k": int(stability_top_k),
            "stability_n_paths": int(stability_n_paths),
            "executable_selection_top_k": int(executable_selection_top_k),
        },
        "runtime": {
            "env": getattr(cfg, "env", None),
            "bucket": bucket,
            "region": region,
            "engine_root": engine_root,
            "market_root": market_root,
        },
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Alpha Edge portfolio search.")

    ap.add_argument("--as-of", default=None, help="Market as-of date YYYY-MM-DD. Default: today.")
    ap.add_argument("--run-dt", default=None, help="Run partition date YYYY-MM-DD. Default: today.")

    ap.add_argument("--equity0", type=float)
    ap.add_argument(
        "--goals",
        default="10000,12500,15000",
        help="Comma-separated 3-goal ladder, e.g. 10000,12500,15000.",
    )
    ap.add_argument("--main-goal", type=float, default=10000.0)

    ap.add_argument("--universe-csv", default=None)
    ap.add_argument("--cache-min-years", type=float, default=5.0)
    ap.add_argument(
        "--rebuild-returns-cache",
        action="store_true",
        help=(
            "Rebuild returns_wide cache before running search. "
            "Use only after build_returns_wide_cache is runtime-aware."
        ),
    )
    ap.add_argument("--no-market-hmm", action="store_true")
    ap.add_argument("--override-target-leverage", type=float, default=None)

    ap.add_argument("--no-write", action="store_true", help="Run search but do not write outputs.")
    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    # Smoke-test / workload knobs
    ap.add_argument("--pop-size", type=int, default=80)
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--elite-frac", type=float, default=0.10)
    ap.add_argument("--max-assets", type=int, default=10)
    ap.add_argument("--min-assets", type=int, default=5)
    ap.add_argument("--n-paths-init", type=int, default=5000)
    ap.add_argument("--n-paths-final", type=int, default=20000)
    ap.add_argument("--pca-k", type=int, default=3)
    ap.add_argument("--block-min", type=int, default=8)
    ap.add_argument("--block-max", type=int, default=12)

    # Milestone 12: universe exploration / mutation diversity knobs
    ap.add_argument("--mutation-sigma-start", type=float, default=0.30)
    ap.add_argument("--mutation-sigma-end", type=float, default=0.05)
    ap.add_argument("--replace-prob-start", type=float, default=0.40)
    ap.add_argument("--replace-prob-end", type=float, default=0.05)
    ap.add_argument("--immigrant-rate-start", type=float, default=0.20)
    ap.add_argument("--immigrant-rate-end", type=float, default=0.03)
    ap.add_argument("--exploration-power", type=float, default=1.5)
    ap.add_argument("--archive-diversity-min-l1", type=float, default=0.15)
    ap.add_argument("--archive-diversity-check-top-k", type=int, default=250)

    ap.add_argument("--skip-stability-rerank", action="store_true")
    ap.add_argument("--stability-top-k", type=int, default=200)
    ap.add_argument("--stability-n-paths", type=int, default=20000)
    ap.add_argument(
        "--executable-selection-top-k",
        type=int,
        default=25,
        help="Number of stability-ranked/archive candidates to discretize, re-evaluate, and validate before final selection.",
    )

    ap.add_argument("--anneal-steps", type=int, default=200)
    ap.add_argument("--anneal-n-paths-init", type=int, default=3000)
    ap.add_argument("--anneal-n-paths-final", type=int, default=20000)

    ap.add_argument(
        "--min-universe-size",
        type=int,
        default=10,
        help="Minimum number of usable assets after returns cleaning. Lower this for dev smoke tests.",
    )

    # Final executable validation knobs
    ap.add_argument("--min-health-score", type=float, default=60.0, help="Minimum normalized 0-100 health score required for final executable acceptance.")
    ap.add_argument("--min-executable-score", type=float, default=None, help="Optional raw optimizer-score floor. Disabled by default because optimizer score scale changes with tuned lambdas.")
    ap.add_argument("--max-score-drop", type=float, default=None, help="Optional raw optimizer-score degradation cap. Disabled by default because optimizer score scale changes with tuned lambdas.")
    ap.add_argument("--max-ruin-increase", type=float, default=0.03)
    ap.add_argument("--max-p-main-drop", type=float, default=0.15)
    ap.add_argument("--max-cash-weight", type=float, default=0.05)
    ap.add_argument("--min-deployment-ratio", type=float, default=0.95)
    ap.add_argument("--max-executable-mdd", type=float, default=0.40)
    ap.add_argument("--max-executable-cdar-95", type=float, default=0.60)
    ap.add_argument("--max-stability-energy", type=float, default=2.00)
    ap.add_argument("--max-dropped-weight", type=float, default=0.04)
    ap.add_argument("--max-weight-drift-l1", type=float, default=0.15)
    ap.add_argument(
        "--actuarial-max-allowed-leverage",
        type=float,
        default=2.0,
        help="Policy cap used by actuarial safe-leverage diagnostics.",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    cfg = load_runtime_config(args.env)
    write_outputs = not bool(args.no_write)

    if write_outputs:
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    run_dt = (
        pd.Timestamp(args.run_dt).tz_localize(None).normalize()
        if args.run_dt
        else pd.Timestamp(dt.date.today()).normalize()
    )
    as_of = args.as_of or run_dt.strftime("%Y-%m-%d")

    run_portfolio_search_asof(
        as_of=as_of,
        equity0=float(args.equity0),
        goals=_parse_goals(args.goals),
        main_goal=float(args.main_goal),
        universe_csv=args.universe_csv,
        use_market_hmm=(not bool(args.no_market_hmm)),
        override_target_leverage=args.override_target_leverage,
        write_outputs=write_outputs,
        run_dt=run_dt,
        cache_min_years=float(args.cache_min_years),
        cfg=cfg,
        confirm_prod_write=bool(args.confirm_prod_write),
        pop_size=int(args.pop_size),
        generations=int(args.generations),
        elite_frac=float(args.elite_frac),
        max_assets=int(args.max_assets),
        min_assets=int(args.min_assets),
        n_paths_init=int(args.n_paths_init),
        n_paths_final=int(args.n_paths_final),
        pca_k=int(args.pca_k),
        block_min=int(args.block_min),
        block_max=int(args.block_max),
        mutation_sigma_start=float(args.mutation_sigma_start),
        mutation_sigma_end=float(args.mutation_sigma_end),
        replace_prob_start=float(args.replace_prob_start),
        replace_prob_end=float(args.replace_prob_end),
        immigrant_rate_start=float(args.immigrant_rate_start),
        immigrant_rate_end=float(args.immigrant_rate_end),
        exploration_power=float(args.exploration_power),
        archive_diversity_min_l1=float(args.archive_diversity_min_l1),
        archive_diversity_check_top_k=int(args.archive_diversity_check_top_k),
        skip_stability_rerank=bool(args.skip_stability_rerank),
        stability_top_k=int(args.stability_top_k),
        stability_n_paths=int(args.stability_n_paths),
        executable_selection_top_k=int(args.executable_selection_top_k),
        anneal_steps=int(args.anneal_steps),
        anneal_n_paths_init=int(args.anneal_n_paths_init),
        anneal_n_paths_final=int(args.anneal_n_paths_final),
        min_universe_size=int(args.min_universe_size),
        rebuild_returns_cache=bool(args.rebuild_returns_cache),
        min_health_score=float(args.min_health_score),
        min_executable_score=args.min_executable_score,
        max_score_drop=args.max_score_drop,
        max_ruin_increase=float(args.max_ruin_increase),
        max_p_main_drop=float(args.max_p_main_drop),
        max_cash_weight=float(args.max_cash_weight),
        min_deployment_ratio=float(args.min_deployment_ratio),
        max_executable_mdd=float(args.max_executable_mdd),
        max_executable_cdar_95=float(args.max_executable_cdar_95),
        max_stability_energy=float(args.max_stability_energy),
        max_dropped_weight=float(args.max_dropped_weight),
        max_weight_drift_l1=float(args.max_weight_drift_l1),
        actuarial_max_allowed_leverage=float(args.actuarial_max_allowed_leverage),
    )

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
        script_name="run_portfolio_search.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            main()

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="portfolio_search",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="run_portfolio_search.py",
                source_mode="portfolio_search",
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
                entity_type="portfolio_search",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="run_portfolio_search.py",
                source_mode="portfolio_search",
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


if __name__ == "__main__":
    from multiprocessing import freeze_support
    
    freeze_support()
    main_with_audit()
