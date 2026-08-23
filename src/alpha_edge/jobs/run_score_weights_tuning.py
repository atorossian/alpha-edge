# run_score_weights_tuning.py
# S3-only I/O; universe filter matches portfolio search.
from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run

import argparse
import datetime as dt
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from alpha_edge import paths
from alpha_edge.core.data_loader import (
    clean_returns_matrix,
    parse_positions_obj,
    s3_init,
    s3_load_latest_json,
    s3_write_json_event,
)
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.portfolio.equity_valuation import resolve_current_equity, print_equity_valuation_result
from alpha_edge.core.schemas import RuntimeConfig
from alpha_edge.core.market_store import MarketStore
from alpha_edge.market.regime_leverage import leverage_from_hmm
from alpha_edge.portfolio.execution_engine import weights_to_discrete_shares
from alpha_edge.portfolio.portfolio_search import evolve_portfolios_ga
from alpha_edge.tuning.tune_score_weights_optimize import tune_lambdas_by_optimization
from alpha_edge.universe.universe import load_universe


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT = "engine/v1"
DEFAULT_MARKET_ROOT = "market"


def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET)).strip()


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION)).strip()


def cfg_engine_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "engine_root", DEFAULT_ENGINE_ROOT)).strip("/")


def cfg_market_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "market_root", DEFAULT_MARKET_ROOT)).strip("/")


def returns_wide_cache_path(cfg: RuntimeConfig, *, min_years: int = 5) -> str:
    return f"s3://{cfg_bucket(cfg)}/{cfg_market_root(cfg)}/cache/v1/returns_wide_min{int(min_years)}y.parquet"


def _safe_conf_str(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.2f}"

def _parse_goals(s: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in str(s).split(",") if str(x).strip()]
    if len(parts) != 3:
        raise ValueError(f"--goals must contain exactly 3 comma-separated numbers. Got {s!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))



def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if np.isfinite(v) else float(default)


def _realized_weights_from_shares(
    shares: dict[str, float],
    prices: dict[str, float],
) -> tuple[dict[str, float], float, dict[str, float]]:
    exposures: dict[str, float] = {}
    gross = 0.0

    for t, q in (shares or {}).items():
        tt = str(t).upper().strip()
        if not tt or tt == "CASH":
            continue
        px = prices.get(tt)
        if px is None:
            continue
        qf = _safe_float(q)
        pxf = _safe_float(px)
        if not np.isfinite(qf) or not np.isfinite(pxf) or pxf <= 0 or abs(qf) <= 0:
            continue
        exp = qf * pxf
        if abs(exp) <= 0:
            continue
        exposures[tt] = float(exp)
        gross += abs(float(exp))

    if gross <= 0 or not np.isfinite(gross):
        raise RuntimeError("Executable candidate produced zero gross notional.")

    weights = {t: float(v / gross) for t, v in exposures.items()}
    return weights, float(gross), exposures


def _execution_quality(
    *,
    theoretical_weights: dict[str, float],
    executable_weights: dict[str, float],
    target_notional: float,
    executable_gross_notional: float,
    cash_left: float,
) -> dict[str, float]:
    tickers = set(str(t).upper().strip() for t in theoretical_weights) | set(str(t).upper().strip() for t in executable_weights)
    l1 = 0.0
    dropped = 0.0
    for t in tickers:
        tw = _safe_float(theoretical_weights.get(t, 0.0), 0.0)
        ew = _safe_float(executable_weights.get(t, 0.0), 0.0)
        l1 += abs(tw - ew)
        if abs(tw) > 1e-8 and abs(ew) <= 1e-8:
            dropped += abs(tw)

    target = float(target_notional)
    gross = float(executable_gross_notional)
    cash = float(cash_left)
    return {
        "deployment_ratio": float(gross / target) if target > 0 else float("nan"),
        "cash_weight": float(cash / target) if target > 0 else float("nan"),
        "weight_drift_l1": float(l1),
        "dropped_theoretical_weight": float(dropped),
    }


def _build_prices_ticker(
    *,
    cfg: RuntimeConfig,
    universe_df: pd.DataFrame,
) -> dict[str, float]:
    try:
        market = MarketStore(
            bucket=cfg_bucket(cfg),
            region=cfg_region(cfg),
            base_prefix=cfg_market_root(cfg),
        )
    except TypeError:
        market = MarketStore(bucket=cfg_bucket(cfg), region=cfg_region(cfg))

    latest_prices_df = market.read_latest_prices_snapshot()
    if latest_prices_df.empty:
        raise RuntimeError("Missing latest_prices snapshot; cannot build executable tuning candidates.")

    latest_prices_df = latest_prices_df.copy()
    latest_prices_df["asset_id"] = latest_prices_df["asset_id"].astype(str).str.strip()

    if "adj_close_usd" in latest_prices_df.columns:
        px_col = "adj_close_usd"
    elif "close_raw_usd" in latest_prices_df.columns:
        px_col = "close_raw_usd"
    elif "close_usd" in latest_prices_df.columns:
        px_col = "close_usd"
    else:
        raise RuntimeError(f"latest_prices snapshot missing usable price column. Columns={list(latest_prices_df.columns)}")

    u2 = universe_df[["asset_id", "ticker"]].copy()
    u2["asset_id"] = u2["asset_id"].astype(str).str.strip()
    u2["ticker"] = u2["ticker"].astype(str).str.upper().str.strip()

    p2 = latest_prices_df[["asset_id", px_col]].copy()
    p2[px_col] = pd.to_numeric(p2[px_col], errors="coerce")

    merged = u2.merge(p2, on="asset_id", how="left")
    merged = merged.dropna(subset=[px_col])
    merged = merged[merged[px_col] > 0]
    return {str(t).upper().strip(): float(px) for t, px in zip(merged["ticker"], merged[px_col])}


def _make_executable_candidate_pool(
    *,
    candidate_metrics: list[Any],
    prices_ticker: dict[str, float],
    notional: float,
    min_weight: float,
    min_units_equity: float,
    min_units_crypto: float,
    min_units_weight_thr: float,
    crypto_decimals: int,
    nearest_step_remaining_frac: float,
    min_deployment_ratio: float,
    max_cash_weight: float,
    max_weight_drift_l1: float,
    max_dropped_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    executable_pool: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for i, m in enumerate(candidate_metrics):
        label = f"ga_archive_{i}"
        theoretical_weights = {str(k).upper().strip(): float(v) for k, v in dict(m.weights).items()}
        try:
            alloc = weights_to_discrete_shares(
                weights=theoretical_weights,
                prices=prices_ticker,
                notional=float(notional),
                min_weight=float(min_weight),
                min_units_equity=float(min_units_equity),
                min_units_crypto=float(min_units_crypto),
                min_units_weight_thr=float(min_units_weight_thr),
                crypto_decimals=int(crypto_decimals),
                nearest_step_remaining_frac=float(nearest_step_remaining_frac),
            )
            final_weights, final_gross, final_exposures = _realized_weights_from_shares(
                {k: float(v) for k, v in alloc.shares.items()},
                prices_ticker,
            )
            quality = _execution_quality(
                theoretical_weights=theoretical_weights,
                executable_weights=final_weights,
                target_notional=float(notional),
                executable_gross_notional=float(final_gross),
                cash_left=float(alloc.cash_left),
            )

            reasons: list[str] = []
            if quality["deployment_ratio"] < float(min_deployment_ratio):
                reasons.append("deployment_ratio")
            if quality["cash_weight"] > float(max_cash_weight):
                reasons.append("cash_weight")
            if quality["weight_drift_l1"] > float(max_weight_drift_l1):
                reasons.append("weight_drift_l1")
            if quality["dropped_theoretical_weight"] > float(max_dropped_weight):
                reasons.append("dropped_theoretical_weight")

            row = {
                "label": label,
                "weights": {k: float(v) for k, v in final_weights.items()},
                "notional": float(final_gross),
                "weight_mode": "gross_signed",
                "source_weights": theoretical_weights,
                "source_score": float(m.score),
                "source_ruin": float(m.ruin_prob_1y),
                "execution_quality": quality,
                "shares": {k: float(v) for k, v in alloc.shares.items()},
                "cash_left": float(alloc.cash_left),
                "exposures": {k: float(v) for k, v in final_exposures.items()},
            }

            if reasons:
                rejected.append({"label": label, "reasons": reasons, "execution_quality": quality})
            else:
                executable_pool.append(row)
        except Exception as exc:
            errors.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "schema_version": "executable_tuning_pool_v1",
        "input_candidates": int(len(candidate_metrics)),
        "accepted_candidates": int(len(executable_pool)),
        "rejected_candidates": int(len(rejected)),
        "error_candidates": int(len(errors)),
        "filters": {
            "min_deployment_ratio": float(min_deployment_ratio),
            "max_cash_weight": float(max_cash_weight),
            "max_weight_drift_l1": float(max_weight_drift_l1),
            "max_dropped_weight": float(max_dropped_weight),
        },
        "rejected_sample": rejected[:25],
        "errors_sample": errors[:25],
    }
    return executable_pool, summary

def run_score_weights_tuning(
    *,
    cfg: RuntimeConfig,
    as_of: str | None = None,
    universe_csv: str | None = None,
    equity0: float = 934.13,
    goals: tuple[float, float, float] = (800.0, 1200.0, 2000.0),
    main_goal: float = 2000.0,
    override_target_leverage: float | None = 7.0,
    use_portfolio_hmm: bool = True,
    write_outputs: bool = True,
    update_latest: bool = True,
    confirm_prod_write: bool = False,
    cache_min_years: int = 5,
    min_universe_size: int = 10,
    max_nan_frac: float = 0.30,
    pop_size: int = 80,
    generations: int = 50,
    elite_frac: float = 0.2,
    max_assets: int = 10,
    min_assets: int = 5,
    weight_mode: str = "long_short",
    n_paths_init: int = 5000,
    n_paths_final: int = 20000,
    pca_k: int = 3,
    block_min: int = 8,
    block_max: int = 12,
    archive_limit: int = 50000,
    candidate_pool_size: int = 2000,
    n_trials: int = 40,
    pool_sample_size: int = 500,
    shortlist_size: int = 60,
    n_paths_train: int = 6000,
    n_paths_valid: int = 20000,
    ruin_cap: float = 0.25,
    alpha_ruin: float = 0.5,
    alpha_stability: float = 0.35,
    alpha_cdar: float = 0.25,
    alpha_path_mdd: float = 0.25,
    alpha_breach: float = 0.40,
    alpha_underwater: float = 0.10,
    alpha_ttr: float = 0.10,
    train_frac: float = 0.7,
    tune_on_executable_candidates: bool = True,
    executable_min_weight: float = 0.01,
    executable_min_units_equity: float = 1.0,
    executable_min_units_crypto: float = 0.0,
    executable_min_units_weight_thr: float = 0.03,
    executable_crypto_decimals: int = 8,
    executable_nearest_step_remaining_frac: float = 0.10,
    executable_min_deployment_ratio: float = 0.95,
    executable_max_cash_weight: float = 0.05,
    executable_max_weight_drift_l1: float = 0.20,
    executable_max_dropped_weight: float = 0.05,
) -> dict:
    if write_outputs:
        require_prod_confirmation(cfg, bool(confirm_prod_write))

    bucket = cfg_bucket(cfg)
    region = cfg_region(cfg)
    engine_root = cfg_engine_root(cfg)

    today = pd.Timestamp(as_of or dt.date.today()).tz_localize(None).normalize()
    as_of_date = today.strftime("%Y-%m-%d")

    s3 = s3_init(region)

    print("\n=== SCORE WEIGHTS TUNING ===")
    print(f"env:           {getattr(cfg, 'env', 'unknown')}")
    print(f"bucket:        {bucket}")
    print(f"region:        {region}")
    print(f"engine_root:   {engine_root}")
    print(f"as_of:         {as_of_date}")
    print(f"write_outputs: {bool(write_outputs)}")
    print("")

    # Universe is local-file based for now, like portfolio search.
    universe_path = universe_csv or str(paths.universe_dir() / "universe.csv")

    u_df = pd.read_csv(universe_path)
    if "include" in u_df.columns:
        u_df = u_df[u_df["include"].fillna(1).astype(int) == 1].copy()
    else:
        u_df = u_df.copy()

    for c in ["asset_id", "ticker"]:
        if c not in u_df.columns:
            raise RuntimeError(f"Universe CSV missing required column '{c}': {universe_path}")

    u_df["asset_id"] = u_df["asset_id"].astype(str).str.strip()
    u_df["ticker"] = u_df["ticker"].astype(str).str.upper().str.strip()

    asset_to_ticker = dict(zip(u_df["asset_id"], u_df["ticker"]))
    allowed_tickers = set(u_df["ticker"].tolist())

    universe_all = load_universe(universe_path)

    # ---------- Load latest positions ----------
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

    # ---------- Regime -> leverage -> notional ----------
    hmm_payload_wrapped = {}
    if use_portfolio_hmm:
        hmm_payload_wrapped = (
            s3_load_latest_json(
                s3,
                bucket=bucket,
                root_prefix=engine_root,
                table="regimes/hmm",
            )
            or {}
        )

    hmm_res = None
    if isinstance(hmm_payload_wrapped, dict):
        hmm_res = hmm_payload_wrapped.get("hmm") or hmm_payload_wrapped

    lev_rec = leverage_from_hmm(
        hmm_res or {},
        default=1.0,
        risk_appetite=0.6,
        low_confidence_floor=0.2,
        hard_cap=12.0,
    )

    if override_target_leverage is not None:
        target_leverage = float(override_target_leverage)
        lev_rec = {
            **dict(lev_rec or {}),
            "mode": "override",
            "leverage": target_leverage,
        }
    else:
        target_leverage = float(lev_rec.get("leverage", 1.0))

    notional = float(equity0) * float(target_leverage)
    if not np.isfinite(notional) or notional <= 0:
        raise RuntimeError(f"Invalid notional={notional} from equity0={equity0} lev={target_leverage}")

    print(
        f"[tuning] equity0={float(equity0):.2f} USD | "
        f"regime={lev_rec.get('chosen_label')} ({lev_rec.get('mode')}, conf={_safe_conf_str(lev_rec.get('confidence'))}) | "
        f"lev={target_leverage:.2f}x -> notional={notional:.2f}"
    )

    # ---------- Load returns wide cache ----------
    # ---------- Load returns wide cache ----------
    returns_path = returns_wide_cache_path(cfg, min_years=int(cache_min_years))

    try:
        returns_wide = pd.read_parquet(returns_path, engine="pyarrow").sort_index()
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Missing returns-wide cache: {returns_path}. "
            "Run dev market ingestion + returns cache build first, or run portfolio search once "
            "after patching it to build the dev cache from the sample universe."
        ) from e

    # Map asset_id columns -> tickers when needed.
    returns_wide = returns_wide.rename(
        columns=lambda c: asset_to_ticker.get(str(c).strip(), str(c).strip())
    )
    returns_wide.columns = [str(c).upper().strip() for c in returns_wide.columns]

    # Slice to <= as_of. This is critical to avoid lookahead.
    returns_wide.index = pd.to_datetime(returns_wide.index, errors="coerce").tz_localize(None).normalize()
    returns_wide = returns_wide.loc[returns_wide.index <= today]

    if returns_wide.shape[0] < 252:
        raise RuntimeError(
            f"Not enough returns history up to as_of={as_of_date}: rows={returns_wide.shape[0]}"
        )

    # IMPORTANT: match portfolio search criteria exactly.
    returns_wide, diag = clean_returns_matrix(
        returns_wide,
        min_history_days=252 * 2,
        max_nan_frac=float(max_nan_frac),
        min_vol=1e-6,
    )

    # Restrict universe exactly like portfolio search.
    universe = {
        str(t).upper().strip(): a
        for t, a in universe_all.items()
        if str(t).upper().strip() in allowed_tickers
        and str(t).upper().strip() in returns_wide.columns
    }

    if len(universe) < int(min_universe_size):
        raise RuntimeError(
            f"Universe too small after returns cleaning: {len(universe)}. "
            f"Required min_universe_size={int(min_universe_size)}. diag={diag}"
        )

    # ---------- GA archive for candidate pool ----------
    ga_params = {
        "pop_size": int(pop_size),
        "generations": int(generations),
        "elite_frac": float(elite_frac),
        "max_assets": int(max_assets),
        "min_assets": int(min_assets),
        "n_paths_init": int(n_paths_init),
        "n_paths_final": int(n_paths_final),
        "path_source": "bootstrap",
        "pca_k": int(pca_k),
        "block_size": (int(block_min), int(block_max)),
        "archive_limit": int(archive_limit),
    }

    print(f"[ga] params={ga_params}")
    print(f"[ga] weight_mode={str(weight_mode)}")

    ga_pop, ga_archive = evolve_portfolios_ga(
        returns=returns_wide,
        universe=universe,
        lw_cov=None,
        equity0=float(equity0),
        notional=float(notional),
        goals=goals,
        main_goal=float(main_goal),
        score_config=None,
        return_archive=True,
        weight_mode=str(weight_mode),
        **ga_params,
    )

    candidate_metrics = list(ga_archive[: int(candidate_pool_size)])
    if not candidate_metrics:
        raise RuntimeError("GA archive is empty; cannot tune lambdas.")

    executable_pool_summary: dict[str, Any] | None = None
    if bool(tune_on_executable_candidates):
        print("[tuning] building executable candidate pool from GA archive...")
        prices_ticker = _build_prices_ticker(cfg=cfg, universe_df=u_df)
        candidate_pool, executable_pool_summary = _make_executable_candidate_pool(
            candidate_metrics=candidate_metrics,
            prices_ticker=prices_ticker,
            notional=float(notional),
            min_weight=float(executable_min_weight),
            min_units_equity=float(executable_min_units_equity),
            min_units_crypto=float(executable_min_units_crypto),
            min_units_weight_thr=float(executable_min_units_weight_thr),
            crypto_decimals=int(executable_crypto_decimals),
            nearest_step_remaining_frac=float(executable_nearest_step_remaining_frac),
            min_deployment_ratio=float(executable_min_deployment_ratio),
            max_cash_weight=float(executable_max_cash_weight),
            max_weight_drift_l1=float(executable_max_weight_drift_l1),
            max_dropped_weight=float(executable_max_dropped_weight),
        )
        print(
            "[tuning] executable_pool "
            f"accepted={executable_pool_summary['accepted_candidates']} "
            f"rejected={executable_pool_summary['rejected_candidates']} "
            f"errors={executable_pool_summary['error_candidates']}"
        )
        if len(candidate_pool) < 50:
            raise RuntimeError(
                "Executable candidate pool too small after rounding/filtering: "
                f"{len(candidate_pool)}. Summary={executable_pool_summary}"
            )
    else:
        candidate_pool = [m.weights for m in candidate_metrics]
        executable_pool_summary = {
            "schema_version": "executable_tuning_pool_v1",
            "enabled": False,
            "input_candidates": int(len(candidate_metrics)),
            "accepted_candidates": int(len(candidate_pool)),
            "rejected_candidates": 0,
            "error_candidates": 0,
        }

    # ---------- Tune lambdas ----------
    tune_params = {
        "n_trials": int(n_trials),
        "pool_sample_size": int(pool_sample_size),
        "shortlist_size": int(shortlist_size),
        "n_paths_train": int(n_paths_train),
        "n_paths_valid": int(n_paths_valid),
        "ruin_cap": float(ruin_cap),
        "alpha_ruin": float(alpha_ruin),
        "alpha_stability": float(alpha_stability),
        "alpha_cdar": float(alpha_cdar),
        "alpha_path_mdd": float(alpha_path_mdd),
        "alpha_breach": float(alpha_breach),
        "alpha_underwater": float(alpha_underwater),
        "alpha_ttr": float(alpha_ttr),
        "train_frac": float(train_frac),
        "weight_mode": "gross_signed" if bool(tune_on_executable_candidates) else str(weight_mode),
    }

    print(f"[tuning] params={tune_params}")

    best_cfg, info = tune_lambdas_by_optimization(
        returns=returns_wide,
        lw_cov=None,
        candidate_pool=candidate_pool,
        equity0=float(equity0),
        notional=float(notional),
        goals=goals,
        main_goal=float(main_goal),
        **tune_params,
    )

    print(best_cfg)
    print(info)

    run_id = f"{today.strftime('%Y%m%d')}-{pd.Timestamp.utcnow().strftime('%H%M%S')}"

    payload = {
        "run_id": run_id,
        "as_of": as_of_date,
        "meta": {
            "env": getattr(cfg, "env", None),
            "bucket": bucket,
            "engine_root": engine_root,
            "market_root": cfg_market_root(cfg),
        },
        "inputs": {
            "equity0": float(equity0),
            "target_leverage": float(target_leverage),
            "notional": float(notional),
            "positions_n": len(positions),
            "universe_n": len(universe),
            "returns_cache": returns_path,
            "returns_clean_diag": diag,
            "goals": list(goals),
            "main_goal": float(main_goal),
            "returns_cache": returns_path,
            "cleaning": {
                "min_history_days": 252 * 2,
                "max_nan_frac": float(max_nan_frac),
                "min_vol": 1e-6,
            },
            "regime": lev_rec,
        },
        "params": {
            "ga": ga_params,
            "tuning": tune_params,
            "candidate_pool_size": int(candidate_pool_size),
            "actual_tuning_candidate_pool_size": int(len(candidate_pool)),
            "executable_pool_summary": executable_pool_summary,
            "cache_min_years": int(cache_min_years),
            "min_universe_size": int(min_universe_size),
        },
        "best_cfg": asdict(best_cfg),
        "info": info,
    }

    if write_outputs:
        # Persist tuned config to S3: append-only + latest pointer.
        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=engine_root,
            table="configs/score_config",
            dt=today,
            filename="score_config.json",
            payload=asdict(best_cfg),
            update_latest=update_latest,
        )

        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=engine_root,
            table="score_tuning/runs",
            dt=today,
            filename=f"tuning_{run_id}.json",
            payload=payload,
            update_latest=False,
        )

        print(f"\n[S3] Saved tuned score_config to s3://{bucket}/{engine_root}/configs/score_config/latest.json")
        print(f"[S3] Saved tuning run to s3://{bucket}/{engine_root}/score_tuning/runs/dt={as_of_date}/")
    else:
        print("\n[NO-WRITE] tuning completed but no S3 outputs were written.")

    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tune ScoreConfig lambda weights using GA candidate archive.")

    ap.add_argument("--as-of", default=None, help="Run/as-of date YYYY-MM-DD. Default: today.")
    ap.add_argument("--universe-csv", default=None)

    ap.add_argument("--equity0", "--equity-override", dest="equity0", type=float, default=None, help="Initial/current equity. If omitted, resolved from ledger + latest prices.")
    ap.add_argument(
        "--goals",
        default="800,1200,2000",
        help="Comma-separated 3-goal ladder, e.g. 800,1200,2000.",
    )
    ap.add_argument("--main-goal", type=float, default=2000.0)
    ap.add_argument("--cache-min-years", type=int, default=5)
    ap.add_argument("--min-universe-size", type=int, default=10)
    ap.add_argument(
        "--max-nan-frac",
        type=float,
        default=0.30,
        help=(
            "Maximum missing-return fraction allowed during tuning universe cleaning. "
            "Default 0.30 because the min5y cache can show ~23%-26% missing data "
            "from calendar alignment even for usable assets."
        ),
    )
    # GA workload knobs
    ap.add_argument("--pop-size", type=int, default=80)
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--elite-frac", type=float, default=0.2)
    ap.add_argument("--max-assets", type=int, default=10)
    ap.add_argument("--min-assets", type=int, default=5)
    ap.add_argument(
        "--weight-mode",
        default="long_short",
        choices=["long_only", "long_short", "gross_signed"],
        help=(
            "Weight interpretation/search space for the GA archive. "
            "Use long_short to match run_portfolio_search.py. "
            "Executable-aware tuning will internally evaluate rounded candidates as gross_signed."
        ),
    )
    ap.add_argument("--n-paths-init", type=int, default=5000)
    ap.add_argument("--n-paths-final", type=int, default=20000)
    ap.add_argument("--pca-k", type=int, default=3)
    ap.add_argument("--block-min", type=int, default=8)
    ap.add_argument("--block-max", type=int, default=12)
    ap.add_argument("--archive-limit", type=int, default=50000)
    ap.add_argument("--candidate-pool-size", type=int, default=2000)

    # Tuning workload knobs
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--pool-sample-size", type=int, default=500)
    ap.add_argument("--shortlist-size", type=int, default=60)
    ap.add_argument("--n-paths-train", type=int, default=6000)
    ap.add_argument("--n-paths-valid", type=int, default=20000)
    ap.add_argument("--ruin-cap", type=float, default=0.25)
    ap.add_argument("--alpha-ruin", type=float, default=0.5)

    # Stability-aware validation objective weights. These do not directly set
    # ScoreConfig lambdas; they tell the tuning optimizer what a good
    # validation portfolio should look like.
    ap.add_argument("--alpha-stability", type=float, default=0.35)
    ap.add_argument("--alpha-cdar", type=float, default=0.25)
    ap.add_argument("--alpha-path-mdd", type=float, default=0.25)
    ap.add_argument("--alpha-breach", type=float, default=0.40)
    ap.add_argument("--alpha-underwater", type=float, default=0.10)
    ap.add_argument("--alpha-ttr", type=float, default=0.10)

    ap.add_argument(
        "--tune-on-theoretical-candidates",
        action="store_true",
        help="Disable executable-aware tuning and tune on theoretical GA weights only. Default is executable-aware tuning.",
    )
    ap.add_argument("--executable-min-weight", type=float, default=0.01)
    ap.add_argument("--executable-min-units-equity", type=float, default=1.0)
    ap.add_argument("--executable-min-units-crypto", type=float, default=0.0)
    ap.add_argument("--executable-min-units-weight-thr", type=float, default=0.03)
    ap.add_argument("--executable-crypto-decimals", type=int, default=8)
    ap.add_argument("--executable-nearest-step-remaining-frac", type=float, default=0.10)
    ap.add_argument("--executable-min-deployment-ratio", type=float, default=0.95)
    ap.add_argument("--executable-max-cash-weight", type=float, default=0.05)
    ap.add_argument("--executable-max-weight-drift-l1", type=float, default=0.20)
    ap.add_argument("--executable-max-dropped-weight", type=float, default=0.05)

    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument(
        "--target-leverage",
        type=float,
        default=7.0,
        help="Override target leverage. Pass --use-regime-leverage to use HMM leverage instead.",
    )
    ap.add_argument(
        "--use-regime-leverage",
        action="store_true",
        help="Use leverage from regimes/hmm instead of --target-leverage.",
    )
    ap.add_argument(
        "--no-portfolio-hmm",
        action="store_true",
        help="Do not read regimes/hmm; only useful together with explicit --target-leverage.",
    )

    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-latest", action="store_true")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    return ap.parse_args()


def _main_impl() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)
    write_outputs = not bool(args.no_write)

    if write_outputs:
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    override_target_leverage = None if bool(args.use_regime_leverage) else float(args.target_leverage)
    equity_result = resolve_current_equity(cfg=cfg, as_of=args.as_of, equity_override=args.equity0)
    print_equity_valuation_result(equity_result)

    run_score_weights_tuning(
        cfg=cfg,
        as_of=args.as_of,
        universe_csv=args.universe_csv,
        equity0=float(equity_result.equity),
        goals=_parse_goals(args.goals),
        main_goal=float(args.main_goal),
        override_target_leverage=override_target_leverage,
        use_portfolio_hmm=(not bool(args.no_portfolio_hmm)),
        write_outputs=write_outputs,
        update_latest=(not bool(args.no_latest)),
        confirm_prod_write=bool(args.confirm_prod_write),
        cache_min_years=int(args.cache_min_years),
        min_universe_size=int(args.min_universe_size),
        max_nan_frac=float(args.max_nan_frac),
        pop_size=int(args.pop_size),
        generations=int(args.generations),
        elite_frac=float(args.elite_frac),
        max_assets=int(args.max_assets),
        min_assets=int(args.min_assets),
        weight_mode=str(args.weight_mode),
        n_paths_init=int(args.n_paths_init),
        n_paths_final=int(args.n_paths_final),
        pca_k=int(args.pca_k),
        block_min=int(args.block_min),
        block_max=int(args.block_max),
        archive_limit=int(args.archive_limit),
        candidate_pool_size=int(args.candidate_pool_size),
        n_trials=int(args.n_trials),
        pool_sample_size=int(args.pool_sample_size),
        shortlist_size=int(args.shortlist_size),
        n_paths_train=int(args.n_paths_train),
        n_paths_valid=int(args.n_paths_valid),
        ruin_cap=float(args.ruin_cap),
        alpha_ruin=float(args.alpha_ruin),
        alpha_stability=float(args.alpha_stability),
        alpha_cdar=float(args.alpha_cdar),
        alpha_path_mdd=float(args.alpha_path_mdd),
        alpha_breach=float(args.alpha_breach),
        alpha_underwater=float(args.alpha_underwater),
        alpha_ttr=float(args.alpha_ttr),
        train_frac=float(args.train_frac),
        tune_on_executable_candidates=(not bool(args.tune_on_theoretical_candidates)),
        executable_min_weight=float(args.executable_min_weight),
        executable_min_units_equity=float(args.executable_min_units_equity),
        executable_min_units_crypto=float(args.executable_min_units_crypto),
        executable_min_units_weight_thr=float(args.executable_min_units_weight_thr),
        executable_crypto_decimals=int(args.executable_crypto_decimals),
        executable_nearest_step_remaining_frac=float(args.executable_nearest_step_remaining_frac),
        executable_min_deployment_ratio=float(args.executable_min_deployment_ratio),
        executable_max_cash_weight=float(args.executable_max_cash_weight),
        executable_max_weight_drift_l1=float(args.executable_max_weight_drift_l1),
        executable_max_dropped_weight=float(args.executable_max_dropped_weight),
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
        script_name="run_score_weights_tuning.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            _main_impl()

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="score_weights_tuning",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="run_score_weights_tuning.py",
                source_mode="score_weights_tuning",
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
                entity_type="score_weights_tuning",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="run_score_weights_tuning.py",
                source_mode="score_weights_tuning",
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
