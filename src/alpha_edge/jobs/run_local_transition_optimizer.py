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
    s3_get_json,
    s3_init,
    s3_load_latest_json,
    s3_write_json_event,
)
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.runtime import (
    RuntimeConfig,
    load_runtime_config,
    require_prod_confirmation,
)
from alpha_edge.portfolio.equity_valuation import resolve_current_equity, print_equity_valuation_result
from alpha_edge.core.schemas import (
    LocalTransitionOptimizerConfig,
    ScoreConfig,
)
from alpha_edge.portfolio.local_transition_optimizer import run_local_transition_optimizer
from alpha_edge.portfolio.regime_asset_preferences import build_portfolio_regime_fit_comparison
from alpha_edge.universe.universe import load_universe


DEFAULT_ENGINE_BUCKET = "alpha-edge-algo"
DEFAULT_ENGINE_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT_PREFIX = "engine/v1"
DEFAULT_MARKET_ROOT = "market"

TRANSITION_ASSESSMENT_TABLE = "portfolio_transition/assessment"
LOCAL_OPTIMIZER_TABLE = "portfolio_transition/local_optimizer"


def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_ENGINE_BUCKET)).strip()


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_ENGINE_REGION)).strip()


def cfg_engine_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "engine_root", DEFAULT_ENGINE_ROOT_PREFIX)).strip("/")


def cfg_market_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "market_root", DEFAULT_MARKET_ROOT)).strip("/")


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        v = float(x)
    except Exception:
        return default
    if not np.isfinite(v):
        return default
    return float(v)


def _parse_goals(s: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"--goals must contain exactly 3 comma-separated numbers. Got {s!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _returns_wide_cache_path(cfg: RuntimeConfig, *, min_years: int) -> str:
    return (
        f"s3://{cfg_bucket(cfg)}/"
        f"{cfg_market_root(cfg)}/cache/v1/"
        f"returns_wide_min{int(min_years)}y.parquet"
    )


def _load_returns_wide(
    *,
    cfg: RuntimeConfig,
    as_of_ts: pd.Timestamp,
    cache_min_years: int,
    min_history_days: int,
    max_nan_frac: float,
) -> pd.DataFrame:
    path = _returns_wide_cache_path(cfg, min_years=int(cache_min_years))
    returns = pd.read_parquet(path, engine="pyarrow").sort_index()

    returns.index = pd.to_datetime(returns.index, errors="coerce").tz_localize(None).normalize()
    returns = returns.loc[returns.index <= as_of_ts]

    if returns.shape[0] < int(min_history_days):
        raise RuntimeError(
            f"Not enough returns history up to as_of={as_of_ts.date()}: "
            f"rows={returns.shape[0]}, required={int(min_history_days)}"
        )

    returns, diag = clean_returns_matrix(
        returns,
        min_history_days=int(min_history_days),
        max_nan_frac=float(max_nan_frac),
        min_vol=1e-6,
    )

    if returns.empty:
        raise RuntimeError(f"returns_wide became empty after cleaning. diag={diag}")

    return returns


def _load_score_config(
    *,
    s3,
    bucket: str,
    root_prefix: str,
) -> ScoreConfig:
    raw = s3_load_latest_json(
        s3,
        bucket=bucket,
        root_prefix=root_prefix,
        table="configs/score_config",
    )

    if not raw:
        return ScoreConfig()

    return ScoreConfig(**raw)


def _latest_prices_by_asset_id(cfg: RuntimeConfig) -> dict[str, float]:
    try:
        market = MarketStore(
            bucket=cfg_bucket(cfg),
            region=cfg_region(cfg),
            base_prefix=cfg_market_root(cfg),
        )
    except TypeError:
        market = MarketStore(
            bucket=cfg_bucket(cfg),
            region=cfg_region(cfg),
        )

    df = market.read_latest_prices_snapshot()
    if df is None or df.empty:
        raise RuntimeError("Missing latest prices snapshot.")

    df = df.copy()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()

    if "adj_close_usd" in df.columns:
        px_col = "adj_close_usd"
    elif "close_usd" in df.columns:
        px_col = "close_usd"
    elif "close_raw_usd" in df.columns:
        px_col = "close_raw_usd"
    else:
        raise RuntimeError(
            "latest prices snapshot has no usable USD close column. "
            f"Columns={list(df.columns)}"
        )

    df[px_col] = pd.to_numeric(df[px_col], errors="coerce")
    df = df.dropna(subset=["asset_id", px_col])
    df = df[df[px_col] > 0].copy()

    return dict(zip(df["asset_id"].tolist(), df[px_col].astype(float).tolist()))


def _load_universe_resolution_maps() -> dict[str, str]:
    """
    Build ticker/symbol -> asset_id resolver from the local universe.

    This keeps the runner tolerant of positions keyed by ticker while the market
    data and returns cache are asset_id-first.
    """
    universe_path = paths.universe_dir() / "universe.csv"
    df = pd.read_csv(universe_path)

    if "include" in df.columns:
        df = df[pd.to_numeric(df["include"], errors="coerce").fillna(1).astype(int) == 1].copy()

    for c in ["asset_id", "ticker", "yahoo_ticker", "yahoo_ticker_norm"]:
        if c not in df.columns:
            df[c] = ""

    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df = df[df["asset_id"] != ""].copy()

    resolver: dict[str, str] = {}

    for _, row in df.iterrows():
        asset_id = str(row.get("asset_id", "")).strip()
        if not asset_id:
            continue

        resolver[asset_id] = asset_id
        resolver[asset_id.upper()] = asset_id
        resolver[asset_id.lower()] = asset_id
        resolver[asset_id.casefold()] = asset_id

        for col in ["ticker", "yahoo_ticker", "yahoo_ticker_norm"]:
            sym = str(row.get(col, "")).strip()
            if sym:
                resolver[sym] = asset_id
                resolver[sym.upper()] = asset_id
                resolver[sym.lower()] = asset_id
                resolver[sym.casefold()] = asset_id

    return resolver


def _resolve_asset_key(key: object, resolver: dict[str, str]) -> str | None:
    if key is None:
        return None

    raw = str(key).strip()
    if not raw or raw.lower() == "nan":
        return None

    return (
        resolver.get(raw)
        or resolver.get(raw.upper())
        or resolver.get(raw.lower())
        or resolver.get(raw.casefold())
    )


def _load_current_positions(
    *,
    s3,
    bucket: str,
    root_prefix: str,
    as_of_date: str,
) -> dict[str, float]:
    """
    Preferred source:
      engine/v1/ledger/dt=YYYY-MM-DD/positions.json

    Fallback:
      engine/v1/inputs/positions/latest.json

    Returns:
      raw position key -> quantity
    """
    ledger_key = f"{root_prefix.strip('/')}/ledger/dt={as_of_date}/positions.json"
    raw = s3_get_json(s3, bucket=bucket, key=ledger_key)

    if not raw:
        raw = s3_load_latest_json(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="inputs/positions",
        )

    if not raw:
        raise RuntimeError(
            "Could not load current positions from ledger dated positions "
            f"or inputs/positions/latest.json for as_of={as_of_date}."
        )

    parsed = parse_positions_obj(raw)

    out: dict[str, float] = {}
    for key, pos in parsed.items():
        raw_key = getattr(pos, "asset_id", None) or getattr(pos, "ticker", None) or key
        qty = _safe_float(getattr(pos, "quantity", None))
        if raw_key and qty is not None and abs(qty) > 0:
            out[str(raw_key)] = float(qty)

    if not out:
        raise RuntimeError("Loaded positions but no non-zero quantities found.")

    return out


def _weights_from_positions(
    *,
    raw_positions_qty: dict[str, float],
    prices_by_asset_id: dict[str, float],
    resolver: dict[str, str],
    returns_columns: set[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    exposures: dict[str, float] = {}
    unresolved: list[dict[str, Any]] = []
    missing_price: list[str] = []
    missing_returns: list[str] = []

    for raw_key, qty in raw_positions_qty.items():
        asset_id = _resolve_asset_key(raw_key, resolver)
        if not asset_id:
            unresolved.append({"key": str(raw_key), "reason": "unresolved_asset"})
            continue

        if asset_id not in returns_columns:
            missing_returns.append(asset_id)
            continue

        px = _safe_float(prices_by_asset_id.get(asset_id))
        if px is None or px <= 0:
            missing_price.append(asset_id)
            continue

        q = float(qty)
        exp = float(q * px)

        if abs(exp) <= 0:
            continue

        exposures[asset_id] = float(exposures.get(asset_id, 0.0) + exp)

    gross = float(sum(abs(x) for x in exposures.values()))

    if gross <= 0:
        raise RuntimeError(
            "Could not build current weights from positions. "
            f"unresolved={unresolved[:10]}, missing_price={missing_price[:10]}, "
            f"missing_returns={missing_returns[:10]}"
        )

    weights = {
        asset_id: float(exp / gross)
        for asset_id, exp in exposures.items()
        if abs(exp) > 0
    }

    diagnostics = {
        "raw_position_count": int(len(raw_positions_qty)),
        "resolved_exposure_count": int(len(exposures)),
        "gross_notional_from_positions": float(gross),
        "unresolved_sample": unresolved[:20],
        "missing_price_sample": missing_price[:20],
        "missing_returns_sample": missing_returns[:20],
    }

    return weights, diagnostics


def _load_transition_assessment(
    *,
    s3,
    bucket: str,
    root_prefix: str,
) -> dict[str, Any]:
    raw = s3_load_latest_json(
        s3,
        bucket=bucket,
        root_prefix=root_prefix,
        table=TRANSITION_ASSESSMENT_TABLE,
    )

    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Missing transition assessment latest.json under "
            f"s3://{bucket}/{root_prefix}/{TRANSITION_ASSESSMENT_TABLE}/latest.json. "
            "Run run_transition_assessment.py first."
        )

    return raw


def _local_optimizer_allowed(transition_assessment: dict[str, Any]) -> bool:
    rec = str(transition_assessment.get("recommendation") or "").strip()
    allowed = bool(transition_assessment.get("local_optimization_allowed"))

    return allowed and rec == "LOCAL_OPTIMIZATION_RECOMMENDED"


def _load_universe_for_returns(
    *,
    returns: pd.DataFrame,
    universe_csv: str | None = None,
) -> dict:
    universe_path = universe_csv or str(paths.universe_dir() / "universe.csv")
    universe_all = load_universe(universe_path)

    returns_cols = {str(c) for c in returns.columns}

    universe = {
        str(k): v
        for k, v in universe_all.items()
        if str(k) in returns_cols
    }

    if len(universe) < 2:
        raise RuntimeError(
            "Universe keys do not overlap returns_wide columns. "
            "Local optimizer requires universe keys and returns columns to use the same asset_id basis. "
            f"universe_sample={list(universe_all.keys())[:10]}, "
            f"returns_sample={list(returns.columns)[:10]}"
        )

    return universe


def _extract_current_regime(
    *,
    transition_assessment: dict[str, Any],
    regime_history: pd.DataFrame,
) -> str | None:
    current_state = transition_assessment.get("current_state") or {}
    if isinstance(current_state, dict):
        regime = current_state.get("regime") or current_state.get("label_or_mixed")
        if regime:
            return str(regime).strip().upper()

    if regime_history is not None and not regime_history.empty and "regime" in regime_history.columns:
        regime = regime_history.sort_values("date").iloc[-1].get("regime")
        if regime:
            return str(regime).strip().upper()

    return None


def _load_regime_fit_diagnostics(
    *,
    cfg: RuntimeConfig,
    returns: pd.DataFrame,
    as_of_date: str,
    transition_assessment: dict[str, Any],
    current_weights: dict[str, float],
    candidate_weights: dict[str, float] | None,
    candidate_name: str,
) -> dict[str, Any]:
    if not candidate_weights:
        return {
            "status": "skipped",
            "reason": "No candidate weights available for regime-fit comparison.",
        }

    try:
        engine_store = MarketStore(
            bucket=cfg_bucket(cfg),
            region=cfg_region(cfg),
            base_prefix=cfg_engine_root(cfg),
        )

        regime_history = engine_store.read_market_hmm_regime_history(
            end=as_of_date,
            include_mixed=True,
            require_point_in_time=False,
        )

        if regime_history is None or regime_history.empty:
            return {
                "status": "unavailable",
                "reason": "No market HMM regime history found under existing regimes/market_hmm path.",
            }

        current_regime = _extract_current_regime(
            transition_assessment=transition_assessment,
            regime_history=regime_history,
        )

        if not current_regime:
            return {
                "status": "unavailable",
                "reason": "Could not determine current regime from transition assessment or regime history.",
            }

        out = build_portfolio_regime_fit_comparison(
            returns_wide=returns,
            regime_history=regime_history,
            regime=current_regime,
            current_weights=current_weights,
            candidate_weights=candidate_weights,
            candidate_name=candidate_name,
        )

        out["inputs"] = {
            "regime_history_rows": int(len(regime_history)),
            "regime_history_start": (
                None if regime_history.empty else str(pd.Timestamp(regime_history["date"].min()).date())
            ),
            "regime_history_end": (
                None if regime_history.empty else str(pd.Timestamp(regime_history["date"].max()).date())
            ),
            "returns_rows": int(len(returns)),
            "returns_cols": int(returns.shape[1]),
            "point_in_time_policy": "regime history loaded only through existing dt=YYYY-MM-DD market_hmm outputs",
        }
        return out

    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def run_local_transition_optimizer_job(
    *,
    cfg: RuntimeConfig,
    as_of: str | None = None,
    write_outputs: bool = True,
    update_latest: bool = True,
    confirm_prod_write: bool = False,
    equity0: float,
    notional: float | None = None,
    goals: tuple[float, float, float] = (7500.0, 10000.0, 12500.0),
    main_goal: float = 10000.0,
    universe_csv: str | None = None,
    cache_min_years: int = 5,
    min_history_days: int = 252 * 2,
    max_nan_frac: float = 0.30,
    random_seed: int = 123,
    anneal_steps: int = 100,
    temp_start: float = 0.50,
    temp_end: float = 0.03,
    n_paths_current: int = 5000,
    n_paths_init: int = 2000,
    n_paths_final: int = 8000,
    path_source: str = "pca",
    pca_k: int | None = 5,
    block_min: int = 8,
    block_max: int = 12,
    max_assets: int = 10,
    min_assets: int = 5,
    weight_mode: str = "long_short",
    max_turnover: float = 0.10,
    min_score_improvement: float = 0.02,
    min_health_improvement: float = 3.0,
) -> dict[str, Any]:
    if write_outputs:
        require_prod_confirmation(cfg, bool(confirm_prod_write))

    bucket = cfg_bucket(cfg)
    region = cfg_region(cfg)
    root_prefix = cfg_engine_root(cfg)

    as_of_ts = pd.Timestamp(as_of or dt.date.today()).tz_localize(None).normalize()
    as_of_date = as_of_ts.strftime("%Y-%m-%d")

    s3 = s3_init(region)

    transition_assessment = _load_transition_assessment(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
    )

    if not _local_optimizer_allowed(transition_assessment):
        payload = {
            "schema_version": "local_transition_optimizer_v1",
            "as_of": as_of_date,
            "status": "skipped",
            "recommendation": "SKIPPED",
            "reason": (
                "Latest transition assessment does not allow local optimization. "
                f"assessment_recommendation={transition_assessment.get('recommendation')!r}"
            ),
            "transition_assessment": {
                "recommendation": transition_assessment.get("recommendation"),
                "full_search_required": transition_assessment.get("full_search_required"),
                "local_optimization_allowed": transition_assessment.get("local_optimization_allowed"),
                "shadow_portfolio_required": transition_assessment.get("shadow_portfolio_required"),
                "reason": transition_assessment.get("reason"),
                "diagnostics": transition_assessment.get("diagnostics"),
            },
        }

        print("\n=== LOCAL TRANSITION OPTIMIZER ===")
        print(f"as_of:          {as_of_date}")
        print("status:         skipped")
        print(f"reason:         {payload['reason']}")

        if write_outputs:
            s3_write_json_event(
                s3,
                bucket=bucket,
                root_prefix=root_prefix,
                table=LOCAL_OPTIMIZER_TABLE,
                dt=as_of_ts,
                filename="local_transition_optimizer.json",
                payload=payload,
                update_latest=update_latest,
            )

        return payload

    returns = _load_returns_wide(
        cfg=cfg,
        as_of_ts=as_of_ts,
        cache_min_years=int(cache_min_years),
        min_history_days=int(min_history_days),
        max_nan_frac=float(max_nan_frac),
    )

    universe = _load_universe_for_returns(
        returns=returns,
        universe_csv=universe_csv,
    )

    score_config = _load_score_config(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
    )

    prices_by_asset_id = _latest_prices_by_asset_id(cfg)
    resolver = _load_universe_resolution_maps()

    raw_positions_qty = _load_current_positions(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
        as_of_date=as_of_date,
    )

    current_weights, position_diag = _weights_from_positions(
        raw_positions_qty=raw_positions_qty,
        prices_by_asset_id=prices_by_asset_id,
        resolver=resolver,
        returns_columns={str(c) for c in returns.columns},
    )

    current_gross = float(position_diag["gross_notional_from_positions"])
    notional_effective = float(notional) if notional is not None else current_gross

    local_cfg = LocalTransitionOptimizerConfig(
        random_seed=int(random_seed),
        anneal_steps=int(anneal_steps),
        temp_start=float(temp_start),
        temp_end=float(temp_end),
        n_paths_current=int(n_paths_current),
        n_paths_init=int(n_paths_init),
        n_paths_final=int(n_paths_final),
        path_source=str(path_source),
        pca_k=None if pca_k is None else int(pca_k),
        block_size=(int(block_min), int(block_max)),
        max_assets=int(max_assets),
        min_assets=int(min_assets),
        weight_mode=str(weight_mode),
        max_turnover=float(max_turnover),
        min_score_improvement=float(min_score_improvement),
        min_health_improvement=float(min_health_improvement),
    )

    result = run_local_transition_optimizer(
        as_of=as_of_date,
        returns=returns,
        universe=universe,
        current_weights=current_weights,
        equity0=float(equity0),
        notional=float(notional_effective),
        goals=goals,
        main_goal=float(main_goal),
        score_config=score_config,
        cfg=local_cfg,
        lw_cov=None,
    )

    candidate_weights = (
        None
        if result.best_candidate is None
        else {k: float(v) for k, v in result.best_candidate.weights.items()}
    )

    regime_fit = _load_regime_fit_diagnostics(
        cfg=cfg,
        returns=returns,
        as_of_date=as_of_date,
        transition_assessment=transition_assessment,
        current_weights=current_weights,
        candidate_weights=candidate_weights,
        candidate_name="candidate",
    )

    payload = {
        "schema_version": "local_transition_optimizer_v1",
        "as_of": as_of_date,
        "status": "success",
        "recommendation": result.recommendation,
        "reason": result.reason,
        "current_weights": {k: float(v) for k, v in result.current_weights.items()},
        "current_score": float(result.current_score),
        "current_health_score": result.current_health_score,
        "best_candidate": (
            None if result.best_candidate is None else asdict(result.best_candidate)
        ),
        "candidates_evaluated": int(result.candidates_evaluated),
        "candidates_accepted_by_turnover": int(result.candidates_accepted_by_turnover),
        "config": asdict(result.config),
        "diagnostics": {
            **dict(result.diagnostics or {}),
            "positions": position_diag,
            "regime_fit": regime_fit,
            "transition_assessment": {
                "recommendation": transition_assessment.get("recommendation"),
                "reason": transition_assessment.get("reason"),
                "diagnostics": transition_assessment.get("diagnostics"),
            },
            "inputs": {
                "equity0": float(equity0),
                "notional": float(notional_effective),
                "goals": [float(x) for x in goals],
                "main_goal": float(main_goal),
                "returns_rows": int(len(returns)),
                "returns_cols": int(returns.shape[1]),
                "universe_size": int(len(universe)),
            },
        },
    }

    print("\n=== LOCAL TRANSITION OPTIMIZER ===")
    print(f"env:                         {getattr(cfg, 'env', 'unknown')}")
    print(f"bucket:                      {bucket}")
    print(f"root_prefix:                 {root_prefix}")
    print(f"as_of:                       {as_of_date}")
    print(f"recommendation:              {payload['recommendation']}")
    print(f"reason:                      {payload['reason']}")
    print(f"current_score:               {payload['current_score']:.4f}")
    print(f"current_position_gross:      {current_gross:,.2f}")
    print(f"notional_used:               {notional_effective:,.2f}")

    if payload["best_candidate"] is not None:
        bc = payload["best_candidate"]
        print(f"best_candidate_score:        {float(bc['score']):.4f}")
        print(f"score_improvement:           {float(bc['score_improvement']):.4f}")
        print(f"turnover:                    {float(bc['turnover']):.2%}")
        print(f"delta_weight_count:          {len(bc.get('delta_weights') or {})}")

    if isinstance(payload.get("diagnostics"), dict):
        rf = payload["diagnostics"].get("regime_fit") or {}
        if rf.get("status") == "success":
            comp = rf.get("comparison") or {}
            adv = comp.get("preference_score_advantage")
            if adv is not None:
                print(f"regime_fit_advantage:       {float(adv):.4f}")

    if write_outputs:
        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table=LOCAL_OPTIMIZER_TABLE,
            dt=as_of_ts,
            filename="local_transition_optimizer.json",
            payload=payload,
            update_latest=update_latest,
        )

        print(
            f"\n[S3] Saved local optimizer result to "
            f"s3://{bucket}/{root_prefix}/{LOCAL_OPTIMIZER_TABLE}/dt={as_of_date}/"
        )

    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run local portfolio transition optimizer using existing simulated annealing."
    )

    p.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    p.add_argument("--as-of", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-prod-write", action="store_true")

    p.add_argument("--equity0", "--equity-override", dest="equity0", type=float, default=None, help="Equity used as MC initial equity. If omitted, resolved from ledger + latest prices.")
    p.add_argument("--notional", type=float, default=None)
    p.add_argument("--goals", default="7500,10000,12500")
    p.add_argument("--main-goal", type=float, default=10000.0)

    p.add_argument("--universe-csv", default=None)
    p.add_argument("--cache-min-years", type=int, default=5)
    p.add_argument("--min-history-days", type=int, default=252 * 2)
    p.add_argument("--max-nan-frac", type=float, default=0.30)

    p.add_argument("--random-seed", type=int, default=123)
    p.add_argument("--anneal-steps", type=int, default=100)
    p.add_argument("--temp-start", type=float, default=0.50)
    p.add_argument("--temp-end", type=float, default=0.03)

    p.add_argument("--n-paths-current", type=int, default=5000)
    p.add_argument("--n-paths-init", type=int, default=2000)
    p.add_argument("--n-paths-final", type=int, default=8000)

    p.add_argument("--path-source", default="pca", choices=["bootstrap", "pca"])
    p.add_argument("--pca-k", type=int, default=5)
    p.add_argument("--block-min", type=int, default=8)
    p.add_argument("--block-max", type=int, default=12)

    p.add_argument("--max-assets", type=int, default=10)
    p.add_argument("--min-assets", type=int, default=5)
    p.add_argument("--weight-mode", default="long_short", choices=["long_only", "long_short", "gross_signed"])

    p.add_argument("--max-turnover", type=float, default=0.10)
    p.add_argument("--min-score-improvement", type=float, default=0.02)
    p.add_argument("--min-health-improvement", type=float, default=3.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = bool(getattr(args, "dry_run", False))

    with capture_script_run(
        cfg=cfg,
        script_name="run_local_transition_optimizer.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            equity_result = resolve_current_equity(cfg=cfg, as_of=args.as_of, equity_override=args.equity0)
            print_equity_valuation_result(equity_result)

            payload = run_local_transition_optimizer_job(
                cfg=cfg,
                as_of=args.as_of,
                write_outputs=not is_dry_run,
                update_latest=True,
                confirm_prod_write=bool(args.confirm_prod_write),
                equity0=float(equity_result.equity),
                notional=args.notional,
                goals=_parse_goals(args.goals),
                main_goal=float(args.main_goal),
                universe_csv=args.universe_csv,
                cache_min_years=int(args.cache_min_years),
                min_history_days=int(args.min_history_days),
                max_nan_frac=float(args.max_nan_frac),
                random_seed=int(args.random_seed),
                anneal_steps=int(args.anneal_steps),
                temp_start=float(args.temp_start),
                temp_end=float(args.temp_end),
                n_paths_current=int(args.n_paths_current),
                n_paths_init=int(args.n_paths_init),
                n_paths_final=int(args.n_paths_final),
                path_source=str(args.path_source),
                pca_k=int(args.pca_k),
                block_min=int(args.block_min),
                block_max=int(args.block_max),
                max_assets=int(args.max_assets),
                min_assets=int(args.min_assets),
                weight_mode=str(args.weight_mode),
                max_turnover=float(args.max_turnover),
                min_score_improvement=float(args.min_score_improvement),
                min_health_improvement=float(args.min_health_improvement),
            )

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="create",
                entity_type="local_transition_optimizer",
                entity_id=str(payload.get("as_of")),
                as_of=str(payload.get("as_of")),
                source_script="run_local_transition_optimizer.py",
                source_mode="local_transition_optimizer",
                status=("dry_run" if is_dry_run else "success"),
                input_args=vars(args),
                output_keys=[] if is_dry_run else [
                    f"{cfg_engine_root(cfg)}/{LOCAL_OPTIMIZER_TABLE}/dt={payload.get('as_of')}/local_transition_optimizer.json",
                    f"{cfg_engine_root(cfg)}/{LOCAL_OPTIMIZER_TABLE}/latest.json",
                ],
                metadata={
                    "recommendation": payload.get("recommendation"),
                    "status": payload.get("status"),
                    "reason": payload.get("reason"),
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)

        except Exception as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="create",
                entity_type="local_transition_optimizer",
                entity_id=None,
                as_of=str(getattr(args, "as_of", "") or ""),
                source_script="run_local_transition_optimizer.py",
                source_mode="local_transition_optimizer",
                status="failed",
                input_args=vars(args),
                metadata={
                    "tier": "local_transition_optimizer",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise


if __name__ == "__main__":
    main()