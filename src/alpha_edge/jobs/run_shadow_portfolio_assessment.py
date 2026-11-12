# src/alpha_edge/jobs/run_shadow_portfolio_assessment.py
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
    ShadowPortfolioConfig,
    ShadowPortfolioState,
)
from alpha_edge.portfolio.portfolio_search import evaluate_weights_for_search
from alpha_edge.portfolio.shadow_portfolio_engine import (
    assess_shadow_portfolio,
    delta_weights,
    weight_turnover,
)
from alpha_edge.portfolio.regime_asset_preferences import build_portfolio_regime_fit_comparison


DEFAULT_ENGINE_BUCKET = "alpha-edge-algo"
DEFAULT_ENGINE_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT_PREFIX = "engine/v1"
DEFAULT_MARKET_ROOT = "market"

TRANSITION_ASSESSMENT_TABLE = "portfolio_transition/assessment"
SHADOW_PORTFOLIO_TABLE = "portfolio_transition/shadow"
PORTFOLIO_RUNS_TABLE = "portfolio_search/runs"
QUAR_CAND_TABLE = "quarantine/candidates"


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


def _safe_str(x: Any) -> str | None:
    if x is None:
        return None
    text = str(x).strip()
    if not text or text.lower() == "nan":
        return None
    return text


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


def _unwrap_positions_payload(raw: Any) -> Any:
    """
    Normalize ledger positions payloads into one position mapping.

    Supported direct shape:
        {
            "AAPL": {...},
            "MSFT": {...}
        }

    Supported wrapped shape:
        {
            "as_of": "...",
            "positions": {...}
        }

    Current ledger rebuild shape:
        {
            "as_of": "...",
            "method": "...",
            "spot_positions": {...},
            "derivatives_positions": {...},
            "stats": {...},
            "sources": {...}
        }

    parse_positions_obj() expects only the actual positions mapping, not
    metadata keys such as as_of, method, stats, or sources.
    """
    if not isinstance(raw, dict):
        return raw

    # Simple wrapper shapes.
    for key in [
        "positions",
        "holdings",
        "positions_by_asset",
        "positions_by_ticker",
    ]:
        value = raw.get(key)
        if isinstance(value, dict):
            return value

    # Current ledger rebuild shape.
    merged: dict[str, Any] = {}

    spot_positions = raw.get("spot_positions")
    if isinstance(spot_positions, dict):
        merged.update(spot_positions)

    derivatives_positions = raw.get("derivatives_positions")
    if isinstance(derivatives_positions, dict):
        merged.update(derivatives_positions)

    if merged:
        return merged

    return raw

def _load_current_positions(
    *,
    s3,
    bucket: str,
    root_prefix: str,
    as_of_date: str,
) -> dict[str, float]:
    ledger_key = f"{root_prefix.strip('/')}/ledger/dt={as_of_date}/positions.json"

    raw = s3_get_json(
        s3,
        bucket=bucket,
        key=ledger_key,
    )

    if not raw:
        raise RuntimeError(
            f"Missing current ledger positions: s3://{bucket}/{ledger_key}"
        )

    out: dict[str, float] = {}

    def _consume_position_row(row: Any) -> None:
        if not isinstance(row, dict):
            return

        asset_key = (
            row.get("asset_id")
            or row.get("ticker")
            or row.get("symbol")
            or row.get("broker_ticker")
        )

        qty = _safe_float(
            row.get("quantity")
            if row.get("quantity") is not None
            else row.get("qty")
        )

        if asset_key and qty is not None and abs(float(qty)) > 0:
            asset_key = str(asset_key)
            out[asset_key] = out.get(asset_key, 0.0) + float(qty)

    if isinstance(raw, dict):
        spot_positions = raw.get("spot_positions")
        derivatives_positions = raw.get("derivatives_positions")

        if isinstance(spot_positions, list):
            for row in spot_positions:
                _consume_position_row(row)

        elif isinstance(spot_positions, dict):
            for fallback_key, row in spot_positions.items():
                if isinstance(row, dict) and not row.get("asset_id"):
                    row = {**row, "asset_id": fallback_key}
                _consume_position_row(row)

        if isinstance(derivatives_positions, list):
            for row in derivatives_positions:
                _consume_position_row(row)

        elif isinstance(derivatives_positions, dict):
            for fallback_key, row in derivatives_positions.items():
                if isinstance(row, dict) and not row.get("asset_id"):
                    row = {**row, "asset_id": fallback_key}
                _consume_position_row(row)

        # Backward-compatible fallback for older payloads.
        if not out:
            positions = raw.get("positions") or raw.get("holdings")
            if isinstance(positions, list):
                for row in positions:
                    _consume_position_row(row)
            elif isinstance(positions, dict):
                for fallback_key, row in positions.items():
                    if isinstance(row, dict) and not row.get("asset_id"):
                        row = {**row, "asset_id": fallback_key}
                    _consume_position_row(row)

    if not out:
        top_keys = list(raw.keys())[:20] if isinstance(raw, dict) else []
        raise RuntimeError(
            f"Loaded ledger positions from s3://{bucket}/{ledger_key}, "
            "but no non-zero quantities were found. "
            f"Top-level keys={top_keys}."
        )

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

        exp = float(float(qty) * float(px))
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


def _shadow_assessment_allowed(transition_assessment: dict[str, Any]) -> bool:
    rec = str(transition_assessment.get("recommendation") or "").strip()
    full_search_required = bool(transition_assessment.get("full_search_required"))
    shadow_required = bool(transition_assessment.get("shadow_portfolio_required"))

    return (
        rec in {"SHADOW_PORTFOLIO_ACTIVE", "FULL_SEARCH_REQUIRED"}
        or full_search_required
        or shadow_required
    )


def _s3_list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None

    while True:
        kwargs: dict[str, Any] = {
            "Bucket": bucket,
            "Prefix": prefix,
            "MaxKeys": 1000,
        }
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)

        for obj in resp.get("Contents", []) or []:
            key = str(obj.get("Key", "")).strip()
            if key:
                keys.append(key)

        if not resp.get("IsTruncated"):
            break

        token = resp.get("NextContinuationToken")

    return keys


def _candidate_latest_key(root_prefix: str, table: str, candidate_id: str) -> str:
    return f"{root_prefix.strip('/')}/{table.strip('/')}/candidate_id={candidate_id}/latest.json"


def _discover_latest_quarantine_approved_portfolio_search(
    *,
    s3,
    bucket: str,
    root_prefix: str,
    as_of_ts: pd.Timestamp,
    lookback_days: int,
) -> dict[str, Any]:
    """
    Discover the latest quarantine-approved portfolio-search candidate.

    Shadow assessment must not read portfolio_search/runs directly. The
    intended lifecycle is:

        portfolio search -> quarantine approval -> shadow assessment

    Source table:
        <root_prefix>/quarantine/candidates/candidate_id=<id>/latest.json

    Expected candidate state shape, produced by run_quarantine_analysis.py:
        {
          "candidate_id": "...",
          "source": {"run_id": "...", "run_key": "...", ...},
          "baseline_weights": {...},
          "baseline_search_eval": {...},
          "quarantine": {"status": "APPROVED", ...}
        }
    """
    prefix = f"{root_prefix.strip('/')}/{QUAR_CAND_TABLE}/candidate_id="
    keys = [
        k
        for k in _s3_list_keys(s3, bucket=bucket, prefix=prefix)
        if k.endswith("/latest.json")
    ]

    candidates: list[dict[str, Any]] = []

    for key in sorted(keys):
        try:
            cand_state = s3_get_json(s3, bucket=bucket, key=key) or {}
        except Exception:
            continue

        if not isinstance(cand_state, dict):
            continue

        q = cand_state.get("quarantine") or {}
        if not isinstance(q, dict):
            continue

        status = str(q.get("status") or cand_state.get("quarantine_status") or "").strip().upper()
        approved_for_shadow = cand_state.get("approved_for_shadow")
        if approved_for_shadow is None:
            approved_for_shadow = q.get("approved_for_shadow")
        if approved_for_shadow is None:
            approved_for_shadow = status == "APPROVED"

        if status != "APPROVED" or not bool(approved_for_shadow):
            continue

        last_eval_raw = (
            q.get("last_eval_as_of")
            or cand_state.get("last_seen_as_of")
            or q.get("start_as_of")
            or cand_state.get("as_of")
        )
        try:
            last_eval_ts = pd.Timestamp(last_eval_raw).tz_localize(None).normalize()
        except Exception:
            last_eval_ts = pd.Timestamp("1900-01-01")

        if last_eval_ts > as_of_ts:
            continue

        age_days = int((as_of_ts - last_eval_ts).days)
        if age_days > int(lookback_days):
            continue

        weights = cand_state.get("baseline_weights")
        if not isinstance(weights, dict) or not weights:
            # Do not infer weights from raw shares here. Shadow assessment
            # compares target portfolio weights; quarantine must persist the
            # portfolio-search baseline weights for approved candidates.
            continue

        metrics = cand_state.get("baseline_search_eval")
        if not isinstance(metrics, dict) or not metrics:
            metrics = (q.get("baseline_eval") if isinstance(q.get("baseline_eval"), dict) else {})

        if not isinstance(metrics, dict) or not metrics:
            continue

        cid = str(cand_state.get("candidate_id") or "").strip()
        if not cid:
            try:
                cid = key.split("candidate_id=")[-1].split("/")[0].strip()
            except Exception:
                cid = key.split("/")[-2]

        source = cand_state.get("source") or {}
        if not isinstance(source, dict):
            source = {}

        run_id = str(source.get("run_id") or source.get("source_run_id") or cid).strip()
        run_key = str(source.get("run_key") or source.get("source_run_key") or "").strip() or None

        health_score = _extract_normalized_health_score(metrics)
        if health_score is None:
            health_score = _safe_float(metrics.get("health_score"))

        final_exec = {
            "status": "quarantine_approved",
            "weights": {str(k): float(v) for k, v in weights.items() if _safe_float(v) is not None},
            "metrics": dict(metrics),
            "health_score": health_score,
            "health_grade": metrics.get("health_grade") or (metrics.get("health") or {}).get("health_grade") if isinstance(metrics.get("health"), dict) else None,
            "source": {
                "candidate_state_key": key,
                "portfolio_search_run_id": run_id,
                "portfolio_search_run_key": run_key,
                "quarantine_status": status,
                "quarantine_last_eval_as_of": None if last_eval_raw is None else str(last_eval_raw),
            },
        }

        candidates.append(
            {
                "run_id": run_id,
                "run_key": run_key,
                "run_as_of": str(source.get("run_as_of") or last_eval_ts.date()),
                "payload": cand_state,
                "final_executable": final_exec,
                "age_days": int(age_days),
                "candidate_id": cid,
                "candidate_state_key": key,
                "quarantine": q,
                "source": source,
            }
        )

    if not candidates:
        raise RuntimeError(
            f"No quarantine-approved portfolio-search candidate found in last {lookback_days} days "
            f"under s3://{bucket}/{root_prefix}/{QUAR_CAND_TABLE}/. "
            "Run run_quarantine_analysis.py until a candidate reaches quarantine.status=APPROVED."
        )

    candidates.sort(
        key=lambda x: (
            pd.Timestamp(x.get("run_as_of") or "1900-01-01"),
            str(x.get("run_id") or ""),
            str(x.get("candidate_state_key") or ""),
        ),
        reverse=True,
    )

    return candidates[0]

def _normalize_weight_dict_to_returns(
    *,
    weights: dict[str, Any],
    resolver: dict[str, str],
    returns_columns: set[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    out: dict[str, float] = {}
    unresolved: list[dict[str, Any]] = []
    missing_returns: list[str] = []

    for raw_key, raw_value in (weights or {}).items():
        v = _safe_float(raw_value)
        if v is None or abs(v) <= 1e-12:
            continue

        asset_id = _resolve_asset_key(raw_key, resolver) or str(raw_key).strip()

        if asset_id not in returns_columns:
            missing_returns.append(asset_id)
            continue

        out[asset_id] = float(out.get(asset_id, 0.0) + float(v))

    gross = float(sum(abs(x) for x in out.values()))
    if gross <= 0:
        raise RuntimeError(
            "Shadow weights could not be normalized to returns columns. "
            f"missing_returns_sample={missing_returns[:20]}, unresolved_sample={unresolved[:20]}"
        )

    out = {k: float(v / gross) for k, v in out.items()}

    diagnostics = {
        "input_weight_count": int(len(weights or {})),
        "resolved_weight_count": int(len(out)),
        "missing_returns_sample": missing_returns[:20],
        "unresolved_sample": unresolved[:20],
    }

    return out, diagnostics


def _load_previous_shadow_state(
    *,
    s3,
    bucket: str,
    root_prefix: str,
) -> dict[str, Any] | None:
    raw = s3_load_latest_json(
        s3,
        bucket=bucket,
        root_prefix=root_prefix,
        table=SHADOW_PORTFOLIO_TABLE,
    )

    return raw if isinstance(raw, dict) else None


def _same_shadow_candidate(
    *,
    previous: dict[str, Any] | None,
    source_run_id: str | None,
    source_run_key: str | None,
) -> bool:
    if not isinstance(previous, dict):
        return False

    state = previous.get("state") or {}
    if not isinstance(state, dict):
        return False

    prev_run_id = _safe_str(state.get("source_run_id"))
    prev_run_key = _safe_str(state.get("source_run_key"))

    if source_run_id and prev_run_id and str(source_run_id) == str(prev_run_id):
        return True

    if source_run_key and prev_run_key and str(source_run_key) == str(prev_run_key):
        return True

    return False


def _previous_shadow_counters(
    *,
    previous: dict[str, Any] | None,
    source_run_id: str | None,
    source_run_key: str | None,
) -> tuple[int, int]:
    if not _same_shadow_candidate(
        previous=previous,
        source_run_id=source_run_id,
        source_run_key=source_run_key,
    ):
        return 1, 0

    state = previous.get("state") or {}
    if not isinstance(state, dict):
        return 1, 0

    days_active = int(_safe_float(state.get("days_active"), 0) or 0) + 1
    days_dominating = int(_safe_float(state.get("days_dominating"), 0) or 0)

    return max(1, days_active), max(0, days_dominating)


def _extract_health_score(final_exec: dict[str, Any]) -> float | None:
    """
    Extract only normalized health score from a final executable payload.

    Do not use optimizer/objective score as health score.
    """
    return _extract_normalized_health_score(final_exec)


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

def _extract_normalized_health_score(raw: Any) -> float | None:
    """
    Extract only a normalized portfolio health score.

    Important:
    - health_score / normalized_health_score are health metrics.
    - score is usually the optimizer/objective score and may be negative.
    - Never fall back to raw["score"] for health_score.
    """
    if not isinstance(raw, dict):
        return None

    candidate_keys = [
        "health_score",
        "normalized_health_score",
        "portfolio_health_score",
    ]

    for key in candidate_keys:
        value = _safe_float(raw.get(key))
        if value is not None and 0.0 <= float(value) <= 100.0:
            return float(value)

    nested_keys = [
        "health",
        "portfolio_health",
        "summary",
        "diagnostics",
        "current_state",
    ]

    for nested_key in nested_keys:
        nested = raw.get(nested_key)
        if not isinstance(nested, dict):
            continue

        for key in candidate_keys:
            value = _safe_float(nested.get(key))
            if value is not None and 0.0 <= float(value) <= 100.0:
                return float(value)

    return None


def _load_current_health(
    *,
    s3,
    bucket: str,
    root_prefix: str,
) -> tuple[float | None, dict[str, Any]]:
    raw = s3_load_latest_json(
        s3,
        bucket=bucket,
        root_prefix=root_prefix,
        table="health",
    ) or {}

    if not isinstance(raw, dict):
        return None, {}

    health_score = _extract_normalized_health_score(raw)

    return health_score, raw

def run_shadow_portfolio_assessment_job(
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
    cache_min_years: int = 5,
    min_history_days: int = 252 * 2,
    max_nan_frac: float = 0.30,
    lookback_days: int = 30,
    n_paths_current: int = 5000,
    n_paths_shadow: int = 5000,
    weight_mode: str = "long_short",
    block_min: int = 8,
    block_max: int = 12,
    min_health_advantage: float = 5.0,
    min_score_advantage: float = 0.02,
    max_turnover_to_accept: float = 0.35,
    confirmation_days: int = 3,
    immediate_accept_health_advantage: float = 10.0,
    immediate_accept_score_advantage: float = 0.05,
    random_seed: int = 123,
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

    if not _shadow_assessment_allowed(transition_assessment):
        payload = {
            "schema_version": "shadow_portfolio_assessment_v1",
            "as_of": as_of_date,
            "status": "skipped",
            "recommendation": "KEEP_CURRENT",
            "reason": (
                "Latest transition assessment does not require shadow portfolio assessment. "
                f"assessment_recommendation={transition_assessment.get('recommendation')!r}"
            ),
            "transition_assessment": {
                "recommendation": transition_assessment.get("recommendation"),
                "full_search_required": transition_assessment.get("full_search_required"),
                "shadow_portfolio_required": transition_assessment.get("shadow_portfolio_required"),
                "reason": transition_assessment.get("reason"),
                "diagnostics": transition_assessment.get("diagnostics"),
            },
        }

        print("\n=== SHADOW PORTFOLIO ASSESSMENT ===")
        print(f"as_of:          {as_of_date}")
        print("status:         skipped")
        print(f"reason:         {payload['reason']}")

        if write_outputs:
            s3_write_json_event(
                s3,
                bucket=bucket,
                root_prefix=root_prefix,
                table=SHADOW_PORTFOLIO_TABLE,
                dt=as_of_ts,
                filename="shadow_portfolio_assessment.json",
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

    score_config_raw = (
        s3_load_latest_json(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="configs/score_config",
        )
        or {}
    )

    from alpha_edge.core.schemas import ScoreConfig

    score_config = ScoreConfig(**score_config_raw) if isinstance(score_config_raw, dict) and score_config_raw else ScoreConfig()

    resolver = _load_universe_resolution_maps()
    prices_by_asset_id = _latest_prices_by_asset_id(cfg)

    raw_positions_qty = _load_current_positions(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
        as_of_date=as_of_date,
    )

    current_weights, current_position_diag = _weights_from_positions(
        raw_positions_qty=raw_positions_qty,
        prices_by_asset_id=prices_by_asset_id,
        resolver=resolver,
        returns_columns={str(c) for c in returns.columns},
    )

    current_gross = float(current_position_diag["gross_notional_from_positions"])
    notional_effective = float(notional) if notional is not None else current_gross

    search_rec = _discover_latest_quarantine_approved_portfolio_search(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
        as_of_ts=as_of_ts,
        lookback_days=int(lookback_days),
    )

    final_exec = search_rec["final_executable"]
    shadow_weights_raw = final_exec.get("weights") or {}

    shadow_weights, shadow_weight_diag = _normalize_weight_dict_to_returns(
        weights=shadow_weights_raw,
        resolver=resolver,
        returns_columns={str(c) for c in returns.columns},
    )

    current_metrics = evaluate_weights_for_search(
        returns=returns,
        weights=current_weights,
        equity0=float(equity0),
        notional=float(notional_effective),
        goals=list(goals),
        main_goal=float(main_goal),
        score_config=score_config,
        n_paths=int(n_paths_current),
        mc_seed=int(random_seed),
        block_size=(int(block_min), int(block_max)),
        weight_mode=str(weight_mode),
    )

    shadow_metrics = evaluate_weights_for_search(
        returns=returns,
        weights=shadow_weights,
        equity0=float(equity0),
        notional=float(notional_effective),
        goals=list(goals),
        main_goal=float(main_goal),
        score_config=score_config,
        n_paths=int(n_paths_shadow),
        mc_seed=int(random_seed) + 10_000,
        block_size=(int(block_min), int(block_max)),
        weight_mode=str(weight_mode),
    )

    current_health_score, raw_current_health = _load_current_health(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
    )

    shadow_health_score = _extract_health_score(final_exec)

    current_score = float(current_metrics.score)
    shadow_score = float(shadow_metrics.score)

    health_advantage = (
        None
        if current_health_score is None or shadow_health_score is None
        else float(shadow_health_score - current_health_score)
    )

    score_advantage = float(shadow_score - current_score)
    turnover = weight_turnover(current_weights, shadow_weights)
    delta = delta_weights(current_weights, shadow_weights)

    regime_fit = _load_regime_fit_diagnostics(
        cfg=cfg,
        returns=returns,
        as_of_date=as_of_date,
        transition_assessment=transition_assessment,
        current_weights=current_weights,
        candidate_weights=shadow_weights,
        candidate_name="shadow",
    )

    previous_shadow = _load_previous_shadow_state(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
    )

    days_active, days_dominating = _previous_shadow_counters(
        previous=previous_shadow,
        source_run_id=search_rec.get("run_id"),
        source_run_key=search_rec.get("run_key"),
    )

    shadow_id = str(search_rec.get("run_id") or f"shadow-{as_of_date}")

    shadow_state = ShadowPortfolioState(
        shadow_id=shadow_id,
        as_of=as_of_date,
        source_run_id=_safe_str(search_rec.get("run_id")),
        source_run_key=_safe_str(search_rec.get("run_key")),
        status="active",
        current_health_score=current_health_score,
        shadow_health_score=shadow_health_score,
        health_advantage=health_advantage,
        current_score=current_score,
        shadow_score=shadow_score,
        score_advantage=score_advantage,
        turnover=float(turnover),
        days_active=int(days_active),
        days_dominating=int(days_dominating),
        current_weights={k: float(v) for k, v in current_weights.items()},
        shadow_weights={k: float(v) for k, v in shadow_weights.items()},
        delta_weights={k: float(v) for k, v in delta.items()},
        diagnostics={
            "current_position_diagnostics": current_position_diag,
            "shadow_weight_diagnostics": shadow_weight_diag,
            "regime_fit": regime_fit,
            "source_search": {
                "candidate_source": "quarantine/candidates",
                "candidate_id": search_rec.get("candidate_id"),
                "candidate_state_key": search_rec.get("candidate_state_key"),
                "quarantine_status": (search_rec.get("quarantine") or {}).get("status"),
                "run_id": search_rec.get("run_id"),
                "run_key": search_rec.get("run_key"),
                "run_as_of": search_rec.get("run_as_of"),
                "age_days": search_rec.get("age_days"),
            },
        },
    )

    shadow_cfg = ShadowPortfolioConfig(
        min_health_advantage=float(min_health_advantage),
        min_score_advantage=float(min_score_advantage),
        max_turnover_to_accept=float(max_turnover_to_accept),
        confirmation_days=int(confirmation_days),
        immediate_accept_health_advantage=float(immediate_accept_health_advantage),
        immediate_accept_score_advantage=float(immediate_accept_score_advantage),
    )

    assessment = assess_shadow_portfolio(
        state=shadow_state,
        cfg=shadow_cfg,
    )

    payload = {
        "schema_version": "shadow_portfolio_assessment_v1",
        "as_of": as_of_date,
        "status": "success",
        "recommendation": assessment.recommendation,
        "reason": assessment.reason,
        "state": asdict(assessment.state),
        "config": asdict(assessment.config),
        "diagnostics": {
            **dict(assessment.diagnostics or {}),
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
                "weight_mode": str(weight_mode),
            },
            "current_metrics": asdict(current_metrics),
            "shadow_metrics": asdict(shadow_metrics),
            "current_health_latest": raw_current_health,
            "regime_fit": regime_fit,
        },
    }

    print("\n=== SHADOW PORTFOLIO ASSESSMENT ===")
    print(f"env:                    {getattr(cfg, 'env', 'unknown')}")
    print(f"bucket:                 {bucket}")
    print(f"root_prefix:            {root_prefix}")
    print(f"as_of:                  {as_of_date}")
    print(f"recommendation:         {payload['recommendation']}")
    print(f"reason:                 {payload['reason']}")
    print(f"source_run_id:          {assessment.state.source_run_id}")
    print(f"current_health_score:   {assessment.state.current_health_score}")
    print(f"shadow_health_score:    {assessment.state.shadow_health_score}")
    print(f"health_advantage:       {assessment.state.health_advantage}")
    print(f"current_score:          {assessment.state.current_score:.4f}")
    print(f"shadow_score:           {assessment.state.shadow_score:.4f}")
    print(f"score_advantage:        {assessment.state.score_advantage:.4f}")
    print(f"turnover:               {assessment.state.turnover:.2%}")
    print(f"days_active:            {assessment.state.days_active}")
    print(f"days_dominating:        {assessment.state.days_dominating}")

    rf = payload.get("diagnostics", {}).get("regime_fit") or {}
    if rf.get("status") == "success":
        comp = rf.get("comparison") or {}
        adv = comp.get("preference_score_advantage")
        if adv is not None:
            print(f"regime_fit_advantage:   {float(adv):.4f}")

    if write_outputs:
        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table=SHADOW_PORTFOLIO_TABLE,
            dt=as_of_ts,
            filename="shadow_portfolio_assessment.json",
            payload=payload,
            update_latest=update_latest,
        )

        print(
            f"\n[S3] Saved shadow portfolio assessment to "
            f"s3://{bucket}/{root_prefix}/{SHADOW_PORTFOLIO_TABLE}/dt={as_of_date}/"
        )

    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assess latest full-search portfolio as a shadow portfolio replacement candidate."
    )

    p.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    p.add_argument("--as-of", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-prod-write", action="store_true")

    p.add_argument("--equity0", "--equity-override", dest="equity0", type=float, default=None, help="Equity used as MC initial equity. If omitted, resolved from ledger + latest prices.")
    p.add_argument("--notional", type=float, default=None)
    p.add_argument("--goals", default="7500,10000,12500")
    p.add_argument("--main-goal", type=float, default=10000.0)

    p.add_argument("--cache-min-years", type=int, default=5)
    p.add_argument("--min-history-days", type=int, default=252 * 2)
    p.add_argument("--max-nan-frac", type=float, default=0.30)
    p.add_argument("--lookback-days", type=int, default=30)

    p.add_argument("--n-paths-current", type=int, default=5000)
    p.add_argument("--n-paths-shadow", type=int, default=5000)
    p.add_argument("--random-seed", type=int, default=123)

    p.add_argument("--weight-mode", default="long_short", choices=["long_only", "long_short", "gross_signed"])
    p.add_argument("--block-min", type=int, default=8)
    p.add_argument("--block-max", type=int, default=12)

    p.add_argument("--min-health-advantage", type=float, default=5.0)
    p.add_argument("--min-score-advantage", type=float, default=0.02)
    p.add_argument("--max-turnover-to-accept", type=float, default=0.35)
    p.add_argument("--confirmation-days", type=int, default=3)
    p.add_argument("--immediate-accept-health-advantage", type=float, default=10.0)
    p.add_argument("--immediate-accept-score-advantage", type=float, default=0.05)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = bool(getattr(args, "dry_run", False))

    with capture_script_run(
        cfg=cfg,
        script_name="run_shadow_portfolio_assessment.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            equity_result = resolve_current_equity(cfg=cfg, as_of=args.as_of, equity_override=args.equity0)
            print_equity_valuation_result(equity_result)

            payload = run_shadow_portfolio_assessment_job(
                cfg=cfg,
                as_of=args.as_of,
                write_outputs=not is_dry_run,
                update_latest=True,
                confirm_prod_write=bool(args.confirm_prod_write),
                equity0=float(equity_result.equity),
                notional=args.notional,
                goals=_parse_goals(args.goals),
                main_goal=float(args.main_goal),
                cache_min_years=int(args.cache_min_years),
                min_history_days=int(args.min_history_days),
                max_nan_frac=float(args.max_nan_frac),
                lookback_days=int(args.lookback_days),
                n_paths_current=int(args.n_paths_current),
                n_paths_shadow=int(args.n_paths_shadow),
                weight_mode=str(args.weight_mode),
                block_min=int(args.block_min),
                block_max=int(args.block_max),
                min_health_advantage=float(args.min_health_advantage),
                min_score_advantage=float(args.min_score_advantage),
                max_turnover_to_accept=float(args.max_turnover_to_accept),
                confirmation_days=int(args.confirmation_days),
                immediate_accept_health_advantage=float(args.immediate_accept_health_advantage),
                immediate_accept_score_advantage=float(args.immediate_accept_score_advantage),
                random_seed=int(args.random_seed),
            )

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="create",
                entity_type="shadow_portfolio_assessment",
                entity_id=str(payload.get("as_of")),
                as_of=str(payload.get("as_of")),
                source_script="run_shadow_portfolio_assessment.py",
                source_mode="shadow_portfolio_assessment",
                status=("dry_run" if is_dry_run else "success"),
                input_args=vars(args),
                output_keys=[] if is_dry_run else [
                    f"{cfg_engine_root(cfg)}/{SHADOW_PORTFOLIO_TABLE}/dt={payload.get('as_of')}/shadow_portfolio_assessment.json",
                    f"{cfg_engine_root(cfg)}/{SHADOW_PORTFOLIO_TABLE}/latest.json",
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
                entity_type="shadow_portfolio_assessment",
                entity_id=None,
                as_of=str(getattr(args, "as_of", "") or ""),
                source_script="run_shadow_portfolio_assessment.py",
                source_mode="shadow_portfolio_assessment",
                status="failed",
                input_args=vars(args),
                metadata={
                    "tier": "shadow_portfolio_assessment",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise


if __name__ == "__main__":
    main()