from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run

import argparse
import datetime as dt
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from alpha_edge.core.data_loader import (
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
from alpha_edge.core.schemas import TransitionExecutionConfig
from alpha_edge.portfolio.execution_engine import build_transition_execution_plan


DEFAULT_ENGINE_BUCKET = "alpha-edge-algo"
DEFAULT_ENGINE_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT_PREFIX = "engine/v1"
DEFAULT_MARKET_ROOT = "market"

LOCAL_OPTIMIZER_TABLE = "portfolio_transition/local_optimizer"
SHADOW_PORTFOLIO_TABLE = "portfolio_transition/shadow"
TRANSITION_EXECUTION_PLAN_TABLE = "portfolio_transition/execution_plan"


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


def _load_current_positions(
    *,
    s3,
    bucket: str,
    root_prefix: str,
    as_of_date: str,
) -> dict[str, float]:
    """
    Load current ledger positions.

    This intentionally expects the dated ledger snapshot to exist.
    If the ledger for as_of_date is missing, the runner should fail.
    """
    ledger_key = f"{root_prefix.strip('/')}/ledger/dt={as_of_date}/positions.json"
    raw = s3_get_json(s3, bucket=bucket, key=ledger_key)

    if not raw:
        raise RuntimeError(
            f"Missing current ledger positions: s3://{bucket}/{ledger_key}"
        )

    parsed = parse_positions_obj(raw)

    out: dict[str, float] = {}

    for key, pos in parsed.items():
        raw_key = getattr(pos, "asset_id", None) or getattr(pos, "ticker", None) or key
        qty = _safe_float(getattr(pos, "quantity", None))

        if raw_key and qty is not None and abs(qty) > 0:
            out[str(raw_key)] = float(qty)

    if not out:
        raise RuntimeError(
            f"Loaded ledger positions from s3://{bucket}/{ledger_key}, "
            "but no non-zero quantities were found."
        )

    return out


def _load_latest_local_optimizer(
    *,
    s3,
    bucket: str,
    root_prefix: str,
) -> dict[str, Any] | None:
    raw = s3_load_latest_json(
        s3,
        bucket=bucket,
        root_prefix=root_prefix,
        table=LOCAL_OPTIMIZER_TABLE,
    )

    return raw if isinstance(raw, dict) else None


def _load_latest_shadow_assessment(
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


def _select_target_weights(
    *,
    local_payload: dict[str, Any] | None,
    shadow_payload: dict[str, Any] | None,
    prefer_shadow: bool = True,
) -> tuple[str, dict[str, float], dict[str, Any]]:
    """
    Select the accepted transition target.

    Priority:
      1. SHADOW_ACCEPTED, if prefer_shadow=True
      2. LOCAL_REBALANCE_RECOMMENDED
      3. SHADOW_ACCEPTED, if prefer_shadow=False and local is unavailable

    Returns:
      source, target_weights, source_payload_summary
    """

    def shadow_candidate() -> tuple[str, dict[str, float], dict[str, Any]] | None:
        if not isinstance(shadow_payload, dict):
            return None

        rec = str(shadow_payload.get("recommendation") or "").strip()

        if rec != "SHADOW_ACCEPTED":
            return None

        state = shadow_payload.get("state") or {}
        if not isinstance(state, dict):
            return None

        weights = state.get("shadow_weights") or {}
        if not isinstance(weights, dict) or not weights:
            return None

        return (
            "shadow",
            {str(k): float(v) for k, v in weights.items() if abs(float(v)) > 1e-12},
            {
                "recommendation": rec,
                "reason": shadow_payload.get("reason"),
                "shadow_id": state.get("shadow_id"),
                "source_run_id": state.get("source_run_id"),
                "source_run_key": state.get("source_run_key"),
                "score_advantage": state.get("score_advantage"),
                "health_advantage": state.get("health_advantage"),
                "turnover": state.get("turnover"),
            },
        )

    def local_candidate() -> tuple[str, dict[str, float], dict[str, Any]] | None:
        if not isinstance(local_payload, dict):
            return None

        rec = str(local_payload.get("recommendation") or "").strip()

        if rec != "LOCAL_REBALANCE_RECOMMENDED":
            return None

        best = local_payload.get("best_candidate") or {}
        if not isinstance(best, dict):
            return None

        weights = best.get("weights") or {}
        if not isinstance(weights, dict) or not weights:
            return None

        return (
            "local_optimizer",
            {str(k): float(v) for k, v in weights.items() if abs(float(v)) > 1e-12},
            {
                "recommendation": rec,
                "reason": local_payload.get("reason"),
                "score": best.get("score"),
                "score_improvement": best.get("score_improvement"),
                "turnover": best.get("turnover"),
            },
        )

    if prefer_shadow:
        for fn in [shadow_candidate, local_candidate]:
            result = fn()
            if result is not None:
                return result
    else:
        for fn in [local_candidate, shadow_candidate]:
            result = fn()
            if result is not None:
                return result

    raise RuntimeError(
        "No accepted transition target found. Expected either "
        "local_optimizer recommendation LOCAL_REBALANCE_RECOMMENDED "
        "or shadow recommendation SHADOW_ACCEPTED."
    )


def run_transition_execution_plan_job(
    *,
    cfg: RuntimeConfig,
    as_of: str | None = None,
    write_outputs: bool = True,
    update_latest: bool = True,
    confirm_prod_write: bool = False,
    notional: float | None = None,
    prefer_shadow: bool = True,
    max_total_turnover: float = 0.35,
    max_daily_turnover: float = 0.10,
    min_trade_value: float = 25.0,
    min_trade_quantity: float = 0.0,
    allow_partial_transition: bool = True,
    min_weight: float = 0.01,
    min_units_equity: float = 1.0,
    min_units_crypto: float = 0.0,
    min_units_weight_thr: float = 0.03,
    crypto_decimals: int = 8,
) -> dict[str, Any]:
    if write_outputs:
        require_prod_confirmation(cfg, bool(confirm_prod_write))

    bucket = cfg_bucket(cfg)
    region = cfg_region(cfg)
    root_prefix = cfg_engine_root(cfg)

    as_of_ts = pd.Timestamp(as_of or dt.date.today()).tz_localize(None).normalize()
    as_of_date = as_of_ts.strftime("%Y-%m-%d")

    s3 = s3_init(region)

    local_payload = _load_latest_local_optimizer(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
    )

    shadow_payload = _load_latest_shadow_assessment(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
    )

    try:
        source, target_weights, target_source_summary = _select_target_weights(
            local_payload=local_payload,
            shadow_payload=shadow_payload,
            prefer_shadow=bool(prefer_shadow),
        )
    except RuntimeError as exc:
        payload = {
            "schema_version": "transition_execution_plan_v1",
            "as_of": as_of_date,
            "status": "skipped",
            "recommendation": "NO_TRADE",
            "reason": str(exc),
            "source": None,
            "notional": None,
            "total_turnover": 0.0,
            "daily_turnover_used": 0.0,
            "blocked_turnover": 0.0,
            "target_allocation": None,
            "trades": [],
            "blocked_trades": [],
            "config": {
                "max_total_turnover": float(max_total_turnover),
                "max_daily_turnover": float(max_daily_turnover),
                "min_trade_value": float(min_trade_value),
                "min_trade_quantity": float(min_trade_quantity),
                "allow_partial_transition": bool(allow_partial_transition),
            },
            "diagnostics": {
                "local_optimizer": {
                    "found": isinstance(local_payload, dict),
                    "recommendation": None if not isinstance(local_payload, dict) else local_payload.get("recommendation"),
                    "status": None if not isinstance(local_payload, dict) else local_payload.get("status"),
                    "reason": None if not isinstance(local_payload, dict) else local_payload.get("reason"),
                },
                "shadow": {
                    "found": isinstance(shadow_payload, dict),
                    "recommendation": None if not isinstance(shadow_payload, dict) else shadow_payload.get("recommendation"),
                    "status": None if not isinstance(shadow_payload, dict) else shadow_payload.get("status"),
                    "reason": None if not isinstance(shadow_payload, dict) else shadow_payload.get("reason"),
                },
            },
        }

        print("\n=== TRANSITION EXECUTION PLAN ===")
        print(f"env:                  {getattr(cfg, 'env', 'unknown')}")
        print(f"bucket:               {bucket}")
        print(f"root_prefix:          {root_prefix}")
        print(f"as_of:                {as_of_date}")
        print("status:               skipped")
        print("recommendation:       NO_TRADE")
        print(f"reason:               {payload['reason']}")

        if write_outputs:
            s3_write_json_event(
                s3,
                bucket=bucket,
                root_prefix=root_prefix,
                table=TRANSITION_EXECUTION_PLAN_TABLE,
                dt=as_of_ts,
                filename="transition_execution_plan.json",
                payload=payload,
                update_latest=update_latest,
            )

        return payload

    current_shares = _load_current_positions(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
        as_of_date=as_of_date,
    )

    prices = _latest_prices_by_asset_id(cfg)

    if notional is None:
        gross = 0.0
        missing_price: list[str] = []

        for asset_id, qty in current_shares.items():
            px = _safe_float(prices.get(asset_id))
            if px is None or px <= 0:
                missing_price.append(str(asset_id))
                continue
            gross += abs(float(qty) * float(px))

        if missing_price:
            raise RuntimeError(
                "Cannot infer notional from current positions because some "
                f"assets are missing prices. missing_price_sample={missing_price[:20]}"
            )

        if gross <= 0:
            raise RuntimeError("Cannot infer notional because current gross exposure is zero.")

        notional_effective = float(gross)
    else:
        notional_effective = float(notional)

    exec_cfg = TransitionExecutionConfig(
        max_total_turnover=float(max_total_turnover),
        max_daily_turnover=float(max_daily_turnover),
        min_trade_value=float(min_trade_value),
        min_trade_quantity=float(min_trade_quantity),
        allow_partial_transition=bool(allow_partial_transition),
    )

    plan = build_transition_execution_plan(
        as_of=as_of_date,
        source=source,
        current_shares=current_shares,
        target_weights=target_weights,
        prices=prices,
        notional=float(notional_effective),
        cfg=exec_cfg,
        min_weight=float(min_weight),
        min_units_equity=float(min_units_equity),
        min_units_crypto=float(min_units_crypto),
        min_units_weight_thr=float(min_units_weight_thr),
        crypto_decimals=int(crypto_decimals),
    )

    payload = {
        "schema_version": "transition_execution_plan_v1",
        "as_of": as_of_date,
        "status": "success",
        "recommendation": plan.recommendation,
        "reason": plan.reason,
        "source": plan.source,
        "notional": float(plan.notional),
        "total_turnover": float(plan.total_turnover),
        "daily_turnover_used": float(plan.daily_turnover_used),
        "blocked_turnover": float(plan.blocked_turnover),
        "target_allocation": asdict(plan.target_allocation),
        "trades": [asdict(t) for t in plan.trades],
        "blocked_trades": [asdict(t) for t in plan.blocked_trades],
        "config": asdict(plan.config),
        "diagnostics": {
            **dict(plan.diagnostics or {}),
            "target_source": target_source_summary,
            "input_counts": {
                "current_share_count": int(len(current_shares)),
                "target_weight_count": int(len(target_weights)),
                "price_count": int(len(prices)),
            },
        },
    }

    print("\n=== TRANSITION EXECUTION PLAN ===")
    print(f"env:                  {getattr(cfg, 'env', 'unknown')}")
    print(f"bucket:               {bucket}")
    print(f"root_prefix:          {root_prefix}")
    print(f"as_of:                {as_of_date}")
    print(f"source:               {payload['source']}")
    print(f"recommendation:       {payload['recommendation']}")
    print(f"reason:               {payload['reason']}")
    print(f"notional:             {float(payload['notional']):,.2f}")
    print(f"total_turnover:       {float(payload['total_turnover']):.2%}")
    print(f"daily_turnover_used:  {float(payload['daily_turnover_used']):.2%}")
    print(f"blocked_turnover:     {float(payload['blocked_turnover']):.2%}")
    print(f"trade_count:          {len(payload['trades'])}")
    print(f"blocked_trade_count:  {len(payload['blocked_trades'])}")

    if payload["trades"]:
        print("\nRecommended trades:")
        for t in payload["trades"][:20]:
            print(
                f"  {t['direction']:>4} {t['asset_id']:<24} "
                f"qty={float(t['delta_quantity']):,.8f} "
                f"value={float(t['delta_value']):,.2f}"
            )

    if write_outputs:
        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table=TRANSITION_EXECUTION_PLAN_TABLE,
            dt=as_of_ts,
            filename="transition_execution_plan.json",
            payload=payload,
            update_latest=update_latest,
        )

        print(
            f"\n[S3] Saved transition execution plan to "
            f"s3://{bucket}/{root_prefix}/{TRANSITION_EXECUTION_PLAN_TABLE}/dt={as_of_date}/"
        )

    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build recommendation-only transition execution plan."
    )

    p.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    p.add_argument("--as-of", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-prod-write", action="store_true")

    p.add_argument("--notional", type=float, default=None)
    p.add_argument("--prefer-local", action="store_true")

    p.add_argument("--max-total-turnover", type=float, default=0.35)
    p.add_argument("--max-daily-turnover", type=float, default=0.10)
    p.add_argument("--min-trade-value", type=float, default=25.0)
    p.add_argument("--min-trade-quantity", type=float, default=0.0)
    p.add_argument("--disable-partial-transition", action="store_true")

    p.add_argument("--min-weight", type=float, default=0.01)
    p.add_argument("--min-units-equity", type=float, default=1.0)
    p.add_argument("--min-units-crypto", type=float, default=0.0)
    p.add_argument("--min-units-weight-thr", type=float, default=0.03)
    p.add_argument("--crypto-decimals", type=int, default=8)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = bool(getattr(args, "dry_run", False))

    with capture_script_run(
        cfg=cfg,
        script_name="run_transition_execution_plan.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            payload = run_transition_execution_plan_job(
                cfg=cfg,
                as_of=args.as_of,
                write_outputs=not is_dry_run,
                update_latest=True,
                confirm_prod_write=bool(args.confirm_prod_write),
                notional=args.notional,
                prefer_shadow=not bool(args.prefer_local),
                max_total_turnover=float(args.max_total_turnover),
                max_daily_turnover=float(args.max_daily_turnover),
                min_trade_value=float(args.min_trade_value),
                min_trade_quantity=float(args.min_trade_quantity),
                allow_partial_transition=not bool(args.disable_partial_transition),
                min_weight=float(args.min_weight),
                min_units_equity=float(args.min_units_equity),
                min_units_crypto=float(args.min_units_crypto),
                min_units_weight_thr=float(args.min_units_weight_thr),
                crypto_decimals=int(args.crypto_decimals),
            )

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="create",
                entity_type="transition_execution_plan",
                entity_id=str(payload.get("as_of")),
                as_of=str(payload.get("as_of")),
                source_script="run_transition_execution_plan.py",
                source_mode="transition_execution_plan",
                status=("dry_run" if is_dry_run else "success"),
                input_args=vars(args),
                output_keys=[] if is_dry_run else [
                    f"{cfg_engine_root(cfg)}/{TRANSITION_EXECUTION_PLAN_TABLE}/dt={payload.get('as_of')}/transition_execution_plan.json",
                    f"{cfg_engine_root(cfg)}/{TRANSITION_EXECUTION_PLAN_TABLE}/latest.json",
                ],
                metadata={
                    "recommendation": payload.get("recommendation"),
                    "source": payload.get("source"),
                    "trade_count": len(payload.get("trades") or []),
                    "blocked_trade_count": len(payload.get("blocked_trades") or []),
                    "reason": payload.get("reason"),
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)

        except Exception as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="create",
                entity_type="transition_execution_plan",
                entity_id=None,
                as_of=str(getattr(args, "as_of", "") or ""),
                source_script="run_transition_execution_plan.py",
                source_mode="transition_execution_plan",
                status="failed",
                input_args=vars(args),
                metadata={
                    "tier": "transition_execution_plan",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise


if __name__ == "__main__":
    main()
