from __future__ import annotations

import argparse
import io
import json
from typing import Any, Optional

import boto3
import pandas as pd

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import (
    RuntimeConfig,
    load_runtime_config,
    require_prod_confirmation,
    runtime_dt_key,
    runtime_engine_key,
)
from alpha_edge.portfolio.open_position_barriers import (
    BarrierMonitorConfig,
    open_lots_from_ledger_trace,
    run_barrier_monitor,
)


BARRIER_TABLE = "open_position_barriers"
LEDGER_TABLE = "ledger"
TRADES_TABLE = "trades"


def _s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


def _s3_get_json(s3, *, bucket: str, key: str) -> dict[str, Any]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    payload = json.loads(obj["Body"].read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _s3_get_json_optional(s3, *, bucket: str, key: str) -> Optional[dict[str, Any]]:
    try:
        return _s3_get_json(s3, bucket=bucket, key=key)
    except Exception:
        return None


def _s3_put_json(s3, *, bucket: str, key: str, payload: dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )


def _s3_put_parquet(s3, *, bucket: str, key: str, df: pd.DataFrame) -> None:
    bio = io.BytesIO()
    df.to_parquet(bio, index=False)
    bio.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=bio.read(), ContentType="application/octet-stream")


def _latest_open_lot_trace_key(cfg: RuntimeConfig) -> str:
    return runtime_engine_key(cfg, LEDGER_TABLE, "open_lot_trace", "latest.json")


def _dated_open_lot_trace_key(cfg: RuntimeConfig, as_of: str) -> str:
    return runtime_dt_key(cfg, LEDGER_TABLE, as_of, "open_lot_trace.json")


def _latest_positions_key(cfg: RuntimeConfig) -> str:
    return runtime_engine_key(cfg, LEDGER_TABLE, "positions", "latest.json")


def _dated_positions_key(cfg: RuntimeConfig, as_of: str) -> str:
    return runtime_dt_key(cfg, LEDGER_TABLE, as_of, "positions.json")


def _load_positions_snapshot(
    *,
    s3,
    cfg: RuntimeConfig,
    as_of: str,
    allow_latest_fallback: bool,
) -> tuple[dict[str, Any], str]:
    dated_key = _dated_positions_key(cfg, as_of)
    payload = _s3_get_json_optional(s3, bucket=cfg.bucket, key=dated_key)
    if payload:
        return payload, dated_key

    if allow_latest_fallback:
        latest_key = _latest_positions_key(cfg)
        payload = _s3_get_json_optional(s3, bucket=cfg.bucket, key=latest_key)
        if payload:
            return payload, latest_key

    raise RuntimeError(
        "Missing positions snapshot. Run rebuild_ledger.py first. "
        f"Expected s3://{cfg.bucket}/{dated_key}"
    )


def _positive_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _current_position_keys(positions_payload: dict[str, Any]) -> tuple[set[tuple[str, str]], dict[str, dict[str, Any]]]:
    """Return monitorable current (asset_id, side) keys from ledger positions.

    This protects the barrier monitor from stale open_lot_trace rows. The
    positions snapshot is the ledger source of truth for currently open
    positions; open_lot_trace is used only to recover lot-level entry details.
    """
    keys: set[tuple[str, str]] = set()
    meta: dict[str, dict[str, Any]] = {}

    for row in positions_payload.get("spot_positions") or []:
        if not isinstance(row, dict):
            continue
        asset_id = str(row.get("asset_id") or "").strip()
        if not asset_id:
            continue
        qty = _positive_float(row.get("quantity"))
        if qty is None:
            continue
        key = (asset_id, "LONG")
        keys.add(key)
        meta["|".join(key)] = row

    for row in positions_payload.get("derivatives_positions") or []:
        if not isinstance(row, dict):
            continue
        asset_id = str(row.get("asset_id") or "").strip()
        if not asset_id:
            continue

        side_raw = str(row.get("side") or row.get("position_side") or row.get("direction") or "").upper().strip()
        if side_raw in {"SHORT", "SELL", "SHORT_NOTIONAL"}:
            side = "SHORT"
        elif side_raw in {"LONG", "BUY", "LONG_NOTIONAL"}:
            side = "LONG"
        else:
            open_notional = row.get("open_notional_usd", row.get("notional", row.get("market_value")))
            try:
                side = "SHORT" if float(open_notional) < 0 else "LONG"
            except Exception:
                side = "LONG"

        key = (asset_id, side)
        keys.add(key)
        meta["|".join(key)] = row

    return keys, meta


def _filter_open_lots_to_current_positions(open_lots, current_keys: set[tuple[str, str]]):
    filtered = [lot for lot in open_lots if (lot.asset_id, lot.side) in current_keys]
    removed = len(open_lots) - len(filtered)
    return filtered, removed


def _trades_index_key(cfg: RuntimeConfig) -> str:
    return runtime_engine_key(cfg, TRADES_TABLE, "index.json")


def _load_open_lot_trace(
    *,
    s3,
    cfg: RuntimeConfig,
    as_of: str,
    allow_latest_fallback: bool,
) -> tuple[dict[str, Any], str]:
    dated_key = _dated_open_lot_trace_key(cfg, as_of)
    payload = _s3_get_json_optional(s3, bucket=cfg.bucket, key=dated_key)
    if payload:
        return payload, dated_key

    if allow_latest_fallback:
        latest_key = _latest_open_lot_trace_key(cfg)
        payload = _s3_get_json_optional(s3, bucket=cfg.bucket, key=latest_key)
        if payload:
            return payload, latest_key

    raise RuntimeError(
        "Missing open lot trace. Run rebuild_ledger.py first. "
        f"Expected s3://{cfg.bucket}/{dated_key}"
    )


def _load_trade_contracts_for_lots(
    *,
    s3,
    cfg: RuntimeConfig,
    open_lot_trace: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    raw_lots = open_lot_trace.get("open_lots") if isinstance(open_lot_trace.get("open_lots"), dict) else open_lot_trace
    if not isinstance(raw_lots, dict):
        return {}, []

    trade_ids: set[str] = set()
    for lot_key in ["spot_long_lots", "spot_short_lots", "notional_long_lots", "notional_short_lots"]:
        for lot in raw_lots.get(lot_key) or []:
            if isinstance(lot, dict) and lot.get("trade_id"):
                trade_ids.add(str(lot["trade_id"]).strip())

    if not trade_ids:
        return {}, []

    index_key = _trades_index_key(cfg)
    index = _s3_get_json_optional(s3, bucket=cfg.bucket, key=index_key) or {}
    out: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    risk_fields = {
        "risk_model",
        "stop_loss_price",
        "target_profit_price",
        "stop_loss_pct",
        "target_profit_pct",
        "risk_amount",
        "target_profit_amount",
        "risk_reward_ratio",
        "max_holding_days",
        "expiry_date",
        "time_barrier_method",
        "barrier_status",
    }

    for trade_id in sorted(trade_ids):
        meta = index.get(trade_id)
        key = meta.get("key") if isinstance(meta, dict) else None
        if not key:
            warnings.append(f"trade_contract_missing_index:{trade_id}")
            continue
        payload = _s3_get_json_optional(s3, bucket=cfg.bucket, key=str(key))
        if not payload:
            warnings.append(f"trade_contract_missing_payload:{trade_id}")
            continue
        out[trade_id] = {k: payload.get(k) for k in risk_fields if k in payload}

    return out, warnings


def _read_latest_indicators_snapshot(cfg: RuntimeConfig) -> tuple[pd.DataFrame, Optional[str]]:
    store = MarketStore(bucket=cfg.bucket, region=cfg.region, base_prefix=cfg.market_root)
    key = f"{store.snapshots_prefix}/latest_indicators.parquet"
    try:
        raw = store._get_bytes(key)
        return pd.read_parquet(io.BytesIO(raw)), key
    except Exception:
        return pd.DataFrame(), None


def _load_market_inputs(
    *,
    cfg: RuntimeConfig,
    asset_ids: list[str],
    start: str,
    end: str,
    include_indicators: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    store = MarketStore(bucket=cfg.bucket, region=cfg.region, base_prefix=cfg.market_root)
    latest_prices = store.read_latest_prices_snapshot()

    columns = [
        "date",
        "asset_id",
        "ticker",
        "open_raw_usd",
        "open_usd",
        "open_adjusted_usd",
        "high_raw_usd",
        "low_raw_usd",
        "high_usd",
        "low_usd",
        "close_raw_usd",
        "close_usd",
        "close_adjusted_usd",
        "adj_close_usd",
    ]
    ohlcv = store.read_ohlcv_usd(asset_ids=asset_ids, start=start, end=end, columns=columns)

    indicators = pd.DataFrame()
    indicators_key = None
    if include_indicators:
        indicators, indicators_key = _read_latest_indicators_snapshot(cfg)

    sources = {
        "latest_prices": f"s3://{cfg.bucket}/{store.snapshots_prefix}/latest_prices.parquet",
        "ohlcv_root": f"s3://{cfg.bucket}/{store.ohlcv_prefix}/",
        "latest_indicators": (f"s3://{cfg.bucket}/{indicators_key}" if indicators_key else None),
    }
    return latest_prices, ohlcv, indicators, sources


def _load_asset_class_overrides(raw: Optional[str]) -> dict[str, dict[str, Any]]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--asset-class-overrides-json must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--asset-class-overrides-json must be a JSON object.")
    out: dict[str, dict[str, Any]] = {}
    for key, val in payload.items():
        if isinstance(val, dict):
            out[str(key).lower().strip()] = dict(val)
    return out


def _latest_barrier_state_key(cfg: RuntimeConfig) -> str:
    return runtime_engine_key(cfg, BARRIER_TABLE, "latest_state.json")


def _dated_barrier_state_key(cfg: RuntimeConfig, as_of: str) -> str:
    return runtime_dt_key(cfg, BARRIER_TABLE, as_of, "state.json")


def _load_previous_barrier_state(*, s3, cfg: RuntimeConfig, disable_state: bool) -> tuple[dict[str, Any], Optional[str]]:
    if disable_state:
        return {}, None
    key = _latest_barrier_state_key(cfg)
    payload = _s3_get_json_optional(s3, bucket=cfg.bucket, key=key)
    if payload:
        return payload, key
    return {}, None


def _config_from_args(args: argparse.Namespace) -> BarrierMonitorConfig:
    return BarrierMonitorConfig(
        default_stop_loss_pct=float(args.default_stop_loss_pct),
        default_target_profit_pct=float(args.default_target_profit_pct),
        chandelier_atr_multiplier=float(args.chandelier_atr_multiplier),
        profit_lock_activation_return=float(args.profit_lock_activation_return),
        profit_lock_ratio=float(args.profit_lock_ratio),
        near_stop_threshold_pct=float(args.near_stop_threshold_pct),
        near_target_threshold_pct=float(args.near_target_threshold_pct),
        max_price_staleness_days=int(args.max_price_staleness_days),
        max_indicator_staleness_days=int(args.max_indicator_staleness_days),
        use_fixed_stop=not bool(args.disable_fixed_stop),
        use_fixed_target=not bool(args.disable_fixed_target),
        use_chandelier_stop=not bool(args.disable_chandelier_stop),
        use_profit_lock=not bool(args.disable_profit_lock),
        fail_on_missing_history=bool(args.fail_on_missing_history),
        fail_on_missing_atr=bool(args.fail_on_missing_atr),
        liquidity_aware_stops=not bool(args.disable_liquidity_aware_stops),
        stop_buffer_atr_multiplier=float(args.stop_buffer_atr_multiplier),
        stop_buffer_pct=float(args.stop_buffer_pct),
        close_breach_reduce_threshold_pct=float(args.close_breach_reduce_threshold_pct),
        confirmed_break_breach_count=int(args.confirmed_break_breach_count),
        emergency_gap_atr_multiplier=float(args.emergency_gap_atr_multiplier),
        emergency_gap_pct=float(args.emergency_gap_pct),
        conservative_advisory_mode=not bool(args.disable_conservative_advisory_mode),
        time_exit_action=str(args.time_exit_action).upper(),
        time_exit_exit_fraction=float(args.time_exit_exit_fraction),
        severe_breach_pct=float(args.severe_breach_pct),
        severe_breach_atr_multiplier=float(args.severe_breach_atr_multiplier),
        confirmed_break_exit_fraction=float(args.confirmed_break_exit_fraction),
        confirmed_break_small_action=str(args.confirmed_break_small_action).upper(),
        confirmed_break_small_exit_fraction=float(args.confirmed_break_small_exit_fraction),
        gap_through_stop_exit_fraction=float(args.gap_through_stop_exit_fraction),
        apply_liquidation_throttle=not bool(args.disable_liquidation_throttle),
        max_daily_exit_fraction_of_portfolio=float(args.max_daily_exit_fraction_of_portfolio),
        max_daily_exit_value=args.max_daily_exit_value,
        max_full_exits_per_day=int(args.max_full_exits_per_day),
        max_reduce_actions_per_day=int(args.max_reduce_actions_per_day),
        asset_class_overrides=_load_asset_class_overrides(args.asset_class_overrides_json),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run daily open-position SL/TP/trailing barrier monitor.")
    p.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    p.add_argument("--as-of", default=None, help="Market date YYYY-MM-DD. Default: today UTC.")
    p.add_argument("--write", action="store_true", help="Persist barrier monitor outputs to S3.")
    p.add_argument("--confirm-prod-write", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Do not persist outputs. Equivalent to omitting --write, but still writes dry-run audit/log markers.")
    p.add_argument("--no-latest-ledger-fallback", action="store_true", help="Require dated open_lot_trace and positions snapshot for --as-of.")
    p.add_argument("--disable-position-snapshot-filter", action="store_true", help="Do not filter open_lot_trace lots against ledger positions/latest.json. Intended only for debugging stale lot traces.")
    p.add_argument("--disable-indicators", action="store_true", help="Skip latest_indicators snapshot; Chandelier may degrade without ATR.")
    p.add_argument("--disable-state", action="store_true", help="Do not read previous barrier state; useful for deterministic first-run validation.")

    p.add_argument("--default-stop-loss-pct", type=float, default=0.08)
    p.add_argument("--default-target-profit-pct", type=float, default=0.15)
    p.add_argument("--chandelier-atr-multiplier", type=float, default=3.0)
    p.add_argument("--profit-lock-activation-return", type=float, default=0.10)
    p.add_argument("--profit-lock-ratio", type=float, default=0.50)
    p.add_argument("--near-stop-threshold-pct", type=float, default=0.02)
    p.add_argument("--near-target-threshold-pct", type=float, default=0.02)
    p.add_argument("--max-price-staleness-days", type=int, default=5)
    p.add_argument("--max-indicator-staleness-days", type=int, default=10)
    p.add_argument("--fail-on-missing-history", action="store_true")
    p.add_argument("--fail-on-missing-atr", action="store_true")

    p.add_argument("--disable-liquidity-aware-stops", action="store_true")
    p.add_argument("--stop-buffer-atr-multiplier", type=float, default=0.25)
    p.add_argument("--stop-buffer-pct", type=float, default=0.003)
    p.add_argument("--close-breach-reduce-threshold-pct", type=float, default=0.015)
    p.add_argument("--confirmed-break-breach-count", type=int, default=2)
    p.add_argument("--emergency-gap-atr-multiplier", type=float, default=1.0)
    p.add_argument("--emergency-gap-pct", type=float, default=0.03)
    p.add_argument("--asset-class-overrides-json", default=None, help="JSON object of asset bucket overrides, e.g. {\"crypto\":{\"stop_buffer_pct\":0.01}}")

    p.add_argument("--disable-conservative-advisory-mode", action="store_true", help="Allow v2-style aggressive EXIT actions. Default v1 behavior is conservative.")
    p.add_argument("--time-exit-action", default="WATCH", choices=["HOLD", "WATCH", "REDUCE", "EXIT", "DATA_ISSUE"])
    p.add_argument("--time-exit-exit-fraction", type=float, default=0.0)
    p.add_argument("--severe-breach-pct", type=float, default=0.05)
    p.add_argument("--severe-breach-atr-multiplier", type=float, default=2.0)
    p.add_argument("--confirmed-break-exit-fraction", type=float, default=0.50)
    p.add_argument("--confirmed-break-small-action", default="REDUCE", choices=["HOLD", "WATCH", "REDUCE", "EXIT", "DATA_ISSUE"])
    p.add_argument("--confirmed-break-small-exit-fraction", type=float, default=0.25)
    p.add_argument("--gap-through-stop-exit-fraction", type=float, default=1.0)

    p.add_argument("--disable-liquidation-throttle", action="store_true")
    p.add_argument("--max-daily-exit-fraction-of-portfolio", type=float, default=0.25)
    p.add_argument("--max-daily-exit-value", type=float, default=None)
    p.add_argument("--max-full-exits-per-day", type=int, default=2)
    p.add_argument("--max-reduce-actions-per-day", type=int, default=8)

    p.add_argument("--disable-fixed-stop", action="store_true")
    p.add_argument("--disable-fixed-target", action="store_true")
    p.add_argument("--disable-chandelier-stop", action="store_true")
    p.add_argument("--disable-profit-lock", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(args.env)
    as_of = pd.Timestamp(args.as_of).date().strftime("%Y-%m-%d") if args.as_of else pd.Timestamp.utcnow().date().strftime("%Y-%m-%d")
    should_write = bool(args.write) and not bool(args.dry_run)

    if should_write:
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    input_args = vars(args)
    with capture_script_run(
        cfg=cfg,
        script_name="run_open_position_barrier_monitor.py",
        input_args=input_args,
        dry_run=not should_write,
    ) as run_id:
        print(f"[runtime] env={cfg.env} bucket={cfg.bucket} root={cfg.engine_root} market_root={cfg.market_root}")
        print(f"[barriers] as_of={as_of} write={should_write}")

        s3 = _s3_client(cfg)
        previous_state, previous_state_key = _load_previous_barrier_state(
            s3=s3,
            cfg=cfg,
            disable_state=bool(args.disable_state),
        )

        open_lot_trace, open_lot_trace_key = _load_open_lot_trace(
            s3=s3,
            cfg=cfg,
            as_of=as_of,
            allow_latest_fallback=not bool(args.no_latest_ledger_fallback),
        )
        trade_contracts, contract_warnings = _load_trade_contracts_for_lots(
            s3=s3,
            cfg=cfg,
            open_lot_trace=open_lot_trace,
        )
        open_lots = open_lots_from_ledger_trace(
            open_lot_trace,
            trade_contracts_by_trade_id=trade_contracts,
        )
        if not open_lots:
            raise RuntimeError("Open lot trace exists but contains no open lots to monitor.")

        positions_key = None
        stale_lots_removed = 0
        if not bool(args.disable_position_snapshot_filter):
            positions_payload, positions_key = _load_positions_snapshot(
                s3=s3,
                cfg=cfg,
                as_of=as_of,
                allow_latest_fallback=not bool(args.no_latest_ledger_fallback),
            )
            current_keys, _current_meta = _current_position_keys(positions_payload)
            if not current_keys:
                raise RuntimeError("Positions snapshot contains no current positions to monitor.")
            open_lots, stale_lots_removed = _filter_open_lots_to_current_positions(open_lots, current_keys)
            if not open_lots:
                raise RuntimeError(
                    "Open lot trace has no lots matching current ledger positions. "
                    "This usually means open_lot_trace is stale or inconsistent with positions/latest.json."
                )

        asset_ids = sorted({lot.asset_id for lot in open_lots})
        start = min(str(lot.entry_date) for lot in open_lots)
        latest_prices, ohlcv, indicators, market_sources = _load_market_inputs(
            cfg=cfg,
            asset_ids=asset_ids,
            start=start,
            end=as_of,
            include_indicators=not bool(args.disable_indicators),
        )

        cfg_barriers = _config_from_args(args)
        result = run_barrier_monitor(
            as_of=as_of,
            open_lots=open_lots,
            ohlcv_history=ohlcv,
            latest_prices=latest_prices,
            indicators=indicators,
            previous_state=previous_state,
            config=cfg_barriers,
        )

        payload = result.to_dict()
        payload["sources"] = {
            "open_lot_trace": f"s3://{cfg.bucket}/{open_lot_trace_key}",
            "positions_snapshot": (f"s3://{cfg.bucket}/{positions_key}" if positions_key else None),
            "trade_index": f"s3://{cfg.bucket}/{_trades_index_key(cfg)}",
            "previous_barrier_state": (f"s3://{cfg.bucket}/{previous_state_key}" if previous_state_key else None),
            **market_sources,
        }
        if contract_warnings:
            payload.setdefault("warnings", []).extend(contract_warnings)
        if stale_lots_removed:
            payload.setdefault("warnings", []).append(f"stale_open_lots_filtered:{stale_lots_removed}")
            payload.setdefault("summary", {})["stale_open_lots_filtered"] = int(stale_lots_removed)

        lots_df = pd.DataFrame(payload["lot_results"])
        positions_df = pd.DataFrame(payload["position_results"])
        actions_df = positions_df[positions_df["recommended_action"].isin(["EXIT", "REDUCE", "WATCH", "DATA_ISSUE"])].copy() if not positions_df.empty else pd.DataFrame()

        print("\n=== OPEN POSITION BARRIER MONITOR ===")
        print(f"open_lots:       {result.summary['open_lots']}")
        print(f"open_positions:  {result.summary['open_positions']}")
        print(f"lot_actions:     {result.summary['lot_actions']}")
        print(f"position_actions:{result.summary['position_actions']}")
        print(f"degraded_lots:   {result.summary['degraded_lots']}")
        print(f"invalid_lots:    {result.summary['invalid_lots']}")
        print(f"stop_status:     {result.summary.get('position_stop_status')}")
        print(f"target_status:   {result.summary.get('position_target_status')}")
        print(f"throttle:        {result.summary.get('liquidation_throttle')}")
        if stale_lots_removed:
            print(f"stale_lots_filtered: {stale_lots_removed}")
        if not actions_df.empty:
            cols = [
                "asset_id",
                "ticker",
                "side",
                "recommended_action",
                "recommended_exit_fraction",
                "recommended_exit_quantity",
                "recommended_exit_value",
                "trigger_reason",
                "asset_bucket",
                "current_price",
                "avg_entry_price",
                "current_return_pct",
                "active_stop_price",
                "active_stop_source",
                "risk_threshold_price",
                "execution_trigger_price",
                "stop_buffer_amount",
                "fixed_stop_loss_price",
                "fixed_target_profit_price",
                "chandelier_stop_price",
                "profit_lock_stop_price",
                "latest_low",
                "latest_close",
                "highest_high_since_entry",
                "atr_14",
                "stop_status",
                "target_status",
                "breach_type",
                "confirmation_status",
                "nearest_distance_to_stop_pct",
                "nearest_distance_to_target_pct",
                "barrier_quality",
            ]
            print("\n=== ACTIONS / WATCHLIST ===")
            print(actions_df[[c for c in cols if c in actions_df.columns]].to_string(index=False))

        dt_json_key = runtime_dt_key(cfg, BARRIER_TABLE, as_of, "barriers.json")
        latest_json_key = runtime_engine_key(cfg, BARRIER_TABLE, "latest.json")
        dt_lots_key = runtime_dt_key(cfg, BARRIER_TABLE, as_of, "lot_barriers.parquet")
        latest_lots_key = runtime_engine_key(cfg, BARRIER_TABLE, "lot_barriers_latest.parquet")
        dt_positions_key = runtime_dt_key(cfg, BARRIER_TABLE, as_of, "position_barriers.parquet")
        latest_positions_key = runtime_engine_key(cfg, BARRIER_TABLE, "position_barriers_latest.parquet")
        dt_actions_key = runtime_dt_key(cfg, BARRIER_TABLE, as_of, "triggered_actions.csv")
        dt_state_key = _dated_barrier_state_key(cfg, as_of)
        latest_state_key = _latest_barrier_state_key(cfg)

        output_keys = [
            dt_json_key,
            latest_json_key,
            dt_lots_key,
            latest_lots_key,
            dt_positions_key,
            latest_positions_key,
            dt_actions_key,
            dt_state_key,
            latest_state_key,
        ]

        audit = build_audit_event(
            cfg=cfg,
            run_id=run_id,
            event_type="evaluate",
            entity_type="open_position_barriers",
            entity_id="daily_monitor",
            as_of=as_of,
            source_script="run_open_position_barrier_monitor.py",
            source_mode="daily_monitor",
            status=("success" if should_write else "dry_run"),
            reason="daily open-position barrier monitor",
            input_args=input_args,
            output_keys=output_keys,
            after_payload={"summary": result.summary, "sources": payload["sources"]},
            metadata={
                "asset_ids_n": len(asset_ids),
                "open_lots_n": len(open_lots),
                "position_actions": result.summary.get("position_actions"),
                "contract_warnings_n": len(contract_warnings),
                "previous_state_loaded": bool(previous_state_key),
                "liquidity_aware_stops": cfg_barriers.liquidity_aware_stops,
            },
        )

        if not should_write:
            print("\n[DRY RUN] Would write:")
            for key in output_keys:
                print(f"  s3://{cfg.bucket}/{key}")
            write_audit_event(cfg=cfg, event=audit, dry_run=True)
            return

        _s3_put_json(s3, bucket=cfg.bucket, key=dt_json_key, payload=payload)
        _s3_put_json(s3, bucket=cfg.bucket, key=latest_json_key, payload=payload)
        _s3_put_parquet(s3, bucket=cfg.bucket, key=dt_lots_key, df=lots_df)
        _s3_put_parquet(s3, bucket=cfg.bucket, key=latest_lots_key, df=lots_df)
        _s3_put_parquet(s3, bucket=cfg.bucket, key=dt_positions_key, df=positions_df)
        _s3_put_parquet(s3, bucket=cfg.bucket, key=latest_positions_key, df=positions_df)
        s3.put_object(
            Bucket=cfg.bucket,
            Key=dt_actions_key,
            Body=actions_df.to_csv(index=False).encode("utf-8"),
            ContentType="text/csv",
        )
        _s3_put_json(s3, bucket=cfg.bucket, key=dt_state_key, payload=result.next_state)
        _s3_put_json(s3, bucket=cfg.bucket, key=latest_state_key, payload=result.next_state)
        write_audit_event(cfg=cfg, event=audit, dry_run=False)

        print("\n[OK] Wrote barrier monitor outputs:")
        for key in output_keys:
            print(f"  s3://{cfg.bucket}/{key}")


if __name__ == "__main__":
    main()
