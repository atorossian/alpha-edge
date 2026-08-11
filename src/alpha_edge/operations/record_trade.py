# record_trade.py  (record + edit + migrate index)
from __future__ import annotations

import argparse
import io
import json
import uuid
from dataclasses import asdict
from typing import Any, Dict, Literal, Optional, Tuple

import boto3
import pandas as pd

from alpha_edge.operations.trade_risk_models import (
    MarketRiskModelConfig,
    calculate_indicator_backed_risk_percentages,
    load_indicator_snapshot,
)

from alpha_edge.core.schemas import RuntimeConfig, Trade
from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.operations.operation_lifecycle import delete_record_with_audit
from alpha_edge.core.runtime import (
    load_runtime_config,
    require_prod_confirmation,
    runtime_dt_key,
    runtime_engine_key,
)


TRADES_TABLE = "trades"


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def s3_get_bytes(s3, *, bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def s3_exists(s3, *, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def s3_copy(s3, *, bucket: str, src_key: str, dst_key: str) -> None:
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        ContentType="application/json",
        MetadataDirective="COPY",
    )


def s3_delete(s3, *, bucket: str, key: str) -> None:
    s3.delete_object(Bucket=bucket, Key=key)


def s3_get_json_optional(s3, *, bucket: str, key: str) -> Optional[dict]:
    try:
        raw = s3_get_bytes(s3, bucket=bucket, key=key)
        obj = json.loads(raw.decode("utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# ----------------------------
# Key helpers
# ----------------------------
def trades_index_key(cfg: RuntimeConfig) -> str:
    return runtime_engine_key(cfg, TRADES_TABLE, "index.json")


def trades_audit_prefix(cfg: RuntimeConfig) -> str:
    return runtime_engine_key(cfg, "trades_audit")


def default_universe_key(cfg: RuntimeConfig) -> str:
    return runtime_engine_key(cfg, "universe", "universe.csv")


# ----------------------------
# Validation
# ----------------------------
def _parse_date(s: str) -> str:
    d = pd.Timestamp(s).date()
    return d.strftime("%Y-%m-%d")


def _iso_utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def _validate_side(s: str) -> Literal["BUY", "SELL"]:
    s = str(s).upper().strip()
    if s not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    return s  # type: ignore[return-value]


def _validate_positive(name: str, x: float) -> float:
    x = float(x)
    if not (x > 0.0):
        raise ValueError(f"{name} must be > 0")
    return x


def _normalize_action_tag(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s == "":
        return None
    allowed = {"open", "close", "add", "reduce"}
    if s not in allowed:
        raise ValueError(f"action_tag must be one of {sorted(allowed)} (got {x!r})")
    return s


def _normalize_quantity_unit(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s == "":
        return None

    unit_map = {
        "share": "shares",
        "shares": "shares",
        "contract": "contracts",
        "contracts": "contracts",
        "coin": "coins",
        "coins": "coins",
        "ounce": "ounces",
        "ounces": "ounces",
    }
    return unit_map.get(s, s)

def _normalize_risk_model(x: Optional[str]) -> str:
    model = str(x or "fixed_by_asset_class").strip().lower()

    allowed = {
        "fixed_by_asset_class",
        "atr_based",
        "volatility_based",
        "hybrid",
    }

    if model not in allowed:
        raise ValueError(
            f"risk_model must be one of {sorted(allowed)} "
            f"(got {x!r})"
        )

    return model

def _normalize_indicator_mode(x: Optional[str]) -> str:
    mode = str(x or "auto").strip().lower()

    allowed = {"auto", "latest", "point_in_time"}

    if mode not in allowed:
        raise ValueError(
            f"indicator_mode must be one of {sorted(allowed)} "
            f"(got {x!r})"
        )

    return mode

# ----------------------------
# Trade calculation engine
# ----------------------------
def _round_money(x: float) -> float:
    return round(float(x), 2)


def _round_quantity(x: float) -> float:
    return round(float(x), 8)


def _round_price(x: float) -> float:
    return round(float(x), 8)


def _validate_optional_positive(name: str, x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return _validate_positive(name, float(x))


def _infer_position_effect(
    *,
    side: Literal["BUY", "SELL"],
    action_tag: Optional[str],
) -> Optional[str]:
    if action_tag is None:
        return None

    if action_tag == "open" and side == "BUY":
        return "open_long"
    if action_tag == "open" and side == "SELL":
        return "open_short"

    if action_tag == "add" and side == "BUY":
        return "add_long"
    if action_tag == "add" and side == "SELL":
        return "add_short"

    if action_tag == "reduce" and side == "SELL":
        return "reduce_long"
    if action_tag == "reduce" and side == "BUY":
        return "reduce_short"

    if action_tag == "close" and side == "SELL":
        return "close_long"
    if action_tag == "close" and side == "BUY":
        return "close_short"

    raise ValueError(f"Invalid side/action_tag combination: side={side}, action_tag={action_tag}")


def _infer_asset_class(
    *,
    asset_id: Optional[str],
    ticker: str,
    quantity_unit: Optional[str],
    explicit_asset_class: Optional[str] = None,
) -> str:
    if explicit_asset_class:
        return str(explicit_asset_class).lower().strip()

    aid = "" if asset_id is None else str(asset_id).upper().strip()
    t = str(ticker).upper().strip()
    unit = "" if quantity_unit is None else str(quantity_unit).lower().strip()

    if aid.startswith("CRYPTO:") or unit == "coins":
        return "crypto"

    if unit == "contracts":
        return "futures"

    fiat = {
        "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
        "BRL", "MXN", "SEK", "NOK", "DKK", "CNH", "CNY",
    }

    if "-" in t:
        left, right = t.split("-", 1)
        if left in fiat and right in fiat:
            return "forex"

    if unit == "ounces":
        return "commodity"

    return "stock"


def _default_risk_params(asset_class: str) -> dict:
    defaults = {
        "stock": {
            "stop_loss_pct": 0.08,
            "target_profit_pct": 0.15,
            "max_holding_days": 45,
            "time_barrier_method": "fixed_business_days_by_asset_class",
        },
        "crypto": {
            "stop_loss_pct": 0.12,
            "target_profit_pct": 0.25,
            "max_holding_days": 30,
            "time_barrier_method": "fixed_calendar_days_by_asset_class",
        },
        "forex": {
            "stop_loss_pct": 0.03,
            "target_profit_pct": 0.06,
            "max_holding_days": 20,
            "time_barrier_method": "fixed_business_days_by_asset_class",
        },
        "futures": {
            "stop_loss_pct": 0.05,
            "target_profit_pct": 0.10,
            "max_holding_days": 15,
            "time_barrier_method": "fixed_business_days_by_asset_class",
        },
        "commodity": {
            "stop_loss_pct": 0.06,
            "target_profit_pct": 0.12,
            "max_holding_days": 30,
            "time_barrier_method": "fixed_business_days_by_asset_class",
        },
        "unknown": {
            "stop_loss_pct": 0.08,
            "target_profit_pct": 0.15,
            "max_holding_days": 30,
            "time_barrier_method": "fixed_calendar_days_by_asset_class",
        },
    }

    return dict(defaults.get(asset_class, defaults["unknown"]))


def _calculate_expiry_date(
    *,
    as_of: str,
    max_holding_days: int,
    time_barrier_method: str,
) -> str:
    start = pd.Timestamp(as_of)

    if "business" in time_barrier_method:
        expiry = start + pd.offsets.BDay(int(max_holding_days))
    else:
        expiry = start + pd.DateOffset(days=int(max_holding_days))

    return expiry.date().strftime("%Y-%m-%d")


def _calculate_quantity_price_value(
    *,
    quantity: Optional[float],
    price: Optional[float],
    value: Optional[float],
) -> tuple[float, float, float, list[str]]:
    """
    Require any two of quantity, price, value.
    Calculate the missing one.

    Preferred for open/add:
      value + price -> quantity

    Preferred for close/reduce:
      quantity + price -> value
    """
    warnings: list[str] = []

    qty = _validate_optional_positive("quantity", quantity)
    px = _validate_optional_positive("price", price)
    val = _validate_optional_positive("value", value)

    provided = sum(x is not None for x in [qty, px, val])

    if provided < 2:
        raise ValueError("Provide at least two of --quantity, --price, --value.")

    if qty is None:
        qty = _round_quantity(float(val) / float(px))
        warnings.append("quantity calculated from value / price")

    if px is None:
        px = _round_price(float(val) / float(qty))
        warnings.append("price calculated from value / quantity")

    if val is None:
        val = _round_money(float(qty) * float(px))
        warnings.append("value calculated from quantity * price")

    implied_value = _round_money(float(qty) * float(px))

    if value is not None and abs(implied_value - float(value)) > 0.05:
        warnings.append(
            f"provided value differs from quantity*price: provided={value}, implied={implied_value}"
        )

    return float(qty), float(px), float(val), warnings


def _calculate_pnl_fields(
    *,
    position_effect: Optional[str],
    quantity: float,
    price: float,
    value: float,
    reported_pnl: Optional[float],
    open_value: Optional[float],
    entry_price: Optional[float],
) -> dict:
    """
    For close/reduce:
      long:  PnL = close_value - open_value
      short: PnL = open_value - close_value

    reported_pnl is broker/manual reported input.
    calculated_pnl is our computed value when enough info exists.
    """
    close_like = position_effect in {
        "close_long",
        "close_short",
        "reduce_long",
        "reduce_short",
    }

    reported_norm = None if reported_pnl is None else _round_money(float(reported_pnl))
    open_value_norm = None if open_value is None else _round_money(float(open_value))

    if open_value_norm is None and entry_price is not None:
        open_value_norm = _round_money(float(quantity) * float(entry_price))

    calculated_pnl = None
    pnl_source = "unavailable"

    if close_like and open_value_norm is not None:
        if position_effect in {"close_long", "reduce_long"}:
            calculated_pnl = _round_money(float(value) - open_value_norm)
        elif position_effect in {"close_short", "reduce_short"}:
            calculated_pnl = _round_money(open_value_norm - float(value))

    pnl_diff = None
    if reported_norm is not None and calculated_pnl is not None:
        pnl_diff = _round_money(reported_norm - calculated_pnl)
        pnl_source = "broker_reported"
    elif reported_norm is not None:
        pnl_source = "manual_override"
    elif calculated_pnl is not None:
        pnl_source = "calculated"

    if position_effect in {"open_long", "open_short", "add_long", "add_short"}:
        return {
            "entry_price": _round_price(price),
            "exit_price": None,
            "open_value": _round_money(value),
            "close_value": None,
            "reported_pnl": reported_norm,
            "calculated_pnl": None,
            "pnl_diff": None,
            "pnl_source": None if reported_norm is None else "manual_override",
        }

    if close_like:
        return {
            "entry_price": None if entry_price is None else _round_price(entry_price),
            "exit_price": _round_price(price),
            "open_value": open_value_norm,
            "close_value": _round_money(value),
            "reported_pnl": reported_norm,
            "calculated_pnl": calculated_pnl,
            "pnl_diff": pnl_diff,
            "pnl_source": pnl_source,
        }

    return {
        "entry_price": None,
        "exit_price": None,
        "open_value": None,
        "close_value": None,
        "reported_pnl": reported_norm,
        "calculated_pnl": None,
        "pnl_diff": None,
        "pnl_source": None if reported_norm is None else "manual_override",
    }


def _empty_risk_contract() -> dict:
    return {
        "risk_model": None,
        "stop_loss_price": None,
        "target_profit_price": None,
        "stop_loss_pct": None,
        "target_profit_pct": None,
        "risk_amount": None,
        "target_profit_amount": None,
        "risk_reward_ratio": None,
        "max_holding_days": None,
        "expiry_date": None,
        "time_barrier_method": None,
        "barrier_status": None,
        "barrier_hit_at_utc": None,
        "barrier_hit_price": None,
        "barrier_hit_type": None,
        "risk_calculation_inputs": None,
    }


def _calculate_risk_contract(
    *,
    cfg: RuntimeConfig,
    as_of: str,
    asset_id: Optional[str],
    position_effect: Optional[str],
    asset_class: str,
    price: float,
    value: float,
    risk_model: str,
    stop_loss_pct: Optional[float],
    target_profit_pct: Optional[float],
    max_holding_days: Optional[int],
    disable_risk_contract: bool,
    indicator_mode: str,
    indicators_snapshot_key: Optional[str],
    indicators_root_prefix: Optional[str],
    max_indicator_staleness_days: int,
    atr_stop_multiplier: float,
    atr_target_multiplier: float,
    volatility_stop_multiplier: float,
    volatility_target_multiplier: float,
    reward_multiple: float,
    min_stop_pct: float,
    max_stop_pct: float,
    max_target_pct: float,
) -> dict:
    if disable_risk_contract:
        return _empty_risk_contract()

    if position_effect not in {
        "open_long",
        "open_short",
        "add_long",
        "add_short",
    }:
        return _empty_risk_contract()

    model = _normalize_risk_model(risk_model)
    params = _default_risk_params(asset_class)

    holding_days = (
        int(max_holding_days)
        if max_holding_days is not None
        else int(params["max_holding_days"])
    )

    if holding_days <= 0:
        raise ValueError("max_holding_days must be > 0.")

    risk_calculation_inputs: dict[str, Any] = {
        "risk_model": model,
        "asset_class": asset_class,
    }

    if model == "fixed_by_asset_class":
        sl_pct = (
            float(stop_loss_pct)
            if stop_loss_pct is not None
            else float(params["stop_loss_pct"])
        )

        tp_pct = (
            float(target_profit_pct)
            if target_profit_pct is not None
            else float(params["target_profit_pct"])
        )

        time_method = str(params["time_barrier_method"])

        risk_calculation_inputs.update(
            {
                "source": "fixed_by_asset_class",
                "default_stop_loss_pct": params["stop_loss_pct"],
                "default_target_profit_pct": (
                    params["target_profit_pct"]
                ),
                "stop_loss_overridden": stop_loss_pct is not None,
                "target_profit_overridden": (
                    target_profit_pct is not None
                ),
            }
        )

    else:
        if not asset_id:
            raise ValueError(
                f"asset_id is required for risk_model={model!r}."
            )

        indicator_snapshot = load_indicator_snapshot(
            cfg=cfg,
            asset_id=str(asset_id),
            as_of=as_of,
            mode=_normalize_indicator_mode(indicator_mode),
            latest_snapshot_key=indicators_snapshot_key,
            historical_root_prefix=indicators_root_prefix,
            max_staleness_days=int(
                max_indicator_staleness_days
            ),
        )

        model_config = MarketRiskModelConfig(
            atr_stop_multiplier=float(atr_stop_multiplier),
            atr_target_multiplier=float(atr_target_multiplier),
            volatility_stop_multiplier=float(
                volatility_stop_multiplier
            ),
            volatility_target_multiplier=float(
                volatility_target_multiplier
            ),
            reward_multiple=float(reward_multiple),
            min_stop_pct=float(min_stop_pct),
            max_stop_pct=float(max_stop_pct),
            max_target_pct=float(max_target_pct),
            max_indicator_staleness_days=int(
                max_indicator_staleness_days
            ),
        )

        calculated = calculate_indicator_backed_risk_percentages(
            risk_model=model,  # type: ignore[arg-type]
            indicators=indicator_snapshot,
            max_holding_days=holding_days,
            config=model_config,
        )

        # Explicit SL/TP values remain available as deliberate overrides.
        sl_pct = (
            float(stop_loss_pct)
            if stop_loss_pct is not None
            else float(calculated["stop_loss_pct"])
        )

        tp_pct = (
            float(target_profit_pct)
            if target_profit_pct is not None
            else float(calculated["target_profit_pct"])
        )

        time_method = "market_indicator_calendar_days"

        risk_calculation_inputs.update(
            {
                "source": "market_indicators",
                "indicator_snapshot": (
                    indicator_snapshot.as_dict()
                ),
                "market_model_inputs": calculated[
                    "model_inputs"
                ],
                "calculated_stop_loss_pct": calculated[
                    "stop_loss_pct"
                ],
                "calculated_target_profit_pct": calculated[
                    "target_profit_pct"
                ],
                "stop_loss_overridden": stop_loss_pct is not None,
                "target_profit_overridden": (
                    target_profit_pct is not None
                ),
            }
        )

    if not (sl_pct > 0):
        raise ValueError("stop_loss_pct must be > 0.")

    if not (tp_pct > 0):
        raise ValueError("target_profit_pct must be > 0.")

    if sl_pct >= 1:
        raise ValueError(
            "stop_loss_pct must be below 1.0."
        )

    if tp_pct >= 1:
        raise ValueError(
            "target_profit_pct must be below 1.0 so short-trade "
            "targets remain positive."
        )

    is_long = position_effect in {"open_long", "add_long"}

    if is_long:
        stop_loss_price = _round_price(
            float(price) * (1.0 - sl_pct)
        )
        target_profit_price = _round_price(
            float(price) * (1.0 + tp_pct)
        )
    else:
        stop_loss_price = _round_price(
            float(price) * (1.0 + sl_pct)
        )
        target_profit_price = _round_price(
            float(price) * (1.0 - tp_pct)
        )

    risk_amount = _round_money(float(value) * sl_pct)
    target_profit_amount = _round_money(float(value) * tp_pct)

    return {
        "risk_model": model,
        "stop_loss_price": stop_loss_price,
        "target_profit_price": target_profit_price,
        "stop_loss_pct": float(sl_pct),
        "target_profit_pct": float(tp_pct),
        "risk_amount": risk_amount,
        "target_profit_amount": target_profit_amount,
        "risk_reward_ratio": _round_price(
            float(tp_pct) / float(sl_pct)
        ),
        "max_holding_days": holding_days,
        "expiry_date": _calculate_expiry_date(
            as_of=as_of,
            max_holding_days=holding_days,
            time_barrier_method=time_method,
        ),
        "time_barrier_method": time_method,
        "barrier_status": "active",
        "barrier_hit_at_utc": None,
        "barrier_hit_price": None,
        "barrier_hit_type": None,
        "risk_calculation_inputs": (
            risk_calculation_inputs
        ),
    }

def _calculate_trade_fields(
    *,
    as_of: str,
    ticker: str,
    side: Literal["BUY", "SELL"],
    action_tag: Optional[str],
    quantity_unit: Optional[str],
    quantity: Optional[float],
    price: Optional[float],
    value: Optional[float],
    reported_pnl: Optional[float],
    open_value: Optional[float],
    entry_price: Optional[float],
    asset_id: Optional[str],
    asset_class: Optional[str],
    risk_model: str,
    stop_loss_pct: Optional[float],
    target_profit_pct: Optional[float],
    max_holding_days: Optional[int],
    disable_risk_contract: bool,
    cfg: RuntimeConfig,
    indicator_mode: str,
    indicators_snapshot_key: Optional[str],
    indicators_root_prefix: Optional[str],
    max_indicator_staleness_days: int,
    atr_stop_multiplier: float,
    atr_target_multiplier: float,
    volatility_stop_multiplier: float,
    volatility_target_multiplier: float,
    reward_multiple: float,
    min_stop_pct: float,
    max_stop_pct: float,
    max_target_pct: float,
) -> dict:
    warnings: list[str] = []

    if action_tag is None:
        raise ValueError("action_tag is required for trade calculation engine.")

    position_effect = _infer_position_effect(side=side, action_tag=action_tag)

    qty, px, val, calc_warnings = _calculate_quantity_price_value(
        quantity=quantity,
        price=price,
        value=value,
    )
    warnings.extend(calc_warnings)

    asset_class_norm = _infer_asset_class(
        asset_id=asset_id,
        ticker=ticker,
        quantity_unit=quantity_unit,
        explicit_asset_class=asset_class,
    )

    pnl_fields = _calculate_pnl_fields(
        position_effect=position_effect,
        quantity=qty,
        price=px,
        value=val,
        reported_pnl=reported_pnl,
        open_value=open_value,
        entry_price=entry_price,
    )

    if position_effect in {"close_long", "close_short", "reduce_long", "reduce_short"}:
        if pnl_fields["reported_pnl"] is None and pnl_fields["calculated_pnl"] is None:
            warnings.append(
                "close/reduce trade has no reported_pnl and insufficient open_value/entry_price "
                "to calculate pnl"
            )

    risk_fields = _calculate_risk_contract(
        cfg=cfg,
        as_of=as_of,
        asset_id=asset_id,
        position_effect=position_effect,
        asset_class=asset_class_norm,
        price=px,
        value=val,
        risk_model=risk_model,
        stop_loss_pct=stop_loss_pct,
        target_profit_pct=target_profit_pct,
        max_holding_days=max_holding_days,
        disable_risk_contract=disable_risk_contract,
        indicator_mode=indicator_mode,
        indicators_snapshot_key=indicators_snapshot_key,
        indicators_root_prefix=indicators_root_prefix,
        max_indicator_staleness_days=(
            max_indicator_staleness_days
        ),
        atr_stop_multiplier=atr_stop_multiplier,
        atr_target_multiplier=atr_target_multiplier,
        volatility_stop_multiplier=(
            volatility_stop_multiplier
        ),
        volatility_target_multiplier=(
            volatility_target_multiplier
        ),
        reward_multiple=reward_multiple,
        min_stop_pct=min_stop_pct,
        max_stop_pct=max_stop_pct,
        max_target_pct=max_target_pct,
    )

    risk_calculation_inputs = risk_fields.pop(
        "risk_calculation_inputs",
        None,
    )

    return {
        "quantity": qty,
        "price": px,
        "value": val,
        "asset_class": asset_class_norm,
        "position_effect": position_effect,
        **pnl_fields,
        **risk_fields,
        "calculation_method": "trade_calculation_engine_v2",
        "calculation_inputs": {
            "quantity": quantity,
            "price": price,
            "value": value,
            "reported_pnl": reported_pnl,
            "open_value": open_value,
            "entry_price": entry_price,
            "asset_class": asset_class,
            "risk_model": risk_model,
            "stop_loss_pct": stop_loss_pct,
            "target_profit_pct": target_profit_pct,
            "max_holding_days": max_holding_days,
            "disable_risk_contract": disable_risk_contract,
            "indicator_mode": indicator_mode,
            "indicators_snapshot_key": indicators_snapshot_key,
            "indicators_root_prefix": indicators_root_prefix,
            "max_indicator_staleness_days": (
                max_indicator_staleness_days
            ),
            "atr_stop_multiplier": atr_stop_multiplier,
            "atr_target_multiplier": atr_target_multiplier,
            "volatility_stop_multiplier": (
                volatility_stop_multiplier
            ),
            "volatility_target_multiplier": (
                volatility_target_multiplier
            ),
            "reward_multiple": reward_multiple,
            "min_stop_pct": min_stop_pct,
            "max_stop_pct": max_stop_pct,
            "max_target_pct": max_target_pct,
            "risk_calculation": risk_calculation_inputs,
        },
        "calculation_warnings": warnings or None,
    }

# ----------------------------
# Universe lookup
# ----------------------------
def _load_universe_df(
    *,
    cfg: RuntimeConfig,
    universe_path: Optional[str],
    universe_key: Optional[str],
) -> Tuple[pd.DataFrame, str]:
    if universe_path:
        df = pd.read_csv(universe_path)
        return df, f"file://{universe_path}"

    if universe_key:
        s3 = s3_client(cfg)
        raw = s3_get_bytes(s3, bucket=cfg.bucket, key=universe_key)
        df = pd.read_csv(io.BytesIO(raw))
        return df, f"s3://{cfg.bucket}/{universe_key}"

    raise RuntimeError("Universe not provided. Use --universe-path or --universe-key to resolve asset_id.")


def _resolve_asset_id_from_universe(
    *,
    cfg: RuntimeConfig,
    broker_ticker: str,
    universe_path: Optional[str],
    universe_key: Optional[str],
) -> Tuple[str, str]:
    df, ref = _load_universe_df(
        cfg=cfg,
        universe_path=universe_path,
        universe_key=universe_key,
    )

    required = {"asset_id", "broker_ticker"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Universe CSV missing required columns {missing} (ref={ref})")

    bt = str(broker_ticker).upper().strip()

    df = df.copy()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["broker_ticker"] = df["broker_ticker"].astype(str).str.upper().str.strip()

    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    m = df[df["broker_ticker"] == bt]
    if len(m) == 1:
        aid = str(m.iloc[0]["asset_id"]).strip()
        if not aid:
            raise RuntimeError(f"Universe match found but asset_id empty (broker_ticker={bt}, ref={ref})")
        return aid, ref

    if len(m) == 0 and "ticker" in df.columns:
        m2 = df[df["ticker"] == bt]
        if len(m2) == 1:
            aid = str(m2.iloc[0]["asset_id"]).strip()
            if not aid:
                raise RuntimeError(f"Universe match found but asset_id empty (ticker={bt}, ref={ref})")
            return aid, ref

        if len(m2) > 1:
            sample_cols = [c for c in ["asset_id", "broker_ticker", "ticker", "yahoo_ticker", "name"] if c in df.columns]
            sample = m2[sample_cols].head(10).to_dict("records")
            raise RuntimeError(
                f"Ambiguous universe mapping for ticker={bt}: {len(m2)} rows match (ref={ref}). "
                f"Sample={sample}"
            )

    if len(m) > 1:
        sample_cols = [c for c in ["asset_id", "broker_ticker", "ticker", "yahoo_ticker", "name"] if c in df.columns]
        sample = m[sample_cols].head(10).to_dict("records")
        raise RuntimeError(
            f"Ambiguous universe mapping for broker_ticker={bt}: {len(m)} rows match (ref={ref}). "
            f"Sample={sample}"
        )

    raise RuntimeError(f"No universe mapping for broker_ticker/ticker={bt} (ref={ref}).")


# ----------------------------
# Trades index
# ----------------------------
def _load_trades_index(s3, *, cfg: RuntimeConfig) -> dict:
    idx = s3_get_json_optional(s3, bucket=cfg.bucket, key=trades_index_key(cfg))
    return idx if isinstance(idx, dict) else {}


def _save_trades_index(s3, *, cfg: RuntimeConfig, idx: dict) -> None:
    s3_put_json(s3, bucket=cfg.bucket, key=trades_index_key(cfg), payload=idx)


def _index_set_trade(
    s3,
    *,
    cfg: RuntimeConfig,
    trade_id: str,
    key: str,
    as_of: str,
) -> None:
    idx = _load_trades_index(s3, cfg=cfg)
    idx[str(trade_id)] = {"key": str(key), "as_of": str(as_of)}
    _save_trades_index(s3, cfg=cfg, idx=idx)


def _audit_backup_key(cfg: RuntimeConfig, src_key: str) -> str:
    safe = src_key.replace("/", "__")
    return f"{trades_audit_prefix(cfg)}/{safe}"


def _rebuild_trades_index(
    s3,
    *,
    cfg: RuntimeConfig,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Rebuild trades/index.json from the S3 trade JSON files.

    Important:
    - Trade IDs are allowed to be arbitrary strings, including broker-prefixed IDs
      such as "broker-20260528-143738-LLY-SELL-close-53482056".
    - The index must use the trade_id stored inside the JSON payload when present.
      The filename remains only a fallback.
    - This makes migration resilient to historical/broker IDs and filename/id
      mismatches created during repair work.
    """
    prefix = runtime_engine_key(cfg, TRADES_TABLE) + "/"

    keys: list[str] = []
    token = None

    while True:
        kwargs = {"Bucket": cfg.bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)

        for it in (resp.get("Contents") or []):
            k = str(it.get("Key", "") or "")
            if not k:
                continue

            name = k.split("/")[-1]

            if not name.startswith("trade_"):
                continue
            if not name.endswith(".json"):
                continue
            if name in {"index.json", "latest.json"}:
                continue

            keys.append(k)

        if not resp.get("IsTruncated"):
            break

        token = resp.get("NextContinuationToken")

    idx: Dict[str, Any] = {}
    scanned = 0
    indexed = 0
    skipped: list[dict] = []
    duplicate_trade_ids: list[dict] = []
    filename_id_mismatches: list[dict] = []

    for k in sorted(keys):
        scanned += 1

        name = k.split("/")[-1]
        filename_trade_id = name[len("trade_") : -len(".json")]

        dt_from_key = ""
        for part in k.split("/"):
            if part.startswith("dt="):
                dt_from_key = part[len("dt=") :]
                break

        obj = s3_get_json_optional(s3, bucket=cfg.bucket, key=k)

        if not isinstance(obj, dict):
            skipped.append(
                {
                    "key": k,
                    "reason": "invalid_json_or_not_object",
                    "filename_trade_id": filename_trade_id,
                    "as_of_from_key": dt_from_key,
                }
            )
            continue

        payload_trade_id = obj.get("trade_id")
        trade_id = str(payload_trade_id).strip() if payload_trade_id not in (None, "") else filename_trade_id

        payload_as_of = obj.get("as_of")
        as_of = str(payload_as_of).strip() if payload_as_of not in (None, "") else dt_from_key

        if not trade_id:
            skipped.append(
                {
                    "key": k,
                    "reason": "missing_trade_id",
                    "filename_trade_id": filename_trade_id,
                    "as_of_from_key": dt_from_key,
                }
            )
            continue

        if filename_trade_id != trade_id:
            filename_id_mismatches.append(
                {
                    "key": k,
                    "filename_trade_id": filename_trade_id,
                    "payload_trade_id": trade_id,
                    "as_of": as_of,
                }
            )

        if trade_id in idx:
            duplicate_trade_ids.append(
                {
                    "trade_id": trade_id,
                    "previous_key": idx[trade_id].get("key"),
                    "new_key": k,
                    "previous_as_of": idx[trade_id].get("as_of"),
                    "new_as_of": as_of,
                }
            )

        idx[str(trade_id)] = {
            "key": str(k),
            "as_of": str(as_of),
            "filename_trade_id": str(filename_trade_id),
        }
        indexed += 1

    meta: Dict[str, Any] = {
        "objects_seen": len(keys),
        "objects_scanned": scanned,
        "objects_indexed": indexed,
        "objects_skipped": len(skipped),
        "duplicate_trade_ids": duplicate_trade_ids[:100],
        "filename_id_mismatches": filename_id_mismatches[:100],
        "skipped": skipped[:100],
        "allows_arbitrary_trade_ids": True,
    }

    if skipped:
        print(f"[WARN] skipped trade json objects: {len(skipped)}")
        for row in skipped[:20]:
            print(f"  - {row}")

    if duplicate_trade_ids:
        print(f"[WARN] duplicate trade_id values while rebuilding index: {len(duplicate_trade_ids)}")
        for row in duplicate_trade_ids[:20]:
            print(f"  - {row}")

    if filename_id_mismatches:
        print(f"[WARN] filename/payload trade_id mismatches: {len(filename_id_mismatches)}")
        for row in filename_id_mismatches[:20]:
            print(f"  - {row}")

    return scanned, indexed, idx


def migrate_trades_index(
    *,
    cfg: RuntimeConfig,
    dry_run: bool = False,
    run_id: Optional[str] = None,
    input_args: Optional[Dict[str, Any]] = None,
) -> None:
    s3 = s3_client(cfg)
    scanned, indexed, idx = _rebuild_trades_index(s3, cfg=cfg)

    print("\n=== MIGRATE TRADES INDEX ===")
    print(f"env:      {cfg.env}")
    print(f"bucket:   {cfg.bucket}")
    print(f"root:     {cfg.engine_root}")
    print(f"scanned:  {scanned}")
    print(f"indexed:  {indexed}")
    print("ids:      arbitrary string trade_ids allowed, including broker-* ids")
    print(f"index:    s3://{cfg.bucket}/{trades_index_key(cfg)}")

    audit = build_audit_event(
        cfg=cfg,
        run_id=run_id,
        event_type="migrate",
        entity_type="trades_index",
        entity_id="index.json",
        as_of=None,
        source_script="record_trade.py",
        source_mode="migrate",
        status=("dry_run" if dry_run else "success"),
        input_args=input_args,
        output_keys=[trades_index_key(cfg)],
        metadata={"objects_scanned": scanned, "objects_indexed": indexed},
    )

    if dry_run:
        print("[DRY RUN] no write performed.")
        write_audit_event(cfg=cfg, event=audit, dry_run=True)
        print("")
        return

    _save_trades_index(s3, cfg=cfg, idx=idx)
    write_audit_event(cfg=cfg, event=audit, dry_run=False)

    print("[OK] index.json written/overwritten.")
    print("")


# ----------------------------
# Core: record
# ----------------------------
def record_trade(
    *,
    cfg: RuntimeConfig,
    as_of: str,
    ticker: str,
    side: str,
    quantity: Optional[float],
    price: Optional[float],
    currency: str = "USD",
    trade_id: Optional[str] = None,
    ts_utc: Optional[str] = None,
    asset_id: Optional[str] = None,
    universe_path: Optional[str] = None,
    universe_key: Optional[str] = None,
    strict_universe: bool = False,
    action_tag: Optional[str] = None,
    quantity_unit: Optional[str] = None,
    value: Optional[float] = None,
    reported_pnl: Optional[float] = None,
    open_value: Optional[float] = None,
    entry_price: Optional[float] = None,
    asset_class: Optional[str] = None,
    risk_model: str = "fixed_by_asset_class",
    stop_loss_pct: Optional[float] = None,
    target_profit_pct: Optional[float] = None,
    max_holding_days: Optional[int] = None,
    disable_risk_contract: bool = False,
    choice_id: Optional[str] = None,
    portfolio_run_id: Optional[str] = None,
    note: Optional[str] = None,
    dry_run: bool = False,
    run_id: Optional[str] = None,
    source_script: str = "record_trade.py",
    source_mode: str = "record",
    input_args: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
    indicator_mode: str = "auto",
    indicators_snapshot_key: Optional[str] = None,
    indicators_root_prefix: Optional[str] = None,
    max_indicator_staleness_days: int = 10,

    atr_stop_multiplier: float = 2.0,
    atr_target_multiplier: float = 4.0,
    volatility_stop_multiplier: float = 1.0,
    volatility_target_multiplier: float = 1.8,
    reward_multiple: float = 2.0,
    min_stop_pct: float = 0.02,
    max_stop_pct: float = 0.25,
    max_target_pct: float = 0.95,
) -> None:
    s3 = s3_client(cfg)

    as_of_norm = _parse_date(as_of)
    side_norm = _validate_side(side)
    ticker_norm = str(ticker).upper().strip()
    currency_norm = str(currency).upper().strip() or "USD"

    action_tag_norm = _normalize_action_tag(action_tag)
    unit_norm = _normalize_quantity_unit(quantity_unit)

    if action_tag_norm is None:
        raise ValueError("action_tag is required for all trades.")

    universe_ref = None
    asset_id_norm = None if asset_id is None else str(asset_id).strip()

    if not asset_id_norm:
        if universe_path or universe_key:
            asset_id_norm, universe_ref = _resolve_asset_id_from_universe(
                cfg=cfg,
                broker_ticker=ticker_norm,
                universe_path=universe_path,
                universe_key=universe_key,
            )
        elif strict_universe:
            raise ValueError("asset_id not provided and --strict-universe requested but no universe provided.")
        else:
            asset_id_norm = None

    if strict_universe and not asset_id_norm:
        raise ValueError(
            "asset_id could not be resolved (strict mode). "
            "Provide --asset-id or --universe-path/--universe-key."
        )

    risk_model_norm = _normalize_risk_model(risk_model)
    indicator_mode_norm = _normalize_indicator_mode(
        indicator_mode
    )

    calc = _calculate_trade_fields(
        cfg=cfg,
        as_of=as_of_norm,
        ticker=ticker_norm,
        side=side_norm,
        action_tag=action_tag_norm,
        quantity_unit=unit_norm,
        quantity=quantity,
        price=price,
        value=value,
        reported_pnl=reported_pnl,
        open_value=open_value,
        entry_price=entry_price,
        asset_id=asset_id_norm,
        asset_class=asset_class,
        risk_model=risk_model_norm,
        stop_loss_pct=stop_loss_pct,
        target_profit_pct=target_profit_pct,
        max_holding_days=max_holding_days,
        disable_risk_contract=disable_risk_contract,
        indicator_mode=indicator_mode_norm,
        indicators_snapshot_key=indicators_snapshot_key,
        indicators_root_prefix=indicators_root_prefix,
        max_indicator_staleness_days=max_indicator_staleness_days,
        atr_stop_multiplier=atr_stop_multiplier,
        atr_target_multiplier=atr_target_multiplier,
        volatility_stop_multiplier=volatility_stop_multiplier,
        volatility_target_multiplier=volatility_target_multiplier,
        reward_multiple=reward_multiple,
        min_stop_pct=min_stop_pct,
        max_stop_pct=max_stop_pct,
        max_target_pct=max_target_pct,
    )

    qty = calc["quantity"]
    px = calc["price"]
    value_norm = calc["value"]
    reported_pnl_norm = calc["reported_pnl"]

    if ts_utc is None:
        ts_utc = _iso_utc_now()

    if trade_id is None:
        trade_id = f"{as_of_norm.replace('-', '')}-{uuid.uuid4().hex[:10]}"

    trade = Trade(
        trade_id=str(trade_id),
        as_of=as_of_norm,
        ts_utc=str(ts_utc),
        asset_id=asset_id_norm,
        ticker=ticker_norm,
        side=side_norm,
        quantity=float(qty),
        price=float(px),
        currency=currency_norm,
        choice_id=(str(choice_id) if choice_id else None),
        portfolio_run_id=(str(portfolio_run_id) if portfolio_run_id else None),
        note=(str(note) if note else None),
        action_tag=action_tag_norm,
        quantity_unit=unit_norm,
        value=value_norm,
        reported_pnl=reported_pnl_norm,

        asset_class=calc["asset_class"],
        position_effect=calc["position_effect"],

        entry_price=calc["entry_price"],
        exit_price=calc["exit_price"],
        open_value=calc["open_value"],
        close_value=calc["close_value"],

        calculated_pnl=calc["calculated_pnl"],
        pnl_diff=calc["pnl_diff"],
        pnl_source=calc["pnl_source"],

        risk_model=calc["risk_model"],
        stop_loss_price=calc["stop_loss_price"],
        target_profit_price=calc["target_profit_price"],
        stop_loss_pct=calc["stop_loss_pct"],
        target_profit_pct=calc["target_profit_pct"],
        risk_amount=calc["risk_amount"],
        target_profit_amount=calc["target_profit_amount"],
        risk_reward_ratio=calc["risk_reward_ratio"],

        max_holding_days=calc["max_holding_days"],
        expiry_date=calc["expiry_date"],
        time_barrier_method=calc["time_barrier_method"],

        barrier_status=calc["barrier_status"],
        barrier_hit_at_utc=calc["barrier_hit_at_utc"],
        barrier_hit_price=calc["barrier_hit_price"],
        barrier_hit_type=calc["barrier_hit_type"],

        calculation_method=calc["calculation_method"],
        calculation_inputs=calc["calculation_inputs"],
        calculation_warnings=calc["calculation_warnings"],
    )

    payload = asdict(trade)

    trade_key = runtime_dt_key(cfg, TRADES_TABLE, as_of_norm, f"trade_{trade.trade_id}.json")
    latest_key = runtime_engine_key(cfg, TRADES_TABLE, "latest.json")

    print("\n=== RECORD TRADE ===")
    print(f"env:       {cfg.env}")
    print(f"bucket:    {cfg.bucket}")
    print(f"root:      {cfg.engine_root}")
    print(f"as_of:     {trade.as_of}")
    print(f"trade_id:  {trade.trade_id}")
    print(f"ts_utc:    {trade.ts_utc}")
    print(f"asset_id:  {trade.asset_id}")
    print(f"{trade.side} {trade.ticker} qty={trade.quantity} px={trade.price} {trade.currency}")

    if universe_ref:
        print(f"universe:  {universe_ref}")
    if trade.choice_id:
        print(f"choice_id: {trade.choice_id}")
    if trade.portfolio_run_id:
        print(f"run_id:    {trade.portfolio_run_id}")
    if trade.note:
        print(f"note:      {trade.note}")
    if trade.action_tag:
        print(f"action_tag:{trade.action_tag}")
    if trade.quantity_unit:
        print(f"unit:      {trade.quantity_unit}")
    if trade.value is not None:
        print(f"value:     {trade.value}")
    if trade.reported_pnl is not None:
        print(f"reported:  {trade.reported_pnl}")
    if trade.asset_class:
        print(f"asset_cls: {trade.asset_class}")
    if trade.position_effect:
        print(f"effect:    {trade.position_effect}")
    if trade.open_value is not None:
        print(f"open_val:  {trade.open_value}")
    if trade.close_value is not None:
        print(f"close_val: {trade.close_value}")
    if trade.calculated_pnl is not None:
        print(f"calc_pnl:  {trade.calculated_pnl}")
    if trade.pnl_diff is not None:
        print(f"pnl_diff:  {trade.pnl_diff}")
    if trade.stop_loss_price is not None:
        print(f"SL:        {trade.stop_loss_price}")
    if trade.target_profit_price is not None:
        print(f"TP:        {trade.target_profit_price}")
    if trade.expiry_date:
        print(f"expires:   {trade.expiry_date}")
    if trade.calculation_warnings:
        print("warnings:")
        for w in trade.calculation_warnings:
            print(f"  - {w}")
    if trade.risk_model:
        print(f"risk_model:{trade.risk_model}")
    if trade.stop_loss_pct is not None:
        print(f"SL pct:    {trade.stop_loss_pct:.4%}")
    if trade.target_profit_pct is not None:
        print(f"TP pct:    {trade.target_profit_pct:.4%}")
    print("")

    output_keys = [trade_key, latest_key, trades_index_key(cfg)]
    audit = build_audit_event(
        cfg=cfg,
        run_id=run_id,
        event_type="create",
        entity_type="trade",
        entity_id=str(trade.trade_id),
        as_of=as_of_norm,
        source_script=source_script,
        source_mode=source_mode,
        status=("dry_run" if dry_run else "success"),
        reason=reason,
        input_args=input_args,
        output_keys=output_keys,
        after_payload=payload,
        metadata={
            "ticker": trade.ticker,
            "side": trade.side,
            "action_tag": trade.action_tag,
            "position_effect": trade.position_effect,
            "asset_class": trade.asset_class,
            "risk_model": trade.risk_model,
            "calculation_method": trade.calculation_method,
            "calculation_warnings": trade.calculation_warnings,
        },
    )

    if dry_run:
        print("[DRY RUN] Would write:")
        print(f"  s3://{cfg.bucket}/{trade_key}")
        print(f"  s3://{cfg.bucket}/{latest_key}")
        print(f"  s3://{cfg.bucket}/{trades_index_key(cfg)} (update trade_id mapping)")
        write_audit_event(cfg=cfg, event=audit, dry_run=True)
        return

    s3_put_json(s3, bucket=cfg.bucket, key=trade_key, payload=payload)
    s3_put_json(s3, bucket=cfg.bucket, key=latest_key, payload=payload)

    _index_set_trade(
        s3,
        cfg=cfg,
        trade_id=str(trade.trade_id),
        key=trade_key,
        as_of=as_of_norm,
    )
    write_audit_event(cfg=cfg, event=audit, dry_run=False)

    print("[OK] Wrote trade:")
    print(f"  s3://{cfg.bucket}/{trade_key}")
    print("[OK] Updated latest:")
    print(f"  s3://{cfg.bucket}/{latest_key}")
    print("[OK] Updated index:")
    print(f"  s3://{cfg.bucket}/{trades_index_key(cfg)}")
    print("")


# ----------------------------
# Core: edit
# ----------------------------
def edit_trade(
    *,
    cfg: RuntimeConfig,
    trade_id: str,
    old_as_of: Optional[str],
    patch: dict,
    new_as_of: Optional[str] = None,
    dry_run: bool = False,
    write_backup: bool = True,
    run_id: Optional[str] = None,
    input_args: Optional[Dict[str, Any]] = None,
    reason: Optional[str] = None,
) -> None:
    s3 = s3_client(cfg)

    old_key = None
    old_dt = None

    if old_as_of:
        old_dt = _parse_date(old_as_of)
        old_key = runtime_dt_key(cfg, TRADES_TABLE, old_dt, f"trade_{trade_id}.json")
    else:
        idx = _load_trades_index(s3, cfg=cfg)
        meta = idx.get(str(trade_id))
        if isinstance(meta, dict) and meta.get("key"):
            old_key = str(meta["key"])
            old_dt = str(meta.get("as_of") or "") or None

    if not old_key:
        raise ValueError(
            "Cannot resolve trade location. Provide --old-as-of once, or run --mode migrate to build index."
        )

    if not s3_exists(s3, bucket=cfg.bucket, key=old_key):
        raise RuntimeError(f"Trade not found: s3://{cfg.bucket}/{old_key} (index may be stale; run --mode migrate)")

    obj = json.loads(s3_get_bytes(s3, bucket=cfg.bucket, key=old_key).decode("utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("Trade JSON is not an object")

    before_payload = dict(obj)

    for k, v in patch.items():
        obj[k] = v

    if new_as_of:
        obj["as_of"] = _parse_date(new_as_of)

    if "action_tag" in obj:
        at = _normalize_action_tag(obj.get("action_tag"))
        if at is None:
            raise ValueError("action_tag cannot be null/empty after edit.")
        obj["action_tag"] = at

    if "side" in obj:
        obj["side"] = _validate_side(obj.get("side"))

    if "quantity" in obj:
        obj["quantity"] = _validate_positive("quantity", obj.get("quantity"))

    if "price" in obj:
        obj["price"] = _validate_positive("price", obj.get("price"))

    if "quantity_unit" in obj and obj.get("quantity_unit") is not None:
        obj["quantity_unit"] = _normalize_quantity_unit(obj.get("quantity_unit"))

    if new_as_of:
        dst_dt = _parse_date(new_as_of)
    else:
        dst_dt = old_dt or _parse_date(str(obj.get("as_of") or ""))

    if not dst_dt:
        raise ValueError("Could not resolve destination dt.")

    dst_key = runtime_dt_key(cfg, TRADES_TABLE, dst_dt, f"trade_{trade_id}.json")

    print("\n=== EDIT TRADE ===")
    print(f"env:       {cfg.env}")
    print(f"bucket:    {cfg.bucket}")
    print(f"root:      {cfg.engine_root}")
    print(f"trade_id:  {trade_id}")
    print(f"from:      s3://{cfg.bucket}/{old_key}")
    print(f"to:        s3://{cfg.bucket}/{dst_key}")
    print(f"patch:     {sorted(list(patch.keys()))}")

    if write_backup:
        print(f"backup:    s3://{cfg.bucket}/{_audit_backup_key(cfg, old_key)}")

    print("")

    backup_key = _audit_backup_key(cfg, old_key) if write_backup else None
    latest_key = runtime_engine_key(cfg, TRADES_TABLE, "latest.json")
    output_keys = [dst_key, latest_key, trades_index_key(cfg)]
    deleted_keys = [old_key] if dst_key != old_key else []
    backup_keys = [backup_key] if backup_key else []

    audit = build_audit_event(
        cfg=cfg,
        run_id=run_id,
        event_type="modify",
        entity_type="trade",
        entity_id=str(trade_id),
        as_of=dst_dt,
        source_script="record_trade.py",
        source_mode="edit",
        status=("dry_run" if dry_run else "success"),
        reason=reason,
        input_args=input_args,
        output_keys=output_keys,
        backup_keys=backup_keys,
        deleted_keys=deleted_keys,
        before_payload=before_payload,
        after_payload=obj,
        metadata={"patch_keys": sorted(list(patch.keys())), "old_key": old_key, "new_key": dst_key},
    )

    if dry_run:
        print("[DRY RUN] no writes performed.")
        write_audit_event(cfg=cfg, event=audit, dry_run=True)
        return

    if backup_key:
        s3_copy(s3, bucket=cfg.bucket, src_key=old_key, dst_key=backup_key)

    if dst_key != old_key:
        s3_copy(s3, bucket=cfg.bucket, src_key=old_key, dst_key=dst_key)
        s3_delete(s3, bucket=cfg.bucket, key=old_key)

    s3_put_json(s3, bucket=cfg.bucket, key=dst_key, payload=obj)
    s3_put_json(s3, bucket=cfg.bucket, key=latest_key, payload=obj)

    _index_set_trade(
        s3,
        cfg=cfg,
        trade_id=trade_id,
        key=dst_key,
        as_of=dst_dt,
    )
    write_audit_event(cfg=cfg, event=audit, dry_run=False)

    print("[OK] Updated:")
    print(f"  s3://{cfg.bucket}/{dst_key}")
    print("[OK] Updated latest:")
    print(f"  s3://{cfg.bucket}/{latest_key}")
    print("[OK] Updated index:")
    print(f"  s3://{cfg.bucket}/{trades_index_key(cfg)}")
    print("")


# ----------------------------
# Core: delete
# ----------------------------
def delete_trade(
    *,
    cfg: RuntimeConfig,
    trade_id: str,
    old_as_of: Optional[str] = None,
    reason: Optional[str] = None,
    dry_run: bool = False,
    run_id: Optional[str] = None,
    input_args: Optional[Dict[str, Any]] = None,
) -> None:
    delete_record_with_audit(
        cfg=cfg,
        table=TRADES_TABLE,
        entity_type="trade",
        entity_id=str(trade_id),
        id_field="trade_id",
        file_prefix="trade",
        as_of=old_as_of,
        index_key=trades_index_key(cfg),
        source_script="record_trade.py",
        source_mode="delete",
        reason=reason,
        dry_run=dry_run,
        run_id=run_id,
        input_args=input_args,
    )


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Record/edit trades in S3.")

    ap.add_argument("--mode", choices=["record", "edit", "delete", "migrate"], default="record")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument("--as-of", required=False, help="Trade date YYYY-MM-DD.")
    ap.add_argument("--ticker", required=False)
    ap.add_argument("--side", required=False, choices=["BUY", "SELL", "buy", "sell"])
    ap.add_argument("--quantity", required=False, type=float)
    ap.add_argument("--price", required=False, type=float)
    ap.add_argument("--currency", default="USD")

    ap.add_argument("--trade-id", default=None)
    ap.add_argument("--ts-utc", default=None)

    ap.add_argument("--asset-id", default=None)

    ap.add_argument("--universe-path", default=None)
    ap.add_argument("--universe-key", default=None)
    ap.add_argument("--strict-universe", action="store_true")

    ap.add_argument("--choice-id", default=None)
    ap.add_argument("--portfolio-run-id", default=None)
    ap.add_argument("--note", default=None)
    ap.add_argument("--action-tag", default=None)
    ap.add_argument("--quantity-unit", default=None)
    ap.add_argument("--value", default=None, type=float)
    ap.add_argument("--reported-pnl", default=None, type=float)

    ap.add_argument("--open-value", default=None, type=float)
    ap.add_argument("--entry-price", default=None, type=float)

    ap.add_argument(
        "--asset-class",
        default=None,
        choices=["stock", "crypto", "forex", "futures", "commodity", "unknown"],
    )

    ap.add_argument(
        "--stop-loss-pct",
        default=None,
        type=float,
        help="Decimal stop loss percentage, e.g. 0.08 for 8%%.",
    )

    ap.add_argument(
        "--target-profit-pct",
        default=None,
        type=float,
        help="Decimal target profit percentage, e.g. 0.15 for 15%%.",
    )

    ap.add_argument(
        "--max-holding-days",
        default=None,
        type=int,
    )

    ap.add_argument(
        "--disable-risk-contract",
        action="store_true",
        help="Disable automatic SL/TP/time-barrier creation for this trade.",
    )

    ap.add_argument("--old-as-of", default=None)
    ap.add_argument("--new-as-of", default=None)
    ap.add_argument("--no-backup", action="store_true")

    ap.add_argument("--reason", default=None, help="Business reason for edit/migration/correction. Recommended for audit trail.")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument(
        "--risk-model",
        default="fixed_by_asset_class",
        choices=[
            "fixed_by_asset_class",
            "atr_based",
            "volatility_based",
            "hybrid",
        ],
    )

    ap.add_argument(
        "--indicator-mode",
        default="auto",
        choices=["auto", "latest", "point_in_time"],
    )

    ap.add_argument(
        "--indicators-snapshot-key",
        default=None,
        help=(
            "Override latest indicators snapshot key. Default is "
            "<market_root>/snapshots/v1/latest_indicators.parquet."
        ),
    )

    ap.add_argument(
        "--indicators-root-prefix",
        default=None,
        help=(
            "Override historical indicator root. Default is "
            "<market_root>/indicators/v1."
        ),
    )

    ap.add_argument(
        "--max-indicator-staleness-days",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--atr-stop-multiplier",
        type=float,
        default=2.0,
    )

    ap.add_argument(
        "--atr-target-multiplier",
        type=float,
        default=4.0,
    )

    ap.add_argument(
        "--volatility-stop-multiplier",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--volatility-target-multiplier",
        type=float,
        default=1.8,
    )

    ap.add_argument(
        "--reward-multiple",
        type=float,
        default=2.0,
    )

    ap.add_argument(
        "--min-stop-pct",
        type=float,
        default=0.02,
    )

    ap.add_argument(
        "--max-stop-pct",
        type=float,
        default=0.25,
    )

    ap.add_argument(
        "--max-target-pct",
        type=float,
        default=0.95,
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)

    if not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    input_args = vars(args)

    with capture_script_run(
        cfg=cfg,
        script_name="record_trade.py",
        input_args=input_args,
        dry_run=bool(args.dry_run),
    ) as run_id:
        print(f"[runtime] env={cfg.env} bucket={cfg.bucket} root={cfg.engine_root}")

        if args.mode == "migrate":
            migrate_trades_index(cfg=cfg, dry_run=bool(args.dry_run), run_id=run_id, input_args=input_args)
            return


        if args.mode == "delete":
            if not args.trade_id:
                raise ValueError("--trade-id is required for --mode delete")

            delete_trade(
                cfg=cfg,
                trade_id=str(args.trade_id),
                old_as_of=(str(args.old_as_of) if args.old_as_of else None),
                reason=args.reason,
                dry_run=bool(args.dry_run),
                run_id=run_id,
                input_args=input_args,
            )
            return

        if args.mode == "edit":
            if not args.trade_id:
                raise ValueError("--trade-id is required for --mode edit")

            patch: Dict[str, Any] = {}

            if args.action_tag is not None:
                patch["action_tag"] = _normalize_action_tag(args.action_tag)
            if args.quantity_unit is not None:
                patch["quantity_unit"] = _normalize_quantity_unit(args.quantity_unit)
            if args.value is not None:
                patch["value"] = float(args.value)
            if args.reported_pnl is not None:
                patch["reported_pnl"] = float(args.reported_pnl)
            if args.note is not None:
                patch["note"] = (str(args.note) if args.note else None)
            if args.side is not None:
                patch["side"] = _validate_side(args.side)
            if args.quantity is not None:
                patch["quantity"] = _validate_positive("quantity", args.quantity)
            if args.price is not None:
                patch["price"] = _validate_positive("price", args.price)

            if not patch and not args.new_as_of:
                raise ValueError("Nothing to edit: provide at least one patch field or --new-as-of.")

            edit_trade(
                cfg=cfg,
                trade_id=str(args.trade_id),
                old_as_of=(str(args.old_as_of) if args.old_as_of else None),
                new_as_of=(str(args.new_as_of) if args.new_as_of else None),
                patch=patch,
                dry_run=bool(args.dry_run),
                write_backup=(not bool(args.no_backup)),
                run_id=run_id,
                input_args=input_args,
                reason=args.reason,
            )
            return

        for name in ["as_of", "ticker", "side", "action_tag"]:
            if getattr(args, name) in (None, ""):
                raise ValueError(f"--{name.replace('_', '-')} is required for --mode record")

        provided_economics = sum(
            getattr(args, name) is not None
            for name in ["quantity", "price", "value"]
        )

        if provided_economics < 2:
            raise ValueError("For --mode record, provide at least two of --quantity, --price, --value.")

        record_trade(
            cfg=cfg,
            as_of=str(args.as_of),
            ticker=str(args.ticker),
            side=str(args.side),
            quantity=(float(args.quantity) if args.quantity is not None else None),
            price=(float(args.price) if args.price is not None else None),
            currency=str(args.currency),
            trade_id=args.trade_id,
            ts_utc=args.ts_utc,
            asset_id=args.asset_id,
            universe_path=args.universe_path,
            universe_key=args.universe_key,
            strict_universe=bool(args.strict_universe),
            choice_id=args.choice_id,
            portfolio_run_id=args.portfolio_run_id,
            note=args.note,
            dry_run=bool(args.dry_run),
            action_tag=args.action_tag,
            quantity_unit=args.quantity_unit,
            value=args.value,
            reported_pnl=args.reported_pnl,
            open_value=args.open_value,
            entry_price=args.entry_price,
            asset_class=args.asset_class,
            risk_model=args.risk_model,
            stop_loss_pct=args.stop_loss_pct,
            target_profit_pct=args.target_profit_pct,
            max_holding_days=args.max_holding_days,
            disable_risk_contract=bool(args.disable_risk_contract),
            run_id=run_id,
            source_script="record_trade.py",
            source_mode="record",
            input_args=input_args,
            reason=args.reason,
            indicator_mode=args.indicator_mode,
            indicators_snapshot_key=args.indicators_snapshot_key,
            indicators_root_prefix=args.indicators_root_prefix,
            max_indicator_staleness_days=int(
                args.max_indicator_staleness_days
            ),
            atr_stop_multiplier=float(args.atr_stop_multiplier),
            atr_target_multiplier=float(args.atr_target_multiplier),
            volatility_stop_multiplier=float(
                args.volatility_stop_multiplier
            ),
            volatility_target_multiplier=float(
                args.volatility_target_multiplier
            ),
            reward_multiple=float(args.reward_multiple),
            min_stop_pct=float(args.min_stop_pct),
            max_stop_pct=float(args.max_stop_pct),
            max_target_pct=float(args.max_target_pct),
        )


if __name__ == "__main__":
    main()