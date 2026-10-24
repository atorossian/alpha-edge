from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd


PositionSide = Literal["LONG", "SHORT"]
Recommendation = Literal["HOLD", "WATCH", "REDUCE", "EXIT", "DATA_ISSUE"]
BreachType = Literal[
    "NONE",
    "WICK_BREACH_RECOVERED",
    "WICK_BREACH_UNCONFIRMED",
    "CLOSE_BREACH",
    "CONFIRMED_BREAK",
    "GAP_THROUGH_STOP",
    "NO_PRICE",
]
ConfirmationStatus = Literal[
    "NONE",
    "WATCH_ONLY",
    "FIRST_CLOSE_BREACH",
    "CONFIRMED",
    "EMERGENCY",
    "RECOVERED",
    "DATA_ISSUE",
]


@dataclass(frozen=True)
class BarrierMonitorConfig:
    """
    Daily open-position barrier settings.

    The monitor is advisory: it calculates risk thresholds and exit-quality
    diagnostics for open lots/positions, but it does not create exit trades.
    v2 adds liquidity-aware stop interpretation so a wick through a level does
    not mechanically become an EXIT recommendation.
    """

    default_stop_loss_pct: float = 0.08
    default_target_profit_pct: float = 0.15

    chandelier_atr_multiplier: float = 3.0
    profit_lock_activation_return: float = 0.10
    profit_lock_ratio: float = 0.50

    near_stop_threshold_pct: float = 0.02
    near_target_threshold_pct: float = 0.02
    max_price_staleness_days: int = 5
    max_indicator_staleness_days: int = 10

    use_fixed_stop: bool = True
    use_fixed_target: bool = True
    use_chandelier_stop: bool = True
    use_profit_lock: bool = True

    # Liquidity-aware interpretation layer.
    liquidity_aware_stops: bool = True
    stop_buffer_atr_multiplier: float = 0.25
    stop_buffer_pct: float = 0.003
    close_breach_reduce_threshold_pct: float = 0.015
    confirmed_break_breach_count: int = 2
    emergency_gap_atr_multiplier: float = 1.0
    emergency_gap_pct: float = 0.03
    wick_breach_action: Recommendation = "WATCH"
    close_breach_action: Recommendation = "REDUCE"
    confirmed_break_action: Recommendation = "EXIT"
    gap_through_stop_action: Recommendation = "EXIT"

    # Conservative v1 guardrails. These intentionally keep Milestone 21 v1
    # independent from regime/portfolio-health logic, which is deferred to v2.
    conservative_advisory_mode: bool = True
    time_exit_action: Recommendation = "WATCH"
    time_exit_exit_fraction: float = 0.0
    severe_breach_pct: float = 0.05
    severe_breach_atr_multiplier: float = 2.0
    confirmed_break_exit_fraction: float = 0.50
    confirmed_break_small_action: Recommendation = "REDUCE"
    confirmed_break_small_exit_fraction: float = 0.25
    gap_through_stop_exit_fraction: float = 1.0

    # Portfolio-level liquidation throttles applied after per-lot/position signals.
    apply_liquidation_throttle: bool = True
    max_daily_exit_fraction_of_portfolio: float = 0.25
    max_daily_exit_value: Optional[float] = None
    max_full_exits_per_day: int = 2
    max_reduce_actions_per_day: int = 8
    throttle_deferred_action: Recommendation = "WATCH"

    # Optional per asset-class/risk-bucket overrides. Keys may be asset classes
    # like stock/crypto/commodity/bond, source lot types, or inferred buckets.
    # Values should be a dict of BarrierMonitorConfig field -> value.
    asset_class_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    fail_on_missing_history: bool = False
    fail_on_missing_atr: bool = False

    version: str = "open_position_barriers_v1_production_ready"

DEFAULT_ASSET_CLASS_OVERRIDES: dict[str, dict[str, Any]] = {
    # Wider crypto buffer and confirmation window to avoid noisy stop-outs.
    "crypto": {
        "stop_buffer_pct": 0.015,
        "emergency_gap_pct": 0.08,
        "confirmed_break_exit_fraction": 0.50,
    },
    # Bond/credit ETFs are lower-vol instruments; tiny breaks should not force full exits.
    "bond": {
        "stop_buffer_pct": 0.002,
        "severe_breach_pct": 0.02,
        "confirmed_break_action": "REDUCE",
        "confirmed_break_exit_fraction": 0.25,
        "gap_through_stop_action": "REDUCE",
        "gap_through_stop_exit_fraction": 0.50,
    },
    "credit": {
        "stop_buffer_pct": 0.002,
        "severe_breach_pct": 0.02,
        "confirmed_break_action": "REDUCE",
        "confirmed_break_exit_fraction": 0.25,
        "gap_through_stop_action": "REDUCE",
        "gap_through_stop_exit_fraction": 0.50,
    },
    # Precious metals can trend, but profit-lock/chandelier outputs need conservative exits.
    "commodity": {
        "severe_breach_pct": 0.06,
        "confirmed_break_exit_fraction": 0.50,
        "gap_through_stop_exit_fraction": 0.50,
    },
}


@dataclass(frozen=True)
class OpenLotInput:
    trade_id: str
    asset_id: str
    ticker: str
    side: PositionSide
    entry_date: str
    entry_price: float
    quantity_open: Optional[float] = None
    quantity_remaining: Optional[float] = None
    value_open: Optional[float] = None
    value_remaining: Optional[float] = None
    quantity_unit: Optional[str] = None
    source_lot_type: str = "unknown"
    trade_contract: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LotBarrierState:
    state_key: str
    trade_id: str
    asset_id: str
    side: PositionSide
    last_as_of: Optional[str] = None
    breach_count: int = 0
    last_breach_type: Optional[str] = None
    last_breach_date: Optional[str] = None
    last_recovery_date: Optional[str] = None
    prior_active_stop_price: Optional[float] = None
    prior_recommendation: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LotBarrierResult:
    as_of: str
    asset_id: str
    ticker: str
    trade_id: str
    side: PositionSide
    source_lot_type: str
    asset_bucket: str
    entry_date: str
    entry_price: float
    quantity_remaining: Optional[float]
    value_remaining: Optional[float]
    current_price: Optional[float]
    current_price_date: Optional[str]
    current_price_staleness_days: Optional[int]
    latest_open: Optional[float]
    latest_high: Optional[float]
    latest_low: Optional[float]
    latest_close: Optional[float]
    latest_bar_date: Optional[str]
    highest_high_since_entry: Optional[float]
    lowest_low_since_entry: Optional[float]
    highest_close_since_entry: Optional[float]
    lowest_close_since_entry: Optional[float]
    atr_14: Optional[float]
    atr_pct_14: Optional[float]
    fixed_stop_loss_price: Optional[float]
    fixed_target_profit_price: Optional[float]
    chandelier_stop_price: Optional[float]
    profit_lock_stop_price: Optional[float]
    raw_active_stop_price: Optional[float]
    active_stop_price: Optional[float]
    active_stop_source: Optional[str]
    risk_threshold_price: Optional[float]
    execution_trigger_price: Optional[float]
    stop_buffer_amount: Optional[float]
    distance_to_stop_pct: Optional[float]
    distance_to_target_pct: Optional[float]
    peak_return: Optional[float]
    current_return: Optional[float]
    profit_lock_activated: bool
    stop_status: str
    target_status: str
    stop_triggered: bool
    target_reached: bool
    time_exit_due: bool
    breach_type: BreachType
    confirmation_status: ConfirmationStatus
    previous_breach_count: int
    breach_count: int
    liquidity_sweep_recovered: bool
    recommended_exit_fraction: float
    recommended_exit_quantity: Optional[float]
    recommended_exit_value: Optional[float]
    recommended_action: Recommendation
    trigger_reason: Optional[str]
    barrier_quality: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PositionBarrierResult:
    as_of: str
    asset_id: str
    ticker: str
    side: PositionSide
    asset_bucket: str
    open_lots: int
    quantity_remaining: Optional[float]
    value_remaining: Optional[float]
    current_price: Optional[float]
    entry_date_min: Optional[str]
    entry_date_max: Optional[str]
    avg_entry_price: Optional[float]
    current_return_pct: Optional[float]
    highest_high_since_entry: Optional[float]
    lowest_low_since_entry: Optional[float]
    atr_14: Optional[float]
    fixed_stop_loss_price: Optional[float]
    fixed_target_profit_price: Optional[float]
    chandelier_stop_price: Optional[float]
    profit_lock_stop_price: Optional[float]
    risk_threshold_price: Optional[float]
    execution_trigger_price: Optional[float]
    stop_buffer_amount: Optional[float]
    latest_open: Optional[float]
    latest_high: Optional[float]
    latest_low: Optional[float]
    latest_close: Optional[float]
    latest_bar_date: Optional[str]
    stop_status: str
    target_status: str
    active_stop_price: Optional[float]
    active_stop_source: Optional[str]
    nearest_distance_to_stop_pct: Optional[float]
    nearest_distance_to_target_pct: Optional[float]
    stop_triggered_lots: int
    target_reached_lots: int
    time_exit_due_lots: int
    breach_type: str
    confirmation_status: str
    max_breach_count: int
    recommended_exit_fraction: float
    recommended_exit_quantity: Optional[float]
    recommended_exit_value: Optional[float]
    recommended_action: Recommendation
    trigger_reason: Optional[str]
    barrier_quality: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BarrierMonitorResult:
    as_of: str
    generated_at_utc: str
    config: dict[str, Any]
    lot_results: list[LotBarrierResult]
    position_results: list[PositionBarrierResult]
    next_state: dict[str, dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "generated_at_utc": self.generated_at_utc,
            "config": self.config,
            "lot_results": [r.to_dict() for r in self.lot_results],
            "position_results": [r.to_dict() for r in self.position_results],
            "next_state": dict(self.next_state),
            "summary": dict(self.summary),
        }


def _as_date(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value!r}")
    return pd.Timestamp(ts).tz_convert(None).normalize()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _positive_or_none(value: Any) -> Optional[float]:
    v = _safe_float(value)
    if v is None or v <= 0:
        return None
    return v


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _price_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = {str(c).lower(): str(c) for c in df.columns}

    def first(names: list[str]) -> str | None:
        for name in names:
            if name.lower() in cols:
                return cols[name.lower()]
        return None

    return {
        "open": first(["open_adjusted_usd", "open_usd", "open_raw_usd", "open"]),
        "close": first(["adj_close_usd", "close_adjusted_usd", "close_usd", "close_raw_usd", "adj_close", "close"]),
        "high": first(["high_adjusted_usd", "high_usd", "high_raw_usd", "high"]),
        "low": first(["low_adjusted_usd", "low_usd", "low_raw_usd", "low"]),
    }


def normalize_ohlcv_history(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame(columns=["date", "asset_id", "ticker", "open", "close", "high", "low"])

    required = {"date", "asset_id"}
    missing = sorted(required - set(ohlcv.columns))
    if missing:
        raise RuntimeError(f"OHLCV history missing required columns: {missing}")

    mapping = _price_columns(ohlcv)
    if mapping["close"] is None:
        raise RuntimeError(f"OHLCV history has no usable close column. Columns={list(ohlcv.columns)}")

    out = ohlcv.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    out["asset_id"] = out["asset_id"].astype(str).str.strip()
    if "ticker" not in out.columns:
        out["ticker"] = None

    close_col = str(mapping["close"])
    out["close"] = pd.to_numeric(out[close_col], errors="coerce")

    open_col = mapping["open"]
    high_col = mapping["high"]
    low_col = mapping["low"]
    out["open"] = pd.to_numeric(out[open_col], errors="coerce") if open_col else out["close"]
    out["high"] = pd.to_numeric(out[high_col], errors="coerce") if high_col else out["close"]
    out["low"] = pd.to_numeric(out[low_col], errors="coerce") if low_col else out["close"]

    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out["close"])
    out["low"] = out["low"].fillna(out["close"])

    out = out.dropna(subset=["date", "asset_id", "close"])
    out = out[(out["close"] > 0) & (out["open"] > 0) & (out["high"] > 0) & (out["low"] > 0)].copy()
    return out[["date", "asset_id", "ticker", "open", "close", "high", "low"]].sort_values(["asset_id", "date"]).reset_index(drop=True)


def latest_prices_by_asset_id(latest_prices: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if latest_prices is None or latest_prices.empty:
        return {}

    if "asset_id" not in latest_prices.columns:
        raise RuntimeError("latest_prices snapshot missing asset_id column.")

    mapping = _price_columns(latest_prices)
    if mapping["close"] is None:
        raise RuntimeError(f"latest_prices snapshot has no usable price column. Columns={list(latest_prices.columns)}")

    df = latest_prices.copy()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["price"] = pd.to_numeric(df[str(mapping["close"])], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    else:
        df["date"] = pd.NaT
    df = df.dropna(subset=["asset_id", "price"])
    df = df[df["price"] > 0].copy()
    df = df.sort_values(["asset_id", "date"], kind="stable").drop_duplicates("asset_id", keep="last")

    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        dt_val = row.get("date")
        out[str(row["asset_id"])] = {
            "price": float(row["price"]),
            "date": None if pd.isna(dt_val) else pd.Timestamp(dt_val).date().isoformat(),
        }
    return out


def indicators_by_asset_id(indicators: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if indicators is None or indicators.empty or "asset_id" not in indicators.columns:
        return {}

    df = indicators.copy()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    elif "indicator_date" in df.columns:
        df["date"] = pd.to_datetime(df["indicator_date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    else:
        df["date"] = pd.NaT

    df = df.sort_values(["asset_id", "date"], kind="stable").drop_duplicates("asset_id", keep="last")
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        aid = str(row["asset_id"])
        date_val = row.get("date")
        out[aid] = {
            "atr_14": _positive_or_none(row.get("atr_14")),
            "atr_pct_14": _positive_or_none(row.get("atr_pct_14")),
            "date": None if pd.isna(date_val) else pd.Timestamp(date_val).date().isoformat(),
        }
    return out


def open_lots_from_ledger_trace(
    open_lot_trace_payload: dict[str, Any],
    *,
    trade_contracts_by_trade_id: Optional[dict[str, dict[str, Any]]] = None,
) -> list[OpenLotInput]:
    payload = open_lot_trace_payload or {}
    raw_lots = payload.get("open_lots") if isinstance(payload.get("open_lots"), dict) else payload
    if not isinstance(raw_lots, dict):
        return []

    contracts = trade_contracts_by_trade_id or {}
    out: list[OpenLotInput] = []

    lot_specs = [
        ("spot_long_lots", "LONG", "spot_long"),
        ("spot_short_lots", "SHORT", "spot_short"),
        ("notional_long_lots", "LONG", "notional_long"),
        ("notional_short_lots", "SHORT", "notional_short"),
    ]

    for key, side, lot_type in lot_specs:
        for lot in raw_lots.get(key) or []:
            if not isinstance(lot, dict):
                continue
            trade_id = str(lot.get("trade_id") or "").strip()
            asset_id = str(lot.get("asset_id") or "").strip()
            ticker = str(lot.get("ticker") or asset_id).strip().upper()
            entry_price = _positive_or_none(lot.get("price") or lot.get("entry_price"))
            entry_date = lot.get("as_of") or lot.get("entry_date")
            if not trade_id or not asset_id or entry_price is None or not entry_date:
                continue
            out.append(
                OpenLotInput(
                    trade_id=trade_id,
                    asset_id=asset_id,
                    ticker=ticker,
                    side=side,  # type: ignore[arg-type]
                    entry_date=str(entry_date),
                    entry_price=float(entry_price),
                    quantity_open=_safe_float(lot.get("quantity_open") or lot.get("quantity")),
                    quantity_remaining=_safe_float(lot.get("quantity_remaining") or lot.get("quantity")),
                    value_open=_safe_float(lot.get("value_open") or lot.get("value")),
                    value_remaining=_safe_float(lot.get("value_remaining") or lot.get("value")),
                    quantity_unit=(str(lot.get("quantity_unit")) if lot.get("quantity_unit") is not None else None),
                    source_lot_type=lot_type,
                    trade_contract=dict(contracts.get(trade_id) or {}),
                )
            )

    return out


def lot_state_key(lot: OpenLotInput) -> str:
    return f"{lot.trade_id}|{lot.asset_id}|{lot.side}"


def normalize_barrier_state(previous_state: Optional[dict[str, Any]]) -> dict[str, LotBarrierState]:
    if not previous_state:
        return {}
    raw = previous_state.get("lot_state") if isinstance(previous_state.get("lot_state"), dict) else previous_state
    if not isinstance(raw, dict):
        return {}

    out: dict[str, LotBarrierState] = {}
    for key, obj in raw.items():
        if not isinstance(obj, dict):
            continue
        try:
            side = str(obj.get("side") or "LONG").upper()
            if side not in {"LONG", "SHORT"}:
                side = "LONG"
            st = LotBarrierState(
                state_key=str(obj.get("state_key") or key),
                trade_id=str(obj.get("trade_id") or ""),
                asset_id=str(obj.get("asset_id") or ""),
                side=side,  # type: ignore[arg-type]
                last_as_of=(str(obj.get("last_as_of")) if obj.get("last_as_of") is not None else None),
                breach_count=int(_safe_float(obj.get("breach_count")) or 0),
                last_breach_type=(str(obj.get("last_breach_type")) if obj.get("last_breach_type") is not None else None),
                last_breach_date=(str(obj.get("last_breach_date")) if obj.get("last_breach_date") is not None else None),
                last_recovery_date=(str(obj.get("last_recovery_date")) if obj.get("last_recovery_date") is not None else None),
                prior_active_stop_price=_safe_float(obj.get("prior_active_stop_price")),
                prior_recommendation=(str(obj.get("prior_recommendation")) if obj.get("prior_recommendation") is not None else None),
            )
            out[str(key)] = st
        except Exception:
            continue
    return out


def _infer_asset_bucket(lot: OpenLotInput) -> str:
    contract = lot.trade_contract or {}
    explicit = contract.get("asset_class") or contract.get("asset_bucket") or contract.get("risk_bucket")
    if explicit:
        return str(explicit).lower().strip()
    aid = str(lot.asset_id).upper().strip()
    unit = str(lot.quantity_unit or "").lower().strip()
    ticker = str(lot.ticker or "").upper().strip()
    if aid.startswith("CRYPTO:") or unit == "coins":
        return "crypto"
    if unit == "ounces" or any(x in ticker for x in ["GLD", "SLV", "GDX", "PPLT", "PALL"]):
        return "commodity"
    if any(x in ticker for x in ["IEF", "TLT", "SHY", "BIL"]):
        return "bond"
    if any(x in ticker for x in ["VCIT", "LQD", "HYG", "IEF", "TLT"]):
        return "credit"
    if lot.source_lot_type.startswith("notional"):
        return "notional"
    return "stock"


def effective_config_for_lot(cfg: BarrierMonitorConfig, lot: OpenLotInput) -> BarrierMonitorConfig:
    bucket = _infer_asset_bucket(lot)
    overrides = dict(DEFAULT_ASSET_CLASS_OVERRIDES.get(bucket) or {})
    overrides.update(dict(cfg.asset_class_overrides.get(bucket) or {}))
    if not overrides:
        return cfg
    valid = {f.name for f in cfg.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    clean = {k: v for k, v in overrides.items() if k in valid and k != "asset_class_overrides"}
    return replace(cfg, **clean)


def _fixed_stop_price(lot: OpenLotInput, cfg: BarrierMonitorConfig) -> Optional[float]:
    contract = lot.trade_contract or {}
    existing = _positive_or_none(contract.get("stop_loss_price"))
    if existing is not None:
        return existing
    pct = _positive_or_none(contract.get("stop_loss_pct")) or float(cfg.default_stop_loss_pct)
    if lot.side == "LONG":
        return float(lot.entry_price) * (1.0 - pct)
    return float(lot.entry_price) * (1.0 + pct)


def _fixed_target_price(lot: OpenLotInput, cfg: BarrierMonitorConfig) -> Optional[float]:
    contract = lot.trade_contract or {}
    existing = _positive_or_none(contract.get("target_profit_price"))
    if existing is not None:
        return existing
    pct = _positive_or_none(contract.get("target_profit_pct")) or float(cfg.default_target_profit_pct)
    if lot.side == "LONG":
        return float(lot.entry_price) * (1.0 + pct)
    target = float(lot.entry_price) * (1.0 - pct)
    return target if target > 0 else None


def _current_return(side: PositionSide, entry: float, current: Optional[float]) -> Optional[float]:
    if current is None or current <= 0 or entry <= 0:
        return None
    if side == "LONG":
        return float(current / entry - 1.0)
    return float(entry / current - 1.0)


def _peak_return(side: PositionSide, entry: float, highest_high: Optional[float], lowest_low: Optional[float]) -> Optional[float]:
    if entry <= 0:
        return None
    if side == "LONG":
        if highest_high is None or highest_high <= 0:
            return None
        return float(highest_high / entry - 1.0)
    if lowest_low is None or lowest_low <= 0:
        return None
    return float(entry / lowest_low - 1.0)


def _better_stop(side: PositionSide, candidates: dict[str, Optional[float]]) -> tuple[Optional[float], Optional[str]]:
    valid = {k: float(v) for k, v in candidates.items() if v is not None and np.isfinite(float(v)) and float(v) > 0}
    if not valid:
        return None, None
    if side == "LONG":
        k = max(valid, key=lambda name: valid[name])
    else:
        k = min(valid, key=lambda name: valid[name])
    return valid[k], k


def _distance_to_stop(side: PositionSide, current: Optional[float], stop: Optional[float]) -> Optional[float]:
    if current is None or stop is None or current <= 0 or stop <= 0:
        return None
    if side == "LONG":
        return float(current / stop - 1.0)
    return float(stop / current - 1.0)


def _distance_to_target(side: PositionSide, current: Optional[float], target: Optional[float]) -> Optional[float]:
    if current is None or target is None or current <= 0 or target <= 0:
        return None
    if side == "LONG":
        return float(target / current - 1.0)
    return float(current / target - 1.0)


def _stop_buffer_amount(*, current_price: Optional[float], atr_14: Optional[float], cfg: BarrierMonitorConfig) -> Optional[float]:
    candidates: list[float] = []
    if atr_14 is not None and atr_14 > 0:
        candidates.append(float(atr_14) * float(cfg.stop_buffer_atr_multiplier))
    if current_price is not None and current_price > 0:
        candidates.append(float(current_price) * float(cfg.stop_buffer_pct))
    if not candidates:
        return None
    return float(max(candidates))


def _execution_trigger(side: PositionSide, raw_stop: Optional[float], buffer_amount: Optional[float]) -> Optional[float]:
    if raw_stop is None or raw_stop <= 0:
        return None
    b = float(buffer_amount or 0.0)
    if side == "LONG":
        return max(0.0, float(raw_stop) - b)
    return float(raw_stop) + b


def _latest_bar(h: pd.DataFrame) -> dict[str, Any]:
    if h is None or h.empty:
        return {}
    row = h.sort_values("date", kind="stable").iloc[-1]
    dt = row.get("date")
    return {
        "date": None if pd.isna(dt) else pd.Timestamp(dt).date().isoformat(),
        "open": _positive_or_none(row.get("open")),
        "high": _positive_or_none(row.get("high")),
        "low": _positive_or_none(row.get("low")),
        "close": _positive_or_none(row.get("close")),
    }


def _breach_magnitude(side: PositionSide, close: Optional[float], stop: Optional[float]) -> float:
    if close is None or stop is None or close <= 0 or stop <= 0:
        return 0.0
    if side == "LONG" and close < stop:
        return float(stop / close - 1.0)
    if side == "SHORT" and close > stop:
        return float(close / stop - 1.0)
    return 0.0


def _severe_breach(
    *,
    side: PositionSide,
    price: Optional[float],
    risk_threshold: Optional[float],
    atr_14: Optional[float],
    cfg: BarrierMonitorConfig,
) -> bool:
    mag = _breach_magnitude(side, price, risk_threshold)
    if mag >= float(cfg.severe_breach_pct):
        return True
    if price is not None and risk_threshold is not None and atr_14 is not None and atr_14 > 0:
        return abs(float(price) - float(risk_threshold)) >= float(cfg.severe_breach_atr_multiplier) * float(atr_14)
    return False


def _stop_status(
    *,
    side: PositionSide,
    risk_threshold: Optional[float],
    execution_trigger: Optional[float],
    latest_high: Optional[float],
    latest_low: Optional[float],
    latest_close: Optional[float],
    current_price: Optional[float],
    distance_stop: Optional[float],
    cfg: BarrierMonitorConfig,
) -> str:
    if risk_threshold is None or current_price is None:
        return "NO_PRICE_OR_STOP"
    close = latest_close or current_price
    high = latest_high or close
    low = latest_low or close
    if side == "LONG":
        if execution_trigger is not None and close <= execution_trigger:
            return "BELOW_EXECUTION_TRIGGER"
        if close <= risk_threshold:
            return "BELOW_RISK_THRESHOLD"
        if low <= risk_threshold and close > risk_threshold:
            return "WICK_BREACH_RECOVERED"
    else:
        if execution_trigger is not None and close >= execution_trigger:
            return "ABOVE_EXECUTION_TRIGGER"
        if close >= risk_threshold:
            return "ABOVE_RISK_THRESHOLD"
        if high >= risk_threshold and close < risk_threshold:
            return "WICK_BREACH_RECOVERED"
    if distance_stop is not None and distance_stop <= float(cfg.near_stop_threshold_pct):
        return "NEAR_STOP"
    return "CLEAR"


def _target_status(
    *,
    side: PositionSide,
    fixed_target: Optional[float],
    current_price: Optional[float],
    distance_target: Optional[float],
    cfg: BarrierMonitorConfig,
) -> str:
    if fixed_target is None or current_price is None:
        return "NO_TARGET"
    reached = bool(current_price >= fixed_target) if side == "LONG" else bool(current_price <= fixed_target)
    if reached:
        return "TARGET_REACHED"
    if distance_target is not None and distance_target <= float(cfg.near_target_threshold_pct):
        return "NEAR_TARGET"
    return "BELOW_TARGET"


def classify_liquidity_aware_breach(
    *,
    side: PositionSide,
    risk_threshold: Optional[float],
    execution_trigger: Optional[float],
    latest_open: Optional[float],
    latest_high: Optional[float],
    latest_low: Optional[float],
    latest_close: Optional[float],
    current_price: Optional[float],
    atr_14: Optional[float],
    previous_breach_count: int,
    cfg: BarrierMonitorConfig,
) -> tuple[BreachType, ConfirmationStatus, int, bool, float, Recommendation, str | None, bool]:
    if risk_threshold is None or risk_threshold <= 0 or current_price is None or current_price <= 0:
        return "NO_PRICE", "DATA_ISSUE", previous_breach_count, False, 0.0, "DATA_ISSUE", "missing_price_or_stop", False

    close = latest_close or current_price
    high = latest_high or max(close, current_price)
    low = latest_low or min(close, current_price)
    open_px = latest_open or close

    if side == "LONG":
        wick_breached = bool(low <= risk_threshold)
        close_breached = bool(close <= risk_threshold)
        execution_breached = bool(execution_trigger is not None and close <= execution_trigger)
        gap_breached = bool(open_px <= (execution_trigger or risk_threshold))
        recovered = bool(wick_breached and close > risk_threshold)
    else:
        wick_breached = bool(high >= risk_threshold)
        close_breached = bool(close >= risk_threshold)
        execution_breached = bool(execution_trigger is not None and close >= execution_trigger)
        gap_breached = bool(open_px >= (execution_trigger or risk_threshold))
        recovered = bool(wick_breached and close < risk_threshold)

    emergency_gap = False
    if gap_breached:
        pct_mag = _breach_magnitude(side, open_px, risk_threshold)
        atr_pct_threshold = None
        if atr_14 is not None and atr_14 > 0 and risk_threshold is not None and risk_threshold > 0:
            atr_pct_threshold = float(cfg.emergency_gap_atr_multiplier) * float(atr_14) / float(risk_threshold)
        # Conservative v1: a small gap just beyond a buffered trigger is not an
        # emergency. Require a genuinely large percentage/ATR-adjusted breach.
        threshold = float(cfg.emergency_gap_pct)
        if atr_pct_threshold is not None:
            threshold = max(threshold, float(atr_pct_threshold))
        emergency_gap = bool(pct_mag >= threshold)

    if emergency_gap:
        fraction = float(cfg.gap_through_stop_exit_fraction)
        action = cfg.gap_through_stop_action
        return "GAP_THROUGH_STOP", "EMERGENCY", max(previous_breach_count + 1, cfg.confirmed_break_breach_count), False, fraction, action, "gap_through_execution_trigger", True

    if recovered:
        return "WICK_BREACH_RECOVERED", "RECOVERED", 0, True, 0.0, cfg.wick_breach_action, "wick_breach_recovered", False

    if not wick_breached and not close_breached and not execution_breached:
        return "NONE", "NONE", 0, False, 0.0, "HOLD", None, False

    if close_breached or execution_breached:
        breach_count = int(previous_breach_count) + 1
        confirmed = breach_count >= int(cfg.confirmed_break_breach_count)
        severe = _severe_breach(side=side, price=close, risk_threshold=risk_threshold, atr_14=atr_14, cfg=cfg)
        if confirmed:
            if bool(cfg.conservative_advisory_mode) and not severe:
                return "CONFIRMED_BREAK", "CONFIRMED", breach_count, False, float(cfg.confirmed_break_small_exit_fraction), cfg.confirmed_break_small_action, "confirmed_stop_break_small", True
            return "CONFIRMED_BREAK", "CONFIRMED", breach_count, False, float(cfg.confirmed_break_exit_fraction), cfg.confirmed_break_action, "confirmed_stop_break", True

        mag = _breach_magnitude(side, close, risk_threshold)
        frac = 0.50 if mag >= float(cfg.close_breach_reduce_threshold_pct) else 0.25
        action = cfg.close_breach_action
        return "CLOSE_BREACH", "FIRST_CLOSE_BREACH", breach_count, False, frac, action, "close_below_risk_threshold" if side == "LONG" else "close_above_risk_threshold", True

    if wick_breached:
        return "WICK_BREACH_UNCONFIRMED", "WATCH_ONLY", previous_breach_count, False, 0.0, cfg.wick_breach_action, "wick_breach_unconfirmed", False

    return "NONE", "NONE", 0, False, 0.0, "HOLD", None, False


def evaluate_open_lot_barrier(
    *,
    lot: OpenLotInput,
    as_of: str,
    ohlcv_history: pd.DataFrame,
    latest_prices: dict[str, dict[str, Any]],
    indicators: Optional[dict[str, dict[str, Any]]] = None,
    previous_state: Optional[dict[str, LotBarrierState]] = None,
    config: Optional[BarrierMonitorConfig] = None,
) -> LotBarrierResult:
    base_cfg = config or BarrierMonitorConfig()
    cfg = effective_config_for_lot(base_cfg, lot)
    warnings: list[str] = []
    as_of_ts = _as_date(as_of)
    entry_ts = _as_date(lot.entry_date)

    state_key = lot_state_key(lot)
    prior_state = (previous_state or {}).get(state_key)
    previous_breach_count = int(prior_state.breach_count) if prior_state else 0

    latest = latest_prices.get(lot.asset_id) or {}
    current_price = _positive_or_none(latest.get("price"))
    current_price_date = latest.get("date")
    price_staleness_days: Optional[int] = None
    if current_price_date:
        try:
            price_staleness_days = int((as_of_ts - _as_date(current_price_date)).days)
            if price_staleness_days < 0:
                warnings.append("latest_price_after_as_of")
            elif price_staleness_days > int(cfg.max_price_staleness_days):
                warnings.append(f"latest_price_stale_{price_staleness_days}d")
        except Exception:
            warnings.append("invalid_latest_price_date")
    else:
        warnings.append("latest_price_date_missing")

    if current_price is None:
        warnings.append("current_price_missing")

    h = normalize_ohlcv_history(ohlcv_history)
    h = h[(h["asset_id"] == str(lot.asset_id)) & (h["date"] >= entry_ts) & (h["date"] <= as_of_ts)].copy()

    highest_high = lowest_low = highest_close = lowest_close = None
    latest_bar = _latest_bar(h)
    if h.empty:
        warnings.append("ohlcv_history_since_entry_missing")
        if cfg.fail_on_missing_history:
            raise RuntimeError(f"Missing OHLCV history for {lot.asset_id} since {lot.entry_date}")
    else:
        highest_high = _positive_or_none(h["high"].max())
        lowest_low = _positive_or_none(h["low"].min())
        highest_close = _positive_or_none(h["close"].max())
        lowest_close = _positive_or_none(h["close"].min())

    latest_open = _positive_or_none(latest_bar.get("open"))
    latest_high = _positive_or_none(latest_bar.get("high"))
    latest_low = _positive_or_none(latest_bar.get("low"))
    latest_close = _positive_or_none(latest_bar.get("close"))
    latest_bar_date = latest_bar.get("date")

    ind = (indicators or {}).get(lot.asset_id) or {}
    atr_14 = _positive_or_none(ind.get("atr_14"))
    atr_pct_14 = _positive_or_none(ind.get("atr_pct_14"))
    if atr_14 is None and atr_pct_14 is not None and current_price is not None:
        atr_14 = float(atr_pct_14) * float(current_price)
    if atr_pct_14 is None and atr_14 is not None and current_price is not None:
        atr_pct_14 = float(atr_14) / float(current_price)

    if atr_14 is None:
        warnings.append("atr_14_missing")
        if cfg.fail_on_missing_atr:
            raise RuntimeError(f"Missing ATR_14 for {lot.asset_id}")

    fixed_stop = _fixed_stop_price(lot, cfg) if cfg.use_fixed_stop else None
    fixed_target = _fixed_target_price(lot, cfg) if cfg.use_fixed_target else None

    chandelier_stop: Optional[float] = None
    if cfg.use_chandelier_stop and atr_14 is not None:
        if lot.side == "LONG" and highest_high is not None:
            chandelier_stop = float(highest_high) - float(cfg.chandelier_atr_multiplier) * float(atr_14)
        elif lot.side == "SHORT" and lowest_low is not None:
            chandelier_stop = float(lowest_low) + float(cfg.chandelier_atr_multiplier) * float(atr_14)
        if chandelier_stop is not None and chandelier_stop <= 0:
            chandelier_stop = None

    peak_ret = _peak_return(lot.side, float(lot.entry_price), highest_high, lowest_low)
    curr_ret = _current_return(lot.side, float(lot.entry_price), current_price)

    profit_lock_activated = bool(
        cfg.use_profit_lock
        and peak_ret is not None
        and float(peak_ret) >= float(cfg.profit_lock_activation_return)
    )
    profit_lock_stop: Optional[float] = None
    if profit_lock_activated:
        if lot.side == "LONG":
            profit_lock_stop = float(lot.entry_price) * (1.0 + float(peak_ret) * float(cfg.profit_lock_ratio))
        else:
            profit_lock_stop = float(lot.entry_price) * (1.0 - float(peak_ret) * float(cfg.profit_lock_ratio))
            if profit_lock_stop <= 0:
                profit_lock_stop = None

    raw_active_stop, active_source = _better_stop(
        lot.side,
        {
            "fixed_stop_loss": fixed_stop,
            "chandelier_stop": chandelier_stop,
            "profit_lock_stop": profit_lock_stop,
        },
    )

    buffer_amount = _stop_buffer_amount(current_price=current_price, atr_14=atr_14, cfg=cfg) if cfg.liquidity_aware_stops else 0.0
    execution_trigger = _execution_trigger(lot.side, raw_active_stop, buffer_amount) if cfg.liquidity_aware_stops else raw_active_stop
    risk_threshold = raw_active_stop
    active_stop = execution_trigger if cfg.liquidity_aware_stops else raw_active_stop

    distance_stop = _distance_to_stop(lot.side, current_price, risk_threshold)
    distance_target = _distance_to_target(lot.side, current_price, fixed_target)

    stop_status = _stop_status(
        side=lot.side,
        risk_threshold=risk_threshold,
        execution_trigger=execution_trigger,
        latest_high=latest_high,
        latest_low=latest_low,
        latest_close=latest_close,
        current_price=current_price,
        distance_stop=distance_stop,
        cfg=cfg,
    )
    target_status = _target_status(
        side=lot.side,
        fixed_target=fixed_target,
        current_price=current_price,
        distance_target=distance_target,
        cfg=cfg,
    )

    target_reached = target_status == "TARGET_REACHED"

    expiry = lot.trade_contract.get("expiry_date") if lot.trade_contract else None
    time_exit_due = False
    if expiry:
        try:
            time_exit_due = bool(as_of_ts >= _as_date(expiry))
        except Exception:
            warnings.append("invalid_expiry_date")

    barrier_quality = "ok"
    if current_price is None or raw_active_stop is None:
        barrier_quality = "invalid"
    elif warnings:
        barrier_quality = "degraded"

    if cfg.liquidity_aware_stops:
        (
            breach_type,
            confirmation_status,
            breach_count,
            liquidity_sweep_recovered,
            exit_fraction,
            breach_recommendation,
            breach_reason,
            stop_triggered,
        ) = classify_liquidity_aware_breach(
            side=lot.side,
            risk_threshold=risk_threshold,
            execution_trigger=execution_trigger,
            latest_open=latest_open,
            latest_high=latest_high,
            latest_low=latest_low,
            latest_close=latest_close,
            current_price=current_price,
            atr_14=atr_14,
            previous_breach_count=previous_breach_count,
            cfg=cfg,
        )
    else:
        stop_triggered = False
        if current_price is not None and raw_active_stop is not None:
            stop_triggered = bool(current_price <= raw_active_stop) if lot.side == "LONG" else bool(current_price >= raw_active_stop)
        breach_type = "CONFIRMED_BREAK" if stop_triggered else "NONE"
        confirmation_status = "CONFIRMED" if stop_triggered else "NONE"
        breach_count = previous_breach_count + 1 if stop_triggered else 0
        liquidity_sweep_recovered = False
        exit_fraction = 1.0 if stop_triggered else 0.0
        breach_recommendation = "EXIT" if stop_triggered else "HOLD"
        breach_reason = f"active_stop_crossed:{active_source}" if stop_triggered else None

    trigger_reason = None
    recommended: Recommendation = "HOLD"
    recommended_exit_fraction = 0.0

    if barrier_quality == "invalid":
        recommended = "DATA_ISSUE"
        confirmation_status = "DATA_ISSUE"
        breach_type = "NO_PRICE"
        trigger_reason = ",".join(warnings[:3]) if warnings else "invalid_barrier"
    elif breach_recommendation != "HOLD":
        recommended = breach_recommendation
        trigger_reason = breach_reason or f"{breach_type.lower()}:{active_source}"
        recommended_exit_fraction = float(exit_fraction)
    elif time_exit_due:
        recommended = cfg.time_exit_action
        trigger_reason = "time_exit_due"
        recommended_exit_fraction = float(cfg.time_exit_exit_fraction)
    elif target_reached and profit_lock_activated:
        recommended = "WATCH"
        trigger_reason = "target_reached_profit_lock_active"
    elif target_reached:
        recommended = "WATCH"
        trigger_reason = "target_reached"
    elif distance_stop is not None and distance_stop <= float(cfg.near_stop_threshold_pct):
        recommended = "WATCH"
        trigger_reason = "near_active_stop"
    elif distance_target is not None and distance_target <= float(cfg.near_target_threshold_pct):
        recommended = "WATCH"
        trigger_reason = "near_fixed_target"

    exit_qty = None
    if lot.quantity_remaining is not None and np.isfinite(float(lot.quantity_remaining)):
        exit_qty = float(lot.quantity_remaining) * float(recommended_exit_fraction)

    exit_value = None
    if exit_qty is not None and current_price is not None:
        exit_value = float(exit_qty) * float(current_price)
    elif lot.value_remaining is not None and np.isfinite(float(lot.value_remaining)):
        exit_value = float(lot.value_remaining) * float(recommended_exit_fraction)

    return LotBarrierResult(
        as_of=pd.Timestamp(as_of_ts).date().isoformat(),
        asset_id=lot.asset_id,
        ticker=lot.ticker,
        trade_id=lot.trade_id,
        side=lot.side,
        source_lot_type=lot.source_lot_type,
        asset_bucket=_infer_asset_bucket(lot),
        entry_date=pd.Timestamp(entry_ts).date().isoformat(),
        entry_price=float(lot.entry_price),
        quantity_remaining=lot.quantity_remaining,
        value_remaining=lot.value_remaining,
        current_price=current_price,
        current_price_date=(str(current_price_date) if current_price_date else None),
        current_price_staleness_days=price_staleness_days,
        latest_open=latest_open,
        latest_high=latest_high,
        latest_low=latest_low,
        latest_close=latest_close,
        latest_bar_date=(str(latest_bar_date) if latest_bar_date else None),
        highest_high_since_entry=highest_high,
        lowest_low_since_entry=lowest_low,
        highest_close_since_entry=highest_close,
        lowest_close_since_entry=lowest_close,
        atr_14=atr_14,
        atr_pct_14=atr_pct_14,
        fixed_stop_loss_price=fixed_stop,
        fixed_target_profit_price=fixed_target,
        chandelier_stop_price=chandelier_stop,
        profit_lock_stop_price=profit_lock_stop,
        raw_active_stop_price=raw_active_stop,
        active_stop_price=active_stop,
        active_stop_source=active_source,
        risk_threshold_price=risk_threshold,
        execution_trigger_price=execution_trigger,
        stop_buffer_amount=buffer_amount,
        distance_to_stop_pct=distance_stop,
        distance_to_target_pct=distance_target,
        peak_return=peak_ret,
        current_return=curr_ret,
        profit_lock_activated=profit_lock_activated,
        stop_status=stop_status,
        target_status=target_status,
        stop_triggered=stop_triggered,
        target_reached=target_reached,
        time_exit_due=time_exit_due,
        breach_type=breach_type,
        confirmation_status=confirmation_status,
        previous_breach_count=previous_breach_count,
        breach_count=int(breach_count),
        liquidity_sweep_recovered=liquidity_sweep_recovered,
        recommended_exit_fraction=float(recommended_exit_fraction),
        recommended_exit_quantity=(None if exit_qty is None else float(exit_qty)),
        recommended_exit_value=(None if exit_value is None else float(exit_value)),
        recommended_action=recommended,
        trigger_reason=trigger_reason,
        barrier_quality=barrier_quality,
        warnings=warnings,
    )


def next_state_from_lot_results(lot_results: list[LotBarrierResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in lot_results:
        key = f"{row.trade_id}|{row.asset_id}|{row.side}"
        breach_date = row.as_of if row.breach_type not in {"NONE", "NO_PRICE"} else None
        recovery_date = row.as_of if row.liquidity_sweep_recovered else None
        out[key] = LotBarrierState(
            state_key=key,
            trade_id=row.trade_id,
            asset_id=row.asset_id,
            side=row.side,
            last_as_of=row.as_of,
            breach_count=int(row.breach_count),
            last_breach_type=(None if row.breach_type == "NONE" else str(row.breach_type)),
            last_breach_date=breach_date,
            last_recovery_date=recovery_date,
            prior_active_stop_price=row.active_stop_price,
            prior_recommendation=row.recommended_action,
        ).to_dict()
    return out


def aggregate_position_results(*, as_of: str, lot_results: list[LotBarrierResult]) -> list[PositionBarrierResult]:
    if not lot_results:
        return []

    priority = {"DATA_ISSUE": 4, "EXIT": 3, "REDUCE": 2, "WATCH": 1, "HOLD": 0}
    quality_rank = {"invalid": 2, "degraded": 1, "ok": 0}
    breach_priority = {
        "NO_PRICE": 6,
        "GAP_THROUGH_STOP": 5,
        "CONFIRMED_BREAK": 4,
        "CLOSE_BREACH": 3,
        "WICK_BREACH_UNCONFIRMED": 2,
        "WICK_BREACH_RECOVERED": 1,
        "NONE": 0,
    }
    confirmation_priority = {
        "DATA_ISSUE": 6,
        "EMERGENCY": 5,
        "CONFIRMED": 4,
        "FIRST_CLOSE_BREACH": 3,
        "WATCH_ONLY": 2,
        "RECOVERED": 1,
        "NONE": 0,
    }
    out: list[PositionBarrierResult] = []

    df = pd.DataFrame([r.to_dict() for r in lot_results])
    for (asset_id, side), g in df.groupby(["asset_id", "side"], sort=True):
        ticker = str(g["ticker"].dropna().iloc[0]) if not g["ticker"].dropna().empty else str(asset_id)
        rec = max((str(x) for x in g["recommended_action"].tolist()), key=lambda x: priority.get(x, -1))
        q = max((str(x) for x in g["barrier_quality"].tolist()), key=lambda x: quality_rank.get(x, -1))
        breach_type = max((str(x) for x in g["breach_type"].tolist()), key=lambda x: breach_priority.get(x, -1))
        confirmation_status = max((str(x) for x in g["confirmation_status"].tolist()), key=lambda x: confirmation_priority.get(x, -1))
        warnings: list[str] = []
        for ws in g["warnings"].tolist():
            if isinstance(ws, list):
                warnings.extend(str(w) for w in ws)
        warnings = sorted(set(warnings))

        dist_stop_values = pd.to_numeric(g["distance_to_stop_pct"], errors="coerce").dropna()
        dist_target_values = pd.to_numeric(g["distance_to_target_pct"], errors="coerce").dropna()
        active_values = pd.to_numeric(g["active_stop_price"], errors="coerce").dropna()
        qty_values = pd.to_numeric(g["quantity_remaining"], errors="coerce").dropna()
        val_values = pd.to_numeric(g["value_remaining"], errors="coerce").dropna()
        exit_frac_values = pd.to_numeric(g["recommended_exit_fraction"], errors="coerce").fillna(0.0)
        exit_qty_values = pd.to_numeric(g.get("recommended_exit_quantity"), errors="coerce").dropna()
        exit_value_values = pd.to_numeric(g.get("recommended_exit_value"), errors="coerce").dropna()
        breach_count_values = pd.to_numeric(g["breach_count"], errors="coerce").fillna(0)

        active_stop = None
        active_source = None
        if not active_values.empty:
            if str(side) == "LONG":
                idx = active_values.idxmax()
            else:
                idx = active_values.idxmin()
            active_stop = float(g.loc[idx, "active_stop_price"])
            active_source = str(g.loc[idx, "active_stop_source"])

        reason = None
        reason_values = [str(x) for x in g["trigger_reason"].dropna().tolist() if str(x).strip()]
        if reason_values:
            reason = reason_values[0]

        current_price_values = pd.to_numeric(g["current_price"], errors="coerce").dropna()
        entry_values = pd.to_numeric(g["entry_price"], errors="coerce").dropna()
        if not entry_values.empty and not qty_values.empty and len(entry_values) == len(qty_values):
            try:
                avg_entry = float(np.average(entry_values.to_numpy(dtype=float), weights=qty_values.to_numpy(dtype=float)))
            except Exception:
                avg_entry = float(entry_values.mean())
        else:
            avg_entry = float(entry_values.mean()) if not entry_values.empty else None
        current_price = float(current_price_values.iloc[-1]) if not current_price_values.empty else None
        current_return_pct = None
        if avg_entry is not None and current_price is not None and avg_entry > 0 and current_price > 0:
            current_return_pct = float(current_price / avg_entry - 1.0) if str(side) == "LONG" else float(avg_entry / current_price - 1.0)

        def _num_col(name: str, how: str = "last"):
            vals = pd.to_numeric(g.get(name), errors="coerce").dropna()
            if vals.empty:
                return None
            if how == "max":
                return float(vals.max())
            if how == "min":
                return float(vals.min())
            if how == "mean":
                return float(vals.mean())
            return float(vals.iloc[-1])

        def _status_col(name: str, ranks: dict[str, int]) -> str:
            vals = [str(x) for x in g.get(name, pd.Series(dtype=object)).dropna().tolist()]
            if not vals:
                return "UNKNOWN"
            return max(vals, key=lambda x: ranks.get(x, -1))

        stop_status = _status_col(
            "stop_status",
            {
                "NO_PRICE_OR_STOP": 6,
                "BELOW_EXECUTION_TRIGGER": 5,
                "ABOVE_EXECUTION_TRIGGER": 5,
                "BELOW_RISK_THRESHOLD": 4,
                "ABOVE_RISK_THRESHOLD": 4,
                "WICK_BREACH_RECOVERED": 3,
                "NEAR_STOP": 2,
                "CLEAR": 1,
            },
        )
        target_status = _status_col("target_status", {"TARGET_REACHED": 3, "NEAR_TARGET": 2, "BELOW_TARGET": 1, "NO_TARGET": 0})
        bucket_values = [str(x) for x in g.get("asset_bucket", pd.Series(dtype=object)).dropna().tolist() if str(x).strip()]
        entry_dates = [str(x) for x in g.get("entry_date", pd.Series(dtype=object)).dropna().tolist() if str(x).strip()]

        out.append(
            PositionBarrierResult(
                as_of=str(as_of),
                asset_id=str(asset_id),
                ticker=ticker,
                side=str(side),  # type: ignore[arg-type]
                asset_bucket=(bucket_values[0] if bucket_values else "unknown"),
                open_lots=int(len(g)),
                quantity_remaining=(float(qty_values.sum()) if not qty_values.empty else None),
                value_remaining=(float(val_values.sum()) if not val_values.empty else None),
                current_price=current_price,
                entry_date_min=(min(entry_dates) if entry_dates else None),
                entry_date_max=(max(entry_dates) if entry_dates else None),
                avg_entry_price=avg_entry,
                current_return_pct=current_return_pct,
                highest_high_since_entry=_num_col("highest_high_since_entry", "max"),
                lowest_low_since_entry=_num_col("lowest_low_since_entry", "min"),
                atr_14=_num_col("atr_14", "last"),
                fixed_stop_loss_price=_num_col("fixed_stop_loss_price", "last"),
                fixed_target_profit_price=_num_col("fixed_target_profit_price", "last"),
                chandelier_stop_price=_num_col("chandelier_stop_price", "last"),
                profit_lock_stop_price=_num_col("profit_lock_stop_price", "last"),
                risk_threshold_price=_num_col("risk_threshold_price", "last"),
                execution_trigger_price=_num_col("execution_trigger_price", "last"),
                stop_buffer_amount=_num_col("stop_buffer_amount", "last"),
                latest_open=_num_col("latest_open", "last"),
                latest_high=_num_col("latest_high", "last"),
                latest_low=_num_col("latest_low", "last"),
                latest_close=_num_col("latest_close", "last"),
                latest_bar_date=(str(g["latest_bar_date"].dropna().iloc[-1]) if "latest_bar_date" in g and not g["latest_bar_date"].dropna().empty else None),
                stop_status=stop_status,
                target_status=target_status,
                active_stop_price=active_stop,
                active_stop_source=active_source,
                nearest_distance_to_stop_pct=(float(dist_stop_values.min()) if not dist_stop_values.empty else None),
                nearest_distance_to_target_pct=(float(dist_target_values.min()) if not dist_target_values.empty else None),
                stop_triggered_lots=int(g["stop_triggered"].astype(bool).sum()),
                target_reached_lots=int(g["target_reached"].astype(bool).sum()),
                time_exit_due_lots=int(g["time_exit_due"].astype(bool).sum()),
                breach_type=breach_type,
                confirmation_status=confirmation_status,
                max_breach_count=int(breach_count_values.max()) if not breach_count_values.empty else 0,
                recommended_exit_fraction=float(exit_frac_values.max()) if not exit_frac_values.empty else 0.0,
                recommended_exit_quantity=(float(exit_qty_values.sum()) if not exit_qty_values.empty else None),
                recommended_exit_value=(float(exit_value_values.sum()) if not exit_value_values.empty else None),
                recommended_action=rec,  # type: ignore[arg-type]
                trigger_reason=reason,
                barrier_quality=q,
                warnings=warnings,
            )
        )

    return out



def _action_priority_for_throttle(row: PositionBarrierResult) -> int:
    reason = str(row.trigger_reason or "")
    breach = str(row.breach_type or "")
    action = str(row.recommended_action or "")
    dist = row.nearest_distance_to_stop_pct
    score = 0
    if action == "DATA_ISSUE":
        return 100
    if action == "EXIT":
        score += 80
    elif action == "REDUCE":
        score += 50
    elif action == "WATCH":
        score += 10
    if "gap_through" in reason or breach == "GAP_THROUGH_STOP":
        score += 30
    if "confirmed" in reason or breach == "CONFIRMED_BREAK":
        score += 20
    if breach == "CLOSE_BREACH":
        score += 10
    if dist is not None and dist < 0:
        score += int(min(20, abs(float(dist)) * 1000.0))
    if reason == "time_exit_due":
        score -= 25
    return int(score)


def apply_liquidation_throttle(
    *,
    position_results: list[PositionBarrierResult],
    cfg: BarrierMonitorConfig,
) -> tuple[list[PositionBarrierResult], dict[str, Any]]:
    """Cap daily liquidation pressure so the monitor remains advisory-safe.

    The throttle intentionally only downgrades actions; it never upgrades them.
    It is independent from regime/portfolio-health awareness, which is deferred
    to Milestone 21 v2.
    """
    if not cfg.apply_liquidation_throttle or not position_results:
        return position_results, {"applied": False, "deferred_actions": 0}

    total_value = 0.0
    for r in position_results:
        if r.value_remaining is not None and np.isfinite(float(r.value_remaining)):
            total_value += abs(float(r.value_remaining))
        elif r.current_price is not None and r.quantity_remaining is not None:
            total_value += abs(float(r.current_price) * float(r.quantity_remaining))

    value_cap = cfg.max_daily_exit_value
    if value_cap is None and total_value > 0:
        value_cap = float(total_value) * float(cfg.max_daily_exit_fraction_of_portfolio)

    remaining_value_cap = float("inf") if value_cap is None else max(0.0, float(value_cap))
    full_exits_used = 0
    reduces_used = 0
    deferred = 0
    consumed_value = 0.0

    ordered = sorted(
        enumerate(position_results),
        key=lambda kv: _action_priority_for_throttle(kv[1]),
        reverse=True,
    )
    updated_by_idx: dict[int, PositionBarrierResult] = {}

    for idx, row in ordered:
        action = row.recommended_action
        exit_value = float(row.recommended_exit_value or 0.0)
        keep = True
        reason_suffix = None

        if action == "EXIT":
            if full_exits_used >= int(cfg.max_full_exits_per_day):
                keep = False
                reason_suffix = "max_full_exits_per_day"
            elif exit_value > remaining_value_cap:
                keep = False
                reason_suffix = "max_daily_exit_value"
            else:
                full_exits_used += 1
                remaining_value_cap -= max(0.0, exit_value)
                consumed_value += max(0.0, exit_value)
        elif action == "REDUCE":
            if reduces_used >= int(cfg.max_reduce_actions_per_day):
                keep = False
                reason_suffix = "max_reduce_actions_per_day"
            elif exit_value > remaining_value_cap:
                keep = False
                reason_suffix = "max_daily_exit_value"
            else:
                reduces_used += 1
                remaining_value_cap -= max(0.0, exit_value)
                consumed_value += max(0.0, exit_value)

        if keep or action not in {"EXIT", "REDUCE"}:
            updated_by_idx[idx] = row
            continue

        deferred += 1
        updated_by_idx[idx] = replace(
            row,
            recommended_action=cfg.throttle_deferred_action,
            recommended_exit_fraction=0.0,
            recommended_exit_quantity=0.0,
            recommended_exit_value=0.0,
            trigger_reason=f"deferred_by_liquidation_throttle:{reason_suffix}; original_action={action}; original_reason={row.trigger_reason}",
            warnings=sorted(set([*row.warnings, f"liquidation_throttle_deferred:{action}"])),
        )

    out = [updated_by_idx[i] for i in range(len(position_results))]
    return out, {
        "applied": True,
        "portfolio_value_basis": float(total_value),
        "max_daily_exit_value": (None if value_cap is None else float(value_cap)),
        "consumed_exit_value": float(consumed_value),
        "remaining_exit_value": (None if value_cap is None else float(max(0.0, remaining_value_cap))),
        "full_exits_used": int(full_exits_used),
        "reduces_used": int(reduces_used),
        "deferred_actions": int(deferred),
        "max_full_exits_per_day": int(cfg.max_full_exits_per_day),
        "max_reduce_actions_per_day": int(cfg.max_reduce_actions_per_day),
    }

def run_barrier_monitor(
    *,
    as_of: str,
    open_lots: list[OpenLotInput],
    ohlcv_history: pd.DataFrame,
    latest_prices: pd.DataFrame | dict[str, dict[str, Any]],
    indicators: Optional[pd.DataFrame | dict[str, dict[str, Any]]] = None,
    previous_state: Optional[dict[str, Any]] = None,
    config: Optional[BarrierMonitorConfig] = None,
) -> BarrierMonitorResult:
    cfg = config or BarrierMonitorConfig()
    as_of_norm = pd.Timestamp(_as_date(as_of)).date().isoformat()

    latest_map = latest_prices if isinstance(latest_prices, dict) else latest_prices_by_asset_id(latest_prices)
    ind_map = None
    if indicators is not None:
        ind_map = indicators if isinstance(indicators, dict) else indicators_by_asset_id(indicators)
    state_map = normalize_barrier_state(previous_state)

    lot_results = [
        evaluate_open_lot_barrier(
            lot=lot,
            as_of=as_of_norm,
            ohlcv_history=ohlcv_history,
            latest_prices=latest_map,
            indicators=ind_map,
            previous_state=state_map,
            config=cfg,
        )
        for lot in open_lots
    ]
    position_results = aggregate_position_results(as_of=as_of_norm, lot_results=lot_results)
    position_results, throttle_summary = apply_liquidation_throttle(position_results=position_results, cfg=cfg)
    next_state = {"lot_state": next_state_from_lot_results(lot_results)}

    def count_rec(items: list[Any], rec: str) -> int:
        return int(sum(1 for x in items if getattr(x, "recommended_action", None) == rec))

    def count_attr(items: list[Any], attr: str, value: str) -> int:
        return int(sum(1 for x in items if getattr(x, attr, None) == value))

    summary = {
        "open_lots": int(len(lot_results)),
        "open_positions": int(len(position_results)),
        "lot_actions": {rec: count_rec(lot_results, rec) for rec in ["EXIT", "REDUCE", "WATCH", "HOLD", "DATA_ISSUE"]},
        "position_actions": {rec: count_rec(position_results, rec) for rec in ["EXIT", "REDUCE", "WATCH", "HOLD", "DATA_ISSUE"]},
        "lot_breach_types": {bt: count_attr(lot_results, "breach_type", bt) for bt in ["GAP_THROUGH_STOP", "CONFIRMED_BREAK", "CLOSE_BREACH", "WICK_BREACH_UNCONFIRMED", "WICK_BREACH_RECOVERED", "NONE", "NO_PRICE"]},
        "degraded_lots": int(sum(1 for r in lot_results if r.barrier_quality == "degraded")),
        "invalid_lots": int(sum(1 for r in lot_results if r.barrier_quality == "invalid")),
        "liquidity_sweep_recovered_lots": int(sum(1 for r in lot_results if r.liquidity_sweep_recovered)),
        "position_stop_status": {ss: count_attr(position_results, "stop_status", ss) for ss in ["BELOW_EXECUTION_TRIGGER", "ABOVE_EXECUTION_TRIGGER", "BELOW_RISK_THRESHOLD", "ABOVE_RISK_THRESHOLD", "WICK_BREACH_RECOVERED", "NEAR_STOP", "CLEAR", "NO_PRICE_OR_STOP"]},
        "position_target_status": {ts: count_attr(position_results, "target_status", ts) for ts in ["TARGET_REACHED", "NEAR_TARGET", "BELOW_TARGET", "NO_TARGET"]},
        "liquidation_throttle": throttle_summary,
        "config_version": cfg.version,
    }

    return BarrierMonitorResult(
        as_of=as_of_norm,
        generated_at_utc=pd.Timestamp.utcnow().isoformat(),
        config=asdict(cfg),
        lot_results=lot_results,
        position_results=position_results,
        next_state=next_state,
        summary=summary,
    )
