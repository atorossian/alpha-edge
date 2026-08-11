from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.schemas import RuntimeConfig


RiskModel = Literal[
    "fixed_by_asset_class",
    "atr_based",
    "volatility_based",
    "hybrid",
]

IndicatorMode = Literal[
    "auto",
    "latest",
    "point_in_time",
]


@dataclass(frozen=True)
class IndicatorSnapshot:
    asset_id: str
    ticker: Optional[str]
    indicator_date: str

    close: Optional[float]
    atr_14: Optional[float]
    atr_pct_14: Optional[float]
    daily_vol_20: Optional[float]
    daily_vol_60: Optional[float]
    annualized_vol_20: Optional[float]
    annualized_vol_60: Optional[float]

    source_key: str
    source_mode: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "ticker": self.ticker,
            "indicator_date": self.indicator_date,
            "close": self.close,
            "atr_14": self.atr_14,
            "atr_pct_14": self.atr_pct_14,
            "daily_vol_20": self.daily_vol_20,
            "daily_vol_60": self.daily_vol_60,
            "annualized_vol_20": self.annualized_vol_20,
            "annualized_vol_60": self.annualized_vol_60,
            "source_key": self.source_key,
            "source_mode": self.source_mode,
        }


@dataclass(frozen=True)
class MarketRiskModelConfig:
    atr_stop_multiplier: float = 2.0
    atr_target_multiplier: float = 4.0

    volatility_stop_multiplier: float = 1.0
    volatility_target_multiplier: float = 1.8

    reward_multiple: float = 2.0
    min_stop_pct: float = 0.02
    max_stop_pct: float = 0.25
    max_target_pct: float = 0.95

    max_indicator_staleness_days: int = 10


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result):
        return None

    return result


def _date(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(parsed):
        raise ValueError(f"Invalid date value: {value!r}")

    return pd.Timestamp(parsed).tz_convert(None).normalize()


def _market_store(cfg: RuntimeConfig) -> MarketStore:
    return MarketStore(
        bucket=cfg.bucket,
        region=cfg.region,
        base_prefix=cfg.market_root,
    )


def latest_indicators_key(store: MarketStore) -> str:
    return f"{store.snapshots_prefix}/latest_indicators.parquet"


def indicators_prefix(store: MarketStore) -> str:
    return f"{store.base_prefix}/indicators/{store.version}"


def _read_parquet(store: MarketStore, key: str) -> pd.DataFrame:
    raw = store._get_bytes(key)
    return pd.read_parquet(io.BytesIO(raw))


def _select_asset_row(
    df: pd.DataFrame,
    *,
    asset_id: str,
    as_of: str,
    source_key: str,
    source_mode: str,
    max_staleness_days: int,
) -> IndicatorSnapshot:
    if df is None or df.empty:
        raise RuntimeError(f"Indicator dataset is empty: s3://{source_key}")

    required = {"asset_id", "date"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            f"Indicator dataset missing required columns {missing}: {source_key}"
        )

    work = df.copy()
    work["asset_id"] = work["asset_id"].astype(str).str.strip()
    work["date"] = pd.to_datetime(
        work["date"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()

    requested_asset_id = str(asset_id).strip()
    as_of_ts = _date(as_of)

    work = work[
        (work["asset_id"] == requested_asset_id)
        & work["date"].notna()
        & (work["date"] <= as_of_ts)
    ].copy()

    if work.empty:
        raise RuntimeError(
            f"No point-in-time indicators found for asset_id={requested_asset_id!r} "
            f"with indicator_date <= {as_of_ts.date()}."
        )

    work = work.sort_values("date", kind="stable")
    row = work.iloc[-1]
    indicator_ts = pd.Timestamp(row["date"]).normalize()

    staleness_days = int((as_of_ts - indicator_ts).days)

    if staleness_days < 0:
        raise RuntimeError(
            f"Indicator lookahead detected: indicator_date={indicator_ts.date()} "
            f"is after trade_as_of={as_of_ts.date()}."
        )

    if staleness_days > int(max_staleness_days):
        raise RuntimeError(
            f"Indicator snapshot is stale for asset_id={requested_asset_id}: "
            f"indicator_date={indicator_ts.date()}, trade_as_of={as_of_ts.date()}, "
            f"staleness_days={staleness_days}, "
            f"maximum={max_staleness_days}."
        )

    ticker = row.get("ticker")
    ticker_norm = None

    if ticker is not None and not pd.isna(ticker):
        ticker_norm = str(ticker).upper().strip() or None

    return IndicatorSnapshot(
        asset_id=requested_asset_id,
        ticker=ticker_norm,
        indicator_date=indicator_ts.date().isoformat(),
        close=_float_or_none(row.get("close")),
        atr_14=_float_or_none(row.get("atr_14")),
        atr_pct_14=_float_or_none(row.get("atr_pct_14")),
        daily_vol_20=_float_or_none(row.get("daily_vol_20")),
        daily_vol_60=_float_or_none(row.get("daily_vol_60")),
        annualized_vol_20=_float_or_none(row.get("annualized_vol_20")),
        annualized_vol_60=_float_or_none(row.get("annualized_vol_60")),
        source_key=source_key,
        source_mode=source_mode,
    )


def load_latest_indicator_snapshot(
    *,
    cfg: RuntimeConfig,
    asset_id: str,
    as_of: str,
    snapshot_key: Optional[str] = None,
    max_staleness_days: int = 10,
) -> IndicatorSnapshot:
    store = _market_store(cfg)
    key = snapshot_key or latest_indicators_key(store)

    if not store._key_exists(key):
        raise RuntimeError(
            f"Latest indicators snapshot does not exist: "
            f"s3://{store.bucket}/{key}"
        )

    df = _read_parquet(store, key)

    return _select_asset_row(
        df,
        asset_id=asset_id,
        as_of=as_of,
        source_key=key,
        source_mode="latest",
        max_staleness_days=max_staleness_days,
    )


def load_point_in_time_indicator_snapshot(
    *,
    cfg: RuntimeConfig,
    asset_id: str,
    as_of: str,
    root_prefix: Optional[str] = None,
    max_staleness_days: int = 10,
) -> IndicatorSnapshot:
    store = _market_store(cfg)
    root = str(root_prefix or indicators_prefix(store)).strip("/")

    as_of_ts = _date(as_of)

    # The relevant row can be in the current year or the preceding year,
    # particularly for January trades and assets with non-daily observations.
    candidate_years = [
        int(as_of_ts.year),
        int(as_of_ts.year) - 1,
    ]

    frames: list[pd.DataFrame] = []
    used_keys: list[str] = []

    for year in candidate_years:
        manifest = store.read_asset_year_manifest(
            table="indicators",
            asset_id=str(asset_id),
            year=year,
        )

        parts = [
            str(key)
            for key in (manifest.get("parts") or [])
            if isinstance(key, str) and key.endswith(".parquet")
        ]

        if not parts:
            prefix = (
                f"{root}/asset_id={str(asset_id).strip()}/"
                f"year={year}/"
            )
            parts = [
                key
                for key in store._list_keys(prefix)
                if key.endswith(".parquet")
            ]

        for key in parts:
            try:
                frame = _read_parquet(store, key)

                if frame is not None and not frame.empty:
                    frames.append(frame)
                    used_keys.append(key)
            except Exception:
                continue

    if not frames:
        raise RuntimeError(
            f"No historical indicator partitions found for "
            f"asset_id={asset_id!r}, as_of={as_of}."
        )

    df = pd.concat(frames, ignore_index=True)

    return _select_asset_row(
        df,
        asset_id=asset_id,
        as_of=as_of,
        source_key=",".join(used_keys),
        source_mode="point_in_time",
        max_staleness_days=max_staleness_days,
    )


def load_indicator_snapshot(
    *,
    cfg: RuntimeConfig,
    asset_id: str,
    as_of: str,
    mode: IndicatorMode = "auto",
    latest_snapshot_key: Optional[str] = None,
    historical_root_prefix: Optional[str] = None,
    max_staleness_days: int = 10,
) -> IndicatorSnapshot:
    mode_norm = str(mode).strip().lower()

    if mode_norm not in {"auto", "latest", "point_in_time"}:
        raise ValueError(
            "indicator_mode must be auto, latest, or point_in_time."
        )

    if mode_norm == "latest":
        return load_latest_indicator_snapshot(
            cfg=cfg,
            asset_id=asset_id,
            as_of=as_of,
            snapshot_key=latest_snapshot_key,
            max_staleness_days=max_staleness_days,
        )

    if mode_norm == "point_in_time":
        return load_point_in_time_indicator_snapshot(
            cfg=cfg,
            asset_id=asset_id,
            as_of=as_of,
            root_prefix=historical_root_prefix,
            max_staleness_days=max_staleness_days,
        )

    # auto:
    # Try the small latest snapshot first. It will reject future or stale rows.
    try:
        return load_latest_indicator_snapshot(
            cfg=cfg,
            asset_id=asset_id,
            as_of=as_of,
            snapshot_key=latest_snapshot_key,
            max_staleness_days=max_staleness_days,
        )
    except Exception:
        return load_point_in_time_indicator_snapshot(
            cfg=cfg,
            asset_id=asset_id,
            as_of=as_of,
            root_prefix=historical_root_prefix,
            max_staleness_days=max_staleness_days,
        )


def _require_positive(name: str, value: Optional[float]) -> float:
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} is required and must be positive for this risk model."
        )

    return float(value)


def calculate_indicator_backed_risk_percentages(
    *,
    risk_model: RiskModel,
    indicators: IndicatorSnapshot,
    max_holding_days: int,
    config: MarketRiskModelConfig,
) -> dict[str, Any]:
    model = str(risk_model).strip().lower()

    if model not in {"atr_based", "volatility_based", "hybrid"}:
        raise ValueError(
            "Indicator-backed risk model must be atr_based, "
            "volatility_based, or hybrid."
        )

    if max_holding_days <= 0:
        raise ValueError("max_holding_days must be > 0.")

    atr_stop_pct: Optional[float] = None
    atr_target_pct: Optional[float] = None
    volatility_stop_pct: Optional[float] = None
    volatility_target_pct: Optional[float] = None
    horizon_volatility: Optional[float] = None

    if model in {"atr_based", "hybrid"}:
        atr_pct = _require_positive(
            "atr_pct_14",
            indicators.atr_pct_14,
        )

        atr_stop_pct = (
            float(config.atr_stop_multiplier) * atr_pct
        )
        atr_target_pct = (
            float(config.atr_target_multiplier) * atr_pct
        )

    if model in {"volatility_based", "hybrid"}:
        daily_volatility = _require_positive(
            "daily_vol_20",
            indicators.daily_vol_20,
        )

        horizon_volatility = (
            daily_volatility * math.sqrt(float(max_holding_days))
        )

        volatility_stop_pct = (
            float(config.volatility_stop_multiplier)
            * horizon_volatility
        )
        volatility_target_pct = (
            float(config.volatility_target_multiplier)
            * horizon_volatility
        )

    if model == "atr_based":
        raw_stop_pct = _require_positive(
            "ATR stop percentage",
            atr_stop_pct,
        )
        raw_target_pct = _require_positive(
            "ATR target percentage",
            atr_target_pct,
        )

    elif model == "volatility_based":
        raw_stop_pct = _require_positive(
            "volatility stop percentage",
            volatility_stop_pct,
        )
        raw_target_pct = _require_positive(
            "volatility target percentage",
            volatility_target_pct,
        )

    else:
        raw_stop_pct = max(
            float(config.min_stop_pct),
            _require_positive("ATR stop percentage", atr_stop_pct),
            _require_positive(
                "volatility stop percentage",
                volatility_stop_pct,
            ),
        )

        raw_target_pct = (
            float(config.reward_multiple) * raw_stop_pct
        )

    final_stop_pct = min(
        max(raw_stop_pct, float(config.min_stop_pct)),
        float(config.max_stop_pct),
    )

    # For a short trade, a target >= 100% would imply a non-positive price.
    final_target_pct = min(
        max(raw_target_pct, 0.0),
        float(config.max_target_pct),
    )

    if final_target_pct <= 0:
        raise ValueError(
            "Calculated target_profit_pct must be positive."
        )

    return {
        "stop_loss_pct": float(final_stop_pct),
        "target_profit_pct": float(final_target_pct),
        "model_inputs": {
            "atr_stop_pct_raw": atr_stop_pct,
            "atr_target_pct_raw": atr_target_pct,
            "volatility_stop_pct_raw": volatility_stop_pct,
            "volatility_target_pct_raw": volatility_target_pct,
            "horizon_volatility": horizon_volatility,
            "atr_stop_multiplier": config.atr_stop_multiplier,
            "atr_target_multiplier": config.atr_target_multiplier,
            "volatility_stop_multiplier": (
                config.volatility_stop_multiplier
            ),
            "volatility_target_multiplier": (
                config.volatility_target_multiplier
            ),
            "reward_multiple": config.reward_multiple,
            "min_stop_pct": config.min_stop_pct,
            "max_stop_pct": config.max_stop_pct,
            "max_target_pct": config.max_target_pct,
            "max_holding_days": max_holding_days,
        },
    }