"""Market data lake v2: pure, testable transformations.

No S3 writes or Yahoo calls live here. Daily bars have a session date; they
do not automatically have a trustworthy UTC close timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """Raised when a source or transformation violates a required contract."""


@dataclass(frozen=True)
class QualityPolicy:
    max_fx_age_days: int = 4
    max_abs_simple_return: float = 0.50


def _require(df: pd.DataFrame, fields: set[str], kind: str) -> None:
    missing = sorted(fields.difference(df.columns))
    if missing:
        raise DataContractError(f"{kind} missing columns: {missing}")


def _session_dates(values: pd.Series) -> pd.Series:
    """Interpret daily-bar labels as exchange session dates, not UTC instants.

    A timezone-aware timestamp supplied as a daily label is interpreted in
    its original displayed timezone; do not first shift it to UTC.
    """

    def parse(value):
        if pd.isna(value):
            return pd.NaT
        try:
            ts = pd.Timestamp(value)
            return ts.date()
        except (ValueError, TypeError):
            return pd.NaT

    return pd.to_datetime(values.map(parse), errors="coerce")


def validate_raw_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Retain all observations with explicit flags; never silently drop rows.

    Expected source columns: asset_id, provider_symbol, currency_original,
    session_date, open, high, low, close, adj_close, volume.
    """
    _require(
        bars,
        {
            "asset_id",
            "provider_symbol",
            "currency_original",
            "session_date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        },
        "raw bars",
    )
    out = bars.copy()
    out["session_date"] = _session_dates(out["session_date"])
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    key = ["asset_id", "session_date"]
    duplicate = out.duplicated(key, keep=False)
    invalid_date = out["session_date"].isna()
    invalid_price = (
        out["close"].isna()
        | ~np.isfinite(out["close"])
        | (out["close"] <= 0)
        | out["adj_close"].isna()
        | ~np.isfinite(out["adj_close"])
        | (out["adj_close"] <= 0)
    )
    invalid_ohlc = out["high"].notna() & out["low"].notna() & (out["high"] < out["low"])
    invalid_volume = out["volume"].notna() & (out["volume"] < 0)
    invalid_identity = (
        out["asset_id"].isna()
        | out["provider_symbol"].isna()
        | out["currency_original"].isna()
        | out["currency_original"].astype(str).str.strip().eq("")
    )
    out["duplicate_key"] = duplicate
    out["quality_reason"] = ""
    conditions = [
        (invalid_identity, "identity_missing"),
        (invalid_date, "invalid_session_date"),
        (invalid_price, "invalid_price"),
        (invalid_ohlc, "invalid_ohlc"),
        (invalid_volume, "negative_volume"),
        (duplicate, "duplicate_session"),
    ]
    for mask, reason in conditions:
        out.loc[mask, "quality_reason"] = (
            out.loc[mask, "quality_reason"].where(
                out.loc[mask, "quality_reason"].eq(""),
                out.loc[mask, "quality_reason"] + ";",
            )
            + reason
        )
    out["quality_status"] = np.where(out["quality_reason"].eq(""), "accepted", "quarantined")
    return out


def normalize_fx(fx: pd.DataFrame) -> pd.DataFrame:
    """FX convention is USD per 1 original currency unit."""
    _require(fx, {"currency_original", "fx_date", "fx_to_usd"}, "FX")
    out = fx.copy()
    out["fx_date"] = _session_dates(out["fx_date"])
    out["fx_to_usd"] = pd.to_numeric(out["fx_to_usd"], errors="coerce").astype(float)
    if out.duplicated(["currency_original", "fx_date"]).any():
        raise DataContractError("FX contains duplicate currency/date keys")
    if out["fx_date"].isna().any() or ~np.isfinite(out["fx_to_usd"]).all() or (out["fx_to_usd"] <= 0).any():
        raise DataContractError("FX contains invalid date or rate")
    return out.sort_values(["currency_original", "fx_date"]).reset_index(drop=True)


def standardize_bars(
    validated_bars: pd.DataFrame,
    fx: pd.DataFrame,
    *,
    policy: QualityPolicy = QualityPolicy(),
) -> pd.DataFrame:
    """Join FX at-or-before session date, with a bounded stale-rate policy.

    Raw quarantines remain visible in the standardized layer. All row counts
    are preserved, including duplicates, to make reconciliation possible.
    """
    _require(
        validated_bars,
        {"asset_id", "currency_original", "session_date", "close", "adj_close", "quality_status", "quality_reason"},
        "validated bars",
    )
    if policy.max_fx_age_days < 0:
        raise DataContractError("max_fx_age_days must be >= 0")
    source = validated_bars.copy().reset_index(drop=True)
    source["_row_id"] = np.arange(len(source))
    source["currency_original"] = source["currency_original"].astype(str).str.upper()
    source["session_date"] = _session_dates(source["session_date"])
    rate_table = normalize_fx(fx)
    segments = []
    for currency, group in source.groupby("currency_original", dropna=False):
        part = group.copy()
        if currency == "USD":
            part["fx_date"] = part["session_date"]
            part["fx_to_usd"] = 1.0
        else:
            rates = rate_table.loc[
                rate_table["currency_original"].eq(currency),
                ["fx_date", "fx_to_usd"],
            ].sort_values("fx_date")
            good = part.loc[part["session_date"].notna()].sort_values("session_date")
            bad = part.loc[part["session_date"].isna()].copy()
            if rates.empty:
                good["fx_date"] = pd.NaT
                good["fx_to_usd"] = np.nan
            elif not good.empty:
                good = pd.merge_asof(
                    good,
                    rates,
                    left_on="session_date",
                    right_on="fx_date",
                    direction="backward",
                    allow_exact_matches=True,
                )
            bad["fx_date"] = pd.NaT
            bad["fx_to_usd"] = np.nan
            part = pd.concat([good, bad], ignore_index=True)
        segments.append(part)
    out = (
        pd.concat(segments, ignore_index=True).sort_values("_row_id")
        if segments
        else source.assign(fx_date=pd.NaT, fx_to_usd=np.nan)
    )
    out["fx_age_days"] = (out["session_date"] - out["fx_date"]).dt.days
    missing_fx = out["fx_to_usd"].isna()
    stale_fx = out["fx_age_days"].gt(policy.max_fx_age_days)
    for mask, reason in ((missing_fx, "fx_missing"), (stale_fx, "fx_stale")):
        out.loc[mask, "quality_reason"] = (
            out.loc[mask, "quality_reason"].where(
                out.loc[mask, "quality_reason"].eq(""),
                out.loc[mask, "quality_reason"] + ";",
            )
            + reason
        )
    out["quality_status"] = np.where(out["quality_reason"].eq(""), "accepted", "quarantined")
    out["close_raw_usd"] = out["close"] * out["fx_to_usd"]
    out["close_adjusted_usd"] = out["adj_close"] * out["fx_to_usd"]
    out.loc[out["quality_status"].ne("accepted"), ["close_raw_usd", "close_adjusted_usd"]] = np.nan
    return out.drop(columns="_row_id").reset_index(drop=True)


def calculate_returns(
    standardized: pd.DataFrame,
    *,
    policy: QualityPolicy = QualityPolicy(),
) -> pd.DataFrame:
    """Calculate arithmetic and log returns without bridging rejected rows.

    Gaps in exchange sessions require separate calendar checks; this function
    makes no assumption that all consecutive accepted rows are contiguous.
    """
    _require(
        standardized,
        {"asset_id", "session_date", "close_adjusted_usd", "quality_status"},
        "standardized bars",
    )
    out = standardized.copy().sort_values(["asset_id", "session_date"], kind="stable")
    if out.duplicated(["asset_id", "session_date"]).any():
        raise DataContractError("Cannot calculate returns over duplicate asset/date keys")
    price = pd.to_numeric(out["close_adjusted_usd"], errors="coerce")
    good = out["quality_status"].eq("accepted") & price.gt(0) & np.isfinite(price)
    safe_price = price.where(good)
    prev = safe_price.groupby(out["asset_id"]).shift(1)
    out["ret_simple_close_adjusted_usd"] = (safe_price / prev - 1).where(prev.gt(0))
    out["ret_log_close_adjusted_usd"] = np.log(safe_price / prev).where(prev.gt(0))
    out["return_quality_status"] = np.where(out["ret_simple_close_adjusted_usd"].isna(), "unavailable", "accepted")
    out.loc[
        out["ret_simple_close_adjusted_usd"].abs().gt(policy.max_abs_simple_return),
        "return_quality_status",
    ] = "review"
    return out.reset_index(drop=True)


def quality_summary(df: pd.DataFrame) -> dict:
    """Simple, serializable reconciliation counts for execution manifests."""
    statuses = df["quality_status"].value_counts().to_dict()
    return {
        "input_rows": int(len(df)),
        "accepted_rows": int(statuses.get("accepted", 0)),
        "quarantined_rows": int(statuses.get("quarantined", 0)),
        "distinct_assets": int(df["asset_id"].nunique()),
        "duplicate_asset_sessions": int(df.duplicated(["asset_id", "session_date"], keep=False).sum()),
    }
