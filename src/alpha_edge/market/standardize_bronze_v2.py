"""Stage 3: from immutable Yahoo Bronze batches to inspectable Silver and Gold.

No network access or production publication. Converts *explicitly configured*
FX symbols, audits expected trading sessions, and prevents returns bridging
missing sessions. Original source columns remain available in Silver/Gold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_edge.market.data_lake_v2 import (
    DataContractError, QualityPolicy, calculate_returns, standardize_bars,
    validate_raw_bars,
)

# Explicit quote direction (base units -> quote units); no ticker inference.
PILOT_FX_PAIRS = {
    "EUR": ("EURUSD=X", "direct"),
    "BRL": ("USDBRL=X", "inverse"),
    "MXN": ("USDMXN=X", "inverse"),
}
# pandas_market_calendars identifiers. Fail closed if an expected calendar is
# unavailable; exchange calendars are not equivalent to business weekdays.
PILOT_CALENDARS = {
    "GLD": "NYSE",
    "SMH": "NYSE",
    "SPY": "NYSE",
    "B3SA3.SA": "BMF",
    "KESKOB.HE": "XHEL",
    "ALSEA.MX": "BMV",
}


def _dates(series: pd.Series) -> pd.Series:
    # Daily bar session labels are *dates*, not instants to be UTC-shifted.
    return pd.to_datetime(series.astype(str).str[:10], format="%Y-%m-%d", errors="coerce")


def normalize_bronze_fx_quotes(
    quotes: pd.DataFrame,
    *,
    pairs: dict[str, tuple[str, str]] | None = None,
    max_abs_daily_change: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (accepted Silver FX rates, row-level FX diagnostics).

    Outlier and invalid rates remain in diagnostics but cannot enter Silver.
    Do not interpolate or fill missing FX observations.
    """
    pairs = pairs if pairs is not None else PILOT_FX_PAIRS
    needed = {"provider_symbol", "session_date", "quote_close"}
    if missing := needed - set(quotes.columns):
        raise DataContractError(f"Bronze FX missing: {sorted(missing)}")
    if not 0 < max_abs_daily_change < 1:
        raise ValueError("FX daily-change review threshold must be in (0,1)")
    q = quotes.copy()
    q["fx_date"] = _dates(q["session_date"])
    q["quote_close"] = pd.to_numeric(q["quote_close"], errors="coerce")
    results = []
    for currency, (symbol, direction) in pairs.items():
        if direction not in {"direct", "inverse"}:
            raise DataContractError(f"{currency}: unsupported quote direction")
        subset = q.loc[q["provider_symbol"].eq(symbol)].copy()
        if subset.empty:
            raise DataContractError(f"Missing FX symbol {symbol} for {currency}")
        subset = subset.sort_values("fx_date", kind="stable")
        subset["currency_original"] = currency
        subset["fx_quote_direction"] = direction
        subset["fx_quote_symbol"] = symbol
        subset["fx_to_usd"] = (
            subset["quote_close"] if direction == "direct"
            else 1.0 / subset["quote_close"]
        )
        dup = subset.duplicated("fx_date", keep=False)
        invalid = (
            subset["fx_date"].isna()
            | ~np.isfinite(subset["quote_close"])
            | subset["quote_close"].le(0)
        )
        # Check only adjacent *available* dates; missing intervals are
        # separately recorded as gaps, not automatically interpreted as jumps.
        prev_date = subset["fx_date"].shift()
        days = (subset["fx_date"] - prev_date).dt.days
        movement = subset["fx_to_usd"].pct_change(fill_method=None).abs()
        high_move = movement.gt(max_abs_daily_change) & days.le(7)
        subset["fx_validation_status"] = np.select(
            [dup, invalid, high_move],
            ["duplicate_quote_date", "invalid_quote", "large_fx_move"],
            default="accepted",
        )
        subset["fx_quote_change_abs"] = movement
        subset["fx_gap_calendar_days"] = days
        results.append(subset)
    evidence = pd.concat(results, ignore_index=True)
    accepted = evidence.loc[
        evidence["fx_validation_status"].eq("accepted"),
        ["currency_original", "fx_date", "fx_to_usd",
         "fx_quote_symbol", "fx_quote_direction", "quote_close"],
    ].copy()
    return accepted.reset_index(drop=True), evidence


def audit_asset_sessions(
    bars: pd.DataFrame,
    *,
    calendars: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (row-level session flags, missing/unexpected session report).

    Only sessions between each asset's first and last observed date are tested.
    Completeness at the request's endpoints is a separate ingestion contract.
    """
    calendars = calendars if calendars is not None else PILOT_CALENDARS
    required = {"asset_id", "provider_symbol", "session_date"}
    if missing := required - set(bars.columns):
        raise DataContractError(f"Session audit missing: {sorted(missing)}")
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:
        raise DataContractError("pandas-market-calendars is required") from exc
    evidence = []
    report = []
    for (asset_id, symbol), group in bars.groupby(["asset_id", "provider_symbol"], sort=True):
        if symbol not in calendars:
            raise DataContractError(f"{symbol}: no approved exchange calendar")
        calendar_id = calendars[symbol]
        try:
            calendar = mcal.get_calendar(calendar_id)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise DataContractError(f"{symbol}: invalid calendar {calendar_id}") from exc
        observed = _dates(group["session_date"]).dropna()
        if observed.empty:
            raise DataContractError(f"{symbol}: no valid session dates")
        expected = pd.DatetimeIndex(
            calendar.valid_days(start_date=observed.min(), end_date=observed.max())
        ).tz_localize(None).normalize()
        observed_set = set(observed.dt.normalize())
        expected_set = set(expected)
        missing_days = sorted(expected_set - observed_set)
        unexpected_days = sorted(observed_set - expected_set)
        for day in missing_days:
            report.append({
                "asset_id": asset_id, "provider_symbol": symbol,
                "session_date": day, "issue": "missing_expected_session",
                "calendar_id": calendar_id,
            })
        for day in unexpected_days:
            report.append({
                "asset_id": asset_id, "provider_symbol": symbol,
                "session_date": day, "issue": "unexpected_session",
                "calendar_id": calendar_id,
            })
        for _, row in group.iterrows():
            current = pd.Timestamp(str(row["session_date"])[:10])
            prev_expected = expected[expected < current]
            prior = prev_expected[-1] if len(prev_expected) else pd.NaT
            previous_missing = pd.notna(prior) and prior not in observed_set
            evidence.append({
                "asset_id": asset_id, "provider_symbol": symbol,
                "session_date": current, "calendar_id": calendar_id,
                "is_expected_session": current in expected_set,
                "previous_expected_session_missing": bool(previous_missing),
            })
    return pd.DataFrame(evidence), pd.DataFrame(
        report, columns=["asset_id", "provider_symbol", "session_date", "issue", "calendar_id"]
    )


def build_silver_gold_from_bronze(
    bars: pd.DataFrame, quotes: pd.DataFrame,
    *, policy: QualityPolicy = QualityPolicy(),
    calendars: dict[str, str] | None = None,
    pairs: dict[str, tuple[str, str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Produce all stage outputs with explicit quality evidence and row counts."""
    bronze = validate_raw_bars(bars)
    session_flags, session_issues = audit_asset_sessions(bronze, calendars=calendars)
    rates, fx_evidence = normalize_bronze_fx_quotes(quotes, pairs=pairs)

    # Require a unique source bar key before reconciling session flags.
    # Duplicate raw observations remain in Bronze and never silently collapse.
    if bronze.duplicated(["asset_id", "session_date"]).any():
        raise DataContractError("Source contains duplicate asset/session observations")
    if session_flags.duplicated(["asset_id", "session_date"]).any():
        raise DataContractError("Session audit returned duplicate session keys")
    flagged = bronze.merge(
        session_flags.drop(columns=["provider_symbol"]),
        on=["asset_id", "session_date"], how="left", validate="one_to_one",
    )
    missing_flag = flagged["previous_expected_session_missing"].fillna(False)
    unexpected_flag = ~flagged["is_expected_session"].fillna(False)
    for mask, reason in (
        (missing_flag, "prior_expected_session_missing"),
        (unexpected_flag, "unexpected_exchange_session"),
    ):
        flagged.loc[mask, "quality_reason"] = (
            flagged.loc[mask, "quality_reason"].where(
                flagged.loc[mask, "quality_reason"].eq(""),
                flagged.loc[mask, "quality_reason"] + ";",
            ) + reason
        )
    flagged.loc[missing_flag | unexpected_flag, "quality_status"] = "quarantined"
    silver = standardize_bars(flagged, rates, policy=policy)
    # Preserve FX lineage through the conversion, including original quote,
    # quote direction and source symbol. Never substitute an unaccepted quote.
    fx_lineage = rates[
        ["currency_original", "fx_date", "fx_quote_symbol", "fx_quote_direction", "quote_close"]
    ].copy()
    silver = silver.merge(
        fx_lineage, on=["currency_original", "fx_date"],
        how="left", validate="many_to_one",
    )
    if len(silver) != len(bronze):
        raise DataContractError("Silver row count does not reconcile with Bronze")

    # The legacy Gold calculator correctly does not bridge a quarantined row.
    # Here we additionally suppress returns after a missing *expected* session.
    gold = calculate_returns(silver, policy=policy)
    return {
        "bronze_validated": bronze,
        "fx_normalized": rates,
        "fx_evidence": fx_evidence,
        "session_flags": session_flags,
        "session_issues": session_issues,
        "silver": silver,
        "gold": gold,
    }
