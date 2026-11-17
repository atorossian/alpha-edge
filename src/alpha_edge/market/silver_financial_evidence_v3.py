"""Financial Silver candidate evidence: corporate actions and FX quotations.

Pure functions only. Reuse the established v2 evidence/math primitives.
No S3/legacy side effects; no double split adjustment, synthetic action,
implicit FX inversion, date-only look-ahead conversion, or Gold release.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_edge.market.audit_corporate_actions_v2 import inspect_adjustment_changes
from alpha_edge.market.standardize_bronze_v2 import normalize_bronze_fx_quotes


class SilverFinancialError(ValueError):
    """Financial evidence cannot be safely reconciled with its Bronze source."""


def extract_source_actions(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Extract reported split/dividend events without modifying provider prices."""
    required = {
        "asset_id", "provider_symbol", "session_date",
        "dividends", "stock_splits", "currency_original",
    }
    missing = required - set(bars)
    if missing or bars.empty:
        raise SilverFinancialError(f"Missing Bronze action source columns: {sorted(missing)}")
    dates = pd.to_datetime(bars["session_date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise SilverFinancialError("Cannot interpret source effective-session dates")
    if bars.duplicated(["asset_id", "session_date"]).any():
        raise SilverFinancialError("Duplicate source sessions cannot yield unique corporate actions")
    splits = pd.to_numeric(bars["stock_splits"], errors="coerce")
    dividends = pd.to_numeric(bars["dividends"], errors="coerce")
    if (~np.isfinite(splits)).any() or splits.lt(0).any():
        raise SilverFinancialError("Invalid source split factor")
    if (~np.isfinite(dividends)).any() or dividends.lt(0).any():
        raise SilverFinancialError("Invalid/negative source dividend; requires independent review")
    events = []
    for i in range(len(bars)):
        row = bars.iloc[i]
        day = dates.iloc[i].date().isoformat()
        for kind, amount in (("SPLIT", splits.iloc[i]), ("DIVIDEND", dividends.iloc[i])):
            if amount == 0:
                continue
            if kind == "SPLIT" and amount <= 0:
                raise SilverFinancialError("Split ratio must be strictly positive")
            events.append(
                {
                    "asset_id": str(row["asset_id"]),
                    "provider_symbol": str(row["provider_symbol"]),
                    "effective_date": day,
                    "action_type": kind,
                    "source_value": float(amount),
                    "source_currency": str(row["currency_original"]) if kind == "DIVIDEND" else None,
                    "source_row_number": i,
                    "origin": "yahoo_bronze_daily_bar",
                    "evidence_status": "provider_reported_not_independently_confirmed",
                }
            )
    columns = [
        "asset_id", "provider_symbol", "effective_date", "action_type",
        "source_value", "source_currency", "source_row_number",
        "origin", "evidence_status",
    ]
    table = pd.DataFrame(events, columns=columns)
    return table, {
        "rows": len(table),
        "reported_splits": int(table["action_type"].eq("SPLIT").sum()),
        "reported_dividends": int(table["action_type"].eq("DIVIDEND").sum()),
        "adjustment_applied": False,
        "independently_verified": False,
        "production_eligible": False,
    }


def source_adjustment_flags(
    bars: pd.DataFrame, actions: pd.DataFrame, *, threshold: float = 0.005
) -> pd.DataFrame:
    """Reuse v2 ratio-change audit as a review signal, never a split repair."""
    columns = ["asset_id", "effective_date", "action_type", "split_factor"]
    if actions.empty:
        reference = pd.DataFrame(columns=columns)
    else:
        reference = actions.rename(columns={"source_value": "split_factor"})[columns]
    return inspect_adjustment_changes(bars, reference, threshold=threshold)


def normalize_fx_candidate(
    quotes: pd.DataFrame, *, pairs: dict[str, tuple[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Use explicit FX quote direction from pinned config; no price joining.

    `pairs` maps local currency to (source Yahoo symbol, direct/inverse).
    Example JPY=X is JPY per USD, hence JPY -> USD requires inverse.
    """
    if not pairs:
        raise SilverFinancialError("FX normalization requires an explicit currency-to-symbol mapping")
    if any(not k or v[1] not in {"direct", "inverse"} for k, v in pairs.items()):
        raise SilverFinancialError("Invalid FX source/direction configuration")
    accepted, evidence = normalize_bronze_fx_quotes(quotes, pairs=pairs)
    return accepted, evidence, {
        "source_quote_rows": len(evidence),
        "accepted_rates": len(accepted),
        "rejected_rates": int(evidence["fx_validation_status"].ne("accepted").sum()),
        "rate_definition": "USD per 1 unit of original currency",
        "conversion_to_underlying_asset_prices_applied": False,
        "observation_time_verified": False,
        "independently_verified": False,
        "production_eligible": False,
    }
