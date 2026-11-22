"""Identify possible dividend double application without inferring a correction."""

from __future__ import annotations

import numpy as np
import pandas as pd

POLICY = "dividend_duplicate_evidence_v1"
TOLERANCE = 0.0001
NEARBY_DAYS = 7
COLUMNS = [
    "effective_date",
    "source_value",
    "assessed_value",
    "source_currency",
    "previous_close",
    "observed_factor",
    "single_factor",
    "double_cash_factor",
    "double_sequential_factor",
    "half_cash_factor",
    "implied_cash",
    "related_date",
    "reason",
    "review_status",
]


def assess(bars: pd.DataFrame, actions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare single and double adjustment hypotheses at each dividend session.

    A double match cannot establish whether the amount, ex-date, share basis,
    or adjusted-close series is wrong. Nearby equal payments can be legitimate.
    Neither signal authorizes removing a dividend.
    """
    frame = bars.sort_values("session_date").reset_index(drop=True)
    dates = pd.to_datetime(frame["session_date"], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or frame["asset_id"].nunique() != 1:
        raise ValueError("Expected unique sessions for a single asset")
    close = pd.to_numeric(frame["close"], errors="coerce")
    adj = pd.to_numeric(frame["adj_close"], errors="coerce")
    ratio = (adj / close).where(np.isfinite(adj) & np.isfinite(close) & adj.gt(0) & close.gt(0))
    observed = ratio.shift() / ratio
    by_day = {day.date().isoformat(): i for i, day in enumerate(dates)}
    dividends = actions.loc[actions["action_type"].eq("DIVIDEND")].sort_values("effective_date")
    splits = set(actions.loc[actions["action_type"].eq("SPLIT"), "effective_date"])
    if dividends["effective_date"].duplicated().any():
        raise ValueError("Ambiguous dividend event identity")
    findings = []
    prior = None
    for event in dividends.to_dict("records"):
        day = event["effective_date"]
        i = by_day[day]
        amount = float(event.get("normalized_value", event["source_value"]))
        price = float(close.iloc[i - 1]) if i else float("nan")
        obs = float(observed.iloc[i])
        record = {
            "effective_date": day,
            "source_value": event["source_value"],
            "assessed_value": amount,
            "source_currency": event["source_currency"],
            "previous_close": price,
            "observed_factor": obs,
            "single_factor": None,
            "double_cash_factor": None,
            "double_sequential_factor": None,
            "half_cash_factor": None,
            "implied_cash": None,
            "related_date": None,
            "review_status": "review_required",
        }
        if (
            day in splits
            or event.get("currency_correction_status") == "proposed"
            or not all(np.isfinite(v) and v > 0 for v in (price, amount, obs))
            or amount >= price
        ):
            findings.append({**record, "reason": "dividend_factor_not_evaluable"})
        else:
            single = 1 - amount / price
            double_cash = 1 - 2 * amount / price
            double_sequential = single**2
            half_cash = 1 - amount / (2 * price)
            record.update(
                single_factor=single,
                double_cash_factor=double_cash,
                double_sequential_factor=double_sequential,
                half_cash_factor=half_cash,
                implied_cash=price * (1 - obs),
            )
            if abs(obs - single) > TOLERANCE:
                matches = [
                    name
                    for name, factor in (("double_cash", double_cash), ("double_sequential", double_sequential))
                    if factor > 0 and abs(obs - factor) <= TOLERANCE
                ]
                reason = (
                    "possible_double_adjustment"
                    if matches
                    else (
                        "possible_doubled_reported_amount"
                        if abs(obs - half_cash) <= TOLERANCE
                        else "dividend_factor_mismatch"
                    )
                )
                findings.append({**record, "reason": reason})
        if prior is not None:
            gap = (pd.Timestamp(day) - pd.Timestamp(prior["effective_date"])).days
            if (
                0 < gap <= NEARBY_DAYS
                and event["source_currency"] == prior["source_currency"]
                and np.isclose(amount, prior["assessed_value"], rtol=1e-7, atol=0)
            ):
                findings.append({**record, "related_date": prior["effective_date"], "reason": "nearby_equal_dividends"})
        prior = record
    table = pd.DataFrame(findings, columns=COLUMNS)
    return table, {
        "policy": POLICY,
        "dividends_examined": len(dividends),
        "findings": len(table),
        "counts": {str(k): int(v) for k, v in table["reason"].value_counts().items()},
        "factor_tolerance": TOLERANCE,
        "nearby_days": NEARBY_DAYS,
        "corrections_applied": 0,
        "production_eligible": False,
    }
