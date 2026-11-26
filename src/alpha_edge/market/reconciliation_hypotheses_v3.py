"""Additional deterministic hypotheses; every accepted step must reconcile on both sources."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from alpha_edge.market.audit_matching_split_basis_v3 import FIELDS, TOL, positive_matrix
from alpha_edge.market.audit_provider_reconciliation_v3 import number, split_ratio

CANDIDATE = "advanced_session_factor_candidate"
POLICY = "calendar_aligned_daily_factor_chain_v2"
MIN_SIGNAL = 0.00008


def coherent(a, b, tolerance=TOL):
    return (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0
            and abs(math.log(a/b)) <= tolerance)


def signal_tolerance(factor):
    signal = abs(math.log(factor))
    return min(TOL, signal/8) if signal >= MIN_SIGNAL else None


def calendar_windows(y, e, calendar, responses, day, missing_expected=()):
    """Use all expected sessions, never bridge a missing expected observation."""
    if calendar is None or y.index.has_duplicates or e.index.has_duplicates:
        return []
    dates = pd.to_datetime(calendar.session_date).dt.strftime("%Y-%m-%d")
    if dates.duplicated().any():
        raise ValueError("Duplicate calendar observation")
    expected = pd.Index(dates[calendar.is_expected_session.eq(True)]).sort_values()
    if day not in expected:
        return []
    j = expected.get_loc(day)
    results = []
    for size in (5, 8):
        if j < size or j+size > len(expected):
            continue
        window = expected[j-size:j+size]
        if (pd.Timestamp(window[-1])-pd.Timestamp(window[0])).days > 40:
            continue
        if any(window[0] <= d <= window[-1] for d in missing_expected):
            continue
        if not window.isin(y.index).all() or not window.isin(e.index).all():
            continue
        eligible = pd.Series(calendar.ohlc_eligible.eq(True).to_numpy(), index=dates)
        if not eligible.loc[window].all():
            continue
        skipped = y.loc[window[0]:window[-1]].index.difference(window)
        if any(pd.to_numeric(y.loc[skipped,c], errors="coerce").fillna(0).ne(0).any()
               for c in ("stock_splits", "dividends")):
            continue
        if any(window[0] <= r["date"] <= window[-1] and r["date"] not in window
               for ep in ("splits","div") for r in responses[ep]):
            continue
        results.append((size, y.loc[window], e.loc[window]))
    return results


def analyze_window(y, e, responses, row, event, size):
    """Test field-level scales and each action separately across the entire window."""
    day = row["date"]
    out = {"method": "daily_factor_chain", "window": f"calendar_{size}_each_side",
           "date_method": "reported_dates", "policy": POLICY, "production_eligible": False,
           "economic_review_required": True, "quarantine_released": False}

    def finish(status):
        return {**out, "status": status}

    ym, em = positive_matrix(y, FIELDS), positive_matrix(e, FIELDS)
    ya, ea = positive_matrix(y, ["adj_close"]), positive_matrix(e, ["adjusted_close"])
    if any(v is None for v in (ym, em, ya, ea)):
        return finish("advanced_invalid_prices")
    levels = em/ym
    daily = np.median(levels, axis=1)
    errors = np.abs(levels/daily[:,None]-1)
    out["field_max_relative_error"] = {f: float(errors[:,i].max()) for i,f in enumerate(FIELDS)}
    out["max_relative_field_error"] = float(errors.max())
    # A decimal rounding hypothesis is reported, not used to waive a mismatch.
    rounded = np.abs(em - ym*daily[:,None])
    out["four_decimal_rounding_compatible"] = bool((rounded <= 0.00005 + np.abs(ym*daily[:,None])*1e-7).all())
    if event["date_status"] != "exact_date" or event["value_status"] != "ratio_match":
        return finish("advanced_independent_event_or_date_evidence_required")
    if not coherent(float(y.loc[day,"stock_splits"]),float(row["ratio"]),1e-7):
        raise ValueError("Yahoo event ratio changed")
    dates = set(y.index)
    shares, cash = {}, {}
    for endpoint, target in (("splits", shares), ("div", cash)):
        for action in responses[endpoint]:
            if action["date"] in dates:
                target.setdefault(action["date"], []).append(action)
    if any(len(v) != 1 for v in shares.values()) or any(len(v) != 1 for v in cash.values()):
        return finish("advanced_duplicate_provider_action")
    # A first-window action has no predecessor; refuse to attribute it.
    if y.index[0] in shares or y.index[0] in cash or float(y.iloc[0].stock_splits) != 0 or float(y.iloc[0].dividends) != 0:
        return finish("advanced_first_session_action")
    checks = []
    out["daily_checks"] = checks
    # Verify each date, including ordinary sessions, with no inferred cash or share action.
    for i in range(1,len(y)):
        d = y.index[i]
        yr = number(y.iloc[i].stock_splits)
        yc = number(y.iloc[i].dividends)
        if yr is None or yc is None or yr < 0 or yc < 0:
            return finish("advanced_invalid_source_action")
        raw_ratio = yr or 1.0
        er = split_ratio(shares[d][0]["split"]) if d in shares else 1.0
        if er is None or not coherent(raw_ratio, er, 0.0001):
            return finish("advanced_share_sequence_disagrees")
        st = signal_tolerance(raw_ratio) if raw_ratio != 1 else TOL
        if st is None:
            return finish("advanced_share_signal_below_resolution")
        if not coherent(float(daily[i-1]/daily[i]),raw_ratio,st):
            return finish("advanced_daily_scale_step_disagrees")
        if float(max(errors[i-1].max(),errors[i].max())) > st:
            return finish("advanced_field_scale_disagrees")
        yf = float((ya[i,0]/ym[i,3])/(ya[i-1,0]/ym[i-1,3]))
        ef = float((ea[i,0]/em[i,3])/(ea[i-1,0]/em[i-1,3]))
        expected_cash = 1.0
        cash_basis = "none"
        ct = TOL
        if yc or d in cash:
            if yc <= 0 or d not in cash:
                return finish("advanced_cash_sequence_disagrees")
            record = cash[d][0]
            currency = y.iloc[i].currency_original
            if not isinstance(currency,str) or currency != record.get("currency"):
                return finish("advanced_cash_currency_disagrees")
            original, adjusted = number(record.get("unadjustedValue")), number(record.get("value"))
            bases = []
            if original and original > 0:
                if coherent(original/daily[i-1],yc):
                    bases.append("pre")
                if coherent(original/daily[i],yc):
                    bases.append("post")
            if adjusted and adjusted > 0 and coherent(adjusted,yc):
                bases.append("reported_adjusted")
            if not bases:
                return finish("advanced_cash_amount_disagrees")
            fraction = yc/float(ym[i-1,3])
            if not 0 < fraction < 1:
                return finish("advanced_cash_fraction_invalid")
            expected_cash = 1/(1-fraction)
            ct = signal_tolerance(expected_cash)
            if ct is None:
                return finish("advanced_cash_signal_below_resolution")
            cash_basis = "+".join(bases)
        if not coherent(yf,expected_cash,ct) or not coherent(ef,raw_ratio*expected_cash,min(ct,st)):
            return finish("advanced_independent_factor_step_disagrees")
        yret, eret = float(ya[i,0]/ya[i-1,0]), float(ea[i,0]/ea[i-1,0])
        if not coherent(yret,eret,min(ct,st)):
            return finish("advanced_adjusted_return_disagrees")
        checks.append({"date": str(d), "share_ratio": raw_ratio, "yahoo_cash": yc, "cash_evidence": cash_basis,
                       "observed_scale_step": float(daily[i-1]/daily[i]), "expected_cash_step": expected_cash,
                       "yahoo_factor_step": yf, "eodhd_factor_step": ef,
                       "yahoo_adjusted_return": yret-1, "eodhd_adjusted_return": eret-1,
                       "large_return_review_required": abs(math.log(yret)) > 0.10})
    # Verify the target was actually a non-unit action and is covered by the chain.
    if not any(c["date"] == day and c["share_ratio"] != 1 for c in checks):
        return finish("advanced_target_not_evaluable")
    out["large_return_review_required"] = any(c["large_return_review_required"] for c in checks)
    out["cash_raw_basis_review_required"] = any(c["cash_evidence"] == "reported_adjusted" for c in checks)
    out["cumulative_share_ratio"] = math.prod(c["share_ratio"] for c in checks)
    out["observed_cumulative_scale_step"] = float(daily[0]/daily[-1])
    # Limit cumulative drift in addition to the individual step tests.
    if not coherent(out["observed_cumulative_scale_step"],out["cumulative_share_ratio"]):
        return finish("advanced_cumulative_scale_disagrees")
    return finish(CANDIDATE)
