"""Additional deterministic hypotheses; every accepted step must reconcile on both sources."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from alpha_edge.market.audit_matching_split_basis_v3 import FIELDS, TOL, positive_matrix
from alpha_edge.market.audit_provider_reconciliation_v3 import number, split_ratio

CANDIDATE = "advanced_session_factor_candidate"
POLICY = "calendar_aligned_daily_factor_chain_v2"
CLOSE_CANDIDATE = "close_basis_only_candidate"
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


def analyze_window(y, e, responses, row, event, size, *, close_only=False):
    """Test field-level scales and each action separately across the entire window."""
    day = row["date"]
    out = {"method": "close_only_daily_factor_chain" if close_only else "daily_factor_chain", "window": f"calendar_{size}_each_side",
           "date_method": "reported_dates", "policy": POLICY, "production_eligible": False,
           "economic_review_required": True, "quarantine_released": False}

    def finish(status):
        return {**out, "status": status}

    fields = ["close"] if close_only else FIELDS
    ym, em = positive_matrix(y, fields), positive_matrix(e, fields)
    ya, ea = positive_matrix(y, ["adj_close"]), positive_matrix(e, ["adjusted_close"])
    if any(v is None for v in (ym, em, ya, ea)):
        return finish("advanced_invalid_prices")
    levels = em/ym
    daily = np.median(levels, axis=1)
    errors = np.abs(levels/daily[:,None]-1)
    out["field_max_relative_error"] = {f: float(errors[:,i].max()) for i,f in enumerate(fields)}
    if close_only:
        eligibility = {"close": True}
        for field in ("open", "high", "low"):
            a, b = positive_matrix(y,[field]), positive_matrix(e,[field])
            if a is None or b is None:
                eligibility[field] = False
                continue
            err = float(np.max(np.abs((b[:,0]/a[:,0])/daily-1)))
            out["field_max_relative_error"][field] = err
            eligibility[field] = err <= TOL
        # Volume is never inferred from a closing-price match.
        out["field_agreement_hypothesis"] = {**eligibility, "volume": False}
        out["candidate_eligible_fields"] = ["close"]
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
        yf = float((ya[i,0]/ym[i,-1])/(ya[i-1,0]/ym[i-1,-1]))
        ef = float((ea[i,0]/em[i,-1])/(ea[i-1,0]/em[i-1,-1]))
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
            fraction = yc/float(ym[i-1,-1])
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
    return finish(CLOSE_CANDIDATE if close_only else CANDIDATE)


def cash_sequence_diagnostic(y, e, responses, row):
    """Locate cash implied by each provider factor, without inferring issuer entitlement."""
    out = {"method": "cash_sequence_factor_diagnostic", "status": "cash_sequence_diagnostic_only", "sessions": []}
    if y.index.has_duplicates or e.index.has_duplicates:
        return out
    ym, em = positive_matrix(y, ["close"]), positive_matrix(e, ["close"])
    ya, ea = positive_matrix(y, ["adj_close"]), positive_matrix(e, ["adjusted_close"])
    if any(x is None for x in (ym,em,ya,ea)):
        return out
    dividends = {}
    splits = {}
    for r in responses["div"]:
        if r["date"] in y.index:
            dividends.setdefault(r["date"], []).append(r)
    for r in responses["splits"]:
        if r["date"] in y.index:
            splits.setdefault(r["date"], []).append(r)
    for i in range(1,len(y)):
        d = y.index[i]
        yr = number(y.iloc[i].stock_splits) or 1.
        erows = splits.get(d,[])
        er = split_ratio(erows[0]["split"]) if len(erows)==1 else 1. if not erows else None
        if er is None or not coherent(yr,er,0.0001):
            continue
        yf = float((ya[i,0]/ym[i,0])/(ya[i-1,0]/ym[i-1,0]))
        ef = float((ea[i,0]/em[i,0])/(ea[i-1,0]/em[i-1,0]))/er
        if min(yf,ef)<=0 or not math.isfinite(yf+ef):
            continue
        yc = number(y.iloc[i].dividends)
        implied_y = float(ym[i-1,0])*(1-1/yf)
        implied_e = float(ym[i-1,0])*(1-1/ef)
        records = dividends.get(d,[])
        if yc or records or max(abs(implied_y),abs(implied_e))/float(ym[i-1,0]) > MIN_SIGNAL:
            out["sessions"].append({"date":str(d),"yahoo_reported_cash":yc,"eodhd_reported_cash":records,
                                    "yahoo_implied_cash":implied_y,"eodhd_implied_cash_on_yahoo_basis":implied_e,
                                    "cash_implied_agreement":coherent(implied_y,implied_e),
                                    "issuer_evidence_required":True})
    return out
