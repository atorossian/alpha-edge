"""Integrate pinned AEA-94 evidence into local-currency Silver candidates.

No production promotion, implicit FX conversion or alteration of Bronze.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import boto3
import numpy as np
import pandas as pd

from alpha_edge.market import (
    build_dividend_event_candidate_v3 as events,
    build_share_basis_candidate_v3 as shares,
    consolidated_split_decisions_v3 as residual,
    dividend_duplicate_evidence_v3 as duplicate_evidence,
    dividend_duplicate_registry_v3 as duplicates,
    historical_quarantine_v3 as quarantine,
)
from alpha_edge.market.audit_financial_universe_v3 import consolidated_receipt, load_consolidated
from alpha_edge.market.dividend_currency_registry_v3 import apply_currency_corrections, load_registry
from alpha_edge.market.ingest_bronze_full_history import DEFAULT_BUCKET, DEFAULT_REGION, load_universe
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT,
    check_id,
    digest,
    json_bytes,
    parquet_bytes,
    put_immutable,
    verified_bytes,
    verified_frame,
)
from alpha_edge.market.silver_financial_evidence_v3 import extract_source_actions
from alpha_edge.market.split_basis_eligibility_v3 import annotate_candidate, load_approved_basis

OUTPUT_ROOT = f"{ROOT}/silver/adjusted_candidates/v1"
OHLC = ("open", "high", "low", "close")
FAMILIES = {
    "reconciliation": "dividend_factor_reconciliation_registries",
    "missing": "missing_session_dividend_candidate_registries",
    "complex": "complex_dividend_candidate_registries",
    "spin": "spin_off_basis_candidate_registries",
    "share": "share_basis_candidate_registries",
}


def _registry(s3, bucket, family, version):
    key = f"{ROOT}/config/{family}/v1/{version}/registry.json"
    doc = json.loads(verified_bytes(s3, bucket, {"key": key, "sha256": version}))
    if doc.get("production_eligible", False) is not False:
        raise ValueError("Expected candidate-only registry")
    return doc


def load_bundle(s3, bucket, version):
    config = _registry(s3, bucket, "financial_integration_registries", version)
    if (
        config.get("schema") != "alpha_edge_financial_integration_registry_v1"
        or config.get("status") != "candidate_only"
    ):
        raise ValueError("Invalid integration registry")
    pins = config["registries"]
    if set(pins) != set(FAMILIES) | {"currency", "duplicates", "quarantine", "additional_events"}:
        raise ValueError("Expected exact integration registry families")
    bundle = {name: _registry(s3, bucket, family, pins[name]) for name, family in FAMILIES.items()}
    r = bundle["reconciliation"]
    refs = {ref["version"] for ref in r["event_reconstruction_registries"]} | set(pins["additional_events"])
    bundle["events"] = [
        events.validate_registry(_registry(s3, bucket, "dividend_event_candidate_registries", v)) for v in sorted(refs)
    ]
    bundle["suppressions"] = [
        _registry(s3, bucket, "dividend_event_suppression_candidate_registries", ref["version"])
        for ref in r["event_suppression_registries"]
    ]
    bundle["currency"] = load_registry(s3, bucket, pins["currency"])
    bundle["duplicates"] = duplicates.load_registry(s3, bucket, pins["duplicates"])
    bundle["quarantine"] = quarantine.load_registry(s3, bucket, pins["quarantine"])
    shares.validate_registry(bundle["share"])
    verified_bytes(s3, bucket, {"key": bundle["share"]["evidence_key"], "sha256": bundle["share"]["evidence_sha256"]})
    # The cumulative 165-case ledger plus 2 VESTA cases is distinct from the
    # underlying action windows; count each source discrepancy exactly once.
    expected = {
        (e["asset_id"], e["effective_date"])
        for name in (
            "entries",
            "additional_currency_candidates",
            "event_reconstruction_candidates",
            "unresolved_events",
        )
        for e in r[name]
    }
    if len(expected) != config["expected_cohort_cases"]:
        raise ValueError("Unexpected reconciliation cohort")
    final = {(e["asset_id"], e["source_dividend_date"]) for e in bundle["missing"]["entries"]}
    final |= {(e["asset_id"], e["source_date"]) for e in bundle["complex"]["entries"]}
    final.add((bundle["spin"]["asset_id"], bundle["spin"]["cash_event"]["ex_date"]))
    if final != {(e["asset_id"], e["effective_date"]) for e in r["unresolved_events"]}:
        raise ValueError("Residual candidates do not cover exact held cohort")
    bundle["cohort"] = expected
    bundle["consolidated_index_version"] = config["consolidated_index_version"]
    bundle["versions"] = {
        **pins,
        "integration": version,
        "events": sorted(refs),
        "suppressions": [r["version"] for r in r["event_suppression_registries"]],
    }
    return bundle


def _same(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-8)


def reconstruct(bars, calendar, findings, bundle, *, asset_id, symbol, source_key, sha, residual_responses=None):
    """Apply each pinned correction once; keep source and candidate fields separate."""
    raw = bars.copy(deep=True).sort_values("session_date").reset_index(drop=True)
    if (
        raw.empty
        or raw.session_date.duplicated().any()
        or not raw.asset_id.eq(asset_id).all()
        or not raw.provider_symbol.eq(symbol).all()
    ):
        raise ValueError("Invalid Bronze identity or sessions")
    raw = raw.set_index("session_date", drop=False)
    f = raw.copy(deep=True)
    for c in (*OHLC, "volume"):
        f[f"candidate_{c}"] = pd.to_numeric(f[c], errors="coerce").astype(float)
    f["candidate_cash"] = f.dividends.astype(float)
    f["candidate_share_multiplier"] = 1.0
    f["candidate_spin_reference"] = 0.0
    f["candidate_spin_actual"] = 0.0
    f["candidate_provisional"] = False
    f["candidate_volume_estimated"] = False
    f["candidate_origin"] = "bronze"
    f["correction_sources"] = ""
    cal = calendar.set_index("session_date")
    if not cal.index.is_unique or set(cal.index) != set(raw.index):
        raise ValueError("Consolidated calendar does not match source dates")
    applied = set()
    cash_overrides = {}
    decisions = bundle.get("residual_decisions", {}).get(asset_id, {})
    if "residual_decisions" in bundle:
        residual.check_source(raw, decisions, asset_id=asset_id, symbol=symbol, sha=sha)

    def pin(e):
        if (
            e["asset_id"] != asset_id
            or e["provider_symbol"] != symbol
            or e.get("source_sha256", e.get("bronze_sha256")) != sha
            or e.get("source_key", e.get("bronze_key", source_key)) != source_key
        ):
            raise ValueError(f"Stale correction source: {asset_id}")

    def rows(doc):
        return [e for e in doc["entries"] if e["asset_id"] == asset_id]

    def check(day, cash):
        if day not in raw.index or not _same(raw.at[day, "dividends"], cash):
            raise ValueError(f"Source dividend changed: {asset_id} {day}")

    def mark(day, name, provisional=False):
        old = f.at[day, "correction_sources"]
        f.at[day, "correction_sources"] = ",".join(sorted(set(filter(None, old.split(","))) | {name}))
        f.at[day, "candidate_provisional"] = bool(f.at[day, "candidate_provisional"] or provisional)

    def override(day, value, name, provisional=False):
        value = float(value)
        if day not in f.index or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid candidate cash target")
        if day in cash_overrides and not _same(cash_overrides[day], value):
            raise ValueError(f"Conflicting candidate cash: {asset_id} {day}")
        cash_overrides[day] = value
        f.at[day, "candidate_cash"] = value
        mark(day, name, provisional)

    for day in cal.index[cal.close_correction_id.fillna("").ne("")]:
        f.at[day, "candidate_close"] = float(cal.at[day, "effective_close"])
        mark(day, "consolidated_close_correction")

    actions, _ = extract_source_actions(raw.reset_index(drop=True))
    currency_normalized, _ = apply_currency_corrections(
        actions, bundle["currency"], asset_id=asset_id, provider_symbol=symbol, bronze_sha256=sha
    )
    normalized, _ = duplicates.apply_suppressions(
        currency_normalized, bundle["duplicates"], asset_id=asset_id, provider_symbol=symbol, bronze_sha256=sha
    )
    for e in normalized.loc[normalized.action_type.eq("DIVIDEND")].to_dict("records"):
        if not _same(e["source_value"], e["normalized_value"]):
            override(e["effective_date"], e["normalized_value"], "approved_currency_or_duplicate")
    # Historical share reconstruction uses its own normalization of the raw
    # share basis; reject an unspecifiable overlapping cash correction.
    share = bundle["share"]
    if share["asset_id"] == asset_id:
        pin(share)
        t, _ = shares.reconstruct(raw.reset_index(drop=True), share, bronze_sha256=sha)
        t = t.set_index("session_date")
        if cash_overrides:
            raise ValueError("Share-basis and cash corrections require explicit combined basis")
        for c in OHLC:
            f[f"candidate_{c}"] = t[f"candidate_{c}"]
        f["candidate_volume"] = f.volume * t.missing_subsequent_share_factor
        f["candidate_cash"] = t.candidate_cash_dividend
        f["candidate_provisional"] = True
        f["correction_sources"] = "missing_share_basis"

    r = bundle["reconciliation"]
    for e in r["entries"] + r["additional_currency_candidates"]:
        if e["asset_id"] != asset_id:
            continue
        pin(e)
        day = e["effective_date"]
        check(day, e["source_cash"])
        if raw.at[day, "currency_original"] != e["price_currency"]:
            raise ValueError("Factor correction currency changed")
        i = raw.index.get_loc(day)
        if i == 0 or not _same(raw.iloc[i - 1].close, e["previous_close"]):
            raise ValueError("Factor correction predecessor changed")
        if not _same(1 - float(e["verified_candidate_cash"]) / float(e["previous_close"]), e["candidate_factor"]):
            raise ValueError("Pinned cash and factor do not reconcile")
        # Currency normalization may precede an explicit final cash decision.
        if day in cash_overrides and _same(cash_overrides[day], e["assessed_cash"]):
            del cash_overrides[day]
        override(day, e["verified_candidate_cash"], "factor_reconciliation", True)
        applied.add((asset_id, day))
    for doc in bundle["events"]:
        for e in rows(doc):
            pin(e)
            t = events.reconstruct(raw.reset_index(drop=True), e, source_sha256=sha)
            for v in t.to_dict("records"):
                if v["source_cash"] or v["candidate_cash"]:
                    override(
                        v["session_date"], v["candidate_cash"], "event_reconstruction", e["basis_status"] != "verified"
                    )
    for doc in bundle["suppressions"]:
        for e in rows(doc):
            pin(e)
            day = e["effective_date"]
            check(day, e["source_cash"])
            for retained in e["retained_source_events"]:
                check(retained["date"], retained["cash"])
                override(retained["date"], retained.get("candidate_cash", retained["cash"]), "retained_event")
            override(day, 0, "event_suppression")
    for e in r["event_reconstruction_candidates"]:
        if e["asset_id"] == asset_id:
            pin(e)
            day = e["effective_date"]
            check(day, e["source_cash"])
            if day not in cash_overrides or not _same(cash_overrides[day], e["candidate_cash"]):
                raise ValueError(f"Cohort event lacks applied reconstruction: {asset_id} {day}")
            applied.add((asset_id, day))

    for e in rows(bundle["missing"]):
        pin(e)
        day, old = e["session_date"], e["source_dividend_date"]
        check(old, e["source_cash"])
        if day in f.index:
            raise ValueError("Refusing to overwrite existing observation")
        ordered = sorted(raw.index)
        i = ordered.index(old)
        if not old < day < ordered[i + 1] or ordered[i + 1] != e["next_observation"]["session_date"]:
            raise ValueError("Missing-session anchors changed")
        if raw.at[old, "currency_original"] != e["currency_original"]:
            raise ValueError("Missing-session currency changed")
        for anchor in (e.get("prior_observation"), e["next_observation"]):
            if anchor is None:
                raise ValueError("Missing source session anchor")
            for col in (*OHLC, "volume", "dividends", "stock_splits"):
                if not _same(raw.at[anchor["session_date"], col], anchor[col]):
                    raise ValueError("Missing-session source anchor changed")
        f.loc[day] = {
            "session_date": day,
            "asset_id": asset_id,
            "provider_symbol": symbol,
            "currency_original": e["currency_original"],
            "candidate_origin": "manual_observation",
            "candidate_provisional": True,
            "candidate_volume_estimated": True,
            "candidate_share_multiplier": 1.0,
            "candidate_spin_reference": 0.0,
            "candidate_spin_actual": 0.0,
            "correction_sources": "",
        }
        for c in OHLC:
            f.at[day, f"candidate_{c}"] = e[c]
        f.at[day, "candidate_volume"] = e["volume_estimate"]
        override(old, 0, "misdated_missing_session", True)
        override(day, e["candidate_cash"], "missing_session", True)
        applied.add((asset_id, old))

    for e in rows(bundle["complex"]):
        pin(e)
        check(e["source_date"], e["source_cash"])
        if e["target"] != e["source_date"]:
            check(e["target"], 0)
        for action in e["actions"]:
            day = action["date"]
            override(day, action["candidate_cash"], "complex_action", True)
            f.at[day, "candidate_share_multiplier"] = action.get("share_multiplier", 1)
        applied.add((asset_id, e["source_date"]))
    spin = bundle["spin"]
    if spin["asset_id"] == asset_id:
        pin(spin)
        for e in spin["rows"]:
            day = e["session_date"]
            for c in (*OHLC, "volume", "dividends", "stock_splits"):
                if not _same(raw.at[day, c], e["source"][c]):
                    raise ValueError("Spin-off source observation changed")
            for c in OHLC:
                f.at[day, f"candidate_{c}"] = e["candidate"][c]
            f.at[day, "candidate_volume"] = e["candidate"]["volume_estimate"]
            f.at[day, "candidate_volume_estimated"] = True
            mark(day, "spin_off_quote_basis", True)
        cash = spin["cash_event"]
        check(cash["ex_date"], cash["source_cash"])
        override(cash["ex_date"], cash["normalized_cash"], "spin_off_cash", True)
        e = spin["spin_off_event"]
        f.at[e["effective_date"], "candidate_spin_reference"] = e["reference_value"]
        f.at[e["effective_date"], "candidate_spin_actual"] = (
            e["actual_distribution_close"] if e.get("actual_distribution_close") is not None else np.nan
        )
        mark(e["effective_date"], "spin_off_distribution", True)
        applied.add((asset_id, cash["ex_date"]))

    if "residual_decisions" in bundle:
        residual.apply_existing_repairs(f, raw, decisions, residual_responses, source_key=source_key, sha=sha)

    # Preserve original calendar decisions and quarantine. The manual session
    # is specifically evidenced; no other missing session is silently filled.
    f = f.sort_index()
    manual = f.candidate_origin.eq("manual_observation")
    expected = cal.is_expected_session.reindex(f.index).eq(True) | manual
    f["candidate_expected_session"] = expected
    f["candidate_session_calendar_provisional"] = manual
    eligible = (
        cal.ohlc_eligible.reindex(f.index).eq(True)
        | manual
        | f.correction_sources.str.contains("spin_off_quote_basis", regex=False)
    )
    gaps = set(findings.loc[findings.issue.eq("missing_expected_session"), "session_date"].astype(str)) - set(f.index)
    dates = pd.Series(f.index, index=f.index)
    # Work on expected trading sessions only; preserve placeholders in output.
    trade = f.loc[expected].copy()
    col = [f"candidate_{c}" for c in OHLC]
    valid = (
        np.isfinite(trade[col]).all(axis=1)
        & trade[col].gt(0).all(axis=1)
        & trade.candidate_high.ge(trade[["candidate_open", "candidate_close"]].max(axis=1))
        & trade.candidate_low.le(trade[["candidate_open", "candidate_close"]].min(axis=1))
        & eligible.loc[trade.index]
    )
    prev = trade.candidate_close.shift()
    prev_day = pd.Series(trade.index, index=trade.index).shift()
    cur_day = dates.loc[trade.index]
    blocked = pd.Series(False, index=trade.index)
    for day in gaps | set(cal.index[cal.calendar_exception_review.eq(True)]):
        blocked |= prev_day.lt(day) & cur_day.gt(day)
    good = valid & valid.shift(fill_value=False) & ~blocked
    if "residual_decisions" in bundle:
        good &= ~trade.candidate_residual_boundary_review & ~trade.candidate_residual_cash_review
    trade["candidate_price_return"] = (trade.candidate_close / prev - 1).where(good)
    value = trade.candidate_close * trade.candidate_share_multiplier + trade.candidate_cash
    trade["candidate_total_return"] = ((value + trade.candidate_spin_actual) / prev - 1).where(good)
    trade["candidate_reference_return"] = ((value + trade.candidate_spin_reference) / prev - 1).where(good)
    trade["candidate_adjustment_factor"] = (
        (1 - (trade.candidate_cash + trade.candidate_spin_reference) / prev) / trade.candidate_share_multiplier
    ).where(good)
    invalid_factor = ~np.isfinite(trade.candidate_adjustment_factor) | trade.candidate_adjustment_factor.le(0)
    trade["candidate_total_return"] = trade.candidate_total_return.mask(invalid_factor)
    trade["candidate_reference_return"] = trade.candidate_reference_return.mask(invalid_factor)
    for c in ("candidate_total_return", "candidate_reference_return", "candidate_adjustment_factor"):
        trade[c] = trade[c].where(np.isfinite(trade[c]) & trade[c].gt(0 if c.endswith("factor") else -1))
    # Chain actual total returns only. Unknown distributed security close or
    # missing session breaks the chain, never masquerades as a zero return.
    trade["candidate_adjusted_close"] = np.nan
    trade["candidate_segment_adjusted_close"] = np.nan
    trade["candidate_segment_id"] = pd.Series(pd.NA, index=trade.index, dtype="string")
    if len(trade):
        ret = trade.candidate_total_return.to_numpy()
        bad = np.flatnonzero(~np.isfinite(ret[1:]) | (ret[1:] <= -1)) + 1
        boundaries = np.r_[0, bad, len(trade)]
        for start, end in pairwise(boundaries):
            if not bool(valid.iloc[end - 1]):
                continue
            chain = np.cumprod(1 + ret[start + 1 : end][::-1])[::-1]
            anchor = trade.candidate_close.iloc[end - 1]
            values = np.r_[anchor / chain, anchor]
            if not np.isfinite(values).all() or not (values > 0).all():
                raise ValueError("Non-finite adjusted candidate chain")
            ix = trade.index[start:end]
            trade.loc[ix, "candidate_segment_adjusted_close"] = values
            trade.loc[ix, "candidate_segment_id"] = str(ix[-1])
            if end == len(trade):
                trade.loc[ix, "candidate_adjusted_close"] = values
    for c in (
        "candidate_price_return",
        "candidate_total_return",
        "candidate_reference_return",
        "candidate_adjustment_factor",
        "candidate_adjusted_close",
        "candidate_segment_adjusted_close",
        "candidate_segment_id",
    ):
        f[c] = trade[c]
    for c in ("open", "high", "low"):
        f[f"candidate_adjusted_{c}"] = f[f"candidate_{c}"] * f.candidate_adjusted_close / f.candidate_close
        f[f"candidate_segment_adjusted_{c}"] = (
            f[f"candidate_{c}"] * f.candidate_segment_adjusted_close / f.candidate_close
        )
    # Cash distributions and spin-off reference coefficients do not change
    # share counts; volume uses only the corrected quotation/share basis.
    f["candidate_adjusted_volume"] = f.candidate_volume
    f["candidate_return_review_required"] = expected & f.candidate_total_return.isna()
    f["candidate_unavailable_factor"] = expected & f.candidate_adjustment_factor.isna()
    f["candidate_large_return_review"] = f.candidate_total_return.abs().gt(0.30)
    dup, _ = duplicate_evidence.assess(raw.reset_index(drop=True), currency_normalized)
    dup = duplicates.mark_legitimate_pairs(
        dup, currency_normalized, bundle["duplicates"], asset_id=asset_id, provider_symbol=symbol, bronze_sha256=sha
    )
    _, qfindings, q = quarantine.assess(
        raw.reset_index(drop=True),
        currency_normalized,
        asset_id=asset_id,
        provider_symbol=symbol,
        bronze_sha256=sha,
        registry=bundle["quarantine"],
        duplicate_findings=dup,
    )
    f["historical_quarantine"] = f.index <= q["quarantine_through"] if q["quarantine_through"] else False
    if "split_basis_approved" in bundle:
        annotate_candidate(
            f,
            raw,
            qfindings,
            bundle["split_basis_approved"].get(asset_id, {}),
            asset_id=asset_id,
            symbol=symbol,
            sha=sha,
        )
    if "residual_decisions" in bundle:
        evidence = (
            residual.close_evidence(raw, calendar, findings, decisions, residual_responses)
            if residual_responses is not None
            else {}
        )
        residual.annotate(f, raw, decisions, evidence, version=bundle["residual_version"])
    f["production_eligible"] = False
    return f.reset_index(drop=True), applied


def build(
    s3,
    bucket,
    *,
    consolidated_version,
    bundle_version,
    run_id,
    selectors=None,
    workers=4,
    execute=False,
    split_basis_approvals=(),
    deterministic_basis_approvals=(),
    expected_split_approvals=None,
    residual_version=None,
    expected_residual_count=None,
    eodhd_run_id=None,
    execution_plan_version=None,
):
    check_id(run_id)
    if not 1 <= workers <= 16:
        raise ValueError("Invalid worker count")
    index = load_consolidated(s3, bucket, consolidated_version)
    active, _ = load_universe(s3, bucket, index["source_fingerprint"])
    if set(index["assets"]) != {r["asset_id"] for r in active}:
        raise ValueError("Consolidated universe mismatch")
    selected = [r for r in active if selectors is None or r["asset_id"] in selectors]
    if not selected or (selectors is not None and len(selected) != len(set(selectors))):
        raise ValueError("Unknown canonical asset selector")
    bundle = load_bundle(s3, bucket, bundle_version)
    if bundle["consolidated_index_version"] != consolidated_version:
        raise ValueError("Integration bundle targets a different consolidated index")
    if split_basis_approvals or deterministic_basis_approvals:
        bundle["split_basis_approved"] = load_approved_basis(
            s3, bucket, split_basis_approvals, deterministic_basis_approvals, expected_split_approvals
        )
        if not set(bundle["split_basis_approved"]) <= set(index["assets"]):
            raise ValueError("Approved split asset outside consolidated index")
    elif expected_split_approvals is not None:
        raise ValueError("Expected split approvals requires registry versions")
    if residual_version is not None:
        if not expected_residual_count or not eodhd_run_id:
            raise ValueError("Residual decisions require an expected count and pinned EODHD run")
        bundle["residual_decisions"] = residual.load(
            s3,
            bucket,
            residual_version,
            consolidated_version=consolidated_version,
            bundle_version=bundle_version,
            expected_count=expected_residual_count,
        )
        bundle["residual_version"] = residual_version
        if not set(bundle["residual_decisions"]) <= set(index["assets"]):
            raise ValueError("Residual decision asset outside consolidated index")
        for aid, rows in bundle["residual_decisions"].items():
            if set(rows) & set(bundle.get("split_basis_approved", {}).get(aid, {})):
                raise ValueError("Duplicate standard and residual split approval")
    elif expected_residual_count is not None or eodhd_run_id is not None:
        raise ValueError("Residual registry required")
    root = f"{OUTPUT_ROOT}/runs/{run_id}"

    def process(row):
        aid = row["asset_id"]
        receipt = consolidated_receipt(s3, bucket, index, row)
        ref = receipt["data"]
        bars = verified_frame(s3, bucket, ref)
        outputs = index["assets"][aid]["outputs"]
        cal = verified_frame(s3, bucket, outputs["candidate"])
        findings = verified_frame(s3, bucket, outputs["findings"])
        responses = None
        decisions = bundle.get("residual_decisions", {}).get(aid, {})
        needed = [
            e
            for e in decisions.values()
            if e["financial_status"] == "close_only_pending_integration"
            or (e["financial_status"] == "approved_price_basis_pending_integration" and e["eodhd_sha256"] is not None)
            or (e["source_doc"] or {}).get("schema") == "alpha_edge_asset_price_reconstruction_approval_v1"
        ]
        if needed:
            hashes = {e["eodhd_sha256"] for e in needed}
            if len(hashes) != 1 or None in hashes:
                raise ValueError(f"Missing or inconsistent pinned EODHD evidence: {aid}")
            responses = residual.read_eodhd(s3, bucket, aid, eodhd_run_id, hashes.pop())
        frame, applied = reconstruct(
            bars,
            cal,
            findings,
            bundle,
            asset_id=aid,
            symbol=row["yahoo_ticker"],
            source_key=ref["key"],
            sha=ref["sha256"],
            residual_responses=responses,
        )
        report = {
            "asset_id": aid,
            "source_rows": len(bars),
            "candidate_rows": len(frame),
            "cohort_cases": len(applied),
            "candidate_returns": int(frame.candidate_total_return.notna().sum()),
            "provisional_rows": int(frame.candidate_provisional.sum()),
            "quarantined_rows": int(frame.historical_quarantine.sum()),
            "adjusted_rows": int(frame.candidate_adjusted_close.notna().sum()),
            "segment_adjusted_rows": int(frame.candidate_segment_adjusted_close.notna().sum()),
            "return_review_rows": int(frame.candidate_return_review_required.sum()),
            "large_return_review_rows": int(frame.candidate_large_return_review.sum()),
            "unresolved_spin_rows": int(frame.candidate_spin_actual.isna().sum()),
        }
        if "split_basis_approved" in bundle:
            report.update(
                {
                    "approved_split_events": int(frame.candidate_split_basis_approved.sum()),
                    "potential_returns_before": int(frame.candidate_return_potential_before_split_basis_review.sum()),
                    "potential_returns_after": int(frame.candidate_return_potential_after_split_basis_review.sum()),
                    "retained_large_return_review_sessions": int(frame.candidate_split_basis_large_return_review.sum()),
                }
            )
        if "residual_decisions" in bundle:
            report.update(residual.validate_candidate(frame, bars, decisions))
        if execute:
            blob = parquet_bytes(frame)
            key = f"{root}/assets/{digest(aid.encode())}/prices-{digest(blob)}.parquet"
            report["output"] = put_immutable(
                s3, bucket, key, blob, content_type="application/vnd.apache.parquet", scope=OUTPUT_ROOT
            )
        return report, applied

    reports, seen = [], set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (report, applied) in enumerate(pool.map(process, sorted(selected, key=lambda r: r["asset_id"])), 1):
            reports.append(report)
            seen.update(applied)
            if i % 50 == 0 or i == len(selected):
                print(f"[adjusted candidate] {i}/{len(selected)}", file=sys.stderr, flush=True)
    selected_ids = {r["asset_id"] for r in selected}
    if selectors is None and not {a for a, _ in bundle["cohort"]} <= selected_ids:
        raise ValueError("Full universe excludes a pinned reconciliation asset")
    if seen != {x for x in bundle["cohort"] if x[0] in selected_ids}:
        raise ValueError("Reconciliation cohort not fully applied")
    if "split_basis_approved" in bundle:
        expected_approved = sum(len(v) for a, v in bundle["split_basis_approved"].items() if a in selected_ids)
        if sum(r["approved_split_events"] for r in reports) != expected_approved:
            raise ValueError("Not all selected split approvals matched source events")
    manifest = {
        "schema": "alpha_edge_silver_adjusted_candidate_v1",
        "run_id": run_id,
        "consolidated_index_version": consolidated_version,
        "registries": bundle["versions"],
        "scope": "full_universe" if selectors is None else "selected_assets",
        "statistics": {
            "assets": len(reports),
            **{
                c: sum(r[c] for r in reports)
                for c in (
                    "source_rows",
                    "candidate_rows",
                    "cohort_cases",
                    "candidate_returns",
                    "provisional_rows",
                    "quarantined_rows",
                    "adjusted_rows",
                    "segment_adjusted_rows",
                    "return_review_rows",
                    "large_return_review_rows",
                    "unresolved_spin_rows",
                )
            },
        },
        "assets": reports,
        "production_eligible": False,
        "currency": "original_local_currency",
        "quarantine_released": False,
    }
    if "split_basis_approved" in bundle:
        manifest["split_basis_approvals"] = {
            "standard": list(split_basis_approvals),
            "deterministic": list(deterministic_basis_approvals),
            "scope": "price_basis_only_counterfactual",
        }
        manifest["statistics"].update(
            {
                k: sum(r[k] for r in reports)
                for k in (
                    "approved_split_events",
                    "potential_returns_before",
                    "potential_returns_after",
                    "retained_large_return_review_sessions",
                )
            }
        )
        manifest["statistics"]["potential_incremental_returns"] = (
            manifest["statistics"]["potential_returns_after"] - manifest["statistics"]["potential_returns_before"]
        )
    if "residual_decisions" in bundle:
        expected = sum(len(v) for a, v in bundle["residual_decisions"].items() if a in selected_ids)
        keys = (
            "residual_decision_events",
            "residual_source_verified_events",
            "residual_reconstruction_applied_events",
            "residual_close_reverified_events",
            "residual_full_price_basis_reverified_events",
            "residual_close_return_rows",
            "residual_financial_blocker_events",
        )
        manifest["statistics"].update({k: sum(r[k] for r in reports) for k in keys})
        if manifest["statistics"]["residual_decision_events"] != expected:
            raise ValueError("Not all selected residual decisions were represented")
        manifest["residual_decisions"] = {
            "version": residual_version,
            "eodhd_run_id": eodhd_run_id,
            "expected_count": expected,
            "scope": "candidate_only_validation",
            "financial_validation_complete": False,
        }
    if execution_plan_version is not None:
        manifest["execution_plan_version"] = execution_plan_version
    if execute:
        put_immutable(
            s3,
            bucket,
            f"{root}/manifest.json",
            json_bytes(manifest),
            content_type="application/json",
            scope=OUTPUT_ROOT,
        )
    return {**manifest, "status": "candidate_written" if execute else "dry_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--consolidated-index-version")
    parser.add_argument("--bundle-version")
    parser.add_argument("--execution-plan-version")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--only", nargs="+")
    parser.add_argument("--split-basis-approvals", nargs="+", default=[])
    parser.add_argument("--deterministic-basis-approvals", nargs="+", default=[])
    parser.add_argument("--expected-split-approvals", type=int)
    parser.add_argument("--residual-version")
    parser.add_argument("--expected-residual-count", type=int)
    parser.add_argument("--eodhd-run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    client = boto3.client("s3", region_name=args.region)
    inputs = {
        "consolidated_version": args.consolidated_index_version,
        "bundle_version": args.bundle_version,
        "split_basis_approvals": args.split_basis_approvals,
        "deterministic_basis_approvals": args.deterministic_basis_approvals,
        "expected_split_approvals": args.expected_split_approvals,
        "residual_version": args.residual_version,
        "expected_residual_count": args.expected_residual_count,
        "eodhd_run_id": args.eodhd_run_id,
    }
    if args.execution_plan_version:
        if any(inputs.values()):
            parser.error("Execution plan cannot be mixed with explicit input versions")
        inputs = residual.load_execution_plan(client, args.bucket, args.execution_plan_version)
    elif not args.consolidated_index_version or not args.bundle_version:
        parser.error("Provide an execution plan or both consolidated index and bundle versions")
    result = build(
        client,
        args.bucket,
        run_id=args.run_id,
        selectors=args.only,
        workers=args.workers,
        execute=args.execute,
        execution_plan_version=args.execution_plan_version,
        **inputs,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "assets"}, indent=2))


if __name__ == "__main__":
    main()
