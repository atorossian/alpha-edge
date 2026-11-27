"""Audit an immutable adjusted Silver candidate without rebuilding or activating it."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pandas as pd

from alpha_edge.market import consolidated_split_decisions_v3 as residual
from alpha_edge.market.audit_financial_universe_v3 import consolidated_receipt, load_consolidated
from alpha_edge.market.build_silver_adjusted_v3 import OUTPUT_ROOT
from alpha_edge.market.ingest_bronze_full_history import DEFAULT_BUCKET, DEFAULT_REGION, load_universe
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT,
    check_id,
    digest,
    json_bytes,
    put_immutable,
    read_object,
    verified_bytes,
    verified_frame,
)
from alpha_edge.market.split_basis_eligibility_v3 import load_approved_basis

AUDIT_ROOT = f"{ROOT}/silver/financial_acceptance_audits/v1"
POLICY = "financial_acceptance_candidate_v1"
OHLC = [f"candidate_{c}" for c in ("open", "high", "low", "close")]


def _number(value):
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


def arithmetic_audit(frame, calendar=None, findings=None):
    """Recompute observed returns and factors, never infer an absent return."""
    f = frame.set_index("session_date")
    if not f.index.is_unique or not f.index.is_monotonic_increasing:
        raise ValueError("Invalid candidate session ordering")
    trade = f.loc[f.candidate_expected_session.eq(True)]
    prev = trade.candidate_close.shift()
    close = trade.candidate_close
    multiplier = trade.candidate_share_multiplier
    cash = trade.candidate_cash
    checks = {
        "price_return": (close / prev - 1, trade.candidate_price_return),
        "total_return": (
            (close * multiplier + cash + trade.candidate_spin_actual) / prev - 1,
            trade.candidate_total_return,
        ),
        "reference_return": (
            (close * multiplier + cash + trade.candidate_spin_reference) / prev - 1,
            trade.candidate_reference_return,
        ),
        "adjustment_factor": (
            (1 - (cash + trade.candidate_spin_reference) / prev) / multiplier,
            trade.candidate_adjustment_factor,
        ),
    }
    failures = {}
    for name, (expected, actual) in checks.items():
        known = actual.notna()
        equal = np.isclose(actual, expected, rtol=1e-7, atol=1e-9, equal_nan=False)
        lower = actual.le(0) if name == "adjustment_factor" else actual.le(-1)
        bad = known & (~equal | ~np.isfinite(actual) | lower)
        failures[name] = set(trade.index[bad])
    valid = (
        np.isfinite(trade[OHLC]).all(axis=1)
        & trade[OHLC].gt(0).all(axis=1)
        & trade.candidate_high.ge(trade[["candidate_open", "candidate_close"]].max(axis=1))
        & trade.candidate_low.le(trade[["candidate_open", "candidate_close"]].min(axis=1))
    )
    failures["ohlc_return_pair"] = set(
        trade.index[trade.candidate_total_return.notna() & ~(valid & valid.shift(fill_value=False))]
    )
    failures["unexpected_session_return"] = set(
        f.index[~f.candidate_expected_session.eq(True) & f.candidate_total_return.notna()]
    )
    if calendar is not None:
        cal = calendar.set_index("session_date")
        manual = f.candidate_origin.eq("manual_observation")
        expected_session = cal.is_expected_session.reindex(f.index).eq(True) | manual
        if not expected_session.equals(f.candidate_expected_session.eq(True)):
            raise ValueError("Candidate expected sessions disagree with pinned calendar")
        calendar_ok = (
            cal.ohlc_eligible.reindex(trade.index).eq(True)
            | manual.loc[trade.index]
            | trade.correction_sources.str.contains("spin_off_quote_basis", regex=False)
        )
        valid &= calendar_ok
        dates = pd.Series(trade.index, index=trade.index)
        prior = dates.shift()
        missing = set(findings.loc[findings.issue.eq("missing_expected_session"), "session_date"].astype(str)) - set(
            f.index
        )
        gap = pd.Series(False, index=trade.index)
        for day in missing | set(cal.index[cal.calendar_exception_review.eq(True)]):
            gap |= prior.lt(day) & dates.gt(day)
        good = (
            valid
            & valid.shift(fill_value=False)
            & ~gap
            & ~trade.candidate_residual_boundary_review
            & ~trade.candidate_residual_cash_review
        )
        factor = checks["adjustment_factor"][0]
        factor_ok = np.isfinite(factor) & factor.gt(0)
        for name, (calculated, actual) in checks.items():
            required = good & np.isfinite(calculated)
            if name == "adjustment_factor":
                required &= calculated.gt(0)
            elif name != "price_return":
                required &= factor_ok & calculated.gt(-1)
            failures[f"{name}_availability"] = set(trade.index[actual.notna().ne(required)])
    all_bad = set().union(*failures.values())
    report = {
        "rows": len(f),
        "expected_sessions": len(trade),
        "observed_total_returns": int(trade.candidate_total_return.notna().sum()),
        "arithmetic_failures": {k: len(v) for k, v in failures.items()},
        "failure_sessions": sorted(all_bad),
        "historical_quarantined_rows": int(f.historical_quarantine.sum()),
        "potential_before": int(f.candidate_return_potential_before_split_basis_review.sum()),
        "potential_after": int(f.candidate_return_potential_after_split_basis_review.sum()),
    }
    return f, trade, all_bad, report


def _event_result(f, trade, day, entry, group, bad):
    row = f.loc[day]
    reasons = []
    status = entry.get("financial_status", "approved_price_basis_only")
    if group == "residual" and status == "quarantined":
        outcome = "retain_documented_quarantine"
    elif group == "residual" and not row.candidate_residual_source_verified:
        outcome = "source_verification_required"
    elif group == "residual" and status in {"cash_review_pending", "economic_review_pending"}:
        outcome = status
    else:
        pos = trade.index.get_loc(day) if day in trade.index else 0
        prev = trade.iloc[pos - 1] if pos else None
        if prev is None:
            reasons.append("missing_expected_predecessor")
        if day in bad:
            reasons.append("candidate_arithmetic_failure")
        if bool(row.candidate_residual_boundary_review):
            reasons.append("boundary_review")
        if bool(row.candidate_residual_cash_review):
            reasons.append("cash_basis_review")
        if entry.get("source_entry", {}).get("session_date_reconciliation_required"):
            reasons.append("issuer_trading_date_review")
        if (
            bool(row.candidate_large_return_review)
            or bool(row.candidate_split_basis_large_return_review)
            or entry.get("large_return_review_dates")
        ):
            reasons.append("large_return_review")
        for field in ("candidate_cash", "candidate_spin_actual", "candidate_spin_reference"):
            value = _number(row[field])
            if value is None or value != 0:
                reasons.append("cash_or_non_cash_distribution")
                break
        if _number(row.candidate_share_multiplier) != 1:
            reasons.append("share_multiplier_requires_economic_review")
        if group == "residual" and status == "close_only_pending_integration":
            if not row.candidate_close_only_basis_reverified:
                outcome = "price_basis_reverification_required"
            elif prev is None or _number(row.candidate_close_only_price_return) is None:
                outcome = "close_only_return_unavailable"
            else:
                p, c = _number(prev.close), _number(row.close)
                actual = _number(row.candidate_close_only_price_return)
                if (
                    p is None
                    or c is None
                    or p <= 0
                    or c <= 0
                    or not math.isclose(actual, c / p - 1, abs_tol=1e-9, rel_tol=1e-7)
                ):
                    reasons.append("close_only_return_mismatch")
                outcome = "close_only_review_required" if reasons else "close_only_price_return_pass"
        elif (
            group == "residual"
            and status == "approved_price_basis_pending_integration"
            and not row.candidate_residual_full_price_basis_reverified
        ):
            outcome = "price_basis_reverification_required"
        elif (
            group == "residual"
            and status in {"reconstruction_pending", "candidate_built_pending_validation"}
            and not row.candidate_residual_reconstruction_applied
        ):
            outcome = "reconstruction_required"
        else:
            if _number(row.candidate_total_return) is None:
                reasons.append("total_return_unavailable")
            if group == "residual" and row.candidate_residual_reconstruction_applied:
                window = trade.iloc[max(0, pos - 4) : pos + 5] if prev is not None else trade.iloc[0:0]
                if window.candidate_residual_cash_review.any() or window.candidate_residual_boundary_review.any():
                    reasons.append("reconstruction_window_review")
                # A successful arithmetic check does not prove the underlying repair or its whole window.
                outcome = "reconstruction_review_required" if reasons else "reconstruction_arithmetic_pass"
            else:
                if prev is not None and not reasons:
                    p, c, r = _number(prev.adj_close), _number(row.adj_close), _number(row.candidate_total_return)
                    if p is None or c is None or p <= 0 or c <= 0 or r is None:
                        reasons.append("provider_adjusted_return_unavailable")
                    elif not math.isclose(c / p - 1, r, rel_tol=0, abs_tol=0.005):
                        reasons.append("provider_adjusted_return_disagreement")
                outcome = "price_basis_return_review_required" if reasons else "price_basis_and_return_pass"
    return {
        "asset_id": str(row.asset_id),
        "symbol": entry["symbol"],
        "date": day,
        "population": group,
        "decision_status": status,
        "outcome": outcome,
        "reasons": sorted(set(reasons)),
        "source_ratio": entry["ratio"],
        "candidate_total_return": _number(row.candidate_total_return),
        "candidate_close_only_price_return": _number(row.candidate_close_only_price_return),
        "historical_quarantine": bool(row.historical_quarantine),
        "economic_acceptance": False,
        "quarantine_release_eligible": False,
    }


def assess(frame, raw, standard, decisions, *, aid, symbol, sha, residual_version, calendar=None, findings=None):
    residual.validate_candidate(frame, raw, decisions)
    f, trade, bad, summary = arithmetic_audit(frame, calendar, findings)
    if set(f.index[f.candidate_split_basis_approved.eq(True)]) != set(standard):
        raise ValueError(f"Standard event population mismatch: {aid}")
    residual.check_source(raw.set_index("session_date"), decisions, asset_id=aid, symbol=symbol, sha=sha)
    results = []
    for group, rows in (("standard", standard), ("residual", decisions)):
        for day, entry in sorted(rows.items()):
            if (
                entry["bronze_sha256"] != sha
                or entry["symbol"] != symbol
                or day not in f.index
                or not math.isclose(float(f.at[day, "stock_splits"]), entry["ratio"], rel_tol=1e-7)
            ):
                raise ValueError(f"Financial audit source pin mismatch: {aid} {day}")
            if group == "standard" and f.at[day, "candidate_split_basis_approval_version"] != entry["approval_version"]:
                raise ValueError("Standard approval version changed")
            if group == "residual" and f.at[day, "candidate_residual_decision_version"] != residual_version:
                raise ValueError("Residual approval version changed")
            results.append(_event_result(f, trade, day, entry, group, bad))
    return summary, results


def audit(s3, bucket, *, candidate_run, manifest_sha256, audit_run, workers=4, execute=False):
    check_id(candidate_run)
    check_id(audit_run)
    if not 1 <= workers <= 16:
        raise ValueError("Invalid worker count")
    manifest_ref = {"key": f"{OUTPUT_ROOT}/runs/{candidate_run}/manifest.json", "sha256": manifest_sha256}
    manifest = json.loads(verified_bytes(s3, bucket, manifest_ref))
    if (
        manifest.get("schema") != "alpha_edge_silver_adjusted_candidate_v1"
        or manifest["run_id"] != candidate_run
        or manifest.get("scope") != "full_universe"
        or manifest["production_eligible"] is not False
        or manifest["quarantine_released"] is not False
    ):
        raise ValueError("Expected completed full-universe candidate manifest")
    plan = residual.load_execution_plan(s3, bucket, manifest["execution_plan_version"])
    index = load_consolidated(s3, bucket, plan["consolidated_version"])
    active, _ = load_universe(s3, bucket, index["source_fingerprint"])
    by_id = {r["asset_id"]: r for r in active}
    reports = {r["asset_id"]: r for r in manifest["assets"]}
    if (
        set(by_id) != set(reports)
        or set(by_id) != set(index["assets"])
        or len(reports) != len(manifest["assets"])
        or manifest["consolidated_index_version"] != plan["consolidated_version"]
        or manifest["registries"]["integration"] != plan["bundle_version"]
        or manifest["split_basis_approvals"]["standard"] != plan["split_basis_approvals"]
        or manifest["split_basis_approvals"]["deterministic"] != plan["deterministic_basis_approvals"]
        or manifest["residual_decisions"]["version"] != plan["residual_version"]
    ):
        raise ValueError("Candidate manifest does not match its execution plan")
    standard = load_approved_basis(
        s3,
        bucket,
        plan["split_basis_approvals"],
        plan["deterministic_basis_approvals"],
        plan["expected_split_approvals"],
    )
    decisions = residual.load(
        s3,
        bucket,
        plan["residual_version"],
        consolidated_version=plan["consolidated_version"],
        bundle_version=plan["bundle_version"],
        expected_count=plan["expected_residual_count"],
    )
    if not (set(standard) | set(decisions)) <= set(by_id):
        raise ValueError("Event asset outside candidate universe")
    expected = {(a, d) for a, rows in standard.items() for d in rows}
    residual_ids = {(a, d) for a, rows in decisions.items() for d in rows}
    if expected & residual_ids:
        raise ValueError("Duplicate event identity")
    expected |= residual_ids
    scope = f"{AUDIT_ROOT}/runs/{audit_run}"
    inputs = {"schema": "alpha_edge_financial_acceptance_inputs_v1", "policy": POLICY, "candidate": manifest_ref}
    fingerprint = digest(json_bytes(inputs))
    if execute:
        put_immutable(
            s3, bucket, f"{scope}/inputs.json", json_bytes(inputs), content_type="application/json", scope=AUDIT_ROOT
        )

    def process(aid):
        ref = reports[aid]["output"]
        if (
            ref["key"]
            != f"{OUTPUT_ROOT}/runs/{candidate_run}/assets/{digest(aid.encode())}/prices-{ref['sha256']}.parquet"
        ):
            raise ValueError("Invalid candidate asset reference")
        key = f"{scope}/assets/{digest(aid.encode())}.json"
        existing = read_object(s3, bucket, key, optional=True) if execute else None
        if existing:
            raw, metadata = existing
            out = json.loads(raw)
            if (
                metadata.get("sha256") != digest(raw)
                or out["input_fingerprint"] != fingerprint
                or out["asset_id"] != aid
                or out["candidate"] != ref
            ):
                raise ValueError("Audit checkpoint mismatch")
            return out, True
        receipt = consolidated_receipt(s3, bucket, index, by_id[aid])
        bars = verified_frame(s3, bucket, receipt["data"])
        frame = verified_frame(s3, bucket, ref)
        cal_refs = index["assets"][aid]["outputs"]
        calendar = verified_frame(s3, bucket, cal_refs["candidate"])
        findings = verified_frame(s3, bucket, cal_refs["findings"])
        if len(frame) != reports[aid]["candidate_rows"] or len(bars) != reports[aid]["source_rows"]:
            raise ValueError("Candidate manifest row counts disagree")
        summary, events = assess(
            frame,
            bars,
            standard.get(aid, {}),
            decisions.get(aid, {}),
            aid=aid,
            symbol=by_id[aid]["yahoo_ticker"],
            sha=receipt["data"]["sha256"],
            residual_version=plan["residual_version"],
            calendar=calendar,
            findings=findings,
        )
        out = {
            "asset_id": aid,
            "input_fingerprint": fingerprint,
            "candidate": ref,
            "source": receipt["data"],
            "summary": summary,
            "events": events,
        }
        if execute:
            put_immutable(s3, bucket, key, json_bytes(out), content_type="application/json", scope=AUDIT_ROOT)
        return out, False

    assets, events, resumed = [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (out, reused) in enumerate(pool.map(process, sorted(by_id)), 1):
            events.extend(out["events"])
            assets.append(
                {"asset_id": out["asset_id"], "summary": out["summary"], "audit_sha256": digest(json_bytes(out))}
            )
            resumed += int(reused)
            if n % 50 == 0 or n == len(by_id):
                print(f"[financial acceptance audit] {n}/{len(by_id)}; resumed={resumed}", file=sys.stderr, flush=True)
    actual = {(e["asset_id"], e["date"]) for e in events}
    if len(actual) != len(events) or actual != expected:
        raise ValueError("Audited event population mismatch")
    stats = {
        "assets": len(assets),
        "events": len(events),
        "outcomes": dict(Counter(e["outcome"] for e in events)),
        "review_reasons": dict(Counter(reason for e in events for reason in e["reasons"])),
        "arithmetic_failure_types": dict(
            sum((Counter(a["summary"]["arithmetic_failures"]) for a in assets), Counter())
        ),
        "by_population": {
            p: dict(Counter(e["outcome"] for e in events if e["population"] == p)) for p in ("standard", "residual")
        },
        "assets_with_arithmetic_failures": sum(bool(a["summary"]["failure_sessions"]) for a in assets),
        "arithmetic_failure_sessions": sum(len(a["summary"]["failure_sessions"]) for a in assets),
        "potential_before": sum(a["summary"]["potential_before"] for a in assets),
        "potential_after": sum(a["summary"]["potential_after"] for a in assets),
    }
    for name, observed in (
        ("assets", len(assets)),
        ("candidate_rows", sum(a["summary"]["rows"] for a in assets)),
        ("candidate_returns", sum(a["summary"]["observed_total_returns"] for a in assets)),
        ("quarantined_rows", sum(a["summary"]["historical_quarantined_rows"] for a in assets)),
        ("potential_returns_before", stats["potential_before"]),
        ("potential_returns_after", stats["potential_after"]),
    ):
        if manifest["statistics"][name] != observed:
            raise ValueError(f"Candidate aggregate disagrees with manifest: {name}")
    result = {
        "schema": "alpha_edge_financial_acceptance_audit_v1",
        "audit_run": audit_run,
        "inputs": inputs,
        "statistics": stats,
        "production_eligible": False,
        "quarantine_released": False,
        "financial_acceptance_complete": False,
    }
    if execute:
        data = json_bytes({"schema": "alpha_edge_financial_acceptance_events_v1", "events": events})
        result["events"] = put_immutable(
            s3,
            bucket,
            f"{scope}/objects/events-{digest(data)}.json",
            data,
            content_type="application/json",
            scope=AUDIT_ROOT,
        )
        result["assets"] = assets
        put_immutable(
            s3, bucket, f"{scope}/manifest.json", json_bytes(result), content_type="application/json", scope=AUDIT_ROOT
        )
    return {k: v for k, v in result.items() if k != "assets"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--candidate-run", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--audit-run", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            audit(
                boto3.client("s3", region_name=a.region),
                a.bucket,
                candidate_run=a.candidate_run,
                manifest_sha256=a.manifest_sha256,
                audit_run=a.audit_run,
                workers=a.workers,
                execute=a.execute,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
