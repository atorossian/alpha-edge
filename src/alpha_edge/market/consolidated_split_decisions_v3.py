"""Load approved residual decisions and retain their distinct financial gates.

Approval of a disposition is not price validation. All market data is read
from checksum-pinned Bronze objects; no source observation is overwritten.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import date

import numpy as np
import pandas as pd

from alpha_edge.market.lakehouse_v3.storage import ROOT, digest, verified_bytes

REGISTRY = f"{ROOT}/config/consolidated_split_decision_registries/v1/objects"
PLAN_ROOT = f"{ROOT}/config/consolidated_financial_execution_plans/v1/objects"
STATUSES = {
    "reconstruction_pending",
    "approved_price_basis_pending_integration",
    "close_only_pending_integration",
    "cash_review_pending",
    "quarantined",
    "economic_review_pending",
    "candidate_built_pending_validation",
}


def _candidate_only(doc):
    if doc.get("production_eligible") is not False or doc.get("quarantine_released") is not False:
        raise ValueError("Expected candidate-only, quarantine-preserving evidence")


def _identity(row, doc=None):
    return row.get("asset_id", (doc or {}).get("asset_id")), row.get("date", row.get("yahoo_date"))


def load_execution_plan(s3, bucket, version):
    """Resolve an explicit, immutable candidate execution plan."""
    doc = json.loads(verified_bytes(s3, bucket, {"key": f"{PLAN_ROOT}/{version}.json", "sha256": version}))
    _candidate_only(doc)
    if doc.get("schema") != "alpha_edge_consolidated_financial_execution_plan_v1":
        raise ValueError("Invalid consolidated financial execution plan")
    args = doc["builder_arguments"]
    expected = {
        "consolidated_version",
        "bundle_version",
        "split_basis_approvals",
        "deterministic_basis_approvals",
        "expected_split_approvals",
        "residual_version",
        "expected_residual_count",
        "eodhd_run_id",
    }
    if (
        set(args) != expected
        or args["expected_split_approvals"] + args["expected_residual_count"] != doc["total_cases"]
    ):
        raise ValueError("Execution plan input population mismatch")
    return args


def load(s3, bucket, version, *, consolidated_version, bundle_version, expected_count):
    """Verify the registry and its original approval references, not free text."""
    ref = {"key": f"{REGISTRY}/{version}.json", "sha256": version}
    doc = json.loads(verified_bytes(s3, bucket, ref))
    _candidate_only(doc)
    if (
        doc.get("schema") != "alpha_edge_consolidated_split_decisions_v1"
        or doc.get("status") != "approved_dispositions_pending_financial_validation"
        or doc["consolidated_index"]["sha256"] != consolidated_version
        or doc["source_refs"]["bundle"]["sha256"] != bundle_version
    ):
        raise ValueError("Consolidated split registry input mismatch")
    rows = doc["entries"]
    if (
        len(rows) != expected_count
        or doc["approved_count"] != expected_count
        or doc["baseline_residual_count"] != expected_count
    ):
        raise ValueError("Unexpected consolidated decision count")
    if Counter(e["financial_status"] for e in rows) != doc["financial_status_counts"]:
        raise ValueError("Consolidated financial status counts disagree")
    by_asset, refs, seen = defaultdict(dict), {}, set()
    for row in rows:
        _candidate_only(row)
        aid, day = _identity(row)
        if (
            not aid
            or date.fromisoformat(day).isoformat() != day
            or (aid, day) in seen
            or row.get("decision_approved") is not True
            or row["financial_status"] not in STATUSES
        ):
            raise ValueError("Invalid or duplicate consolidated decision")
        seen.add((aid, day))
        d = row["decision"]
        direct = d.get("previous_evidence", d)
        if _identity(direct) != (aid, day) or direct["symbol"] != row["symbol"]:
            raise ValueError("Consolidated decision identity mismatch")
        source_doc = None
        if "source" in direct:
            pin = direct["source"]
            token = (pin["key"], pin["sha256"])
            if token not in refs:
                refs[token] = json.loads(verified_bytes(s3, bucket, pin))
                _candidate_only(refs[token])
            source_doc = refs[token]
            matches = [e for e in source_doc["entries"] if _identity(e, source_doc) == (aid, day)]
            if len(matches) != 1:
                raise ValueError("Original decision reference does not match exactly one event")
            source = matches[0]
            if not math.isclose(float(source["ratio"]), float(direct["ratio"]), rel_tol=1e-7):
                raise ValueError("Original decision ratio mismatch")
        else:
            source = direct
        ratio = float(direct["ratio"])
        if not math.isfinite(ratio) or ratio <= 0 or ratio == 1:
            raise ValueError("Invalid consolidated source ratio")
        sha = source.get("bronze_sha256") or (source_doc or {}).get("yahoo", {}).get("sha256")
        eod = source.get("eodhd_sha256") or (source_doc or {}).get("eodhd", {}).get("sha256")
        by_asset[aid][day] = {
            **row,
            "ratio": ratio,
            "bronze_sha256": sha,
            "eodhd_sha256": eod,
            "source_doc": source_doc,
            "source_entry": source,
        }
    return dict(by_asset)


def check_source(raw, decisions, *, asset_id, symbol, sha):
    """Never let a stale source hash or changed action silently pass."""
    for day, e in decisions.items():
        if (
            e["asset_id"] != asset_id
            or e["symbol"] != symbol
            or day not in raw.index
            or not math.isclose(float(raw.at[day, "stock_splits"]), e["ratio"], rel_tol=1e-7)
            or (e["bronze_sha256"] is not None and e["bronze_sha256"] != sha)
        ):
            raise ValueError(f"Consolidated split source mismatch: {asset_id} {day}")


def read_eodhd(s3, bucket, asset_id, run_id, sha):
    """Verify the exact original provider bundle and response digests."""
    from alpha_edge.market.eodhd_source_v3 import verify_bundle
    from alpha_edge.market.lakehouse_v3.storage import check_id

    check_id(run_id)
    key = f"{ROOT}/bronze/eodhd_full_history/v1/runs/{run_id}/assets/{digest(asset_id.encode())}/objects/{sha}.json"
    blob = verified_bytes(s3, bucket, {"key": key, "sha256": sha})
    doc = json.loads(blob)
    record = {
        "asset_id": asset_id,
        "provider_symbol": doc["provider_symbol"],
        "mapping_version": doc["mapping_version"],
    }
    if "api_base" in doc:
        record["api_base"] = doc["api_base"]
    verify_bundle(blob, record, doc["end_exclusive"])
    return {name: json.loads(x["raw_json"]) for name, x in doc["responses"].items()}


def close_evidence(raw, calendar, findings, decisions, responses):
    """Repeat the original daily-chain tests against raw evidence.

    A passing window is local evidence only, not a release of a historical
    prefix or permission to reuse the disputed OHLC/volume fields.
    """
    from alpha_edge.market.audit_matching_split_basis_v3 import FIELDS, TOL, positive_matrix, prepare
    from alpha_edge.market.audit_provider_reconciliation_v3 import split_ratio
    from alpha_edge.market.reconciliation_hypotheses_v3 import (
        CANDIDATE,
        CLOSE_CANDIDATE,
        analyze_window,
        calendar_windows,
    )

    y, e = prepare(raw.reset_index(drop=True), responses)
    missing = tuple(findings.loc[findings.issue.eq("missing_expected_session"), "session_date"].astype(str))
    out = {}
    for day, row in decisions.items():
        close_only = row["financial_status"] == "close_only_pending_integration"
        if not close_only and row["financial_status"] != "approved_price_basis_pending_integration":
            continue
        windows = calendar_windows(y, e, calendar, responses, day, missing)
        checks = [
            analyze_window(
                yy,
                ee,
                responses,
                {"date": day, "ratio": row["ratio"]},
                {"date_status": "exact_date", "value_status": "ratio_match"},
                size,
                close_only=close_only,
            )
            for size, yy, ee in windows
        ]
        if checks and all(x["status"] == (CLOSE_CANDIDATE if close_only else CANDIDATE) for x in checks):
            # The helper checks all split and cash observations including the target.
            out[day] = checks[-1]
            continue
        # Some issuer-verified Yahoo splits have no observed scale step because
        # both providers already normalized the price basis. Require stable
        # *all-field* scales and adjusted returns, and no other nearby action.
        if not close_only and windows:

            def normalized(yy, ee):
                if yy.dividends.ne(0).any() or yy.stock_splits.drop(day).ne(0).any():
                    return False
                if any(yy.index[0] <= x["date"] <= yy.index[-1] for x in responses["div"]):
                    return False
                if any(yy.index[0] <= x["date"] <= yy.index[-1] and x["date"] != day for x in responses["splits"]):
                    return False
                events = [x for x in responses["splits"] if x["date"] == day]
                if len(events) > 1 or (
                    events and not math.isclose(split_ratio(events[0]["split"]) or 0, row["ratio"], rel_tol=1e-4)
                ):
                    return False
                a, b = positive_matrix(yy, FIELDS), positive_matrix(ee, FIELDS)
                ya, ea = positive_matrix(yy, ["adj_close"]), positive_matrix(ee, ["adjusted_close"])
                if any(x is None for x in (a, b, ya, ea)):
                    return False
                levels = b / a
                if np.max(np.abs(levels / np.median(levels) - 1)) > TOL:
                    return False
                return bool(np.max(np.abs((ya[1:] / ya[:-1]) / (ea[1:] / ea[:-1]) - 1)) <= TOL)

            if all(normalized(yy, ee) for _, yy, ee in windows):
                out[day] = {"status": "already_normalized_price_basis_reverified", "daily_checks": []}
                continue
        out[day] = {"status": "basis_reverification_failed", "daily_checks": []}
    return out


def apply_existing_repairs(frame, raw, decisions, responses, *, source_key, sha):
    """Reuse explicitly approved price repairs; refuse overlapping corrections.

    Only typed existing plans are executable. A research disposition (including
    suggested issuer terms) is not a price transformation specification.
    """
    from alpha_edge.market.build_asset_price_reconstruction_v3 import reconstruct

    frame["candidate_residual_reconstruction_applied"] = False
    frame["candidate_residual_boundary_review"] = False
    frame["candidate_residual_cash_review"] = False
    frame["candidate_residual_volume_review"] = False
    frame["candidate_residual_split_event"] = frame.stock_splits
    plans = {}
    for day, entry in decisions.items():
        doc = entry["source_doc"]
        if doc and doc.get("schema") == "alpha_edge_asset_price_reconstruction_approval_v1":
            if doc["yahoo"] != {"key": source_key, "sha256": sha}:
                raise ValueError("Existing reconstruction source pin mismatch")
            plans.setdefault(entry["decision"]["source"]["sha256"], (doc, []))[1].append(day)
    for doc, days in plans.values():
        if {e["date"] for e in doc["entries"]} != set(days):
            raise ValueError("Existing reconstruction plan not fully represented in decision inventory")
        if responses is None:
            raise ValueError("Missing independent reconstruction evidence")
        repaired, _ = reconstruct(raw.reset_index(drop=True), responses, doc)
        repaired = repaired.set_index("session_date")
        fields = ("open", "high", "low", "close", "volume")
        touched = pd.Series(False, index=raw.index)
        for c in fields:
            original, new = raw[c], repaired[f"candidate_{c}"]
            touched |= ~(original.eq(new) | (original.isna() & new.isna()))
        split_changed = ~raw.stock_splits.eq(repaired.candidate_split_event)
        touched |= split_changed | repaired.candidate_boundary_review | repaired.candidate_cash_review
        if "candidate_cash_dividend" in repaired:
            cash = repaired.candidate_cash_dividend
            touched |= ~(raw.dividends.eq(cash) | (raw.dividends.isna() & cash.isna()))
        ix = raw.index[touched]
        # Conflicts are explicitly retained, not combined from unrelated bases.
        if frame.loc[ix, "correction_sources"].ne("").any():
            continue
        for c in fields:
            frame.loc[ix, f"candidate_{c}"] = repaired.loc[ix, f"candidate_{c}"]
        frame.loc[ix, "candidate_cash"] = repaired.loc[ix, "candidate_cash_dividend"]
        frame.loc[ix, "candidate_residual_split_event"] = repaired.loc[ix, "candidate_split_event"]
        for name in ("boundary", "cash", "volume"):
            frame.loc[ix, f"candidate_residual_{name}_review"] = repaired.loc[ix, f"candidate_{name}_review"]
        frame.loc[ix, "candidate_provisional"] = True
        frame.loc[ix, "correction_sources"] = "approved_asset_price_reconstruction"
        frame.loc[days, "candidate_residual_reconstruction_applied"] = True
    for day, entry in decisions.items():
        terms = entry["decision"].get("proposed_decision", {})
        if terms.get("decision") != "single_treasury_share_distribution":
            continue
        delivery = terms["duplicate_reported_cash_date"]
        amount = terms["duplicate_reported_cash_adjusted"]
        # The entitlement is already present as a normalized-share event.
        # Suppress only the explicitly pinned duplicate on the delivery date.
        if (
            entry["bronze_sha256"] != sha
            or delivery not in raw.index
            or not math.isclose(float(raw.at[delivery, "dividends"]), amount, rel_tol=1e-7)
            or not math.isclose(float(raw.at[day, "stock_splits"]), terms["ratio"], rel_tol=1e-7)
        ):
            raise ValueError("Treasury-share duplicate source mismatch")
        if frame.loc[[day, delivery], "correction_sources"].ne("").any():
            continue
        frame.at[delivery, "candidate_cash"] = 0.0
        frame.at[delivery, "candidate_provisional"] = True
        frame.at[delivery, "correction_sources"] = "approved_treasury_share_delivery_duplicate"
        frame.at[day, "candidate_residual_reconstruction_applied"] = True


def annotate(frame, raw, decisions, evidence, *, version):
    """Expose independent local close diagnostics and unresolved decisions.

    Never alter the existing OHLC/return columns or historical quarantine.
    The close-only price return is separate from total return and never bridges
    a gap, conflicting repair, or an unvalidated corporate-action session.
    """
    frame["candidate_residual_decision_status"] = ""
    frame["candidate_residual_decision_version"] = ""
    frame["candidate_residual_source_verified"] = False
    frame["candidate_residual_validation_status"] = ""
    frame["candidate_residual_full_price_basis_reverified"] = False
    frame["candidate_close_only_basis_reverified"] = False
    frame["candidate_close_only_price_return"] = np.nan
    frame["candidate_close_only_return_review"] = False
    frame["candidate_residual_financial_blocker"] = False
    for day, row in decisions.items():
        frame.at[day, "candidate_residual_decision_status"] = row["financial_status"]
        frame.at[day, "candidate_residual_decision_version"] = version
        frame.at[day, "candidate_residual_source_verified"] = row["bronze_sha256"] is not None
        frame.at[day, "candidate_residual_financial_blocker"] = True
        result = evidence.get(day, {})
        state = "held_pending_financial_validation"
        if row["bronze_sha256"] is None:
            state = "held_missing_source_pin"
        elif (
            "candidate_residual_reconstruction_applied" in frame
            and frame.at[day, "candidate_residual_reconstruction_applied"]
        ):
            state = "reconstruction_applied_pending_validation"
        elif result:
            state = result["status"]
        frame.at[day, "candidate_residual_validation_status"] = state
        if result.get("status") in {"advanced_session_factor_candidate", "already_normalized_price_basis_reverified"}:
            frame.at[day, "candidate_residual_full_price_basis_reverified"] = True
        if result.get("status") != "close_basis_only_candidate":
            continue
        frame.at[day, "candidate_close_only_basis_reverified"] = True
        source = row["source_entry"]
        if source.get("session_date_reconciliation_required") or result.get("cash_raw_basis_review_required"):
            continue
        checks = result["daily_checks"]
        for check in checks:
            d = check["date"]
            if d not in frame.index or d not in raw.index or (d in decisions and d != day):
                continue
            if d in decisions and decisions[d]["financial_status"] != "close_only_pending_integration":
                continue
            if check["yahoo_cash"] or check["cash_evidence"] != "none":
                continue
            # Verify the exact pair from the audited window, not the nearest
            # arbitrary observation. No candidate correction may overlap it.
            expected = frame.index[frame.candidate_expected_session]
            pos = expected.get_loc(d) if d in expected else 0
            if not pos:
                continue
            prev = expected[pos - 1]
            if prev not in raw.index or frame.loc[[prev, d], "correction_sources"].ne("").any():
                continue
            if not all(math.isfinite(float(raw.at[x, "close"])) and float(raw.at[x, "close"]) > 0 for x in (prev, d)):
                continue
            if not math.isclose(
                float(raw.at[d, "close"]) / float(raw.at[prev, "close"]) - 1,
                check["yahoo_adjusted_return"],
                rel_tol=0,
                abs_tol=0.005,
            ):
                continue
            frame.at[d, "candidate_close_only_price_return"] = (
                float(raw.at[d, "close"]) / float(raw.at[prev, "close"]) - 1
            )
            frame.at[d, "candidate_close_only_return_review"] = True
    return len(decisions)


def validate_candidate(frame, bars, decisions):
    """Quantitative invariants for the candidate, not a production acceptance."""
    f = frame.set_index("session_date")
    raw = bars.set_index("session_date")
    if not f.index.is_unique or not raw.index.is_unique or not set(raw.index) <= set(f.index):
        raise ValueError("Financial candidate lost or duplicated source sessions")
    pd.testing.assert_frame_equal(f.loc[raw.index, raw.columns], raw, check_dtype=False)
    if f.production_eligible.any():
        raise ValueError("Unexpected production eligibility")
    trade = f.loc[f.candidate_expected_session]
    adjusted = trade.candidate_segment_adjusted_close
    known = adjusted.notna()
    if not np.isfinite(adjusted[known]).all() or not adjusted[known].gt(0).all():
        raise ValueError("Invalid segmented adjusted prices")
    segment = trade.candidate_segment_id
    same = segment.notna() & segment.eq(segment.shift()).fillna(False)
    ratio = adjusted / adjusted.shift() - 1
    if not np.allclose(
        ratio[same].to_numpy(dtype=float),
        trade.candidate_total_return[same].to_numpy(dtype=float),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError("Adjusted return chain disagrees with candidate returns")
    observed = set(f.index[f.candidate_residual_decision_status.ne("")])
    if observed != set(decisions):
        raise ValueError("Residual event accounting mismatch")
    if not f.loc[list(observed), "candidate_residual_financial_blocker"].all():
        raise ValueError("Residual financial gate silently released")
    if not f.loc[f.candidate_close_only_price_return.notna(), "candidate_close_only_return_review"].all():
        raise ValueError("Unreviewed close-only return")
    return {
        "residual_decision_events": len(observed),
        "residual_source_verified_events": int(f.candidate_residual_source_verified.sum()),
        "residual_reconstruction_applied_events": int(f.candidate_residual_reconstruction_applied.sum()),
        "residual_close_reverified_events": int(f.candidate_close_only_basis_reverified.sum()),
        "residual_full_price_basis_reverified_events": int(f.candidate_residual_full_price_basis_reverified.sum()),
        "residual_close_return_rows": int(f.candidate_close_only_price_return.notna().sum()),
        "residual_financial_blocker_events": int(f.candidate_residual_financial_blocker.sum()),
    }
