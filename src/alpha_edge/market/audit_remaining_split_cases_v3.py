"""Consolidate residual split cases and cross-reference existing financial evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date

import boto3
import pandas as pd

from alpha_edge.market.approve_split_basis_v3 import approve
from alpha_edge.market.build_silver_adjusted_v3 import load_bundle
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT, check_id, digest, json_bytes, parquet_bytes, put_immutable, read_json,
    verified_bytes, verified_frame,
)

SCOPE = f"{ROOT}/silver/remaining_split_cases/v1"
MODES = ({}, {"fractional": True}, {"small_fraction": True}, {"irregular": True}, {"yahoo_only": True})
STATUSES = ("provider_normalized_candidate", "fractional_price_basis_candidate",
            "small_fraction_price_basis_candidate", "irregular_price_basis_candidate",
            "yahoo_only_implicit_price_basis_candidate")


def identity(row):
    return row["asset_id"], row["date"]


def consolidate(screens, unmatched, batches):
    """Partition by asset/date; latest targeted result wins, never count a case twice."""
    sources = {identity(r): r for r in screens}
    if len(sources) != len(screens):
        raise ValueError("Duplicate screened split")
    latest = {}

    def overlay(rows, *, unmatched=False):
        seen = set()
        for row in rows:
            key = identity(row)
            if key not in sources or key in seen:
                raise ValueError("Missing or duplicate split identity")
            seen.add(key)
            raw = sources[key]
            sha = row.get("yahoo_sha256", row.get("bronze_sha256"))
            if sha != raw["bronze_sha256"] or not math.isclose(float(row["ratio"]), float(raw["ratio"]), rel_tol=1e-7):
                raise ValueError("Split evidence source mismatch")
            if key in latest and latest[key]["status"] in STATUSES:
                raise ValueError("Attempt to replace validated candidate")
            latest[key] = {**raw, **row, "status": row["reason"] if unmatched else row["status"]}

    overlay(unmatched, unmatched=True)
    for rows in batches:
        overlay(rows)
    if set(latest) != set(sources):
        raise ValueError("Incomplete split case inventory")
    return latest


def evidence_index(bundle):
    """Index explicit registry dates and bounded event windows, not entire asset histories."""
    index = defaultdict(list)

    def add(entry, family, days=(), window=None, role="cash_or_action"):
        sha = entry.get("source_sha256", entry.get("bronze_sha256"))
        # An unpinned prior finding is context, never a verified source match.
        for day in sorted(set(d for d in days if d)):
            index[entry["asset_id"]].append({"family": family, "date": day, "role": role,
                                             "sha256": sha, "status": entry.get("basis_status", "registry_evidence")})
        if window:
            index[entry["asset_id"]].append({"family": family, "start": window[0], "end": window[1],
                                             "role": "event_window", "sha256": sha,
                                             "status": entry.get("basis_status", "registry_evidence")})

    for e in bundle["currency"]["entries"]:
        add(e, "currency", [e["effective_date"]])
    for e in bundle["duplicates"]["entries"]:
        add(e, "duplicates", [e["effective_date"], e["retained_date"], e["issuer_ex_date"]])
    for e in bundle["duplicates"].get("legitimate_pairs", []):
        add(e, "legitimate_pairs", [e["first_date"], e["second_date"]])
    for name in ("entries", "additional_currency_candidates", "event_reconstruction_candidates", "unresolved_events"):
        for e in bundle["reconciliation"][name]:
            add(e, "reconciliation_" + name, [e["effective_date"]])
    for doc in bundle["events"]:
        for e in doc["entries"]:
            add(e, "event_reconstruction", [e["issuer_ex_date"], *(x["date"] for x in e["source_events"])],
                (e["window_start"], e["window_end"]))
            for factor in e["basis_factors"]:
                add({**e, "basis_status": factor["status"]}, "event_basis_factor",
                    [factor["effective_date"]], role=factor["kind"])
    for doc in bundle["suppressions"]:
        for e in doc["entries"]:
            add(e, "event_suppression", [e["effective_date"], *(x["date"] for x in e["retained_source_events"])])
    for e in bundle["missing"]["entries"]:
        add(e, "missing_session", [e["source_dividend_date"], e["session_date"]])
    for e in bundle["complex"]["entries"]:
        add(e, "complex_action", [e["source_date"], e["target"], *(x["date"] for x in e["actions"])])
    share = bundle["share"]
    for e in share["events"]:
        add(share, "share_basis", [e["effective_date"]], role="share_ratio")
    for e in share.get("cash_reconciliations", []):
        add(share, "share_cash_reconciliation", [e["effective_date"]])
    spin = bundle["spin"]
    add(spin, "spin_off", [spin["spin_off_event"]["effective_date"]], role="noncash_distribution")
    add(spin, "spin_off_cash", [spin["cash_event"]["ex_date"]])
    for e in bundle["quarantine"]["entries"]:
        add(e, "historical_quarantine", [e["effective_date"]], role=e["reason"])
    return index


def match_evidence(row, index):
    day = date.fromisoformat(row["date"])
    matches = []
    for item in index.get(row["asset_id"], []):
        if "date" in item:
            offset = (date.fromisoformat(item["date"]) - day).days
            if abs(offset) > 4:
                continue
            relation = "exact_date" if offset == 0 else "near_date"
        elif item["start"] <= row["date"] <= item["end"]:
            offset, relation = None, "within_event_window"
        else:
            continue
        matches.append({**item, "relation": relation, "offset_days": offset,
                        "source_matches": item["sha256"] == row["bronze_sha256"]})
    return matches


def classify_reason(status):
    if status in {"same_day_cash_event", "nearby_eodhd_cash_event", "nearby_yahoo_cash_event"}:
        return "cash_and_share"
    if status in {"outside_price_history_overlap", "historical_asset_without_eodhd_coverage", "insufficient_window"}:
        return "coverage_limits"
    if status in {"session_windows_disagree", "missing_or_duplicate_session"}:
        return "session_alignment"
    if status in {"near_date_same_ratio", "near_date_ratio_mismatch", "near_date_ratio_not_evaluable", "ambiguous_near_date"}:
        return "event_date"
    if status in {"exact_date_ratio_mismatch", "exact_date_ratio_not_evaluable", "implied_share_ratio_disagreement",
                  "ratio_too_small_to_distinguish_from_unit", "invalid_or_unit_ratio"}:
        return "ratio_disagreement"
    if status in {"multiple_or_missing_share_events_in_window", "multiple_or_conflicting_share_events_in_window", "duplicate_provider_event_date"}:
        return "overlapping_share_events"
    if status in {"ohlc_window_disagreement", "adjustment_factor_disagreement", "invalid_price_window"}:
        return "price_or_factor_disagreement"
    if status in {"yahoo_continuity_not_confirmed", "large_event_session_return_review"}:
        return "continuity"
    if status in {"no_independent_split_basis_change", "yahoo_only_within_overlap"}:
        return "missing_independent_event_evidence"
    return "other_review"


def audit(s3, bucket, *, screening_run, reconciliation_run, unmatched_run, basis_runs, bundle_version,
          run_id, expected_count, execute=False):
    if len(basis_runs) != len(MODES) or len(set(basis_runs)) != len(MODES):
        raise ValueError("Expected five distinct basis runs in integer/fractional/small/irregular/Yahoo-only order")
    for v in [screening_run, reconciliation_run, unmatched_run, run_id, *basis_runs]:
        check_id(v)
    def manifest(family, run):
        return read_json(s3, bucket, f"{ROOT}/silver/{family}/v1/runs/{run}/manifest.json")
    screening = manifest("split_screening", screening_run)
    recon = manifest("provider_reconciliation", reconciliation_run)
    unmatched = manifest("unmatched_split_cases", unmatched_run)
    refs = {"screening_events": screening["event_output"], "reconciliation_events": recon["outputs"]["events"]}
    if any(unmatched["inputs"][k] != v for k, v in refs.items()):
        raise ValueError("Unmatched evidence inputs differ")
    screens = verified_frame(s3, bucket, refs["screening_events"]).to_dict("records")
    unmatched_rows = verified_frame(s3, bucket, unmatched["outputs"]["events"]).to_dict("records")
    batches, approvals, approved_ids, event_refs = [], [], set(), []
    for run, mode, status in zip(basis_runs, MODES, STATUSES):
        m = manifest("split_basis_candidates", run)
        if any(m["inputs"][k] != v for k, v in refs.items()):
            raise ValueError("Basis evidence inputs differ")
        rows = verified_frame(s3, bucket, m["outputs"]["events"]).to_dict("records")
        candidates = {identity(r) for r in rows if r["status"] == status}
        ref = m["outputs"]["candidate_registry"]
        reg = json.loads(verified_bytes(s3, bucket, ref))
        if {identity(r) for r in reg["entries"]} != candidates or approved_ids & candidates:
            raise ValueError("Candidate population disagreement or repeated approval")
        a = approve(s3, bucket, ref["sha256"], expected_count=len(candidates), **mode)
        verified_bytes(s3, bucket, {"key": a["key"], "sha256": a["approval_version"]})
        approvals.append({"key": a["key"], "sha256": a["approval_version"], "count": len(candidates)})
        approved_ids |= candidates
        batches.append(rows)
        event_refs.append(m["outputs"]["events"])
    inventory = consolidate(screens, unmatched_rows, batches)
    remaining = [r for key, r in inventory.items() if key not in approved_ids]
    if len(remaining) != expected_count or len(screens) != len(approved_ids) + expected_count:
        raise ValueError("Unexpected remaining population")
    bundle = load_bundle(s3, bucket, bundle_version)
    index = evidence_index(bundle)
    diagnostics = manifest("financial_diagnostics", recon["inputs"]["yahoo_diagnostic"])
    if diagnostics["inputs"]["consolidated_index_version"] != bundle["consolidated_index_version"]:
        raise ValueError("Existing financial bundle uses different source index")
    dividends = verified_frame(s3, bucket, refs["reconciliation_events"])
    dividends = dividends.loc[dividends.kind.eq("dividend")]
    cash_by_asset = defaultdict(list)
    for r in dividends.to_dict("records"):
        cash_by_asset[r["asset_id"]].append(r)
    findings = verified_frame(s3, bucket, diagnostics["outputs"]["dividend_findings"])
    findings_by_asset = defaultdict(list)
    for r in findings.to_dict("records"):
        findings_by_asset[r["asset_id"]].append(r)
    reasons, categories, evidence_counts, overlaps = Counter(), Counter(), Counter(), defaultdict(Counter)
    family_counts, exact_family_counts, cash_counts = Counter(), Counter(), Counter()
    for row in remaining:
        day = date.fromisoformat(row["date"])
        matches = match_evidence(row, index)
        related = [m for m in matches if m["source_matches"]]
        families = sorted({m["family"] for m in related})
        exact = any(m["relation"] == "exact_date" for m in related)
        evidence_status = "exact_registry_date" if exact else "nearby_registry_evidence" if related else "source_mismatch_only" if matches else "no_registry_match"
        cash = []
        for r in cash_by_asset[row["asset_id"]]:
            if r["yahoo_sha256"] != row["bronze_sha256"]:
                continue
            dates = [d for d in (r["yahoo_date"], r["eodhd_date"]) if pd.notna(d)]
            if any(abs((date.fromisoformat(d)-day).days) <= 4 for d in dates):
                cash.append({k: r[k] for k in ("yahoo_date", "eodhd_date", "date_status", "value_status") if pd.notna(r[k])})
        diagnostic_matches = []
        for r in findings_by_asset[row["asset_id"]]:
            ds = r.get("effective_date")
            if pd.notna(ds) and abs((date.fromisoformat(str(ds)[:10])-day).days) <= 4:
                diagnostic_matches.append({"date": str(ds)[:10], "reason": r["reason"]})
        family_counts.update(families)
        exact_family_counts.update({m["family"] for m in related if m["relation"] == "exact_date"})
        if any(row["date"] in (m.get("yahoo_date"), m.get("eodhd_date")) for m in cash) or row["same_day_dividend"]:
            cash_counts["same_day_cash"] += 1
        elif cash:
            cash_counts["nearby_cash"] += 1
        else:
            cash_counts["no_cash_match"] += 1
        if diagnostic_matches:
            cash_counts["prior_diagnostic_match"] += 1
        category = classify_reason(row["status"])
        row.update({"root_category": category, "registry_evidence_status": evidence_status,
                    "registry_families": ",".join(families),
                    "registry_matches_json": json.dumps(matches, sort_keys=True),
                    "provider_cash_matches_json": json.dumps(cash, sort_keys=True),
                    "diagnostic_matches_json": json.dumps(diagnostic_matches, sort_keys=True),
                    "production_eligible": False, "quarantine_released": False})
        reasons[row["status"]] += 1
        categories[category] += 1
        evidence_counts[evidence_status] += 1
        overlaps[category][evidence_status] += 1
    scope = f"{SCOPE}/runs/{run_id}"
    outputs = {}
    if execute:
        blob = parquet_bytes(pd.DataFrame(remaining))
        key = f"{scope}/objects/events-{digest(blob)}.parquet"
        outputs["events"] = {**put_immutable(s3, bucket, key, blob, content_type="application/vnd.apache.parquet",
                                             scope=SCOPE), "rows": len(remaining), "format": "parquet"}
    result = {"schema": "alpha_edge_remaining_split_cases_v1", "run_id": run_id,
              "inputs": {**refs, "unmatched_events": unmatched["outputs"]["events"], "basis_events": event_refs,
                         "approvals": approvals, "financial_bundle_version": bundle_version,
                         "dividend_findings": diagnostics["outputs"]["dividend_findings"]},
              "statistics": {"screened": len(screens), "approved": len(approved_ids), "remaining": len(remaining),
                             "assets": len({r["asset_id"] for r in remaining}),
                             "reasons": dict(reasons.most_common()), "root_categories": dict(categories.most_common()),
                             "registry_evidence": dict(evidence_counts),
                             "registry_families_nonexclusive": dict(family_counts.most_common()),
                             "exact_registry_families_nonexclusive": dict(exact_family_counts.most_common()),
                             "cash_evidence": dict(cash_counts),
                             "registry_evidence_by_category": {k: dict(v) for k, v in overlaps.items()}},
              "outputs": outputs, "production_eligible": False, "quarantine_released": False}
    key = f"{scope}/manifest.json"
    if execute:
        put_immutable(s3, bucket, key, json_bytes(result), content_type="application/json", scope=SCOPE)
    return {**result, "manifest_key": key, "status": "candidate_written" if execute else "dry_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    for name in ("screening-run", "reconciliation-run", "unmatched-run", "bundle-version", "run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--basis-runs", required=True, nargs=5)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(boto3.client("s3", region_name=args.region), args.bucket,
                           screening_run=args.screening_run, reconciliation_run=args.reconciliation_run,
                           unmatched_run=args.unmatched_run, basis_runs=args.basis_runs, bundle_version=args.bundle_version,
                           run_id=args.run_id, expected_count=args.expected_count, execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
