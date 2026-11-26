"""Record deterministic candidate approval and prioritize remaining corporate-action evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

import boto3
import pandas as pd

from alpha_edge.market.audit_matching_split_basis_v3 import close_to
from alpha_edge.market.reconcile_corporate_actions_v3 import PASS
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT, check_id, digest, json_bytes, parquet_bytes, put_immutable,
    read_json, verified_bytes, verified_frame,
)

SCOPE = f"{ROOT}/silver/deterministic_residual_diagnostics/v1"
APPROVAL_SCOPE = f"{ROOT}/config/deterministic_corporate_action_approval_registries/v1"
CANDIDATE = "reconciled_price_basis_candidate"


def identity(row):
    return row["asset_id"], row["date"]


def validate_approval(manifest, rows, registry, version, count):
    if (manifest["schema"] != "alpha_edge_deterministic_corporate_action_reconciliation_v1"
            or registry["schema"] != "alpha_edge_deterministic_corporate_action_candidate_registry_v1"
            or registry["inputs"] != manifest["inputs"]
            or manifest["outputs"]["candidate_registry"]["sha256"] != version
            or registry["production_eligible"] is not False or registry["quarantine_released"] is not False):
        raise ValueError("Candidate registry or manifest mismatch")
    by_id = {identity(r): r for r in rows}
    candidates = {identity(r) for r in rows if r["status"] == CANDIDATE}
    entries = registry["entries"]
    if (len(by_id) != len(rows) or len(rows) != manifest["statistics"]["events"]
            or count <= 0 or len(entries) != count or len(candidates) != count
            or manifest["statistics"]["candidate_events"] != count
            or {identity(r) for r in entries} != candidates):
        raise ValueError("Unexpected candidate population")
    for entry in entries:
        row = by_id[identity(entry)]
        chosen = entry["chosen_hypothesis"]
        if (entry["status"] != CANDIDATE or entry["production_eligible"] is not False
                or entry["quarantine_released"] is not False or not entry["economic_review_required"]
                or chosen["status"] not in PASS or chosen["date_method"] != "reported_dates"
                or chosen != json.loads(row["chosen_hypothesis"])
                or entry["bronze_sha256"] != row["bronze_sha256"]
                or entry["eodhd_sha256"] != row["eodhd_sha256"]
                or not math.isclose(float(entry["ratio"]), float(row["ratio"]), rel_tol=1e-7)):
            raise ValueError("Candidate evidence mismatch")
    return candidates


def market(symbol):
    if not isinstance(symbol, str) or not symbol:
        return "unknown"
    return symbol.rsplit(".", 1)[-1].upper() if "." in symbol else "unsuffixed"


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def pattern(row):
    attempts = json.loads(row["attempts"])
    statuses = {a["status"] for a in attempts}
    prior = json.loads(row["prior_factors"])
    flags = set()
    for a in attempts:
        if a["status"] == "ratio_diagnostic_only":
            flags.update("observed_ratio_matches_"+x for x in a["matching_hypotheses"])
        if positive(a.get("yahoo_cash")) and positive(a.get("eodhd_adjusted_cash")):
            if close_to(a["yahoo_cash"], a["eodhd_adjusted_cash"]):
                flags.add("adjusted_cash_matches_yahoo")
        if positive(a.get("pre_scale")) and positive(a.get("post_scale")):
            if close_to(a["pre_scale"], a["post_scale"]):
                flags.add("stable_relative_quote_scale")
            if any(close_to(a["post_scale"], k) for k in (0.01, 0.1, 10., 100.)):
                flags.add("possible_decimal_quote_unit_scale")
        if positive(a.get("yahoo_close_step")) and positive(a.get("eodhd_close_step")):
            if close_to(a["yahoo_close_step"], a["eodhd_close_step"]):
                flags.add("same_observed_price_return")
        if positive(a.get("yahoo_adjustment_factor_step")) and positive(a.get("eodhd_adjustment_factor_step")):
            if close_to(a["yahoo_adjustment_factor_step"], a["eodhd_adjustment_factor_step"]):
                flags.add("same_observed_adjustment_step")
    if prior:
        flags.add("prior_documented_factor")
    rules = [
        ("no_independent_split_basis_change", "independent_event_evidence_required"),
        ("cash_not_uniquely_reported_by_both", "cash_event_identity_or_missing_amount"),
        ("nearby_cash_requires_separate_reconciliation", "nearby_cash_date_or_multiple_distributions"),
        ("nearby_cash_review", "nearby_cash_date_or_multiple_distributions"),
        ("cash_amount_or_share_basis_disagreement", "cash_amount_basis_or_provider_definition"),
        ("cash_factor_too_small_to_distinguish", "small_cash_factor_below_resolution"),
        ("cash_and_share_factor_disagreement", "cash_factor_conflict"),
        ("ohlc_window_disagreement", "unstable_cross_provider_ohlc_scale"),
        ("large_cash_adjusted_session_return_review", "large_cash_adjusted_return"),
        ("large_event_session_return_review", "large_event_return"),
        ("implied_share_ratio_disagreement", "observed_ratio_conflict"),
        ("multiple_or_conflicting_share_events_in_window", "overlapping_share_events"),
        ("multiple_or_missing_share_events_in_window", "overlapping_share_events"),
        ("conflicting_eodhd_share_events", "overlapping_share_events"),
        ("ratio_invalid_or_indistinguishable", "small_or_invalid_ratio"),
        ("ratio_too_small_to_distinguish_from_unit", "small_or_invalid_ratio"),
        ("missing_or_duplicate_session", "missing_or_duplicate_session"),
        ("session_windows_disagree", "session_window_disagreement"),
        ("insufficient_window", "insufficient_price_history"),
        ("provider_ratio_or_date_not_corroborated", "provider_date_ratio_or_coverage_conflict"),
    ]
    reason = next((label for status,label in rules if status in statuses), "other_evidence_required")
    return reason, sorted(flags), sorted(statuses)


def audit(s3, bucket, *, deterministic_run, candidate_version, approved_count, run_id, expected_count=None, execute=False):
    for value in (deterministic_run, run_id):
        check_id(value)
    manifest = read_json(s3, bucket, f"{ROOT}/silver/corporate_action_reconciliation/v1/runs/{deterministic_run}/manifest.json")
    rows = verified_frame(s3, bucket, manifest["outputs"]["events"]).to_dict("records")
    registry = json.loads(verified_bytes(s3, bucket, manifest["outputs"]["candidate_registry"]))
    approved = validate_approval(manifest, rows, registry, candidate_version, approved_count)
    remaining = [r for r in rows if identity(r) not in approved]
    if expected_count is not None and len(remaining) != expected_count:
        raise ValueError("Unexpected residual count")
    approval = {"schema": "alpha_edge_deterministic_corporate_action_approval_v1",
                "candidate_registry": manifest["outputs"]["candidate_registry"], "events": manifest["outputs"]["events"],
                "approved_count": approved_count, "decision": "approved_price_basis_only",
                "production_eligible": False, "quarantine_released": False, "economic_review_required": True}
    body = json_bytes(approval)
    approval_ref = {"key": f"{APPROVAL_SCOPE}/objects/{digest(body)}.json", "sha256": digest(body)}
    categories, patterns, markets, flags, attempts = Counter(), Counter(), Counter(), Counter(), Counter()
    asset_counts, by_pattern_market = defaultdict(Counter), defaultdict(Counter)
    output = []
    for row in remaining:
        reason, evidence_flags, attempt_statuses = pattern(row)
        suffix = market(row.get("symbol"))
        categories[row["root_category"]] += 1
        patterns[reason] += 1
        markets[suffix] += 1
        flags.update(evidence_flags)
        attempts.update(attempt_statuses)
        by_pattern_market[reason][suffix] += 1
        asset_counts[(row["asset_id"], str(row.get("symbol", "")))][reason] += 1
        output.append({**row, "diagnostic_pattern": reason, "market_suffix": suffix,
                       "evidence_flags_json": json.dumps(evidence_flags),
                       "attempt_statuses_json": json.dumps(attempt_statuses)})
    top = sorted(asset_counts.items(), key=lambda x: (-sum(x[1].values()), x[0]))
    scope = f"{SCOPE}/runs/{run_id}"
    outputs = {}
    if execute:
        # Validate every input and produce the full analysis before recording the approval.
        put_immutable(s3, bucket, approval_ref["key"], body, content_type="application/json", scope=APPROVAL_SCOPE)
        blob = parquet_bytes(pd.DataFrame(output))
        outputs["events"] = {**put_immutable(s3, bucket, f"{scope}/objects/events-{digest(blob)}.parquet", blob,
                                             content_type="application/vnd.apache.parquet", scope=SCOPE),
                             "rows": len(output), "format": "parquet"}
    result = {"schema": "alpha_edge_deterministic_residual_diagnostics_v1", "run_id": run_id,
              "inputs": {"deterministic_events": manifest["outputs"]["events"], "approval": approval_ref},
              "statistics": {"approved": approved_count, "remaining": len(remaining), "affected_assets": len(asset_counts),
                             "root_categories": dict(categories.most_common()), "patterns": dict(patterns.most_common()),
                             "market_suffixes": dict(markets.most_common()),
                             "evidence_flags_nonexclusive": dict(flags.most_common()),
                             "attempt_statuses_by_unique_event_nonexclusive": dict(attempts.most_common()),
                             "markets_by_pattern": {k: dict(v.most_common()) for k,v in by_pattern_market.items()},
                             "top_assets": [{"asset_id": a, "symbol": s, "events": sum(c.values()), "patterns": dict(c)}
                                            for (a,s),c in top[:10]]},
              "outputs": outputs, "production_eligible": False, "quarantine_released": False}
    key = f"{scope}/manifest.json"
    if execute:
        put_immutable(s3, bucket, key, json_bytes(result), content_type="application/json", scope=SCOPE)
    return {**result, "manifest_key": key, "status": "candidate_written" if execute else "dry_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--deterministic-run", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--approved-count", required=True, type=int)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(boto3.client("s3", region_name=args.region), args.bucket,
                           deterministic_run=args.deterministic_run, candidate_version=args.candidate_version,
                           approved_count=args.approved_count, expected_count=args.expected_count,
                           run_id=args.run_id, execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
