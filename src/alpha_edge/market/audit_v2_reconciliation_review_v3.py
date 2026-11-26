"""Review accepted price-basis hypotheses and prioritize the remaining evidence."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import boto3
import pandas as pd

from alpha_edge.market.audit_deterministic_residual_v3 import identity, validate_approval
from alpha_edge.market.audit_matching_split_basis_v3 import TOL, close_to
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT, check_id, digest, json_bytes, parquet_bytes, put_immutable, read_json,
    verified_bytes, verified_frame,
)

SCOPE = f"{ROOT}/silver/deterministic_candidate_review/v1"


def candidate_flags(row):
    choice = json.loads(row["chosen_hypothesis"])
    attempts = json.loads(row["attempts"])
    checks = choice.get("daily_checks", [])
    target = [c for c in checks if c["date"] == row["date"]]
    return {
        "large_return_any_window": bool(choice.get("large_return_review_required", False)),
        "large_return_target": any(c.get("large_return_review_required", False) for c in target),
        "raw_cash_basis_any_window": bool(choice.get("cash_raw_basis_review_required", False)),
        "raw_cash_basis_target": any(c.get("cash_evidence") == "reported_adjusted" for c in target),
        "prior_documented_factor": bool(json.loads(row["prior_factors"])),
        "passing_windows": sum(a["status"] == choice["status"] for a in attempts),
        "economic_review_required": True,
    }


def residual_flags(row):
    attempts = json.loads(row["attempts"])
    advanced = [a for a in attempts if a.get("method") == "daily_factor_chain"]
    statuses = sorted({a["status"] for a in attempts})
    fields = set()
    isolated = set()
    observed = set()
    adjusted_cash_agrees = False
    for a in attempts:
        errors = a.get("field_max_relative_error", {})
        bad = {f for f,v in errors.items() if v is not None and v > TOL}
        fields.update(bad)
        if len(bad) == 1:
            isolated.update(bad)
        observed.update(a.get("matching_hypotheses", []))
        yc, ec = a.get("yahoo_cash"), a.get("eodhd_adjusted_cash")
        if isinstance(yc,(int,float)) and isinstance(ec,(int,float)) and close_to(yc,ec):
            adjusted_cash_agrees = True
    return {
        "failed_fields": sorted(fields), "isolated_field_hypotheses": sorted(isolated),
        "rounding_compatible_window": any(a.get("four_decimal_rounding_compatible") is True for a in advanced),
        "observed_ratio_hypotheses": sorted(observed), "adjusted_cash_matches_yahoo": adjusted_cash_agrees,
        "advanced_statuses": sorted({a["status"] for a in advanced}),
        "all_statuses": statuses,
    }


def audit(s3,bucket,*,deterministic_run,candidate_version,expected_candidates,expected_remaining,run_id,execute=False):
    for value in (deterministic_run,run_id):
        check_id(value)
    m = read_json(s3,bucket,f"{ROOT}/silver/corporate_action_reconciliation/v1/runs/{deterministic_run}/manifest.json")
    rows = verified_frame(s3,bucket,m["outputs"]["events"]).to_dict("records")
    registry = json.loads(verified_bytes(s3,bucket,m["outputs"]["candidate_registry"]))
    candidates = validate_approval(m,rows,registry,candidate_version,expected_candidates)
    remaining = [r for r in rows if identity(r) not in candidates]
    if len(remaining) != expected_remaining:
        raise ValueError("Unexpected residual population")
    candidate_rows, residual_rows, research = [], [], []
    flag_counts, intersections, advanced_counts, field_counts, categories = Counter(),Counter(),Counter(),Counter(),Counter()
    advanced_by_category = defaultdict(Counter)
    targets = Counter()
    for row in rows:
        if identity(row) in candidates:
            flags = candidate_flags(row)
            flag_counts.update(k for k,v in flags.items() if v is True)
            label = ("large_and_cash" if flags["large_return_any_window"] and flags["raw_cash_basis_any_window"]
                     else "large_return_only" if flags["large_return_any_window"] else
                     "cash_basis_only" if flags["raw_cash_basis_any_window"] else "neither_flag")
            intersections[label] += 1
            candidate_rows.append({**row, **flags, "review_group":label})
            continue
        flags = residual_flags(row)
        category = row["root_category"]
        categories[category] += 1
        advanced_counts.update(flags["advanced_statuses"])
        advanced_by_category[category].update(flags["advanced_statuses"])
        field_counts.update(flags["failed_fields"])
        if flags["isolated_field_hypotheses"]:
            targets["isolated_field_disagreement"] += 1
        if flags["rounding_compatible_window"]:
            targets["rounding_compatible_window"] += 1
        if flags["adjusted_cash_matches_yahoo"]:
            targets["adjusted_cash_matches_yahoo"] += 1
        if "yahoo" in flags["observed_ratio_hypotheses"]:
            targets["observed_ratio_matches_yahoo"] += 1
        detail = {**row, "review_evidence_json":json.dumps(flags,sort_keys=True)}
        residual_rows.append(detail)
        if category == "missing_independent_event_evidence":
            research.append({"asset_id":row["asset_id"], "symbol":row.get("symbol"), "date":row["date"],
                             "ratio":row["ratio"], "bronze_sha256":row["bronze_sha256"],
                             "eodhd_sha256":row["eodhd_sha256"],
                             "required_evidence":"issuer_or_exchange_event_date_ratio_and_economic_type",
                             "production_eligible":False})
    scope = f"{SCOPE}/runs/{run_id}"
    outputs = {}
    if execute:
        for name,table in (("candidate_review",candidate_rows),("residual_review",residual_rows),("issuer_research_queue",research)):
            blob = parquet_bytes(pd.DataFrame(table))
            outputs[name] = {**put_immutable(s3,bucket,f"{scope}/objects/{name}-{digest(blob)}.parquet",blob,
                                              content_type="application/vnd.apache.parquet",scope=SCOPE),
                             "rows":len(table),"format":"parquet"}
    result = {"schema":"alpha_edge_deterministic_candidate_review_v1","run_id":run_id,
              "inputs":{"deterministic_events":m["outputs"]["events"],"candidate_registry":m["outputs"]["candidate_registry"]},
              "statistics":{"candidates":len(candidates),"remaining":len(remaining),
                            "candidate_review_groups":dict(intersections),"candidate_flags_nonexclusive":dict(flag_counts),
                            "remaining_categories":dict(categories.most_common()),"advanced_statuses_nonexclusive":dict(advanced_counts.most_common()),
                            "advanced_by_category":{k:dict(v.most_common()) for k,v in advanced_by_category.items()},
                            "field_disagreement_nonexclusive":dict(field_counts.most_common()),
                            "residual_evidence_flags_nonexclusive":dict(targets.most_common()),
                            "issuer_research_events":len(research),"issuer_research_assets":len({r["asset_id"] for r in research})},
              "outputs":outputs,"approval_written":False,"production_eligible":False,"quarantine_released":False}
    key = f"{scope}/manifest.json"
    if execute:
        put_immutable(s3,bucket,key,json_bytes(result),content_type="application/json",scope=SCOPE)
    return {**result,"manifest_key":key,"status":"review_written" if execute else "dry_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket",default="alpha-edge-algo")
    parser.add_argument("--region",default="eu-west-1")
    parser.add_argument("--deterministic-run",required=True)
    parser.add_argument("--candidate-version",required=True)
    parser.add_argument("--expected-candidates",required=True,type=int)
    parser.add_argument("--expected-remaining",required=True,type=int)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    print(json.dumps(audit(boto3.client("s3",region_name=args.region),args.bucket,
                           deterministic_run=args.deterministic_run,candidate_version=args.candidate_version,
                           expected_candidates=args.expected_candidates,expected_remaining=args.expected_remaining,
                           run_id=args.run_id,execute=args.execute),indent=2))


if __name__ == "__main__":
    main()
