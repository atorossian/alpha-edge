"""Classify unmatched Yahoo splits from immutable reconciliation and screening evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date

import boto3
import pandas as pd

from alpha_edge.market.lakehouse_v3.storage import (
    ROOT, check_id, digest, json_bytes, parquet_bytes, put_immutable, read_json, verified_frame,
)

SCOPE = f"{ROOT}/silver/unmatched_split_cases/v1"


def reason(row):
    if row["value_status"] == "duplicate_date":
        return "duplicate_provider_event_date"
    ds, vs = row["date_status"], row["value_status"]
    if ds == "exact_date":
        return {"ratio_mismatch": "exact_date_ratio_mismatch",
                "ratio_not_evaluable": "exact_date_ratio_not_evaluable"}[vs]
    if ds == "near_date_candidate":
        return {"ratio_match": "near_date_same_ratio",
                "ratio_mismatch": "near_date_ratio_mismatch",
                "ratio_not_evaluable": "near_date_ratio_not_evaluable"}[vs]
    return {"ambiguous_near_date": "ambiguous_near_date",
            "yahoo_only": "yahoo_only_within_overlap",
            "outside_overlap": "outside_price_history_overlap"}[ds]


def classify_cases(screens, events, selected, historical):
    """One row per screened Yahoo split; EODHD-only rows are never counted as Yahoo cases."""
    if screens.duplicated(["asset_id", "date"]).any():
        raise ValueError("Duplicate screening identity")
    splits = events.loc[events.kind.eq("split")]
    yahoo = splits.loc[splits.yahoo_event_rows.gt(0)]
    if yahoo.duplicated(["asset_id", "yahoo_date"]).any():
        raise ValueError("Duplicate reconciliation identity")
    by_key = {(r["asset_id"], r["yahoo_date"]): r for r in yahoo.to_dict("records")}
    screen_keys = set(zip(screens.asset_id, screens.date))
    if set(by_key) - screen_keys:
        raise ValueError("Reconciliation contains unscreened Yahoo splits")
    rows, matched = [], 0
    for screen in screens.to_dict("records"):
        aid, day = screen["asset_id"], screen["date"]
        row = by_key.get((aid, day))
        if row is None:
            if aid in selected or aid not in historical:
                raise ValueError(f"Missing reconciliation evidence for {aid} {day}")
            why = "historical_asset_without_eodhd_coverage"
            detail = {}
        else:
            if row["yahoo_sha256"] != screen["bronze_sha256"]:
                raise ValueError(f"Source digest mismatch for {aid} {day}")
            if row["yahoo_event_rows"] != 1:
                raise ValueError(f"Screened Yahoo event is not unique: {aid} {day}")
            # A non-evaluable ratio is retained as an exception, not normalized.
            if pd.notna(row["yahoo_value"]) and not math.isclose(
                    float(row["yahoo_value"]), float(screen["ratio"]), rel_tol=1e-7):
                raise ValueError(f"Yahoo ratio mismatch for {aid} {day}")
            if row["date_status"] == "exact_date" and row["value_status"] == "ratio_match":
                matched += 1
                continue
            why = reason(row)
            detail = {k: row.get(k) for k in (
                "yahoo_symbol", "eodhd_symbol", "eodhd_sha256", "eodhd_date", "eodhd_value",
                "date_status", "value_status", "yahoo_event_rows", "eodhd_event_rows",
                "yahoo_raw", "eodhd_raw", "previous_shared_date", "yahoo_close_step",
                "eodhd_close_step", "cross_provider_step_ratio")}
            detail["date_offset_days"] = (
                (date.fromisoformat(row["eodhd_date"]) - date.fromisoformat(day)).days
                if pd.notna(row["eodhd_date"]) else None)
        rows.append({**screen, **detail, "reason": why,
                     "production_eligible": False, "quarantine_released": False})
    return rows, matched


def audit(s3, bucket, *, reconciliation_run, screening_run, run_id, expected_count, execute=False):
    for value in (reconciliation_run, screening_run, run_id):
        check_id(value)
    recon = read_json(s3, bucket, f"{ROOT}/silver/provider_reconciliation/v1/runs/{reconciliation_run}/manifest.json")
    screening = read_json(s3, bucket, f"{ROOT}/silver/split_screening/v1/runs/{screening_run}/manifest.json")
    if (recon["schema"] != "alpha_edge_provider_reconciliation_v1"
            or screening["schema"] != "alpha_edge_split_screening_v1"
            or recon["inputs"]["yahoo_diagnostic"] != screening["diagnostic_run_id"]):
        raise ValueError("Incompatible reconciliation and screening runs")
    events = verified_frame(s3, bucket, recon["outputs"]["events"])
    screens = verified_frame(s3, bucket, screening["event_output"])
    selected = set(recon["inputs"]["selected_asset_ids"])
    historical = set(recon["inputs"]["yahoo_historical_only_ids"])
    if selected & historical:
        raise ValueError("Overlapping provider coverage categories")
    rows, matched = classify_cases(screens, events, selected, historical)
    expected_matched = recon["statistics"]["split"]["exact_date_value_status"]["ratio_match"]
    if len(screens) != screening["events"] or matched != expected_matched or len(rows) != expected_count:
        raise ValueError("Unmatched population does not equal expected count")
    counts = Counter(r["reason"] for r in rows)
    groups = defaultdict(Counter)
    for row in rows:
        groups[row["reason"]][row["group"]] += 1
    e_only = events.loc[events.kind.eq("split") & events.yahoo_event_rows.eq(0)]
    stats = {"yahoo_screened_events": len(screens), "exact_ratio_matches": matched,
             "unmatched_yahoo_events": len(rows), "affected_assets": len({r["asset_id"] for r in rows}),
             "reasons": dict(counts.most_common()),
             "groups_by_reason": {k: dict(v) for k, v in groups.items()},
             "same_day_yahoo_cash_by_reason": dict(Counter(r["reason"] for r in rows if r["same_day_dividend"])),
             "eodhd_only_event_rows_by_date_status": {str(k): int(v) for k, v in e_only.groupby("date_status").eodhd_event_rows.sum().items()}}
    scope = f"{SCOPE}/runs/{run_id}"
    outputs = {}
    if execute:
        blob = parquet_bytes(pd.DataFrame(rows))
        key = f"{scope}/objects/events-{digest(blob)}.parquet"
        outputs["events"] = {**put_immutable(s3, bucket, key, blob,
                                             content_type="application/vnd.apache.parquet", scope=SCOPE),
                             "rows": len(rows), "format": "parquet"}
    manifest = {"schema": "alpha_edge_unmatched_split_cases_v1", "run_id": run_id,
                "inputs": {"reconciliation_run": reconciliation_run, "screening_run": screening_run,
                           "reconciliation_events": recon["outputs"]["events"],
                           "screening_events": screening["event_output"]},
                "statistics": stats, "outputs": outputs,
                "production_eligible": False, "quarantine_released": False}
    key = f"{scope}/manifest.json"
    if execute:
        put_immutable(s3, bucket, key, json_bytes(manifest), content_type="application/json", scope=SCOPE)
    return {**manifest, "manifest_key": key, "status": "candidate_written" if execute else "dry_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--reconciliation-run", required=True)
    parser.add_argument("--screening-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(boto3.client("s3", region_name=args.region), args.bucket,
                           reconciliation_run=args.reconciliation_run, screening_run=args.screening_run,
                           run_id=args.run_id, expected_count=args.expected_count, execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
