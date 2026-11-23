"""Read-only split-event screening and marginal quarantine coverage assessment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import boto3
import pandas as pd

from alpha_edge.market.lakehouse_v3.storage import (
    ROOT,
    check_id,
    digest,
    json_bytes,
    parquet_bytes,
    put_immutable,
    read_json,
    verified_frame,
)

SPLIT = "split_price_basis_unverified"
GROUPS = ("integer_forward", "integer_reverse", "simple_fraction", "irregular_ratio", "invalid_or_unit_ratio")


def classify(ratio: float) -> str:
    """Ratio shape, not legal/economic event identity; no implicit approval."""
    if not math.isfinite(ratio) or ratio <= 0 or math.isclose(ratio, 1, abs_tol=1e-8):
        return "invalid_or_unit_ratio"
    if ratio > 1 and math.isclose(ratio, round(ratio), rel_tol=1e-7):
        return "integer_forward"
    inverse = 1 / ratio
    if ratio < 1 and math.isclose(inverse, round(inverse), rel_tol=1e-7):
        return "integer_reverse"
    candidate = Fraction(ratio).limit_denominator(20)
    if candidate.numerator <= 100 and math.isclose(ratio, float(candidate), rel_tol=1e-7):
        return "simple_fraction"
    return "irregular_ratio"


def screen_asset(frame, findings, *, asset_id):
    """Reconcile every source split with its exact audit date and estimate bounds."""
    f = frame.sort_values("session_date").reset_index(drop=True)
    if not f.asset_id.eq(asset_id).all() or f.session_date.duplicated().any():
        raise ValueError("Invalid source identity or duplicate sessions")
    dates = pd.to_datetime(f.session_date).dt.strftime("%Y-%m-%d")
    split_rows = f.loc[f.stock_splits.ne(0)]
    split_dates = set(findings.loc[findings.reason.eq(SPLIT), "effective_date"])
    if set(dates.loc[split_rows.index]) != split_dates or len(split_rows) != len(split_dates):
        raise ValueError("Split audit/source mismatch")
    prefix = findings.loc[findings.scope.eq("prefix")]
    old = prefix.effective_date.max()
    nonsplit = prefix.loc[prefix.reason.ne(SPLIT), "effective_date"]
    valid = f.close.gt(0) & f.adj_close.gt(0) & f.close.map(math.isfinite) & f.adj_close.map(math.isfinite)
    events = []
    for i in split_rows.index:
        row = f.iloc[i]
        ratio = float(row.stock_splits)
        day = dates.iloc[i]
        same_day = bool(row.dividends != 0)
        continuity = "not_evaluable"
        raw_return = None
        if i > 0 and valid.iloc[i - 1] and valid.iloc[i] and ratio > 0 and math.isfinite(ratio):
            q = float(row.close / f.close.iloc[i - 1])
            raw_return = q - 1
            if not same_day:
                d0, d1 = abs(math.log(q)), abs(math.log(q * ratio))
                if d0 <= 0.10 and d1 - d0 >= 0.10:
                    continuity = "provider_normalized_signal"
                elif d1 <= 0.10 and d0 - d1 >= 0.10:
                    continuity = "unadjusted_signal"
                else:
                    continuity = "ambiguous"
        events.append(
            {
                "asset_id": asset_id,
                "symbol": str(row.provider_symbol),
                "date": day,
                "ratio": ratio,
                "group": classify(ratio),
                "same_day_dividend": same_day,
                "price_return": raw_return,
                "continuity": continuity,
            }
        )

    def bound(remove):
        remaining = list(nonsplit) + [e["date"] for e in events if e["group"] not in remove]
        boundary = max(remaining) if remaining else None
        return int((valid & dates.le(old) & (dates.gt(boundary) if boundary else True)).sum())

    release = {g: bound({g}) for g in GROUPS}
    release["integer_both"] = bound({"integer_forward", "integer_reverse"})
    release["all_reported_splits"] = bound(set(GROUPS))
    binding = [e["group"] for e in events if e["date"] == old]
    return events, {
        "asset_id": asset_id,
        "symbol": str(f.provider_symbol.iloc[0]),
        "binding_group": binding[0] if binding else "other",
        "binding_date": old,
        "binding_has_other": old in set(nonsplit),
        "source_prefix_rows": int(dates.le(old).sum()),
        "upper_bounds": release,
    }


def audit(s3, bucket, *, diagnostic_run_id, adjusted_run_id, workers=8, run_id=None, execute=False):
    check_id(diagnostic_run_id)
    check_id(adjusted_run_id)
    if not 1 <= workers <= 16:
        raise ValueError("Invalid worker count")
    if execute:
        check_id(run_id)
    diagnostic = read_json(s3, bucket, f"{ROOT}/silver/financial_diagnostics/v1/runs/{diagnostic_run_id}/manifest.json")
    adjusted = read_json(s3, bucket, f"{ROOT}/silver/adjusted_candidates/v1/runs/{adjusted_run_id}/manifest.json")
    if diagnostic["inputs"]["consolidated_index_version"] != adjusted["consolidated_index_version"]:
        raise ValueError("Diagnostic and adjusted candidate source index differ")
    for key in ("currency", "duplicate", "quarantine"):
        registry = key if key != "duplicate" else "duplicates"
        if diagnostic["inputs"][f"{key}_registry_version"] != adjusted["registries"][registry]:
            raise ValueError("Diagnostic and adjusted candidate registry differ")
    q = verified_frame(s3, bucket, diagnostic["outputs"]["quarantine_findings"])
    c = verified_frame(s3, bucket, diagnostic["outputs"]["coverage"])
    split = q.loc[q.reason.eq(SPLIT)]
    ids = set(split.asset_id)
    rows = c.loc[c.asset_id.isin(ids)].to_dict("records")
    if len(rows) != len(ids):
        raise ValueError("Missing or duplicate source coverage")
    by_asset = {a: g for a, g in q.groupby("asset_id")}
    manifest_assets = {e["asset_id"]: e for e in adjusted["assets"]}

    def process(row):
        aid = row["asset_id"]
        f = verified_frame(s3, bucket, {"key": row["bronze_key"], "sha256": row["bronze_sha256"]})
        if len(f) != row["source_rows"] or len(f) != manifest_assets[aid]["source_rows"]:
            raise ValueError("Source row count mismatch")
        events, report = screen_asset(f, by_asset[aid], asset_id=aid)
        for event in events:
            event["bronze_sha256"] = row["bronze_sha256"]
        report["bronze_sha256"] = row["bronze_sha256"]
        return events, report

    event_rows, reports = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (events, report) in enumerate(pool.map(process, rows), 1):
            event_rows.extend(events)
            reports.append(report)
            if i % 50 == 0 or i == len(rows):
                print(f"[split classification] {i}/{len(rows)}", file=sys.stderr, flush=True)
    if len(event_rows) != len(split) or len(event_rows) != diagnostic["statistics"]["reported_splits"]:
        raise ValueError("Incomplete split classification")
    e = pd.DataFrame(event_rows)
    a = pd.DataFrame(reports)
    result = {
        "schema": "alpha_edge_split_screening_v1",
        "diagnostic_run_id": diagnostic_run_id,
        "adjusted_run_id": adjusted_run_id,
        "events": len(e),
        "assets": len(a),
        "groups": [],
    }
    for group in GROUPS:
        g = e.loc[e.group.eq(group)]
        binding = a.loc[a.binding_group.eq(group)]
        result["groups"].append(
            {
                "group": group,
                "events": len(g),
                "assets": int(g.asset_id.nunique()),
                "same_day_dividend_events": int(g.same_day_dividend.sum()),
                "continuity_signals": {str(k): int(v) for k, v in g.continuity.value_counts().items()},
                "binding_assets": len(binding),
                "binding_candidate_rows": sum(manifest_assets[x]["quarantined_rows"] for x in binding.asset_id),
                "source_rows_potentially_unblocked_if_all_group_events_verified": sum(
                    r["upper_bounds"][group] for r in reports
                ),
            }
        )
    result["combined_upper_bounds"] = {
        g: sum(r["upper_bounds"][g] for r in reports) for g in ("integer_both", "all_reported_splits")
    }
    result["top_irregular_binding_assets"] = [
        {
            "symbol": r["symbol"],
            "asset_id": r["asset_id"],
            "quarantined_candidate_rows": manifest_assets[r["asset_id"]]["quarantined_rows"],
            "binding_date": r["binding_date"],
            "other_blocker_same_date": bool(r["binding_has_other"]),
        }
        for r in sorted(reports, key=lambda r: -manifest_assets[r["asset_id"]]["quarantined_rows"])
        if r["binding_group"] == "irregular_ratio"
    ][:20]
    result["production_eligible"] = False
    result["quarantine_released"] = False
    if execute:
        scope = f"{ROOT}/silver/split_screening/v1"
        root = f"{scope}/runs/{run_id}"
        blob = parquet_bytes(e)
        result["event_output"] = put_immutable(
            s3,
            bucket,
            f"{root}/events-{digest(blob)}.parquet",
            blob,
            content_type="application/vnd.apache.parquet",
            scope=scope,
        )
        asset_table = pd.DataFrame(
            [
                {
                    **{k: v for k, v in r.items() if k != "upper_bounds"},
                    **{f"potential_{k}": v for k, v in r["upper_bounds"].items()},
                }
                for r in reports
            ]
        )
        blob = parquet_bytes(asset_table)
        result["asset_output"] = put_immutable(
            s3,
            bucket,
            f"{root}/assets-{digest(blob)}.parquet",
            blob,
            content_type="application/vnd.apache.parquet",
            scope=scope,
        )
        result["manifest_key"] = f"{root}/manifest.json"
        put_immutable(
            s3, bucket, result["manifest_key"], json_bytes(result), content_type="application/json", scope=scope
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--diagnostic-run-id", required=True)
    parser.add_argument("--adjusted-run-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                boto3.client("s3", region_name=args.region),
                args.bucket,
                diagnostic_run_id=args.diagnostic_run_id,
                adjusted_run_id=args.adjusted_run_id,
                workers=args.workers,
                run_id=args.run_id,
                execute=args.execute,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
