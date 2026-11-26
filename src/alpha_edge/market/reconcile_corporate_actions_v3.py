"""Deterministic residual corporate-action reconciliation; candidate evidence only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.market.audit_combined_cash_share_v3 import (
    CANDIDATE as CASH_CANDIDATE, assess_prior, prior_factors, screen_cash_share,
)
from alpha_edge.market.audit_matching_split_basis_v3 import (
    CANDIDATE, FRACTIONAL, IRREGULAR_CANDIDATE, YAHOO_ONLY_CANDIDATE,
    WINDOW, FIELDS, TOL, close_to, positive_matrix, prepare, screen_event,
)
from alpha_edge.market.audit_provider_reconciliation_v3 import split_ratio
from alpha_edge.market.build_silver_adjusted_v3 import load_bundle
from alpha_edge.market.audit_financial_universe_v3 import load_consolidated, consolidated_receipt
from alpha_edge.market.build_share_basis_candidate_v3 import reconstruct as reconstruct_share_basis
from alpha_edge.market.reconciliation_hypotheses_v3 import CANDIDATE as ADVANCED_CANDIDATE, calendar_windows, analyze_window
from alpha_edge.market.eodhd_source_v3 import load_mapping, verify_bundle
from alpha_edge.market.ingest_bronze_full_history import EODHD_ROOT, _completed, load_universe
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT, check_id, digest, json_bytes, parquet_bytes, put_immutable, read_json,
    verified_bytes, verified_frame,
)

SCOPE = f"{ROOT}/silver/corporate_action_reconciliation/v1"
POLICY = "deterministic_residual_reconciliation_v1"
PASS = {CANDIDATE, FRACTIONAL, IRREGULAR_CANDIDATE, YAHOO_ONLY_CANDIDATE, CASH_CANDIDATE, ADVANCED_CANDIDATE}
ADVANCED_POLICY = "deterministic_residual_reconciliation_v2"


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def shared_windows(y, e, responses, day):
    """Permit missing peripheral quotes only; never bridge missing event/predecessor or actions."""
    if y.index.has_duplicates or e.index.has_duplicates or day not in y.index or day not in e.index:
        return None
    jy, je = y.index.get_loc(day), e.index.get_loc(day)
    if not jy or not je or y.index[jy-1] != e.index[je-1]:
        return None
    common = y.index.intersection(e.index).sort_values()
    j = common.get_loc(day)
    if j < WINDOW or j + WINDOW > len(common):
        return None
    window = common[j-WINDOW:j+WINDOW]
    dropped_y = y.loc[window[0]:window[-1]].index.difference(window)
    dropped_e = e.loc[window[0]:window[-1]].index.difference(window)
    if len(dropped_y) + len(dropped_e) > 2 or (pd.Timestamp(window[-1])-pd.Timestamp(window[0])).days > 30:
        return None
    if not len(dropped_y) and not len(dropped_e):
        return None
    if any(pd.to_numeric(y.loc[dropped_y, c], errors="coerce").fillna(0).ne(0).any()
           for c in ("stock_splits", "dividends")):
        return None
    if any(window[0] <= r["date"] <= window[-1] and r["date"] not in window
           for endpoint in ("splits", "div") for r in responses[endpoint]):
        return None
    return y.loc[window], e.loc[window]


def ratio_diagnostic(y, e, row, event):
    """Measure the observed scale step without authorizing a conflicting event."""
    day = row["date"]
    if y.index.has_duplicates or e.index.has_duplicates or day not in y.index or day not in e.index:
        return {"method": "observed_ratio", "status": "ratio_window_unavailable"}
    jy, je = y.index.get_loc(day), e.index.get_loc(day)
    if jy < WINDOW or je < WINDOW or jy + WINDOW > len(y) or je + WINDOW > len(e):
        return {"method": "observed_ratio", "status": "ratio_window_unavailable"}
    yw, ew = y.iloc[jy-WINDOW:jy+WINDOW], e.iloc[je-WINDOW:je+WINDOW]
    if not yw.index.equals(ew.index):
        return {"method": "observed_ratio", "status": "ratio_window_not_aligned"}
    ym, em = positive_matrix(yw, FIELDS), positive_matrix(ew, FIELDS)
    if ym is None or em is None:
        return {"method": "observed_ratio", "status": "ratio_window_invalid"}
    levels = em / ym
    pre, post = float(np.median(levels[:WINDOW])), float(np.median(levels[WINDOW:]))
    error = max(float(np.max(np.abs(levels[:WINDOW]/pre-1))),
                float(np.max(np.abs(levels[WINDOW:]/post-1))))
    if error > TOL or not math.isfinite(pre/post):
        return {"method": "observed_ratio", "status": "ratio_window_unstable", "max_relative_error": error}
    implied = pre/post
    yr = float(row["ratio"])
    er = event.get("eodhd_value")
    hypotheses = {"yahoo": yr, "yahoo_reciprocal": 1/yr if yr > 0 else 0, "unit": 1.0}
    if er is not None and pd.notna(er) and float(er) > 0:
        hypotheses.update({"eodhd": float(er), "eodhd_reciprocal": 1/float(er)})
    return {"method": "observed_ratio", "status": "ratio_diagnostic_only",
            "implied_ratio": implied, "matching_hypotheses": [k for k,v in hypotheses.items() if close_to(implied,v)],
            "max_relative_error": error}


def investigate(y, e, responses, row, event, prior, *, advanced=False, calendar=None, missing_expected=(), share_context=None):
    """Try every applicable bounded hypothesis; conflicts remain held."""
    day, ratio = row["date"], float(row["ratio"])
    attempts = [clean(ratio_diagnostic(y, e, row, event))]
    passed = []
    # A nearby source split may explain a date difference, but cannot establish
    # an issuer ex-date. Preserve both original dates in every attempted hypothesis.
    variants = [("reported_dates", responses, event)]
    if event["date_status"] == "near_date_candidate" and event["value_status"] == "ratio_match":
        ed = event["eodhd_date"]
        splits = [r for r in responses["splits"] if r["date"] == ed]
        if len(splits) == 1 and math.isclose(split_ratio(splits[0]["split"]) or 0, ratio, rel_tol=0.0001):
            shifted = {**responses, "splits": [{**r, "date": day} if r is splits[0] else r for r in responses["splits"]]}
            variants.append(("near_date_price_transition", shifted, {**event, "date_status": "exact_date"}))
    windows = [("native", y, e)]
    aligned = shared_windows(y, e, responses, day)
    if aligned is not None:
        windows.append(("shared_peripheral_sessions", *aligned))
    group = row["group"]
    for date_method, rsp, ev in variants:
        exact = ev["date_status"] == "exact_date" and ev["value_status"] == "ratio_match"
        yahoo_only = ev["date_status"] == "yahoo_only"
        if not (exact or yahoo_only):
            attempts.append({"method": date_method, "status": "provider_ratio_or_date_not_corroborated"})
            continue
        for window_method, yy, ee in windows:
            cash_here = (bool(row["same_day_dividend"])
                         or any(r["date"] == day for r in rsp["div"]))
            cash_related = cash_here or row["root_category"] == "cash_and_share"
            # The cash path tests both original amount conventions independently.
            if cash_related:
                results = [(f"cash_{basis}", screen_cash_share(yy, ee, rsp, row, ev, cash_basis=basis))
                           for basis in ("pre", "post")]
            else:
                options = {"yahoo_only": True} if yahoo_only else (
                    {"fractional": True} if group == "simple_fraction" else
                    {"irregular": True} if group == "irregular_ratio" else {})
                # A nearby cash event can explain an otherwise ordinary split.
                # Keep it held instead of guessing its economic relation.
                cash_dates_y = yy.index[pd.to_numeric(yy.dividends, errors="coerce").fillna(0).ne(0)]
                if (any(abs((pd.Timestamp(r["date"])-pd.Timestamp(day)).days) <= 4 for r in rsp["div"])
                        or any(abs((pd.Timestamp(d)-pd.Timestamp(day)).days) <= 4 for d in cash_dates_y)):
                    results = [("share_only", {"status": "nearby_cash_review"})]
                else:
                    results = [("share_only", screen_event(yy, ee, rsp, ev, row, recheck_continuity=True, **options))]
            for method, decision in results:
                record = clean({"method": method, "window": window_method, "date_method": date_method,
                                "eodhd_reported_date": event.get("eodhd_date"), **decision})
                attempts.append(record)
                if decision["status"] in PASS:
                    passed.append(record)
    if advanced:
        window_results = calendar_windows(y, e, calendar, responses, day, missing_expected)
        if not window_results:
            attempts.append({"method": "calendar_daily_chain", "status": "advanced_calendar_window_unavailable"})
        advanced_results = [clean(analyze_window(yy, ee, responses, row, event, size)) for size, yy, ee in window_results]
        attempts.extend(advanced_results)
        # A wider complete window is an independent challenge, not a reason to cherry-pick.
        if not passed and advanced_results and all(a["status"] == ADVANCED_CANDIDATE for a in advanced_results):
            passed.append(advanced_results[-1])
        if share_context is not None:
            attempts.append({"method": "existing_share_reconstruction_context", "status": "prior_reconstruction_context_only", **share_context})
    prior_conflict = any(f["comparison"] not in {"documented_share_ratio_match"} for f in prior)
    # No candidate can silently override documented coefficients, a date conflict,
    # or two distinct cash conventions that both fit within tolerance.
    if prior_conflict:
        status, choice = "prior_factor_requires_review", None
    elif not passed:
        status, choice = "evidence_exhausted", None
    else:
        substantive = {(r["method"], r["date_method"]) for r in passed}
        if len(substantive) != 1:
            status, choice = "ambiguous_passing_hypotheses", None
        else:
            choice = passed[0]
            status = ("price_basis_candidate_date_review" if choice["date_method"] != "reported_dates"
                      else "reconciled_price_basis_candidate")
    return {"status": status, "chosen_hypothesis": choice, "attempts": attempts,
            "prior_factors": prior, "production_eligible": False, "quarantine_released": False,
            "economic_review_required": True}


def optional_checkpoint(s3, bucket, key):
    try:
        result = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    blob = result["Body"].read()
    if result.get("Metadata", {}).get("sha256") != digest(blob):
        raise ValueError("Checkpoint checksum mismatch")
    return json.loads(blob)


def audit(s3, bucket, *, remaining_run, reconciliation_run, run_id, expected_count=None, workers=8, execute=False, residual_diagnostics_run=None):
    for value in (remaining_run, reconciliation_run, run_id):
        check_id(value)
    if not 1 <= workers <= 16:
        raise ValueError("workers must be 1–16")
    remaining = read_json(s3, bucket, f"{ROOT}/silver/remaining_split_cases/v1/runs/{remaining_run}/manifest.json")
    recon = read_json(s3, bucket, f"{ROOT}/silver/provider_reconciliation/v1/runs/{reconciliation_run}/manifest.json")
    if remaining["inputs"]["reconciliation_events"] != recon["outputs"]["events"]:
        raise ValueError("Reconciliation evidence mismatch")
    cases = verified_frame(s3, bucket, remaining["outputs"]["events"])
    if len(cases) != remaining["statistics"]["remaining"] or cases.duplicated(["asset_id", "date"]).any():
        raise ValueError("Unexpected residual population")
    residual_ref = None
    if residual_diagnostics_run:
        check_id(residual_diagnostics_run)
        diagnostic = read_json(s3, bucket, f"{ROOT}/silver/deterministic_residual_diagnostics/v1/runs/{residual_diagnostics_run}/manifest.json")
        approval = json.loads(verified_bytes(s3, bucket, diagnostic["inputs"]["approval"]))
        registry = json.loads(verified_bytes(s3, bucket, approval["candidate_registry"]))
        if (approval["decision"] != "approved_price_basis_only" or approval["production_eligible"] is not False
                or registry["inputs"]["remaining_events"] != remaining["outputs"]["events"]
                or registry["inputs"]["reconciliation_events"] != recon["outputs"]["events"]
                or approval["events"] != diagnostic["inputs"]["deterministic_events"]):
            raise ValueError("Residual approval source mismatch")
        excluded = {(r["asset_id"], r["date"]) for r in registry["entries"]}
        residual_ref = diagnostic["outputs"]["events"]
        residual = verified_frame(s3, bucket, residual_ref)
        original_ids = set(zip(cases.asset_id, cases.date))
        residual_ids = set(zip(residual.asset_id, residual.date))
        if (len(excluded) != approval["approved_count"] or len(residual_ids) != len(residual)
                or excluded & residual_ids or excluded | residual_ids != original_ids):
            raise ValueError("Residual population does not partition original inventory")
        cases = cases.loc[[(a,d) in residual_ids for a,d in zip(cases.asset_id,cases.date)]]
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError("Unexpected selected residual count")
    pins = recon["inputs"]
    bundle_version = remaining["inputs"]["financial_bundle_version"]
    bundle = load_bundle(s3, bucket, bundle_version)
    factors = prior_factors(bundle)
    consolidated = load_consolidated(s3, bucket, bundle["consolidated_index_version"]) if residual_ref else None
    calendar_records = {}
    if consolidated:
        active, _ = load_universe(s3, bucket, consolidated["source_fingerprint"])
        calendar_records = {r["asset_id"]: r for r in active}
    coverage = verified_frame(s3, bucket, pins["yahoo_coverage"]).set_index("asset_id")
    if not coverage.index.is_unique:
        raise ValueError("Duplicate coverage")
    all_events = verified_frame(s3, bucket, recon["outputs"]["events"])
    splits = all_events.loc[all_events.kind.eq("split") & all_events.yahoo_event_rows.gt(0)]
    if splits.duplicated(["asset_id", "yahoo_date"]).any():
        raise ValueError("Duplicate split evidence")
    by_event = {(r["asset_id"], r["yahoo_date"]): r for r in splits.to_dict("records")}
    by_asset = {a: g.to_dict("records") for a, g in cases.groupby("asset_id")}
    config = read_json(s3, bucket, f"{EODHD_ROOT}/runs/{pins['eodhd_run']}/run.json")
    if digest(json_bytes(config)) != pins["eodhd_run_config_sha256"]:
        raise ValueError("EODHD config changed")
    universe, _ = load_universe(s3, bucket, pins["eodhd_fingerprint"])
    mapped = load_mapping(s3, bucket, [r for r in universe if r["asset_id"] in by_asset], pins["mapping_version"])
    by_mapping = {r["asset_id"]: r for r in mapped}
    absent = set(by_asset) - set(by_mapping)
    if absent - set(pins["yahoo_historical_only_ids"]):
        raise ValueError("Unexpected missing provider mapping")
    scope = f"{SCOPE}/runs/{run_id}"
    inputs = {"remaining_events": remaining["outputs"]["events"], "reconciliation_events": recon["outputs"]["events"],
              "yahoo_coverage": pins["yahoo_coverage"], "eodhd_run_config_sha256": pins["eodhd_run_config_sha256"],
              "financial_bundle_version": bundle_version, "policy": ADVANCED_POLICY if residual_ref else POLICY}
    if residual_ref:
        inputs["residual_diagnostics_events"] = residual_ref
        inputs["prior_approval"] = diagnostic["inputs"]["approval"]
        inputs["consolidated_index_version"] = bundle["consolidated_index_version"]
    fingerprint = digest(json_bytes(inputs))

    def process(aid):
        source = by_asset[aid]
        ids = sorted(r["date"] for r in source)
        key = f"{scope}/checkpoints/{digest(aid.encode())}.json"
        if execute:
            checkpoint = optional_checkpoint(s3, bucket, key)
            if checkpoint is not None:
                if (checkpoint["input_fingerprint"] != fingerprint or checkpoint["asset_id"] != aid
                        or sorted(r["date"] for r in checkpoint["rows"]) != ids
                        or any(r["asset_id"] != aid for r in checkpoint["rows"])):
                    raise ValueError("Checkpoint identity mismatch")
                return checkpoint["rows"], True
        output = []
        if aid in absent:
            for row in source:
                output.append({"asset_id": aid, "date": row["date"], "ratio": row["ratio"], "root_category": row["root_category"],
                               "previous_status": row["status"], "bronze_sha256": row["bronze_sha256"],
                               "status": "historical_provider_coverage_missing", "attempts": [],
                               "chosen_hypothesis": None, "prior_factors": [], "production_eligible": False,
                               "quarantine_released": False, "economic_review_required": True})
        else:
            record, yr = by_mapping[aid], coverage.loc[aid]
            bars = verified_frame(s3, bucket, {"key": yr["bronze_key"], "sha256": yr["bronze_sha256"]})
            if len(bars) != yr["source_rows"] or not bars.asset_id.eq(aid).all() or not bars.provider_symbol.eq(record["yahoo_ticker"]).all():
                raise ValueError(f"Yahoo source mismatch: {aid}")
            receipt = _completed(s3, bucket, run_id=pins["eodhd_run"], record=record, verify_hash=False)
            if receipt is None or receipt["end_exclusive"] != config["end_exclusive"]:
                raise ValueError(f"Missing provider receipt: {aid}")
            blob = verified_bytes(s3, bucket, receipt["data"])
            if verify_bundle(blob, record, config["end_exclusive"]) != receipt["validation"]:
                raise ValueError(f"Provider receipt mismatch: {aid}")
            responses = {k: json.loads(v["raw_json"]) for k, v in json.loads(blob)["responses"].items()}
            y, e = prepare(bars, responses)
            calendar, missing_expected, reconstructed = None, (), None
            if consolidated and aid in calendar_records:
                cal_receipt = consolidated_receipt(s3, bucket, consolidated, calendar_records[aid])
                if cal_receipt["data"]["sha256"] != yr["bronze_sha256"]:
                    raise ValueError("Calendar source mismatch")
                refs = consolidated["assets"][aid]["outputs"]
                calendar = verified_frame(s3, bucket, refs["candidate"])
                findings = verified_frame(s3, bucket, refs["findings"])
                missing_expected = tuple(findings.loc[findings.issue.eq("missing_expected_session"), "session_date"].astype(str))
                share_doc = bundle["share"]
                if share_doc["asset_id"] == aid:
                    reconstructed, _ = reconstruct_share_basis(bars, share_doc, bronze_sha256=yr["bronze_sha256"])
                    reconstructed = reconstructed.set_index("session_date")
            for row in source:
                event = by_event[(aid, row["date"])]
                if (row["bronze_sha256"] != yr["bronze_sha256"] or event["yahoo_sha256"] != yr["bronze_sha256"]
                        or event["eodhd_sha256"] != receipt["data"]["sha256"]
                        or not math.isclose(float(event["yahoo_value"]), float(row["ratio"]), rel_tol=1e-7)):
                    raise ValueError(f"Event source mismatch: {aid} {row['date']}")
                prior = assess_prior(row, factors.get((aid, row["date"]), []))
                context = None
                if reconstructed is not None and row["date"] in reconstructed.index:
                    r = reconstructed.loc[row["date"]]
                    context = clean({k: r[k] for k in ("missing_subsequent_share_factor", "candidate_cash_dividend",
                                                        "candidate_share_event", "unresolved_cash")})
                decision = investigate(y, e, responses, row, event, prior, advanced=bool(residual_ref),
                                       calendar=calendar, missing_expected=missing_expected, share_context=context)
                output.append({"asset_id": aid, "symbol": record["yahoo_ticker"], "date": row["date"], "ratio": row["ratio"],
                               "root_category": row["root_category"], "previous_status": row["status"],
                               "bronze_sha256": yr["bronze_sha256"], "eodhd_sha256": receipt["data"]["sha256"], **decision})
        output = clean(output)
        if execute:
            put_immutable(s3, bucket, key, json_bytes({"asset_id": aid, "input_fingerprint": fingerprint, "rows": output}),
                          content_type="application/json", scope=SCOPE)
        return output, False

    rows, resumed = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (out, reused) in enumerate(pool.map(process, sorted(by_asset)), 1):
            rows.extend(out)
            resumed += int(reused)
            if i % 25 == 0 or i == len(by_asset):
                print(f"[corporate action reconciliation] {i}/{len(by_asset)} assets; resumed={resumed}", file=sys.stderr, flush=True)
    if len(rows) != len(cases) or {(r["asset_id"], r["date"]) for r in rows} != set(zip(cases.asset_id, cases.date)):
        raise ValueError("Incomplete reconciliation inventory")
    candidates = [r for r in rows if r["status"] == "reconciled_price_basis_candidate"]
    by_category = defaultdict(Counter)
    attempt_counts = Counter()
    for row in rows:
        by_category[row["root_category"]][row["status"]] += 1
        attempt_counts.update(a["status"] for a in row["attempts"])
    outputs = {}
    if execute:
        table = pd.DataFrame([{**r, "attempts": json.dumps(r["attempts"], sort_keys=True),
                               "chosen_hypothesis": json.dumps(r["chosen_hypothesis"], sort_keys=True),
                               "prior_factors": json.dumps(r["prior_factors"], sort_keys=True)} for r in rows])
        blob = parquet_bytes(table)
        outputs["events"] = {**put_immutable(s3, bucket, f"{scope}/objects/events-{digest(blob)}.parquet", blob,
                                             content_type="application/vnd.apache.parquet", scope=SCOPE),
                             "rows": len(rows), "format": "parquet"}
        registry = {"schema": "alpha_edge_deterministic_corporate_action_candidate_registry_v1", "inputs": inputs,
                    "entries": candidates, "production_eligible": False, "quarantine_released": False}
        body = json_bytes(registry)
        root = f"{ROOT}/config/deterministic_corporate_action_candidate_registries/v1"
        outputs["candidate_registry"] = put_immutable(s3, bucket, f"{root}/objects/{digest(body)}.json", body,
                                                       content_type="application/json", scope=root)
    result = {"schema": "alpha_edge_deterministic_corporate_action_reconciliation_v1", "run_id": run_id,
              "inputs": inputs, "statistics": {"events": len(rows), "assets": len(by_asset), "candidate_events": len(candidates),
                    "statuses": dict(Counter(r["status"] for r in rows)), "categories": {k: dict(v) for k,v in by_category.items()},
                    "attempt_statuses_nonexclusive": dict(attempt_counts), "prior_factor_cases": sum(bool(r["prior_factors"]) for r in rows)},
              "outputs": outputs, "production_eligible": False, "quarantine_released": False}
    key = f"{scope}/manifest.json"
    if execute:
        put_immutable(s3, bucket, key, json_bytes(result), content_type="application/json", scope=SCOPE)
    return {**result, "manifest_key": key, "status": "candidate_written" if execute else "dry_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--remaining-run", required=True)
    parser.add_argument("--reconciliation-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--residual-diagnostics-run", help="Apply v2 hypotheses only to the prior approved residual inventory")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(boto3.client("s3", region_name=args.region), args.bucket,
                           remaining_run=args.remaining_run, reconciliation_run=args.reconciliation_run,
                           run_id=args.run_id, expected_count=args.expected_count, workers=args.workers, execute=args.execute, residual_diagnostics_run=args.residual_diagnostics_run), indent=2))


if __name__ == "__main__":
    main()
