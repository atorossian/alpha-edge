"""Reconcile residual simultaneous cash/share events and prior documented factors."""

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

from alpha_edge.market.audit_matching_split_basis_v3 import (
    FIELDS, TOL, WINDOW, close_to, positive_matrix, prepare,
)
from alpha_edge.market.audit_provider_reconciliation_v3 import number, split_ratio
from alpha_edge.market.build_silver_adjusted_v3 import load_bundle
from alpha_edge.market.eodhd_source_v3 import load_mapping, verify_bundle
from alpha_edge.market.ingest_bronze_full_history import EODHD_ROOT, _completed, load_universe
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT, check_id, digest, json_bytes, parquet_bytes, put_immutable, read_json,
    verified_bytes, verified_frame,
)

SCOPE = f"{ROOT}/silver/combined_cash_share_candidates/v1"
POLICY = "combined_cash_share_independent_factor_steps_v1"
CANDIDATE = "combined_cash_share_price_basis_candidate"


def prior_factors(bundle):
    index = defaultdict(list)
    for doc in bundle["events"]:
        for entry in doc["entries"]:
            for factor in entry["basis_factors"]:
                index[(entry["asset_id"], factor["effective_date"])].append({
                    **factor, "source_sha256": entry["source_sha256"],
                    "provider_symbol": entry["provider_symbol"],
                    "issuer_dividend_date": entry["issuer_ex_date"],
                })
    return index


def assess_prior(row, factors):
    pinned = [f for f in factors if f["source_sha256"] == row["bronze_sha256"]]
    findings = []
    for f in pinned:
        value = float(f["factor"])
        ratio = float(row["ratio"])
        if f["status"] != "verified":
            status = "provisional_factor_review"
        elif f["kind"] == "share_ratio":
            status = "documented_share_ratio_match" if math.isclose(value, ratio, rel_tol=0.0001) else "documented_share_ratio_conflict"
        elif f["kind"] == "price_coefficient":
            status = "documented_price_coefficient_reciprocal_match" if math.isclose(1/value, ratio, rel_tol=0.0001) else "documented_price_coefficient_other"
        else:
            status = "unknown_factor_kind"
        findings.append({**f, "comparison": status})
    return findings


def screen_cash_share(y, e, responses, row, split_event, *, cash_basis="post"):
    """Confirm both cash and price-basis steps; never infer legal share issuance."""
    day, ratio = row["date"], float(row["ratio"])
    details = {"policy": POLICY, "status": "not_evaluated", "production_eligible": False,
               "quarantine_released": False, "economic_review_required": True}

    def finish(status):
        return {**details, "status": status}

    if not math.isfinite(ratio) or ratio <= 0 or abs(math.log(ratio)) <= 2*TOL:
        return finish("ratio_invalid_or_indistinguishable")
    if y.index.has_duplicates or e.index.has_duplicates or day not in y.index or day not in e.index:
        return finish("missing_or_duplicate_session")
    jy, je = y.index.get_loc(day), e.index.get_loc(day)
    if jy < WINDOW or je < WINDOW or jy + WINDOW > len(y) or je + WINDOW > len(e):
        return finish("insufficient_window")
    yw, ew = y.iloc[jy-WINDOW:jy+WINDOW], e.iloc[je-WINDOW:je+WINDOW]
    if not yw.index.equals(ew.index):
        return finish("session_windows_disagree")
    if not math.isclose(float(y.loc[day, "stock_splits"]), ratio, rel_tol=1e-7):
        raise ValueError("Yahoo split ratio changed")
    y_shares = set(yw.index[pd.to_numeric(yw.stock_splits, errors="coerce").fillna(0).ne(0)])
    e_shares = [r for r in responses["splits"] if ew.index[0] <= r["date"] <= ew.index[-1]]
    if y_shares != {day}:
        return finish("overlapping_yahoo_share_events")
    if split_event["date_status"] == "exact_date" and split_event["value_status"] == "ratio_match":
        if len(e_shares) != 1 or e_shares[0]["date"] != day or not close_to(split_ratio(e_shares[0]["split"]) or 0, ratio):
            return finish("conflicting_eodhd_share_events")
    elif split_event["date_status"] == "yahoo_only":
        if e_shares:
            return finish("conflicting_eodhd_share_events")
    else:
        return finish("share_date_or_ratio_review")
    y_cash = number(y.loc[day, "dividends"])
    e_cash = [r for r in responses["div"] if r["date"] == day]
    y_other = [str(d) for d in yw.index if d != day and number(yw.loc[d, "dividends"]) not in (None, 0)]
    e_other = [r["date"] for r in responses["div"] if ew.index[0] <= r["date"] <= ew.index[-1] and r["date"] != day]
    details.update({"yahoo_cash": y_cash, "eodhd_cash_rows_json": json.dumps(e_cash, sort_keys=True),
                    "nearby_yahoo_cash_dates_json": json.dumps(y_other),
                    "nearby_eodhd_cash_dates_json": json.dumps(e_other)})
    if y_other or e_other:
        return finish("nearby_cash_requires_separate_reconciliation")
    if y_cash is None or y_cash <= 0 or len(e_cash) != 1:
        return finish("cash_not_uniquely_reported_by_both")
    cash = e_cash[0]
    currency = y.loc[day, "currency_original"]
    raw_cash = number(cash.get("unadjustedValue"))
    if not isinstance(currency, str) or cash.get("currency") != currency:
        return finish("cash_currency_unverified")
    if raw_cash is None or raw_cash <= 0:
        return finish("eodhd_original_cash_unavailable")
    ym, em = positive_matrix(yw, FIELDS), positive_matrix(ew, FIELDS)
    ya, ea = positive_matrix(yw, ["adj_close"]), positive_matrix(ew, ["adjusted_close"])
    if any(x is None for x in (ym, em, ya, ea)):
        return finish("invalid_price_window")
    scales = em / ym
    pre, post = float(np.median(scales[:WINDOW])), float(np.median(scales[WINDOW:]))
    pre_error = float(np.max(np.abs(scales[:WINDOW] / pre - 1)))
    post_error = float(np.max(np.abs(scales[WINDOW:] / post - 1)))
    implied = pre/post
    y_step, e_step = float(ym[WINDOW, 3]/ym[WINDOW-1, 3]), float(em[WINDOW, 3]/em[WINDOW-1, 3])
    yf = float((ya[WINDOW, 0]/ym[WINDOW, 3])/(ya[WINDOW-1, 0]/ym[WINDOW-1, 3]))
    ef = float((ea[WINDOW, 0]/em[WINDOW, 3])/(ea[WINDOW-1, 0]/em[WINDOW-1, 3]))
    # EODHD original cash is tested on the post-event quote basis. No alternative
    # basis is guessed when that hypothesis fails.
    if cash_basis not in {"pre", "post"}:
        raise ValueError("Unknown cash basis")
    e_cash_normalized = raw_cash / (pre if cash_basis == "pre" else post)
    details.update({"pre_scale": pre, "post_scale": post, "implied_ratio": implied,
                    "pre_max_relative_error": pre_error, "post_max_relative_error": post_error,
                    "yahoo_adjustment_factor_step": yf, "eodhd_adjustment_factor_step": ef,
                    "yahoo_close_step": y_step, "eodhd_close_step": e_step,
                    "eodhd_original_cash": raw_cash, "eodhd_cash_on_yahoo_basis": e_cash_normalized,
                    "cash_currency": currency, "cash_basis_hypothesis": cash_basis,
                    "eodhd_adjusted_cash": number(cash.get("value"))})
    if not all(math.isfinite(x) for x in (pre, post, implied, pre_error, post_error, yf, ef, y_step, e_step, e_cash_normalized)):
        return finish("invalid_price_window")
    if pre_error > TOL or post_error > TOL:
        return finish("ohlc_window_disagreement")
    if not close_to(implied, ratio) or not close_to((e_step/y_step)*ratio, 1.0):
        return finish("implied_share_ratio_disagreement")
    if not close_to(e_cash_normalized, y_cash):
        return finish("cash_amount_or_share_basis_disagreement")
    cash_fraction_y = y_cash/float(ym[WINDOW-1, 3])
    cash_fraction_e = raw_cash / (float(em[WINDOW-1, 3]) if cash_basis == "pre" else float(em[WINDOW-1, 3])/ratio)
    if not (0 < cash_fraction_y < 1 and 0 < cash_fraction_e < 1):
        return finish("invalid_cash_fraction")
    expected_y, expected_e = 1/(1-cash_fraction_y), ratio/(1-cash_fraction_e)
    details.update({"expected_yahoo_factor_step": expected_y, "expected_eodhd_factor_step": expected_e,
                    "cash_fraction_yahoo": cash_fraction_y, "cash_fraction_eodhd": cash_fraction_e})
    if abs(math.log(expected_y)) <= 2*TOL:
        return finish("cash_factor_too_small_to_distinguish")
    if not close_to(yf, expected_y) or not close_to(ef, expected_e):
        return finish("cash_and_share_factor_disagreement")
    if abs(math.log(y_step*expected_y)) > 0.10:
        return finish("large_cash_adjusted_session_return_review")
    return finish(CANDIDATE)


def audit(s3, bucket, *, remaining_run, reconciliation_run, run_id, expected_count, workers=8, execute=False):
    for v in (remaining_run, reconciliation_run, run_id):
        check_id(v)
    if not 1 <= workers <= 16:
        raise ValueError("workers must be 1–16")
    remaining = read_json(s3, bucket, f"{ROOT}/silver/remaining_split_cases/v1/runs/{remaining_run}/manifest.json")
    recon = read_json(s3, bucket, f"{ROOT}/silver/provider_reconciliation/v1/runs/{reconciliation_run}/manifest.json")
    if remaining["inputs"]["reconciliation_events"] != recon["outputs"]["events"]:
        raise ValueError("Remaining and reconciliation source mismatch")
    cases = verified_frame(s3, bucket, remaining["outputs"]["events"])
    selected = cases.loc[cases.root_category.eq("cash_and_share") | cases.registry_evidence_status.eq("exact_registry_date")]
    if len(selected) != expected_count or selected.duplicated(["asset_id", "date"]).any():
        raise ValueError("Unexpected combined cash/share population")
    bundle_version = remaining["inputs"]["financial_bundle_version"]
    bundle = load_bundle(s3, bucket, bundle_version)
    factors = prior_factors(bundle)
    pins = recon["inputs"]
    coverage = verified_frame(s3, bucket, pins["yahoo_coverage"]).set_index("asset_id")
    events = verified_frame(s3, bucket, recon["outputs"]["events"])
    splits = events.loc[events.kind.eq("split") & events.yahoo_event_rows.gt(0)]
    if splits.duplicated(["asset_id", "yahoo_date"]).any():
        raise ValueError("Duplicate source split identities")
    by_event = {(r["asset_id"], r["yahoo_date"]): r for r in splits.to_dict("records")}
    by_asset = {a: g.to_dict("records") for a, g in selected.groupby("asset_id")}
    config = read_json(s3, bucket, f"{EODHD_ROOT}/runs/{pins['eodhd_run']}/run.json")
    if digest(json_bytes(config)) != pins["eodhd_run_config_sha256"]:
        raise ValueError("EODHD config changed")
    universe, _ = load_universe(s3, bucket, pins["eodhd_fingerprint"])
    mapped = load_mapping(s3, bucket, [r for r in universe if r["asset_id"] in by_asset], pins["mapping_version"])
    if {r["asset_id"] for r in mapped} != set(by_asset):
        raise ValueError("Missing provider mapping")

    def process(record):
        aid = record["asset_id"]
        yr = coverage.loc[aid]
        bars = verified_frame(s3, bucket, {"key": yr["bronze_key"], "sha256": yr["bronze_sha256"]})
        if len(bars) != yr["source_rows"] or not bars.asset_id.eq(aid).all() or not bars.provider_symbol.eq(record["yahoo_ticker"]).all():
            raise ValueError(f"Yahoo source mismatch: {aid}")
        receipt = _completed(s3, bucket, run_id=pins["eodhd_run"], record=record, verify_hash=False)
        if receipt is None or receipt["end_exclusive"] != config["end_exclusive"]:
            raise ValueError(f"Missing EODHD receipt: {aid}")
        blob = verified_bytes(s3, bucket, receipt["data"])
        if verify_bundle(blob, record, config["end_exclusive"]) != receipt["validation"]:
            raise ValueError(f"EODHD receipt mismatch: {aid}")
        responses = {k: json.loads(v["raw_json"]) for k, v in json.loads(blob)["responses"].items()}
        y, e = prepare(bars, responses)
        out = []
        for row in by_asset[aid]:
            key = (aid, row["date"])
            event = by_event[key]
            if row["bronze_sha256"] != yr["bronze_sha256"] or event["yahoo_sha256"] != yr["bronze_sha256"] or event["eodhd_sha256"] != receipt["data"]["sha256"]:
                raise ValueError(f"Stale event evidence: {key}")
            prior = assess_prior(row, factors.get(key, []))
            if row["registry_evidence_status"] == "exact_registry_date" and not prior:
                raise ValueError(f"Missing prior factor evidence: {key}")
            decision = (screen_cash_share(y, e, responses, row, event) if row["root_category"] == "cash_and_share"
                        else {"status": "prior_factor_review_only", "policy": POLICY, "production_eligible": False,
                              "quarantine_released": False, "economic_review_required": True})
            out.append({"asset_id": aid, "symbol": record["yahoo_ticker"], "date": row["date"], "ratio": row["ratio"],
                        "bronze_sha256": yr["bronze_sha256"], "eodhd_sha256": receipt["data"]["sha256"],
                        "previous_status": row["status"], "root_category": row["root_category"],
                        "prior_factors_json": json.dumps(prior, sort_keys=True), **decision})
        return out

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, group in enumerate(pool.map(process, mapped), 1):
            rows.extend(group)
            if i % 25 == 0 or i == len(mapped):
                print(f"[combined cash/share] {i}/{len(mapped)}", file=sys.stderr, flush=True)
    if len(rows) != expected_count:
        raise ValueError("Incomplete combined cash/share audit")
    prior_rows = [r for r in rows if json.loads(r["prior_factors_json"])]
    candidates = [r for r in rows if r["status"] == CANDIDATE]
    scope = f"{SCOPE}/runs/{run_id}"
    outputs = {}
    if execute:
        blob = parquet_bytes(pd.DataFrame(rows))
        key = f"{scope}/objects/events-{digest(blob)}.parquet"
        outputs["events"] = {**put_immutable(s3, bucket, key, blob, content_type="application/vnd.apache.parquet", scope=SCOPE),
                             "rows": len(rows), "format": "parquet"}
        obj = {"schema": "alpha_edge_combined_cash_share_candidate_registry_v1", "policy": POLICY,
               "remaining_events": remaining["outputs"]["events"], "financial_bundle_version": bundle_version,
               "entries": candidates, "production_eligible": False, "quarantine_released": False}
        body = json_bytes(obj)
        root = f"{ROOT}/config/combined_cash_share_candidate_registries/v1"
        outputs["candidate_registry"] = put_immutable(s3, bucket, f"{root}/objects/{digest(body)}.json", body,
                                                       content_type="application/json", scope=root)
    result = {"schema": "alpha_edge_combined_cash_share_candidates_v1", "run_id": run_id, "policy": POLICY,
              "inputs": {"remaining_events": remaining["outputs"]["events"], "reconciliation_events": recon["outputs"]["events"],
                         "financial_bundle_version": bundle_version},
              "statistics": {"events": len(rows), "assets": len(mapped), "candidates": len(candidates),
                             "statuses": dict(Counter(r["status"] for r in rows)),
                             "prior_factor_cases": len(prior_rows),
                             "prior_factor_comparisons": dict(Counter(f["comparison"] for r in prior_rows for f in json.loads(r["prior_factors_json"])))},
              "prior_factor_cases": prior_rows, "outputs": outputs, "production_eligible": False, "quarantine_released": False}
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
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(boto3.client("s3", region_name=args.region), args.bucket,
                           remaining_run=args.remaining_run, reconciliation_run=args.reconciliation_run,
                           run_id=args.run_id, expected_count=args.expected_count, workers=args.workers,
                           execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
