"""Screen exact Yahoo/EODHD split matches for batch price-basis approval candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pandas as pd

from alpha_edge.market.audit_provider_reconciliation_v3 import split_ratio
from alpha_edge.market.audit_split_classification_v3 import SPLIT, classify
from alpha_edge.market.eodhd_source_v3 import load_mapping, verify_bundle
from alpha_edge.market.ingest_bronze_full_history import EODHD_ROOT, _completed, load_universe
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT,
    check_id,
    digest,
    json_bytes,
    parquet_bytes,
    put_immutable,
    read_json,
    verified_bytes,
    verified_frame,
)

SCOPE = f"{ROOT}/silver/split_basis_candidates/v1"
POLICY = "integer_split_five_session_ohlc_basis_v1"
WINDOW = 5
TOL = 0.005
FIELDS = ["open", "high", "low", "close"]
CANDIDATE = "provider_normalized_candidate"
FRACTIONAL = "fractional_price_basis_candidate"
FRACTIONAL_POLICY = "fractional_five_session_ohlc_basis_economic_review_v1"
SMALL_POLICY = "small_fraction_cross_provider_ohlc_basis_v1"
SMALL_CANDIDATE = "small_fraction_price_basis_candidate"
IRREGULAR_POLICY = "irregular_cross_provider_ohlc_basis_economic_review_v1"
IRREGULAR_CANDIDATE = "irregular_price_basis_candidate"
YAHOO_ONLY_POLICY = "yahoo_only_implicit_cross_provider_price_basis_v1"
YAHOO_ONLY_CANDIDATE = "yahoo_only_implicit_price_basis_candidate"


def is_small_fraction(ratio):
    return any(math.isclose(ratio, r, rel_tol=1e-7) for r in (1.05, 1.10))


def close_to(a, b):
    return math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0 and math.isclose(a, b, rel_tol=TOL)


def positive_matrix(frame, columns):
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return values if np.isfinite(values).all() and (values > 0).all() else None


def prepare(yahoo, responses):
    y = yahoo.copy()
    y["date"] = pd.to_datetime(y.session_date).dt.strftime("%Y-%m-%d")
    y = y.set_index("date").sort_index()
    e = pd.DataFrame(responses["eod"]).set_index("date").sort_index()
    return y, e


def screen_event(y, e, responses, event, screen, *, fractional=False, small_fraction=False, irregular=False, yahoo_only=False, recheck_continuity=False):
    day = event["yahoo_date"]
    ratio = float(event["yahoo_value"])
    group = classify(ratio)
    policy = YAHOO_ONLY_POLICY if yahoo_only else IRREGULAR_POLICY if irregular else SMALL_POLICY if small_fraction else FRACTIONAL_POLICY if fractional else POLICY
    details = {"group": group, "date": day, "ratio": ratio, "policy": policy,
               "status": "not_evaluated", "production_eligible": False}

    def finish(status):
        return {**details, "status": status}

    if sum((fractional, small_fraction, irregular, yahoo_only)) > 1:
        raise ValueError("Choose one screening mode")
    if small_fraction and not is_small_fraction(ratio):
        return finish("outside_small_fraction_scope")
    if yahoo_only and group == "invalid_or_unit_ratio":
        return finish("invalid_or_unit_ratio")
    if not yahoo_only and group not in ({"irregular_ratio"} if irregular else {"simple_fraction"} if fractional or small_fraction else {"integer_forward", "integer_reverse"}):
        return finish("non_integer_economic_review")
    if bool(screen["same_day_dividend"]) or any(r["date"] == day for r in responses["div"]):
        return finish("same_day_cash_event")
    if not (small_fraction or irregular or yahoo_only or recheck_continuity) and screen["continuity"] != "provider_normalized_signal":
        return finish("yahoo_continuity_not_confirmed")
    if y.index.has_duplicates or e.index.has_duplicates or day not in y.index or day not in e.index:
        return finish("missing_or_duplicate_session")
    jy, je = y.index.get_loc(day), e.index.get_loc(day)
    if jy < WINDOW or je < WINDOW or jy + WINDOW > len(y) or je + WINDOW > len(e):
        return finish("insufficient_window")
    yw, ew = y.iloc[jy-WINDOW:jy+WINDOW], e.iloc[je-WINDOW:je+WINDOW]
    if not yw.index.equals(ew.index):
        return finish("session_windows_disagree")
    split_dates_y = set(yw.index[pd.to_numeric(yw.stock_splits, errors="coerce").fillna(0).ne(0)])
    split_dates_e = [r["date"] for r in responses["splits"] if r["date"] in ew.index]
    if yahoo_only:
        split_dates_e = [r["date"] for r in responses["splits"] if ew.index[0] <= r["date"] <= ew.index[-1]]
        if split_dates_y != {day} or split_dates_e:
            return finish("multiple_or_conflicting_share_events_in_window")
        if any(ew.index[0] <= r["date"] <= ew.index[-1] for r in responses["div"]):
            return finish("nearby_eodhd_cash_event")
        if "dividends" in yw and pd.to_numeric(yw.dividends, errors="coerce").fillna(0).ne(0).any():
            return finish("nearby_yahoo_cash_event")
    elif split_dates_y != {day} or split_dates_e != [day]:
        return finish("multiple_or_missing_share_events_in_window")
    # The events are pinned to their original source bytes; also assert the raw ratios.
    if not math.isclose(float(y.loc[day, "stock_splits"]), ratio, rel_tol=0.0001):
        raise ValueError("Yahoo event ratio changed")
    if not yahoo_only:
        er = next(r for r in responses["splits"] if r["date"] == day)
        if not close_to(split_ratio(er["split"]) or 0, ratio):
            raise ValueError("EODHD event ratio changed")
    ym, em = positive_matrix(yw, FIELDS), positive_matrix(ew, FIELDS)
    ya, ea = positive_matrix(yw, ["adj_close"]), positive_matrix(ew, ["adjusted_close"])
    if any(x is None for x in (ym, em, ya, ea)):
        return finish("invalid_price_window")
    level = em / ym
    pre, post = float(np.median(level[:WINDOW])), float(np.median(level[WINDOW:]))
    error_pre = float(np.max(np.abs(level[:WINDOW] / pre - 1)))
    error_post = float(np.max(np.abs(level[WINDOW:] / post - 1)))
    implied = pre / post
    y_step = float(ym[WINDOW, 3] / ym[WINDOW-1, 3])
    e_step = float(em[WINDOW, 3] / em[WINDOW-1, 3])
    yf = float((ya[WINDOW, 0] / ym[WINDOW, 3]) / (ya[WINDOW-1, 0] / ym[WINDOW-1, 3]))
    ef = float((ea[WINDOW, 0] / em[WINDOW, 3]) / (ea[WINDOW-1, 0] / em[WINDOW-1, 3]))
    details.update({"window_start": str(yw.index[0]), "window_end": str(yw.index[-1]),
                    "pre_price_scale": pre, "post_price_scale": post,
                    "pre_max_relative_error": error_pre, "post_max_relative_error": error_post,
                    "implied_ratio": implied, "yahoo_close_step": y_step, "eodhd_close_step": e_step,
                    "yahoo_adjustment_factor_step": yf, "eodhd_adjustment_factor_step": ef})
    if not all(math.isfinite(x) for x in (pre, post, error_pre, error_post, implied, y_step, e_step, yf, ef)):
        return finish("invalid_price_window")
    if error_pre > TOL or error_post > TOL:
        return finish("ohlc_window_disagreement")
    if yahoo_only:
        if abs(math.log(ratio)) <= 2 * TOL:
            return finish("ratio_too_small_to_distinguish_from_unit")
        if close_to(implied, 1.0) and close_to(yf, 1.0) and close_to(ef, 1.0):
            return finish("no_independent_split_basis_change")
    if not close_to(implied, ratio) or not close_to((e_step / y_step) * ratio, 1.0):
        return finish("implied_share_ratio_disagreement")
    if not close_to(yf, 1.0) or not close_to(ef, ratio):
        return finish("adjustment_factor_disagreement")
    if (small_fraction or irregular or yahoo_only or recheck_continuity) and abs(math.log(y_step)) > 0.10:
        return finish("large_event_session_return_review")
    # Cross-provider OHLC and factor-step checks replace the impossible 0.10 log-margin gate.
    # This corroborates price basis, not the legal identity of every corporate action.
    return finish(YAHOO_ONLY_CANDIDATE if yahoo_only else IRREGULAR_CANDIDATE if irregular else SMALL_CANDIDATE if small_fraction else FRACTIONAL if fractional else CANDIDATE)


def recovery_bound(y, findings, approved_dates):
    """Marginal source-row upper bound; never release an existing quarantine."""
    prefix = findings.loc[findings.scope.eq("prefix")]
    if prefix.empty:
        return 0
    dates = pd.to_datetime(y.session_date).dt.strftime("%Y-%m-%d")
    old = prefix.effective_date.max()
    keep = ~(prefix.reason.eq(SPLIT) & prefix.effective_date.isin(approved_dates))
    remaining = prefix.loc[keep, "effective_date"]
    valid = pd.to_numeric(y.close, errors="coerce").gt(0) & pd.to_numeric(y.adj_close, errors="coerce").gt(0)
    valid &= np.isfinite(pd.to_numeric(y.close, errors="coerce")) & np.isfinite(pd.to_numeric(y.adj_close, errors="coerce"))
    return int((valid & dates.le(old) & (dates.gt(remaining.max()) if not remaining.empty else True)).sum())


def audit(s3, bucket, *, reconciliation_run, screening_run, run_id, workers=8, execute=False, fractional=False, small_fraction=False, irregular=False, yahoo_only=False):
    for value in (reconciliation_run, screening_run, run_id):
        check_id(value)
    if not 1 <= workers <= 16:
        raise ValueError("workers must be 1–16")
    recon = read_json(s3, bucket, f"{ROOT}/silver/provider_reconciliation/v1/runs/{reconciliation_run}/manifest.json")
    screening = read_json(s3, bucket, f"{ROOT}/silver/split_screening/v1/runs/{screening_run}/manifest.json")
    pins = recon["inputs"]
    if (recon.get("schema") != "alpha_edge_provider_reconciliation_v1"
            or screening.get("schema") != "alpha_edge_split_screening_v1"
            or screening["diagnostic_run_id"] != pins["yahoo_diagnostic"]):
        raise ValueError("Reconciliation/screening input mismatch")
    diagnostic = read_json(s3, bucket, f"{ROOT}/silver/financial_diagnostics/v1/runs/{pins['yahoo_diagnostic']}/manifest.json")
    if diagnostic["outputs"]["coverage"] != pins["yahoo_coverage"]:
        raise ValueError("Yahoo coverage mismatch")
    coverage = verified_frame(s3, bucket, pins["yahoo_coverage"])
    quarantine = verified_frame(s3, bucket, diagnostic["outputs"]["quarantine_findings"])
    events = verified_frame(s3, bucket, recon["outputs"]["events"])
    matched = events.loc[events.kind.eq("split") & events.date_status.eq("exact_date") & events.value_status.eq("ratio_match")]
    expected = recon["statistics"]["split"]["exact_date_value_status"]["ratio_match"]
    if len(matched) != expected or matched.duplicated(["asset_id", "yahoo_date"]).any():
        raise ValueError("Exact split population mismatch")
    screens = verified_frame(s3, bucket, screening["event_output"])
    if screens.duplicated(["asset_id", "date"]).any() or coverage.asset_id.duplicated().any():
        raise ValueError("Duplicate source evidence")
    by_screen = {(x["asset_id"], x["date"]): x for x in screens.to_dict("records")}
    if sum((fractional, small_fraction, irregular, yahoo_only)) > 1:
        raise ValueError("Choose one screening mode")
    if yahoo_only:
        matched = events.loc[events.kind.eq("split") & events.date_status.eq("yahoo_only") & events.yahoo_event_rows.eq(1)]
        expected = recon["statistics"]["split"]["date_status"]["yahoo_only"]
        if len(matched) != expected or matched.duplicated(["asset_id", "yahoo_date"]).any():
            raise ValueError("Yahoo-only split population mismatch")
    elif irregular:
        matched = matched.loc[matched.yahoo_value.map(lambda v: classify(float(v)) == "irregular_ratio")]
        expected = len(matched)
    elif small_fraction:
        matched = matched.loc[matched.yahoo_value.map(lambda v: is_small_fraction(float(v)))]
        expected = len(matched)
    elif fractional:
        matched = matched.loc[matched.yahoo_value.map(lambda v: classify(float(v)) == "simple_fraction")]
        expected = len(matched)
    policy = YAHOO_ONLY_POLICY if yahoo_only else IRREGULAR_POLICY if irregular else SMALL_POLICY if small_fraction else FRACTIONAL_POLICY if fractional else POLICY
    candidate_status = YAHOO_ONLY_CANDIDATE if yahoo_only else IRREGULAR_CANDIDATE if irregular else SMALL_CANDIDATE if small_fraction else FRACTIONAL if fractional else CANDIDATE
    by_asset = {a: g.to_dict("records") for a, g in matched.groupby("asset_id")}
    by_cov = coverage.set_index("asset_id").to_dict("index")
    by_q = {a: g for a, g in quarantine.groupby("asset_id")}
    config = read_json(s3, bucket, f"{EODHD_ROOT}/runs/{pins['eodhd_run']}/run.json")
    if digest(json_bytes(config)) != pins["eodhd_run_config_sha256"]:
        raise ValueError("EODHD config changed")
    universe, _ = load_universe(s3, bucket, pins["eodhd_fingerprint"])
    mapped = load_mapping(s3, bucket, [r for r in universe if r["asset_id"] in by_asset], pins["mapping_version"])
    if {r["asset_id"] for r in mapped} != set(by_asset):
        raise ValueError("Missing mapped split population")

    def process(row):
        aid = row["asset_id"]
        yr = by_cov[aid]
        yf = verified_frame(s3, bucket, {"key": yr["bronze_key"], "sha256": yr["bronze_sha256"]})
        if len(yf) != yr["source_rows"] or not yf.asset_id.eq(aid).all() or not yf.provider_symbol.eq(row["yahoo_ticker"]).all():
            raise ValueError(f"Yahoo identity mismatch: {aid}")
        receipt = _completed(s3, bucket, run_id=pins["eodhd_run"], record=row, verify_hash=False)
        if receipt is None or receipt["end_exclusive"] != config["end_exclusive"]:
            raise ValueError(f"Missing EODHD receipt: {aid}")
        blob = verified_bytes(s3, bucket, receipt["data"])
        if verify_bundle(blob, row, config["end_exclusive"]) != receipt["validation"]:
            raise ValueError(f"EODHD validation mismatch: {aid}")
        responses = {k: json.loads(v["raw_json"]) for k, v in json.loads(blob)["responses"].items()}
        y, e = prepare(yf, responses)
        findings = []
        for event in by_asset[aid]:
            screen = by_screen[(aid, event["yahoo_date"])]
            if (event["yahoo_sha256"] != yr["bronze_sha256"] or screen["bronze_sha256"] != yr["bronze_sha256"]
                    or event["eodhd_sha256"] != receipt["data"]["sha256"]
                    or not math.isclose(float(screen["ratio"]), float(event["yahoo_value"]), rel_tol=0.0001)):
                raise ValueError(f"Event source evidence mismatch: {aid}")
            decision = screen_event(y, e, responses, event, screen, fractional=fractional, small_fraction=small_fraction, irregular=irregular, yahoo_only=yahoo_only)
            findings.append({"asset_id": aid, "yahoo_symbol": row["yahoo_ticker"], "eodhd_symbol": row["provider_symbol"],
                             "yahoo_sha256": yr["bronze_sha256"], "eodhd_sha256": receipt["data"]["sha256"],
                             **decision})
        dates = {x["date"] for x in findings if x["status"] == candidate_status}
        bound = recovery_bound(yf, by_q[aid], dates)
        return findings, {"asset_id": aid, "symbol": row["yahoo_ticker"], "candidate_events": len(dates),
                          "source_rows_potentially_unblocked": bound}

    decisions, assets = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (rows, asset) in enumerate(pool.map(process, mapped), 1):
            decisions.extend(rows)
            assets.append(asset)
            if i % 50 == 0 or i == len(mapped):
                print(f"[split basis] {i}/{len(mapped)}", file=sys.stderr, flush=True)
    if len(decisions) != expected:
        raise ValueError("Incomplete split candidate audit")
    counts = Counter(x["status"] for x in decisions)
    groups = {}
    for x in decisions:
        groups.setdefault(x["group"], Counter())[x["status"]] += 1
    ratios = {}
    for x in decisions:
        label = str(Fraction(x["ratio"]).limit_denominator(20)) if fractional or small_fraction else str(x["ratio"])
        if irregular:
            r = x["ratio"]
            label = "reverse" if r < 1 else "1_to_1.1" if r <= 1.1 else "1.1_to_1.5" if r <= 1.5 else "above_1.5"
        ratios.setdefault(label, Counter())[x["status"]] += 1
    scope = f"{SCOPE}/runs/{run_id}"
    outputs = {}
    candidates = [x for x in decisions if x["status"] == candidate_status]
    if execute:
        for name, rows in (("events", decisions), ("assets", assets)):
            blob = parquet_bytes(pd.DataFrame(rows))
            key = f"{scope}/objects/{name}-{digest(blob)}.parquet"
            outputs[name] = {**put_immutable(s3, bucket, key, blob, content_type="application/vnd.apache.parquet",
                                             scope=SCOPE), "rows": len(rows), "format": "parquet"}
        # Explicitly candidate-only; this is not a production split registry.
        obj = {"schema": "alpha_edge_split_basis_candidate_registry_v1", "policy": policy,
               "reconciliation_events": recon["outputs"]["events"],
               "screening_events": screening["event_output"],
               "quarantine_findings": diagnostic["outputs"]["quarantine_findings"],
               "production_eligible": False, "quarantine_released": False, "entries": candidates}
        body = json_bytes(obj)
        key = f"{ROOT}/config/split_basis_candidate_registries/v1/objects/{digest(body)}.json"
        outputs["candidate_registry"] = put_immutable(s3, bucket, key, body, content_type="application/json",
                                                       scope=f"{ROOT}/config/split_basis_candidate_registries/v1")
    result = {"schema": "alpha_edge_split_basis_candidate_v1", "run_id": run_id, "policy": policy,
              "inputs": {"reconciliation_run": reconciliation_run, "reconciliation_events": recon["outputs"]["events"],
                         "screening_run": screening_run, "screening_events": screening["event_output"],
                         "quarantine_findings": diagnostic["outputs"]["quarantine_findings"]},
              "statistics": {"events": len(decisions), "assets": len(assets), "statuses": dict(counts),
                             "groups": {k: dict(v) for k, v in groups.items()},
                             "ratios": {k: dict(v) for k, v in ratios.items()} if fractional or small_fraction else {},
                             **({"ratio_bands": {k: dict(v) for k, v in ratios.items()}} if irregular else {}),
                             "economic_review_required": fractional or small_fraction or irregular or yahoo_only,
                             "candidate_assets": sum(x["candidate_events"] > 0 for x in assets),
                             "source_rows_potentially_unblocked": sum(x["source_rows_potentially_unblocked"] for x in assets)},
              "production_eligible": False, "quarantine_released": False, "outputs": outputs,
              "manifest_key": f"{scope}/manifest.json"}
    if execute:
        put_immutable(s3, bucket, result["manifest_key"], json_bytes(result), content_type="application/json", scope=SCOPE)
    return {**result, "status": "candidate_written" if execute else "dry_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--reconciliation-run", required=True)
    parser.add_argument("--screening-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fractional", action="store_true", help="Screen simple fractional matches; economic identity remains unverified")
    parser.add_argument("--small-fraction", action="store_true", help="Reassess 21:20 and 11:10 using cross-provider basis evidence")
    parser.add_argument("--irregular", action="store_true", help="Assess irregular ratio price basis; economic identity remains unverified")
    parser.add_argument("--yahoo-only", action="store_true", help="Assess implicit EODHD price evidence for Yahoo-only split events")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(boto3.client("s3", region_name=args.region), args.bucket,
                           reconciliation_run=args.reconciliation_run, screening_run=args.screening_run,
                           run_id=args.run_id, workers=args.workers, execute=args.execute, fractional=args.fractional, small_fraction=args.small_fraction, irregular=args.irregular, yahoo_only=args.yahoo_only), indent=2))


if __name__ == "__main__":
    main()
