"""Apply approved, pinned asset price-basis repairs to a non-production candidate."""

from __future__ import annotations

import argparse
import json
import math

import boto3
import numpy as np
import pandas as pd

from alpha_edge.market.lakehouse_v3.storage import (
    ROOT, check_id, digest, json_bytes, parquet_bytes, put_immutable, verified_bytes, verified_frame,
)

REGISTRY = f"{ROOT}/config/asset_price_reconstruction_approvals/v1/objects"
OUTPUT = f"{ROOT}/silver/asset_price_reconstruction_candidates/v1"
OHLC = ("open", "high", "low", "close")


def _precision_reconstruction(f, e, doc):
    """Correct only the missing precision of an already-normalized share chain."""
    factor = pd.Series(1.0, index=f.index)
    cash_factor = factor.copy()
    exact = pd.to_numeric(f.stock_splits, errors="raise").copy()
    reports, seen = [], set()
    for event in sorted(doc["entries"], key=lambda x: x["date"]):
        day = event["date"]
        reported, issuer = float(event["ratio"]), float(event["issuer_ratio"])
        if (day in seen or day not in f.index or day not in e.index or not math.isfinite(reported)
                or not math.isfinite(issuer) or min(reported, issuer) <= 0
                or not math.isclose(float(exact.at[day]), reported, rel_tol=1e-7)):
            raise ValueError(f"Invalid precision correction: {day}")
        seen.add(day)
        change = reported / issuer
        pos = f.index.get_loc(day)
        before = f.index[max(0, pos-5):pos].intersection(e.index)
        after = f.index[pos:pos+5].intersection(e.index)
        if len(before) < 3 or len(after) < 3:
            raise ValueError(f"Missing precision corroboration window: {day}")
        pre_levels = (e.loc[before, list(OHLC)] / f.loc[before, list(OHLC)]).to_numpy(dtype=float)
        post_levels = (e.loc[after, list(OHLC)] / f.loc[after, list(OHLC)]).to_numpy(dtype=float)
        pre, post = float(np.median(pre_levels)), float(np.median(post_levels))
        observed = pre / post if post > 0 else math.nan
        error = max(float(np.max(np.abs(pre_levels / pre - 1))), float(np.max(np.abs(post_levels / post - 1)))) if pre > 0 and post > 0 else math.inf
        if not math.isfinite(observed) or not math.isfinite(error) or error > 0.0002 or not math.isclose(observed, change, rel_tol=0.0002):
            raise ValueError(f"Independent precision step mismatch: {day}")
        factor.loc[f.index < day] *= change
        # Cash paid for pre-distribution shares includes today's ratio as well.
        cash_factor.loc[f.index <= day] *= change
        exact.at[day] = issuer
        reports.append({"date": day, "source_ratio": reported, "issuer_ratio": issuer,
                        "incremental_factor": change, "observed_ratio": observed, "ohlc_scale_max_relative_error": error})
    if not np.isfinite(factor).all() or not factor.gt(0).all():
        raise ValueError("Invalid cumulative precision factor")
    for c in OHLC:
        f[f"candidate_{c}"] = pd.to_numeric(f[c], errors="raise") * factor
    f["candidate_split_event"] = exact
    f["candidate_share_precision_factor"] = factor
    f["candidate_cash_precision_factor"] = cash_factor
    f["candidate_price_source"] = np.where(factor.ne(1.0), "yahoo_issuer_ratio_precision", "yahoo")
    f["candidate_volume"] = pd.to_numeric(f.volume, errors="coerce").where(factor.eq(1.0))
    f["candidate_volume_review"] = factor.ne(1.0)
    f["candidate_boundary_review"] = False
    f["candidate_cash_review"] = False
    f["candidate_cash_dividend"] = pd.to_numeric(f.dividends, errors="coerce") * cash_factor
    future = exact.replace(0, 1).iloc[::-1].cumprod().iloc[::-1]
    if not np.isfinite(future).all() or not future.gt(0).all():
        raise ValueError("Invalid recorded share chain")
    f["candidate_declared_cash_on_basis"] = np.nan
    for event in doc["entries"]:
        day = event["date"]
        declared = float(event["declared_cash"])
        if not math.isfinite(declared) or declared <= 0:
            raise ValueError("Invalid declared cash evidence")
        expected = declared / float(future.at[day])
        f.at[day, "candidate_declared_cash_on_basis"] = expected
        # Do not change the economic cash amount. Hold discrepancies for review.
        same = math.isclose(float(f.at[day, "candidate_cash_dividend"]), expected, rel_tol=0.001, abs_tol=0.0001)
        f.at[day, "candidate_cash_review"] = not same
    o, h, l, c = (f[f"candidate_{x}"] for x in OHLC)
    valid = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & l.gt(0) & h.ge(o) & h.ge(c) & l.le(o) & l.le(c)
    f["candidate_ohlc_valid"] = valid
    f["candidate_price_return"] = (c / c.shift() - 1).where(valid & valid.shift(fill_value=False))
    f["candidate_total_return"] = ((c + f.candidate_cash_dividend) / c.shift() - 1).where(f.candidate_price_return.notna() & ~f.candidate_cash_review)
    f["production_eligible"] = False
    return f.reset_index(), {"events": reports, "corrected_ratios": sum(x["source_ratio"] != x["issuer_ratio"] for x in reports),
                             "touched_rows": int(factor.ne(1.0).sum()), "cash_review_rows": int(f.candidate_cash_review.sum()),
                             "volume_review_rows": int(f.candidate_volume_review.sum()), "invalid_touched_ohlc_rows": int((~valid & factor.ne(1.0)).sum())}


def reconstruct(bars, responses, doc):
    """Preserve source columns; apply explicit local factors or anchored provider windows."""
    if (doc.get("schema") != "alpha_edge_asset_price_reconstruction_approval_v1"
            or doc.get("approval") != "approved_candidate_reconstruction"
            or doc.get("production_eligible") is not False or doc.get("quarantine_released") is not False):
        raise ValueError("Expected candidate-only approval")
    f = bars.copy(deep=True)
    f["session_date"] = pd.to_datetime(f.session_date).dt.strftime("%Y-%m-%d")
    f = f.set_index("session_date").sort_index()
    if not f.index.is_unique or not f.asset_id.eq(doc["asset_id"]).all() or not f.provider_symbol.eq(doc["symbol"]).all():
        raise ValueError("Source identity mismatch")
    e = pd.DataFrame(responses["eod"]).set_index("date").sort_index()
    if not e.index.is_unique:
        raise ValueError("Duplicate provider session")
    if doc["entries"] and all(x["method"] == "issuer_ratio_precision" for x in doc["entries"]):
        return _precision_reconstruction(f, e, doc)
    for c in OHLC:
        f[f"candidate_{c}"] = pd.to_numeric(f[c], errors="raise")
    f["candidate_volume"] = pd.to_numeric(f.volume, errors="coerce")
    f["candidate_price_source"] = "yahoo"
    f["candidate_volume_review"] = False
    f["candidate_boundary_review"] = False
    f["candidate_cash_review"] = False
    f["candidate_split_event"] = pd.to_numeric(f.stock_splits, errors="raise")
    cash_by_day = {}
    for row in responses["div"]:
        cash_by_day.setdefault(row["date"], []).append(row)
    cash_days = set(cash_by_day)
    f["candidate_eodhd_cash_evidence"] = ""
    touched = set()
    reports = []
    for event in doc["entries"]:
        day, ratio = event["date"], float(event["ratio"])
        if not math.isfinite(ratio) or ratio <= 0 or day not in f.index or not math.isclose(float(f.at[day, "stock_splits"]), ratio, rel_tol=1e-7):
            raise ValueError(f"Source split mismatch: {day}")
        method = event["method"]
        if method == "preserve_normalized":
            reports.append({"date": day, "method": method, "rows": 0})
            continue
        if method in {"local_ohlc_scale", "historical_quote_basis_reconstruct"}:
            days = list(event["sessions"])
            multiplier = float(event["multiplier"])
            if not math.isfinite(multiplier) or multiplier <= 0 or not days:
                raise ValueError("Invalid local correction")
            if not set(days) <= set(f.index) or touched.intersection(days):
                raise ValueError("Missing or overlapping correction sessions")
            for c in OHLC:
                f.loc[days, f"candidate_{c}"] = f.loc[days, c] * multiplier
            f.loc[days, "candidate_price_source"] = "yahoo_explicit_scale"
        elif method == "eodhd_price_basis_reconstruct":
            effective = event["effective_trading_date"]
            pos = e.index.get_loc(effective)
            if pos < 4 or pos + 4 >= len(e):
                raise ValueError("Incomplete independent price window")
            window = e.iloc[pos-4:pos+5]
            days = list(window.index.intersection(f.index))
            if effective not in days or touched.intersection(days):
                raise ValueError("Missing or overlapping provider sessions")
            p, q = e.iloc[pos-1], e.iloc[pos]
            observed = (q.adjusted_close / q.close) / (p.adjusted_close / p.close)
            if not math.isclose(observed, ratio, rel_tol=1e-4):
                raise ValueError(f"Independent factor mismatch: {day}")
            after = [d for d in days if d > effective]
            if len(after) < 2:
                raise ValueError("Insufficient post-event anchors")
            scale = float((f.loc[after, "close"] / e.loc[after, "close"]).median())
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Invalid post-event quote scale")
            for c in OHLC:
                f.loc[days, f"candidate_{c}"] = [float(e.at[d, c]) * scale / (ratio if d < effective else 1.0) for d in days]
            f.loc[days, "candidate_price_source"] = "eodhd_anchored_split_basis"
            # Keep the originally reported event, but expose the candidate trading date separately.
            if effective != day:
                if float(f.at[effective, "stock_splits"]) != 0:
                    raise ValueError("Conflicting split session")
                f.at[day, "candidate_split_event"] = 0.0
                f.at[effective, "candidate_split_event"] = ratio
        else:
            raise ValueError(f"Unsupported correction method: {method}")
        touched.update(days)
        f.loc[days, "candidate_volume"] = np.nan
        f.loc[days, "candidate_volume_review"] = True
        review = [d for d in days if d in cash_days or float(f.at[d, "dividends"]) != 0]
        f.loc[review, "candidate_cash_review"] = True
        for d in review:
            f.at[d, "candidate_eodhd_cash_evidence"] = json.dumps(cash_by_day.get(d, []), sort_keys=True)
        # Provider switches are not asserted to be continuous at either edge.
        first, last = f.index.get_loc(min(days)), f.index.get_loc(max(days))
        if method == "eodhd_price_basis_reconstruct" or event.get("boundary_unverified"):
            f.iloc[first, f.columns.get_loc("candidate_boundary_review")] = True
            if last + 1 < len(f):
                f.iloc[last+1, f.columns.get_loc("candidate_boundary_review")] = True
        reports.append({"date": day, "method": method, "rows": len(days)})
    o, h, l, c = (f[f"candidate_{x}"] for x in OHLC)
    valid = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & l.gt(0) & h.ge(o) & h.ge(c) & l.le(o) & l.le(c)
    f["candidate_ohlc_valid"] = valid
    f["candidate_price_return"] = (c / c.shift() - 1).where(valid & valid.shift(fill_value=False) & ~f.candidate_boundary_review)
    # Cash remains explicit and unverified; do not replace it with an invented entitlement.
    f["candidate_cash_dividend"] = f.dividends.where(~f.candidate_cash_review)
    f["candidate_total_return"] = ((c + f.candidate_cash_dividend) / c.shift() - 1).where(f.candidate_price_return.notna() & ~f.candidate_cash_review)
    f["production_eligible"] = False
    return f.reset_index(), {"events": reports, "touched_rows": len(touched), "cash_review_rows": int(f.candidate_cash_review.sum()), "volume_review_rows": int(f.candidate_volume_review.sum()), "boundary_review_rows": int(f.candidate_boundary_review.sum()), "invalid_touched_ohlc_rows": int((~valid & f.index.isin(touched)).sum())}


def build(s3, bucket, version, run_id, execute=False):
    check_id(run_id)
    ref = {"key": f"{REGISTRY}/{version}.json", "sha256": version}
    doc = json.loads(verified_bytes(s3, bucket, ref))
    if "checkpoint" in doc:
        verified_bytes(s3, bucket, doc["checkpoint"])
    bars = verified_frame(s3, bucket, doc["yahoo"])
    source = json.loads(verified_bytes(s3, bucket, doc["eodhd"]))
    responses = {k: json.loads(v["raw_json"]) for k, v in source["responses"].items()}
    result, stats = reconstruct(bars, responses, doc)
    manifest = {"schema": "alpha_edge_asset_price_reconstruction_candidate_v1", "run_id": run_id, "approval": ref, "asset_id": doc["asset_id"], "statistics": stats, "production_eligible": False, "quarantine_released": False}
    if execute:
        blob = parquet_bytes(result)
        root = f"{OUTPUT}/runs/{run_id}/assets/{digest(doc['asset_id'].encode())}"
        manifest["output"] = put_immutable(s3, bucket, f"{root}/objects/prices-{digest(blob)}.parquet", blob, content_type="application/vnd.apache.parquet", scope=OUTPUT)
        put_immutable(s3, bucket, f"{root}/manifest.json", json_bytes(manifest), content_type="application/json", scope=OUTPUT)
        manifest["status"] = "candidate_written"
    else:
        manifest["status"] = "dry_run"
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--approval-version", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    print(json.dumps(build(boto3.client("s3", region_name=a.region), a.bucket, a.approval_version, a.run_id, a.execute), indent=2))


if __name__ == "__main__":
    main()
