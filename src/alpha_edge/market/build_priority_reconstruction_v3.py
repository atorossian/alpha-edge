"""Build a quarantined, source-pinned four-asset AEA-94 reconstruction candidate."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]
ROOT = "market/lake_v3/silver/financial_acceptance_reviews/v1/runs"
INPUT = "aea94-first-four-reconstruction-hypotheses-20261128"
VERSION = "3889cf46b5b1b6b3f94290746a68569ae126b5d8691d09ded98822fb1ee50932"
DIAGNOSTIC = "3651e2747c29c54795916a549234206103783d66b5db96f29d72a028dd5a3e5a"
INDEX = "fd00093539bdd4fa7538b0ae42c920aa1d6f2c94b6cf975cdceda2b3452814ee"
TERMS = "6bcc0d501a1e9747276826e4414e72e4fbf2505cb53ca8aec1a26f82e134e91e"


def indexed(frame):
    f = frame.copy()
    f["session_date"] = pd.to_datetime(f.session_date).dt.strftime("%Y-%m-%d")
    f = f.set_index("session_date").sort_index()
    if not f.index.is_unique:
        raise ValueError("Duplicate session")
    return f


def scale_window(f, e, days):
    """Use independent close anchors only; do not claim OHLC agreement."""
    days = [d for d in days if d in e.index and f.at[d, "volume"] > 0]
    if len(days) < 3:
        return None
    values = f.loc[days, "close"] / e.loc[days, "close"]
    mid = float(values.median())
    if not math.isfinite(mid) or mid <= 0 or (values / mid - 1).abs().max() > .005:
        return None
    return mid


def reconstruct(bars, eod, calendar, entries):
    f = indexed(bars)
    cal = indexed(calendar)
    if set(cal.index) != set(f.index):
        raise ValueError("Calendar/source mismatch")
    e = pd.DataFrame(eod).set_index("date").sort_index()
    if not e.index.is_unique:
        raise ValueError("Duplicate EODHD session")
    for col in OHLC + ["volume"]:
        f["candidate_" + col] = f[col].astype(float)
    f["candidate_cash"] = f.dividends.astype(float)
    f["candidate_price_factor"] = 1.0
    f["candidate_review"] = True
    f["candidate_price_source"] = "yahoo"
    f["candidate_cash_source"] = "yahoo"
    f["candidate_issuer_share_ratio"] = np.nan
    f["production_eligible"] = False
    f["candidate_expected_session"] = cal.is_expected_session.eq(True)
    trade = f.index[f.candidate_expected_session]
    splits = f.stock_splits.astype(float).replace(0, 1)
    chain = splits.iloc[::-1].cumprod().iloc[::-1]
    if not np.isfinite(chain).all() or not chain.gt(0).all():
        raise ValueError("Invalid share chain")
    reports = []
    for event in sorted(entries, key=lambda x: x["date"]):
        day, route = event["date"], event["route"]
        if day not in trade or day not in e.index:
            reports.append({"date": day, "status": "session_unverified"})
            continue
        pos = trade.get_loc(day)
        before, after = trade[max(0, pos - 5):pos], trade[pos:pos + 5]
        pre, post = scale_window(f, e, before), scale_window(f, e, after)
        report = {"date": day, "status": "retained_pending_validation", "pre_scale": pre, "post_scale": post}
        if event.get("issuer_share_or_ads_ratio") is not None:
            f.at[day, "candidate_issuer_share_ratio"] = float(event["issuer_share_or_ads_ratio"])
        if route in {"quote_basis_break_estimate_pre_segment_factor", "compound_quote_basis_break_estimate_pre_segment_factor"}:
            # EODHD quote conversion includes non-legal units in 2000.
            ratio = float(f.at[day, "stock_splits"])
            multiplier = post / (pre * ratio) if pre and post and ratio > 0 else None
            hint = event["price_multiplier_hypothesis"]
            if multiplier and math.isclose(multiplier, hint, rel_tol=.005):
                ix = f.index < day
                f.loc[ix, "candidate_price_factor"] *= multiplier
                report.update(status="prefix_scale_candidate", multiplier=multiplier)
        elif route == "stale_yahoo_replace_only_verified_trading_sessions":
            # Identify the complete contiguous flat, zero-volume run, not an arbitrary event window.
            stale = f.loc[trade, "volume"].eq(0) & f.loc[trade, OHLC].eq(f.at[day, "close"]).all(axis=1)
            left = right = pos
            if stale.iloc[pos]:
                while left > 0 and stale.iloc[left - 1]:
                    left -= 1
                while right + 1 < len(trade) and stale.iloc[right + 1]:
                    right += 1
                days = trade[left:right + 1]
                a = scale_window(f, e, trade[max(0, left - 5):left])
                b = scale_window(f, e, trade[right + 1:right + 6])
                ratio = float(f.at[day, "stock_splits"])
                if (a and b and ratio > 0 and math.isclose(a * ratio, b, rel_tol=.005)
                        and set(days) <= set(e.index) and e.loc[days, "volume"].gt(0).all()
                        and splits.loc[days].ne(1).sum() == 1 and f.loc[days, "dividends"].eq(0).all()):
                    for d in days:
                        f.loc[d, ["candidate_" + c for c in OHLC]] = e.loc[d, OHLC].to_numpy(dtype=float) * b / (ratio if d < day else 1)
                    f.loc[days, "candidate_volume"] = np.nan
                    f.loc[days, "candidate_price_source"] = "eodhd_stale_reconstruction"
                    report.update(status="stale_window_candidate", sessions=len(days), first=days[0], last=days[-1])
        elif route == "issuer_gross_cash_candidate_verify_exact_share_chain":
            gross = float(event["issuer_gross_cash_candidate"]["gross_old"])
            amount = gross / float(chain.at[day])
            # Check the existing normalized price basis against the complete recorded chain.
            if pre and math.isclose(pre, 1 / float(chain.at[day]), rel_tol=.005):
                f.at[day, "candidate_cash"] = amount
                f.at[day, "candidate_cash_source"] = "issuer_gross_on_recorded_share_basis"
                report.update(status="gross_cash_candidate", cash=amount, source_cash=float(f.at[day, "dividends"]))
        elif route == "preserve_existing_normalized_close" and pre and post:
            report["status"] = "preserve_close_candidate"
        reports.append(report)
    scaled = f.candidate_price_factor.ne(1)
    for col in OHLC:
        f.loc[scaled, "candidate_" + col] *= f.loc[scaled, "candidate_price_factor"]
    f.loc[scaled, "candidate_volume"] = np.nan
    f.loc[scaled, "candidate_price_source"] = "yahoo_quote_basis_candidate"
    # No cash scaling of questionable historical quote denominations; keep the source cash for review.
    o, h, l, c = (f["candidate_" + x] for x in OHLC)
    valid = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & l.gt(0) & h.ge(o) & h.ge(c) & l.le(o) & l.le(c)
    touched = scaled | f.candidate_price_source.ne("yahoo")
    if (~valid & touched).any():
        raise ValueError("Invalid reconstructed OHLC")
    # Return checks are diagnostic only; do not store derived return columns.
    for report in reports:
        day = report["date"]
        if day in trade and trade.get_loc(day) > 0:
            prev = trade[trade.get_loc(day) - 1]
            p, q = float(f.at[prev, "candidate_close"]), float(f.at[day, "candidate_close"])
            if p > 0 and q > 0:
                report["price_return_diagnostic"] = q / p - 1
                report["cash_inclusive_diagnostic"] = (q + float(f.at[day, "candidate_cash"])) / p - 1
    return f.reset_index(), reports


def main():
    import boto3

    from alpha_edge.market.lakehouse_v3.storage import (
        check_id, digest, json_bytes, parquet_bytes, put_immutable, verified_bytes, verified_frame,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--run-id", required=True)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    check_id(a.run_id)
    s3 = boto3.client("s3", region_name=a.region)

    def read(key, sha):
        return json.loads(verified_bytes(s3, a.bucket, {"key": key, "sha256": sha}))

    doc = read(f"{ROOT}/{INPUT}/objects/{VERSION}.json", VERSION)
    terms = read(f"{ROOT}/aea94-priority-33-economic-terms-20261128/objects/{TERMS}.json", TERMS)
    term_index = {(r["asset_id"], r["date"]): r for r in terms["events"]}
    diag = read(f"{ROOT}/aea94-first-four-reconstruction-diagnostic-20261128/objects/{DIAGNOSTIC}.json", DIAGNOSTIC)
    idx = read(f"market/lake_v3/silver/session_candidates/consolidated/{INDEX}/index.json", INDEX)
    if len(doc["events"]) != 16 or len(diag["source_refs"]) != 4:
        raise ValueError("Unexpected approved population")
    out = []
    scope = "market/lake_v3/silver/priority_reconstruction_candidates/v1"
    root = f"{scope}/runs/{a.run_id}"
    for i, ref in enumerate(diag["source_refs"], 1):
        aid = ref["asset_id"]
        entries = [{**term_index[(x["asset_id"], x["date"])], **x} for x in doc["events"] if x["asset_id"] == aid]
        if len(entries) != 4 or any(x["bronze_sha256"] != ref["bronze_sha256"] or x["eodhd_sha256"] != ref["eodhd_sha256"] for x in entries):
            raise ValueError("Source pin mismatch")
        bars = verified_frame(s3, a.bucket, {"key": ref["bronze_key"], "sha256": ref["bronze_sha256"]})
        if not bars.asset_id.eq(aid).all() or not bars.provider_symbol.eq(ref["symbol"]).all():
            raise ValueError("Bronze identity mismatch")
        cal = verified_frame(s3, a.bucket, idx["assets"][aid]["outputs"]["candidate"])
        source = read(ref["eodhd_key"], ref["eodhd_sha256"])
        eod = json.loads(source["responses"]["eod"]["raw_json"])
        frame, reports = reconstruct(bars, eod, cal, entries)
        item = {"asset_id": aid, "symbol": ref["symbol"], "events": reports, "rows": len(frame)}
        if a.execute:
            blob = parquet_bytes(frame)
            item["output"] = put_immutable(s3, a.bucket, f"{root}/assets/{digest(aid.encode())}/prices-{digest(blob)}.parquet", blob, content_type="application/vnd.apache.parquet", scope=scope)
        out.append(item)
        print(f"[priority reconstruction] {i}/4 {ref['symbol']}", flush=True)
    manifest = {"schema": "alpha_edge_priority_reconstruction_candidate_v1", "run_id": a.run_id, "input_sha256": VERSION, "assets": out, "production_eligible": False, "quarantine_released": False, "financial_validation_complete": False, "status": "candidate_written" if a.execute else "dry_run"}
    if a.execute:
        put_immutable(s3, a.bucket, f"{root}/manifest.json", json_bytes(manifest), content_type="application/json", scope=scope)
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
