"""Read-only two-provider corporate-action reconciliation; writes candidate evidence only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd

from alpha_edge.market.eodhd_source_v3 import load_mapping, verify_bundle
from alpha_edge.market.ingest_bronze_full_history import EODHD_ROOT, _completed, load_universe
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT, check_id, digest, json_bytes, parquet_bytes, put_immutable, read_json, verified_bytes, verified_frame,
)

SCOPE = f"{ROOT}/silver/provider_reconciliation/v1"
POLICY = "raw_events_exact_and_unique_near_date_v1"


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def split_ratio(value):
    try:
        a, b = str(value).split("/")
        n, d = float(a), float(b)
        return number(n / d) if n > 0 and d > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def relative_match(a, b, *, tol):
    return a is not None and b is not None and a > 0 and b > 0 and math.isclose(a, b, rel_tol=tol)


def groups(events, start, end):
    inside, outside = defaultdict(list), defaultdict(list)
    for row in events:
        (inside if start <= row["date"] <= end else outside)[row["date"]].append(row)
    return dict(inside), dict(outside)


def reconcile_events(yahoo, eodhd, *, kind, start, end, currency, context):
    """Exact date first; near-date pairing only for mutually unique unmatched dates."""
    y, yo = groups(yahoo, start, end)
    e, eo = groups(eodhd, start, end)
    output = []

    def emit(yd, ed, date_status, year=None, ear=None):
        yr, er = year or [], ear or []
        yv = number(yr[0]["value"]) if len(yr) == 1 else None
        ev = (split_ratio(er[0]["split"]) if kind == "split" else number(er[0].get("value"))) if len(er) == 1 else None
        raw = number(er[0].get("unadjustedValue")) if kind == "dividend" and len(er) == 1 else None
        ec = er[0].get("currency") if kind == "dividend" and len(er) == 1 else None
        yc = yr[0].get("currency") if kind == "dividend" and len(yr) == 1 else None
        # Currency hints are not evidence of the historical dividend denomination.
        yc = yc or None
        if len(yr) > 1 or len(er) > 1:
            value_status = "duplicate_date"
        elif not yr or not er:
            value_status = "unpaired"
        elif kind == "split":
            value_status = ("ratio_not_evaluable" if yv is None or ev is None or yv <= 0 or ev <= 0
                            else "ratio_match" if relative_match(yv, ev, tol=0.0001) else "ratio_mismatch")
        elif not yc or not ec:
            value_status = "currency_unknown"
        elif yc != ec:
            value_status = "currency_mismatch"
        else:
            adj = relative_match(yv, ev, tol=0.005)
            orig = relative_match(yv, raw, tol=0.005)
            value_status = (
                "both_amounts_match" if adj and orig else "adjusted_amount_match" if adj else
                "raw_amount_match" if orig else "amount_not_evaluable" if yv is None or (ev is None and raw is None)
                else "amount_mismatch"
            )
        item = {
            "kind": kind, "yahoo_date": yd, "eodhd_date": ed,
            "date_status": date_status, "value_status": value_status,
            "yahoo_event_rows": len(yr), "eodhd_event_rows": len(er),
            "yahoo_value": yv, "eodhd_value": ev, "eodhd_unadjusted_value": raw,
            "yahoo_currency": yc if kind == "dividend" else None, "eodhd_currency": ec,
            "yahoo_raw": json.dumps(yr, sort_keys=True, allow_nan=False),
            "eodhd_raw": json.dumps(er, sort_keys=True),
            **context.get(yd or ed, {}),
        }
        output.append(item)

    both = set(y) & set(e)
    for day in sorted(both):
        emit(day, day, "exact_date", y[day], e[day])
    yl, el = set(y) - both, set(e) - both
    # At most four calendar days (weekends/holidays); never merge ambiguous clusters.
    candidates = {}
    reverse = defaultdict(list)
    for day in sorted(yl):
        dt = date.fromisoformat(day)
        near = [(dt + timedelta(days=offset)).isoformat() for offset in range(-4, 5) if offset]
        candidates[day] = [x for x in near if x in el]
        for other in candidates[day]:
            reverse[other].append(day)
    paired_y, paired_e = set(), set()
    for day, possible in candidates.items():
        if len(possible) == 1 and len(reverse[possible[0]]) == 1:
            other = possible[0]
            emit(day, other, "near_date_candidate", y[day], e[other])
            paired_y.add(day)
            paired_e.add(other)
    for day in sorted(yl - paired_y):
        emit(day, None, "ambiguous_near_date" if candidates[day] else "yahoo_only", y[day])
    for day in sorted(el - paired_e):
        emit(None, day, "ambiguous_near_date" if reverse[day] else "eodhd_only", ear=e[day])
    for day, rows in sorted(yo.items()):
        emit(day, None, "outside_overlap", rows)
    for day, rows in sorted(eo.items()):
        emit(None, day, "outside_overlap", ear=rows)
    assert sum(x["yahoo_event_rows"] for x in output) == len(yahoo)
    assert sum(x["eodhd_event_rows"] for x in output) == len(eodhd)
    return output


def price_stats(a, b, tolerance):
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    valid = np.isfinite(a) & np.isfinite(b) & a.gt(0) & b.gt(0)
    rel = (a[valid] / b[valid] - 1).abs()
    return {
        "evaluated": int(valid.sum()), "not_evaluable": int((~valid).sum()),
        "within_tolerance": int(rel.le(tolerance).sum()), "outside_tolerance": int(rel.gt(tolerance).sum()),
        "median_relative_error": number(rel.median()), "p95_relative_error": number(rel.quantile(0.95)),
    }


def compare_asset(yahoo, responses, *, currency):
    y = yahoo.copy()
    y["date"] = pd.to_datetime(y["session_date"]).dt.strftime("%Y-%m-%d")
    e = pd.DataFrame(responses["eod"])
    if y.empty or e.empty or y["date"].isna().any():
        raise ValueError("Missing source price dates")
    start, end = max(y.date.min(), e.date.min()), min(y.date.max(), e.date.max())
    # Preserve duplicate observations in Bronze; exclude all duplicate-date quotes from comparison.
    dup_y = int(y.date.duplicated(keep=False).sum())
    dup_e = int(e.date.duplicated(keep=False).sum())
    yy = y.loc[~y.date.duplicated(keep=False)].set_index("date").sort_index()
    ee = e.loc[~e.date.duplicated(keep=False)].set_index("date").sort_index()
    common = yy.index.intersection(ee.index)
    yi, ei = yy.loc[common], ee.loc[common]
    factors = pd.Series(1.0, index=common)
    split_valid = True
    dates = [r["date"] for r in responses["splits"]]
    if len(dates) != len(set(dates)):
        split_valid = False
    for row in responses["splits"]:
        ratio = split_ratio(row["split"])
        if ratio is None:
            split_valid = False
        else:
            factors.loc[factors.index < row["date"]] *= ratio
    if not split_valid or not np.isfinite(factors).all() or not factors.gt(0).all():
        factors[:] = np.nan
        split_valid = False
    fields = {}
    for field in ("open", "high", "low", "close"):
        raw = pd.to_numeric(ei[field], errors="coerce")
        fields[field] = {"as_reported": price_stats(yi[field], raw, 0.005),
                         "eodhd_split_normalized": price_stats(yi[field], raw / factors, 0.005)}
    fields["volume"] = {"as_reported": price_stats(yi["volume"], ei["volume"], 0.05)}
    # Comparing normalized adj-close series avoids differences solely from latest-price anchors.
    ya = pd.to_numeric(yi["adj_close"], errors="coerce")
    ea = pd.to_numeric(ei["adjusted_close"], errors="coerce")
    anchors = ya.index[np.isfinite(ya) & np.isfinite(ea) & ya.gt(0) & ea.gt(0)]
    if len(anchors):
        anchor = anchors[-1]
        fields["adjusted_close"] = {"anchor_normalized": price_stats(ya / ya.loc[anchor], ea / ea.loc[anchor], 0.005)}
    else:
        anchor = None
        fields["adjusted_close"] = {"anchor_normalized": price_stats(ya * np.nan, ea * np.nan, 0.005)}
    context = {}
    yc, ec = pd.to_numeric(yi.close, errors="coerce"), pd.to_numeric(ei.close, errors="coerce")
    # Event-local step. Do not use a long interval across missing sessions as a daily return.
    event_dates = set(y.loc[pd.to_numeric(y.stock_splits, errors="coerce").fillna(0).ne(0)
                            | pd.to_numeric(y.dividends, errors="coerce").fillna(0).ne(0), "date"])
    event_dates.update(row["date"] for endpoint in ("splits", "div") for row in responses[endpoint])
    positions = common.get_indexer(sorted(event_dates))
    for i in sorted(p for p in positions if p > 0):
        day, prev = common[i], common[i - 1]
        if (date.fromisoformat(day) - date.fromisoformat(prev)).days > 7:
            continue
        p, q, r, s = (number(v) for v in (yc.iloc[i-1], yc.iloc[i], ec.iloc[i-1], ec.iloc[i]))
        if all(v is not None and v > 0 for v in (p, q, r, s)):
            context[day] = {"previous_shared_date": prev, "yahoo_close_step": q/p,
                            "eodhd_close_step": s/r, "cross_provider_step_ratio": (s/r)/(q/p)}
    raw_yahoo = {}
    for kind, column in (("split", "stock_splits"), ("dividend", "dividends")):
        rows = []
        for row in y.loc[pd.to_numeric(y[column], errors="coerce").fillna(0).ne(0)].to_dict("records"):
            rows.append({"date": row["date"], "value": number(row[column]),
                         "currency": row["currency_original"] if isinstance(row["currency_original"], str) else None})
        raw_yahoo[kind] = rows
    events = []
    for kind, endpoint in (("split", "splits"), ("dividend", "div")):
        events.extend(reconcile_events(raw_yahoo[kind], responses[endpoint], kind=kind,
                                       start=start, end=end, currency=currency, context=context))
    summary = {
        "yahoo_rows": len(y), "eodhd_rows": len(e), "overlap_start": start, "overlap_end": end,
        "shared_sessions": len(common), "yahoo_duplicate_price_rows": dup_y, "eodhd_duplicate_price_rows": dup_e,
        "yahoo_only_sessions": len(yy.loc[start:end].index.difference(ee.index)),
        "eodhd_only_sessions": len(ee.loc[start:end].index.difference(yy.index)),
        "split_factor_hypothesis_evaluable": split_valid, "adjusted_close_anchor": anchor,
        "yahoo_currency_hint": currency,
        "field_metrics": json.dumps(fields, sort_keys=True, allow_nan=False),
    }
    return summary, events


def audit(s3, bucket, *, yahoo_diagnostic, eodhd_run, fingerprint, run_id, only=None, execute=False, workers=8):
    for value in (yahoo_diagnostic, eodhd_run, run_id):
        check_id(value)
    if not 1 <= workers <= 16:
        raise ValueError("workers must be 1–16")
    dkey = f"{ROOT}/silver/financial_diagnostics/v1/runs/{yahoo_diagnostic}/manifest.json"
    diagnostic = read_json(s3, bucket, dkey)
    coverage = verified_frame(s3, bucket, diagnostic["outputs"]["coverage"])
    if coverage.asset_id.duplicated().any():
        raise ValueError("Duplicate Yahoo diagnostic asset")
    by_id = coverage.set_index("asset_id").to_dict("index")
    config_key = f"{EODHD_ROOT}/runs/{eodhd_run}/run.json"
    config = read_json(s3, bucket, config_key)
    if config["source_fingerprint"] != fingerprint or config["provider_parameters"]["provider"] != "eodhd":
        raise ValueError("EODHD run identity mismatch")
    current, _ = load_universe(s3, bucket, fingerprint)
    ids = {r["asset_id"] for r in current}
    if len(config["requested_asset_ids"]) != len(ids) or set(config["requested_asset_ids"]) != ids:
        raise ValueError("EODHD run does not cover pinned universe")
    if not ids <= set(by_id):
        raise ValueError("Yahoo coverage missing active asset")
    if only:
        selected = [r for r in current if r["asset_id"] in only or r["yahoo_ticker"] in only]
        if len(selected) != len(set(only)):
            raise ValueError("Unknown or ambiguous selector")
    else:
        selected = current
    mapped = load_mapping(s3, bucket, selected, config["provider_parameters"]["mapping_version"])

    def process(row):
        aid = row["asset_id"]
        yref = by_id[aid]
        if yref["provider_symbol"] != row["yahoo_ticker"]:
            raise ValueError(f"Yahoo symbol mismatch: {aid}")
        yf = verified_frame(s3, bucket, {"key": yref["bronze_key"], "sha256": yref["bronze_sha256"]})
        if len(yf) != yref["source_rows"] or not yf.asset_id.eq(aid).all() or not yf.provider_symbol.eq(row["yahoo_ticker"]).all():
            raise ValueError(f"Yahoo source mismatch: {aid}")
        receipt = _completed(s3, bucket, run_id=eodhd_run, record=row, verify_hash=False)
        if receipt is None or receipt["end_exclusive"] != config["end_exclusive"]:
            raise ValueError(f"Missing or incorrect EODHD receipt: {aid}")
        blob = verified_bytes(s3, bucket, receipt["data"])
        if len(blob) != receipt["data"]["bytes"]:
            raise ValueError(f"EODHD source size mismatch: {aid}")
        validation = verify_bundle(blob, row, config["end_exclusive"])
        if validation != receipt["validation"]:
            raise ValueError(f"EODHD receipt validation mismatch: {aid}")
        bundle = json.loads(blob)
        responses = {k: json.loads(v["raw_json"]) for k, v in bundle["responses"].items()}
        summary, events = compare_asset(yf, responses, currency=row["currency_hint"])
        identity = {"asset_id": aid, "yahoo_symbol": row["yahoo_ticker"], "eodhd_symbol": row["provider_symbol"],
                    "yahoo_sha256": yref["bronze_sha256"], "eodhd_sha256": receipt["data"]["sha256"]}
        return {**identity, **summary}, [{**identity, **x} for x in events]

    summaries, events = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (summary, findings) in enumerate(pool.map(process, mapped), 1):
            summaries.append(summary)
            events.extend(findings)
            if i % 50 == 0 or i == len(mapped):
                print(f"[provider reconciliation] {i}/{len(mapped)}", file=sys.stderr, flush=True)
    tables = {"coverage": pd.DataFrame(summaries), "events": pd.DataFrame(events)}
    root = f"{SCOPE}/runs/{run_id}"
    outputs = {}
    if execute:
        for name, table in tables.items():
            blob = parquet_bytes(table)
            key = f"{root}/objects/{name}-{digest(blob)}.parquet"
            outputs[name] = {**put_immutable(s3, bucket, key, blob, content_type="application/vnd.apache.parquet",
                                             scope=SCOPE), "rows": len(table), "format": "parquet"}
    stats = {}
    for kind in ("split", "dividend"):
        rows = [x for x in events if x["kind"] == kind]
        stats[kind] = {"yahoo_rows": sum(x["yahoo_event_rows"] for x in rows),
                       "eodhd_rows": sum(x["eodhd_event_rows"] for x in rows),
                       "date_status": dict(Counter(x["date_status"] for x in rows)),
                       "value_status": dict(Counter(x["value_status"] for x in rows)),
                       "exact_date_value_status": dict(Counter(x["value_status"] for x in rows if x["date_status"]=="exact_date"))}
    price_totals = defaultdict(Counter)
    for row in summaries:
        for field, variants in json.loads(row["field_metrics"]).items():
            for basis, metrics in variants.items():
                price_totals[f"{field}:{basis}"].update({
                    k: metrics[k] for k in ("evaluated", "not_evaluable", "within_tolerance", "outside_tolerance")
                })
    manifest = {
        "schema": "alpha_edge_provider_reconciliation_v1", "run_id": run_id, "policy": POLICY,
        "inputs": {"yahoo_diagnostic": yahoo_diagnostic, "yahoo_coverage": diagnostic["outputs"]["coverage"],
                   "yahoo_fingerprint": diagnostic["inputs"]["source_fingerprint"], "eodhd_run": eodhd_run,
                   "eodhd_fingerprint": fingerprint, "mapping_version": config["provider_parameters"]["mapping_version"],
                   "eodhd_run_config_sha256": digest(json_bytes(config)), "selected_asset_ids": [r["asset_id"] for r in mapped],
                   "yahoo_historical_only_ids": sorted(set(by_id) - ids)},
        "statistics": {"assets": len(summaries), "shared_sessions": sum(x["shared_sessions"] for x in summaries),
                       "price_comparison": {k: dict(v) for k, v in price_totals.items()}, **stats},
        "outputs": outputs, "production_eligible": False, "corrections_applied": False, "quarantine_released": False,
    }
    if execute:
        put_immutable(s3, bucket, f"{root}/manifest.json", json_bytes(manifest), content_type="application/json", scope=SCOPE)
    return {**manifest, "manifest_key": f"{root}/manifest.json", "status": "candidate_written" if execute else "dry_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--yahoo-diagnostic", required=True)
    parser.add_argument("--eodhd-run", required=True)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = audit(boto3.client("s3", region_name=args.region), args.bucket,
                   yahoo_diagnostic=args.yahoo_diagnostic, eodhd_run=args.eodhd_run,
                   fingerprint=args.source_fingerprint, run_id=args.run_id,
                   only=[x.strip() for x in args.only.split(",") if x.strip()] or None,
                   execute=args.execute, workers=args.workers)
    # Keep terminal output compact; detailed rows and exact selection remain in S3.
    report["inputs"].pop("selected_asset_ids")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
