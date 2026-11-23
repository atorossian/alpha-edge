"""Reconstruct independently documented dividend events without altering Bronze."""

from __future__ import annotations

import argparse
import json
import re
from decimal import ROUND_HALF_UP, Decimal

import boto3
import pandas as pd

from alpha_edge.market.build_share_basis_candidate_v3 import _date, _decimal
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT,
    check_id,
    digest,
    json_bytes,
    parquet_bytes,
    put_immutable,
    verified_bytes,
    verified_frame,
)

REGISTRY_ROOT = f"{ROOT}/config/dividend_event_candidate_registries/v1"
OUTPUT_ROOT = f"{ROOT}/silver/dividend_event_candidates/v1"
SCHEMA = "alpha_edge_dividend_event_candidate_registry_v1"


def normalized_amount(e):
    """Price coefficients are NOT share ratios (rights/spin-off semantics)."""
    amount = _decimal(e["issuer_amount"])
    if amount <= 0:
        raise ValueError("Expected positive declared dividend")
    for factor in e["basis_factors"]:
        v = _decimal(factor["factor"])
        if v <= 0 or not factor.get("evidence_url", "").startswith("https://"):
            raise ValueError("Invalid basis factor or evidence")
        if factor["kind"] == "share_ratio":
            amount /= v
        elif factor["kind"] == "price_coefficient":
            amount *= v
        else:
            raise ValueError("Unknown corporate-action factor type")
    return amount.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def validate_registry(doc):
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("status") != "candidate_only":
        raise ValueError("Expected candidate-only event registry")
    entries = doc.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Expected explicit entries")
    seen = set()
    for e in entries:
        for field in ("asset_id", "provider_symbol", "source_key", "source_sha256", "reason", "currency"):
            if not isinstance(e.get(field), str) or not e[field]:
                raise ValueError(f"Missing event field {field}")
        if e["asset_id"] in seen or not re.fullmatch(r"[0-9a-f]{64}", e["source_sha256"]):
            raise ValueError("Repeated asset or invalid source hash")
        seen.add(e["asset_id"])
        if not e["source_key"].startswith(f"{ROOT}/bronze/") or not re.fullmatch(r"[A-Z]{3}", e["currency"]):
            raise ValueError("Expected Bronze key and source currency")
        start, end, target = map(_date, (e["window_start"], e["window_end"], e["issuer_ex_date"]))
        if not start <= target <= end or e["basis_status"] not in {"verified", "provisional"}:
            raise ValueError("Invalid date window or basis status")
        if not isinstance(e.get("basis_factors"), list) or not isinstance(e.get("source_events"), list):
            raise TypeError("Expected explicit factors and source events")
        factor_dates = [_date(x["effective_date"]) for x in e["basis_factors"]]
        if len(set(factor_dates)) != len(factor_dates) or any(d <= target for d in factor_dates):
            raise ValueError("Basis factors must be distinct subsequent events")
        if any(x.get("status") not in {"verified", "provisional"} for x in e["basis_factors"]) or (
            e["basis_status"] == "verified" and any(x["status"] != "verified" for x in e["basis_factors"])
        ):
            raise ValueError("Unverified factor cannot support a verified basis")
        events = e["source_events"]
        days = [_date(x["date"]) for x in events]
        if len(set(days)) != len(days) or any(not start <= d <= end for d in days):
            raise ValueError("Repeated or out-of-window source event")
        if len(events) < 2 or any(_decimal(x["cash"]) <= 0 for x in events):
            raise ValueError("Expected positive original candidate events")
        # A retained matching amount must actually be stored on some source
        # session. Prohibit a fabricated replacement of an unknown payment.
        amount = normalized_amount(e)
        correction = e.get("issuer_cash_correction")
        if correction is not None:
            # An independently documented cash correction is distinct from an
            # inferred duplicate. Pin the existing target and require verified
            # basis, so the original cash remains auditable in the candidate.
            target_cash = next((_decimal(x["cash"]) for x in events if _date(x["date"]) == target), None)
            if (
                not isinstance(correction, dict)
                or e["basis_status"] != "verified"
                or target_cash is None
                or target_cash != _decimal(correction.get("source_target_cash", "0"))
                or not isinstance(correction.get("reason"), str)
                or not correction["reason"].strip()
                or correction.get("policy") != "issuer_declared_cash_excluding_tax_credit"
            ):
                raise ValueError("Invalid independently documented issuer cash correction")
        elif amount not in {_decimal(x["cash"]) for x in events}:
            raise ValueError("No correctly sized source observation to reconstruct")
        urls = e.get("evidence_urls")
        if not isinstance(urls, list) or len(set(urls)) < 2 or not all(u.startswith("https://") for u in urls):
            raise ValueError("Independent declared amount and date evidence required")
    return doc


def publish_registry(s3, bucket, doc, *, execute=False):
    payload = json_bytes(validate_registry(doc))
    version = digest(payload)
    key = f"{REGISTRY_ROOT}/{version}/registry.json"
    if execute:
        put_immutable(s3, bucket, key, payload, content_type="application/json", scope=REGISTRY_ROOT)
    return {"version": version, "key": key, "executed": execute}


def reconstruct(bars, entry, *, source_sha256):
    validate_registry({"schema": SCHEMA, "status": "candidate_only", "entries": [entry]})
    if source_sha256 != entry["source_sha256"]:
        raise ValueError("Source hash mismatch")
    if (
        bars.empty
        or not bars.asset_id.eq(entry["asset_id"]).all()
        or not bars.provider_symbol.eq(entry["provider_symbol"]).all()
    ):
        raise ValueError("Source identity mismatch")
    dates = bars.session_date.astype(str)
    if dates.duplicated().any():
        raise ValueError("Duplicate source sessions")
    recorded = {
        str(d): Decimal(str(v))
        for d, v in zip(dates, bars.stock_splits, strict=True)
        if d > entry["issuer_ex_date"] and v != 0
    }
    expected_splits = {
        x["effective_date"]: _decimal(x["factor"]) for x in entry["basis_factors"] if x["kind"] == "share_ratio"
    }
    if recorded != expected_splits:
        raise ValueError("Recorded subsequent share ratios differ from pinned basis")
    frame = bars.loc[dates.between(entry["window_start"], entry["window_end"])].copy(deep=True)
    frame = frame.sort_values("session_date").reset_index(drop=True)
    day = frame.session_date.astype(str)
    if day.eq(entry["issuer_ex_date"]).sum() != 1 or not frame.currency_original.eq(entry["currency"]).all():
        raise ValueError("No unique target session or currency mismatch")
    if frame.stock_splits.ne(0).any():
        raise ValueError("Unexpected same-window split; independent treatment required")
    actual = {str(d): Decimal(str(v)) for d, v in zip(day, frame.dividends, strict=True) if v != 0}
    expected = {x["date"]: _decimal(x["cash"]) for x in entry["source_events"]}
    if actual != expected:
        raise ValueError("Source dividend window differs from pinned observations")
    out = frame[["asset_id", "provider_symbol", "session_date", "currency_original", "dividends"]].rename(
        columns={"dividends": "source_cash"}
    )
    out["candidate_cash"] = 0.0
    target = day.eq(entry["issuer_ex_date"])
    out.loc[target, "candidate_cash"] = float(normalized_amount(entry))
    out["action_status"] = "unchanged_no_event"
    out.loc[out.source_cash.ne(0) & ~target, "action_status"] = "candidate_suppressed_source_event"
    out.loc[target, "action_status"] = "candidate_reconstructed_event"
    if entry.get("issuer_cash_correction") is not None:
        out.loc[target, "action_status"] = "candidate_corrected_issuer_cash"
    out["basis_status"] = entry["basis_status"]
    out["source_sha256"] = source_sha256
    out["production_eligible"] = False
    return out


def build(s3, bucket, *, version, run_id, execute=False):
    check_id(run_id)
    if not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("Expected pinned registry checksum")
    raw = verified_bytes(s3, bucket, {"key": f"{REGISTRY_ROOT}/{version}/registry.json", "sha256": version})
    doc = validate_registry(json.loads(raw))
    if json_bytes(doc) != raw:
        raise ValueError("Noncanonical registry")
    frames = []
    for e in doc["entries"]:
        source = verified_frame(s3, bucket, {"key": e["source_key"], "sha256": e["source_sha256"]})
        frames.append(reconstruct(source, e, source_sha256=e["source_sha256"]))
    result = pd.concat(frames, ignore_index=True)
    manifest = {
        "schema": "alpha_edge_dividend_event_candidate_v1",
        "run_id": run_id,
        "registry_version": version,
        "assets": len(frames),
        "rows": len(result),
        "source_positive_events": int(result.source_cash.gt(0).sum()),
        "candidate_positive_events": int(result.candidate_cash.gt(0).sum()),
        "provisional_basis_assets": sum(e["basis_status"] == "provisional" for e in doc["entries"]),
        "production_eligible": False,
        "prices_changed": False,
    }
    if not execute:
        return {**manifest, "status": "dry_run"}
    root = f"{OUTPUT_ROOT}/runs/{run_id}"
    blob = parquet_bytes(result)
    key = f"{root}/objects/events-{digest(blob)}.parquet"
    manifest["output"] = put_immutable(
        s3, bucket, key, blob, content_type="application/vnd.apache.parquet", scope=OUTPUT_ROOT
    )
    manifest_key = f"{root}/manifest.json"
    put_immutable(s3, bucket, manifest_key, json_bytes(manifest), content_type="application/json", scope=OUTPUT_ROOT)
    return {**manifest, "manifest_key": manifest_key, "status": "candidate_written"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--registry-version", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                boto3.client("s3", region_name=args.region),
                args.bucket,
                version=args.registry_version,
                run_id=args.run_id,
                execute=args.execute,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
