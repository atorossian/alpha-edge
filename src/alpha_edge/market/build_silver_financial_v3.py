"""Silver v3 stage 2: immutable S3 corporate-action and FX source evidence.

This is a candidate-evidence stage only. It does NOT apply splits to Yahoo
Close, attach future FX to stock prices, make return calculations, or promote
financially unverified data to Gold.
"""

from __future__ import annotations

import argparse
import io
import json
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.market.ingest_bronze_full_history import (
    BRONZE_ROOT,
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    SAFE_RUN,
    _completed,
    _digest,
    _get_optional,
    _json_bytes,
    _read,
    load_universe,
)
from alpha_edge.market.silver_financial_evidence_v3 import (
    extract_source_actions,
    normalize_fx_candidate,
    source_adjustment_flags,
)

ROOT = "market/lake_v3/silver/financial_evidence"
SCHEMA = "alpha_edge_silver_financial_evidence_v1"


class SilverEvidenceError(ValueError):
    """Financial evidence source or immutable publication contract failed."""


def _put(s3: Any, bucket: str, key: str, payload: bytes, content_type: str) -> None:
    if not key.startswith(ROOT + "/"):
        raise SilverEvidenceError("Refusing write outside Silver financial evidence")
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=payload,
            ContentType=content_type,
            Metadata={"sha256": _digest(payload)},
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"PreconditionFailed", "412"}:
            raise
        if _read(s3, bucket, key) != payload:
            raise SilverEvidenceError(f"Immutable S3 financial evidence differs: {key}") from exc


def build(
    s3: Any,
    bucket: str,
    *,
    fingerprint: str,
    bronze_run_id: str,
    silver_run_id: str,
    selector: str,
    fx_currency: str | None = None,
    fx_direction: str | None = None,
    execute: bool = False,
) -> dict:
    """Only one pinned, verified Bronze asset; no broad implicit universe run."""
    if not SAFE_RUN.fullmatch(bronze_run_id) or not SAFE_RUN.fullmatch(silver_run_id):
        raise SilverEvidenceError("Invalid pinned run ID")
    if (fx_currency is None) != (fx_direction is None):
        raise SilverEvidenceError("FX currency and direction must be supplied together")
    if fx_currency is not None:
        if fx_currency != fx_currency.upper() or len(fx_currency) != 3 or not fx_currency.isalpha():
            raise SilverEvidenceError("Expected explicit ISO currency code")
        if fx_direction not in {"direct", "inverse"}:
            raise SilverEvidenceError("Explicit FX quote direction must be direct or inverse")
    active, source = load_universe(s3, bucket, fingerprint)
    selected = [row for row in active if selector in (row["asset_id"], row["ticker"], row["yahoo_ticker"])]
    if len(selected) != 1:
        raise SilverEvidenceError("Exactly one included canonical asset must match")
    record = selected[0]
    config = json.loads(_read(s3, bucket, f"{BRONZE_ROOT}/runs/{bronze_run_id}/run.json"))
    if config.get("source_fingerprint") != fingerprint or record["asset_id"] not in config.get(
        "requested_asset_ids", []
    ):
        raise SilverEvidenceError("Bronze configuration/asset selection mismatch")
    receipt = _completed(s3, bucket, run_id=bronze_run_id, record=record, verify_hash=True)
    if receipt is None:
        raise SilverEvidenceError("Selected asset has no verified committed Bronze source")
    payload = _read(s3, bucket, receipt["data"]["key"])
    if _digest(payload) != receipt["data"]["sha256"]:
        raise SilverEvidenceError("Bronze data byte checksum mismatch")
    frame = pd.read_parquet(io.BytesIO(payload), engine="pyarrow")
    actions, action_stats = extract_source_actions(frame)
    flags = source_adjustment_flags(frame, actions)
    outputs = {"actions": actions, "ratio_flags": flags}
    fx_stats = None
    if fx_currency:
        quotes = frame[["provider_symbol", "session_date", "close"]].rename(columns={"close": "quote_close"})
        rates, fx_issues, fx_stats = normalize_fx_candidate(
            quotes, pairs={fx_currency: (record["yahoo_ticker"], fx_direction)}
        )
        outputs["fx_rates"] = rates
        outputs["fx_findings"] = fx_issues
    config_manifest = {
        "schema": SCHEMA,
        "run_id": silver_run_id,
        "bronze_run_id": bronze_run_id,
        "source_fingerprint": fingerprint,
        "source_git_commit": source["source_git_commit"],
        "asset_id": record["asset_id"],
        "provider_symbol": record["yahoo_ticker"],
        "source_data_key": receipt["data"]["key"],
        "source_sha256": receipt["data"]["sha256"],
        "fx_currency": fx_currency or "",
        "fx_direction": fx_direction or "",
        "action_statistics": action_stats,
        "ratio_flags": len(flags),
        "fx_statistics": fx_stats,
        "production_eligible": False,
        "financial_validation": "source_evidence_only",
    }
    root = f"{ROOT}/runs/{silver_run_id}/assets/{_digest(record['asset_id'].encode('utf-8'))}"
    prior = _get_optional(s3, bucket, root + "/manifest.json")
    if prior is not None:
        previous = json.loads(prior)
        if previous.get("inputs") != config_manifest:
            raise SilverEvidenceError("Existing immutable financial evidence run uses different inputs")
        return {
            "status": "already_committed",
            "source": receipt["data"]["key"],
            "statistics": config_manifest,
            "objects": previous["outputs"],
            "production_eligible": False,
        }
    result = {
        "status": "dry_run" if not execute else "candidate_written",
        "source": receipt["data"]["key"],
        "statistics": config_manifest,
        "outputs": {name: len(df) for name, df in outputs.items()},
        "production_eligible": False,
    }
    if not execute:
        return result
    published = {}
    for name, data in outputs.items():
        stream = io.BytesIO()
        data.to_parquet(stream, index=False)
        blob = stream.getvalue()
        key = f"{root}/objects/{name}-{_digest(blob)}.parquet"
        _put(s3, bucket, key, blob, "application/vnd.apache.parquet")
        published[name] = {
            "key": key,
            "sha256": _digest(blob),
            "bytes": len(blob),
            "rows": len(data),
        }
    manifest = {"inputs": config_manifest, "outputs": published}
    _put(s3, bucket, root + "/manifest.json", _json_bytes(manifest), "application/json")
    result["manifest_s3_uri"] = f"s3://{bucket}/{root}/manifest.json"
    result["output_objects"] = published
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Silver v3 immutable source-financial evidence pilot")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--bronze-run-id", required=True)
    parser.add_argument("--silver-run-id", required=True)
    parser.add_argument("--only", required=True)
    parser.add_argument("--fx-currency")
    parser.add_argument("--fx-direction", choices=["direct", "inverse"])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    report = build(
        s3,
        args.bucket,
        fingerprint=args.source_fingerprint,
        bronze_run_id=args.bronze_run_id,
        silver_run_id=args.silver_run_id,
        selector=args.only,
        fx_currency=args.fx_currency,
        fx_direction=args.fx_direction,
        execute=args.execute,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
