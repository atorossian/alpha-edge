"""S3-only immutable, checksum-pinned corporate-action diagnostic evidence."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter

import boto3
import pandas as pd

from alpha_edge.market import (
    dividend_duplicate_evidence_v3 as duplicate,
    dividend_duplicate_registry_v3 as duplicate_registry,
    historical_quarantine_v3 as quarantine,
)
from alpha_edge.market.dividend_currency_registry_v3 import apply_currency_corrections, load_registry
from alpha_edge.market.ingest_bronze_full_history import (
    BRONZE_ROOT,
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    SAFE_RUN,
    _completed,
    _read,
    load_universe,
)
from alpha_edge.market.lakehouse_v3.storage import ROOT, digest, json_bytes, parquet_bytes, put_immutable
from alpha_edge.market.silver_financial_evidence_v3 import extract_source_actions

AUDIT_ROOT = f"{ROOT}/silver/financial_diagnostics/v1"
SCHEMA = "alpha_edge_financial_diagnostic_v1"


def assess_asset(frame, *, asset_id, provider_symbol, source_key, sha, currency_registry, quarantine_registry):
    """Shared in-memory evidence calculation; does not write or repair anything."""
    actions, action_stats = extract_source_actions(frame)
    corrected, corrections = apply_currency_corrections(
        actions,
        currency_registry,
        asset_id=asset_id,
        provider_symbol=provider_symbol,
        bronze_sha256=sha,
    )
    duplicates, duplicate_stats = duplicate.assess(frame, corrected)
    _, findings, quarantine_stats = quarantine.assess(
        frame,
        corrected,
        asset_id=asset_id,
        provider_symbol=provider_symbol,
        bronze_sha256=sha,
        registry=quarantine_registry,
        duplicate_findings=duplicates,
    )
    identity = {
        "asset_id": asset_id,
        "provider_symbol": provider_symbol,
        "bronze_key": source_key,
        "bronze_sha256": sha,
    }
    for table in (duplicates, findings):
        for key, value in identity.items():
            table[key] = value
    summary = {
        **identity,
        "source_rows": len(frame),
        "reported_dividends": action_stats["reported_dividends"],
        "reported_splits": action_stats["reported_splits"],
        "approved_currency_corrections": corrections["approved"],
        "duplicate_findings": len(duplicates),
        "quarantine_findings": len(findings),
        "quarantine_through": quarantine_stats["quarantine_through"],
        "quarantined_sessions": quarantine_stats["quarantined_sessions"],
        "candidate_sessions": quarantine_stats["candidate_sessions"],
        "candidate_returns": quarantine_stats["candidate_returns"],
    }
    return summary, duplicates, findings, duplicate_stats


def publish(
    s3,
    bucket,
    *,
    run_id,
    inputs,
    coverage,
    duplicate_findings,
    quarantine_findings,
    execute=False,
    artifact_format="parquet",
):
    """Publish content-addressed tables, then the immutable completion manifest."""
    if not SAFE_RUN.fullmatch(run_id):
        raise ValueError("Invalid diagnostic run ID")
    if artifact_format not in {"parquet", "json"}:
        raise ValueError("Unsupported diagnostic artifact format")
    content_type = "application/vnd.apache.parquet" if artifact_format == "parquet" else "application/json"
    root = f"{AUDIT_ROOT}/runs/{run_id}"
    frames = {"coverage": coverage, "dividend_findings": duplicate_findings, "quarantine_findings": quarantine_findings}
    counts = Counter(duplicate_findings["reason"].tolist())
    refs = {}
    payloads = {}
    for name, table in frames.items():
        blob = (
            parquet_bytes(table)
            if artifact_format == "parquet"
            else json_bytes({"rows": json.loads(table.to_json(orient="records", double_precision=15))})
        )
        sha = digest(blob)
        refs[name] = {
            "key": f"{root}/objects/{name}-{sha}.{artifact_format}",
            "sha256": sha,
            "bytes": len(blob),
            "rows": len(table),
            "format": artifact_format,
        }
        payloads[name] = blob
    manifest = {
        "schema": SCHEMA,
        "run_id": run_id,
        "inputs": inputs,
        "outputs": refs,
        "statistics": {
            "assets": len(coverage),
            "source_rows": int(coverage["source_rows"].sum()),
            "reported_dividends": int(coverage["reported_dividends"].sum()),
            "reported_splits": int(coverage["reported_splits"].sum()),
            "dividend_reasons": dict(counts),
            "quarantine_findings": len(quarantine_findings),
        },
        "production_eligible": False,
        "corrections_applied_to_prices": False,
    }
    if execute:
        for name, ref in refs.items():
            put_immutable(s3, bucket, ref["key"], payloads[name], content_type=content_type, scope=AUDIT_ROOT)
        put_immutable(
            s3, bucket, f"{root}/manifest.json", json_bytes(manifest), content_type="application/json", scope=AUDIT_ROOT
        )
    return {
        **manifest,
        "status": "candidate_written" if execute else "dry_run",
        "manifest_key": f"{root}/manifest.json",
    }


def audit(
    s3,
    bucket,
    *,
    fingerprint,
    bronze_run_id,
    run_id,
    currency_version,
    quarantine_version=None,
    duplicate_registry_version=None,
    selectors=None,
    execute=False,
    artifact_format="parquet",
):
    if not all(SAFE_RUN.fullmatch(v) for v in (bronze_run_id, run_id)):
        raise ValueError("Invalid pinned run ID")
    active, source = load_universe(s3, bucket, fingerprint)
    config = json.loads(_read(s3, bucket, f"{BRONZE_ROOT}/runs/{bronze_run_id}/run.json"))
    if config.get("source_fingerprint") != fingerprint:
        raise ValueError("Bronze fingerprint mismatch")
    selected = active if selectors is None else [row for row in active if row["asset_id"] in selectors]
    if not selected or (selectors is not None and len(selected) != len(set(selectors))):
        raise ValueError("Expected known canonical asset IDs")
    currency = load_registry(s3, bucket, currency_version)
    external = quarantine.load_registry(s3, bucket, quarantine_version) if quarantine_version else None
    proposals = (
        duplicate_registry.load_registry(s3, bucket, duplicate_registry_version) if duplicate_registry_version else None
    )
    summaries, dividends, quarantines = [], [], []
    for row in sorted(selected, key=lambda r: r["asset_id"]):
        asset_id = row["asset_id"]
        if asset_id not in config.get("requested_asset_ids", []):
            raise ValueError(f"Asset not included in pinned Bronze run: {asset_id}")
        receipt = _completed(s3, bucket, run_id=bronze_run_id, record=row, verify_hash=False)
        if receipt is None:
            raise ValueError(f"Missing verified Bronze receipt: {asset_id}")
        ref = receipt["data"]
        blob = _read(s3, bucket, ref["key"])
        if digest(blob) != ref["sha256"] or len(blob) != ref["bytes"]:
            raise ValueError(f"Bronze data checksum mismatch: {asset_id}")
        frame = pd.read_parquet(io.BytesIO(blob), engine="pyarrow")
        if len(frame) != receipt["validation"]["rows"]:
            raise ValueError(f"Bronze row count mismatch: {asset_id}")
        if proposals is not None:
            actions, _ = extract_source_actions(frame)
            duplicate_registry.apply_suppressions(
                actions, proposals, asset_id=asset_id, provider_symbol=row["yahoo_ticker"], bronze_sha256=ref["sha256"]
            )
        summary, d, q, _ = assess_asset(
            frame,
            asset_id=asset_id,
            provider_symbol=row["yahoo_ticker"],
            source_key=ref["key"],
            sha=ref["sha256"],
            currency_registry=currency,
            quarantine_registry=external,
        )
        summaries.append(summary)
        dividends.append(d)
        quarantines.append(q)
    inputs = {
        "bronze_run_id": bronze_run_id,
        "source_fingerprint": fingerprint,
        "source_git_commit": source["source_git_commit"],
        "currency_registry_version": currency_version,
        "quarantine_registry_version": quarantine_version,
        "duplicate_registry_version": duplicate_registry_version,
        "duplicate_policy": duplicate.POLICY,
        "quarantine_policy": quarantine.POLICY,
        "scope": "full_universe" if selectors is None else "selected_assets",
    }
    return publish(
        s3,
        bucket,
        run_id=run_id,
        inputs=inputs,
        coverage=pd.DataFrame(summaries),
        duplicate_findings=pd.concat([d for d in dividends if not d.empty] or dividends[:1], ignore_index=True),
        quarantine_findings=pd.concat([q for q in quarantines if not q.empty] or quarantines[:1], ignore_index=True),
        execute=execute,
        artifact_format=artifact_format,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--bronze-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dividend-currency-registry-version", required=True)
    parser.add_argument("--historical-quarantine-registry-version")
    parser.add_argument("--dividend-duplicate-registry-version")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--only", nargs="+")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--artifact-format", choices=["parquet", "json"], default="parquet")
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                boto3.client("s3", region_name=args.region),
                args.bucket,
                fingerprint=args.source_fingerprint,
                bronze_run_id=args.bronze_run_id,
                run_id=args.run_id,
                currency_version=args.dividend_currency_registry_version,
                quarantine_version=args.historical_quarantine_registry_version,
                duplicate_registry_version=args.dividend_duplicate_registry_version,
                selectors=args.only,
                execute=args.execute,
                artifact_format=args.artifact_format,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
