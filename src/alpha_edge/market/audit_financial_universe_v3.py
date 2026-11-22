"""S3-only immutable, checksum-pinned corporate-action diagnostic evidence."""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import boto3
import pandas as pd

from alpha_edge.market import (
    dividend_duplicate_evidence_v3 as duplicate,
    dividend_duplicate_registry_v3 as duplicate_registry,
    historical_quarantine_v3 as quarantine,
)
from alpha_edge.market.build_silver_sessions_v3 import SILVER_ROOT, SILVER_SCHEMA
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
from alpha_edge.market.lakehouse_v3.storage import (
    ROOT,
    digest,
    json_bytes,
    parquet_bytes,
    put_immutable,
    verified_bytes,
)
from alpha_edge.market.silver_financial_evidence_v3 import extract_source_actions

AUDIT_ROOT = f"{ROOT}/silver/financial_diagnostics/v1"
SCHEMA = "alpha_edge_financial_diagnostic_v1"


def load_consolidated(s3, bucket, version):
    if not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("Expected consolidated index SHA-256")
    key = f"{SILVER_ROOT}/consolidated/{version}/index.json"
    index = json.loads(verified_bytes(s3, bucket, {"key": key, "sha256": version}))
    if (
        index.get("schema") != "alpha_edge_silver_consolidated_candidate_v1"
        or index.get("production_eligible") is not False
    ):
        raise ValueError("Invalid consolidated candidate index")
    return index


def consolidated_receipt(s3, bucket, index, row):
    aid = row["asset_id"]
    entry = index["assets"][aid]
    run = entry["silver_run_id"]
    expected_key = f"{SILVER_ROOT}/runs/{run}/assets/{digest(aid.encode())}/manifest.json"
    if not SAFE_RUN.fullmatch(run) or run not in index["source_runs"] or entry["manifest_key"] != expected_key:
        raise ValueError(f"Invalid consolidated manifest reference: {aid}")
    manifest = json.loads(verified_bytes(s3, bucket, {"key": expected_key, "sha256": entry["manifest_sha256"]}))
    if (
        manifest.get("schema") != SILVER_SCHEMA
        or manifest.get("asset_id") != aid
        or manifest.get("provider_symbol") != row["yahoo_ticker"]
        or manifest.get("silver_run_id") != run
        or manifest.get("production_eligible") is not False
        or manifest.get("outputs") != entry["outputs"]
    ):
        raise ValueError(f"Consolidated manifest identity mismatch: {aid}")
    bronze_run = manifest["bronze_run_id"]
    if not SAFE_RUN.fullmatch(bronze_run):
        raise ValueError(f"Invalid consolidated Bronze run: {aid}")
    expected_receipt = f"{BRONZE_ROOT}/runs/{bronze_run}/assets/{digest(aid.encode())}/manifest.json"
    if manifest.get("source_manifest_key") != expected_receipt:
        raise ValueError(f"Consolidated Bronze receipt reference mismatch: {aid}")
    receipt = _completed(s3, bucket, run_id=bronze_run, record=row, verify_hash=False)
    if (
        receipt is None
        or receipt["data"]["key"] != manifest.get("source_data_key")
        or receipt["data"]["sha256"] != manifest.get("source_data_sha256")
    ):
        raise ValueError(f"Consolidated Bronze evidence mismatch: {aid}")
    return receipt


def assess_asset(
    frame,
    *,
    asset_id,
    provider_symbol,
    source_key,
    sha,
    currency_registry,
    quarantine_registry,
    duplicate_correction_registry=None,
):
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
    if duplicate_correction_registry is not None:
        duplicates = duplicate_registry.mark_legitimate_pairs(
            duplicates,
            corrected,
            duplicate_correction_registry,
            asset_id=asset_id,
            provider_symbol=provider_symbol,
            bronze_sha256=sha,
        )
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
        "verified_legitimate_pairs": int(duplicates.review_status.eq("verified_legitimate_distributions").sum()),
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
            "approved_duplicate_suppressions_validated": int(
                coverage["approved_duplicate_suppressions_validated"].sum()
            )
            if "approved_duplicate_suppressions_validated" in coverage
            else 0,
            "verified_legitimate_pairs": int(coverage["verified_legitimate_pairs"].sum())
            if "verified_legitimate_pairs" in coverage
            else 0,
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
    fingerprint=None,
    bronze_run_id=None,
    consolidated_version=None,
    run_id,
    currency_version,
    quarantine_version=None,
    duplicate_registry_version=None,
    selectors=None,
    execute=False,
    artifact_format="parquet",
    workers=1,
):
    if not SAFE_RUN.fullmatch(run_id) or not 1 <= workers <= 16:
        raise ValueError("Invalid pinned run ID")
    if consolidated_version:
        if fingerprint is not None or bronze_run_id is not None:
            raise ValueError("Specify consolidated index or single Bronze run, not both")
        index = load_consolidated(s3, bucket, consolidated_version)
        fingerprint = index["source_fingerprint"]
    else:
        if not fingerprint or not bronze_run_id or not SAFE_RUN.fullmatch(bronze_run_id):
            raise ValueError("Expected pinned source fingerprint and Bronze run")
        index = None
    active, source = load_universe(s3, bucket, fingerprint)
    if index is not None:
        if set(index["assets"]) != {row["asset_id"] for row in active}:
            raise ValueError("Consolidated index does not cover exact active universe")
    else:
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

    def process(row):
        asset_id = row["asset_id"]
        if index is not None:
            receipt = consolidated_receipt(s3, bucket, index, row)
        else:
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
        approved = 0
        if proposals is not None:
            actions, _ = extract_source_actions(frame)
            actions, _ = apply_currency_corrections(
                actions, currency, asset_id=asset_id, provider_symbol=row["yahoo_ticker"], bronze_sha256=ref["sha256"]
            )
            _, stats = duplicate_registry.apply_suppressions(
                actions, proposals, asset_id=asset_id, provider_symbol=row["yahoo_ticker"], bronze_sha256=ref["sha256"]
            )
            approved = stats["approved"]
        summary, d, q, _ = assess_asset(
            frame,
            asset_id=asset_id,
            provider_symbol=row["yahoo_ticker"],
            source_key=ref["key"],
            sha=ref["sha256"],
            currency_registry=currency,
            quarantine_registry=external,
            duplicate_correction_registry=proposals,
        )
        summary["approved_duplicate_suppressions_validated"] = approved
        return summary, d, q

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (summary, d, q) in enumerate(pool.map(process, sorted(selected, key=lambda r: r["asset_id"])), 1):
            summaries.append(summary)
            dividends.append(d)
            quarantines.append(q)
            if i % 50 == 0 or i == len(selected):
                print(f"[financial audit] {i}/{len(selected)}", file=sys.stderr, flush=True)
    inputs = {
        "bronze_run_id": bronze_run_id,
        "consolidated_index_version": consolidated_version,
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
    parser.add_argument("--source-fingerprint")
    parser.add_argument("--bronze-run-id")
    parser.add_argument("--consolidated-index-version")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dividend-currency-registry-version", required=True)
    parser.add_argument("--historical-quarantine-registry-version")
    parser.add_argument("--dividend-duplicate-registry-version")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--only", nargs="+")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--artifact-format", choices=["parquet", "json"], default="parquet")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                boto3.client("s3", region_name=args.region),
                args.bucket,
                fingerprint=args.source_fingerprint,
                bronze_run_id=args.bronze_run_id,
                consolidated_version=args.consolidated_index_version,
                run_id=args.run_id,
                currency_version=args.dividend_currency_registry_version,
                quarantine_version=args.historical_quarantine_registry_version,
                duplicate_registry_version=args.dividend_duplicate_registry_version,
                selectors=args.only,
                execute=args.execute,
                artifact_format=args.artifact_format,
                workers=args.workers,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
