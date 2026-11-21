"""Index existing Silver candidates and classify findings without rewriting bars."""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict

import boto3
import pandas as pd

from alpha_edge.market.build_silver_sessions_v3 import SILVER_ROOT, SILVER_SCHEMA, SilverSessionError, _put
from alpha_edge.market.ingest_bronze_full_history import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    SAFE_RUN,
    _digest,
    _get_optional,
    _json_bytes,
    _read,
    load_universe,
)
from alpha_edge.market.lakehouse_v3.storage import verified_bytes


def classify(issue: str) -> str:
    if issue == "missing_endpoint_session":
        return "coverage_endpoint_review"
    if issue == "missing_expected_session":
        return "historical_gap_review"
    if issue == "unexpected_session":
        return "calendar_review"
    if issue == "non_trading_placeholder":
        return "provider_placeholder"
    if issue == "calendar_mapping_not_configured":
        return "mapping_blocker"
    if issue == "inconsistent_raw_ohlc":
        return "price_integrity_review"
    if issue in {"invalid_session_date", "duplicate_source_session", "missing_offset_aware_timestamp", "negative_volume"}:
        return "source_integrity_review"
    return "unclassified_review"


def consolidate(s3, bucket: str, run_ids: list[str], *, execute: bool = False) -> dict:
    if not run_ids or len(set(run_ids)) != len(run_ids) or any(not SAFE_RUN.fullmatch(r) for r in run_ids):
        raise SilverSessionError("Invalid or repeated Silver run IDs")
    configs = {}
    for run in run_ids:
        key = f"{SILVER_ROOT}/runs/{run}/run.json"
        raw = _read(s3, bucket, key)
        cfg = json.loads(raw)
        if cfg["silver_run_id"] != run or cfg["status"] != "candidate_only_not_certified":
            raise SilverSessionError("Unexpected run identity or status")
        configs[run] = (cfg, {"key": key, "sha256": _digest(raw)})
    base = configs[run_ids[0]][0]
    for cfg, _ in configs.values():
        for field in ("source_fingerprint", "source_git_commit", "registry_version", "calendar_policy_version",
                      "calendar_library_version", "exchange_calendars_version", "session_eligibility_policy"):
            if cfg.get(field) != base.get(field):
                raise SilverSessionError(f"Incompatible run configuration: {field}")
    universe, _ = load_universe(s3, bucket, base["source_fingerprint"])
    expected = {a["asset_id"]: a for a in universe}
    indexed = {}
    counts = Counter()
    groups = defaultdict(set)
    totals = Counter()
    by_source = Counter()
    by_calendar = Counter()
    affected = defaultdict(set)
    for run, (cfg, _) in configs.items():
        for i, aid in enumerate(cfg["selected_asset_ids"], 1):
            if i == 1 or i % 50 == 0:
                print(f"[consolidate] {run} {i}/{len(cfg['selected_asset_ids'])}", file=sys.stderr, flush=True)
            if aid not in expected:
                raise SilverSessionError(f"Asset outside pinned universe: {aid}")
            prefix = f"{SILVER_ROOT}/runs/{run}/assets/{_digest(aid.encode())}"
            key = f"{prefix}/manifest.json"
            raw = _get_optional(s3, bucket, key)
            if raw is None:
                continue
            if aid in indexed:
                raise SilverSessionError(f"Duplicate committed candidate: {aid}")
            sha = _digest(raw)
            if s3.head_object(Bucket=bucket, Key=key)["Metadata"].get("sha256") != sha:
                raise SilverSessionError(f"Manifest checksum mismatch: {aid}")
            m = json.loads(raw)
            if (m["schema"] != SILVER_SCHEMA or m["asset_id"] != aid or m["silver_run_id"] != run
                    or m["production_eligible"] or m["provider_symbol"] != expected[aid]["yahoo_ticker"]
                    or m.get("registry_version") != cfg.get("registry_version")
                    or m.get("calendar_policy_version") != cfg.get("calendar_policy_version")
                    or (m.get("correction_version") and m["correction_version"] != cfg.get("correction_version"))):
                raise SilverSessionError(f"Incompatible candidate manifest: {aid}")
            for name in ("candidate", "findings"):
                if not m["outputs"][name]["key"].startswith(f"{prefix}/objects/{name}-"):
                    raise SilverSessionError(f"Invalid artifact reference: {aid}")
            ref = m["outputs"]["findings"]
            blob = verified_bytes(s3, bucket, ref)
            if len(blob) != ref["bytes"]:
                raise SilverSessionError(f"Findings length mismatch: {aid}")
            findings = pd.read_parquet(io.BytesIO(blob))
            if len(findings) != m["quality"]["issue_rows"] or not findings["asset_id"].eq(aid).all():
                raise SilverSessionError(f"Findings reconciliation failed: {aid}")
            calendar = m["calendar_id"] or "UNRESOLVED"
            by_calendar[calendar] += 1
            for row in findings.itertuples(index=False):
                if (row.calendar_id or "UNRESOLVED") != calendar:
                    raise SilverSessionError(f"Findings calendar mismatch: {aid}")
                issue = str(row.issue)
                day = str(row.session_date) if pd.notna(row.session_date) else "NO_DATE"
                counts[(calendar, issue, m["bronze_run_id"])] += 1
                groups[(calendar, issue, day)].add(aid)
                affected[issue].add(aid)
            totals.update({k: v for k, v in m["quality"].items() if type(v) is int})
            by_source[m["bronze_run_id"]] += 1
            indexed[aid] = {"manifest_key": key, "manifest_sha256": sha, "silver_run_id": run,
                            "calendar_id": calendar, "correction_version": m.get("correction_version"),
                            "outputs": m["outputs"]}
    missing = sorted(set(expected) - set(indexed))
    if missing:
        raise SilverSessionError(f"Incomplete consolidated coverage: {missing}")
    classification = Counter()
    by_issue = Counter()
    for (_, issue, _), count in counts.items():
        classification[classify(issue)] += count
        by_issue[issue] += count
    report = {
        "schema": "alpha_edge_silver_consolidated_candidate_v1",
        "production_eligible": False,
        "source_fingerprint": base["source_fingerprint"],
        "registry_version": base.get("registry_version"),
        "calendar_policy_version": base.get("calendar_policy_version"),
        "source_runs": {r: ref for r, (_, ref) in configs.items()},
        "assets": indexed,
        "summary": {"selected": len(expected), "indexed": len(indexed), "quality_totals": dict(totals),
                    "by_bronze_run": dict(by_source), "assets_by_calendar": dict(by_calendar),
                    "by_category": dict(classification), "findings_by_issue": dict(by_issue),
                    "affected_assets_by_issue": {k: len(v) for k, v in affected.items()}},
        "by_calendar_issue_source": [
            {"calendar": cal, "issue": issue, "bronze_run": source, "findings": n}
            for (cal, issue, source), n in sorted(counts.items())
        ],
        "by_calendar_issue_date": [
            {"calendar": cal, "issue": issue, "session_date": day, "affected_assets": len(ids),
             "calendar_assets": by_calendar[cal],
             "asset_ids": sorted(ids)}
            for (cal, issue, day), ids in sorted(groups.items())
        ],
        "validation_scope": "manifest_and_findings_integrity; candidate_bytes_not_reverified",
    }
    payload = _json_bytes(report)
    key = f"{SILVER_ROOT}/consolidated/{_digest(payload)}/index.json"
    if execute:
        _put(s3, bucket, key, payload, "application/json")
    return {"executed": execute, "index_s3_uri": f"s3://{bucket}/{key}", **report["summary"],
            "by_calendar_issue_source": report["by_calendar_issue_source"],
            "top_date_clusters": [
                {k: v for k, v in row.items() if k != "asset_ids"}
                for row in sorted(report["by_calendar_issue_date"], key=lambda r: -r["affected_assets"])[:30]
            ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(consolidate(boto3.client("s3", region_name=args.region), args.bucket,
                                 args.run_id, execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
