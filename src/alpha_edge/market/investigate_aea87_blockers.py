"""AEA-87 targeted blocked-asset investigation; immutable staging evidence only.

For stale manifests, compares manifest/partition contents with staged replacements
and identifies newer dates or divergent overlapping adjusted closes. For quarantines,
records exact excluded dates and unresolved flagged counts, without auto-waivers.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import uuid
from collections import Counter
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd


def _bytes(s3, bucket, key):
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def _json(s3, bucket, key):
    raw = _bytes(s3, bucket, key)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _frame(s3, bucket, key, sha=None):
    raw = _bytes(s3, bucket, key)
    if sha is not None and hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"Staged partition checksum mismatch: {key}")
    df = pd.read_parquet(io.BytesIO(raw))
    if "date" not in df or "asset_id" not in df:
        raise ValueError(f"Missing date/asset identity: {key}")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.normalize()
    if df["date"].isna().any() or df["date"].duplicated().any():
        raise ValueError(f"Invalid/duplicate dates: {key}")
    return df.set_index("date").sort_index()


def _compare(staged, active, *, table):
    staged_dates, active_dates = staged.index, active.index
    new_dates = sorted(set(active_dates) - set(staged_dates))
    only_staged = sorted(set(staged_dates) - set(active_dates))
    common = staged_dates.intersection(active_dates)
    metric = "close_adjusted_usd" if table == "ohlcv_usd" else "ret_log_close_adjusted_usd"
    # Legacy production return partitions may predate the normalized return
    # column. Never misclassify an uncomparable column as matching data.
    alternatives = ("ret_log_close_adjusted_usd", "ret_close_adjusted_usd",
                    "ret_adj_close_usd")
    def values(frame):
        if metric in frame:
            return pd.to_numeric(frame.loc[common, metric], errors="coerce").to_numpy(), metric
        if table == "returns_usd":
            for column in alternatives[1:]:
                if column in frame:
                    simple = pd.to_numeric(frame.loc[common, column], errors="coerce").to_numpy()
                    if np.isfinite(simple).all() and (simple > -1).all():
                        return np.log1p(simple), column + " (converted via log1p)"
        return None, None
    a, staged_metric = values(staged)
    b, active_metric = values(active)
    comparable = a is not None and b is not None
    if comparable:
        differs = ~np.isclose(a, b, rtol=1e-9, atol=1e-11, equal_nan=True)
        changed = common[differs]
    else:
        differs = None
        changed = []
    return {
        "comparison_status": "COMPARED" if comparable else "INCOMPARABLE_SCHEMA",
        "staged_metric": staged_metric,
        "active_metric": active_metric,
        "staged_columns": list(staged.columns),
        "active_columns": list(active.columns),
        "active_only_dates": [x.strftime("%Y-%m-%d") for x in new_dates],
        "staged_only_dates": [x.strftime("%Y-%m-%d") for x in only_staged],
        "changed_overlap_dates": [x.strftime("%Y-%m-%d") for x in changed],
        "active_only_count": len(new_dates),
        "staged_only_count": len(only_staged),
        "changed_overlap_count": int(differs.sum()) if comparable else None,
    }


def investigate(s3, bucket, market_prefix, reconciliation_id, release_plan_key):
    root = f"{market_prefix.strip('/')}/repairs/aea87/staging/"
    rec, rec_hash = _json(s3, bucket, root + reconciliation_id + "/reconciliation.json")
    plan, plan_hash = _json(s3, bucket, release_plan_key)
    if rec.get("run_id") != reconciliation_id or plan.get("reconciliation_id") != reconciliation_id:
        raise ValueError("Reconciliation/release plan mismatch")
    if plan.get("reconciliation_sha256") != rec_hash:
        raise ValueError("Release plan references different reconciliation bytes")
    blocked = {x["asset_id"]: x for x in plan["assets"] if x["status"] == "BLOCKED"}
    selected = {x["asset_id"]: x for x in rec["assets"]}
    source_campaigns = {}
    for source in rec["sources"].values():
        manifest, sha = _json(s3, bucket, source["manifest_key"])
        if sha != source["manifest_sha256"]:
            raise ValueError("Staged source manifest changed")
        source_campaigns[source["run_id"]] = {a["asset_id"]: a for a in manifest["prepared"]}
    findings = []
    for aid in sorted(blocked):
        asset = selected[aid]
        records = source_campaigns[asset["campaign"]][aid]["records"]
        changes = []
        for part in records:
            key = part["manifest_key"]
            if "STALE_ACTIVE_MANIFEST" not in blocked[aid]["blockers"]:
                continue
            active_blob = _bytes(s3, bucket, key)
            actual_sha = hashlib.sha256(active_blob).hexdigest()
            if actual_sha == part.get("original_manifest_sha256"):
                continue
            manifest = json.loads(active_blob)
            active_parts = manifest.get("parts", [])
            if not active_parts:
                raise ValueError(f"Active manifest has no parts: {key}")
            active_df = pd.concat([_frame(s3, bucket, k) for k in active_parts])
            if active_df.index.has_duplicates:
                raise ValueError(f"Active manifest duplicate dates: {key}")
            stage_df = _frame(s3, bucket, part["staged_part"], part["sha256"])
            if (active_df["asset_id"].astype(str) != aid).any() or (stage_df["asset_id"].astype(str) != aid).any():
                raise ValueError(f"Identity mismatch: {key}")
            comparison = _compare(stage_df, active_df, table=part["table"])
            changes.append({"table": part["table"], "year": part["year"],
                            "manifest_key": key, "expected_sha256": part["original_manifest_sha256"],
                            "current_sha256": actual_sha, **comparison})
        findings.append({
            "asset_id": aid,
            "release_blockers": blocked[aid]["blockers"],
            "flagged_retained": asset["flagged_retained"],
            "quarantine_rows": asset["quarantine_rows"],
            "return_break_dates": asset["return_break_dates"],
            "changed_manifests": changes,
            "decision": "MANUAL_REVIEW_REQUIRED",
            "promotable": False,
        })
    counts = Counter(reason for x in findings for reason in x["release_blockers"])
    report = {
        "kind": "AEA87_TARGETED_REMEDIATION_EVIDENCE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reconciliation_id": reconciliation_id,
        "reconciliation_sha256": rec_hash,
        "release_plan_key": release_plan_key,
        "release_plan_sha256": plan_hash,
        "blocked_assets_checked": len(findings),
        "blocker_counts": dict(counts),
        "assets": findings,
        "production_promotion_allowed": False,
    }
    rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    key = root + reconciliation_id + "/remediation/" + rid + "/report.json"
    s3.put_object(Bucket=bucket, Key=key,
                  Body=json.dumps(report, indent=2, sort_keys=True, default=str).encode(),
                  ContentType="application/json", IfNoneMatch="*")
    print(json.dumps({"blocked_assets_checked": len(findings),
                      "blocker_counts": dict(counts)}, indent=2))
    print(f"REMEDIATION_REPORT s3://{bucket}/{key}")
    return report


def main():
    p = argparse.ArgumentParser(description="AEA-87 targeted blocked-assets review (read-only production)")
    p.add_argument("--reconciliation-id", required=True)
    p.add_argument("--release-plan-key", required=True, help="S3 object key, without s3://bucket")
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--market-prefix", default="market")
    args = p.parse_args()
    investigate(boto3.client("s3", region_name=args.region), args.bucket,
                args.market_prefix, args.reconciliation_id, args.release_plan_key)


if __name__ == "__main__":
    main()
