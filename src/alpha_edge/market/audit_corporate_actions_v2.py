"""Read-only corporate-action evidence audit for Market Data Lake v2.

Queries the existing MarketStore corporate-action partition keys with S3
GetObject only; differentiates absent partitions from unreadable/corrupt data.
It does not apply adjustment factors or mutate Bronze, Silver, Gold or S3.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.market.publish_lake_v2 import validate_local_candidate


class CorporateActionAuditError(ValueError):
    pass


def _hash(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _fetch(s3, bucket: str, key: str) -> tuple[str, bytes | None]:
    try:
        return "present", s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404", "NotFound"}:
            return "missing", None
        raise CorporateActionAuditError(f"Unable to read s3://{bucket}/{key}: {code}") from exc


def _date_column(value: pd.Series, label: str) -> pd.Series:
    dates = pd.to_datetime(value.astype(str).str[:10], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise CorporateActionAuditError(f"Invalid {label} date")
    return dates


def load_existing_actions(
    s3, *, bucket: str, asset_ids: list[str],
    prefix: str = "market/reference/v1/corporate_actions",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict counterpart to MarketStore.read_corporate_actions for audit use."""
    if len(asset_ids) != len(set(asset_ids)) or not asset_ids:
        raise CorporateActionAuditError("Expected unique nonempty asset IDs")
    if any(not a or "/" in a or ".." in a for a in asset_ids):
        raise CorporateActionAuditError("Unsafe asset ID")
    rows = []
    sources = []
    for asset_id in asset_ids:
        key = f"{prefix}/asset_id={asset_id}/part-00000.parquet"
        status, raw = _fetch(s3, bucket, key)
        record = {"asset_id": asset_id, "source_key": key, "source_status": status,
                  "sha256": None, "rows": 0}
        if raw is None:
            sources.append(record)
            continue
        record["sha256"] = _hash(raw)
        try:
            table = pd.read_parquet(io.BytesIO(raw))
        except Exception as exc:
            raise CorporateActionAuditError(f"Corrupt corporate-action partition: {key}") from exc
        required = {"asset_id", "effective_date", "action_type", "split_factor"}
        if missing := required - set(table.columns):
            raise CorporateActionAuditError(f"Corporate-action partition {key} missing {sorted(missing)}")
        if not table["asset_id"].astype(str).eq(asset_id).all():
            raise CorporateActionAuditError(f"Corporate-action partition asset mismatch: {key}")
        table = table.copy()
        if not table.empty:
            table["effective_date"] = _date_column(table["effective_date"], "corporate action")
            if table.duplicated(["asset_id", "effective_date", "action_type"]).any():
                raise CorporateActionAuditError(f"Duplicate corporate actions: {key}")
            if table["action_type"].astype(str).ne("SPLIT").any():
                raise CorporateActionAuditError(f"Unrecognized legacy action type: {key}")
            factors = pd.to_numeric(table["split_factor"], errors="coerce")
            if not np.isfinite(factors).all() or factors.le(0).any():
                raise CorporateActionAuditError(f"Invalid split factor in {key}")
            table["split_factor"] = factors
            rows.append(table)
        record["rows"] = len(table)
        record["source_status"] = "present_with_rows" if len(table) else "present_empty"
        sources.append(record)
    columns = ["asset_id", "effective_date", "action_type", "split_factor",
               "ticker", "yahoo_ticker", "source", "source_action_id", "detected_at_utc", "notes"]
    actions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    return actions, pd.DataFrame(sources)


def inspect_adjustment_changes(
    gold: pd.DataFrame, actions: pd.DataFrame, *, threshold: float = 0.005,
    event_window_days: int = 7,
) -> pd.DataFrame:
    """Flag changes in Yahoo adjusted/raw close ratio without assigning causality."""
    required = {"asset_id", "session_date", "close", "adj_close"}
    if missing := required - set(gold.columns):
        raise CorporateActionAuditError(f"Gold missing fields: {sorted(missing)}")
    if threshold <= 0 or event_window_days < 0:
        raise CorporateActionAuditError("Invalid detection settings")
    data = gold[["asset_id", "session_date", "close", "adj_close"]].copy()
    data["session_date"] = _date_column(data["session_date"], "session")
    if data.duplicated(["asset_id", "session_date"]).any():
        raise CorporateActionAuditError("Duplicate Gold asset/date")
    data = data.sort_values(["asset_id", "session_date"], kind="stable")
    close = pd.to_numeric(data["close"], errors="coerce")
    adj = pd.to_numeric(data["adj_close"], errors="coerce")
    ratio = (adj / close).where(close.gt(0) & adj.gt(0))
    data["adj_to_close_ratio"] = ratio
    prev = data.groupby("asset_id")["adj_to_close_ratio"].shift()
    data["ratio_change"] = (ratio / prev - 1).where(prev.gt(0))
    flags = data.loc[data["ratio_change"].abs().gt(threshold)].copy()
    event_dates = actions[["asset_id", "effective_date", "action_type", "split_factor"]].copy()
    for col in ["nearest_action_date", "nearest_action_type", "nearest_action_distance_days"]:
        flags[col] = None
    for i, row in flags.iterrows():
        candidates = event_dates.loc[event_dates["asset_id"].eq(row["asset_id"])]
        if candidates.empty:
            continue
        dates = pd.to_datetime(candidates["effective_date"])
        distance = (dates - row["session_date"]).dt.days.abs()
        j = distance.idxmin()
        if distance.loc[j] <= event_window_days:
            flags.at[i, "nearest_action_date"] = dates.loc[j].date().isoformat()
            flags.at[i, "nearest_action_type"] = candidates.loc[j, "action_type"]
            flags.at[i, "nearest_action_distance_days"] = int(distance.loc[j])
    flags["diagnosis"] = "adjustment_ratio_change_requires_review"
    return flags.reset_index(drop=True)


def audit(
    *, s3, bucket: str, candidate_dir: Path, output: Path,
    threshold: float = 0.005, event_window_days: int = 7,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output}")
    manifest, _ = validate_local_candidate(candidate_dir)
    gold = pd.read_parquet(candidate_dir / "gold.parquet")
    assets = sorted(gold["asset_id"].dropna().astype(str).unique().tolist())
    actions, sources = load_existing_actions(s3, bucket=bucket, asset_ids=assets)
    flags = inspect_adjustment_changes(gold, actions, threshold=threshold,
                                       event_window_days=event_window_days)
    output.mkdir(parents=True)
    for filename, df in [
        ("corporate_action_sources.csv", sources),
        ("corporate_actions.csv", actions),
        ("adjustment_ratio_flags.csv", flags),
    ]:
        df.to_csv(output / filename, index=False)
    report = {
        "schema": "alpha_edge_lake_v2_corporate_action_evidence_1",
        "candidate_manifest_sha256": _hash((candidate_dir / "manifest.json").read_bytes()),
        "candidate_acceptance": manifest.get("acceptance"),
        "bucket": bucket,
        "source_prefix": "market/reference/v1/corporate_actions",
        "assets": len(assets),
        "present_partitions": int(sources["source_status"].ne("missing").sum()),
        "missing_partitions": int(sources["source_status"].eq("missing").sum()),
        "split_events": len(actions),
        "adjustment_ratio_flags": len(flags),
        "threshold": threshold,
        "event_window_days": event_window_days,
        "s3_writes": 0,
        "outcome": "review_required",
        "limitations": (
            "Legacy reference dataset contains split events only. Absent partitions do not "
            "prove there were no corporate actions. Ratio changes cannot diagnose rights issues, "
            "cash dividends, or Yahoo provider revisions without independent provenance. "
            "A constant scaling of both Close and Adj Close is invisible to this ratio check."
        ),
        "files_sha256": {
            filename: _hash((output / filename).read_bytes())
            for filename in ("corporate_action_sources.csv", "corporate_actions.csv",
                             "adjustment_ratio_flags.csv")
        },
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--event-window-days", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(audit(
        s3=boto3.client("s3", region_name=args.region), bucket=args.bucket,
        candidate_dir=args.candidate_dir, output=args.output,
        threshold=args.threshold, event_window_days=args.event_window_days,
    ), indent=2))


if __name__ == "__main__":
    main()
