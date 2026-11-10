"""Join immutable corporate-action candidate events to Lake v2 adjustment flags.

Local read-only evidence; source event proximity is not proof of causal adjustment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from alpha_edge.market.audit_corporate_actions_v2 import inspect_adjustment_changes
from alpha_edge.market.ingest_corporate_actions_v2 import load_verified_batch
from alpha_edge.market.publish_lake_v2 import validate_local_candidate


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def reconcile_events(
    *, gold: pd.DataFrame, events: pd.DataFrame, threshold: float = 0.005,
) -> pd.DataFrame:
    actions = events.rename(columns={
        "value": "split_factor",
        "effective_date": "effective_date",
    })
    flags = inspect_adjustment_changes(
        gold, actions, threshold=threshold, event_window_days=0,
    )
    if flags.empty:
        flags["evidence_status"] = pd.Series(dtype="object")
        return flags
    flags["evidence_status"] = flags["nearest_action_type"].apply(
        lambda value: "same_day_reported_action" if pd.notna(value)
        else "unmatched_requires_review"
    )
    return flags


def run(*, candidate_dir: Path, action_batch_dir: Path, output: Path,
        threshold: float = 0.005) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    validate_local_candidate(candidate_dir)
    events, batch_manifest = load_verified_batch(action_batch_dir)
    gold = pd.read_parquet(candidate_dir / "gold.parquet")
    flags = reconcile_events(gold=gold, events=events, threshold=threshold)
    unexplained = int(flags["evidence_status"].eq("unmatched_requires_review").sum())
    explained = int(flags["evidence_status"].eq("same_day_reported_action").sum())
    source_assets = set(gold["asset_id"].astype(str).unique())
    event_assets = set(events["asset_id"].astype(str).unique())
    # A symbol can have no actions and thus no rows in events. The acquisition
    # manifest, not the event table, tracks the count of fully queried assets.
    if not event_assets.issubset(source_assets):
        raise ValueError("Event batch contains assets not in candidate Gold")
    if set(batch_manifest["asset_ids"]) != source_assets:
        raise ValueError("Corporate action batch does not cover all pilot assets")
    output.mkdir(parents=True, exist_ok=False)
    flags.to_csv(output / "adjustment_event_matches.csv", index=False)
    events.to_csv(output / "reported_events.csv", index=False)
    summary = {
        "schema": "alpha_edge_corporate_action_v2_reconciliation_1",
        "candidate_manifest_sha256": _sha((candidate_dir / "manifest.json").read_bytes()),
        "corporate_actions_manifest_sha256": _sha((action_batch_dir / "manifest.json").read_bytes()),
        "reported_dividends": int(events["action_type"].eq("DIVIDEND").sum()),
        "reported_splits": int(events["action_type"].eq("SPLIT").sum()),
        "ratio_alerts": len(flags),
        "same_day_reported_actions": explained,
        "unmatched_alerts": unexplained,
        "threshold": threshold,
        "outcome": "review_required",
        "s3_writes": 0,
        "limitations": (
            "Same-day action event is corroborating evidence, not proof of exact adjustment math. "
            "Rights issues and common raw/adjusted price rescalings remain outside this audit. "
            "No price, return, or corporate-action dataset is changed."
        ),
        "files_sha256": {
            fn: _sha((output / fn).read_bytes())
            for fn in ("adjustment_event_matches.csv", "reported_events.csv")
        },
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--actions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.005)
    args = parser.parse_args()
    print(json.dumps(run(
        candidate_dir=args.candidate_dir, action_batch_dir=args.actions_dir,
        output=args.output, threshold=args.threshold,
    ), indent=2))


if __name__ == "__main__":
    main()
