"""Compare TWO immutable Yahoo Bronze v2 snapshots; never use AEA-87.

The result is revision evidence, NOT automatic correction or source truth.
No Yahoo calls, S3 access, or production writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_edge.market.run_standardization_pilot_v2 import _source_inputs


class RevisionContractError(ValueError):
    pass


PRICE_FIELDS = ("open", "high", "low", "close", "adj_close", "volume",
                "dividends", "stock_splits")
FX_FIELDS = ("quote_close",)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values.astype(str).str[:10], errors="coerce", format="%Y-%m-%d")


def _validate_window(manifest: dict, label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RevisionContractError(f"{label}: missing source-window provenance")
    windows = set()
    for source in sources:
        if source.get("interval") != "1d" or source.get("auto_adjust") is not False or source.get("repair") is not False:
            raise RevisionContractError(f"{label}: incompatible Yahoo acquisition parameters")
        if not source.get("start") or not source.get("end_exclusive"):
            raise RevisionContractError(f"{label}: missing bounded source acquisition range")
        windows.add((source["start"], source["end_exclusive"]))
    if len(windows) != 1:
        raise RevisionContractError(f"{label}: mixed source windows within a batch")
    start, end = next(iter(windows))
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first >= last:
        raise RevisionContractError(f"{label}: invalid acquisition date range")
    return first, last


def _normalized(frame: pd.DataFrame, *, key: list[str], columns: tuple[str, ...],
                label: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = set(key) | set(columns)
    if missing := required - set(frame.columns):
        raise RevisionContractError(f"{label}: missing fields {sorted(missing)}")
    data = frame[[*key, *columns]].copy()
    data["session_date"] = _dates(data["session_date"])
    if data["session_date"].isna().any() or data[key[0]].isna().any():
        raise RevisionContractError(f"{label}: invalid source key")
    if data.duplicated(key).any():
        raise RevisionContractError(f"{label}: duplicate source key")
    if ((data["session_date"] < start) | (data["session_date"] >= end)).any():
        raise RevisionContractError(f"{label}: observation outside declared batch range")
    if not len(data):
        raise RevisionContractError(f"{label}: no source observations")
    for field in columns:
        data[field] = pd.to_numeric(data[field], errors="coerce")
    return data.sort_values(key, kind="stable").reset_index(drop=True)


def compare_table(
    old: pd.DataFrame, new: pd.DataFrame, *,
    key: list[str], columns: tuple[str, ...], label: str,
    rtol: float = 1e-8, atol: float = 1e-10,
) -> tuple[pd.DataFrame, dict]:
    both = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"),
                     indicator=True, validate="one_to_one").sort_values(key)
    details = []
    counts = {"shared": int(both["_merge"].eq("both").sum()),
              "old_only": int(both["_merge"].eq("left_only").sum()),
              "new_only": int(both["_merge"].eq("right_only").sum()),
              "changed_keys": 0, "changed_fields": 0}
    for row in both.itertuples(index=False, name=None):
        data = dict(zip(both.columns, row))
        presence = data["_merge"]
        if presence != "both":
            details.append({**{k: data[k] for k in key}, "table": label,
                            "change_type": str(presence), "field": "",
                            "old_value": None, "new_value": None,
                            "relative_change": None})
            continue
        revised = False
        for col in columns:
            x, y = data[f"{col}_old"], data[f"{col}_new"]
            x_na, y_na = pd.isna(x), pd.isna(y)
            same = (x_na and y_na) or (
                not x_na and not y_na and
                np.isclose(float(x), float(y), rtol=rtol, atol=atol)
            )
            if same:
                continue
            revised = True
            relative = ((float(y) / float(x) - 1) if
                        not x_na and not y_na and float(x) != 0 else np.nan)
            details.append({
                **{k: data[k] for k in key}, "table": label,
                "change_type": "revised", "field": col,
                "old_value": None if x_na else float(x),
                "new_value": None if y_na else float(y),
                "relative_change": relative,
            })
            counts["changed_fields"] += 1
        if revised:
            counts["changed_keys"] += 1
    result = pd.DataFrame(details, columns=[*key, "table", "change_type",
                                             "field", "old_value", "new_value",
                                             "relative_change"])
    return result, counts


def classify_revisions(changes: pd.DataFrame, *, snapshot_asof: str,
                       price_relative_tolerance: float = 1e-5,
                       provisional_volume_days: int = 3,
                       provisional_volume_tolerance: float = 0.05) -> pd.DataFrame:
    """Classify, never delete, source revisions. Provisional volume is not final."""
    if not 0 < price_relative_tolerance < 1 or not 0 < provisional_volume_tolerance < 1:
        raise RevisionContractError("Materiality tolerances must be in (0,1)")
    if provisional_volume_days < 0:
        raise RevisionContractError("Invalid provisional volume window")
    asof = pd.Timestamp(snapshot_asof).tz_localize(None) if pd.Timestamp(snapshot_asof).tzinfo is None else pd.Timestamp(snapshot_asof).tz_convert(None)
    result = changes.copy()
    labels = []
    for row in result.itertuples(index=False):
        if row.change_type != "revised":
            label = "requires_review"
        elif row.field == "volume":
            age = (asof.normalize() - pd.Timestamp(row.session_date)).days
            label = ("provisional_volume" if 0 <= age <= provisional_volume_days
                     and pd.notna(row.relative_change)
                     and abs(float(row.relative_change)) <= provisional_volume_tolerance
                     else "requires_review")
        elif row.field in {"open", "high", "low", "close", "adj_close", "quote_close"}:
            label = ("immaterial_numeric" if pd.notna(row.relative_change)
                     and abs(float(row.relative_change)) <= price_relative_tolerance
                     else "requires_review")
        else:
            label = "requires_review"
        labels.append(label)
    result["materiality"] = pd.Series(labels, index=result.index, dtype="object")
    return result


def compare_batches(*, old_dir: Path, new_dir: Path,
                    rtol: float = 1e-8, atol: float = 1e-10) -> tuple[pd.DataFrame, dict]:
    if rtol < 0 or atol < 0:
        raise RevisionContractError("Numeric tolerances must be nonnegative")
    if old_dir.resolve() == new_dir.resolve():
        raise RevisionContractError("Two DISTINCT snapshots are required")
    old_bars, old_fx, old_manifest = _source_inputs(old_dir)
    new_bars, new_fx, new_manifest = _source_inputs(new_dir)
    if old_manifest["batch_id"] == new_manifest["batch_id"]:
        raise RevisionContractError("Batch IDs must be distinct")
    a0, a1 = _validate_window(old_manifest, "old")
    b0, b1 = _validate_window(new_manifest, "new")
    start, end = max(a0, b0), min(a1, b1)
    if start >= end:
        raise RevisionContractError("Snapshots have no common acquisition date window")
    comparisons = [
        ("ohlcv", old_bars, new_bars, ["asset_id", "session_date"], PRICE_FIELDS),
        ("fx", old_fx, new_fx, ["provider_symbol", "session_date"], FX_FIELDS),
    ]
    results, summaries = [], {}
    for name, first, second, key, fields in comparisons:
        first = _normalized(first, key=key, columns=fields, label=f"old {name}",
                            start=a0, end=a1)
        second = _normalized(second, key=key, columns=fields, label=f"new {name}",
                             start=b0, end=b1)
        ids_old = set(first[key[0]].astype(str))
        ids_new = set(second[key[0]].astype(str))
        if ids_old != ids_new:
            raise RevisionContractError(f"{name}: source asset/symbol universes differ")
        first = first.loc[first["session_date"].ge(start) & first["session_date"].lt(end)]
        second = second.loc[second["session_date"].ge(start) & second["session_date"].lt(end)]
        if first.empty or second.empty:
            raise RevisionContractError(f"{name}: no overlapping observations")
        changes, counts = compare_table(first, second, key=key, columns=fields,
                                        label=name, rtol=rtol, atol=atol)
        results.append(changes)
        summaries[name] = counts
    findings = pd.concat([x for x in results if not x.empty], ignore_index=True,
                         sort=False) if any(not x.empty for x in results) else results[0].copy()
    asof = new_manifest.get("created_at_utc")
    if not asof:
        raise RevisionContractError("New batch lacks acquisition timestamp")
    findings = classify_revisions(findings, snapshot_asof=asof)
    result = {
        "schema": "alpha_edge_bronze_source_revision_comparison_1",
        "old_batch_id": old_manifest["batch_id"],
        "new_batch_id": new_manifest["batch_id"],
        "old_bronze_manifest_sha256": _sha(old_dir / "manifest.json"),
        "new_bronze_manifest_sha256": _sha(new_dir / "manifest.json"),
        "common_window_start": start.date().isoformat(),
        "common_window_end_exclusive": end.date().isoformat(),
        "comparisons": summaries,
        "revision_rows": len(findings),
        "revision_materiality": {
            name: int(findings["materiality"].eq(name).sum())
            for name in ("immaterial_numeric", "provisional_volume", "requires_review")
        },
        "materiality_policy": {
            "price_relative_tolerance": 1e-5,
            "provisional_volume_days": 3,
            "provisional_volume_relative_tolerance": 0.05,
            "classification_only": True,
        },
        "source_equivalence": (
            "observationally_equal_within_tolerances" if findings.empty
            else "differences_require_review"
        ),
        "acceptance": "review_required",
        "legacy_v1_used_as_reference": False,
        "s3_writes": 0,
        "limitations": ("Comparison reveals provider revisions but does not determine which "
                        "snapshot is correct. Only intersecting declared acquisition dates "
                        "are compared; missing observations inside that window are flagged. "
                        "No automatic repairs, adjusted-price overrides or production writes."),
    }
    return findings, result


def generate(*, old_dir: Path, new_dir: Path, output: Path,
             rtol: float = 1e-8, atol: float = 1e-10) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite revision audit: {output}")
    findings, report = compare_batches(old_dir=old_dir, new_dir=new_dir,
                                       rtol=rtol, atol=atol)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "source_revisions.csv"
    findings.to_csv(target, index=False)
    report["files_sha256"] = {"source_revisions.csv": _sha(target)}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-bronze-dir", type=Path, required=True)
    parser.add_argument("--new-bronze-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()
    print(json.dumps(generate(old_dir=args.old_bronze_dir,
                              new_dir=args.new_bronze_dir, output=args.output,
                              rtol=args.rtol, atol=args.atol), indent=2))


if __name__ == "__main__":
    main()
