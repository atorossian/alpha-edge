"""Read-only historical reconciliation: AEA-87 extracts versus Market Lake v2.

Legacy inputs must be explicitly supplied as local CSV/Parquet extracts; the
tool never queries or modifies active S3 datasets. Comparison never assumes
that legacy 'ret_close_adjusted_usd' is arithmetic: callers must assert its
semantics or returns are marked NOT_COMPARABLE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_edge.market.publish_lake_v2 import validate_local_candidate

PRICE_COL = "close_adjusted_usd"
V2_RETURN_COL = "ret_simple_close_adjusted_usd"
V2_LOG_COL = "ret_log_close_adjusted_usd"
LEGACY_RETURN_COL = "ret_close_adjusted_usd"


class ReconciliationError(ValueError):
    pass


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ReconciliationError("Only CSV and Parquet inputs are accepted")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(df: pd.DataFrame, *, required: set[str], label: str) -> pd.DataFrame:
    if missing := required - set(df.columns):
        raise ReconciliationError(f"{label} missing required fields: {sorted(missing)}")
    result = df.copy()
    # Legacy 'date' is a session-date LABEL, never a UTC instant.
    if "session_date" not in result:
        if "date" not in result:
            raise ReconciliationError(f"{label} missing date/session_date")
        result = result.rename(columns={"date": "session_date"})
    result["session_date"] = pd.to_datetime(
        result["session_date"].astype(str).str[:10], format="%Y-%m-%d", errors="coerce"
    )
    if result["session_date"].isna().any():
        raise ReconciliationError(f"{label} has invalid session dates")
    result["asset_id"] = result["asset_id"].astype(str).str.strip()
    if result["asset_id"].eq("").any():
        raise ReconciliationError(f"{label} has empty asset IDs")
    if result.duplicated(["asset_id", "session_date"]).any():
        raise ReconciliationError(f"{label} has duplicate asset/session observations")
    return result


def reconcile(
    candidate_dir: Path, legacy_prices: Path, legacy_returns: Path,
    *, return_kind: str = "unknown",
    price_atol: float = 1e-8, price_rtol: float = 1e-6,
    return_atol: float = 1e-9, return_rtol: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (full outer join detail, asset summary, reproducible manifest)."""
    if return_kind not in {"simple", "log", "unknown"}:
        raise ReconciliationError("legacy return kind must be simple, log, or unknown")
    if min(price_atol, price_rtol, return_atol, return_rtol) < 0:
        raise ReconciliationError("Negative comparison tolerance")
    manifest, _ = validate_local_candidate(candidate_dir)
    candidate = _canonical(
        pd.read_parquet(candidate_dir / "gold.parquet"),
        required={"asset_id", "session_date", PRICE_COL, V2_RETURN_COL, V2_LOG_COL,
                  "quality_status", "return_quality_status"},
        label="V2 Gold",
    )
    old_prices = _canonical(
        _read(legacy_prices), required={"asset_id", PRICE_COL}, label="AEA-87 prices"
    )
    old_returns = _canonical(
        _read(legacy_returns), required={"asset_id", LEGACY_RETURN_COL},
        label="AEA-87 returns",
    )
    keys = ["asset_id", "session_date"]
    price_table = old_prices[keys + [PRICE_COL]].rename(
        columns={PRICE_COL: "legacy_close_adjusted_usd"}
    )
    return_table = old_returns[keys + [LEGACY_RETURN_COL]].rename(
        columns={LEGACY_RETURN_COL: "legacy_return"}
    )
    old = price_table.merge(return_table, on=keys, how="outer", indicator="legacy_price_return_presence",
                            validate="one_to_one")
    v2_cols = keys + [PRICE_COL, V2_RETURN_COL, V2_LOG_COL,
                      "quality_status", "return_quality_status"]
    for optional in ("provider_symbol", "currency_original", "observation_origin",
                     "correction_id", "verification_status", "quality_reason"):
        if optional in candidate.columns:
            v2_cols.append(optional)
    new = candidate[v2_cols].rename(columns={
        PRICE_COL: "v2_close_adjusted_usd",
        V2_RETURN_COL: "v2_return_simple",
        V2_LOG_COL: "v2_return_log",
    })
    detail = old.merge(new, on=keys, how="outer", indicator="historical_presence",
                       validate="one_to_one").sort_values(keys).reset_index(drop=True)

    both = detail["historical_presence"].eq("both")
    lp = pd.to_numeric(detail["legacy_close_adjusted_usd"], errors="coerce")
    vp = pd.to_numeric(detail["v2_close_adjusted_usd"], errors="coerce")
    prices_finite = np.isfinite(lp) & np.isfinite(vp)
    detail["price_diff_usd"] = vp - lp
    detail["price_diff_abs_usd"] = detail["price_diff_usd"].abs()
    detail["price_diff_relative"] = detail["price_diff_abs_usd"] / lp.abs().where(lp.ne(0))
    close = pd.Series(
        np.isclose(lp, vp, atol=price_atol, rtol=price_rtol, equal_nan=False),
        index=detail.index,
    )
    detail["price_comparison"] = np.select(
        [~both, both & ~prices_finite, both & close],
        ["not_shared_session", "unavailable_price", "match"], default="mismatch"
    )

    lr = pd.to_numeric(detail["legacy_return"], errors="coerce")
    column = "v2_return_simple" if return_kind == "simple" else "v2_return_log"
    vr = pd.to_numeric(detail[column], errors="coerce")
    returns_finite = np.isfinite(lr) & np.isfinite(vr)
    close_ret = pd.Series(
        np.isclose(lr, vr, atol=return_atol, rtol=return_rtol, equal_nan=False),
        index=detail.index,
    )
    if return_kind == "unknown":
        detail["return_comparison"] = "not_comparable_semantics_unknown"
        detail["return_diff"] = np.nan
    else:
        detail["return_diff"] = vr - lr
        detail["return_comparison"] = np.select(
            [~both, both & ~returns_finite, both & close_ret],
            ["not_shared_session", "unavailable_return", "match"],
            default="mismatch"
        )

    detail["review_reason"] = ""
    def add_reason(mask: pd.Series, reason: str) -> None:
        detail.loc[mask, "review_reason"] += reason + ";"
    add_reason(detail["historical_presence"].eq("left_only"), "aea87_only_session")
    add_reason(detail["historical_presence"].eq("right_only"), "v2_only_session")
    add_reason(detail["legacy_price_return_presence"].eq("left_only").fillna(False),
               "aea87_price_without_return")
    add_reason(detail["legacy_price_return_presence"].eq("right_only").fillna(False),
               "aea87_return_without_price")
    add_reason(detail["price_comparison"].eq("mismatch"), "price_mismatch")
    add_reason(detail["price_comparison"].eq("unavailable_price"), "price_unavailable")
    add_reason(detail["return_comparison"].eq("mismatch"), "return_mismatch")
    add_reason(detail["return_comparison"].eq("unavailable_return"), "return_unavailable")
    if "observation_origin" in detail:
        add_reason(detail["observation_origin"].eq("user_supplied_manual").fillna(False),
                   "manual_correction")
    if "quality_status" in detail:
        add_reason(detail["quality_status"].eq("quarantined").fillna(False),
                   "v2_quarantined")
    detail["review_reason"] = detail["review_reason"].str.rstrip(";")
    summaries = []
    for asset_id, frame in detail.groupby("asset_id", sort=True):
        summaries.append({
            "asset_id": asset_id,
            "shared_sessions": int(frame["historical_presence"].eq("both").sum()),
            "aea87_only_sessions": int(frame["historical_presence"].eq("left_only").sum()),
            "v2_only_sessions": int(frame["historical_presence"].eq("right_only").sum()),
            "price_matches": int(frame["price_comparison"].eq("match").sum()),
            "price_mismatches": int(frame["price_comparison"].eq("mismatch").sum()),
            "price_unavailable": int(frame["price_comparison"].eq("unavailable_price").sum()),
            "return_matches": int(frame["return_comparison"].eq("match").sum()),
            "return_mismatches": int(frame["return_comparison"].eq("mismatch").sum()),
            "review_rows": int(frame["review_reason"].ne("").sum()),
        })
    summary = pd.DataFrame(summaries)
    source = {
        "schema": "alpha_edge_aea87_lake_v2_reconciliation_1",
        "candidate_manifest_sha256": _sha(candidate_dir / "manifest.json"),
        "legacy_prices_sha256": _sha(legacy_prices),
        "legacy_returns_sha256": _sha(legacy_returns),
        "legacy_return_kind": return_kind,
        "tolerance": {
            "price_atol": price_atol, "price_rtol": price_rtol,
            "return_atol": return_atol, "return_rtol": return_rtol,
        },
        "legacy_price_rows": len(old_prices),
        "legacy_return_rows": len(old_returns),
        "v2_gold_rows": len(candidate),
        "joined_rows": len(detail),
        "assets": len(summary),
        "comparison_status": "review_required" if (
            detail["review_reason"].ne("").any() or return_kind == "unknown"
            or manifest.get("acceptance") != "initial_contract_pass"
        ) else "comparison_pass",
        "note": "Historical equivalence is not production approval. Missing/ambiguous legacy "
                "return conventions must be explicitly resolved. Session-date aliases are "
                "interpreted as exchange date labels, not UTC instants.",
    }
    return detail, summary, source


def generate(
    candidate_dir: Path, legacy_prices: Path, legacy_returns: Path, *,
    output: Path, return_kind: str = "unknown",
) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing reconciliation: {output}")
    detail, summary, manifest = reconcile(
        candidate_dir, legacy_prices, legacy_returns, return_kind=return_kind
    )
    output.mkdir(parents=True)
    detail_path = output / "session_differences.csv"
    summary_path = output / "asset_reconciliation_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest["files_sha256"] = {
        detail_path.name: _sha(detail_path),
        summary_path.name: _sha(summary_path),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--legacy-prices", type=Path, required=True)
    parser.add_argument("--legacy-returns", type=Path, required=True)
    parser.add_argument("--legacy-return-kind", choices=["unknown", "simple", "log"],
                        default="unknown")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(
        args.candidate_dir, args.legacy_prices, args.legacy_returns,
        output=args.output, return_kind=args.legacy_return_kind,
    ), indent=2))


if __name__ == "__main__":
    main()
