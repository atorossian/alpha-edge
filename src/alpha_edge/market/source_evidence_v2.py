"""Independent source endpoint, calendar and corporate action factor checks.

Always returns evidence, never certifies provider correctness or writes S3.
Daily Yahoo index timestamps are session labels, not verified close instants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_edge.market.independent_acceptance_v2 import _bronze
from alpha_edge.market.ingest_corporate_actions_v2 import load_verified_batch
from alpha_edge.market.preflight_universe_v2 import prepare_registry


class EvidenceError(ValueError):
    pass


def validate_source_coverage(
    *,
    bars: pd.DataFrame,
    universe: pd.DataFrame,
    start: str,
    end: str,
    tail_grace_sessions: int = 0,
) -> pd.DataFrame:
    """Expected exchange trading sessions including the bounded request edges.

    Leading/trailing missing sessions are NOT silently suppressed.
    """
    if tail_grace_sessions < 0:
        raise EvidenceError("Negative grace period")
    import pandas_market_calendars as mcal

    requested_start, requested_end = pd.Timestamp(start), pd.Timestamp(end)
    if requested_start >= requested_end:
        raise EvidenceError("Invalid request window")
    issues = []
    for r in universe.itertuples(index=False):
        source = bars.loc[bars["asset_id"].astype(str).eq(r.asset_id)]
        observed = pd.to_datetime(source["session_date"].astype(str).str[:10], errors="coerce")
        if observed.isna().any() or observed.duplicated().any():
            raise EvidenceError(f"{r.asset_id}: invalid or duplicate sessions")
        cal = mcal.get_calendar(r.calendar_id)
        expected = (
            pd.DatetimeIndex(cal.valid_days(start_date=requested_start, end_date=requested_end - pd.Timedelta(days=1)))
            .tz_localize(None)
            .normalize()
        )
        observed_set = set(observed)
        missing = sorted(set(expected) - observed_set)
        unexpected = sorted(observed_set - set(expected))
        for session in missing:
            issue = (
                "tail_grace"
                if tail_grace_sessions and session in set(expected[-tail_grace_sessions:])
                else "missing_expected_session"
            )
            issues.append(
                {
                    "asset_id": r.asset_id,
                    "provider_symbol": r.yahoo_ticker,
                    "session_date": session.date().isoformat(),
                    "issue": issue,
                }
            )
        for session in unexpected:
            issues.append(
                {
                    "asset_id": r.asset_id,
                    "provider_symbol": r.yahoo_ticker,
                    "session_date": session.date().isoformat(),
                    "issue": "unexpected_session",
                }
            )
    return pd.DataFrame(issues, columns=["asset_id", "provider_symbol", "session_date", "issue"])


def validate_adjustment_factors(
    *,
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    tolerance: float = 0.002,
) -> pd.DataFrame:
    """Compare implied ex-dividend adjustment jumps with cash dividends.

    This is an *approximate diagnostic*, not a substitute for a verified
    independent provider/exchange corporate-action factor feed.
    """
    if not 0 < tolerance < 1:
        raise EvidenceError("Invalid dividend factor tolerance")
    required = {"asset_id", "session_date", "close", "adj_close"}
    if required - set(bars) or {"asset_id", "effective_date", "action_type", "value"} - set(actions):
        raise EvidenceError("Missing action or price evidence columns")
    grouped = bars.copy()
    grouped["session_date"] = pd.to_datetime(grouped["session_date"].astype(str).str[:10])
    grouped = grouped.sort_values(["asset_id", "session_date"])
    ratio = pd.to_numeric(grouped["adj_close"]) / pd.to_numeric(grouped["close"])
    grouped["ratio"] = ratio
    grouped["prior_ratio"] = ratio.groupby(grouped["asset_id"]).shift()
    grouped["prior_close"] = pd.to_numeric(grouped["close"]).groupby(grouped["asset_id"]).shift()
    findings = []
    for action in actions.itertuples(index=False):
        if action.action_type != "DIVIDEND":
            continue
        effective = pd.Timestamp(action.effective_date)
        rows = grouped.loc[grouped["asset_id"].eq(action.asset_id) & grouped["session_date"].eq(effective)]
        if len(rows) != 1:
            status, actual, implied = "missing_price_evidence", None, None
        else:
            row = rows.iloc[0]
            previous = row.prior_close
            if not np.isfinite(previous) or previous <= 0 or not np.isfinite(row.prior_ratio) or row.prior_ratio <= 0:
                status, actual, implied = "missing_prior_evidence", None, None
            else:
                cash = float(action.value)
                if cash < 0 or cash >= previous:
                    status, actual, implied = "invalid_dividend_amount", None, None
                else:
                    # Theoretical prior ratio vs ex-date ratio: R_t/R_(t-1)
                    # ~= 1 / (1 - D / P_(t-1)) for pure cash dividends.
                    implied = 1.0 / (1.0 - cash / previous)
                    actual = float(row.ratio / row.prior_ratio)
                    status = (
                        "approx_factor_consistent"
                        if abs(actual / implied - 1.0) <= tolerance
                        else "factor_requires_review"
                    )
        findings.append(
            {
                "asset_id": action.asset_id,
                "effective_date": effective.date().isoformat(),
                "source_dividend": float(action.value),
                "observed_ratio_step": actual,
                "approx_expected_ratio_step": implied,
                "status": status,
            }
        )
    return pd.DataFrame(
        findings,
        columns=[
            "asset_id",
            "effective_date",
            "source_dividend",
            "observed_ratio_step",
            "approx_expected_ratio_step",
            "status",
        ],
    )


def run(
    *,
    bronze_dir: Path,
    action_dir: Path,
    universe_csv: Path,
    fx_config_csv: Path,
    output: Path,
    tail_grace_sessions: int = 0,
) -> dict:
    if output.exists():
        raise FileExistsError("Refusing to overwrite evidence")
    bars, quotes, bronze_manifest = _bronze(bronze_dir)
    u, _ = prepare_registry(pd.read_csv(universe_csv), pd.read_csv(fx_config_csv))
    expected_assets = set(u["asset_id"])
    if set(bars["asset_id"].astype(str)) != expected_assets:
        raise EvidenceError("Bronze asset universe differs from supplied registry")
    windows = {(entry.get("start"), entry.get("end_exclusive")) for entry in bronze_manifest["sources"]}
    if len(windows) != 1:
        raise EvidenceError("Mixed source windows")
    start, end = next(iter(windows))
    if not start or not end:
        raise EvidenceError("Unbounded source window")
    events, actions_manifest = load_verified_batch(action_dir)
    if set(actions_manifest.get("asset_ids", [])) != expected_assets:
        raise EvidenceError("Corporate actions universe mismatch")
    coverage = validate_source_coverage(
        bars=bars,
        universe=u,
        start=start,
        end=end,
        tail_grace_sessions=tail_grace_sessions,
    )
    factors = validate_adjustment_factors(bars=bars, actions=events)
    unresolved_factors = int(factors["status"].ne("approx_factor_consistent").sum())
    unresolved_coverage = int(coverage["issue"].ne("tail_grace").sum())
    output.mkdir(parents=True, exist_ok=False)
    checksums = {}
    for name, frame in [("coverage_issues.csv", coverage), ("dividend_factor_evidence.csv", factors)]:
        path = output / name
        frame.to_csv(path, index=False)
        checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "schema": "alpha_edge_source_coverage_factor_evidence_v2_1",
        "bronze_manifest_sha256": hashlib.sha256((bronze_dir / "manifest.json").read_bytes()).hexdigest(),
        "corporate_action_manifest_sha256": hashlib.sha256((action_dir / "manifest.json").read_bytes()).hexdigest(),
        "coverage_issues": len(coverage),
        "unresolved_coverage": unresolved_coverage,
        "dividend_events": len(factors),
        "unresolved_approx_dividend_factors": unresolved_factors,
        "acceptance": "review_required",
        "source_independently_verified": False,
        "fx_close_timing_independently_verified": False,
        "s3_writes": 0,
        "files_sha256": checksums,
        "limitations": (
            "Dividend factors are approximations using prior native close, not an "
            "independently verified corporate-action adjustment convention. "
            "Trading calendar sessions do not prove bar finality, quote timestamp "
            "or rights-issue completeness. No production eligibility granted."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bronze-dir", type=Path, required=True)
    parser.add_argument("--actions-dir", type=Path, required=True)
    parser.add_argument("--universe-csv", type=Path, required=True)
    parser.add_argument("--fx-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-grace-sessions", type=int, default=0)
    a = parser.parse_args()
    print(
        json.dumps(
            run(
                bronze_dir=a.bronze_dir,
                action_dir=a.actions_dir,
                universe_csv=a.universe_csv,
                fx_config_csv=a.fx_csv,
                output=a.output,
                tail_grace_sessions=a.tail_grace_sessions,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
