"""Independent arithmetic and lineage checks for S3-only daily v2 candidates.

Pure DataFrame validation: complements the existing file-based independent
acceptance module, reusing its equality and date normalization helpers.
Never claims independent provider accuracy, corporate-action verification
or external evidence-gate completion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_edge.market.independent_acceptance_v2 import _equal, _dates


def review(tables: dict[str, pd.DataFrame], *, manual_corrections: int = 0) -> dict:
    gold = tables["gold"].copy()
    silver = tables["silver"].copy()
    source = tables["bronze_validated"].copy()
    issues = tables["session_issues"]
    flags = tables["session_flags"].copy()
    fx_evidence = tables["fx_evidence"]
    required = {
        "asset_id", "session_date", "close", "adj_close", "currency_original",
        "quality_status", "return_quality_status", "close_raw_usd",
        "close_adjusted_usd", "fx_to_usd", "ret_simple_close_adjusted_usd",
        "ret_log_close_adjusted_usd",
    }
    if missing := required - set(gold):
        raise ValueError(f"Daily Gold missing {sorted(missing)}")
    findings = []
    keys = ["asset_id", "session_date"]
    for frame, label in ((gold, "Gold"), (silver, "Silver"), (source, "source")):
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} duplicate asset/session")
        frame["session_date"] = _dates(frame["session_date"])
        if frame["session_date"].isna().any():
            raise ValueError(f"{label} invalid dates")
    for frame, label in ((silver, "Silver"), (source, "source")):
        if set(map(tuple, frame[keys].itertuples(index=False, name=None))) != set(
            map(tuple, gold[keys].itertuples(index=False, name=None))
        ):
            raise ValueError(f"{label} asset/date coverage differs from Gold")
    ordered = gold.sort_values(keys).reset_index(drop=True)
    aligned = source.merge(ordered, on=keys, validate="one_to_one",
                           suffixes=("_source", "_gold"))
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        if not _equal(aligned[f"{col}_source"], aligned[f"{col}_gold"] ).all():
            findings.append(f"source_lineage_{col}")
    known = ordered["quality_status"].eq("accepted")
    fx = pd.to_numeric(ordered["fx_to_usd"], errors="coerce")
    close = pd.to_numeric(ordered["close"], errors="coerce")
    adj = pd.to_numeric(ordered["adj_close"], errors="coerce")
    for col, expected in (
        ("close_raw_usd", close * fx),
        ("close_adjusted_usd", adj * fx),
    ):
        observed = pd.to_numeric(ordered[col], errors="coerce")
        if not _equal(observed.where(known), expected.where(known)).all():
            findings.append(f"usd_arithmetic_{col}")
    good_price = (known & np.isfinite(
        pd.to_numeric(ordered["close_adjusted_usd"], errors="coerce")
    ))
    safe = pd.to_numeric(ordered["close_adjusted_usd"], errors="coerce").where(good_price)
    previous = safe.groupby(ordered["asset_id"]).shift()
    expected_simple = safe.div(previous).sub(1).where(previous.gt(0))
    expected_log = np.log(safe.div(previous)).where(previous.gt(0))
    flags["session_date"] = _dates(flags["session_date"])
    if flags.duplicated(keys).any():
        raise ValueError("Calendar evidence duplicates asset/date")
    matched = ordered[keys].merge(
        flags[keys + ["previous_expected_session_missing"]],
        on=keys, how="left", validate="one_to_one",
    )
    if matched["previous_expected_session_missing"].isna().any():
        raise ValueError("Missing exchange calendar evidence")
    gaps = matched["previous_expected_session_missing"].astype(bool)
    expected_simple = expected_simple.mask(gaps)
    expected_log = expected_log.mask(gaps)
    for col, expected in (
        ("ret_simple_close_adjusted_usd", expected_simple),
        ("ret_log_close_adjusted_usd", expected_log),
    ):
        if not _equal(ordered[col], expected, atol=1e-10).all():
            findings.append(f"return_arithmetic_{col}")
    statuses = pd.Series(np.where(expected_simple.isna(), "unavailable", "accepted"))
    statuses.loc[expected_simple.abs().gt(0.5)] = "review"
    if statuses.ne(ordered["return_quality_status"].reset_index(drop=True)).any():
        findings.append("return_quality_status")
    blockers = {
        "integrity_findings": len(findings),
        "unverified_manual_corrections": manual_corrections,
        "quarantined_prices": int((~known).sum()),
        "review_returns": int(ordered["return_quality_status"].eq("review").sum()),
        "exchange_calendar_issues": len(issues),
        "fx_quote_rejections": int(fx_evidence["fx_validation_status"].ne("accepted").sum()),
    }
    return {
        "schema": "alpha_edge_daily_independent_arithmetic_v2_1",
        "internal_integrity_status": "passed" if not findings else "failed",
        "integrity_findings": findings,
        "acceptance": "blocked" if any(blockers.values()) else "review_required",
        "blocker_counts": blockers,
        "assets": int(ordered["asset_id"].nunique()),
        "gold_rows": len(ordered),
        "open_external_evidence_gates": [
            "independent_source_observation_verification",
            "corporate_action_adjustment_factor_validation",
            "historical_provider_revision_comparison",
            "exchange_specific_fx_observation_timing_policy",
            "endpoints_and_market_close_completeness",
        ],
        "production_eligible": False,
    }
