"""Checksum-pinned price-basis approvals and candidate-only quarantine impact."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict

import pandas as pd

from alpha_edge.market.lakehouse_v3.storage import ROOT, verified_bytes

STANDARD = "split_basis_approval_registries"
DETERMINISTIC = "deterministic_corporate_action_approval_registries"


def load_approved_basis(s3, bucket, standard_versions, deterministic_versions, expected_count=None):
    """Validate immutable approvals; never reinterpret an approval as economic clearance."""
    by_asset = defaultdict(dict)
    ids = set()
    for family, versions in ((STANDARD, standard_versions), (DETERMINISTIC, deterministic_versions)):
        for version in versions:
            if not re.fullmatch(r"[0-9a-f]{64}", version):
                raise ValueError("Expected approval digest")
            ref = {"key": f"{ROOT}/config/{family}/v1/objects/{version}.json", "sha256": version}
            approval = json.loads(verified_bytes(s3,bucket,ref))
            expected_schema = ("alpha_edge_split_basis_approval_v1" if family == STANDARD
                               else "alpha_edge_deterministic_corporate_action_approval_v1")
            if (approval["schema"] != expected_schema or approval["decision"] != "approved_price_basis_only"
                    or approval["production_eligible"] is not False or approval["quarantine_released"] is not False):
                raise ValueError("Invalid price-basis-only approval")
            registry = json.loads(verified_bytes(s3,bucket,approval["candidate_registry"]))
            schema = ("alpha_edge_split_basis_candidate_registry_v1" if family == STANDARD
                      else "alpha_edge_deterministic_corporate_action_candidate_registry_v1")
            entries = registry["entries"]
            if (registry["schema"] != schema or registry["production_eligible"] is not False
                    or registry["quarantine_released"] is not False or len(entries) != approval["approved_count"]):
                raise ValueError("Invalid approved candidate registry")
            if family == STANDARD:
                from alpha_edge.market.approve_split_basis_v3 import approve
                from alpha_edge.market.audit_matching_split_basis_v3 import (
                    POLICY, FRACTIONAL_POLICY, SMALL_POLICY, IRREGULAR_POLICY, YAHOO_ONLY_POLICY,
                )
                modes = {POLICY: {}, FRACTIONAL_POLICY: {"fractional":True}, SMALL_POLICY: {"small_fraction":True},
                         IRREGULAR_POLICY: {"irregular":True}, YAHOO_ONLY_POLICY: {"yahoo_only":True}}
                check = approve(s3,bucket,approval["candidate_registry"]["sha256"],
                                expected_count=len(entries),**modes[registry["policy"]])
                if check["approval_version"] != version:
                    raise ValueError("Approved registry policy mismatch")
            for e in entries:
                key=(e["asset_id"],e["date"])
                if key in ids:
                    raise ValueError("Duplicate approved split identity")
                ids.add(key)
                if family == DETERMINISTIC:
                    from alpha_edge.market.reconcile_corporate_actions_v3 import PASS
                    chosen=e["chosen_hypothesis"]
                    if (e["status"] != "reconciled_price_basis_candidate" or chosen["status"] not in PASS
                            or chosen["date_method"] != "reported_dates" or e["production_eligible"] is not False
                            or e["quarantine_released"] is not False):
                        raise ValueError("Unexpected deterministic approval entry")
                    sha,symbol=e["bronze_sha256"],e["symbol"]
                    review_days=[c["date"] for c in chosen.get("daily_checks",[]) if c.get("large_return_review_required")]
                else:
                    sha,symbol=e["yahoo_sha256"],e["yahoo_symbol"]
                    review_days=[]
                ratio=float(e["ratio"])
                if not math.isfinite(ratio) or ratio<=0 or ratio==1:
                    raise ValueError("Invalid approved ratio")
                by_asset[key[0]][key[1]]={"ratio":ratio,"bronze_sha256":sha,"symbol":symbol,
                                          "approval_version":version,"large_return_review_dates":review_days}
    if expected_count is not None and len(ids)!=expected_count:
        raise ValueError("Unexpected approved population")
    return dict(by_asset)


def annotate_candidate(frame, raw, findings, approvals, *, asset_id, symbol, sha):
    """Remove only approved split blockers in a parallel hypothetical eligibility view.

    Preserve original historical quarantine, all other blockers, and production flags.
    """
    days=set()
    review=set()
    for day,entry in approvals.items():
        if (entry["bronze_sha256"]!=sha or entry["symbol"]!=symbol or day not in raw.index
                or not math.isclose(float(raw.at[day,"stock_splits"]),entry["ratio"],rel_tol=1e-7)
                or not ((findings.effective_date.eq(day)) & findings.reason.eq("split_price_basis_unverified")).any()):
            raise ValueError(f"Approved split source mismatch: {asset_id} {day}")
        days.add(day)
        review.update(entry["large_return_review_dates"])
    prefix=findings.loc[findings.scope.eq("prefix")]
    remaining=prefix.loc[~(prefix.reason.eq("split_price_basis_unverified") & prefix.effective_date.isin(days))]
    boundary=remaining.effective_date.max() if not remaining.empty else None
    potential=pd.Series(True,index=frame.index)
    if boundary is not None:
        potential &= frame.index > boundary
    session_blockers=set(findings.loc[findings.scope.eq("session"),"effective_date"])
    potential &= ~frame.index.isin(session_blockers)
    expected=frame.candidate_expected_session.eq(True)
    session=potential & expected
    previous=session.loc[expected].shift(fill_value=False).reindex(frame.index,fill_value=False)
    frame["candidate_split_basis_approved"]=frame.index.isin(days)
    frame["candidate_split_basis_economic_review_required"]=frame.index.isin(days)
    frame["candidate_split_basis_approval_version"]=[approvals[d]["approval_version"] if d in days else "" for d in frame.index]
    frame["candidate_split_basis_large_return_review"]=frame.index.isin(review)
    frame["candidate_split_basis_remaining_prefix_quarantine"]=frame.index<=boundary if boundary is not None else False
    frame["candidate_return_potential_after_split_basis_review"]=(session & previous & frame.candidate_total_return.notna())
    baseline=session & ~frame.historical_quarantine
    baseline_previous=baseline.loc[expected].shift(fill_value=False).reindex(frame.index,fill_value=False)
    frame["candidate_return_potential_before_split_basis_review"]=(baseline & baseline_previous & frame.candidate_total_return.notna())
    return len(days)
