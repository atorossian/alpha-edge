"""Conservative historical corporate-action quarantine; never changes prices.

Candidate sessions are not certified prices. Yahoo's adjustment ratio is a
consistency check, not independent evidence of a share basis.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal

import numpy as np
import pandas as pd

from alpha_edge.market.lakehouse_v3.storage import ROOT, digest, json_bytes, put_immutable, verified_bytes

REGISTRY_ROOT = f"{ROOT}/config/historical_quarantine_registries/v1"
SCHEMA = "alpha_edge_historical_quarantine_registry_v1"
POLICY = "strict_historical_quarantine_v1"


def normalize_declared_dividend(
    amount: Decimal, *, split_ratios: tuple[Decimal, ...], price_basis_factors: tuple[Decimal, ...]
) -> Decimal:
    """Normalize onto an explicitly verified later basis, not inferred Yahoo ratios.

    Split ratios are new shares / old shares. Rights price factors are separate
    multiplicative price coefficients, never share counts. Callers must verify
    which factors actually belong to the target price basis before using this.
    """
    values = (amount, *split_ratios, *price_basis_factors)
    if any(not isinstance(v, Decimal) or not v.is_finite() or v <= 0 for v in values):
        raise ValueError("Expected positive finite exact Decimal basis inputs")
    for ratio in split_ratios:
        amount /= ratio
    for factor in price_basis_factors:
        amount *= factor
    return amount


def load_registry(s3, bucket: str, version: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("Expected SHA-256 quarantine registry version")
    blob = verified_bytes(s3, bucket, {"key": f"{REGISTRY_ROOT}/{version}/registry.json", "sha256": version})
    doc = json.loads(blob)
    if digest(json_bytes(doc)) != version:
        raise ValueError("Non-canonical quarantine registry")
    validate_registry(doc)
    return doc


def validate_registry(doc: dict) -> None:
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or not isinstance(doc.get("entries"), list):
        raise ValueError("Invalid quarantine registry")
    seen = set()
    for entry in doc["entries"]:
        if not isinstance(entry, dict):
            raise TypeError("Expected quarantine entry object")
        for key in ("asset_id", "provider_symbol", "bronze_sha256", "effective_date", "reason", "evidence_url"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"Missing quarantine field {key}")
        day = pd.Timestamp(entry["effective_date"])
        if day.date().isoformat() != entry["effective_date"] or not re.fullmatch(
            r"[0-9a-f]{64}", entry["bronze_sha256"]
        ):
            raise ValueError("Invalid quarantine date or Bronze checksum")
        if not entry["evidence_url"].startswith("https://"):
            raise ValueError("Missing independent evidence URL")
        key = (entry["asset_id"], entry["effective_date"], entry["reason"])
        if key in seen:
            raise ValueError("Duplicate quarantine finding")
        seen.add(key)


def publish_registry(s3, bucket: str, doc: dict, *, execute: bool = False) -> dict:
    """Publish only the explicit content-addressed registry, never activate it."""
    validate_registry(doc)
    payload = json_bytes(doc)
    version = digest(payload)
    key = f"{REGISTRY_ROOT}/{version}/registry.json"
    if execute:
        put_immutable(s3, bucket, key, payload, content_type="application/json", scope=REGISTRY_ROOT)
    return {"version": version, "key": key, "entries": len(doc["entries"]), "executed": execute}


def assess(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    asset_id: str,
    provider_symbol: str,
    bronze_sha256: str,
    registry: dict | None = None,
    duplicate_findings: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Block the prefix through every unresolved backward-adjusting event.

    A bad raw close blocks its session and any adjacent return. All remaining
    sessions/returns are merely candidates for subsequent financial gates.
    """
    frame = bars.sort_values("session_date").reset_index(drop=True)
    dates = pd.to_datetime(frame["session_date"], errors="raise")
    if (
        frame.empty
        or dates.duplicated().any()
        or not frame["asset_id"].eq(asset_id).all()
        or not frame["provider_symbol"].eq(provider_symbol).all()
    ):
        raise ValueError("Expected unique sessions for one pinned asset")
    close = pd.to_numeric(frame["close"], errors="coerce")
    adj = pd.to_numeric(frame["adj_close"], errors="coerce")
    raw_valid = np.isfinite(close) & close.gt(0)
    adj_valid = np.isfinite(adj) & adj.gt(0)
    ratio = adj.where(adj_valid) / close.where(raw_valid)
    observed = ratio.shift() / ratio
    findings = []

    def flag(day, reason, scope="prefix"):
        findings.append({"effective_date": day.date().isoformat(), "reason": reason, "scope": scope})

    events = {}
    for event in actions.to_dict("records"):
        events.setdefault(event["effective_date"], []).append(event)
    date_text = dates.dt.strftime("%Y-%m-%d")
    review = (
        ~raw_valid | ~adj_valid | date_text.isin(events) | (np.isfinite(observed) & observed.sub(1).abs().gt(0.0001))
    )
    for i in np.flatnonzero(review):
        day = dates.iloc[i]
        if not raw_valid.iloc[i]:
            flag(day, "invalid_raw_close", "session")
        if not adj_valid.iloc[i]:
            flag(day, "invalid_adjusted_close")
        day_events = events.get(day.date().isoformat(), [])
        split = any(e["action_type"] == "SPLIT" for e in day_events)
        dividends = [e for e in day_events if e["action_type"] == "DIVIDEND"]
        if split:
            flag(day, "split_price_basis_unverified")
        if dividends:
            event = dividends[0]
            amount = event.get("normalized_value", event["source_value"])
            if event.get("currency_correction_status") == "proposed":
                flag(day, "dividend_currency_correction_unapproved")
            if split or i == 0 or not raw_valid.iloc[i - 1] or not np.isfinite(amount) or amount <= 0:
                flag(day, "dividend_factor_not_evaluable")
                continue
            expected = 1 - amount / close.iloc[i - 1]
            if expected <= 0 or not np.isfinite(observed.iloc[i]):
                flag(day, "dividend_factor_not_evaluable")
            elif abs(observed.iloc[i] - expected) > 0.0001:
                flag(day, "dividend_factor_mismatch")
        elif not split and i > 0 and np.isfinite(observed.iloc[i]) and abs(observed.iloc[i] - 1) > 0.0001:
            flag(day, "unexplained_adjustment_factor_change")
    if registry is not None:
        validate_registry(registry)
        for entry in registry["entries"]:
            if entry["asset_id"] != asset_id:
                continue
            if entry["provider_symbol"] != provider_symbol or entry["bronze_sha256"] != bronze_sha256:
                raise ValueError("Quarantine registry source identity or checksum changed")
            day = pd.Timestamp(entry["effective_date"])
            if not dates.iloc[0] <= day <= dates.iloc[-1]:
                raise ValueError("Quarantine finding outside pinned history")
            flag(day, entry["reason"])
    if duplicate_findings is not None:
        source_dates = set(dates)
        for finding in duplicate_findings.loc[duplicate_findings["reason"].eq("nearby_equal_dividends")].to_dict(
            "records"
        ):
            if finding.get("review_status") == "verified_legitimate_distributions":
                continue
            day = pd.Timestamp(finding["effective_date"])
            if day not in source_dates:
                raise ValueError("Duplicate finding outside pinned sessions")
            flag(day, "nearby_equal_dividends_require_review")
    table = pd.DataFrame(findings, columns=["effective_date", "reason", "scope"])
    prefix = table.loc[table["scope"].eq("prefix"), "effective_date"]
    boundary = max(prefix) if not prefix.empty else None
    allowed = raw_valid & adj_valid
    if boundary is not None:
        allowed &= dates.gt(pd.Timestamp(boundary))
    sessions = pd.DataFrame(
        {
            "session_date": dates.dt.strftime("%Y-%m-%d"),
            "candidate_session": allowed,
            "candidate_return": allowed & allowed.shift(fill_value=False),
            "production_eligible": False,
        }
    )
    return (
        sessions,
        table,
        {
            "policy": POLICY,
            "quarantine_through": boundary,
            "quarantined_sessions": int((~allowed).sum()),
            "candidate_sessions": int(allowed.sum()),
            "candidate_returns": int(sessions["candidate_return"].sum()),
            "findings": len(table),
            "independently_verified": False,
            "production_eligible": False,
        },
    )
