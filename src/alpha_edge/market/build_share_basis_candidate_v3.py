"""Explicit, source-pinned missing-share-factor reconstruction (candidate only)."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation

import boto3
import numpy as np
import pandas as pd

from alpha_edge.market.lakehouse_v3.storage import (
    ROOT,
    check_id,
    digest,
    json_bytes,
    parquet_bytes,
    put_immutable,
    verified_bytes,
    verified_frame,
)

REGISTRY_ROOT = f"{ROOT}/config/share_basis_candidate_registries/v1"
OUTPUT_ROOT = f"{ROOT}/silver/share_basis_candidates/v1"
SCHEMA = "alpha_edge_share_basis_candidate_registry_v1"


def _date(value):
    if not isinstance(value, str) or pd.Timestamp(value).date().isoformat() != value:
        raise ValueError("Expected exact ISO session date")
    return value


def _decimal(value):
    if not isinstance(value, str):
        raise TypeError("Expected exact decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal") from exc
    if not result.is_finite():
        raise ValueError("Non-finite decimal")
    return result


def validate_registry(doc):
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or doc.get("status") != "candidate_only":
        raise ValueError("Expected candidate-only share basis registry")
    for k in ("asset_id", "provider_symbol", "source_key", "source_sha256", "evidence_key", "evidence_sha256"):
        if not isinstance(doc.get(k), str) or not doc[k]:
            raise ValueError(f"Missing registry field {k}")
    for k in ("source_sha256", "evidence_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", doc[k]):
            raise ValueError("Expected SHA-256 input")
    if not doc["source_key"].startswith(f"{ROOT}/bronze/") or not doc["evidence_key"].startswith(f"{ROOT}/config/"):
        raise ValueError("Expected pinned Bronze and configuration evidence objects")
    scope = _date(doc["scope_start"])
    events = doc.get("events")
    blocked = doc.get("unresolved_cash_dates")
    if not isinstance(events, list) or not events or not isinstance(blocked, list):
        raise ValueError("Expected explicit events and unresolved cash dates")
    dates = set()
    for e in events:
        day = _date(e["effective_date"])
        if day <= scope or day in dates or _decimal(e["ratio"]) <= 0 or _decimal(e["ratio"]) == 1:
            raise ValueError("Invalid or repeated missing share-factor event")
        if _decimal(e["source_split"]) != 0 or _decimal(e["source_cash"]) < 0:
            raise ValueError("Only a missing source split can be reconstructed")
        if e.get("date_status") not in {"candidate", "issuer_verified"}:
            raise ValueError("Explicit effective-date evidence status required")
        if not isinstance(e.get("evidence_url"), str) or not e["evidence_url"].startswith("https://"):
            raise ValueError("Independent evidence URL required")
        dates.add(day)
    if len(set(blocked)) != len(blocked) or any(_date(d) < scope for d in blocked):
        raise ValueError("Invalid unresolved cash dates")
    cash_days = set()
    for e in doc.get("cash_reconciliations", []):
        day = _date(e["effective_date"])
        if day < scope or day in cash_days or day in blocked or day not in dates:
            raise ValueError("Conflicting cash reconciliation date")
        if _decimal(e["source_cash"]) <= 0 or _decimal(e["declared_cash"]) <= 0:
            raise ValueError("Expected positive source and declared cash")
        if e.get("entitlement_basis") != "pre_share_event" or e.get("status") != "candidate":
            raise ValueError("Expected explicit candidate pre-share entitlement basis")
        if not isinstance(e.get("currency"), str) or not re.fullmatch(r"[A-Z]{3}", e["currency"]):
            raise ValueError("Expected currency code")
        if (
            not isinstance(e.get("evidence_urls"), list)
            or len(e["evidence_urls"]) < 2
            or not all(isinstance(url, str) and url.startswith("https://") for url in e["evidence_urls"])
            or not e.get("reason")
        ):
            raise ValueError("Cash reconciliation requires independent evidence and rationale")
        cash_days.add(day)
    return doc


def publish_registry(s3, bucket, doc, *, execute=False):
    payload = json_bytes(validate_registry(doc))
    version = digest(payload)
    key = f"{REGISTRY_ROOT}/{version}/registry.json"
    if execute:
        put_immutable(s3, bucket, key, payload, content_type="application/json", scope=REGISTRY_ROOT)
    return {"key": key, "version": version, "executed": execute}


def reconstruct(bars, doc, *, bronze_sha256):
    """Normalize only missing subsequent share factors, never provider adj_close.

    Other actions on the source basis are not independently certified. All
    returns are provisional; unresolved cash dates break the adjusted chain.
    """
    validate_registry(doc)
    if bronze_sha256 != doc["source_sha256"]:
        raise ValueError("Bronze checksum changed")
    if any(col.startswith("candidate_") for col in bars.columns):
        raise ValueError("Refusing to reconstruct an already transformed candidate")
    f = bars.sort_values("session_date").reset_index(drop=True).copy(deep=True)
    dates = f.session_date.astype(str)
    if (
        f.empty
        or dates.duplicated().any()
        or not f.asset_id.eq(doc["asset_id"]).all()
        or not f.provider_symbol.eq(doc["provider_symbol"]).all()
    ):
        raise ValueError("Expected unique sessions for pinned asset")
    if any(_date(d) != d for d in dates):
        raise ValueError("Invalid source dates")
    scope = dates.ge(doc["scope_start"])
    if not scope.any() or doc["scope_start"] not in set(dates):
        raise ValueError("Scope start missing from source")
    known = set(dates)
    blocked = set(doc["unresolved_cash_dates"])
    if not blocked <= known:
        raise ValueError("Unresolved cash date missing from source")
    missing = pd.Series(1.0, index=f.index)
    f["candidate_share_event"] = 0.0
    for e in sorted(doc["events"], key=lambda x: x["effective_date"]):
        mask = dates.eq(e["effective_date"])
        if mask.sum() != 1:
            raise ValueError("Missing event session")
        idx = f.index[mask][0]
        if Decimal(str(f.at[idx, "stock_splits"])) != _decimal(e["source_split"]) or Decimal(
            str(f.at[idx, "dividends"])
        ) != _decimal(e["source_cash"]):
            raise ValueError("Source action changed or already recorded")
        ratio = float(_decimal(e["ratio"]))
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("Share factor outside finite numeric range")
        missing.loc[dates.lt(e["effective_date"])] *= ratio
        f.at[idx, "candidate_share_event"] = ratio
    if not np.isfinite(missing).all() or not missing.gt(0).all():
        raise ValueError("Invalid cumulative share factor")
    f["missing_subsequent_share_factor"] = missing.where(scope)
    for col in ("open", "high", "low", "close"):
        raw = pd.to_numeric(f[col], errors="coerce")
        f[f"candidate_{col}"] = (raw / missing).where(scope & np.isfinite(raw) & raw.gt(0))
    cash = pd.to_numeric(f.dividends, errors="coerce")
    f["unresolved_cash"] = dates.isin(blocked)
    # Cash on a newly reconstructed split date requires an independent quote
    # basis decision; no automatic conversion of a possibly cum-share amount.
    f["unresolved_cash"] |= f.candidate_share_event.ne(0) & cash.ne(0)
    f["candidate_cash_dividend"] = (cash / missing).where(scope & ~f.unresolved_cash & np.isfinite(cash) & cash.ge(0))
    if doc.get("cash_reconciliations"):
        f["candidate_cash_reconciled"] = False
    # Original declarations are on the pre-event share basis. Reverse the
    # entire subsequent share chain (recorded AND missing), including today's
    # newly reconstructed share dividend, exactly once. Other cash events keep
    # the provider basis and the missing-only correction above.
    recorded = pd.to_numeric(f.stock_splits, errors="raise").replace(0, 1)
    if not np.isfinite(recorded).all() or not recorded.gt(0).all():
        raise ValueError("Invalid recorded share ratio")
    recorded_future = recorded.iloc[::-1].cumprod().iloc[::-1] / recorded
    for e in doc.get("cash_reconciliations", []):
        mask = dates.eq(e["effective_date"])
        if mask.sum() != 1:
            raise ValueError("Cash reconciliation session missing")
        i = f.index[mask][0]
        if (
            Decimal(str(f.at[i, "dividends"])) != _decimal(e["source_cash"])
            or f.at[i, "currency_original"] != e["currency"]
        ):
            raise ValueError("Cash reconciliation source value or currency changed")
        factor = recorded_future.iat[i] * missing.iat[i] * f.at[i, "candidate_share_event"]
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid complete declared-cash share basis")
        f.at[i, "candidate_cash_dividend"] = float(_decimal(e["declared_cash"])) / factor
        f.at[i, "candidate_cash_reconciled"] = True
        f.at[i, "unresolved_cash"] = False
    f["candidate_price_return"] = f.candidate_close / f.candidate_close.shift() - 1
    f["candidate_total_return"] = (f.candidate_close + f.candidate_cash_dividend) / f.candidate_close.shift() - 1
    f["candidate_adjusted_close"] = np.nan
    # Backward chain from latest close; never bridge a missing/unresolved return.
    if pd.notna(f.candidate_close.iat[-1]):
        f.loc[f.index[-1], "candidate_adjusted_close"] = f.candidate_close.iat[-1]
        for i in range(len(f) - 1, 0, -1):
            r = f.candidate_total_return.iat[i]
            if not np.isfinite(r) or r <= -1:
                break
            f.loc[i - 1, "candidate_adjusted_close"] = f.candidate_adjusted_close.iat[i] / (1 + r)
    f["production_eligible"] = False
    stats = {
        "rows": len(f),
        "scope_rows": int(scope.sum()),
        "missing_share_events": len(doc["events"]),
        "unresolved_cash_sessions": int((scope & f.unresolved_cash).sum()),
        "candidate_total_returns": int(f.candidate_total_return.notna().sum()),
        "candidate_adjusted_prices": int(f.candidate_adjusted_close.notna().sum()),
        "production_eligible": False,
    }
    if doc.get("cash_reconciliations"):
        stats["cash_reconciliations"] = int(f.candidate_cash_reconciled.sum())
    return f, stats


def build(s3, bucket, *, version, run_id, execute=False):
    check_id(run_id)
    if not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("Expected registry version checksum")
    registry_key = f"{REGISTRY_ROOT}/{version}/registry.json"
    raw = verified_bytes(s3, bucket, {"key": registry_key, "sha256": version})
    doc = validate_registry(json.loads(raw))
    if json_bytes(doc) != raw:
        raise ValueError("Noncanonical registry")
    verified_bytes(s3, bucket, {"key": doc["evidence_key"], "sha256": doc["evidence_sha256"]})
    source = verified_frame(s3, bucket, {"key": doc["source_key"], "sha256": doc["source_sha256"]})
    result, stats = reconstruct(source, doc, bronze_sha256=doc["source_sha256"])
    manifest = {
        "schema": "alpha_edge_share_basis_candidate_v1",
        "run_id": run_id,
        "registry_version": version,
        "source_key": doc["source_key"],
        "source_sha256": doc["source_sha256"],
        "scope_start": doc["scope_start"],
        "asset_id": doc["asset_id"],
        "statistics": stats,
        "production_eligible": False,
    }
    if not execute:
        return {**manifest, "status": "dry_run"}
    root = f"{OUTPUT_ROOT}/runs/{run_id}/assets/{digest(doc['asset_id'].encode())}"
    blob = parquet_bytes(result)
    key = f"{root}/objects/prices-{digest(blob)}.parquet"
    ref = put_immutable(s3, bucket, key, blob, content_type="application/vnd.apache.parquet", scope=OUTPUT_ROOT)
    manifest["output"] = {**ref, "rows": len(result)}
    manifest_key = f"{root}/manifest.json"
    put_immutable(s3, bucket, manifest_key, json_bytes(manifest), content_type="application/json", scope=OUTPUT_ROOT)
    return {**manifest, "manifest_key": manifest_key, "status": "candidate_written"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--registry-version", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                boto3.client("s3", region_name=args.region),
                args.bucket,
                version=args.registry_version,
                run_id=args.run_id,
                execute=args.execute,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
