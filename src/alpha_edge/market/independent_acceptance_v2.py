"""Independent, fail-closed Lake v2 pilot integrity and acceptance report.

No AEA-87/v1 input or comparison. Verifies immutable Bronze hashes, candidate
hashes, source-to-Gold lineage, arithmetic, FX quote mapping/as-of dates,
returns, and unresolved quality evidence. Produces local evidence only.

Passing arithmetic is NOT independent verification of Yahoo observations,
corporate actions, or closing-time semantics: publication remains blocked
until those explicit external evidence gates are satisfied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_edge.market.publish_lake_v2 import validate_local_candidate
from alpha_edge.market.ingest_corporate_actions_v2 import load_verified_batch
from alpha_edge.market.standardize_bronze_v2 import PILOT_FX_PAIRS


class AcceptanceError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str[:10], format="%Y-%m-%d", errors="coerce")


def _bronze(batch_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise AcceptanceError("Immutable Bronze manifest missing or unsafe")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("schema_version") != "bronze-v2-1":
        raise AcceptanceError("Bronze manifest schema mismatch")
    tables = []
    for name in ("source_ohlcv.parquet", "source_fx_quotes.parquet"):
        path = batch_dir / name
        if not path.is_file() or path.is_symlink() or m.get("files", {}).get(name, {}).get("sha256") != _sha(path):
            raise AcceptanceError(f"Bronze integrity failure: {name}")
        tables.append(pd.read_parquet(path))
    return tables[0], tables[1], m


def _equal(a: pd.Series, b: pd.Series, *, atol: float = 1e-9, rtol: float = 1e-8) -> np.ndarray:
    x = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    return np.isclose(x, y, atol=atol, rtol=rtol, equal_nan=True)


def _event(events: list[dict], *, check: str, frame: pd.DataFrame, mask, reason: str) -> None:
    bad = frame.loc[np.asarray(mask, dtype=bool)]
    for row in bad.itertuples(index=False):
        events.append({
            "check": check, "asset_id": str(row.asset_id),
            "session_date": str(pd.Timestamp(row.session_date).date()),
            "reason": reason,
        })


def validate(
    *, candidate_dir: Path, bronze_dir: Path, action_batch_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    manifest, _ = validate_local_candidate(candidate_dir)
    bars, quotes, bronze_manifest = _bronze(bronze_dir)
    if _sha(bronze_dir / "manifest.json") != manifest.get("bronze_manifest_sha256"):
        raise AcceptanceError("Candidate Bronze manifest digest differs from supplied source")
    if bronze_manifest.get("batch_id") != manifest.get("bronze_batch_id"):
        raise AcceptanceError("Candidate Bronze batch identity mismatch")
    silver = pd.read_parquet(candidate_dir / "silver.parquet")
    gold = pd.read_parquet(candidate_dir / "gold.parquet")
    fx = pd.read_parquet(candidate_dir / "fx_normalized.parquet")
    issues = pd.read_parquet(candidate_dir / "session_issues.parquet")
    evidence = pd.read_parquet(candidate_dir / "fx_evidence.parquet")
    required = {
        "asset_id", "session_date", "provider_symbol", "currency_original",
        "open", "high", "low", "close", "adj_close", "volume",
        "fx_date", "fx_to_usd", "quality_status", "quality_reason",
        "close_raw_usd", "close_adjusted_usd",
    }
    required_gold = required | {"ret_simple_close_adjusted_usd",
                                "ret_log_close_adjusted_usd", "return_quality_status"}
    if missing := required_gold - set(gold):
        raise AcceptanceError(f"Gold missing contract fields: {sorted(missing)}")
    if missing := required - set(silver):
        raise AcceptanceError(f"Silver missing contract fields: {sorted(missing)}")
    source_required = {"asset_id", "session_date", "provider_symbol", "currency_original",
                       "open", "high", "low", "close", "adj_close", "volume"}
    if missing := source_required - set(bars):
        raise AcceptanceError(f"Bronze missing contract fields: {sorted(missing)}")
    if missing := {"currency_original", "fx_date", "fx_to_usd", "quote_close",
                   "fx_quote_direction", "fx_quote_symbol"} - set(fx):
        raise AcceptanceError(f"Normalized FX missing fields: {sorted(missing)}")
    if missing := {"provider_symbol", "session_date", "quote_close"} - set(quotes):
        raise AcceptanceError(f"Raw FX missing fields: {sorted(missing)}")
    if missing := {"issue", "asset_id", "session_date"} - set(issues):
        raise AcceptanceError(f"Calendar issue evidence missing fields: {sorted(missing)}")
    if "fx_validation_status" not in evidence:
        raise AcceptanceError("FX validation evidence missing status")
    for df, label in ((gold, "Gold"), (silver, "Silver"), (bars, "Bronze")):
        if df.duplicated(["asset_id", "session_date"]).any():
            raise AcceptanceError(f"{label} duplicates source asset/session keys")
        if _dates(df["session_date"]).isna().any():
            raise AcceptanceError(f"{label} invalid session labels")
    for df in (gold, silver, bars):
        df["session_date"] = _dates(df["session_date"])
    gold = gold.sort_values(["asset_id", "session_date"], kind="stable").reset_index(drop=True)
    silver = silver.sort_values(["asset_id", "session_date"], kind="stable").reset_index(drop=True)
    bars = bars.sort_values(["asset_id", "session_date"], kind="stable").reset_index(drop=True)
    key = ["asset_id", "session_date"]
    if not gold[key].equals(silver[key]):
        raise AcceptanceError("Silver and Gold keys differ")
    if len(gold) != manifest["gold_rows"] or len(silver) != manifest["silver_rows"]:
        raise AcceptanceError("Candidate row counts differ from manifest")

    findings: list[dict] = []
    # Gold must be a lossless extension of Silver, excluding calculated returns.
    overlap = sorted(set(silver) & set(gold))
    for col in overlap:
        a, b = silver[col], gold[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            mismatch = ~_equal(a, b)
        else:
            mismatch = ~a.astype("string").fillna("<NULL>").eq(
                b.astype("string").fillna("<NULL>")
            ).to_numpy()
        _event(findings, check=f"silver_gold_{col}", frame=gold,
               mask=mismatch, reason="Gold differs from Silver source value")

    # Every original Yahoo observation must remain present, bit-for-bit as
    # interpreted numerically; manual additions may exist but not overwrite.
    source = bars.merge(gold, on=key, how="left", suffixes=("_bronze", "_gold"),
                        indicator=True, validate="one_to_one")
    if source["_merge"].ne("both").any():
        raise AcceptanceError("Gold missing original Bronze observations")
    for col in sorted(source_required - set(key)):
        a = source[f"{col}_bronze"]
        b = source[f"{col}_gold"]
        if col in {"open", "high", "low", "close", "adj_close", "volume"}:
            mismatch = ~_equal(a, b)
        else:
            mismatch = ~a.astype("string").fillna("<NULL>").eq(
                b.astype("string").fillna("<NULL>")
            ).to_numpy()
        _event(findings, check=f"bronze_lineage_{col}",
               frame=source.rename(columns={"asset_id": "asset_id",
                                            "session_date": "session_date"}),
               mask=mismatch, reason="Candidate differs from immutable Yahoo Bronze")

    # Local USD arithmetic is independently recomputed; unaccepted rows
    # must never expose a purported valid USD price.
    accepted = gold["quality_status"].eq("accepted").to_numpy()
    for name, native in (("close_raw_usd", "close"), ("close_adjusted_usd", "adj_close")):
        expected = pd.to_numeric(gold[native], errors="coerce") * pd.to_numeric(
            gold["fx_to_usd"], errors="coerce")
        observed = pd.to_numeric(gold[name], errors="coerce")
        mismatch = np.where(accepted, ~_equal(expected, observed),
                            observed.notna().to_numpy())
        _event(findings, check=f"usd_arithmetic_{name}", frame=gold, mask=mismatch,
               reason="USD value or rejected-row masking inconsistent with native price and FX")
    for asset_id, group in gold.groupby("asset_id"):
        currencies = group["currency_original"].dropna().unique()
        if len(currencies) != 1:
            raise AcceptanceError(f"Asset {asset_id} has multiple or missing currencies")

    # Re-derive provider FX quote for the exact FX date (no reliance on v1).
    fx_native = fx.copy()
    fx_native["fx_date"] = _dates(fx_native["fx_date"])
    raw_fx = quotes[["provider_symbol", "session_date", "quote_close"]].copy()
    raw_fx["session_date"] = _dates(raw_fx["session_date"])
    if raw_fx.duplicated(["provider_symbol", "session_date"]).any():
        raise AcceptanceError("Raw FX contains duplicate symbol/session keys")
    quoted = fx_native.merge(raw_fx, left_on=["fx_quote_symbol", "fx_date"],
                             right_on=["provider_symbol", "session_date"],
                             how="left", validate="many_to_one", suffixes=("", "_raw"))
    if quoted["provider_symbol"].isna().any():
        raise AcceptanceError("Normalized FX includes quote not found in immutable Bronze")
    quoted_rate = pd.to_numeric(quoted["quote_close_raw"], errors="coerce")
    normalized_rate = pd.to_numeric(quoted["fx_to_usd"], errors="coerce")
    expected_rate = np.where(quoted["fx_quote_direction"].eq("direct"),
                             quoted_rate, 1.0 / quoted_rate)
    invalid_direction = ~quoted["fx_quote_direction"].isin(["direct", "inverse"])
    if invalid_direction.any() or not _equal(pd.Series(expected_rate), normalized_rate).all():
        raise AcceptanceError("FX inversion/direct quote arithmetic inconsistency")
    for currency, (symbol, direction) in PILOT_FX_PAIRS.items():
        subset = quoted.loc[quoted["currency_original"].eq(currency)]
        if not subset.empty and (
            subset["fx_quote_symbol"].ne(symbol).any()
            or subset["fx_quote_direction"].ne(direction).any()
        ):
            raise AcceptanceError(f"Unexpected FX quote mapping for {currency}")
    if set(gold["currency_original"].astype(str)) - {"USD", *PILOT_FX_PAIRS}:
        raise AcceptanceError("Unconfigured original currency in pilot")
    rates = fx_native[["currency_original", "fx_date", "fx_to_usd"]].copy()
    if rates.duplicated(["currency_original", "fx_date"]).any():
        raise AcceptanceError("Duplicate normalized FX date")
    # Independently determine max available accepted quote date at or before
    # each session, rather than assuming a UTC label for an exchange close.
    expected_dates = []
    expected_rates = []
    for row in gold.itertuples(index=False):
        if row.currency_original == "USD":
            expected_dates.append(row.session_date)
            expected_rates.append(1.0)
            continue
        subset = rates.loc[rates["currency_original"].eq(row.currency_original)]
        subset = subset.loc[subset["fx_date"].le(row.session_date)]
        if subset.empty:
            expected_dates.append(pd.NaT)
            expected_rates.append(np.nan)
        else:
            latest = subset.sort_values("fx_date").iloc[-1]
            expected_dates.append(latest.fx_date)
            expected_rates.append(float(latest.fx_to_usd))
    fx_dates_match = _dates(pd.Series(expected_dates)).fillna(pd.Timestamp("1900-01-01")).eq(
        _dates(gold["fx_date"]).fillna(pd.Timestamp("1900-01-01"))
    ).to_numpy()
    _event(findings, check="fx_asof_date", frame=gold,
           mask=~fx_dates_match, reason="FX as-of date differs from latest available accepted quote")
    _event(findings, check="fx_asof_value", frame=gold,
           mask=~_equal(pd.Series(expected_rates), gold["fx_to_usd"]),
           reason="FX conversion differs from selected quote")

    # Recompute simple and logarithmic returns solely from acceptable Gold
    # prices; a quarantine on either side must break the return chain.
    price = pd.to_numeric(gold["close_adjusted_usd"], errors="coerce")
    good = gold["quality_status"].eq("accepted") & np.isfinite(price) & price.gt(0)
    safe = price.where(good)
    previous = safe.groupby(gold["asset_id"]).shift()
    simple = (safe / previous - 1).where(previous.gt(0))
    log = np.log(safe / previous).where(previous.gt(0))
    for name, expected in (
        ("ret_simple_close_adjusted_usd", simple),
        ("ret_log_close_adjusted_usd", log),
    ):
        _event(findings, check=f"returns_{name}", frame=gold,
               mask=~_equal(expected, gold[name], atol=1e-10),
               reason="Return differs from independently computed accepted price path")
    expected_status = pd.Series(np.where(simple.isna(), "unavailable", "accepted"))
    expected_status.loc[simple.abs().gt(0.5)] = "review"
    _event(findings, check="returns_quality_status", frame=gold,
           mask=expected_status.ne(gold["return_quality_status"]).to_numpy(),
           reason="Return review/unavailable status inconsistent with computed return")

    # Manifest signals remain mandatory unresolved evidence even if arithmetic
    # is mathematically consistent. Corporate actions are external correlation
    # only and never automatically waive a quality gate.
    manual = int(manifest.get("manual_correction_rows", 0))
    quarantined = int(gold["quality_status"].ne("accepted").sum())
    review_returns = int(gold["return_quality_status"].eq("review").sum())
    issue_count = len(issues)
    rejected_fx = int(evidence["fx_validation_status"].ne("accepted").sum())
    actions = None
    if action_batch_dir is not None:
        actions, action_manifest = load_verified_batch(action_batch_dir)
        if set(action_manifest.get("asset_ids", [])) != set(gold["asset_id"].astype(str)):
            raise AcceptanceError("Corporate-action batch and candidate have different assets")
    failures = pd.DataFrame(findings, columns=["check", "asset_id", "session_date", "reason"])
    blockers = {
        "integrity_findings": len(failures),
        "unverified_manual_corrections": manual,
        "quarantined_prices": quarantined,
        "review_returns": review_returns,
        "exchange_calendar_issues": issue_count,
        "fx_quote_rejections": rejected_fx,
    }
    # Source capture, corporate action coverage/rights issues, provider revision
    # cross-checks and market-close timestamp interpretation are external gates
    # not independently certified by this local arithmetic audit.
    obligations = [
        "independent_source_observation_verification",
        "corporate_action_adjustment_factor_validation",
        "historical_provider_revision_comparison",
        "exchange_specific_fx_observation_timing_policy",
        "endpoints_and_market_close_completeness",
    ]
    result = {
        "schema": "alpha_edge_lake_v2_independent_acceptance_1",
        "candidate_manifest_sha256": _sha(candidate_dir / "manifest.json"),
        "bronze_manifest_sha256": _sha(bronze_dir / "manifest.json"),
        "corporate_actions_manifest_sha256": (
            _sha(action_batch_dir / "manifest.json") if action_batch_dir else None
        ),
        "legacy_v1_used_as_reference": False,
        "assets": int(gold["asset_id"].nunique()),
        "bronze_rows": len(bars), "gold_rows": len(gold),
        "corporate_action_events": len(actions) if actions is not None else None,
        "blocker_counts": blockers,
        "open_external_evidence_gates": obligations,
        "internal_integrity_status": "failed" if len(failures) else "passed",
        "acceptance": "blocked" if any(blockers.values()) else "review_required",
        "production_eligible": False,
        "s3_writes": 0,
        "limitations": (
            "No AEA-87 dependency. Source hashes and arithmetic consistency do not prove "
            "Yahoo quotations, corporate action adjustments, FX fixing time or absence "
            "of historical provider revisions. No production promotion authorized."
        ),
    }
    return failures, result


def generate(*, candidate_dir: Path, bronze_dir: Path, output: Path,
             action_batch_dir: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {output}")
    failures, report = validate(
        candidate_dir=candidate_dir, bronze_dir=bronze_dir,
        action_batch_dir=action_batch_dir,
    )
    output.mkdir(parents=True, exist_ok=False)
    target = output / "integrity_findings.csv"
    failures.to_csv(target, index=False)
    report["files_sha256"] = {"integrity_findings.csv": _sha(target)}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--bronze-dir", type=Path, required=True)
    parser.add_argument("--actions-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(candidate_dir=args.candidate_dir,
                              bronze_dir=args.bronze_dir,
                              action_batch_dir=args.actions_dir,
                              output=args.output), indent=2))


if __name__ == "__main__":
    main()
