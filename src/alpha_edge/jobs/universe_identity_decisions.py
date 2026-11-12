"""Conservative identity-collision decision report for the canonical universe.

This is an audit/decision layer, NOT an automatic asset deletion mechanism.
Row order is a tie-breaker only after instrument identity evidence agrees.
Historical IDs and rows remain unchanged. Optional provider lookup reuses the
existing strict Yahoo resolver; no suggested override is applied automatically.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from alpha_edge.universe.universe import (
    fetch_yahoo_meta,
    resolve_ticker_strict,
    validate_yahoo_meta_for_row,
)

_CORPORATE_WORDS = {
    "inc", "incorporated", "corp", "corporation", "company", "co",
    "ltd", "limited", "plc", "sa", "s", "a", "ag", "nv", "se",
    "holdings", "holding", "group", "the",
}
_REQUIRED = {
    "row_id", "asset_id", "ticker", "yahoo_ticker", "name",
    "asset_class", "role", "region", "include",
}


def _txt(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    result = str(value).strip()
    return "" if result.lower() in {"nan", "none", "null"} else result


def _name_tokens(value: object) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", _txt(value).casefold())
        if token not in _CORPORATE_WORDS
    }


def _company_similarity(first: str, second: str) -> float:
    a, b = _name_tokens(first), _name_tokens(second)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _excluded_rows(exclusions: pd.DataFrame | None) -> tuple[set[str], set[tuple[str, str]]]:
    if exclusions is None or exclusions.empty:
        return set(), set()
    row_ids, legacy = set(), set()
    for _, row in exclusions.iterrows():
        rid = _txt(row.get("target_row_id")).upper()
        if rid:
            row_ids.add(rid)
        else:
            ticker = _txt(row.get("ticker")).upper()
            if ticker:
                legacy.add((ticker, _txt(row.get("asset_class")).lower()))
    return row_ids, legacy


def _is_excluded(row: pd.Series, by_row: set[str], legacy: set[tuple[str, str]]) -> bool:
    rid = _txt(row.get("row_id")).upper()
    ticker = _txt(row.get("ticker")).upper()
    cls = _txt(row.get("asset_class")).lower()
    return rid in by_row or (ticker, cls) in legacy or (ticker, "") in legacy


def identify_collisions(
    universe: pd.DataFrame,
    *,
    exclusions: pd.DataFrame | None = None,
    provider_resolver=None,
    provider_metadata_fetcher=None,
) -> pd.DataFrame:
    """One decision record per shared active Yahoo symbol.

    Supported actions: PROPOSE_EXCLUSION, PROPOSE_OVERRIDE, REVIEW_IDENTITY,
    REVIEW_PROVIDER_MAPPING. All suggestions require separate approval.
    """
    if missing := _REQUIRED - set(universe.columns):
        raise ValueError(f"Identity audit missing columns: {sorted(missing)}")
    if universe["row_id"].astype(str).duplicated().any():
        raise ValueError("Ambiguous duplicate row_id in input universe")
    by_row, legacy = _excluded_rows(exclusions)
    u = universe.copy()
    u["_ordinal"] = range(len(u))
    u["_active"] = pd.to_numeric(u["include"], errors="coerce").eq(1)
    u["_active"] &= ~u.apply(lambda r: _is_excluded(r, by_row, legacy), axis=1)
    u["_symbol"] = u["yahoo_ticker"].map(lambda x: _txt(x).upper())
    u = u.loc[u["_active"] & u["_symbol"].ne("")].copy()
    proposals = []
    for symbol, group in u.groupby("_symbol", sort=True):
        if len(group) < 2:
            continue
        # Handle non-pair collisions conservatively; never assume a single
        # redundant later record when 3+ assets share the symbol.
        ordered = group.sort_values("_ordinal", kind="stable")
        first = ordered.iloc[0]
        for _, later in ordered.iloc[1:].iterrows():
            similar = _company_similarity(first["name"], later["name"])
            broker_equal = _txt(first["ticker"]).upper() == _txt(later["ticker"]).upper()
            market_equal = _txt(first["region"]).casefold() == _txt(later["region"]).casefold()
            type_equal = (
                _txt(first["asset_class"]).casefold() == _txt(later["asset_class"]).casefold()
                and _txt(first["role"]).casefold() == _txt(later["role"]).casefold()
            )
            # Strong identity agreement required for any exclusion proposal.
            duplicate = (
                len(ordered) == 2 and broker_equal and market_equal and type_equal
                and bool(_name_tokens(first["name"])) and similar >= 0.90
            )
            different = (
                similar <= 0.20
                and bool(_name_tokens(first["name"]))
                and bool(_name_tokens(later["name"]))
            )
            action = (
                "PROPOSE_EXCLUSION" if duplicate else
                "PROPOSE_OVERRIDE" if different else "REVIEW_IDENTITY"
            )
            reason = (
                "same broker ticker, market, type and normalized company name; earlier row wins"
                if duplicate else
                "different company names share a provider symbol; verify distinct listing"
                if different else
                "name, instrument, or venue evidence insufficient for automatic identity decision"
            )
            suggested_symbol, source = "", ""
            mapped_row = later
            if different and provider_resolver is not None and provider_metadata_fetcher is not None:
                # Validate which company the *current* Yahoo symbol describes.
                # Never infer the mismatched row from its order alone.
                meta = provider_metadata_fetcher(symbol)
                matches = []
                for target in (first, later):
                    ok, validation_reason, name_score = validate_yahoo_meta_for_row(
                        meta,
                        expected_name=_txt(target["name"]),
                        expected_region=_txt(target["region"]),
                        asset_class=_txt(target["asset_class"]),
                        role=_txt(target["role"]),
                    )
                    matches.append((bool(ok), validation_reason, name_score))
                if sum(ok for ok, _, _ in matches) == 1:
                    mapped_row = later if matches[0][0] else first
                    candidate, evidence = provider_resolver(
                        _txt(mapped_row["ticker"]),
                        expected_name=_txt(mapped_row["name"]),
                        expected_region=_txt(mapped_row["region"]),
                        asset_class=_txt(mapped_row["asset_class"]),
                        role=_txt(mapped_row["role"]),
                    )
                    if candidate and candidate.upper() != symbol:
                        suggested_symbol = candidate
                        source = json.dumps(
                            {"shared_symbol_validation": matches, "resolved_candidate": evidence},
                            sort_keys=True, default=str,
                        )
                    else:
                        action = "REVIEW_PROVIDER_MAPPING"
                        reason += "; no distinct alternative symbol verified"
                else:
                    action = "REVIEW_PROVIDER_MAPPING"
                    reason += "; shared Yahoo symbol did not uniquely identify either company"
            elif different:
                action = "REVIEW_PROVIDER_MAPPING"
                reason += "; independent shared-symbol verification not yet run"
            proposals.append({
                "shared_yahoo_symbol": symbol,
                "group_size": len(ordered),
                "first_row_index": int(first["_ordinal"]),
                "first_row_id": _txt(first["row_id"]),
                "first_asset_id": _txt(first["asset_id"]),
                "first_broker_ticker": _txt(first["ticker"]),
                "first_company_name": _txt(first["name"]),
                "first_region": _txt(first["region"]),
                "second_row_index": int(later["_ordinal"]),
                "second_row_id": _txt(later["row_id"]),
                "second_asset_id": _txt(later["asset_id"]),
                "second_broker_ticker": _txt(later["ticker"]),
                "second_company_name": _txt(later["name"]),
                "second_region": _txt(later["region"]),
                "name_similarity": round(similar, 4),
                "proposed_action": action,
                "retain_asset_id": _txt(first["asset_id"]) if duplicate else "",
                "exclude_row_id": _txt(later["row_id"]) if duplicate else "",
                "override_row_id": _txt(mapped_row["row_id"]) if suggested_symbol else "",
                "suggested_yahoo_ticker": suggested_symbol,
                "evidence": source,
                "rationale": reason,
                "approved": False,
                "applied": False,
            })
    return pd.DataFrame(proposals, columns=[
        "shared_yahoo_symbol", "group_size", "first_row_index",
        "first_row_id", "first_asset_id", "first_broker_ticker",
        "first_company_name", "first_region", "second_row_index",
        "second_row_id", "second_asset_id", "second_broker_ticker",
        "second_company_name", "second_region", "name_similarity",
        "proposed_action", "retain_asset_id", "exclude_row_id",
        "override_row_id", "suggested_yahoo_ticker", "evidence",
        "rationale", "approved", "applied",
    ])


def report_collisions(
    universe: pd.DataFrame, *, exclusions: pd.DataFrame | None = None,
    output_csv: Path | None = None, resolve_online: bool = False,
) -> dict:
    resolver = resolve_ticker_strict if resolve_online else None
    fetcher = fetch_yahoo_meta if resolve_online else None
    report = identify_collisions(
        universe, exclusions=exclusions,
        provider_resolver=resolver,
        provider_metadata_fetcher=fetcher,
    )
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_csv, index=False)
    result = {
        "collision_groups": int(report["shared_yahoo_symbol"].nunique()),
        "decision_rows": len(report),
        "proposed_actions": dict(sorted(Counter(report["proposed_action"]).items())),
        "online_verification": resolve_online,
        "universe_mutations": 0,
        "ledger_mutations": 0,
        "report_csv": str(output_csv) if output_csv is not None else None,
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", type=Path, default=Path("data/universe/universe.csv"))
    p.add_argument("--exclusions", type=Path, default=Path("data/universe/asset_excluded.csv"))
    p.add_argument("--output", type=Path, default=Path("outputs/universe_audit/identity_decisions.csv"))
    p.add_argument(
        "--resolve-online", action="store_true",
        help="Use existing strict Yahoo resolver for different-company pairs",
    )
    args = p.parse_args()
    u = pd.read_csv(args.universe)
    e = pd.read_csv(args.exclusions) if args.exclusions.exists() else None
    print(json.dumps(report_collisions(
        u, exclusions=e, output_csv=args.output,
        resolve_online=args.resolve_online,
    ), indent=2))


if __name__ == "__main__":
    main()
