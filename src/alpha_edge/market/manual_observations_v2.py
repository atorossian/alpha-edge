"""Apply explicit manual OHLCV observations to a Bronze-derived *candidate*.

Original Yahoo Bronze parquet and manifests are never modified. A correction
is an appended, attributable observation, not a provider rewrite. Do not
invent adjusted close, dividends, split factors or a precise close timestamp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_edge.market.data_lake_v2 import DataContractError


REQUIRED_CORRECTION = {
    "asset_id", "provider_symbol", "currency_original", "session_date",
    "open", "high", "low", "close", "volume", "correction_id",
    "reported_by", "verification_status",
}
NUMERIC = ("open", "high", "low", "close", "volume")
OPTIONAL = ("adj_close", "source_reference", "notes", "adj_close_policy")
SCHEMA_COLS = (
    "provider", "source_timestamp", "source_timezone", "interval",
    "ingestion_batch_id", "ingested_at_utc", "dividends", "stock_splits",
)


def append_manual_observations(
    source_bars: pd.DataFrame, corrections: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (candidate source observations, detailed correction audit).

    Fail closed on contradictory identities, duplicate keys, malformed
    observations, or attempts to replace any existing provider observation.
    Unverified corrections are intentionally kept from adjusted-close returns.
    """
    if missing := REQUIRED_CORRECTION - set(corrections.columns):
        raise DataContractError(f"Manual corrections missing fields: {sorted(missing)}")
    for col in ("asset_id", "provider_symbol", "currency_original", "session_date"):
        if col not in source_bars:
            raise DataContractError(f"Source Bronze missing {col}")

    records = corrections.copy()
    for col in OPTIONAL:
        if col not in records:
            records[col] = pd.NA

    for key in ("asset_id", "provider_symbol", "currency_original",
                "correction_id", "reported_by", "verification_status"):
        records[key] = records[key].astype("string").str.strip()
        if records[key].isna().any() or records[key].eq("").any():
            raise DataContractError(f"Manual correction has missing {key}")
    if records["correction_id"].duplicated().any():
        raise DataContractError("Correction IDs must be unique")
    allowed = {"pending_independent_verification", "user_asserted_unverified"}
    if not records["verification_status"].isin(allowed).all():
        raise DataContractError("Unexpected correction approval state")

    records["session_date"] = pd.to_datetime(
        records["session_date"], format="%Y-%m-%d", errors="coerce"
    )
    if records["session_date"].isna().any():
        raise DataContractError("Invalid manual correction session date")
    records["currency_original"] = records["currency_original"].str.upper()

    for col in NUMERIC:
        records[col] = pd.to_numeric(records[col], errors="coerce")
    if records[list(NUMERIC)].isna().any().any() or not np.isfinite(
        records[list(NUMERIC)].to_numpy(dtype=float)
    ).all():
        raise DataContractError("Manual OHLCV must be finite")
    if (
        records["open"].le(0).any()
        or records["high"].le(0).any()
        or records["low"].le(0).any()
        or records["close"].le(0).any()
        or records["volume"].lt(0).any()
        or records["low"].gt(records["high"]).any()
        or records["high"].lt(records[["open", "close"]].max(axis=1)).any()
        or records["low"].gt(records[["open", "close"]].min(axis=1)).any()
    ):
        raise DataContractError("Manual OHLCV violates price range or volume constraints")

    # Accept only an explicit, traceable user assertion of adj_close == close.
    records["adj_close"] = pd.to_numeric(records["adj_close"], errors="coerce")
    policy = records["adj_close_policy"].astype("string").fillna("").str.strip()
    asserted = records["verification_status"].eq("user_asserted_unverified")
    if (asserted & (~policy.eq("explicit_equal_close")
                    | records["adj_close"].isna()
                    | ~np.isclose(records["adj_close"].fillna(-1),
                                  records["close"], rtol=0, atol=1e-9))).any():
        raise DataContractError("User asserted adjusted close must explicitly equal close")
    if ((~asserted) & records["adj_close"].notna()).any():
        raise DataContractError("Unverified corrections must not provide adjusted close")
    if ((~asserted) & policy.ne("")).any():
        raise DataContractError("Unexpected adjusted close policy")
    records["adj_close_origin"] = np.where(asserted, "user_asserted_equal_close", "not_available")

    identity = source_bars[
        ["asset_id", "provider_symbol", "currency_original"]
    ].drop_duplicates()
    for entry in records.itertuples(index=False):
        existing = identity.loc[identity["asset_id"].eq(entry.asset_id)]
        if len(existing) != 1 or (
            existing.iloc[0]["provider_symbol"] != entry.provider_symbol
            or str(existing.iloc[0]["currency_original"]).upper() != entry.currency_original
        ):
            raise DataContractError(f"Identity mismatch for {entry.asset_id}")

    source_date = pd.to_datetime(
        source_bars["session_date"].astype(str).str[:10],
        format="%Y-%m-%d", errors="coerce"
    )
    original_keys = pd.MultiIndex.from_arrays(
        [source_bars["asset_id"].astype(str), source_date]
    )
    correction_keys = pd.MultiIndex.from_arrays(
        [records["asset_id"].astype(str), records["session_date"]]
    )
    if correction_keys.has_duplicates:
        raise DataContractError("Duplicate correction asset/session key")
    if correction_keys.isin(original_keys).any():
        raise DataContractError("Refusing to overwrite existing Yahoo observation")

    candidate = source_bars.copy()
    candidate["observation_origin"] = "yahoo_finance"
    candidate["correction_id"] = pd.NA
    candidate["verification_status"] = "provider_observation"
    candidate["reported_by"] = pd.NA
    candidate["source_reference"] = pd.NA
    candidate["notes"] = pd.NA
    candidate["adj_close_origin"] = "yahoo_finance"
    candidate["adj_close_policy"] = pd.NA

    records["observation_origin"] = "user_supplied_manual"
    records["provider"] = "manual_correction"
    records["source_timestamp"] = pd.NA
    records["source_timezone"] = pd.NA
    records["interval"] = "1d"
    records["ingestion_batch_id"] = pd.NA
    records["ingested_at_utc"] = pd.NA
    records["dividends"] = np.nan
    records["stock_splits"] = np.nan
    candidate = pd.concat([candidate, records], ignore_index=True, sort=False)
    if candidate.duplicated(["asset_id", "session_date"]).any():
        raise DataContractError("Candidate has duplicate session keys")

    audit = records[[
        "correction_id", "asset_id", "provider_symbol", "currency_original",
        "session_date", "open", "high", "low", "close", "volume",
        "adj_close", "adj_close_origin", "adj_close_policy",
        "observation_origin", "verification_status",
        "reported_by", "source_reference", "notes",
    ]].copy()
    return candidate, audit
