"""Read-only cross-market diagnostics; optionally stage the user-supplied DB1 evidence."""

from __future__ import annotations

import argparse
import io
import json

import boto3
import numpy as np
import pandas as pd

from alpha_edge.market.build_silver_sessions_v3 import CORRECTION_ROOT, prepare_session_candidate
from alpha_edge.market.ingest_bronze_full_history import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    _completed,
    _digest,
    _json_bytes,
    _read,
    load_universe,
)
from alpha_edge.market.ingest_bronze_yahoo_v2 import acquire_yahoo_bars
from alpha_edge.market.lakehouse_v3.storage import put_immutable

FINGERPRINT = "50e7096fc839c9f3d4b372a75ec539e9ee63982eefa3e2d68fa808ea71c90399"
RUN_ID = "bronze-aea93-cross-market-20261118"
DB1 = "EQH1ef5845b5bfba875"
AUD = "EQHd70d5a1a729cd8f3"
FIELDS = ["open", "high", "low", "close"]


def read_bronze(s3, bucket, asset):
    receipt = _completed(s3, bucket, run_id=RUN_ID, record=asset, verify_hash=True)
    if receipt is None:
        raise ValueError(f"Missing verified Bronze: {asset['asset_id']}")
    frame = pd.read_parquet(io.BytesIO(_read(s3, bucket, receipt["data"]["key"])))
    return frame, receipt


def db1_correction(frame, receipt):
    # Investing.com observations supplied by the user, with displayed rounded volume.
    evidence = [
        ("2007-09-12", 78.400, 78.750, 77.270, 78.320, "887.29K", -0.62),
        ("2007-09-13", 78.180, 81.150, 78.000, 80.930, "2.20M", 3.33),
        ("2007-09-14", 80.700, 80.760, 78.500, 80.180, "1.24M", -0.93),
    ]
    records = [
        dict(zip(["session_date", *FIELDS, "volume_display", "change_percent"], row, strict=True)) for row in evidence
    ]
    indexed = frame.set_index("session_date", verify_integrity=True)
    for record in records:
        source = indexed.loc[record["session_date"]]
        fields = FIELDS if record["session_date"] != "2007-09-12" else FIELDS[:3]
        if not all(np.isclose(source[field], record[field], atol=0.0005, rtol=0) for field in fields):
            raise ValueError("DB1 provider observation differs from the corroborating evidence")
        displayed = record["volume_display"]
        scale = 1000 if displayed.endswith("K") else 1000000
        rounded_volume = float(displayed[:-1]) * scale
        if abs(float(source["volume"]) - rounded_volume) > 0.005 * scale:
            raise ValueError("DB1 volume differs from the rounded external evidence")
    original = float(indexed.loc["2007-09-12", "close"])
    if not np.isclose(original, 78.809998, atol=1e-5, rtol=0):
        raise ValueError("DB1 original closing price changed")
    return {
        "schema": "alpha_edge_silver_close_correction_v1",
        "correction_id": "DB1.DE-2007-09-12-investing-close",
        "kind": "close_correction",
        "asset_id": DB1,
        "provider_symbol": "DB1.DE",
        "currency_original": "EUR",
        "session_date": "2007-09-12",
        "original_close": original,
        "corrected_close": 78.32,
        "source_fingerprint": FINGERPRINT,
        "source_data_sha256": receipt["data"]["sha256"],
        "source_reference": "Investing.com, historical prices supplied by Alejandro Torossian",
        "reason": "Yahoo close exceeds the daily high; neighboring OHLC and rounded volumes corroborate identity.",
        "verification_status": "user_supplied_candidate_evidence",
        "evidence": records,
    }


def yahoo_history(symbol, asset_id, currency):
    frame, _ = acquire_yahoo_bars(
        asset_id,
        symbol,
        start=None,
        end="2026-11-18",
        batch_id="aea93-peer-validation",
        currency_hint=currency,
        full_history=True,
    )
    return frame


def compare_fx(aud):
    inverse = yahoo_history("AUD=X", "FX:USD-AUD-VALIDATION", "AUD")
    inverse_prices = inverse[FIELDS].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(inverse_prices).all(axis=1) & inverse_prices.gt(0).all(axis=1)
    usable = inverse_prices.where(valid)
    converted = inverse[["session_date", "source_timestamp"]].rename(columns={"source_timestamp": "inverse_timestamp"})
    for field, inverse_field in zip(FIELDS, ["open", "low", "high", "close"], strict=True):
        converted[f"inverse_{field}"] = 1 / usable[inverse_field]
    candidate, findings, _ = prepare_session_candidate(
        aud, asset_id=AUD, symbol="AUDUSD=X", calendar_id="YAHOO_FX_WEEKDAYS"
    )
    joined = candidate.merge(converted, on="session_date", how="left", validate="one_to_one")
    flagged = joined.loc[joined["session_quality_status"].eq("quarantined")].copy()
    for field in FIELDS:
        flagged[f"{field}_difference_bps"] = (flagged[f"inverse_{field}"] / flagged[field] - 1) * 10000
    inverse_ohlc = flagged[[f"inverse_{field}" for field in FIELDS]]
    valid_range = (
        inverse_ohlc.notna().all(axis=1)
        & flagged["inverse_high"].ge(flagged[["inverse_open", "inverse_close", "inverse_low"]].max(axis=1))
        & flagged["inverse_low"].le(flagged[["inverse_open", "inverse_close"]].min(axis=1))
    )
    timestamps_match = pd.to_datetime(flagged["source_timestamp"], utc=True).eq(
        pd.to_datetime(flagged["inverse_timestamp"], utc=True)
    )
    print("\nAUD/USD inverse comparison")
    print("Flagged:", len(flagged), "valid inverse OHLC:", int(valid_range.sum()))
    print("Matching timestamp labels:", int(timestamps_match.sum()))
    print(flagged[[f"{field}_difference_bps" for field in FIELDS]].abs().describe().to_string())
    missing = findings.loc[findings["issue"].eq("missing_expected_session"), ["session_date"]]
    print(missing.merge(converted, on="session_date", how="left").to_string(index=False))


def compare_german_peers(db1, assets):
    _, findings, _ = prepare_session_candidate(db1, asset_id=DB1, symbol="DB1.DE", calendar_id="XETR")
    dates = findings.loc[
        findings["issue"].isin(["missing_expected_session", "unexpected_session"]), ["session_date", "issue"]
    ].copy()
    for symbol in ("DBK.DE", "DTE.DE"):
        matches = [a for a in assets if a["yahoo_ticker"] == symbol]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely resolve {symbol} in the pinned universe")
        peer = yahoo_history(symbol, matches[0]["asset_id"], "EUR")
        dates[symbol] = dates["session_date"].isin(peer["session_date"])
    print("\nDB1 calendar findings — peer observation present")
    print(dates.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--stage-db1-correction", action="store_true")
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=DEFAULT_REGION)
    assets, _ = load_universe(s3, args.bucket, FINGERPRINT)
    by_id = {asset["asset_id"]: asset for asset in assets}
    db1, receipt = read_bronze(s3, args.bucket, by_id[DB1])
    correction = db1_correction(db1, receipt)
    payload = _json_bytes(correction)
    version = _digest(payload)
    if args.stage_db1_correction:
        put_immutable(
            s3,
            args.bucket,
            f"{CORRECTION_ROOT}/{version}.json",
            payload,
            content_type="application/json",
            scope=CORRECTION_ROOT,
        )
    print(json.dumps({"correction_version": version, "staged": args.stage_db1_correction}, indent=2))
    aud, _ = read_bronze(s3, args.bucket, by_id[AUD])
    compare_fx(aud)
    compare_german_peers(db1, assets)


if __name__ == "__main__":
    main()
