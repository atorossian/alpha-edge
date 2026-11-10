"""Immutable, local-first Yahoo corporate-action acquisition for Market Data Lake v2.

A separate additive reference candidate. Does not modify market/reference/v1,
production OHLCV/returns, or apply actions to prices. No S3 writes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SAFE_BATCH = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
EVENT_COLUMNS = [
    "asset_id", "provider_symbol", "action_type", "effective_date",
    "source_timestamp", "source_timezone", "value", "unit",
    "currency", "source", "retrieved_at_utc", "ingestion_batch_id",
]
ACTION_FIELDS = {"Dividends": ("DIVIDEND", "currency_per_share"),
                 "Stock Splits": ("SPLIT", "new_shares_per_old_share")}
LEGACY_ACTION_PREFIX = "market/reference/v1/corporate_actions"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _parquet(table: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    table.to_parquet(buffer, index=False)
    return buffer.getvalue()


def acquire_actions(
    asset_id: str, symbol: str, currency: str,
    *, start: str, end: str, batch_id: str,
    ticker_factory: Callable | None = None, retrieved_at: str | None = None,
) -> pd.DataFrame:
    if ticker_factory is None:
        import yfinance as yf
        ticker_factory = yf.Ticker
    if not asset_id or not symbol or not currency:
        raise ValueError("Asset ID, symbol and currency must be specified")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("Expected start before exclusive end")
    ticker = ticker_factory(symbol)
    frame = ticker.history(start=start, end=end, interval="1d",
                           auto_adjust=False, actions=True, repair=False)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{symbol}: Yahoo returned empty history; cannot assert zero actions")
    if frame.index.has_duplicates:
        raise ValueError(f"{symbol}: duplicate Yahoo session labels")
    if not {"Dividends", "Stock Splits"}.issubset(frame.columns):
        raise ValueError(f"{symbol}: action columns missing; cannot assert zero actions")
    observed_currency = None
    try:
        info = ticker.fast_info
        observed_currency = info.get("currency") if hasattr(info, "get") else getattr(info, "currency", None)
    except Exception:
        pass
    if observed_currency and str(observed_currency).upper() != currency.upper():
        raise ValueError(f"{symbol}: supplied currency {currency} conflicts with provider {observed_currency}")
    obtained = retrieved_at or _utc()
    rows = []
    for source_column, (kind, units) in ACTION_FIELDS.items():
        series = pd.to_numeric(frame[source_column], errors="coerce")
        if series.isna().any() or not np.isfinite(series).all() or series.lt(0).any():
            raise ValueError(f"{symbol}: invalid {source_column} evidence")
        for stamp, value in series[series.gt(0)].items():
            ts = pd.Timestamp(stamp)
            rows.append({
                "asset_id": str(asset_id), "provider_symbol": str(symbol),
                "action_type": kind, "effective_date": ts.date().isoformat(),
                "source_timestamp": ts.isoformat(),
                "source_timezone": str(ts.tzinfo) if ts.tzinfo else "",
                "value": float(value), "unit": units, "currency": currency.upper() if kind == "DIVIDEND" else None,
                "source": "yfinance_history", "retrieved_at_utc": obtained,
                "ingestion_batch_id": batch_id,
            })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def create_local_batch(
    *, universe: pd.DataFrame, start: str, end: str,
    output: Path, batch_id: str,
    ticker_factory: Callable | None = None,
    retrieved_at: str | None = None,
) -> dict:
    if not SAFE_BATCH.fullmatch(batch_id):
        raise ValueError("Unsafe batch ID")
    required = {"asset_id", "yahoo_ticker", "currency"}
    if missing := required - set(universe.columns):
        raise ValueError(f"Universe missing {sorted(missing)}")
    if universe.empty or universe["asset_id"].duplicated().any():
        raise ValueError("Universe empty or duplicate asset IDs")
    if (universe[list(required)].isna().any().any() or
        universe[list(required)].astype(str).apply(lambda c: c.str.strip().eq("").any()).any()):
        raise ValueError("Universe contains blank required fields")
    retrieved = retrieved_at or _utc()
    frames = []
    for record in universe.sort_values("asset_id").to_dict("records"):
        frames.append(acquire_actions(
            str(record["asset_id"]), str(record["yahoo_ticker"]),
            str(record["currency"]), start=start, end=end, batch_id=batch_id,
            ticker_factory=ticker_factory, retrieved_at=retrieved,
        ))
    events = pd.concat(frames, ignore_index=True)
    if events.duplicated(["asset_id", "action_type", "effective_date"]).any():
        raise ValueError("Duplicate event keys")
    events = events.sort_values(["asset_id", "effective_date", "action_type"]).reset_index(drop=True)
    payload = _parquet(events)
    base = output / batch_id
    base.mkdir(parents=True, exist_ok=False)
    (base / "events.parquet").write_bytes(payload)
    manifest = {
        "schema": "alpha_edge_corporate_actions_v2_candidate_1",
        "batch_id": batch_id, "created_at_utc": retrieved,
        "start": start, "end_exclusive": end,
        "assets": len(universe), "events": len(events),
        "dividends": int(events["action_type"].eq("DIVIDEND").sum()),
        "splits": int(events["action_type"].eq("SPLIT").sum()),
        "files_sha256": {"events.parquet": _sha(payload)},
        "publication_state": "local_candidate_only", "s3_writes": 0,
        "legacy_production_unchanged": True,
        "limitations": "Yahoo-reported splits and dividends only. Does not cover rights issues; never applies adjustment factors.",
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "local_path": str(base)}


def load_verified_batch(base: Path) -> tuple[pd.DataFrame, dict]:
    manifest_file = base / "manifest.json"
    parquet_file = base / "events.parquet"
    if any(not f.is_file() or f.is_symlink() for f in (manifest_file, parquet_file)):
        raise ValueError("Missing or unsafe corporate-action candidate artifacts")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != "alpha_edge_corporate_actions_v2_candidate_1" or manifest.get("publication_state") != "local_candidate_only":
        raise ValueError("Corporate action candidate manifest contract mismatch")
    if _sha(parquet_file.read_bytes()) != manifest.get("files_sha256", {}).get("events.parquet"):
        raise ValueError("Corporate action candidate hash mismatch")
    events = pd.read_parquet(parquet_file)
    if set(EVENT_COLUMNS) - set(events.columns) or len(events) != manifest.get("events"):
        raise ValueError("Corporate action events inconsistent with manifest")
    return events, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-csv", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="Exclusive end session date")
    parser.add_argument("--output", default="outputs/market_corporate_actions_v2")
    parser.add_argument("--batch-id", default=None)
    args = parser.parse_args()
    u = pd.read_csv(args.universe_csv)
    if "include" in u:
        u = u.loc[pd.to_numeric(u["include"], errors="coerce").fillna(0).eq(1)]
    batch_id = args.batch_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    print(json.dumps(create_local_batch(
        universe=u, start=args.start, end=args.end,
        batch_id=batch_id, output=Path(args.output),
    ), indent=2))


if __name__ == "__main__":
    main()
