"""Immutable Yahoo Finance Bronze acquisition for Alpha Edge market lake v2.

This module does NOT do currency conversion, UTC normalization, deduplication,
return calculation, cache publication, or legacy writes.

Yahoo/yfinance does not expose the original HTTP response here; "raw" means
untransformed provider bar values and source index representation with
provenance, not byte-for-byte Yahoo API responses.
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
from typing import Any, Callable

import pandas as pd


FIELDS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}
REQUIRED = {"open", "high", "low", "close", "volume"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _single_symbol_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Reject unexpected/multiple tickers instead of assigning data to wrong asset."""
    result = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        scores = [
            sum(str(v).strip().lower().replace(" ", "_") in
                {"open", "high", "low", "close", "adj_close", "volume", "dividends",
                 "stock_splits"} for v in result.columns.get_level_values(i))
            for i in range(result.columns.nlevels)
        ]
        level = scores.index(max(scores))
        if scores[level] < 4:
            raise ValueError(f"{symbol}: cannot identify OHLCV columns")
        for i in range(result.columns.nlevels):
            if i != level:
                observed = {str(v).upper() for v in result.columns.get_level_values(i)}
                if observed != {symbol.upper()}:
                    raise ValueError(f"{symbol}: unexpected provider symbol(s): {observed}")
        result.columns = result.columns.get_level_values(level)
    renamed = {}
    for col in result.columns:
        key = str(col).strip().replace("_", " ").lower()
        possible = [target for source, target in FIELDS.items() if source.lower() == key]
        if possible:
            renamed[col] = possible[0]
    result = result.rename(columns=renamed)
    if result.columns.duplicated().any():
        raise ValueError(f"{symbol}: duplicate provider columns")
    if not REQUIRED.issubset(result.columns):
        raise ValueError(f"{symbol}: missing OHLCV fields {sorted(REQUIRED - set(result.columns))}")
    return result


def _original_index_fields(index: pd.Index) -> tuple[list[str], list[str], list[str]]:
    """Preserve local time/offset without casting dates to UTC."""
    original, sessions, zones = [], [], []
    for value in index:
        stamp = pd.Timestamp(value)
        original.append(stamp.isoformat())
        sessions.append(stamp.date().isoformat())
        zones.append(str(stamp.tzinfo) if stamp.tzinfo is not None else "")
    return original, sessions, zones


def raw_bars_from_frame(
    frame: pd.DataFrame, *, asset_id: str, symbol: str, currency: str,
    batch_id: str, retrieved_at_utc: str, exchange: str = "",
    quote_type: str = "", interval: str = "1d",
) -> pd.DataFrame:
    """Preserve price numbers and source index representation as provided.

    A daily bar's session_date is taken from its ORIGINAL timezone, never from
    converting the timestamp into UTC first.
    """
    if frame.empty:
        raise ValueError(f"{symbol}: empty Yahoo response")
    if interval != "1d":
        raise ValueError("Bronze v2 pilot supports 1d bars only")
    if not asset_id or not symbol or not currency:
        raise ValueError("asset_id, provider symbol and currency are required")
    if currency.strip().upper() in {"NAN", "NONE", "NULL"}:
        raise ValueError(f"{symbol}: currency metadata unknown")
    df = _single_symbol_columns(frame, symbol)
    source_ts, session_date, zones = _original_index_fields(df.index)
    values: dict[str, Any] = {
        "asset_id": [asset_id] * len(df),
        "provider_symbol": [symbol] * len(df),
        "provider": ["yahoo_finance"] * len(df),
        "currency_original": [currency] * len(df),
        "exchange": [exchange] * len(df),
        "quote_type": [quote_type] * len(df),
        "source_timestamp": source_ts,
        "source_timezone": zones,
        "session_date": session_date,
        "interval": [interval] * len(df),
        "ingestion_batch_id": [batch_id] * len(df),
        "ingested_at_utc": [retrieved_at_utc] * len(df),
    }
    for name in FIELDS.values():
        values[name] = df[name].to_numpy() if name in df.columns else [float("nan")] * len(df)
    return pd.DataFrame(values)


def _metadata(ticker: Any) -> dict:
    """Read metadata defensively but NEVER assume USD if currency is missing."""
    result: dict[str, str] = {}
    try:
        info = ticker.fast_info
        for field in ("currency", "exchange", "timezone"):
            value = info.get(field) if hasattr(info, "get") else getattr(info, field, None)
            if value:
                result[field] = str(value)
    except Exception:
        pass
    # Slow .info only when fast_info lacks a currency.
    if not result.get("currency"):
        try:
            info = ticker.info or {}
            for field in ("currency", "exchange", "exchangeTimezoneName", "quoteType"):
                value = info.get(field)
                if value:
                    result[field] = str(value)
        except Exception:
            pass
    return result


def acquire_yahoo_bars(
    asset_id: str, symbol: str, *, start: str, end: str | None,
    batch_id: str, ticker_factory: Callable[[str], Any] | None = None,
    currency_hint: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Read single-symbol Yahoo history WITHOUT auto-adjust or repair."""
    if ticker_factory is None:
        import yfinance as yf
        ticker_factory = yf.Ticker
    ticker = ticker_factory(symbol)
    meta = _metadata(ticker)
    provider_currency = str(meta.get("currency") or "").strip()
    # A hint cannot silently overrule contradictory provider metadata.
    if provider_currency and currency_hint and provider_currency.upper() != currency_hint.upper():
        raise ValueError(f"{symbol}: provider currency {provider_currency} conflicts with hint {currency_hint}")
    currency = provider_currency or str(currency_hint or "").strip()
    if not currency or currency.upper() in {"NAN", "NONE", "NULL"}:
        raise ValueError(f"{symbol}: currency unknown; Bronze acquisition quarantined")
    frame = ticker.history(start=start, end=end, interval="1d",
                           auto_adjust=False, actions=True, repair=False)
    retrieved = _utc_now()
    bars = raw_bars_from_frame(
        frame, asset_id=asset_id, symbol=symbol, currency=currency,
        batch_id=batch_id, retrieved_at_utc=retrieved,
        exchange=meta.get("exchange", ""), quote_type=meta.get("quoteType", ""),
    )
    provenance = {
        "asset_id": asset_id, "provider_symbol": symbol,
        "currency_original": currency,
        "currency_metadata_source": "provider" if provider_currency else "hint",
        "provider_metadata": meta, "start": start, "end_exclusive": end,
        "interval": "1d", "auto_adjust": False, "repair": False,
        "actions": True, "retrieved_at_utc": retrieved, "row_count": len(bars),
        "provider": "yahoo_finance", "client": "yfinance",
    }
    return bars, provenance


def acquire_yahoo_fx_quote(
    symbol: str, *, start: str, end: str | None, batch_id: str,
    ticker_factory: Callable[[str], Any] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Preserve Yahoo FX quotes, not normalized USD-per-currency rates."""
    if ticker_factory is None:
        import yfinance as yf
        ticker_factory = yf.Ticker
    if not symbol.upper().endswith("=X"):
        raise ValueError("FX ticker must be a Yahoo =X quote")
    frame = ticker_factory(symbol).history(
        start=start, end=end, interval="1d", auto_adjust=False,
        actions=False, repair=False,
    )
    if frame.empty:
        raise ValueError(f"{symbol}: empty FX history")
    frame = _single_symbol_columns(frame, symbol)
    timestamps, sessions, zones = _original_index_fields(frame.index)
    retrieved = _utc_now()
    out = pd.DataFrame({
        "provider_symbol": symbol,
        "source_timestamp": timestamps,
        "source_timezone": zones,
        "session_date": sessions,
        "quote_close": frame["close"].to_numpy(),
        "ingestion_batch_id": batch_id,
        "ingested_at_utc": retrieved,
        "provider": "yahoo_finance",
    })
    meta = {"provider_symbol": symbol, "start": start, "end_exclusive": end,
            "interval": "1d", "row_count": len(out),
            "auto_adjust": False, "repair": False, "retrieved_at_utc": retrieved}
    return out, meta


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


def _save_batch(
    *, destination: str, batch_id: str, tables: dict[str, pd.DataFrame],
    provenance: list[dict], bucket: str | None = None,
    region: str = "eu-west-1", upload: bool = False,
    market_root: str = "dev/market",
) -> dict:
    """Store immutable stage-local objects, manifest last. No legacy writes."""
    if upload and not bucket:
        raise ValueError("S3 upload requires a bucket")
    root = market_root.strip("/")
    if upload and (not root or root == "market"):
        raise ValueError("Refusing legacy production market root; select isolated namespace")
    base = Path(destination) / batch_id
    base.mkdir(parents=True, exist_ok=False)
    payloads = {f"{name}.parquet": _parquet_bytes(table) for name, table in tables.items()}
    manifest = {
        "schema_version": "bronze-v2-1", "batch_id": batch_id,
        "created_at_utc": _utc_now(), "sources": provenance,
        "files": {
            name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
            for name, value in payloads.items()
        },
        "publication_state": "bronze_only", "legacy_production_unchanged": True,
    }
    for name, value in payloads.items():
        (base / name).write_bytes(value)
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    (base / "manifest.json").write_bytes(manifest_bytes)
    if upload:
        import boto3
        s3 = boto3.client("s3", region_name=region)
        prefix = f"{root}/lake_v2/bronze/batches/{batch_id}"
        manifest_key = f"{prefix}/manifest.json"
        try:
            s3.head_object(Bucket=bucket, Key=manifest_key)
        except s3.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in ("404", "NotFound", "NoSuchKey"):
                raise
        else:
            raise FileExistsError(f"Already published: s3://{bucket}/{prefix}")
        for name, value in payloads.items():
            s3.put_object(Bucket=bucket, Key=f"{prefix}/{name}", Body=value)
        s3.put_object(Bucket=bucket, Key=manifest_key, Body=manifest_bytes,
                      ContentType="application/json")
        manifest["s3_prefix"] = f"s3://{bucket}/{prefix}"
    manifest["local_path"] = str(base)
    return manifest


def run_batch(
    *, universe: pd.DataFrame, fx_symbols: list[str], start: str,
    end: str | None, batch_id: str, output: str,
    ticker_factory: Callable[[str], Any] | None = None,
    bucket: str | None = None, upload: bool = False,
    market_root: str = "dev/market",
) -> dict:
    """All-or-nothing Bronze batch; any failed symbol prevents publication."""
    if not SAFE_ID.fullmatch(batch_id):
        raise ValueError("Unsafe batch identifier")
    required = {"asset_id", "yahoo_ticker"}
    if not required.issubset(universe.columns):
        raise ValueError(f"Universe missing {sorted(required - set(universe.columns))}")
    if universe.duplicated("asset_id").any():
        raise ValueError("Universe has duplicate asset IDs")
    if not len(universe):
        raise ValueError("Universe is empty")
    frames, metas = [], []
    for row in universe.to_dict("records"):
        symbol = str(row["yahoo_ticker"]).strip()
        currency_hint = row.get("currency")
        if pd.isna(currency_hint):
            currency_hint = None
        bars, provenance = acquire_yahoo_bars(
            str(row["asset_id"]), symbol, start=start, end=end,
            batch_id=batch_id, ticker_factory=ticker_factory,
            currency_hint=currency_hint,
        )
        frames.append(bars)
        metas.append(provenance)
    fx_frames = []
    for symbol in fx_symbols:
        fx, provenance = acquire_yahoo_fx_quote(
            symbol, start=start, end=end, batch_id=batch_id,
            ticker_factory=ticker_factory,
        )
        fx_frames.append(fx)
        metas.append(provenance)
    empty_fx = pd.DataFrame(columns=[
        "provider_symbol", "source_timestamp", "source_timezone",
        "session_date", "quote_close", "ingestion_batch_id",
        "ingested_at_utc", "provider",
    ])
    return _save_batch(
        destination=output, batch_id=batch_id,
        tables={"source_ohlcv": pd.concat(frames, ignore_index=True),
                "source_fx_quotes": pd.concat(fx_frames, ignore_index=True)
                if fx_frames else empty_fx},
        provenance=metas, bucket=bucket, upload=upload,
        market_root=market_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Yahoo Bronze v2 acquisition (pilot)")
    parser.add_argument("--universe-csv", required=True)
    parser.add_argument("--fx-symbols", default="", help="Semicolon-separated Yahoo FX pairs")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None, help="Exclusive Yahoo end date")
    parser.add_argument("--output", default="./outputs/market_bronze_v2")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--market-root", default="dev/market")
    args = parser.parse_args()
    universe = pd.read_csv(args.universe_csv)
    if "include" in universe:
        universe = universe.loc[
            pd.to_numeric(universe["include"], errors="coerce").fillna(0).eq(1)
        ]
    batch = args.batch_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    print(json.dumps(run_batch(
        universe=universe,
        fx_symbols=[s.strip() for s in args.fx_symbols.split(";") if s.strip()],
        start=args.start, end=args.end, batch_id=batch,
        output=args.output, bucket=args.bucket, upload=args.upload,
        market_root=args.market_root,
    ), indent=2))


if __name__ == "__main__":
    main()
