from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import boto3
import pandas as pd

BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"
TRADES_TABLE = "trades"
UNIVERSE_PREFIX = "engine/v1/universe/"


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(region: str = REGION):
    return boto3.client("s3", region_name=region)


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def s3_list_objects(s3, *, bucket: str, prefix: str) -> list[dict]:
    out: list[dict] = []
    token = None
    while True:
        kwargs: dict[str, Any] = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        out.extend(resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


def discover_latest_key(s3, *, bucket: str, prefix: str) -> str | None:
    objs = s3_list_objects(s3, bucket=bucket, prefix=prefix)
    if not objs:
        return None
    objs = [o for o in objs if "Key" in o and "LastModified" in o]
    if not objs:
        return None
    objs.sort(key=lambda o: o["LastModified"], reverse=True)
    return str(objs[0]["Key"])


def s3_get_bytes(s3, *, bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


# ----------------------------
# Load universe (local or S3)
# ----------------------------
def load_df_from_local(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path)


def load_df_from_s3(s3, *, bucket: str, key: str) -> pd.DataFrame:
    data = s3_get_bytes(s3, bucket=bucket, key=key)
    if key.lower().endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    return pd.read_csv(io.StringIO(data.decode("utf-8", errors="replace")))


def load_universe_df(
    s3,
    *,
    bucket: str,
    universe_path: str | None = None,
    universe_key: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Returns (universe_df, universe_ref_string)
    """
    if universe_path:
        df = load_df_from_local(universe_path)
        ref = f"file://{universe_path}"
    else:
        key = universe_key or discover_latest_key(s3, bucket=bucket, prefix=UNIVERSE_PREFIX)
        if not key:
            raise RuntimeError(
                f"Could not find universe under s3://{bucket}/{UNIVERSE_PREFIX}. "
                "Pass --universe-path or --universe-key explicitly."
            )
        df = load_df_from_s3(s3, bucket=bucket, key=key)
        ref = f"s3://{bucket}/{key}"

    # required universe columns
    for c in ["row_id", "ticker", "asset_id"]:
        if c not in df.columns:
            raise RuntimeError(f"Universe missing required column: {c}")

    df = df.copy()
    df["row_id"] = df["row_id"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()

    return df, ref


def build_broker_map(universe_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    ticker -> list[asset_id] (keeps duplicates so we can detect ambiguity)
    """
    m: dict[str, list[str]] = {}
    for bt, sub in universe_df.groupby("ticker"):
        ids = [str(x) for x in sub["asset_id"].dropna().unique().tolist() if str(x).strip()]
        if ids:
            m[str(bt)] = ids
    return m


def build_rowid_to_assetid_map(universe_df: pd.DataFrame) -> dict[str, str]:
    """
    row_id -> asset_id (must be unique per row_id)
    """
    m: dict[str, str] = {}
    sub = universe_df[["row_id", "asset_id"]].dropna().copy()
    for _, r in sub.iterrows():
        rid = str(r["row_id"]).strip()
        aid = str(r["asset_id"]).strip()
        if rid and aid:
            m[rid] = aid
    return m


# ----------------------------
# Overrides (local or S3) - row-based
# ----------------------------
def load_overrides_df_from_local(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_overrides_df_from_s3(s3, *, bucket: str, key: str) -> pd.DataFrame:
    data = s3_get_bytes(s3, bucket=bucket, key=key)
    return pd.read_csv(io.StringIO(data.decode("utf-8", errors="replace")))


def load_overrides_row_pick(
    s3,
    *,
    bucket: str,
    overrides_path: str | None = None,
    overrides_key: str | None = None,
    overrides_ticker_col: str = "ticker",
    overrides_id_col: str = "target_row_id",
) -> tuple[dict[str, str], str | None]:
    """
    Overrides file selects a universe row to use for a given trade ticker.

    Expected columns (defaults):
      - ticker              (trade/broker ticker)
      - target_row_id       (universe row_id to use)
    Returns:
      ticker -> target_row_id
    """
    if not overrides_path and not overrides_key:
        return {}, None

    if overrides_path:
        df = load_overrides_df_from_local(overrides_path)
        ref = f"file://{overrides_path}"
    else:
        df = load_overrides_df_from_s3(s3, bucket=bucket, key=str(overrides_key))
        ref = f"s3://{bucket}/{overrides_key}"

    if overrides_ticker_col not in df.columns:
        raise RuntimeError(f"Overrides CSV missing required column: {overrides_ticker_col}")
    if overrides_id_col not in df.columns:
        raise RuntimeError(f"Overrides CSV missing required column: {overrides_id_col}")

    df = df.copy()
    df[overrides_ticker_col] = df[overrides_ticker_col].astype(str).str.upper().str.strip()
    df[overrides_id_col] = df[overrides_id_col].astype(str).str.strip()

    out: dict[str, str] = {}
    for _, r in df.iterrows():
        t = str(r[overrides_ticker_col]).upper().strip()
        rid = str(r[overrides_id_col]).strip()
        if t and rid:
            out[t] = rid

    return out, ref


# ----------------------------
# Backfill logic
# ----------------------------
@dataclass
class BackfillResult:
    key: str
    status: str  # "updated" | "skipped" | "error"
    reason: str | None = None
    ticker: str | None = None
    asset_id: str | None = None


def resolve_asset_id_for_trade(
    *,
    ticker: str,
    broker_map: dict[str, list[str]],
    overrides_rowpick: dict[str, str],
    rowid_to_assetid: dict[str, str],
    mode: str,
) -> tuple[str | None, list[str] | None, str | None]:
    """
    Returns (asset_id, candidates, err_reason)

    mode:
      - strict: missing/ambiguous => error
      - null:   missing/ambiguous => asset_id=None, candidates populated if ambiguous

    resolution order:
      1) overrides_rowpick (ticker -> target_row_id -> asset_id)
      2) broker_map (ticker -> [asset_id])
    """
    bt = str(ticker).upper().strip()

    # (1) overrides pick exact universe row
    if bt in overrides_rowpick:
        rid = overrides_rowpick[bt]
        aid = rowid_to_assetid.get(str(rid).strip())
        if not aid:
            if mode == "null":
                return None, None, None
            return None, None, f"override row_id not found in universe: ticker={bt} target_row_id={rid}"
        return str(aid), None, None

    # (2) fallback: broker_map
    ids = broker_map.get(bt)
    if not ids:
        if mode == "null":
            return None, None, None
        return None, None, f"no universe mapping for ticker={bt}"

    if len(ids) == 1:
        return ids[0], None, None

    if mode == "null":
        return None, [str(x) for x in ids], None
    return None, [str(x) for x in ids], f"ambiguous ticker={bt} asset_ids={ids}"


def backfill_one_key(
    *,
    s3,
    bucket: str,
    key: str,
    broker_map: dict[str, list[str]],
    overrides_rowpick: dict[str, str],
    rowid_to_assetid: dict[str, str],
    universe_ref: str,
    overrides_ref: str | None,
    mode: str,
    dry_run: bool,
) -> BackfillResult:
    try:
        payload = json.loads(s3_get_bytes(s3, bucket=bucket, key=key).decode("utf-8"))

        if not isinstance(payload, dict):
            return BackfillResult(key=key, status="error", reason="payload is not a dict")

        if payload.get("asset_id"):
            return BackfillResult(key=key, status="skipped", reason="already has asset_id")

        ticker = payload.get("ticker")
        if ticker is None:
            return BackfillResult(key=key, status="error", reason="missing ticker field")

        asset_id, candidates, err = resolve_asset_id_for_trade(
            ticker=str(ticker),
            broker_map=broker_map,
            overrides_rowpick=overrides_rowpick,
            rowid_to_assetid=rowid_to_assetid,
            mode=mode,
        )
        if err is not None:
            return BackfillResult(key=key, status="error", reason=err, ticker=str(ticker))

        payload["asset_id"] = asset_id  # may be None if mode=null
        if candidates:
            payload["asset_id_candidates"] = candidates

        payload["_universe_ref"] = universe_ref
        if overrides_ref:
            payload["_overrides_ref"] = overrides_ref

        payload["_backfill"] = {
            "job": "backfill_trades_asset_id",
            "ts_utc": pd.Timestamp.utcnow().isoformat(),
            "mode": mode,
        }

        if dry_run:
            return BackfillResult(key=key, status="updated", ticker=str(ticker), asset_id=asset_id)

        s3_put_json(s3, bucket=bucket, key=key, payload=payload)
        return BackfillResult(key=key, status="updated", ticker=str(ticker), asset_id=asset_id)

    except Exception as e:
        return BackfillResult(key=key, status="error", reason=f"{type(e).__name__}: {e}")


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Backfill asset_id into engine/v1/trades/dt=*/trade_*.json using universe ticker mapping + row-based overrides."
    )
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--region", default=REGION)

    ap.add_argument("--universe-path", default=None, help="Local path to universe snapshot (csv/parquet).")
    ap.add_argument("--universe-key", default=None, help="S3 key to universe snapshot (csv/parquet).")

    ap.add_argument("--overrides-path", default=None, help="Local path to universe_overrides.csv (row-based).")
    ap.add_argument("--overrides-key", default=None, help="S3 key to overrides CSV.")

    ap.add_argument("--overrides-ticker-col", default="ticker", help="Column in overrides that contains the broker ticker.")
    ap.add_argument("--overrides-id-col", default="target_row_id", help="Column in overrides that contains the universe row_id to use.")

    ap.add_argument("--start", default=None, help="Only process dt >= start (YYYY-MM-DD).")
    ap.add_argument("--end", default=None, help="Only process dt <= end (YYYY-MM-DD).")

    ap.add_argument("--mode", default="strict", choices=["strict", "null"], help="strict: fail on missing/ambiguous; null: write asset_id=None (and candidates for ambiguous).")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="Optional max number of trade files to process.")
    ap.add_argument("--dry-run", action="store_true")

    return ap.parse_args()


def _extract_dt_from_key(key: str) -> str | None:
    parts = key.split("/")
    for p in parts:
        if p.startswith("dt="):
            return p.replace("dt=", "")
    return None


def main() -> None:
    args = parse_args()

    s3 = s3_client(args.region)

    universe_df, universe_ref = load_universe_df(
        s3,
        bucket=args.bucket,
        universe_path=args.universe_path,
        universe_key=args.universe_key,
    )
    broker_map = build_broker_map(universe_df)
    rowid_to_assetid = build_rowid_to_assetid_map(universe_df)

    overrides_rowpick, overrides_ref = load_overrides_row_pick(
        s3,
        bucket=args.bucket,
        overrides_path=args.overrides_path,
        overrides_key=args.overrides_key,
        overrides_ticker_col=args.overrides_ticker_col,
        overrides_id_col=args.overrides_id_col,
    )

    prefix = engine_key(TRADES_TABLE, "dt=")
    objs = s3_list_objects(s3, bucket=args.bucket, prefix=prefix)
    keys = [o["Key"] for o in objs if isinstance(o.get("Key"), str)]
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]

    # filter by dt range if provided
    if args.start or args.end:
        start_d = pd.Timestamp(args.start).date() if args.start else None
        end_d = pd.Timestamp(args.end).date() if args.end else None
        filtered: list[str] = []
        for k in keys:
            dts = _extract_dt_from_key(k)
            if not dts:
                continue
            d = pd.Timestamp(dts).date()
            if start_d and d < start_d:
                continue
            if end_d and d > end_d:
                continue
            filtered.append(k)
        keys = filtered

    keys.sort()
    if args.limit is not None:
        keys = keys[: int(args.limit)]

    print("=== BACKFILL TRADES ASSET_ID ===")
    print(f"bucket:         {args.bucket}")
    print(f"prefix:         s3://{args.bucket}/{prefix}")
    print(f"universe_ref:   {universe_ref}")
    print(f"overrides_ref:  {overrides_ref or '(none)'}")
    print(f"mode:           {args.mode}")
    print(f"dry_run:        {bool(args.dry_run)}")
    print(f"keys_to_check:  {len(keys)}")
    if overrides_rowpick:
        print(f"overrides:      {len(overrides_rowpick)} tickers (row-pick)")
    print("")

    updated = 0
    skipped = 0
    errors = 0

    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futs = [
            ex.submit(
                backfill_one_key,
                s3=s3,
                bucket=args.bucket,
                key=k,
                broker_map=broker_map,
                overrides_rowpick=overrides_rowpick,
                rowid_to_assetid=rowid_to_assetid,
                universe_ref=universe_ref,
                overrides_ref=overrides_ref,
                mode=args.mode,
                dry_run=bool(args.dry_run),
            )
            for k in keys
        ]

        for i, fut in enumerate(cf.as_completed(futs), start=1):
            r: BackfillResult = fut.result()
            if r.status == "updated":
                updated += 1
            elif r.status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"[ERROR] {r.key} :: {r.reason}")

            if i % 250 == 0 or i == len(futs):
                print(f"[progress] done={i}/{len(futs)} updated={updated} skipped={skipped} errors={errors}")

    print("\n=== DONE ===")
    print(f"updated: {updated}")
    print(f"skipped: {skipped}")
    print(f"errors:  {errors}")
    print("")

    if errors > 0 and args.mode == "strict":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
