from __future__ import annotations

import argparse
import io
import json
from typing import Any, Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alpha_edge.warehouse.schemas import (
    DIM_ASSETS_SCHEMA,
    FCT_ACCOUNT_PNL_DAILY_SCHEMA,
    FCT_DAILY_REPORT_STATS_SCHEMA,
    FCT_POSITIONS_DAILY_SCHEMA,
    FCT_TRADES_SCHEMA,
    enforce_schema,
)

DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT = "engine/v1"
WAREHOUSE_ROOT = "warehouse"
WAREHOUSE_VERSION = "v=1"


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(region: str):
    return boto3.client("s3", region_name=region)


def join_key(*parts: str) -> str:
    return "/".join([p.strip("/") for p in parts if p is not None and str(p).strip("/") != ""])


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


def s3_get_bytes(s3, *, bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def s3_put_bytes(s3, *, bucket: str, key: str, data: bytes, content_type: str) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def s3_put_parquet_table(
    s3,
    *,
    bucket: str,
    key: str,
    table: pa.Table,
    compression: str = "snappy",
) -> None:
    buf = io.BytesIO()
    pq.write_table(table, buf, compression=compression)
    s3_put_bytes(s3, bucket=bucket, key=key, data=buf.getvalue(), content_type="application/octet-stream")


def s3_get_json(s3, *, bucket: str, key: str) -> dict:
    data = s3_get_bytes(s3, bucket=bucket, key=key)
    return json.loads(data.decode("utf-8", errors="replace"))


def now_ts_utc_ms() -> pd.Timestamp:
    # stored as timestamp (no tz) but in UTC “meaning”
    return pd.Timestamp.utcnow().tz_localize(None).floor("ms")


def parse_date(s: str) -> str:
    d = pd.Timestamp(s).date()
    return d.strftime("%Y-%m-%d")


# ----------------------------
# Paths
# ----------------------------
def lake_key(engine_root: str, *parts: str) -> str:
    return join_key(engine_root, *parts)


def wh_key(engine_root: str, table: str, *parts: str) -> str:
    # engine/v1/warehouse/<table>/v=1/...
    return join_key(engine_root, WAREHOUSE_ROOT, table, WAREHOUSE_VERSION, *parts)


# ----------------------------
# Builders
# ----------------------------
def build_dim_assets_from_universe_csv(
    *,
    universe_path: str,
    source_ref: str,
    load_ts: pd.Timestamp,
) -> pa.Table:
    u = pd.read_csv(universe_path)

    # Minimum required columns
    for c in ["asset_id", "row_id", "broker_ticker", "include"]:
        if c not in u.columns:
            raise RuntimeError(f"Universe CSV missing required column: {c}")

    # normalize tickers
    for c in ["broker_ticker", "ticker", "yahoo_ticker"]:
        if c in u.columns:
            u[c] = u[c].astype(str).str.upper().str.strip()

    # include -> is_tradable
    u["include"] = pd.to_numeric(u["include"], errors="coerce").fillna(0).astype("int32")
    u["is_tradable"] = u["include"].astype(int) == 1

    out = pd.DataFrame(
        {
            "asset_id": u["asset_id"].astype(str).str.strip(),
            "row_id": u.get("row_id", None),
            "broker_ticker": u.get("broker_ticker", None),
            "ticker": u.get("ticker", None),
            "yahoo_ticker": u.get("yahoo_ticker", None),
            "name": u.get("name", None),
            "asset_class": u.get("asset_class", None),
            "role": u.get("role", None),
            "currency": u.get("currency", None),
            "exchange": u.get("exchange", None),
            "country": u.get("country", None),
            "market": u.get("market", None),
            "include": u.get("include", None),
            "is_tradable": u.get("is_tradable", None),
            "lock_yahoo_ticker": u.get("lock_yahoo_ticker", None),
            "yahoo_ok": u.get("yahoo_ok", None),
            "yahoo_symbol_used": u.get("yahoo_symbol_used", None),
            "resolver_debug": u.get("resolver_debug", None),
            "valid_from": u.get("valid_from", None),
            "valid_to": u.get("valid_to", None),
            "load_ts_utc": load_ts,
            "source_ref": source_ref,
        }
    )

    res = enforce_schema(out, DIM_ASSETS_SCHEMA)
    return res.table


def build_fct_trades_for_dt(
    s3,
    *,
    bucket: str,
    engine_root: str,
    dt: str,
    account_id: str,
    load_ts: pd.Timestamp,
) -> pa.Table:
    prefix = lake_key(engine_root, "trades", f"dt={dt}/")
    objs = s3_list_objects(s3, bucket=bucket, prefix=prefix)
    keys = [o["Key"] for o in objs if isinstance(o.get("Key"), str)]
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]

    rows: list[dict] = []
    for k in sorted(keys):
        try:
            t = s3_get_json(s3, bucket=bucket, key=k)
            if not isinstance(t, dict):
                continue

            trade_id = str(t.get("trade_id") or "").strip()
            as_of = str(t.get("as_of") or "").strip()
            if not trade_id or not as_of:
                continue

            rows.append(
                {
                    "trade_id": trade_id,
                    "as_of_date": pd.Timestamp(as_of).date(),
                    "ts_utc": t.get("ts_utc", None),
                    "account_id": account_id,
                    "asset_id": t.get("asset_id", None),
                    "broker_ticker": t.get("ticker", None),
                    "side": t.get("side", None),
                    "quantity": t.get("quantity", None),
                    "price": t.get("price", None),
                    "currency": t.get("currency", None),
                    "action_tag": t.get("action_tag", None),
                    "quantity_unit": t.get("quantity_unit", None),
                    "value": t.get("value", None),
                    "reported_pnl": t.get("reported_pnl", None),
                    "choice_id": t.get("choice_id", None),
                    "portfolio_run_id": t.get("portfolio_run_id", None),
                    "note": t.get("note", None),
                    "source_key": k,
                    "load_ts_utc": load_ts,
                }
            )
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame({f.name: [] for f in FCT_TRADES_SCHEMA})

    res = enforce_schema(df, FCT_TRADES_SCHEMA)
    return res.table


def build_fct_positions_daily_for_dt(
    s3,
    *,
    bucket: str,
    engine_root: str,
    dt: str,
    account_id: str,
    load_ts: pd.Timestamp,
) -> pa.Table:
    key = lake_key(engine_root, "ledger", f"dt={dt}", "positions.json")
    payload = s3_get_json(s3, bucket=bucket, key=key)

    as_of = str(payload.get("as_of") or dt)
    as_of_date = pd.Timestamp(as_of).date()

    spot = payload.get("spot_positions") or []
    deriv = payload.get("derivatives_positions") or []

    rows: list[dict] = []

    for p in spot:
        if not isinstance(p, dict):
            continue
        rows.append(
            {
                "as_of_date": as_of_date,
                "account_id": account_id,
                "asset_id": p.get("asset_id", None),
                "broker_ticker": p.get("ticker", None),
                "position_type": "SPOT",
                "side": None,
                "quantity": p.get("quantity", None),
                "avg_cost_usd": p.get("avg_cost", None),
                "last_price_usd": p.get("last_price", None),
                "market_value_usd": p.get("market_value", None),
                "cost_value_usd": p.get("cost_value", None),
                "unrealized_pnl_usd": p.get("unrealized_pnl", None),
                "open_notional_usd": None,
                "avg_entry_price_usd": None,
                "currency": p.get("currency", None),
                "missing_price_flag": (p.get("last_price", None) is None),
                "source_key": key,
                "load_ts_utc": load_ts,
            }
        )

    for p in deriv:
        if not isinstance(p, dict):
            continue
        rows.append(
            {
                "as_of_date": as_of_date,
                "account_id": account_id,
                "asset_id": p.get("asset_id", None),
                "broker_ticker": p.get("ticker", None),
                "position_type": "NOTIONAL",
                "side": p.get("side", None),
                "quantity": None,
                "avg_cost_usd": None,
                "last_price_usd": None,
                "market_value_usd": None,
                "cost_value_usd": None,
                "unrealized_pnl_usd": None,
                "open_notional_usd": p.get("open_notional_usd", None),
                "avg_entry_price_usd": p.get("avg_entry_price", None),
                "currency": p.get("currency", None),
                "missing_price_flag": None,
                "source_key": key,
                "load_ts_utc": load_ts,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame({f.name: [] for f in FCT_POSITIONS_DAILY_SCHEMA})

    res = enforce_schema(df, FCT_POSITIONS_DAILY_SCHEMA)
    return res.table


def build_fct_account_pnl_daily_for_dt(
    s3,
    *,
    bucket: str,
    engine_root: str,
    dt: str,
    account_id: str,
    load_ts: pd.Timestamp,
) -> pa.Table:
    key = lake_key(engine_root, "ledger", f"dt={dt}", "pnl.json")
    payload = s3_get_json(s3, bucket=bucket, key=key)

    as_of = str(payload.get("as_of") or payload.get("summary", {}).get("as_of") or dt)
    as_of_date = pd.Timestamp(as_of).date()

    summary = payload.get("summary") or {}
    method = payload.get("method", None)

    equity_usd = summary.get("equity", None)
    if equity_usd is None:
        equity_usd = summary.get("equity_usd", None)

    row = {
        "as_of_date": as_of_date,
        "account_id": account_id,
        "realized_pnl_usd": summary.get("realized_pnl", None),
        "unrealized_pnl_usd": summary.get("unrealized_pnl_spot", None),
        "total_pnl_usd": summary.get("total_pnl", None),
        "equity_usd": equity_usd,
        "trade_count": summary.get("trade_count", None),
        "tickers_spot": summary.get("tickers_spot", None),
        "tickers_derivatives": summary.get("tickers_derivatives", None),
        "method": method,
        "source_key": key,
        "load_ts_utc": load_ts,
    }

    df = pd.DataFrame([row])
    res = enforce_schema(df, FCT_ACCOUNT_PNL_DAILY_SCHEMA)
    return res.table


def build_fct_daily_report_stats_for_dt(
    s3,
    *,
    bucket: str,
    report_key: str,
    dt: str,
    account_id: str,
    load_ts: pd.Timestamp,
) -> Optional[pa.Table]:
    try:
        payload = s3_get_json(s3, bucket=bucket, key=report_key)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    as_of = str(payload.get("date") or payload.get("as_of") or dt)
    as_of_date = pd.Timestamp(as_of).date()

    row = {
        "as_of_date": as_of_date,
        "account_id": account_id,
        "total_notional_usd": payload.get("total_notional_usd", payload.get("total_notional", None)),
        "equity_usd": payload.get("equity_usd", payload.get("equity", None)),
        "leverage": payload.get("leverage", None),
        "ann_return": payload.get("ann_return", None),
        "ann_vol": payload.get("ann_vol", payload.get("ann_volatility", None)),
        "sharpe": payload.get("sharpe", None),
        "max_drawdown": payload.get("max_drawdown", None),
        "ruin_prob": payload.get("ruin_prob", None),
        "score": payload.get("score", None),
        "alpha_vs_bench": payload.get("alpha_vs_bench", None),
        "source_key": report_key,
        "load_ts_utc": load_ts,
    }

    df = pd.DataFrame([row])
    res = enforce_schema(df, FCT_DAILY_REPORT_STATS_SCHEMA)
    return res.table


# ----------------------------
# Runner
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Alpha Edge warehouse partitions for a single date (dt).")

    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--engine-root", default=DEFAULT_ENGINE_ROOT)

    ap.add_argument("--dt", required=True, help="Partition date YYYY-MM-DD")
    ap.add_argument("--account-id", default="main")

    ap.add_argument("--universe-path", default=None, help="Local path to universe.csv (to build dim_assets).")
    ap.add_argument("--build-dim-assets", action="store_true", help="Rewrite dim_assets snapshot (unpartitioned).")

    # NEW: dim-only mode
    ap.add_argument(
        "--dim-assets-only",
        action="store_true",
        help="If set, build dim_assets (requires --build-dim-assets) and exit without building any fact tables.",
    )

    ap.add_argument(
        "--report-key",
        default=None,
        help="S3 key to report.json for this dt. If omitted, defaults to engine_root/reports/dt=DT/report.json",
    )
    ap.add_argument("--dry-run", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    dt_str = parse_date(args.dt)
    s3 = s3_client(args.region)
    load_ts = now_ts_utc_ms()

    bucket = str(args.bucket)
    engine_root = str(args.engine_root).strip("/")
    account_id = str(args.account_id)

    # default report key
    report_key = args.report_key
    if report_key is None:
        report_key = lake_key(engine_root, "reports", f"dt={dt_str}", "report.json")

    # 1) dim_assets (optional full snapshot rewrite)
    if args.build_dim_assets:
        if not args.universe_path:
            raise SystemExit("--build-dim-assets requires --universe-path")
        dim_assets_key = wh_key(engine_root, "dim_assets", "dim_assets.parquet")

        table = build_dim_assets_from_universe_csv(
            universe_path=args.universe_path,
            source_ref=f"file://{args.universe_path}",
            load_ts=load_ts,
        )

        print(f"[dim_assets] rows={table.num_rows} -> s3://{bucket}/{dim_assets_key}")
        if not args.dry_run:
            s3_put_parquet_table(s3, bucket=bucket, key=dim_assets_key, table=table)

        # NEW: stop here if dim-only
        if args.dim_assets_only:
            print("[OK] dim_assets-only mode: skipped fact tables.")
            return

    # guard: dim-assets-only without build-dim-assets is a misuse
    if args.dim_assets_only and (not args.build_dim_assets):
        raise SystemExit("--dim-assets-only requires --build-dim-assets")

    # 2) facts for dt (required)
    trades_table = build_fct_trades_for_dt(
        s3,
        bucket=bucket,
        engine_root=engine_root,
        dt=dt_str,
        account_id=account_id,
        load_ts=load_ts,
    )
    trades_out_key = wh_key(engine_root, "fct_trades", f"dt={dt_str}", "part-00000.parquet")
    print(f"[fct_trades] rows={trades_table.num_rows} -> s3://{bucket}/{trades_out_key}")
    if not args.dry_run:
        s3_put_parquet_table(s3, bucket=bucket, key=trades_out_key, table=trades_table)

    positions_table = build_fct_positions_daily_for_dt(
        s3,
        bucket=bucket,
        engine_root=engine_root,
        dt=dt_str,
        account_id=account_id,
        load_ts=load_ts,
    )
    positions_out_key = wh_key(engine_root, "fct_positions_daily", f"dt={dt_str}", "part-00000.parquet")
    print(f"[fct_positions_daily] rows={positions_table.num_rows} -> s3://{bucket}/{positions_out_key}")
    if not args.dry_run:
        s3_put_parquet_table(s3, bucket=bucket, key=positions_out_key, table=positions_table)

    pnl_table = build_fct_account_pnl_daily_for_dt(
        s3,
        bucket=bucket,
        engine_root=engine_root,
        dt=dt_str,
        account_id=account_id,
        load_ts=load_ts,
    )
    pnl_out_key = wh_key(engine_root, "fct_account_pnl_daily", f"dt={dt_str}", "part-00000.parquet")
    print(f"[fct_account_pnl_daily] rows={pnl_table.num_rows} -> s3://{bucket}/{pnl_out_key}")
    if not args.dry_run:
        s3_put_parquet_table(s3, bucket=bucket, key=pnl_out_key, table=pnl_table)

    report_table = build_fct_daily_report_stats_for_dt(
        s3,
        bucket=bucket,
        report_key=report_key,
        dt=dt_str,
        account_id=account_id,
        load_ts=load_ts,
    )
    if report_table is None:
        print(f"[fct_daily_report_stats] missing -> skipped (expected key: s3://{bucket}/{report_key})")
    else:
        report_out_key = wh_key(engine_root, "fct_daily_report_stats", f"dt={dt_str}", "part-00000.parquet")
        print(f"[fct_daily_report_stats] rows={report_table.num_rows} -> s3://{bucket}/{report_out_key}")
        if not args.dry_run:
            s3_put_parquet_table(s3, bucket=bucket, key=report_out_key, table=report_table)

    print("[OK] warehouse build done.")


if __name__ == "__main__":
    main()