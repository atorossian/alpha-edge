from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run

import argparse
import io
import json
from typing import Any, Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import RuntimeConfig
from alpha_edge.warehouse.schemas import (
    DIM_ASSETS_SCHEMA,
    FCT_ACCOUNT_PNL_DAILY_SCHEMA,
    FCT_DAILY_REPORT_STATS_SCHEMA,
    FCT_POSITIONS_DAILY_SCHEMA,
    FCT_TRADES_SCHEMA,
    enforce_schema,
)


WAREHOUSE_ROOT = "warehouse"
WAREHOUSE_VERSION = "v=1"


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


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
    s3_put_bytes(
        s3,
        bucket=bucket,
        key=key,
        data=buf.getvalue(),
        content_type="application/octet-stream",
    )


def s3_get_json(s3, *, bucket: str, key: str) -> dict:
    data = s3_get_bytes(s3, bucket=bucket, key=key)
    return json.loads(data.decode("utf-8", errors="replace"))


def now_ts_utc_ms() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None).floor("ms")


def parse_date(s: str) -> str:
    d = pd.Timestamp(s).date()
    return d.strftime("%Y-%m-%d")


# ----------------------------
# Paths
# ----------------------------
def lake_key(cfg: RuntimeConfig, *parts: str) -> str:
    return join_key(cfg.engine_root, *parts)


def wh_key(cfg: RuntimeConfig, table: str, *parts: str) -> str:
    return join_key(cfg.engine_root, WAREHOUSE_ROOT, table, WAREHOUSE_VERSION, *parts)


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

    for c in ["asset_id", "row_id", "broker_ticker", "include"]:
        if c not in u.columns:
            raise RuntimeError(f"Universe CSV missing required column: {c}")

    for c in ["broker_ticker", "ticker", "yahoo_ticker"]:
        if c in u.columns:
            u[c] = u[c].astype(str).str.upper().str.strip()

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
    cfg: RuntimeConfig,
    dt: str,
    account_id: str,
    load_ts: pd.Timestamp,
) -> pa.Table:
    prefix = lake_key(cfg, "trades", f"dt={dt}/")
    objs = s3_list_objects(s3, bucket=cfg.bucket, prefix=prefix)
    keys = [o["Key"] for o in objs if isinstance(o.get("Key"), str)]
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]

    rows: list[dict] = []

    for k in sorted(keys):
        try:
            t = s3_get_json(s3, bucket=cfg.bucket, key=k)
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
        except Exception as e:
            raise RuntimeError(f"Failed to build fct_trades from s3://{cfg.bucket}/{k}: {e}") from e

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame({f.name: [] for f in FCT_TRADES_SCHEMA})

    res = enforce_schema(df, FCT_TRADES_SCHEMA)
    return res.table


def build_fct_positions_daily_for_dt(
    s3,
    *,
    cfg: RuntimeConfig,
    dt: str,
    account_id: str,
    load_ts: pd.Timestamp,
) -> pa.Table:
    key = lake_key(cfg, "ledger", f"dt={dt}", "positions.json")
    payload = s3_get_json(s3, bucket=cfg.bucket, key=key)

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
    cfg: RuntimeConfig,
    dt: str,
    account_id: str,
    load_ts: pd.Timestamp,
) -> pa.Table:
    key = lake_key(cfg, "ledger", f"dt={dt}", "pnl.json")
    payload = s3_get_json(s3, bucket=cfg.bucket, key=key)

    as_of = str(payload.get("as_of") or payload.get("summary", {}).get("as_of") or dt)
    as_of_date = pd.Timestamp(as_of).date()

    summary = payload.get("summary") or {}
    method = payload.get("method", None)

    dividends_pnl_usd = summary.get("dividends_pnl_usd", summary.get("dividends_usd", None))
    net_cashflow_usd = summary.get("net_cashflow_usd", summary.get("cashflow_usd", None))
    equity_usd = summary.get("equity_usd", summary.get("equity", None))

    row = {
        "as_of_date": as_of_date,
        "account_id": account_id,
        "realized_pnl_usd": summary.get("realized_pnl", summary.get("realized_pnl_usd", None)),
        "unrealized_pnl_usd": summary.get("unrealized_pnl_spot", summary.get("unrealized_pnl_usd", None)),
        "dividends_pnl_usd": dividends_pnl_usd,
        "net_cashflow_usd": net_cashflow_usd,
        "total_pnl_usd": summary.get("total_pnl", summary.get("total_pnl_usd", None)),
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
    cfg: RuntimeConfig,
    report_key: str,
    dt: str,
    account_id: str,
    load_ts: pd.Timestamp,
) -> Optional[pa.Table]:
    try:
        payload = s3_get_json(s3, bucket=cfg.bucket, key=report_key)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    as_of = str(payload.get("date") or payload.get("as_of") or dt)
    as_of_date = pd.Timestamp(as_of).date()

    eval_obj = None
    if isinstance(payload.get("report"), dict):
        rep = payload["report"]
        if isinstance(rep.get("eval"), dict):
            eval_obj = rep["eval"]

    def pick(*keys, default=None):
        for k in keys:
            if k in payload and payload[k] is not None:
                return payload[k]

        if isinstance(eval_obj, dict):
            for k in keys:
                if k in eval_obj and eval_obj[k] is not None:
                    return eval_obj[k]

        return default

    row = {
        "as_of_date": as_of_date,
        "account_id": account_id,
        "total_notional_usd": pick("total_notional_usd", "total_notional"),
        "equity_usd": pick("equity_usd", "equity"),
        "leverage": pick("leverage"),
        "ann_return": pick("ann_return"),
        "ann_vol": pick("ann_vol", "ann_volatility"),
        "sharpe": pick("sharpe"),
        "max_drawdown": pick("max_drawdown"),
        "ruin_prob": pick("ruin_prob"),
        "score": pick("score"),
        "alpha_vs_bench": pick("alpha_vs_bench"),
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
    ap = argparse.ArgumentParser(description="Build Alpha Edge warehouse partitions for a single date.")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument("--dt", required=True, help="Partition date YYYY-MM-DD")
    ap.add_argument("--account-id", default="main")

    ap.add_argument("--universe-path", default=None, help="Local path to universe.csv to build dim_assets.")
    ap.add_argument("--build-dim-assets", action="store_true", help="Rewrite dim_assets snapshot.")
    ap.add_argument(
        "--dim-assets-only",
        action="store_true",
        help="Build dim_assets and exit without building fact tables.",
    )

    ap.add_argument(
        "--report-key",
        default=None,
        help="S3 key to report.json. If omitted, defaults to <env-root>/daily_reports/dt=DT/report.json",
    )

    ap.add_argument("--dry-run", action="store_true")

    return ap.parse_args()


def _main_impl() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)

    if not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    dt_str = parse_date(args.dt)
    s3 = s3_client(cfg)
    load_ts = now_ts_utc_ms()

    account_id = str(args.account_id)

    print("\n=== BUILD WAREHOUSE ===")
    print(f"env:        {cfg.env}")
    print(f"bucket:     {cfg.bucket}")
    print(f"region:     {cfg.region}")
    print(f"root:       {cfg.engine_root}")
    print(f"dt:         {dt_str}")
    print(f"account_id: {account_id}")
    print(f"dry_run:    {bool(args.dry_run)}")
    print("")

    report_key = args.report_key
    if report_key is None:
        report_key = lake_key(cfg, "daily_reports", f"dt={dt_str}", "report.json")

    # 1) dim_assets
    if args.build_dim_assets:
        if not args.universe_path:
            raise SystemExit("--build-dim-assets requires --universe-path")

        dim_assets_key = wh_key(cfg, "dim_assets", "dim_assets.parquet")

        table = build_dim_assets_from_universe_csv(
            universe_path=args.universe_path,
            source_ref=f"file://{args.universe_path}",
            load_ts=load_ts,
        )

        print(f"[dim_assets] rows={table.num_rows} -> s3://{cfg.bucket}/{dim_assets_key}")

        if not args.dry_run:
            s3_put_parquet_table(s3, bucket=cfg.bucket, key=dim_assets_key, table=table)

        if args.dim_assets_only:
            print("[OK] dim_assets-only mode: skipped fact tables.")
            return

    if args.dim_assets_only and not args.build_dim_assets:
        raise SystemExit("--dim-assets-only requires --build-dim-assets")

    # 2) fct_trades
    trades_table = build_fct_trades_for_dt(
        s3,
        cfg=cfg,
        dt=dt_str,
        account_id=account_id,
        load_ts=load_ts,
    )
    trades_out_key = wh_key(cfg, "fct_trades", f"dt={dt_str}", "part-00000.parquet")
    print(f"[fct_trades] rows={trades_table.num_rows} -> s3://{cfg.bucket}/{trades_out_key}")

    if not args.dry_run:
        s3_put_parquet_table(s3, bucket=cfg.bucket, key=trades_out_key, table=trades_table)

    # 3) fct_positions_daily
    positions_table = build_fct_positions_daily_for_dt(
        s3,
        cfg=cfg,
        dt=dt_str,
        account_id=account_id,
        load_ts=load_ts,
    )
    positions_out_key = wh_key(cfg, "fct_positions_daily", f"dt={dt_str}", "part-00000.parquet")
    print(f"[fct_positions_daily] rows={positions_table.num_rows} -> s3://{cfg.bucket}/{positions_out_key}")

    if not args.dry_run:
        s3_put_parquet_table(s3, bucket=cfg.bucket, key=positions_out_key, table=positions_table)

    # 4) fct_account_pnl_daily
    pnl_table = build_fct_account_pnl_daily_for_dt(
        s3,
        cfg=cfg,
        dt=dt_str,
        account_id=account_id,
        load_ts=load_ts,
    )
    pnl_out_key = wh_key(cfg, "fct_account_pnl_daily", f"dt={dt_str}", "part-00000.parquet")
    print(f"[fct_account_pnl_daily] rows={pnl_table.num_rows} -> s3://{cfg.bucket}/{pnl_out_key}")

    if not args.dry_run:
        s3_put_parquet_table(s3, bucket=cfg.bucket, key=pnl_out_key, table=pnl_table)

    # 5) fct_daily_report_stats
    report_table = build_fct_daily_report_stats_for_dt(
        s3,
        cfg=cfg,
        report_key=report_key,
        dt=dt_str,
        account_id=account_id,
        load_ts=load_ts,
    )

    if report_table is None:
        print(f"[fct_daily_report_stats] missing -> skipped expected=s3://{cfg.bucket}/{report_key}")
    else:
        report_out_key = wh_key(cfg, "fct_daily_report_stats", f"dt={dt_str}", "part-00000.parquet")
        print(f"[fct_daily_report_stats] rows={report_table.num_rows} -> s3://{cfg.bucket}/{report_out_key}")

        if not args.dry_run:
            s3_put_parquet_table(s3, bucket=cfg.bucket, key=report_out_key, table=report_table)

    print("[OK] warehouse build done.")


# ----------------------------
# Audit/logging entrypoint wrapper
# ----------------------------
def _tier1_audit_is_dry_run(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "dry_run", False) or getattr(args, "no_write", False))


def main_with_audit() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = _tier1_audit_is_dry_run(args)

    with capture_script_run(
        cfg=cfg,
        script_name="build_warehouse.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            _main_impl()

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="warehouse",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="build_warehouse.py",
                source_mode="warehouse",
                status=("dry_run" if is_dry_run else "success"),
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                    "note": "Tier 1 audit event is entrypoint-level. Detailed output keys/row counts are available in the script log stdout and script-specific metadata where emitted by the script.",
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
        except Exception as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="warehouse",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="build_warehouse.py",
                source_mode="warehouse",
                status="failed",
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise


def main() -> None:
    main_with_audit()


if __name__ == "__main__":
    main()
