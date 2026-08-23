# run_current_equity_valuation.py
from __future__ import annotations

import argparse
import json

import boto3
import pandas as pd

from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation, runtime_dt_key, runtime_engine_key
from alpha_edge.portfolio.equity_valuation import (
    resolve_current_equity,
    print_equity_valuation_result,
)

EQUITY_VALUATION_TABLE = "portfolio/equity_valuation"


def _s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calculate current portfolio equity as a standalone process.")
    p.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    p.add_argument("--as-of", default=None, help="Valuation date YYYY-MM-DD. Default: today UTC.")
    p.add_argument("--equity-override", type=float, default=None, help="Manual bypass value. If provided, ledger/prices are not required.")
    p.add_argument("--price-column", default=None, help="Optional latest_prices.parquet column to use for valuation.")
    p.add_argument("--no-latest-fallback", action="store_true", help="Do not fall back to ledger latest.json when dt partition is missing.")
    p.add_argument("--write", action="store_true", help="Persist valuation result to S3.")
    p.add_argument("--confirm-prod-write", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(args.env)
    as_of = pd.Timestamp(args.as_of).date().strftime("%Y-%m-%d") if args.as_of else pd.Timestamp.utcnow().date().strftime("%Y-%m-%d")

    if bool(args.write):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    result = resolve_current_equity(
        cfg=cfg,
        as_of=as_of,
        equity_override=args.equity_override,
        allow_latest_fallback=not bool(args.no_latest_fallback),
        price_column=args.price_column,
    )
    print_equity_valuation_result(result)

    if args.write:
        s3 = boto3.client("s3", region_name=cfg.region)
        payload = result.to_dict()
        dt_key = runtime_dt_key(cfg, EQUITY_VALUATION_TABLE, result.as_of, "equity_valuation.json")
        latest_key = runtime_engine_key(cfg, EQUITY_VALUATION_TABLE, "latest.json")
        _s3_put_json(s3, bucket=cfg.bucket, key=dt_key, payload=payload)
        _s3_put_json(s3, bucket=cfg.bucket, key=latest_key, payload=payload)
        print("[OK] wrote equity valuation:")
        print(f"  s3://{cfg.bucket}/{dt_key}")
        print(f"  s3://{cfg.bucket}/{latest_key}")


if __name__ == "__main__":
    main()
