# debug_trade_visibility.py
from __future__ import annotations

import argparse
import json
import boto3

from alpha_edge.core.runtime import load_runtime_config
from alpha_edge.operations.rebuild_ledger import _load_trades, engine_key, TRADES_TABLE


def s3_list_all(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            keys.append(it["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return sorted(keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="prod", choices=["dev", "staging", "prod"])
    ap.add_argument("--dt", default="2026-05-28")
    ap.add_argument("--trade-id", default="broker-20260528-143738-LLY-SELL-close-53482056")
    args = ap.parse_args()

    cfg = load_runtime_config(args.env)
    s3 = boto3.client("s3", region_name=cfg.region)

    target_key = engine_key(cfg, TRADES_TABLE, f"dt={args.dt}", f"trade_{args.trade_id}.json")
    date_prefix = engine_key(cfg, TRADES_TABLE, f"dt={args.dt}") + "/"
    all_prefix = engine_key(cfg, TRADES_TABLE, "dt=")

    print("=== RUNTIME ===")
    print("env:", cfg.env)
    print("bucket:", cfg.bucket)
    print("region:", cfg.region)
    print("engine_root:", cfg.engine_root)
    print("target_key:", target_key)
    print("")

    print("=== HEAD TARGET ===")
    try:
        head = s3.head_object(Bucket=cfg.bucket, Key=target_key)
        print("HEAD OK")
        print("ContentLength:", head.get("ContentLength"))
        print("LastModified:", head.get("LastModified"))
    except Exception as e:
        print("HEAD FAILED:", type(e).__name__, str(e))
    print("")

    print("=== GET TARGET JSON ===")
    try:
        obj = s3.get_object(Bucket=cfg.bucket, Key=target_key)
        payload = json.loads(obj["Body"].read().decode("utf-8"))
        print("GET OK")
        for k in [
            "trade_id", "as_of", "ts_utc", "ticker", "asset_id", "side",
            "action_tag", "quantity", "price", "value", "reported_pnl",
            "quantity_unit", "currency",
        ]:
            print(f"{k}:", payload.get(k))
    except Exception as e:
        print("GET FAILED:", type(e).__name__, str(e))
    print("")

    print("=== LIST DATE PREFIX ===")
    date_keys = s3_list_all(s3, bucket=cfg.bucket, prefix=date_prefix)
    print("date_prefix:", date_prefix)
    print("date_keys_n:", len(date_keys))
    print("target in date listing:", target_key in date_keys)
    for k in date_keys:
        if args.trade_id in k or "LLY" in k or "broker-20260528" in k:
            print("MATCH DATE KEY:", k)
    print("")

    print("=== LIST ALL TRADE PREFIX QUICK CHECK ===")
    all_keys = s3_list_all(s3, bucket=cfg.bucket, prefix=all_prefix)
    json_trade_keys = [
        k for k in all_keys
        if k.endswith(".json") and "/trade_" in k
    ]
    print("all_prefix:", all_prefix)
    print("all_keys_n:", len(all_keys))
    print("json_trade_keys_n:", len(json_trade_keys))
    print("target in all listing:", target_key in all_keys)
    print("target in json_trade_keys:", target_key in json_trade_keys)
    for k in all_keys:
        if args.trade_id in k or "LLY" in k or "broker-20260528" in k:
            print("MATCH ALL KEY:", k)
    print("")

    print("=== rebuild_ledger._load_trades CHECK ===")
    trades = _load_trades(s3, cfg=cfg, start=None, end=None)
    print("_load_trades_n:", len(trades))
    matches = [
        t for t in trades
        if str(t.get("trade_id")) == args.trade_id
        or str(t.get("_s3_key")) == target_key
        or str(t.get("ticker")).upper().strip() == "LLY"
    ]
    print("matches_n:", len(matches))
    for t in matches:
        print(
            json.dumps(
                {
                    "trade_id": t.get("trade_id"),
                    "as_of": t.get("as_of"),
                    "ts_utc": t.get("ts_utc"),
                    "ticker": t.get("ticker"),
                    "asset_id": t.get("asset_id"),
                    "side": t.get("side"),
                    "action_tag": t.get("action_tag"),
                    "quantity": t.get("quantity"),
                    "price": t.get("price"),
                    "value": t.get("value"),
                    "reported_pnl": t.get("reported_pnl"),
                    "_s3_key": t.get("_s3_key"),
                },
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
