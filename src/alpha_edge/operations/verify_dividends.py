from __future__ import annotations

import argparse
import json
from collections import defaultdict

import boto3

from alpha_edge.core.runtime import load_runtime_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify dividend count, duplicate fingerprints, and native-currency totals.")
    ap.add_argument("--env", default="prod", choices=["dev", "staging", "prod"])
    args = ap.parse_args()

    cfg = load_runtime_config(args.env)
    s3 = boto3.client("s3", region_name=cfg.region)

    prefix = f"{cfg.engine_root.strip('/')}/dividends/dt="
    keys = []
    token = None
    while True:
        kwargs = {"Bucket": cfg.bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        keys.extend([
            x["Key"]
            for x in resp.get("Contents", [])
            if x["Key"].endswith(".json") and "/dividend_" in x["Key"]
        ])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

    rows = []
    for k in sorted(keys):
        obj = s3.get_object(Bucket=cfg.bucket, Key=k)
        dv = json.loads(obj["Body"].read().decode("utf-8"))
        rows.append((k, dv))

    by_id = defaultdict(list)
    by_fp = defaultdict(list)
    by_year_currency = defaultdict(float)
    by_currency = defaultdict(float)

    positive = 0
    negative = 0

    for k, dv in rows:
        did = str(dv.get("dividend_id", "")).strip()
        as_of = str(dv.get("as_of", "")).strip()
        year = as_of[:4]
        ts_utc = str(dv.get("ts_utc", "")).strip()
        ticker = str(dv.get("ticker", "")).upper().strip()
        amount = round(float(dv.get("amount", 0.0)), 2)
        currency = str(dv.get("currency", "USD")).upper().strip()

        if amount >= 0:
            positive += 1
        else:
            negative += 1

        by_id[did].append(k)
        by_fp[(as_of, ts_utc, ticker, amount, currency)].append((k, did))
        by_year_currency[(year, currency)] += amount
        by_currency[currency] += amount

    dupe_ids = {k: v for k, v in by_id.items() if k and len(v) > 1}
    dupe_fps = {k: v for k, v in by_fp.items() if len(v) > 1}

    print("=== DIVIDEND VERIFY ===")
    print("env:", cfg.env)
    print("bucket:", cfg.bucket)
    print("dividend json objects:", len(rows))
    print("positive rows:", positive)
    print("negative adjustment rows:", negative)
    print("duplicate dividend_id groups:", len(dupe_ids))
    print("duplicate economic fingerprint groups:", len(dupe_fps))
    print("")
    print("by currency native total:")
    for ccy in sorted(by_currency):
        print(ccy, round(by_currency[ccy], 2))
    print("")
    print("by year/currency native total:")
    for (year, ccy), amt in sorted(by_year_currency.items()):
        print(year, ccy, round(amt, 2))

    if dupe_fps:
        print("")
        print("first duplicate fingerprints:")
        for fp, vals in list(dupe_fps.items())[:20]:
            print("DUPLICATE", fp)
            for k, did in vals:
                print(" ", did, k)


if __name__ == "__main__":
    main()
