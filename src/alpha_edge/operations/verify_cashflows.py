from __future__ import annotations

import argparse
import json
from collections import defaultdict

import boto3

from alpha_edge.core.runtime import load_runtime_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify cashflow count, duplicate fingerprints, and net cashflow.")
    ap.add_argument("--env", default="prod", choices=["dev", "staging", "prod"])
    args = ap.parse_args()

    cfg = load_runtime_config(args.env)
    s3 = boto3.client("s3", region_name=cfg.region)

    prefix = f"{cfg.engine_root.strip('/')}/cashflows/dt="
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
            if x["Key"].endswith(".json") and "/cashflow_" in x["Key"]
        ])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

    rows = []
    for k in sorted(keys):
        obj = s3.get_object(Bucket=cfg.bucket, Key=k)
        cf = json.loads(obj["Body"].read().decode("utf-8"))
        rows.append((k, cf))

    by_id = defaultdict(list)
    by_fp = defaultdict(list)
    net = 0.0
    deposits = 0.0
    withdrawals = 0.0

    by_year = defaultdict(lambda: {"deposits": 0.0, "withdrawals": 0.0, "net": 0.0, "n": 0})

    for k, cf in rows:
        cid = str(cf.get("cashflow_id", "")).strip()
        as_of = str(cf.get("as_of", "")).strip()
        year = as_of[:4]
        ts_utc = str(cf.get("ts_utc", "")).strip()
        typ = str(cf.get("type") or cf.get("direction") or cf.get("cashflow_type") or "").upper().strip()
        amount = round(float(cf.get("amount", 0.0)), 2)
        currency = str(cf.get("currency", "USD")).upper().strip()

        sign = +1.0 if typ in {"DEPOSIT", "IN", "CREDIT"} else -1.0 if typ in {"WITHDRAWAL", "OUT", "DEBIT"} else 0.0
        signed = sign * amount
        net += signed
        by_year[year]["net"] += signed
        by_year[year]["n"] += 1
        if sign > 0:
            deposits += amount
            by_year[year]["deposits"] += amount
        elif sign < 0:
            withdrawals += amount
            by_year[year]["withdrawals"] += amount

        by_id[cid].append(k)
        by_fp[(as_of, ts_utc, typ, amount, currency)].append((k, cid))

    dupe_ids = {k: v for k, v in by_id.items() if k and len(v) > 1}
    dupe_fps = {k: v for k, v in by_fp.items() if len(v) > 1}

    print("=== CASHFLOW VERIFY ===")
    print("env:", cfg.env)
    print("bucket:", cfg.bucket)
    print("cashflow json objects:", len(rows))
    print("gross deposits by JSON amount:", round(deposits, 2))
    print("gross withdrawals by JSON amount:", round(withdrawals, 2))
    print("net by JSON amount:", round(net, 2))
    print("duplicate cashflow_id groups:", len(dupe_ids))
    print("duplicate economic fingerprint groups:", len(dupe_fps))
    print("")
    print("by year:")
    for y in sorted(by_year):
        v = by_year[y]
        print(y, "n=", int(v["n"]), "deposits=", round(v["deposits"], 2), "withdrawals=", round(v["withdrawals"], 2), "net=", round(v["net"], 2))

    if dupe_fps:
        print("")
        print("first duplicate fingerprints:")
        for fp, vals in list(dupe_fps.items())[:20]:
            print("DUPLICATE", fp)
            for k, cid in vals:
                print(" ", cid, k)


if __name__ == "__main__":
    main()
