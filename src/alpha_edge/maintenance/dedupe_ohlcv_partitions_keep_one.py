# dedupe_ohlcv_partitions_keep_one.py
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import boto3


BUCKET = "alpha-edge-algo"
ROOT_PREFIX = "market/ohlcv_usd/v1/"   # must end with "/"
REGION = "eu-west-1"

# If True: only prints what it WOULD delete
DRY_RUN = False

# Choose which file to keep inside each ticker/year partition:
#   "newest" => keep most recently modified
#   "largest" => keep biggest size
KEEP_STRATEGY = "newest"  # or "largest"


@dataclass
class Obj:
    key: str
    size: int
    last_modified: object  # datetime, but keep generic


PART_RE = re.compile(r"^market/ohlcv_usd/v1/ticker=([^/]+)/year=(\d{4})/.*\.parquet$")


def list_objects(bucket: str, prefix: str) -> List[Obj]:
    s3 = boto3.client("s3", region_name=REGION)
    out: List[Obj] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            k = it["Key"]
            if k.endswith(".parquet"):
                out.append(Obj(key=k, size=int(it["Size"]), last_modified=it["LastModified"]))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return out


def choose_keep(objs: List[Obj]) -> Obj:
    if KEEP_STRATEGY == "largest":
        return max(objs, key=lambda o: o.size)
    # default: newest
    return max(objs, key=lambda o: o.last_modified)


def delete_keys(bucket: str, keys: List[str]) -> None:
    if not keys:
        return
    s3 = boto3.client("s3", region_name=REGION)
    # S3 delete_objects supports up to 1000 keys per request
    for i in range(0, len(keys), 1000):
        chunk = keys[i : i + 1000]
        if DRY_RUN:
            continue
        s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in chunk]})


def main():
    objs = list_objects(BUCKET, ROOT_PREFIX)

    groups: Dict[Tuple[str, str], List[Obj]] = defaultdict(list)
    skipped = 0
    for o in objs:
        m = PART_RE.match(o.key)
        if not m:
            skipped += 1
            continue
        ticker, year = m.group(1), m.group(2)
        groups[(ticker, year)].append(o)

    print(f"[scan] parquet_objects={len(objs)} partitions={len(groups)} skipped_nonmatching={skipped}")
    print(f"[mode] DRY_RUN={DRY_RUN} KEEP_STRATEGY={KEEP_STRATEGY}")

    total_delete = 0
    total_keep = 0

    # Sort for stable output
    for (ticker, year) in sorted(groups.keys(), key=lambda x: (x[0], x[1])):
        items = groups[(ticker, year)]
        if len(items) <= 1:
            continue

        keep = choose_keep(items)
        to_delete = [o.key for o in items if o.key != keep.key]

        total_keep += 1
        total_delete += len(to_delete)

        print(f"\n[ticker={ticker} year={year}] files={len(items)}")
        print(f"  KEEP   : {keep.key} (size={keep.size} last_modified={keep.last_modified})")
        for k in to_delete:
            print(f"  DELETE : {k}")

        delete_keys(BUCKET, to_delete)

    print("\n[summary]")
    print(f"  partitions_with_dupes={total_keep}")
    print(f"  files_to_delete={total_delete}")
    if DRY_RUN:
        print("  (dry run) No deletions executed. Set DRY_RUN=False to apply.")


if __name__ == "__main__":
    main()
