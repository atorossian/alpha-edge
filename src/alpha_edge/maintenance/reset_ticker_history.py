from __future__ import annotations

import argparse
import json

import boto3
import fsspec

from alpha_edge.core.market_store import MarketStore


def delete_prefix(bucket: str, prefix: str, dry_run: bool = True) -> int:
    s3 = boto3.client("s3")
    deleted = 0
    token = None

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        contents = resp.get("Contents") or []
        keys = [c["Key"] for c in contents]

        if keys:
            if dry_run:
                for k in keys[:30]:
                    print(f"[DRY_RUN] would delete s3://{bucket}/{k}")
                if len(keys) > 30:
                    print(f"[DRY_RUN] ... and {len(keys) - 30} more in this page")
                deleted += len(keys)
            else:
                # batch delete
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": k} for k in keys], "Quiet": True},
                )
                deleted += len(keys)

        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

    return deleted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="alpha-edge-algo")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", action="store_true", help="Required to actually delete + mutate state")
    args = ap.parse_args()

    t = str(args.ticker).upper().strip()
    store = MarketStore(bucket=args.bucket)

    # compute prefixes (keys, not s3://...)
    ohlcv_key_prefix = f"{store.base_prefix}/ohlcv_usd/{store.version}/ticker={t}/"
    returns_key_prefix = f"{store.base_prefix}/returns_usd/{store.version}/ticker={t}/"

    print(f"[INFO] target ticker={t}")
    print(f"[INFO] ohlcv prefix:   s3://{args.bucket}/{ohlcv_key_prefix}")
    print(f"[INFO] returns prefix: s3://{args.bucket}/{returns_key_prefix}")

    n1 = delete_prefix(args.bucket, ohlcv_key_prefix, dry_run=bool(args.dry_run or not args.confirm))
    n2 = delete_prefix(args.bucket, returns_key_prefix, dry_run=bool(args.dry_run or not args.confirm))

    print(f"[INFO] matched ohlcv objects:   {n1}")
    print(f"[INFO] matched returns objects: {n2}")

    # state edit
    state_path = f"{store.state_prefix}/last_date_by_ticker.json"
    if args.dry_run or not args.confirm:
        print(f"[DRY_RUN] would remove '{t}' from {state_path}")
        return

    with fsspec.open(state_path, "r") as f:
        st = json.load(f)

    if t in st:
        st.pop(t, None)
        with fsspec.open(state_path, "w") as f:
            json.dump(st, f, indent=2)
        print(f"[OK] removed {t} from last_date_by_ticker.json")
    else:
        print(f"[OK] {t} not present in state (nothing to remove)")


if __name__ == "__main__":
    main()
