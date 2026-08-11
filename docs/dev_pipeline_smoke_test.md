To smoke test in dev, the goal is not to fully reprocess everything yet. The goal is to verify that:

scripts resolve cfg.env=dev
writes go to dev/engine/v1/...
reads use dev/market/... where intended
prod is not touched
the pipeline fails clearly if dev input data is missing

Start with dry runs, then one real dev write.

0. Set env variables in your terminal

From the project root:

set ALPHA_EDGE_ENV=dev
set ALPHA_EDGE_BUCKET=alpha-edge-algo
set ALPHA_EDGE_REGION=eu-west-1

On PowerShell instead:

$env:ALPHA_EDGE_ENV="dev"
$env:ALPHA_EDGE_BUCKET="alpha-edge-algo"
$env:ALPHA_EDGE_REGION="eu-west-1"
1. Confirm runtime resolves to dev

Run this:

poetry run python -c "from alpha_edge.core.runtime import load_runtime_config; print(load_runtime_config('dev'))"

Expected shape:

RuntimeConfig(
  env='dev',
  bucket='alpha-edge-algo',
  region='eu-west-1',
  engine_root='dev/engine/v1',
  market_root='dev/market',
  warehouse_root='dev/engine/v1/warehouse',
  is_prod=False
)

If this does not print dev/engine/v1 and dev/market, stop there.

2. Check whether dev has any data yet

Run:

poetry run python - <<'PY'
import boto3

bucket = "alpha-edge-algo"
s3 = boto3.client("s3", region_name="eu-west-1")

prefixes = [
    "dev/engine/v1/trades/",
    "dev/engine/v1/cashflows/",
    "dev/engine/v1/dividends/",
    "dev/market/ohlcv_usd/v1/",
    "dev/market/snapshots/v1/",
    "dev/market/cache/v1/",
]

for p in prefixes:
    r = s3.list_objects_v2(Bucket=bucket, Prefix=p, MaxKeys=5)
    n = len(r.get("Contents", []) or [])
    print(f"{p:40s} objects_sample={n}")
PY

Interpretation:

Result	Meaning
dev prefixes have data	good, continue
dev market has no data	expected if we have not copied/ingested dev data yet
dev engine has no trades	ledger smoke test cannot rebuild real positions yet

This is important: if dev is empty, the first smoke test should prove path isolation, not full business correctness.

3. Dry-run ledger rebuild in dev

Pick a date where you know prod has activity, but dev may not.

poetry run python -m alpha_edge.operations.rebuild_ledger \
  --env dev \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --as-of 2024-01-31 \
  --prices-mode asof \
  --dry-run

PowerShell / CMD with \ works in CMD. In PowerShell use backticks:

poetry run python -m alpha_edge.operations.rebuild_ledger `
  --env dev `
  --start 2024-01-01 `
  --end 2024-01-31 `
  --as-of 2024-01-31 `
  --prices-mode asof `
  --dry-run

Expected if dev has no activity:

No activity found under dev/engine/v1/{trades,cashflows,dividends}/

That is not a bad failure. It means it is looking at dev.

Bad failure would be seeing paths like:

engine/v1/trades
market/ohlcv_usd/v1

without dev/.

4. Create a tiny dev fixture from prod

To properly test the ledger, copy a very small sample of prod data into dev. Do not copy everything yet.

Use one known trade date. Example:

poetry run python - <<'PY'
import boto3

bucket = "alpha-edge-algo"
s3 = boto3.client("s3", region_name="eu-west-1")

# CHANGE THIS to a date where you know trades exist
dates = ["2024-01-31"]

tables = ["trades", "cashflows", "dividends"]

for table in tables:
    for d in dates:
        src_prefix = f"engine/v1/{table}/dt={d}/"
        dst_prefix = f"dev/engine/v1/{table}/dt={d}/"

        resp = s3.list_objects_v2(Bucket=bucket, Prefix=src_prefix, MaxKeys=20)
        objs = resp.get("Contents", []) or []

        print(f"{table} {d}: copying {len(objs)} objects")

        for obj in objs:
            src_key = obj["Key"]
            dst_key = dst_prefix + src_key.split("/")[-1]
            s3.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": src_key},
                Key=dst_key,
                MetadataDirective="COPY",
            )
            print(f"  {src_key} -> {dst_key}")
PY

This copies only a tiny event fixture.

5. Make sure dev market data exists for the fixture assets

The ledger rebuild needs dev/market/ohlcv_usd/v1/....

For a quick smoke test, copy the relevant market partitions from prod to dev. The brute-force but safe small test is to copy only a few asset/year partitions used by those trades. First list copied dev trades:

poetry run python - <<'PY'
import boto3, json

bucket = "alpha-edge-algo"
s3 = boto3.client("s3", region_name="eu-west-1")

prefix = "dev/engine/v1/trades/"
resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=100)

asset_ids = set()
for obj in resp.get("Contents", []) or []:
    key = obj["Key"]
    if not key.endswith(".json"):
        continue
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    payload = json.loads(body.decode("utf-8"))
    aid = str(payload.get("asset_id") or "").strip()
    if aid:
        asset_ids.add(aid)

print("asset_ids:")
for aid in sorted(asset_ids):
    print(aid)
PY

Then copy the required OHLCV years. Example for 2024:

poetry run python - <<'PY'
import boto3, json

bucket = "alpha-edge-algo"
year = 2024
s3 = boto3.client("s3", region_name="eu-west-1")

# Load asset_ids from dev fixture trades
resp = s3.list_objects_v2(Bucket=bucket, Prefix="dev/engine/v1/trades/", MaxKeys=1000)
asset_ids = set()

for obj in resp.get("Contents", []) or []:
    key = obj["Key"]
    if not key.endswith(".json"):
        continue
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    payload = json.loads(body.decode("utf-8"))
    aid = str(payload.get("asset_id") or "").strip()
    if aid:
        asset_ids.add(aid)

for aid in sorted(asset_ids):
    src_prefix = f"market/ohlcv_usd/v1/asset_id={aid}/year={year}/"
    dst_prefix = f"dev/market/ohlcv_usd/v1/asset_id={aid}/year={year}/"

    r = s3.list_objects_v2(Bucket=bucket, Prefix=src_prefix, MaxKeys=100)
    objs = r.get("Contents", []) or []
    print(f"{aid}: copying {len(objs)} parquet files")

    for obj in objs:
        src_key = obj["Key"]
        dst_key = dst_prefix + src_key.split("/")[-1]
        s3.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": src_key},
            Key=dst_key,
            MetadataDirective="COPY",
        )
        print(f"  {src_key} -> {dst_key}")
PY

If the trade date is another year, change year.

6. Run the actual dev ledger smoke test

Now run without --dry-run:

poetry run python -m alpha_edge.operations.rebuild_ledger \
  --env dev \
  --start 2024-01-01 \
  --end 2024-01-31 \
  --as-of 2024-01-31 \
  --prices-mode asof

Expected output should include:

env:                  dev
bucket:               alpha-edge-algo
engine_root:          dev/engine/v1
...
[OK] Wrote:
  s3://alpha-edge-algo/dev/engine/v1/ledger/dt=2024-01-31/positions.json
  s3://alpha-edge-algo/dev/engine/v1/ledger/dt=2024-01-31/pnl.json

This is the first real pass.

7. Verify prod was not touched

Run:

poetry run python - <<'PY'
import boto3

bucket = "alpha-edge-algo"
s3 = boto3.client("s3", region_name="eu-west-1")

checks = [
    "dev/engine/v1/ledger/",
    "engine/v1/ledger/",
]

for p in checks:
    r = s3.list_objects_v2(Bucket=bucket, Prefix=p, MaxKeys=10)
    print(f"\n{p}")
    for obj in r.get("Contents", []) or []:
        print(" ", obj["Key"])
PY

You should see the new write under:

dev/engine/v1/ledger/...

The prod path may already have objects, but it should not show a new unexpected date from your dev test.

8. Then smoke test warehouse build for the same date

Once ledger dev output exists:

poetry run python -m alpha_edge.warehouse.build_warehouse \
  --env dev \
  --dt 2024-01-31 \
  --account-id main \
  --dry-run

If dry-run looks correct, run:

poetry run python -m alpha_edge.warehouse.build_warehouse \
  --env dev \
  --dt 2024-01-31 \
  --account-id main

Expected writes should go to something like:

dev/engine/v1/warehouse/...
Minimum smoke-test success criteria

You are good to continue if:

rebuild_ledger.py writes only to dev/engine/v1
rebuild_ledger.py reads market prices from dev/market
build_warehouse.py reads dev/engine/v1/ledger
build_warehouse.py writes dev/engine/v1/warehouse
prod confirmation is not required for --env dev
prod confirmation is still required for --env prod