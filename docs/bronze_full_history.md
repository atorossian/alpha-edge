# Bronze — full-history Yahoo Finance ingestion (clean slate)

This is one independent, executable Bronze stage. It **does not** reuse previous
market prices, v1/v2 lake data, backfills, release inventories or historical
identity crosswalks. It reuses the Yahoo provider parsing from
`ingest_bronze_yahoo_v2.acquire_yahoo_bars` and the original
`ingest_market_data.py` operational lessons (rate limiting, retries, per-symbol
processing). It does **not** import `ingest_market_data.py`, which creates a
local cache on import, converts FX, calculates returns and writes legacy paths.

**Runtime:** Python 3.10, Poetry, boto3 and Yahoo Finance via yfinance.
The script runs from Git Bash on Windows, but all market data remains in
memory until written directly to S3. There are no local Parquet/CSV outputs.

## The only input universe

Reads precisely these five immutable S3 objects:

```text
s3://alpha-edge-algo/market/lake_v3/config/source_snapshots/
  f0089502a02a8d13d4c162bec31248595aa939935c6507478f29e603802b34a6/
    universe.csv
    universe_overrides.csv
    asset_excluded.csv
    ingest_force_refresh.csv
    universe_manual_additions.csv
```

These were captured from pinned Git commit
`cfb36a1deb4eaa58615018702cdf4cbee6a79cf2`.
Every file is checksum-verified against the snapshot metadata. The
five-file metadata fingerprint is verified before any write.

Selection: exactly `include == 1` from `universe.csv`: **2,035**
instruments out of 2,207 (172 excluded). Overrides and manual additions
are cross-checked; the resolved `yahoo_ticker` in the universe is the
provider symbol used for acquisition. Manual Yahoo-symbol conflicts (e.g.
EUR-USD vs EURUSD=X) are recorded as warnings, not replaced using
historical data. Rows missing a currency hint are still attempted:
Yahoo's live metadata may supply it. No automatic USD assumption.

**This is a source snapshot, not live configuration.** If any of the five
files changes, capture a new immutable snapshot and use its new
fingerprint in a new run. An earlier run can always resume from its exact
source snapshot.

## What is written into Bronze

Each included asset is requested independently using
`yfinance.Ticker(symbol).history(period="max", end=END, interval="1d",
auto_adjust=False, actions=True, repair=False)`.

Yahoo's `Close` may already be split-adjusted by the provider;
the script does **not** apply splits or dividends, does **not** normalize
timezones into UTC session dates, does **not** convert FX/quotes into USD,
does **not** calculate returns, and does **not** fill missing bars.
It stores source OHLCV, Yahoo `Adj Close`, dividends, stock splits,
original source timestamp/timezone labels and request provenance.

Every successful asset writes an immutable, content-hashed Parquet object
and then a per-asset committed manifest. A manifest is never written
before the referenced data has reached S3. Failure events and run reports
are stored separately in S3. Incomplete asset objects without a manifest
are not considered committed.

S3 layout:

```text
market/lake_v3/bronze/full_history/runs/<run-id>/
  run.json                                  # immutable source selection
  assets/<sha256(asset_id)>/
    objects/<sha256(parquet)>.parquet        # immutable Yahoo source bars
    manifest.json                           # committed asset receipt
    failures/<timestamp-and-uuid>.json      # retry/failure evidence (if any)
  reports/<timestamp-and-uuid>.json          # counts and failure details
```

A different source response or new ingestion end date requires a **new run
ID**. The same run ID is resumable: committed assets are skipped after
checking object identity and size/hash metadata. An explicit `verify`
reads every completed Parquet object back into memory, verifies its
SHA256 and row count, and reports missing or corrupt assets.

## 1. Checkout the new implementation (Git Bash)

```bash
git fetch origin
git switch feature/bronze-full-history-s3
git pull --ff-only origin feature/bronze-full-history-s3
poetry install
```

Do **not** run the legacy `alphaedge-ingest` command for this Bronze stage.

## 2. Pilot: eight edge-case assets (same exact code path)

```bash
poetry run python -m alpha_edge.market.ingest_bronze_full_history ingest \
  --bucket alpha-edge-algo \
  --source-fingerprint f0089502a02a8d13d4c162bec31248595aa939935c6507478f29e603802b34a6 \
  --run-id bronze-pilot-20261116 \
  --end 2026-11-16 \
  --only SPY,GLD,BTC-USD,RENT3,TRAXIONA,LOG,REY,EUR-USD
```

**First invocation without `--execute` is read-only.** Append
`--execute` to actually download and store the eight assets.
The `--end` date is an exclusive boundary and prevents including a
still-forming daily candle. This particular run ends before the
2026-11-16 market sessions. For a later run, choose the actual
next suitable UTC date and a **different run ID**.

Verify the pilot independently:

```bash
poetry run python -m alpha_edge.market.ingest_bronze_full_history verify \
  --bucket alpha-edge-algo \
  --source-fingerprint f0089502a02a8d13d4c162bec31248595aa939935c6507478f29e603802b34a6 \
  --run-id bronze-pilot-20261116
```

## 3. Full universe: 2,035 included assets

**One command for all included instruments (no `--only`):**

```bash
poetry run python -m alpha_edge.market.ingest_bronze_full_history ingest \
  --bucket alpha-edge-algo \
  --source-fingerprint f0089502a02a8d13d4c162bec31248595aa939935c6507478f29e603802b34a6 \
  --run-id bronze-full-20261116 \
  --end 2026-11-16 \
  --execute
```

The output includes the S3 run-report URI and the number of successes,
pre-existing commitments and failures. Failures are **not hidden**:
the command exits with status 2 if any asset failed, even though
successful records are preserved. Individual failed assets may need a
correction to their Yahoo ticker or manual review.

If interrupted or if Yahoo errors occur, **rerun the exact same
command**. Assets already committed are skipped; missing/failed assets
are tried again. Do not change the source fingerprint, end date,
asset selection or run ID while resuming. Use `--retries 2` and
`--pause-seconds 0.5` as defaults; increase pause if rate-limited.

## 4. Independently validate all written Bronze objects

```bash
poetry run python -m alpha_edge.market.ingest_bronze_full_history verify \
  --bucket alpha-edge-algo \
  --source-fingerprint f0089502a02a8d13d4c162bec31248595aa939935c6507478f29e603802b34a6 \
  --run-id bronze-full-20261116
```

**Bronze is complete only when:**
- `expected=2035`, `committed=2035`, `missing=0`, `corrupt=0`,
  `state="pass"`.
- Each asset has an immutable S3 Parquet object, SHA-256 and row count,
  explicit observed historical start/end session, and source metadata.
- Any source gaps, duplicates, stale values or suspicious prices are
  preserved as observed and flagged for independent quality review.
- No legacy production paths or local market data are written.

**Important:** passing Bronze object integrity does not prove Yahoo's
observations economically accurate or a valid calendar complete.
That independent market-data quality check is the next stage.
A ticker genuinely unsupported by Yahoo will remain a failed asset;
do not silently omit it merely to manufacture a 100% result.

## CLI controls

- `--execute`: permission to make S3 writes (absent = dry-run).
- `--only`: asset IDs or unambiguous ticker selectors for pilots.
- `--end`: exclusive UTC date; choose a completed session window.
- `--retries`: retry count *after* the initial Yahoo attempt (default 2).
- `--pause-seconds`: pacing between per-asset requests (default 0.5).
- `verify`: read-only, expensive full-object SHA verification; performs no writes.

Required AWS credentials: permission to read the five snapshot objects,
and `s3:PutObject`, `s3:GetObject`, `s3:HeadObject` on
`market/lake_v3/bronze/full_history/`. No write permission to
`market/*/v1`, v2 datasets, trades or ledger is required.

The `data/universe/*.csv` files are NOT accessed locally by this command.


## Visible progress and audit trail (compatible with existing runs)

The CLI now prints a startup summary, the first processed asset, and a
progress line every ten selected assets (or at completion), in the spirit
of `ingest_market_data.py`:

```text
=== BRONZE FULL HISTORY INGEST ===
run_id:          bronze-full-20261116
included_total:  2035
selected:        2035
[bronze][progress] done=1/2035 (0.0%) ok=1 resumed=0 failed=0 rate=... assets/s elapsed=...m
[bronze][fail][1]  ... asset_id=... yahoo=... error=...
[bronze][progress] done=10/2035 (0.5%) ok=... resumed=... failed=... rate=... assets/s elapsed=...m
```

These are **illustrative format examples**, not a live ingestion report.
All counters reflect actual processed assets. `resumed` counts existing
successful S3 commits skipped during this invocation; it is not the number
of provider downloads performed. Console output is flushed immediately.

New options:

- `--progress-every 10`: periodic progress interval (use `1` to print
  every asset).
- `--print-first-failures 20`: number of individual failure messages
  printed; all failures remain recorded in S3.
- `--verbose-yahoo`: restore raw yfinance diagnostics when investigating
  specific ticker failures. By default the script suppresses the
  misleading/repeated provider logging and prints the classified
  final per-asset exception instead.

Every `--execute` invocation writes **append-only** S3 audit events:

```text
market/lake_v3/bronze/full_history/runs/<run-id>/audit/dt=YYYY-MM-DD/
  bronze_<timestamp>-<uuid>.json
```

The audit event's fields follow Alpha Edge's established
`build_dataset` pattern: audit ID, run ID, entity type/asset ID,
script, status, source fingerprint, asset provider symbol, output keys,
observed metrics and errors. Events cover invocation start, each asset's
success/already-committed/failure, normal run completion, and
`KeyboardInterrupt` interruption. This audit stays under the **new
Bronze S3 namespace**, not the old engine/v1 audit paths. An
interrupted run retains its earlier committed manifests and audit
records.

**If the previous command is still executing:** the running Python
process will not automatically load code changes. Stop it with
`Ctrl+C`, then update this branch:

```bash
git pull --ff-only origin feature/bronze-full-history-s3
```

Rerun **the exact same** `ingest` command, maintaining
`--run-id bronze-full-20261116`, the same `--end` and source
fingerprint. Existing successful S3 manifests will be skipped, and
the newly started invocation will emit progress and new audit events.
Don't run two copies against the same run ID concurrently.

If S3 auditing itself fails, the script reports an exception rather than
silently declaring un-audited work complete. For any pre-update assets
that already committed successfully, the original immutable S3
manifests remain the authoritative ingestion evidence.


## Post-run remediation — 2026-11-17

The original run `bronze-full-20261116` is immutable and remains a
correctly preserved historical acquisition attempt:
`expected=2035`, `committed=2026`, `missing=9`, `corrupt=0`.
Its five-file source fingerprint is pinned; changing Git-tracked
configuration **does not change that snapshot or run's included assets**.

Updated only the five authoritative current universe CSV files on this
feature branch:

- `ATKR`, `NWE`, `ACVA`: `include=0`, with explicit acquisition/
  merger exclusion reasons; keep their stable IDs and any prior data.
- `BKH`: retain `include=1` and the existing stable asset ID, but
  resolve the **current Yahoo symbol** to `HZN`. Do not join the former
  NWE source history or assume an economically continuous BKH/HZN price
  series without verifying the corporate action and historical provider
  semantics.
- `USD-CNY`, `USD-BRL`, `USD-JPY`: resolve Yahoo tickers to
  `CNY=X`, `BRL=X`, `JPY=X`, with quote currencies
  `CNY`, `BRL`, `JPY`, respectively. Keep source pairs and FX
  conversions separate; Bronze records raw provider quote values only.
- `EMBJ`: leave active and **do not apply or reverse a split in
  Bronze**. Preserve the Yahoo splits/actions column for inspection
  (reported 4:1 ADS split effective November 16, 2026).
  Live data acquisition should be verified separately; seeing prices
  in the Yahoo UI does not itself prove the API response will work.
- `PSKY`: **deferred** pending a separate instrument review. No change
  to its included status or identity has been made.

**Important:** after deciding PSKY and approving the updated config,
capture a **new immutable S3 snapshot** from the five edited files,
generate a **new run ID**, and ingest the unresolved eligible assets
using the new fingerprint. The old run cannot be resumed with different
ticker mappings or new eligibility flags. Never edit or overwrite the
old run's S3 manifests and raw Parquet observations. Reconcile the old
and new per-run asset receipts explicitly before judging overall Bronze
coverage. The snapshot and next-run command must be generated using a
tested version-controlled script, not an improvised direct S3 upload.
