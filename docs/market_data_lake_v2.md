# Market data lake v2 — migration plan (pilot)

## Why

Legacy `ingest_market_data.py` currently combines Yahoo acquisition, timezone/date handling,
FX conversion, USD OHLCV publication, return calculation, and snapshots.
The legacy rebuild may write log returns under legacy aliases that the wide-cache
reader interprets as simple returns. This makes failures difficult to trace.

This branch begins a **non-destructive, separate** implementation. It does not
replace `market/ohlcv_usd/v1`, `market/returns_usd/v1`, snapshots,
existing manifests, or morning pipeline orchestration.

## Data contracts

- **Bronze**: Immutable source observation, original currency, provider symbol,
  session date, original OHLC, adjusted close, volume. The raw source file is
  retained separately from Bronze validation flags.
- **Silver**: Session dates are not treated as precise UTC closing instants.
  Genuine timestamps require source timezone and explicit conversion rules.
  FX convention is **USD per one original currency unit**; reverse quotes must
  be inverted upstream and penny/cent quotes must be rescaled explicitly.
  FX is matched backward only within a configured maximum age; missing/stale
  FX quarantines the row. Raw/adjusted USD close prices are separate.
- **Gold**: Simple daily return `P_t/P_(t-1)-1`; log return
  `log(P_t/P_(t-1))`. Both are named explicitly. Gold has a review flag
  for extreme returns rather than silently clipping them.
- No source observation is overwritten to fix a data anomaly.
- No legacy consumer should read this pilot until its contracts are verified.

## First pilot commands

Input raw OHLCV (CSV/Parquet) required fields:
`asset_id,provider_symbol,currency_original,session_date,open,high,low,close,adj_close,volume`

Input FX (CSV/Parquet) required fields:
`currency_original,fx_date,fx_to_usd`

Local-only pilot (no S3 writes):

```bash
python -m alpha_edge.market.run_data_lake_v2 \
  --bars samples/source_bars.csv --fx samples/source_fx.csv \
  --output ./outputs/market_data_lake_v2
```

Inspect one asset and export CSV for plotting in Excel:

```bash
python -m alpha_edge.market.inspect_data_lake_v2 \
  --run-dir ./outputs/market_data_lake_v2/RUN_ID \
  --asset-id EQ:EXAMPLE --export-csv ./outputs/asset_trace.csv
```

Optional isolated S3 persistence only (explicit write switch):

```bash
python -m alpha_edge.market.run_data_lake_v2 \
  --bars samples/source_bars.csv --fx samples/source_fx.csv \
  --bucket alpha-edge-algo --market-root dev/market --write-s3
```

**Do not use `--market-root market`**. This tool refuses the legacy prod root.

## Acceptance gates before any daily integration

1. Provider adapter retrieves Yahoo OHLCV and FX into Bronze **before any
   normalization**, with ingestion batch IDs, source timezone, original quote
   units, provider versions, and reproducible response metadata.
2. Exchange calendar coverage and closed-candle checks, including holidays,
   half-days, US/EU/Asia/crypto and FX calendars.
3. Strict duplicate conflict resolution policy (raw duplicates retained,
   canonical observation selected by an explicit approved rule).
4. Corporate-action/split/dividend reconciliation, historical ticker/asset
   mapping validation, and the distinction between total-return analytics
   and execution/ledger prices.
5. Return calculation must not bridge missing expected sessions invisibly.
6. Independent stage reconciliation, quarantine reports and exception review.
7. Atomic publication model with immutable run manifests, checksums, validated
   candidate snapshot, and one controlled current-version pointer. Retrying
   a partial run must never silently adopt a partially staged output.
8. Asset time-series explorer with per-observation lineage, date zoom, raw
   versus silver/gold comparison, anomaly drill-down.
9. Historical backfill in a staging prefix; compare legacy versus v2 for
   representative assets and full universe before switching readers.
10. Run unit/integration tests and CI, then enable shadow-mode morning
    orchestration, then controlled consumer migration and rollback.

## Proposed work breakdown

- PR 1: Transformations, contracts, test fixtures, pilot run manifests and
  trace export (**this branch**).
- PR 2: Yahoo Bronze ingestion and canonical source revision handling.
- PR 3: Exchange/session, FX calendar, corporate-action, and quality gates.
- PR 4: Versioned Silver/Gold publisher, schema registry, compatibility
  adapter and immutable promotion pointers.
- PR 5: Read-only explorer, asset trace API and anomaly review workflow.
- PR 6: Morning orchestration, backfill, shadow comparisons, alerts,
  fail-fast gates and documented rollback.

**Current state:** PR 1 is a pilot only. No live Yahoo fetching, production
S3 pipeline promotion or daily pipeline integration is claimed.
