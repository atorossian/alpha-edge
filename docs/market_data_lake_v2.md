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

## Stage 2: Yahoo Bronze acquisition (initial implementation)

Module: `alpha_edge.market.ingest_bronze_yahoo_v2`.
This is a **standalone shadow ingestion pilot**, not a replacement of
`ingest_market_data.py`. The legacy AEA-87 repair and publishing jobs remain
unchanged. No historical data has been modified.

The Stage-2 source contract records:

- Original OHLC, adjusted close, volume and Yahoo dividend/split columns
  without USD conversion, return calculation, price repair or deduplication.
- Original index timestamp representation, source timezone, local session date,
  provider symbol, exchange, reported original currency and batch metadata.
- Unmodified Yahoo FX quote prices (`quote_close`) under the requested
  provider symbol; these are **not yet** the Silver `fx_to_usd` rates.
- Immutable local per-run Parquet data plus manifest and SHA-256 checksums.
  Optional S3 output is under `dev/market/lake_v2/bronze/batches/<batch-id>`
  by default. The legacy `market/` root is refused.

Run a small asset sample using a CSV with `asset_id,yahoo_ticker` and optional
`currency` / `include` columns:

```bash
python -m alpha_edge.market.ingest_bronze_yahoo_v2 \
  --universe-csv ./samples/bronze_universe.csv \
  --fx-symbols 'EURUSD=X;USDEUR=X' \
  --start 2026-10-01 --end 2026-10-31 \
  --output ./outputs/market_bronze_v2
```

To stage to S3 as well, add `--upload --bucket alpha-edge-algo
--market-root dev/market`. Use a unique batch ID. The process fails the
entire batch on a failed provider response or missing currency; this is
intentional for the initial small-pilot contract. It must evolve to preserve
per-symbol failures and successful responses independently for large universes.

**Important source caveats:** Yahoo/yfinance does not expose exact raw HTTP
payload bytes via this adapter. Parquet contains the original numerical
values returned by yfinance plus the index time representation, not a
byte-for-byte provider payload. Yahoo currency units such as GBp versus GBP
require an explicit future Silver quote-unit normalization policy; do not
treat them as equivalent. These new raw FX quote tables must not be passed
directly to `run_data_lake_v2`, which currently requires already oriented
USD-per-unit FX values. A validated FX normalization adapter is still needed.

Run tests:
```bash
pytest tests/unit/test_ingest_bronze_yahoo_v2.py tests/unit/test_data_lake_v2.py -q
```

**Remaining before Stage 2 production acceptance:**
- Implement Yahoo retry/rate limiting and per-asset response failure manifests,
  reusing the tested AEA-87 mechanisms where appropriate.
- Extend the ingest metadata with explicit native quote-unit conventions,
  provider/library version and source exchange timezone confidence.
- Introduce object immutability enforcement / conditional writes for S3
  and robust orphaned-batch cleanup. A manifest-last publication on its own
  is not a multi-object atomic transaction.
- Validate full-history run performance and compare with existing AEA-87
  source extracts; do not enable daily scheduling yet.

## Stage 3: Exchange-session and FX standardization candidate

The Stage-3 modules are `standardize_bronze_v2.py` and
`run_standardization_pilot_v2.py`. They consume the checksummed Bronze
batch and **never download data**; output is a local, versioned candidate
directory with a manifest written last. They do not write S3 or modify
legacy `market/ohlcv_usd/v1` and `market/returns_usd/v1`.

Run on the six-asset Bronze batch:

```bash
python -m alpha_edge.market.run_standardization_pilot_v2 \
  --bronze-batch-dir ./outputs/market_bronze_v2/six_asset_pilot \
  --output ./outputs/market_silver_gold_v2/six_asset_pilot
```

The report includes source checksum verification, Bronze/Silver/Gold row-count
reconciliation, FX diagnostics, missing/unexpected exchange sessions,
quarantine counts, and explicit arithmetic/log returns.
Each asset receives a CSV trace with source price, original currency,
session date, source timestamp UTC (if offset-aware), FX quote symbol,
direction, original quote, derived USD price and quality flags.

Pilot FX pairs are **explicitly** mapped: EURUSD=X (EUR→USD direct),
USDBRL=X (USD→BRL inverse) and USDMXN=X (USD→MXN inverse). Never infer
direction from a string or apply a quote before specifying its unit.

Pilot exchange calendars: NYSE for GLD/SMH/SPY, BMF for B3SA3.SA,
XHEL for KESKOB.HE, and BMV for ALSEA.MX. Calendar checks cover
expected exchange sessions **between first and last observed dates**.
That is not yet a request-window completeness guarantee, nor proof of
a source candle being fully closed.

Stage-3 tests: `pytest -q tests/unit/test_standardize_bronze_v2.py`.
The default-branch manual launcher now runs Stage 1–3 tests, re-acquires
six assets, builds the candidate and archives its outputs to GitHub Actions.
Results must be reviewed before promotion.

**Pending production correctness controls:**

- Validate official calendar identifiers and holiday schedules against real
  pilot evidence. Fail closed on an unknown calendar.
- FX daily-bar fixing times are not necessarily synchronous with the
  underlying security close. Define a valuation cutoff policy and
  prove no look-ahead across markets. The current same-session as-of
  matching is a diagnostic candidate, not a point-in-time trading feed.
- Implement multi-day gap classification, corporate action consistency
  (including dividends/splits), and adjusted-close provenance checks.
- Support direct/inverse quote cross-checks and disallow corrupt FX values
  from subsequently entering the rate selection; Stage-3 currently uses
  one requested Yahoo quote direction per currency.
- Extend missing-session diagnostics to the requested start/end range and
  define eligibility rules for current partial candles.
- Add staging promotion, AEA-87 historical reconciliation and operational
  backfill before integrating the morning pipeline.

No claim of successful CI execution or real-data Stage-3 acceptance is made
until the workflow is rerun and its candidate report reviewed.

## Manual market observation corrections (pilot, 2026-11-10)

A user-supplied observation for ALSEA.MX on 2026-04-22 is maintained as
`data/quality/market_lake_v2/manual_ohlcv_corrections.csv`:

| Field | Value |
| --- | --- |
| Asset ID | EQHd491d3c90f3be544 |
| Symbol | ALSEA.MX |
| Currency | MXN |
| Session | 2026-04-22 |
| Open | 51.50 |
| High | 52.06 |
| Low | 51.10 |
| Close | 51.80 |
| Volume | 140975 |
| Adjusted close | **unknown / not supplied** |
| Verification | **pending independent verification** |
| Provenance | user-supplied historical trading observation; original source link not yet provided |

The original Yahoo Bronze batch is never edited. To overlay corrections into a
Silver/Gold candidate while preserving the original Bronze checksum:

```bash
python -m alpha_edge.market.run_standardization_pilot_v2 \
  --bronze-batch-dir ./outputs/market_bronze_v2/six_asset_pilot \
  --corrections-csv data/quality/market_lake_v2/manual_ohlcv_corrections.csv \
  --output ./outputs/market_silver_gold_v2/six_asset_pilot
```

The overlay is strictly **append-only**: an existing Yahoo asset/session
observation cannot be overwritten; identity and OHLC ranges are checked.
The source CSV hash and row-level correction evidence are included in the
candidate manifest and `manual_corrections_audit.parquet`.

The missing exchange-calendar date is covered by the manual observation,
but its adjusted close remains unknown. This means its adjusted USD close
and the neighboring adjusted-return calculation must remain unavailable;
**do not set adjusted close equal to raw close or manufacture a corporate-action
adjustment factor**. The candidate is still `review_required`. Before promoting
to Gold, reconcile the user-supplied OHLCV against BMV or another independently
trusted primary record, confirm quote/volume units and obtain a supported
adjusted-close calculation policy, with documented corporate-action evidence.

The default-branch GitHub Actions pilot launcher includes this correction
overlay, and its offline unit test covers append-only behavior and the
unavailable adjusted-return boundary. The production daily ingestion and AEA-87
historical repair pipelines remain unchanged.

## Stage 4: Immutable, versioned Silver/Gold candidate publication

Module: `alpha_edge.market.publish_lake_v2`. Implemented as **opt-in
shadow publication** only; the default CI launcher runs local verification
without AWS credentials or network writes.

The candidate manifest records the Bronze manifest hash, row counts,
manual-correction status, every artifact's SHA-256 checksum, and acceptance
gates. Local verification rejects missing files, altered hashes, invalid
schemas, or row-count mismatches.

```bash
poetry run python -m alpha_edge.market.publish_lake_v2 verify-local \
  --candidate-dir ./outputs/market_silver_gold_v2/six_asset_pilot
```

Optional isolated staging (requires explicit AWS credentials and invocation;
not performed by CI):

```bash
poetry run python -m alpha_edge.market.publish_lake_v2 stage \
  --candidate-dir ./outputs/market_silver_gold_v2/six_asset_pilot \
  --bucket YOUR_DEV_BUCKET --run-id six-asset-v2-001 --root dev/market

poetry run python -m alpha_edge.market.publish_lake_v2 verify-staged \
  --bucket YOUR_DEV_BUCKET --run-id six-asset-v2-001 --root dev/market
```

Staged objects live solely under
`dev/market/lake_v2/candidates/<run-id>/`. Each artifact is written with
a conditional create; original bytes are re-read and verified, then
`_committed.json` is written last. A partial upload is never committed,
and a retry must use a new run ID. Do not infer S3 multi-object atomicity
from this protocol: consumers must require the final marker.

Optional **dev-only** promotion is separate and requires a manually reviewed
candidate-manifest SHA-256:

```bash
poetry run python -m alpha_edge.market.publish_lake_v2 promote-dev \
  --bucket YOUR_DEV_BUCKET --run-id APPROVED_RUN_ID \
  --root dev/market --approval REVIEWED_MANIFEST_SHA256
```

Promotion writes only `dev/market/lake_v2/current.json`, using a
conditional update. It cannot modify any production reader pointer.
It rejects a candidate with missing/unexpected sessions, quarantined
prices, review returns, rejected FX, missing acceptance, or any manual
correction. **The current ALSEA.MX correction is pending independent
verification, so its candidate is not eligible for promotion.**

Remaining items before a production-grade publisher:
- Formal provenance sign-off, fully adjusted-close policy and exchange-
  session evidence for the ALSEA correction.
- Remote S3 IAM / bucket policy tests (least-privilege write paths),
  versioning, incomplete-batch garbage collection and disaster recovery.
- Reader-side immutable manifest/pointer protocol, rollback drill, golden-
  dataset reconciliation and consumer compatibility tests.
- End-to-end CI dry run against an isolated AWS sandbox only after separate
  approval; no production writes or dev pointer update in routine CI.

## Stage 5: Read-only Data Quality Explorer (initial implementation)

Module: `alpha_edge.market.data_quality_explorer_v2`.

The Explorer renders a **self-contained, local HTML snapshot** from a
Silver/Gold candidate after checking the candidate's immutable manifest and
SHA-256 hashes. It performs no S3 writes, starts no server, needs no login,
and never retrieves live prices. No external JavaScript libraries are loaded.
Run:

```bash
poetry run python -m alpha_edge.market.data_quality_explorer_v2 \
  --candidate-dir ./outputs/market_silver_gold_v2/six_asset_pilot \
  --output ./outputs/market_explorer_v2/index.html
```

Open `index.html` in a browser. The report provides: an asset selector,
date-range filters, source/adjusted/USD price and simple/log-return charts,
anomalies and calendar diagnostics, FX rejection evidence, a per-observation
drill-down (including correction provenance), a CSV export and manifest
checksum identity. Null values are left as gaps; the ALSEA.MX user-supplied
raw close remains visible without fabricating an adjusted close.

GitHub Actions' six-asset launcher builds the HTML report into the downloadable
`market-lake-v2-six-asset-pilot` artifact. Its unit tests include checksum
tamper detection, null handling and HTML script-injection safety.

**Current scope:** 100,000 rows maximum in a single static snapshot. This
is a pilot explorer for locally verified candidate datasets, not a persistent
asset service or production-facing website. It does not yet fetch arbitrary
historical revisions from S3 or independently compare AEA-87 and lake-v2
history: historical reconciliation is a separate milestone. Source and
corporate-action verification are still required for manual corrections.

## Historical reconciliation: AEA-87 versus Lake v2 (read-only)

Module: `alpha_edge.market.reconcile_aea87_lake_v2`.

The reconciler compares candidate Gold against **explicitly supplied**
local AEA-87 history extracts; it does not fetch or write S3 objects.
It requires the legacy price series and return series, each containing
`asset_id`, a `date` or `session_date` session label, and their
corresponding values:

- prices: `close_adjusted_usd`
- returns: `ret_close_adjusted_usd`

Input CSV or Parquet files must be exported from the AEA-87 repaired
dataset and retain asset IDs/dates without reinterpretation. The tool
verifies the lake-v2 candidate manifest and records SHA-256 hashes of
both supplied AEA-87 files. Comparison is a full outer join on
`(asset_id, session_date)`, not a potentially lossy inner join.

Example:

```bash
poetry run python -m alpha_edge.market.reconcile_aea87_lake_v2 \
  --candidate-dir ./outputs/market_silver_gold_v2/six_asset_pilot \
  --legacy-prices ./evidence/aea87_prices.parquet \
  --legacy-returns ./evidence/aea87_returns.parquet \
  --legacy-return-kind unknown \
  --output ./outputs/aea87_v2_reconciliation/six_asset_pilot
```

Outputs: `session_differences.csv`, `asset_reconciliation_summary.csv`
and `manifest.json`. The output directory is created once, with no
overwrite. Daily session presence, missing/invalid adjusted USD prices,
return comparison eligibility, price/return discrepancies, quarantines
and manual corrections are recorded separately.

**Crucial return convention:** AEA-87 legacy returns may have been
stored as log values under an ambiguously named `ret_close_adjusted_usd`
column. The default `--legacy-return-kind unknown` deliberately
marks returns NOT COMPARABLE rather than assuming arithmetic returns.
Use `simple` or `log` only after independently verifying the specific
legacy extract's provenance. Do not infer the return kind from a ticker,
filename or the numeric values alone.

No AEA-87 source extracts have been supplied to this implementation yet:
the unit tests exercise synthetic fixtures only. Genuine historical
equivalence is **not** claimed and the current ALSEA.MX manual
adjusted-close gap remains an explicit review item.

Next: obtain read-only AEA-87 extracts for the same six asset IDs and
date interval, run the reconciler, review differences and add its
results as a separate comparison view in the Explorer.

## Existing corporate-actions reference: read-only lake-v2 audit

`build_corporate_actions.py` already stores Yahoo split observations through
`MarketStore.write_corporate_actions_partitioned()` to
`market/reference/v1/corporate_actions/asset_id=.../part-00000.parquet`.
The reference dataset is **not** a price adjustment engine.

Use the new isolated audit to inspect that existing S3 reference dataset
against the immutable six-asset Gold candidate. It checks for missing versus
corrupt reference partitions, stores per-partition SHA-256 provenance, and
flags changes in the Yahoo `adj_close / close` ratio over time. It does not
modify Gold, overwrite S3, or infer that an adjustment factor is justified.

```bash
poetry run python -m alpha_edge.market.audit_corporate_actions_v2 \
  --bucket alpha-edge-algo \
  --region eu-west-1 \
  --candidate-dir outputs/pilot_artifact/market_silver_gold_v2/six_asset_pilot \
  --output outputs/corporate_action_audit_v2/six_asset_pilot
```

Outputs: `corporate_action_sources.csv`, `corporate_actions.csv`,
`adjustment_ratio_flags.csv`, and hash-checked `manifest.json`.
A missing partition means **reference unavailable**, not proof of no action.
Corrupt or unreadable existing partitions cause the audit to fail.

**Limitations:** legacy corporate actions contain splits only, not rights
issues or cash dividends. A constant common multiplier in both Yahoo
`close` and `adj_close` (as observed for some KESKOB.HE sessions)
cannot be detected using the adjusted/raw ratio. Compare immutable source
snapshots and obtain independent corporate-action evidence before correcting
prices. This is a supplemental, manually executed read-only audit;
it does not turn the existing split table into a mandatory production gate.

## Milestone: versioned corporate-actions acquisition and audit correlation

This milestone adds an **additive, local-only immutable reference candidate**;
the legacy corporate-actions S3 paths are preserved untouched.

1. Acquire Yahoo \`Dividends\` and \`Stock Splits\` with source session
   timestamps, currency/value units, retrieval time and ingestion batch IDs.
2. Preserve all reported events (including those below the ratio-alert
   threshold), including asset IDs with **zero** reported events in the batch
   manifest. The manifest pins the exact asset universe and event parquet hash.
3. Verify batch identity and hashes before correlating event dates with Gold
   \`Adj Close / Close\` ratio-change alerts.
4. Emit matched/unmatched flags as **diagnostic evidence only**, not automatic
   acceptance. Same-day dividend does not prove the provider's adjustment
   calculation; rights issues and simultaneous scaling of raw and adjusted
   prices are explicitly unhandled.

Use the six-asset pilot universe, which must include known currency hints:

\`\`\`bash
poetry run python -m alpha_edge.market.ingest_corporate_actions_v2 \
  --universe-csv data/quality/market_lake_v2/pilot_six_assets.csv \
  --start 2025-01-01 \
  --end 2026-11-10 \
  --output outputs/market_corporate_actions_v2
\`\`\`

The acquisition prints its new \`local_path\` (timestamped batch directory).
Use that exact path in the local correlation:

\`\`\`bash
poetry run python -m alpha_edge.market.reconcile_corporate_actions_v2 \
  --candidate-dir outputs/pilot_artifact/market_silver_gold_v2/six_asset_pilot \
  --actions-dir outputs/market_corporate_actions_v2/<batch-id> \
  --output outputs/corporate_action_event_matches_v2/six_asset_pilot
\`\`\`

No S3 permissions are required for this step. The two candidate filesets have
separate manifests, hashes and provenance. The existing split-only S3 audit is
still available via \`audit_corporate_actions_v2\`.

### Market Data Lake v2 roadmap snapshot (November 10, 2026)

| Workstream | Current state | Remaining gate |
| --- | --- | --- |
| Canonical asset universe and source identifiers | Six-asset pilot implemented | Full-universe contract and scaling |
| Yahoo Bronze raw acquisition and provenance | Implemented and CI-piloted | Incremental and multi-source support |
| Silver exchange-session and FX normalization | Implemented in pilot | Explain BRL/MXN 228 shared FX mismatch dates and verify timezone source |
| Silver price/corporate-action consistency | Split reference audit and dividends/splits candidate implemented | Correlate real batch, rights-issue source evidence, detect provider revisions |
| Gold USD adjusted prices and returns | Implemented in pilot | Verify return conventions and approve quality gates |
| Manual correction, quarantine and review | Implemented; ALSEA adjusted close still unknown | Resolve manual provenance before approval |
| Historical AEA-87 reconciliation | Run locally: 2,758 shared prices, 32 additional v2 sessions | Resolve 917 mismatches across three foreign-currency assets |
| Explorer and CI | Implemented; six-asset workflow generated artifact | Surface new reference correlations and validate updated CI |
| Isolated dev publication | Opt-in publisher implemented | Production acceptance and rollback runbook |
| Full production cutover and daily processing | Not started | End-to-end incremental orchestration, versioning, compatibility and monitoring |

**Important:** the 16 ratio flags were each on a reported dividend date in the
manual Yahoo diagnostic; this is not yet an independently automated reconciled
batch result. KESKOB.HE October 28–November 4 raw-price scaling remains
unexplained. Do not promote the six-asset candidate to production.

## Milestone: independent data integrity and acceptance (v1 not authoritative)

New standalone module:
\`alpha_edge.market.independent_acceptance_v2\`. This is a **local,
read-only shadow assessment**, intentionally excluding AEA-87/v1 from the
acceptance decision. Source records and stored transformation results are
assessed against their own immutable Bronze and candidate SHA-256 manifests,
not legacy prices.

The module checks:
- Exact source batch ID and manifest checksum, and Bronze file integrity.
- No loss or mutation of source OHLCV observations in Gold (except separate
  explicitly disclosed manual additions); one asset/session key per stage.
- Silver/Gold observation-column identity; independent computation of USD raw
  and adjusted prices, with rejected rows masked.
- Direct/inverted FX arithmetic against preserved Bronze quote values; selection
  of the latest accepted quote at-or-before the exchange session label.
- Simple and logarithmic daily return arithmetic, quarantine chain breaks and
  review statuses.
- Separate **unresolved evidence** counts for unverified manual corrections,
  quarantines, exchange-calendar issues, extreme returns and rejected FX rates.
- Optional exact-universe, checksum-validated dividend/split acquisition batch.

The output has \`internal_integrity_status\` and an intentionally more
conservative \`acceptance\`. Even a fully consistent candidate remains
\`review_required\`, **not production eligible**, until independent provider
source observation verification, external corporate-action adjustment factors,
historical source revision comparisons, FX timing policy, and endpoint
completeness are evidenced. The existing publisher is unchanged: no automated
production promotion is introduced.

First locate the original immutable Bronze pilot under the extracted artifact
(with \`find outputs/pilot_artifact -name source_ohlcv.parquet\`). The
\`--bronze-dir\` argument must contain \`source_ohlcv.parquet\`,
\`source_fx_quotes.parquet\` and \`manifest.json\`, and its manifest digest
must equal the source digest listed in the Silver/Gold candidate.

\`\`\`bash
poetry run python -m alpha_edge.market.independent_acceptance_v2 \
  --candidate-dir outputs/pilot_artifact/market_silver_gold_v2/six_asset_pilot \
  --bronze-dir <path-to-extracted-original-bronze-batch> \
  --actions-dir outputs/market_corporate_actions_v2/20261110T193325Z-9fc45259 \
  --output outputs/independent_acceptance_v2/six_asset_pilot
\`\`\`

The local output contains \`integrity_findings.csv\` and
\`manifest.json\`; neither source nor S3 is modified. If the exact Bronze
source batch is not available, the tool refuses to assess data rather than
substituting another provider snapshot. AEA-87 comparisons remain optional
migration-impact diagnostics, not correctness gates.

### Roadmap position

Bronze acquisition, Silver normalization, Gold returns, quality explorer,
manual quarantine and corporate action evidence have pilot implementations;
the **current phase** is proving independent correctness. Pilot acceptance,
provider revision monitoring, formal rights-issue handling, full-universe
incremental ingestion and controlled production cutover remain pending.
