# Alpha Edge market lakehouse v3 (S3-only)

## Scope and maturity

This is an **isolated, review-gated implementation**, not a production cutover.
Legacy `market/*/v1`, `market/lake_v2`, portfolio consumers and v2 immutable
evidence are deliberately untouched.

**No persistent local market-data files:** every dataset, transformation artifact,
candidate, Athena result and final Iceberg file is stored in S3. ETL uses
in-memory pandas and PyArrow. GitHub holds only code/tests/documentation.
Tests use `moto` simulated S3 and in-memory fixtures.

### Dependency graph

1. **S3 config registry:** immutable asset ID, provider symbol, source currency,
   exchange calendar, explicit FX quote direction. Registry revision SHA is pinned.
2. **Bronze:** source OHLCV, source dividends/splits, source FX, Yahoo provenance,
   SHA-256 artifact references. A final manifest makes an ingestion batch readable.
3. **Silver:** reuse v2 source contracts, session-calendar gap check,
   source-date preservation, validation, direct/inverse FX normalization, bounded
   prior FX join and USD conversion; retain quarantined observations.
4. **Gold candidate:** reuse v2 adjusted-price/return calculations and existing
   indicator engine. Keep source price and total-return series distinctly named;
   calculate USD price/total/log returns, indicators and quality reports.
5. **Publication:** external reviewer signs an exact candidate SHA and independent
   evidence gates. Athena reads only `candidates/<id>/gold/`, builds a new
   release-specific Iceberg table, verifies row and key counts, then switches
   `alpha_edge_market.market_asset_daily` to the certified release.
6. **Consumers:** query the stable Athena view; past releases remain queryable
   as immutable Iceberg tables. Do not use unapproved dev candidates.

This structure is a **Gold warehouse layer**, not a separate fourth physical
copy. The Parquet Gold staging copy may be lifecycle-managed after verification
and retention-policy approval; it is not automatically deleted.

## Contract definitions

- `asset_id`: immutable internal identifier. A ticker can change without
  changing instrument identity.
- `session_date`: exchange-local daily trading-session label, never inferred
  by converting a Yahoo daily label into UTC.
- `source_timestamp`: provider's original label, NOT assumed actual market
  close timestamp. `ingested_at_utc` is the retrieval timestamp.
- `currency_original`: source quotation currency; quote units (GBp etc.) must
  be explicitly normalized and validated before approval. The current v3
  processor does **not** implement every quote-unit conversion.
- `fx_to_usd`: number of USD per local currency unit. Inverse source quotes
  are inverted before applying `price_usd = price_local * fx_to_usd`.
- `close_price_usd`: Yahoo Close (under `auto_adjust=False`) translated
  into USD. Yahoo Close is normally ALREADY split-adjusted by its provider;
  re-applying a split multiplier would double-count it.
- `close_total_return_index_usd`: Yahoo Adj Close translated into USD.
  This is an adjusted-price proxy for total-return growth, not a
  unit-normalized wealth index nor independently verified cash reinvestment.
- `ret_price_usd`: change in split-consistent translated Close.
- `ret_total_usd`: change in translated Adj Close (provider-adjusted
  dividend/split history), with `ret_log_total_usd = log(1+ret_total_usd)`.
- `dividend_per_share_local`, `split_new_shares_per_old_share`: Yahoo-reported
  corporate actions retained as explicit observations. These are validated
  against the SAME original OHLCV source response; they are not independently
  verified against an external registry. **Do not double apply actions to
  Yahoo-adjusted values.**
- `quality_status`: accepted or quarantined price observation.
- `return_quality_status`: accepted, unavailable or review; unavailable
  and review returns are never silently filled. The first return of an
  asset is legitimately unavailable.
- The warehouse `market_asset_daily` has one row per canonical
  `(asset_id,session_date)` for one approved release.

The indicator adapter reuses `indicator_calculations.py` (SMA, RSI, ATR,
MACD, volatility, drawdowns) with state resets across invalid observations.
New assets legitimately have null warm-up indicators.

## Commands

Run in an AWS-hosted job/container with Python 3.10 and an IAM role authorized
for the dedicated v3 prefixes. None of these CLI commands write market data to
local files. **No production execution is authorized merely by these examples.**

### 1. Bootstrap reference registry (read-only first)

Historical source:
`s3://alpha-edge-algo/dev/market/lake_v2/archives/old-outputs-20261110/`.
The S3 plan is CHECKSUM verified. Because it is historical, it is only a
starting point, **not the current production registry**.

```bash
poetry run python -m alpha_edge.market.lakehouse_v3.pipeline bootstrap \
  --bucket alpha-edge-algo \
  --asset-ids 'CRYPTO:BTC-USD,CRYPTO:ETH-USD'
```

To store the **review candidate** registry in S3, append `--execute`; the
result contains the immutable `target_key`. The current universe and source
identity changes must be independently reconciled and approved.

### 2. Ingest + calculate Bronze/Silver/Gold (dry-run first)

```bash
poetry run python -m alpha_edge.market.lakehouse_v3.pipeline build \
  --bucket alpha-edge-algo \
  --registry-key market/lake_v3/config/registry_<hash>.json \
  --run-id pilot-20261113 \
  --start 2025-01-01 --end 2026-11-13
```

Append `--execute` only after reviewing the registry and AWS policy.
The `end` date is exclusive. A full cohort is first acquired in memory and
its Bronze files are committed before Silver and Gold are computed. Failure
may leave orphaned immutable objects, but **never a committed partial
manifest**. On retry, only byte-identical objects can reuse the same key.
A new provider response/revision requires a **new run ID**.

The candidate manifest always records `review_required` and
`approved=false` and will not change the warehouse on its own.

### 3. Independent review (mandatory)

Independent evidence must cover:

- instrument identity and source exchange calendar,
- source OHLCV reconciliation and adjusted-close conventions,
- splits/dividends, including special distributions/rights issues,
- FX timing, quote currency/units and absence of look-ahead,
- provider revisions and gaps,
- consumer schema and point-in-time return semantics.

Reviewers store an immutable JSON approval under
`market/lake_v3/approvals/<run_id>/<decision_id>.json` with the schema below.
This is an **example of the contract**, not an authorization to mark
unverified evidence as passed.

```json
{
  "schema": "alpha_edge_gold_approval_v3_1",
  "run_id": "pilot-20261113",
  "candidate_sha256": "<actual SHA256 of candidate manifest>",
  "reviewer": "<independent reviewer identifier>",
  "approved_at_utc": "<verified UTC timestamp>",
  "decision": "approved",
  "gates": {
    "identity": "passed",
    "exchange_calendar": "passed",
    "source_prices": "passed",
    "corporate_actions": "passed",
    "fx_timing": "passed",
    "provider_revisions": "passed",
    "warehouse_contract": "passed"
  }
}
```

**Important:** present implementation checks the approval structure and
pinned hash; independent evidence gathering, authorization of reviewers,
IAM separation-of-duties and verification of reviewer identity must be
operationally enforced before production deployment.

### 4. Inspect release SQL, then explicitly publish

```bash
poetry run python -m alpha_edge.market.lakehouse_v3.pipeline publish \
  --bucket alpha-edge-algo \
  --run-id pilot-20261113 \
  --approval-key market/lake_v3/approvals/pilot-20261113/<decision_id>.json \
  --athena-workgroup primary
```

The dry run verifies the S3 candidate/approval, returns SQL and performs
no Athena DDL. Only after all gates pass, append `--execute`.
The publisher creates an Athena database, an external staging table over
the Gold candidate, a fresh release-specific Iceberg table, validates
row/key counts and updates a stable view.

**Recommended deployment:** create the Athena database, Glue catalog and
workgroup through reviewed IaC. Use a role with scoped
S3 read to the registry/Bronze/candidates, write to v3 storage and Athena
results, Glue catalog permissions, and Athena execution permissions. The
warehouse role should be the only identity able to update the consumer
view. No v1/v2 mutation IAM rights are required.

### Query the user-facing warehouse

```sql
SELECT asset_id, session_date, close_price_usd, ret_total_usd,
       sma_50, annualized_vol_20
FROM alpha_edge_market.market_asset_daily
WHERE asset_id = 'CRYPTO:BTC-USD'
ORDER BY session_date DESC
LIMIT 20;
```

## Test and CI

`tests/unit/test_market_lakehouse_v3.py` exercises immutable S3 writes,
hash verification, registry contracts, provider evidence matching, the
real v2 Silver/Gold transformations, indicator calculations, the S3-only
Bronze→Silver→Gold path and fail-closed warehouse approval.

```bash
poetry run pytest -q tests/unit/test_market_lakehouse_v3.py
```

GitHub Actions workflow:
`.github/workflows/market-lakehouse-v3.yml`.

**Deployment acceptance still required:** integration tests against
isolated AWS Athena/Glue/Iceberg resources, fully verified source-evidence
gates, a daily incremental replay/backfill policy, historical revisions and
corporate actions with externally confirmed adjustments, service-account
authorization, physical-table rollback drill, and consumer cutover tests.

## Non-destructive migration and rollback

- Legacy `market/*/v1` and `market/lake_v2` are never modified by this code.
- S3 bucket versioning/recovery protection and retention must be reviewed
  before any cleanup. Existing immutable manifests are audit evidence.
- Published Gold snapshots get their own `market_gold_<run_id>` tables.
  Repointing the view to an earlier approved table is the intended rollback
  mechanism; this needs a separate reviewed rollback command and test.
- Athena view replacement is a single metadata operation, but the current
  publisher is **not a distributed transaction** spanning Athena and S3
  receipt creation. Operations must be serialized, and interrupted
  publications must be reconciled manually before a retry.
- **No automatic production promotion**, no deleting old buckets, no local
  market-data caching, no implicit fallback to legacy tables.

## Known limitations / remaining implementation work

This branch provides a working **full-request-range pilot** and a warehouse
publisher, **not a certified daily production migration**.

1. A full daily historical release currently requires acquisition of the
   entire requested period. Efficient bounded-overlap, dependency-aware
   incremental Iceberg MERGE/compaction has not been implemented.
2. Date/session semantics and FX matching reuse current v2 policy, but
   independent timing evidence and special exchange/asset exceptions remain
   to be approved.
3. Yahoo-adjusted close is used as the provider's total-return adjustment
   proxy; external corporate-action sources and full independent
   historical split/dividend factor reconstruction remain outstanding.
4. This initial implementation does not yet cover all quote-unit conventions,
   instruments, point-in-time corporate-action revision snapshots or
   signed approval principal authentication.
5. Athena workgroup/IAM/table validation and retry/rollback integration
   must pass in a dedicated AWS test environment before any production
   view is switched.
6. The current GitHub Action runs offline mock tests only; it does not assert
   actual external Yahoo or Athena results.

Do not treat candidate Parquet or an externally unchecked approval JSON as
certified market data.
