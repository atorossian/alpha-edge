# AEA-90 — Canonical market-data contracts and eight-asset edge-case pilot

**Status:** Contract and CI hardening only. No production data changes or live market-source validation are implied. This work extends [PR #17](https://github.com/atorossian/alpha-edge/pull/17) and implements Jira [AEA-90](https://alejandrotorossian.atlassian.net/browse/AEA-90).

## Pilot chosen from the actual Alpha Edge universe

These identifiers and Yahoo source symbols were read from
`data/universe/universe.csv` on branch
`integration/aea87-market-data-lake-v2`, HEAD `cfb36a1`.
All eight are included (`include=1`). The sample deliberately includes
several trading calendars, three non-USD quote currencies, and both a
**forex asset** and a **forex conversion source**. Neither role is inferred
from the display name.

| User label | asset_id | Yahoo asset symbol | Quote currency | Asset role | Calendar / verification |
|---|---|---|---|---|---|
| SPY | `EQH7f9e4349dabafdd9` | `SPY` | USD | ETF | NYSE calendar policy |
| GLD | `EQH4561ad9433b2a662` | `GLD` | USD | Gold-backed ETF | NYSE calendar policy |
| BTC/USD | `CRYPTO:BTC-USD` | `BTC-USD` | USD | Crypto | 24/7 UTC daily boundary |
| RENT3 | `EQH0f232d452fb06a46` | `RENT3.SA` | BRL | Brazil equity | BMF calendar |
| TRAXIONA | `EQH11f97bdc8d8dfe85` | `TRAXIONA.MX` | MXN | Mexico equity | XMEX calendar |
| LOG | `EQH2325a2fe8044198c` | `LOG.MC` | EUR | Spain equity | XMAD **to validate** |
| REY | `EQH2834655e74af1b24` | `REY.MI` | EUR | Italy equity | XMIL **to validate** |
| EUR/USD | `EQHe874aca04f607837` | `EURUSD=X` | USD | Forex asset | FOREX daily-bar calendar |

The calendar choices for SPY/GLD/Brazil/Mexico follow existing v2 pilot
policy. Exchange-calendar identifiers `XMAD` and `XMIL` are
**proposed**; their availability and historical behavior must be tested
against the installed calendar library before using them for ingestion.
The repo universe did not have a `calendar_id` for the five listed
non-US equities. No guessed calendar is certified here.

The three independent **FX feed roles** are EUR/USD direct
(`EURUSD=X`), BRL/USD inverse from `USDBRL=X`, and MXN/USD inverse from
`USDMXN=X`. BTC/USD remains a crypto security, not an FX normalization
series; EUR/USD also appears as a *market asset* under a different ID.
This double role requires distinct lineage.

## Bronze / Silver / Gold / warehouse contracts

**Bronze** (`alpha_edge_bronze_v3_1`):
an immutable S3 batch under
`market/lake_v3/bronze/runs/<run_id>/`. The committed manifest requires
unique run ID, exact registry key+SHA256, source window [start,end),
nonempty source provenance, original provider adjustment basis, row
counts, content SHA256 and byte size for the original bars/actions/FX
Parquet objects. No consumer reads a batch without committed manifest,
valid artifact references and per-object hash checking.

**Silver:** schema and semantics inherit existing v2 dataframe contracts
pending the independent financial checks in AEA-93/94/95. Distinguish
exchange-local `session_date` (date, not UTC instant),
`source_timestamp` (provider label, not necessarily actual close time),
`ingested_at_utc` (actual UTC source retrieval), and
`available_at_utc` (known-as-of instant, only if independently proven).
Preserve provider raw values, action source events, quote-unit factors,
normalized USD-per-currency FX and quality evidence. Never infer a
missing timezone/currency or invent missing prices.

**Gold candidate** (`alpha_edge_gold_candidate_v3_1`):
must refer to its exact Bronze run, pin registry digest, reconcile
Bronze/Silver/Gold row counts, and SHA-pin four expected artifacts
(Silver, Gold, session evidence, quality report) in isolated S3 prefixes.
The stored candidate state is **always `review_required` and
`approved=false`**. Financial acceptance happens separately under
AEA-94/95/96/98, *not* by schema validation alone.

**Warehouse:** `alpha_edge_market.market_asset_daily` is a read
interface over an independently approved Apache Iceberg Gold release.
One row per (`asset_id`, `session_date`) per approved release;
no fourth physically duplicated layer. Additional release-history and
point-in-time availability fields require actual Athena integration
and verification under AEA-101/102.

### Source units and financial semantics

- All eight quote currencies are explicit. Use BRL, MXN, EUR conversions
  via verified FX quotes; USD and USD-quoted EURUSD=X use identity rate 1.
- Distinguish original provider `Close`, provider-adjusted `Adj Close`,
  split-consistent price, total-return adjusted series and execution
  price. Yahoo `Close` commonly reflects its own split adjustments;
  never apply the same split again. AEA-94 must prove behavior
  independently for selected historical splits and dividends.
- The standard currency-conversion invariant is
  `USD_per_unit_of_currency * normalized_quote_price`; GBp vs GBP
  conversion remains an explicit separate operation and must not be
  inferred silently.
- For each transformation, preserve source observation version,
  calculation version, FX quote/time, action basis, calendar version and
  quality/approval state.
- Historical source corrections become new immutable source versions and
  induce downstream recalculation. Do not silently overwrite earlier
  Bronze records.

### Contract versioning

Breaking changes require a new `schema` identifier and a deliberate,
documented converter and compatibility test. The v3 contract validators
fail closed on missing/unknown major schema versions and unexpected
artifact names, paths or SHA digests. Intra-version optional metadata
may be added without redefining the financial meaning of an existing
column. Source IDs must remain stable; ticker aliases are handled in
the next registry task AEA-91.

### Failure and retry semantics

- An uncommitted Bronze object or Silver/Gold candidate artifact is
  **not a published dataset**. Only the final manifest determines a
  complete publication unit.
- S3 objects are written with `IfNoneMatch='*'`, verified by checksum
  after writing. If the object already exists, its bytes must match
  exactly; collisions fail and require a **new versioned run ID**.
- Retry with a changed source response or changed retrieval timestamps
  is not byte-identical. The caller must use a new `run_id`; reusable
  deterministic source-content IDs are a later design choice
  (AEA-92/100).
- The contract validator is independent of S3; the storage adapter
  checks actual objects, not merely the manifest's assertion.
- Gold approval is independently signed/authorized under AEA-98/102.
  Failure to validate one stage never changes the active warehouse.

### AEA-90 implementation and CI

`src/alpha_edge/market/lakehouse_v3/contracts.py` provides versioned,
strict, reusable registry/Bronze/Gold manifest validation.
`guardrails.py` blocks common local-file market-data operations,
unsafe Parquet output and temp file imports in the v3 data plane.
Two explicit S3 BytesIO reads are permitted and reviewed.

`tests/unit/test_market_lakehouse_v3_contracts.py` uses synthetic
in-memory descriptions of all eight canonical securities and
direct/inverse FX mappings; corruption, duplicate IDs, missing FX,
wrong calendars, incompatible schema, missing provenance,
unexpected artifact keys and local I/O patterns must fail.

GitHub Actions runs the new suite alongside the existing v3 unit tests
and Ruff. Tests do not touch AWS accounts; integration with real
S3/Glue/Athena is **not** proven by passing offline CI.

## What's deliberately next, not AEA-90

- AEA-91: full S3 universe reconciliation and effective-dated aliases.
- AEA-92: actual Yahoo acquisition and provider-version acceptance.
- AEA-93: verify market calendars and true known-as-of timing.
- AEA-94: independently validate special dividends and splits.
- AEA-95: FX timing, quote-unit normalization and causal USD conversion.
- AEA-98/101/102: financial acceptance, Iceberg integration, approvals.
- AEA-100: efficient incremental replay and dependency versioning.

**No market data was downloaded or stored locally to create these
contracts. The production bucket remains unchanged.**


## Live read-only S3 reconciliation — 2026-11-16

The AWS connection was reauthorized. Inspection was **read-only** against
`s3://alpha-edge-algo` in `eu-west-1`; no mutation, ingestion, deletion,
approval, or warehouse activation was executed.

### Sources inspected

1. Git-tracked current identity/eligibility configuration:
   `data/universe/universe.csv` at integration HEAD `cfb36a1`.
2. S3 archived v2 first-review plan:
   `dev/market/lake_v2/archives/old-outputs-20261110/files/full_universe_preflight_v2/first_review/`.
   All 79 cohort CSVs and `fx_config.csv` could be read successfully.
   Their S3 archive inventory declares SHA-256 checksums; **those
   content digests were not recomputed in this inspection**, so reads
   alone are *not* final archived-plan integrity approval.
3. Two S3 committed release-inventory candidates under
   `dev/market/lake_v2/release_candidates/`, including the newer
   exclusions-adjusted snapshot (2,207 assessed, 2,078 currently eligible).
4. S3 prefix presence: **no objects** currently under
   `market/lake_v3/`, `market/lake_v2/`, or
   `engine/v1/universe/`. Existing v2 research evidence is under
   `dev/market/lake_v2/`. The Git-tracked universe remains the
   available current configuration; the v3 canonical S3 registry
   has **not** been installed.

### Eight-asset reconciliation matrix

| Asset | Archived cohort | Archived Yahoo symbol | Current Git symbol | Registry state | Existing v2 release candidate |
|---|---:|---|---|---|---|
| BTC/USD | 0001 | BTC-USD | BTC-USD | Same ID/symbol | Unverified marker |
| RENT3 | 0010 | RENT3.SA | RENT3.SA | Same ID/symbol | Unverified marker |
| TRAXIONA | 0012 | TRAXIONA.MX | TRAXIONA.MX | Same ID/symbol | Unverified marker |
| LOG | 0021 | LOG.MC | LOG.MC | Same ID/symbol | Unverified marker |
| REY | 0023 | REY.MI | REY.MI | Same ID/symbol | Unverified marker |
| GLD | 0036 | GLD | GLD | Same ID/symbol | Unverified marker |
| SPY | 0058 | SPY | SPY | Same ID/symbol | Unverified marker |
| EUR/USD | 0078 | **EUR-USD** | **EURUSD=X** | **Identity preserved; provider symbol differs** | Reprocessed marker unverified |

**Significant finding:** archived EUR/USD had calendar `24/7` and
Yahoo ticker `EUR-USD`; current universe uses `EURUSD=X` while the
v3 pilot proposes a dedicated `FOREX` session-date policy. The
asset_id `EQHe874aca04f607837` is unchanged, and v2 release evidence
explicitly references `reviewed_symbol=EURUSD=X`. The historical
reprocessing has a committed source marker, **not final audited price
approval**. This is an intentional source-symbol/calendar migration
requiring effective-dated alias evidence and data comparison under
AEA-91 and AEA-93, not an automatic rename or rewriting of Bronze.

FX mappings in the archived plan agree with the v3 pilot:
BRL via inverse `USDBRL=X`; MXN via inverse `USDMXN=X`;
EUR via direct `EURUSD=X`. All eight are `current_eligible=true`
in the older v2 inventory. **All eight** also have
`external_evidence_approved=false` and
`production_eligible=false`. No observed source-data accuracy or
corporate action has been independently certified.

### Decisions and blockers

- **Accepted for AEA-90 contract pilot:** the eight canonical asset
  IDs, current provider-symbol mapping, FX pair roles, and distinct
  exchange calendar support in CI.
- **Not yet accepted for production:** historical EUR/USD alias/date
  transition, independently verified prices/actions/FX timing, and
  approval of the real S3 registry and Gold release.
- **AEA-91:** reconcile archived `EUR-USD/24/7` against current
  `EURUSD=X/FOREX`; preserve original observations, attach
  effective-dated alias history, and publish a reviewed canonical
  S3 registry with immutable hash.
- **AEA-93:** verify the actual `FOREX` daily-bar calendar semantics;
  daily Yahoo FX dates are source session labels, not proof of
  24/7 trading or actual Sunday-close timestamps.
- **S3 no-write guarantee for this audit:** listed and read only
  `GetObject`/`ListObjectsV2`, no `PutObject`/delete/publish.

