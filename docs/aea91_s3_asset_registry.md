# AEA-91 — Canonical S3 registry from the five current universe files

**Jira:** https://alejandrotorossian.atlassian.net/browse/AEA-91

## Authoritative inputs — no historical data

Only these configuration files may determine membership, identity, provider
symbols and exclusions:

1. `data/universe/universe.csv`
2. `data/universe/universe_overrides.csv`
3. `data/universe/asset_excluded.csv`
4. `data/universe/ingest_force_refresh.csv`
5. `data/universe/universe_manual_additions.csv`

**Explicitly prohibited as registry inputs:** legacy v1/v2 market prices,
backfills, historical release inventories, recovery crosswalks, old cohort
metadata and previous data-quality approval flags. None are imported by the
new AEA-91 registry builder.

The latest inspected Git configuration had **2,207 rows**,
**2,035 with `include=1`** and **172 with `include=0`**.
The supporting files had 153 overrides, 170 explicit exclusion records,
36 manual additions and one force-refresh ticker. A separate excluded record
is not required for every `include=0` row; the include flag is the
authoritative membership rule.

All 2,035 included rows remain represented in the versioned registry
even when operationally blocked by missing configuration.
Excluded assets are recorded separately for an auditable membership
decision, but not sent to ingestion.

## Registry architecture

```text
Five pinned Git configuration files [bootstrap only, read in memory]
  -> strict source validation and reconciliation
  -> immutable S3 source snapshot (all five files + SHA-256)
  -> immutable S3 canonical registry
        assets: 2035 included, with individual blockers if necessary
        excluded: 172 excluded, with reasons
        FX configuration candidates: generated from current currency needs
  -> explicitly authorized S3 active pointer (separate activation step)
  -> read-only, version/hash-pinned operational registry reader
  -> downstream Bronze ingestion selects valid configured assets only
```

**Gold market-data approval is not implied by registry activation.** The
registry says which instruments are included and how to attempt ingestion.
Independent source/FX/corporate-action validation is a later gate.

### Config precedence and safe conflict handling

- `universe.csv` owns stable `asset_id`, `row_id` and `include`.
- `universe_overrides.csv` must resolve to an existing row; locked Yahoo
  symbol conflicts are blocked, not silently overwritten.
- `asset_excluded.csv` must resolve to an existing row. Any conflicting
  `include=1` entry is blocked for review.
- `universe_manual_additions.csv` must be materialized in the universe;
  a locked-symbol mismatch blocks that asset.
- `ingest_force_refresh.csv` marks a present ticker for refresh but cannot
  add a missing asset or override exclusion.
- Duplicate IDs/row IDs/tickers, malformed includes, orphaned references or
  unexpected extra data files fail before S3 publication.

The `EUR-USD` manual-addition entry still requests `EUR-USD`, while
the universe's canonical provider symbol is `EURUSD=X`.
The registry retains the current `EURUSD=X` mapping and marks the
locked-manual conflict. **No historical rename or date is invented.**

### Exchange calendars, currencies and FX

Most equities have no `calendar_id` populated in the five source files.
The new `MARKET_POLICY` maps the *current universe's own market values*
to proposed exchange calendar and timezone policies, retaining explicit
conflicts and marking unsupported/blank markets as blocked.
Each venue's historical session conventions still require AEA-93 checks.

Five included assets have missing currency in the current config:
`HYPE-USD`, `KMB`, `USD-CNY`, `USD-BRL`, `USD-JPY`.
They are retained as **included but blocked** until corrected in the five
configuration sources.

The FX configuration builder generates candidate direct/inverse symbols
from the currencies required by included assets, without consulting old
market data. These are **unverified quotation-source suggestions** and are
not accepted conversion observations. AEA-95 must verify source quote
existence, quote units and timing. Any ambiguity should block USD return
publication.

## S3 location and versioning

Immutable source files:
`s3://alpha-edge-algo/market/lake_v3/config/source_snapshots/<source_fingerprint>/<filename>`

Immutable, content-addressed registry:
`s3://alpha-edge-algo/market/lake_v3/config/registries/registry_<SHA256>.json`

Active pointer (only once separately authorized):
`s3://alpha-edge-algo/market/lake_v3/config/active.json`

The runtime registry reader must validate exact content SHA, registered
schema, membership counts and the active pointer. Merely staging a candidate
must never cause activation or ingestion.

## Bootstrap CLI

To generate the registry from **exactly the five current universe config
files** at a pinned Git commit, in memory, without a local data file or
historical S3 source:

```bash
poetry run python -m alpha_edge.market.lakehouse_v3.bootstrap \
  --bucket alpha-edge-algo \
  --source-commit <40-character commit SHA>
```

This is **dry-run by default** and prints a registry key, SHA and
inclusion/blocker summary. `--execute` stages the five immutable S3 source
snapshots and the registry candidate. Staging does not activate it, run
ingestion, or certify market-data observations.

On subsequent config changes, take the new five files from an immutable
Git commit, stage a **new** S3 registry and approve/activate that
version only after reconciling the membership differences.
Never overwrite the prior registry.

## Validation status and outstanding gates

- Clean-slate build and synthetic in-memory S3 tests cover inclusion,
  exclusions, override conflicts, missing data, source fingerprints,
  idempotent staging, and failure to load a registry without an active
  pointer.
- Real integration must check all 2,207 config rows and confirm blocker
  counts, calendar policies and constrained FX configuration.
- Explicit activation and IAM-controlled pointer update must be tested
  before anyone relies on the registry in production.
- The S3 bootstrap snapshot/registry must be staged and verified before
  declaring S3 the operative source of truth.
- Independent verification of source prices, actions and conversions is
  outside AEA-91 and remains mandatory before Gold publication.

The old historical-v2 reconciliation work was incorrectly included in
an earlier draft of AEA-91; that dependency is now removed by design.
