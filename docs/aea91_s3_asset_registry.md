# AEA-91 — Canonical S3 Asset Registry: staged pilot design

**Jira:** https://alejandrotorossian.atlassian.net/browse/AEA-91

**Implementation:** `src/alpha_edge/market/lakehouse_v3/registry_v3.py`
and `contracts.validate_pilot_registry_document`.

**Release status:** review-only pilot candidate; **not approved for ingestion**.
The source of truth for any runtime ingestion must eventually be a
separately promoted immutable S3 registry, never the Git configuration.

## What the first delivery does

- Reads the 79-cohort historical plan exclusively from the archived S3
  evidence via `universe_registry_v2.load_archived_plan`, including the
  plan and per-cohort SHA-256 verification.
- Reads and verifies the SHA-256 and byte lengths of all three files
  declared by the *committed* v2 exclusions-adjusted release inventory,
  checks that it is explicitly `production_eligible=false`, and
  reconciles 2,207 inventory rows / 2,078 eligible identities.
- Checks the exact stable IDs, historical quote currencies, calendar
  expectations, inclusion flags, and independently unapproved state for
  the eight pilot assets against both sets of S3 evidence.
- Reads historical `fx_config.csv` from S3 and verifies its archived
  checksum; retains USD per local-unit conventions.
- Carries current provider symbols, approved/not-approved flag,
  exchange-specific timezone, trading-calendar key, quote currency,
  quote unit and factor (identity 1.0 for this pilot) and historical
  aliases without inventing an effective date.
- Generates a sorted, deterministic JSON schema
  `alpha_edge_registry_v3_2`, with an immutable S3 candidate key
  `market/lake_v3/config/candidates/registry_<SHA256>.json`.
- Never activates, promotes, or replaces the current production
  registry. The existing ingestion contract rejects this review-only
  `v3_2` document. A future versioned approval/promotion contract
  must explicitly authorize ingestion.

## Sample CLI: live-read-only dry-run by default

```bash
poetry run python -m alpha_edge.market.lakehouse_v3.registry_v3 \
  --bucket alpha-edge-algo
```

With `--execute`, this new command writes **one immutable
review-only candidate** under the `market/lake_v3/config/candidates/`
prefix using `IfNoneMatch="*"`, checksum verification, and exact-byte
collision rejection. It does not create an active pointer, approve
the source history or run Yahoo ingestion.

Do not execute before the AWS IAM role and remaining evidence are
reviewed. `--execute` is not a production activation.

## Pilot identity and policy

| Asset | Canonical ID | Current provider symbol | Exchange calendar | Currency |
|---|---|---|---|---|
| SPY | `EQH7f9e4349dabafdd9` | `SPY` | NYSE | USD |
| GLD | `EQH4561ad9433b2a662` | `GLD` | NYSE | USD |
| BTC/USD | `CRYPTO:BTC-USD` | `BTC-USD` | 24/7 | USD |
| RENT3 | `EQH0f232d452fb06a46` | `RENT3.SA` | BMF | BRL |
| TRAXIONA | `EQH11f97bdc8d8dfe85` | `TRAXIONA.MX` | XMEX | MXN |
| LOG | `EQH2325a2fe8044198c` | `LOG.MC` | XMAD | EUR |
| REY | `EQH2834655e74af1b24` | `REY.MI` | XMIL | EUR |
| EUR/USD | `EQHe874aca04f607837` | `EURUSD=X` | FOREX | USD |

### EUR/USD: unresolved ticker observation, NOT a verified rename date

The original archived source was `EUR-USD` / `24/7`;
the current source is `EURUSD=X` / `FOREX`.
Both belong to the SAME stable asset ID according to the existing
registry and v2 review evidence. The schema captures them as two
separate `source_aliases` observations. Both
`effective_from` and `effective_to` remain null, and
`effective_dates_verified=false` until a reviewer checks the exact
historical transition and the daily Yahoo FX session convention.

The archive alone does not prove that the two provider symbols are
interchangeable or their prices equivalent. Do not backfill across the
change without a controlled source/series reconciliation.

### FX normalization inputs

- EUR direct `EURUSD=X`.
- BRL inverse `USDBRL=X`.
- MXN inverse `USDMXN=X`.

These source quotes are available as an archived configuration;
their actual timing and values require validation in AEA-95.

## Immutable data version and validation policy

- A historical or excluded asset never disappears simply because
  its modern `include` flag is false; it remains represented in the
  historical registry and separate audit evidence.
- One current symbol per active asset and no duplicate canonical IDs.
  Unresolved current eligibility/identity/quote policies fail.
- A changed input (archive plan, eligible source inventory, FX evidence,
  pilot mapping, alias/review resolution) changes the serialized
  candidate and its SHA, producing a new S3 key.
- A previously written key may only be reused when its bytes match.
  Candidate structure and lineage are validated before writing.
- A candidate must keep `review_state=review_required` and
  nonempty `release_blockers`; attempting to mark it approved
  without independent verification fails.

## Remaining work before AEA-91 is complete

1. Establish a **full universe** S3 registry; this pilot covers eight
   assets only. Current Git-tracked configuration is not yet an
   immutable, checksum-pinned S3 upstream source for all 2,207 identities.
2. Resolve the historical EUR/USD alias dates and price-series
   equivalence; no automatic historical remapping.
3. Close the 911 historical identity crosswalk candidates: S3 asset
   plan reports **888 single-symbol matches requiring instrument review,
   14 ambiguous current symbols, 2 ambiguous historical symbols, and
   7 historical symbols absent from the current universe**.
   There are **zero approved identity merges** in that plan.
4. Implement independent registry review, active-pointer promotion
   with IAM separation/concurrency control, and approved-reader
   version pinning. Do not reuse a reviewer-created JSON boolean as
   a production authorization.
5. Verify full-universe exclusions/overrides, quote units, instrument
   identity and source preferences from immutable S3 configuration.
   Keep all unresolved identities explicitly blocked.
6. Run an isolated AWS S3 end-to-end integration, verify actual
   candidate checksum and ensure no accidental local data persistence.

**No real market-data observations are approved by this task.**
