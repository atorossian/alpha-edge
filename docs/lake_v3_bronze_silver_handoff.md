# Alpha Edge Lake v3 — Bronze handoff and Silver implementation ledger

_Last verified: 2026-11-17. This is a status ledger, not a certification or a production cutover._

## Immutable Bronze baseline (accepted with one explicit exception)

- Bucket: `alpha-edge-algo`; AWS region `eu-west-1`.
- Full-history run: `bronze-recovery-20261117`.
- Pinned 5-file source fingerprint:
  `87fc22f9c87353b9b3b5c69d816ee18bb2699c2aec501aa8e6be10cedf421179`.
- Run end date, **exclusive**: `2026-11-17` (includes provider observations through November 16).
- Verified by `ingest_bronze_full_history verify`:
  `expected=2032 committed=2031 missing=1 corrupt=0 state=incomplete`.
- The sole missing asset is canonical `EQH3ed12a76f622ff34`,
  broker ticker `BKH`, Yahoo ticker `HZN`. Yahoo returned an empty
  provider response after the November 2026 ticker transition.
- **User-approved exception:** proceed with 2,031 committed assets. Keep HZN
  active and visible as missing. Retry HZN later using the exact original
  run ID, fingerprint and selection. Do not synthesize prices or exclude HZN.
- All verified historical Parquet bytes and their S3 manifests are immutable.
  Storage integrity does **not** establish financial, calendar or corporate
  action accuracy.
- Earlier run: `bronze-full-20261116` has an immutable original universe
  snapshot (`2035 expected, 2026 committed, 9 missing`). Retain it as audit
  evidence; it is not the active Silver baseline.

## Bronze implementation: implemented versus pending

| Area | Verified / implemented | Remaining |
| --- | --- | --- |
| Full-history Yahoo adapter | Raw provider OHLCV/Adj Close/dividends/splits, timezone label and currency; actions with no internal adjustment | Independent source accuracy and revision reconciliation |
| Immutable S3 histories | Five-file source snapshots, per-asset SHA-256 Parquet objects/manifests, retries and S3 audit, full readback verification | HZN provider availability, targeted retry |
| Universe identity | Current canonical IDs; BKH→HZN, PSKY→SKYD, USD-CNY→CNY=X, USD-BRL→BRL=X, USD-JPY→JPY=X; NWE/ATKR/ACVA excluded | Corporate-action relationship NWE→HZN (0.98 stock exchange), PARA→PSKY and WBD cash treatment as **Silver reference evidence** |
| Daily incremental acquisition | `ingest_bronze_incremental.py` uses source S3 manifests and bounded date windows; unit tests prove no `period=max` fallback | Live Yahoo/S3 pilot; calendar-aware watermarks; revision diff and dependency replay; daily orchestration |
| Operational safety | Dedicated S3 Bronze namespaces, additive writes, no local market artifacts, working CI | Full integration and operational reliability, controlled release/cutover |

**Caution:** the standalone `ingest_bronze_full_history` CLI still
requests `period=max` on a *new* run. Never use it for normal daily
operations. Same-run retries skip already committed assets; new-run daily
updates require `ingest_bronze_incremental`. The incremental implementation
is not yet production-certified.

## Jira accountability

- **AEA-89:** lakehouse epic and overall release criteria.
- **AEA-92:** Bronze provider adapter, immutable source ingestion and live
  source exceptions.
- **AEA-99:** historical coverage, active-universe reconciliation and
  HZN exception acceptance.
- **AEA-100:** incremental operation, live integration, provider revisions,
  watermark management and downstream recomputation (still open).
- **AEA-93:** Silver exchange dates, timezones, session QA and completeness.
- **AEA-94:** splits, dividends, ticker changes, mergers and corporate actions.
- **AEA-95:** FX pairs, currency quote directions, units and USD conversion.
- **AEA-98:** independent quality evidence and release acceptance.
- **AEA-103/104/108:** orchestration, end-to-end integration and handover.

Existing Jira ticket status values do not prove production readiness. Do not
transition remaining work to Done until its acceptance criteria are satisfied.

## Silver phase 1: session candidates (AEA-93)

Implementation: `src/alpha_edge/market/build_silver_sessions_v3.py`.

Reuses the existing `audit_asset_sessions` from
`standardize_bronze_v2.py`; does **not** import the legacy
`ingest_market_data.py` runtime (which has local disk I/O and USD conversion).

Scope of this first stage:
- Read only committed, byte-verified source Parquet from the pinned Bronze run
  and produce a **candidate**, not approved Silver publication.
- Preserve `session_date` as the source exchange-local daily label; do not
  confuse it with the UTC date of `source_timestamp`.
- Convert only explicitly timezone-aware original timestamp strings to
  independent UTC instants, retaining `timestamp_semantics` as a daily
  provider label rather than a verified market close.
- Preserve all original columns and source rows. Flag invalid/duplicate
  sessions, invalid OHLCV, missing timezone metadata and calendar gaps. Never
  repair, drop, fill or fabricate observations.
- Use only an explicitly supplied, approved market calendar. No automatic
  NYSE assignment to unrelated assets, nor weekday assumption for crypto.
- Write candidate frames, QA findings and SHA-256 manifests only to
  `market/lake_v3/silver/session_candidates/` if `--execute` is requested.
- HZN without a committed Bronze source remains an explicit missing-source
  entry, not a synthetic zero-length file.
- Candidate data is marked `production_eligible=false`. There is no Gold
  publication in this stage.

**Read-only pilot, one approved NYSE equity ETF:**

```bash
poetry run python -m alpha_edge.market.build_silver_sessions_v3 \
  --bucket alpha-edge-algo \
  --source-fingerprint 87fc22f9c87353b9b3b5c69d816ee18bb2699c2aec501aa8e6be10cedf421179 \
  --bronze-run-id bronze-recovery-20261117 \
  --silver-run-id silver-session-pilot-20261117 \
  --only SPY \
  --calendar NYSE
```

Only when the dry-run QA report has been inspected and validated should
`--execute` be supplied to write the pilot candidate to S3.
Do not process the full universe using the first-stage pilot.

**Outstanding before AEA-93 closure:** versioned per-asset market-calendar
registry, robust exchange coverage (US/EU/Asia), half-days, holiday/DST,
24/7 crypto versus FX daily labeling, source endpoint freshness and
provider-availability evidence, independently established observation time,
data-quality quarantine policy and approved release gate.

## Silver SPY S3 candidate publication — 2026-11-17

Source: user-provided execution of `build_silver_sessions_v3` with
`--execute`; independently inspected S3 run configuration, published
report, committed asset manifest and S3 object HEAD responses.

- Silver run: `silver-session-pilot-20261117`; explicit calendar: NYSE.
- Source Bronze run: `bronze-recovery-20261117`; canonical SPY asset:
  `EQH7f9e4349dabafdd9`.
- User's report: `processed=1`, `missing_bronze=0`,
  `status=candidate_written`, `executed=true`,
  `production_eligible=false`.
- 8,508 input/source rows preserved, no reported duplicate/invalid sessions,
  invalid OHLC, negative volume, calendar gaps or quality findings.
- S3 prefix:
  `market/lake_v3/silver/session_candidates/runs/silver-session-pilot-20261117/`.
- S3 candidate object: 604,688 bytes, SHA-256 metadata
  `34d1b86b52ceb081bf4df076d0a0efbd86924279afc531870a9f2730171d928f`.
- S3 findings object: 3,170 bytes, SHA-256 metadata
  `24432b507b6ddfdaf9d89e6b94b2f6250b745e4468ddc9639513db21a1709026`.
- Both object byte sizes and stored SHA-256 metadata match the committed
  Silver asset manifest. S3 run config and report agree on the pinned
  Bronze identity, NYSE calendar and candidate-only status.
- Published report:
  `s3://alpha-edge-algo/market/lake_v3/silver/session_candidates/runs/silver-session-pilot-20261117/reports/4fc0971579514ae59c11bc38b6c54a53.json`.
- **Unverified:** independently rehashing the downloaded Parquet bytes and
  examining every resulting row, as well as broader financial-data acceptance,
  cross-provider checks, corporate actions and FX conversion. No Gold
  release or production acceptance is implied.

**Next step:** approve a versioned per-asset calendar registry rather than
scaling manual `--calendar NYSE` across the entire 2,031-asset Bronze
baseline. Missing/unsupported calendars must remain explicit. Run an
exchange-diverse sample (US equities, EU, Asia, FX, 24/7 crypto), audit
its findings and full byte-level S3 readback before broader Silver
publication.

## Silver continuation — pinned calendar registry and financial evidence (2026-11-17)

**Added on the feature branch (draft PR #18):**

- `src/alpha_edge/market/calendar_registry_v3.py`: derive candidate mappings
  using the exact five-file Bronze universe snapshot and the pre-existing
  `exchange_calendar_mapping_v2.csv` policy. Record SHA-256 content version,
  canonical asset IDs, provider symbols, exchange code, mapping origin and
  approval status. `--execute` writes a single immutable, readback-verified
  registry to `market/lake_v3/config/calendar_registries/v1/<sha>/registry.json`.
  **All mappings remain proposals, not independently verified or approved.**
- `build_silver_sessions_v3.py`: optional `--registry-version <sha>`
  uses source-pinned registry calendars *per asset*, including explicit
  `CRYPTO_24_7` and `YAHOO_FX_WEEKDAYS` daily-label conventions.
  Unknown mappings stay unverified; no default NYSE fallback.
  Registry/version and mapping status are recorded in Silver candidate
  manifests. An existing Silver run is immutable: use a NEW Silver run ID
  when changing to registry mode.
- `src/alpha_edge/market/silver_financial_evidence_v3.py`:
  in-memory, no-I/O provider-reported split/dividend extraction and
  adjustment-ratio discrepancy flags reusing v2 utilities; explicit
  direct/inverse FX conversion to **USD per original currency unit** using
  `normalize_bronze_fx_quotes`. Original Bronze prices remain unchanged.
  This code does not yet orchestrate/release USD-converted security histories
  or independently validate action factors and FX timestamp availability.
- Added CI tests for registry immutability, unresolved/crypto/FX calendar
  conventions, source actions and direct/inverse FX arithmetic.

**Observed mapping coverage from the pinned 2,032-asset source snapshot (pre-approval):** the first registry scan identified 2,023 proposed and 9 unresolved mappings. Two of the nine were confirmed metadata edge cases—not unsupported calendars: `HYPE-USD` has an explicit `24/7` calendar but empty `quoteType`, and `USD-BRL` has an explicit `FOREX` calendar with `asset_class=fx_pair`. The resolver now checks the explicit calendar and source asset class instead of requiring complete Yahoo quote-type metadata. This gives an *expected* 2,025 proposed / 7 unresolved proposals, subject to the actual registry dry run. The remaining unresolved asset tickers are `COP`, `DB1`, `DDS`, `IBOC`, `KMB`, `CFR`, `ET`. Their exchange metadata (`HAM`, `MUN`, `TEX`, or empty) is not automatically mapped. Do not infer their calendars from ticker symbols or mark mappings approved. A proposed mapping is not a calendar-quality pass.

**Registry dry run against the active AEA-91 universe (read-only):**

The calendar builder, Bronze full-history CLI and Silver session CLI default
to the snapshot in the active S3 registry. Use `--source-fingerprint` for
historical replays and to verify an earlier run after the active registry has
changed. Use the same fingerprint throughout a multi-step run. The Silver
session builder rejects a Bronze run pinned to a different snapshot.

```bash
poetry run python -m alpha_edge.market.calendar_registry_v3 --bucket alpha-edge-algo
```

Inspect the reported `proposed` and `unresolved` counts and representative
mapping evidence before publishing with `--execute`.
Use `--registry-version` for subsequent Silver *candidate* pilots.
This does not authorize a full-universe release.

- `src/alpha_edge/market/build_silver_financial_v3.py`: runnable
  one-asset S3 candidate pipeline, consuming only SHA-verified, committed
  Bronze Parquet from the pinned source run, extracting reported action
  events and adjustment-ratio review signals. When an explicit
  `--fx-currency` and `--fx-direction` are supplied for an FX quote asset,
  it publishes accepted rate *evidence* with a source manifest and separate
  anomaly evidence. Outputs are immutable and content-addressed under
  `market/lake_v3/silver/financial_evidence/`. Without `--execute`, it
  performs only S3 reads and in-memory calculations. Tests use mocked AWS.
  This is NOT an approved adjusted-price or FX-converted-price table;
  no corporate action is applied and no cross-asset FX as-of join is done.

**Two focused read-only pilots after CI:**

```bash
poetry run python -m alpha_edge.market.build_silver_financial_v3 \
  --bucket alpha-edge-algo \
  --source-fingerprint 87fc22f9c87353b9b3b5c69d816ee18bb2699c2aec501aa8e6be10cedf421179 \
  --bronze-run-id bronze-recovery-20261117 \
  --silver-run-id silver-action-evidence-spy-20261117 \
  --only SPY

poetry run python -m alpha_edge.market.build_silver_financial_v3 \
  --bucket alpha-edge-algo \
  --source-fingerprint 87fc22f9c87353b9b3b5c69d816ee18bb2699c2aec501aa8e6be10cedf421179 \
  --bronze-run-id bronze-recovery-20261117 \
  --silver-run-id silver-fx-evidence-jpy-20261117 \
  --only USD-JPY \
  --fx-currency JPY \
  --fx-direction inverse
```

Both commands deliberately omit `--execute`: do not publish a new
candidate until its dry-run statistics, provider conventions and source
lineage have been inspected. Do not interpret `USD-JPY` itself as a
JPY-denominated equity or convert its quote series as if it were one.

**Remaining before complete Silver:** independent per-asset calendar
approval (e.g. valid non-US exchange coverage and effective dating), FX
quote-unit metadata and observation-time/no-look-ahead policy, independently
validated split/dividend adjustment semantics (Yahoo `Close` can already
be adjusted), historical corporate-action revision handling, S3-native
versioned financial transformation outputs, audit/readback and full-universe
acceptance gates. None of this is claimed complete by the candidate code.

## Silver subsequent stages (not yet production-ready)

1. **AEA-94:** independently validate split and dividend adjustment factors
   against source corporate-action history; record effective dates and
   asset-identity changes. Treat NWE→HZN 0.98 as stock-for-stock merger
   **not** a regular stock split, and do not splice NWE and HZN into the
   immutable Bronze series. Retain EMBJ's November 16 split evidence.
2. **AEA-95:** normalize FX pair direction and quote units (USD/JPY means
   JPY per USD; convert JPY stocks by division), apply validated timing
   and staleness rules, convert accepted prices to USD in Silver with
   original quote lineage. Do not turn an FX instrument's USD-value
   `USD/JPY` rate into constant 1.
3. **AEA-98:** independent cross-provider and finance-quality checks,
   controlled Gold release certification after all Silver stages.

## Next acceptance checkpoints

1. CI passes unit tests and Ruff for the candidate Silver stage.
2. Run one **read-only** S3-backed Silver pilot (SPY) and inspect
   calendar discrepancies and observed source timestamp semantics.
3. Publish only a separately authorized immutable Silver *candidate*
   and verify S3 hashes/readback; no consumer cutover.
4. Follow through on AEA-100 live incremental Bronze acquisition pilot
   and HZN targeted retry when Yahoo starts publishing the ticker.
5. Scale Silver only with approved explicit calendar policies, FX and
   corporate-action evidence; do not represent the 2,031 Bronze histories
   as finance-validated.
