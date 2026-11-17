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

## Silver subsequent stages (not implemented in this ticket)

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
