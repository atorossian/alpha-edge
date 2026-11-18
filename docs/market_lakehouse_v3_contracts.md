# Alpha Edge Lakehouse v3 — canonical data contracts (AEA-90/AEA-91)

This architecture is implemented **from scratch with respect to persisted market
data**. The operational universe registry is derived only from five current
configuration files, never previously acquired prices or historical lake
assets. Existing pure data-processing Python functions may be reused, but
no persisted v1/v2 data is accepted as a source.

See [AEA-91 full S3 registry specification](aea91_s3_asset_registry.md)
and [pipeline runbook](market_lakehouse_v3.md).

## Universe membership

Permitted data inputs:

1. `data/universe/universe.csv`
2. `data/universe/universe_overrides.csv`
3. `data/universe/asset_excluded.csv`
4. `data/universe/ingest_force_refresh.csv`
5. `data/universe/universe_manual_additions.csv`

`universe.csv.include=1` selects the full operational population;
`include=0` excludes. Stable `asset_id` and `row_id` disambiguate
broker tickers appearing on several exchanges. Supporting files explain
exclusions, corrections and refresh behavior, but cannot silently change
membership. Source conflicts or incomplete currency/venue metadata produce
visible per-asset blockers rather than deleting members.

## Eight-asset validation subset

| Ticker label | Canonical Yahoo source | Currency | Venue/calendar policy |
|---|---|---|---|
| SPY | SPY | USD | US exchange |
| GLD | GLD | USD | US exchange |
| BTC/USD | BTC-USD | USD | Crypto 24/7 |
| RENT3 | RENT3.SA | BRL | Brazil |
| TRAXIONA | TRAXIONA.MX | MXN | Mexico |
| LOG | LOG.MC | EUR | Madrid |
| REY | REY.MI | EUR | Milan |
| EUR/USD | EURUSD=X | USD | Forex |

These are **representative test cases within the full included universe**,
not a substitute for the complete 2,032-member configuration.
The EUR/USD manual-addition source is inconsistent with the resolved
`EURUSD=X` ticker; this is a current-five-file conflict and stays blocked.
Do not invent an alias transition by consulting an old data archive.

## Storage contracts

- Config: five immutable S3 snapshot files and a content-addressed
  registry under `market/lake_v3/config/`.
- Bronze: fresh provider OHLCV, corporate actions and FX quotations
  captured exactly as provided, with source parameters, retrieval
  timestamp, checksums and a committed manifest.
- Silver: deterministic session/calendar, quote-unit normalization,
  FX conversion and corporate-action handling with separate quality
  evidence; no invented market prices or UTC shifts.
- Gold: explicit price-only vs dividend-adjusted total USD returns
  and indicators with checksummed lineage and no return bridging.
- Warehouse: one approved Apache Iceberg Gold release, exposed through
  a stable read-only Athena view. The registry's active configuration
  pointer does not authorize any Gold release.

Config validation and source acceptance are different gates. A symbol
can be in the included universe and still require missing
quote-currency, market calendar, FX source or source-price verification
before data may be published to Gold.

All market datasets, config snapshots and transformations persist in
S3; there are no local operational market-data files. The GitHub
repository contains code, the five bootstrap configs, documentation
and in-memory/mock tests only.

## Integrated contract boundary (AEA-90)

The historical baseline and Silver evidence builders use the formats below.
PR #17's prototype Bronze and Gold manifests have separate schema identifiers.
Their readers cannot consume the full-history Bronze manifest directly, and
the prototype Gold output still requires financial acceptance before use.

| Artifact | Version and identity | Price and time semantics |
|---|---|---|
| Bronze full history | `alpha_edge_full_history_asset_v1`, one manifest per run and canonical asset, rows unique by asset and provider session | Provider OHLC, Adj Close and events are retained on the provider's reported basis. The source timestamp and timezone are evidence, and the requested upper date is exclusive. |
| Bronze incremental | `alpha_edge_bronze_incremental_asset_v1`, one manifest per run and canonical asset, with a pinned parent | Provider observations and revisions are retained separately from the historical baseline. |
| Silver sessions | `alpha_edge_silver_session_candidate_v1`, one candidate per run and asset, pinned to verified Bronze bytes and the chosen calendar policy | Exchange date and UTC timestamps have separate meanings. Session validation retains source observations and reports findings. |
| Silver financial evidence | `alpha_edge_silver_financial_evidence_v1`, pinned to one Bronze asset and the requested FX direction if supplied | Provider actions, adjustment discrepancies and currency rates are evidence. No adjustment-factor or point-in-time availability approval is conveyed. |
| Asset and calendar configuration | PR #17 asset registry and `alpha_edge_calendar_registry_v1` calendar proposals are separate versioned objects | Canonical identity and source snapshot fingerprints are explicit. Calendar proposals carry their review status. |
| Gold and warehouse prototype | `alpha_edge_gold_candidate_v3_1`, release-specific candidate and Iceberg table | USD analytical contract and stable reader design. Acceptance of price-basis and availability semantics is pending. |

All immutable artifact writers use conditional creation followed by S3
readback of the actual bytes and required metadata. A retry after a conflict
is accepted only if the object matches. A missing or different object fails
the write. Artifacts are written before the corresponding manifest so a
partial run does not present a completed asset. The registry active pointer
uses a separate compare-and-swap contract.

An operator-selected run ID is a stable namespace, with run configuration and
manifest checks binding its inputs where applicable. Content fingerprints
identify source/configuration bytes. Use a new run ID for incompatible inputs
or a changed data contract. Introduce a new schema identifier for breaking
changes to keys, fields, price basis or timestamp meaning. Readers should
reject unknown versions until they explicitly support them.

The two CI workflows cover the integrated source guard and synthetic S3
tests. Actual S3 readback and failure injection for the integrated code,
resolution of the locked yfinance 0.2.66 timezone and cookie disk caches,
and independent review of AWS write permissions are still required before AEA-90 acceptance.

Integration evidence for the combined PR #17/#18 code state:

- 130 targeted tests passed under Python 3.12 using locked dependencies.
  The guard checked 17 modules, and Ruff and formatting passed.
- A read-only AWS check of the existing
  `bronze/full_history/runs/bronze-recovery-20261117/run.json` object
  confirmed 45,031 bytes and a matching SHA-256. This checks the stored
  baseline, while post-write readback and failure scenarios for the integrated
  writer were tested with simulated S3.
