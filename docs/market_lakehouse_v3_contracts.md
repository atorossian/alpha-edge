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
not a substitute for the complete 2,035-member configuration.
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
