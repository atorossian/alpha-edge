# AEA-87 — approved historical data repair contract

Status: **requirements approved; implementation NOT production-ready**.

## Approved policy decisions
1. **Scope:** every unique `asset_id` in the 6,667-cell anomalies report (854 observed), including excluded assets when they can be mapped to a unique source symbol. An absent or ambiguous mapping is explicitly reported, never silently dropped.
2. **History:** download full provider history; newer valid observations replace stored observations by `(asset_id, date)`. Preserve historical S3 observations absent from the refreshed feed; label these `PRESERVED_UNREFRESHED`, never `REPAIRED`. Avoid calendar-wide FX forward filling.
3. **Failure:** continue independent assets, quarantine failed assets, report partial completion. Never rebuild or publish a trusted cache with unresolved affected observations incorrectly marked clean.
4. **Publication:** stage all replacement asset/year OHLCV and returns partitions and the required manifests before promotion; retain original manifests/parts for rollback; commit recoverably per asset; do not publish a replacement as a normal daily-ingest append.
5. **Remaining anomalies:** preserve available source history, identify and quarantine anomalous dates/assets and exclude unresolved observations from trusted analytics until clearance.

## Required operational integrity
- Immutable run ID and durable campaign journal (asset-level progress/status, original manifest keys/checksums, staged partition keys/checksums, committed revisions, recovery status).
- A single asset's OHLCV/returns manifest switch must be recoverable from journal; detect concurrent ingestion/writers. S3 has no multi-object transaction; recovery state must be explicit.
- Handle delistings, missing Yahoo symbols, splits/dividends/adjustments, FX gaps, mismatched quote currencies, missing corporate-action history, and new vs absent dates as classifications not silent data loss.
- Recalculate the entire affected asset return series *after* merging raw/adjusted USD prices; maintain simple-vs-log field contract.
- Rebuild active wide return cache **only from committed trusted returns**, quarantine untrustworthy cells, and refresh price/return snapshots and daily-ingest checkpoints coherently.
- Consumers (search, optimizer, simulation, daily reports, regime/risk analytics) must honor quarantine, or affected assets must be excluded from their data inputs.
- Idempotent resume for interrupted campaigns; persistent results for SUCCESS / PRESERVED_UNREFRESHED / QUARANTINED / UNMAPPED / SOURCE_UNAVAILABLE / FAILED / ROLLBACK_FAILED.
- Stop normal market-data writers during promotion; expose a non-mutating preparation mode and explicit confirmation for production commits.

## Acceptance
- Every requested asset classified once and completely.
- All modified manifest references backed up, checksummed and recoverable.
- No unintentional date loss; source, adjustment, FX and return semantics consistent.
- Trusted cache and snapshots do not contain identified unresolved abnormal observations.
- A successful dry run, controlled end-to-end pilot including forced failure/rollback, and successful campaign report precede a full production campaign.
- **Do not describe the current branch as production-ready until these conditions are implemented and verified.**

## Current gaps (as reviewed 2026-10-28)
- Current `repair_full_histories` filters `include=1`, contrary to approved scope.
- Existing code stages and switches per asset, not a complete preflighted campaign; writes to live manifests before entire campaign completes.
- Unresolved flagged dates are reported, but quarantine is not enforced across trusted analytics consumers.
- No integrated coherent snapshots/checkpoint updates; wide-cache rebuild alone is insufficient.
- Rollback is best-effort per asset and not durable crash-resumable.
- No complete runtime integration/rollback test has been demonstrated.
