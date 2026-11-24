# Provider reconciliation (AEA-94)

This diagnostic reconciles the pinned Yahoo financial-diagnostic Bronze receipts
with the EODHD full-history run using canonical asset IDs. It does not publish
Silver prices, change action registries or release quarantine.

- Compare dates in their common historical window. Report earlier/later events
  separately; do not count them as missing events in the other provider.
- Match exact dates first; suggest mutually unique unmatched dates within four
  calendar days. Near-date pairs are not approvals. Keep duplicate rows and
  ambiguous groups as evidence.
- Compare split ratios to relative tolerance 0.01% and dividend amounts to
  0.5%, distinguishing EODHD adjusted and original amounts; unknown or differing
  currencies are not deemed comparable. Quote currency is not proof of the
  historical cash denomination. Raw source events are not overwritten.
- Compare reported OHLC and a diagnostic hypothesis using EODHD's cumulative
  reported split ratios (0.5% tolerance), volume (5%), and adjusted close
  normalized to a common anchor (0.5%). A matching split ratio does not imply
  that the event is legally a split or that its price basis is valid.
- Save event-level source values, price context, raw event records and
  per-asset coverage to immutable, checksum-verified S3 Parquet objects.
  The manifest records all inputs and excludes assets absent from the revised
  universe, while leaving their historic source receipts untouched.

Run `python -m alpha_edge.market.audit_provider_reconciliation_v3 --help`.
Use `--only` for a small pilot and a different run ID for the full universe.
Without `--execute` the command reads and computes but writes nothing.
Financial decisions must be made in a subsequent step using this report and
the existing AEA-94 correction registries and authoritative event evidence.
