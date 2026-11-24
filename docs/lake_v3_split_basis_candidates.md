# AEA-94: matching split price-basis candidates

Run the exact-date/ratio population from the full provider reconciliation against
the prior split screening and original checksum-pinned Yahoo and EODHD source
objects. This is an evidence-only candidate registry, not an approved adjustment
registry.

A conventional integer forward/reverse candidate must have:

- a prior Yahoo normalized-price continuity signal;
- no same-day cash event in either source;
- five matching price sessions before and five starting on the event date,
  all positive finite OHLC and adjusted close, no duplicate sessions or other
  reported splits within that window;
- stable EODHD/Yahoo OHLC price-scale ratios on each side (maximum relative
  deviation 0.5%), with the ratio of those scales matching the reported split
  at 0.5% tolerance;
- a stable Yahoo adjusted-close/close step and an EODHD adjusted-close/close
  step consistent with the split.

Noninteger ratios undergo separate economic review even if their dates and
factors agree. Provider agreement is not authoritative proof of legal event
identity. The source-row recovery estimate removes only qualifying split
prefix blockers from the old diagnostic, retaining every other blocker;
it is an upper bound, not released or production-ready data.

Run `python -m alpha_edge.market.audit_matching_split_basis_v3 --help`.
Use `--execute` to persist the complete event decisions, per-asset estimates
and immutable candidate registry. No provider calls or price writes occur.
