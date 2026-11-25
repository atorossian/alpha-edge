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

## Integer approval and fractional screening

The 1,674 integer decisions in candidate registry `e2e88b74a4578a1f50b777845b7c32364ce5cc7fb023ac6f49a27e6b35482eff` were approved for price basis only. Record the approval with `approve_split_basis_v3 --candidate-version <digest> --expected-count 1674 --execute`. The immutable record references the hash-verified candidate registry; it does not release quarantine or publish prices.

Run the existing matching split audit with `--fractional` and a new run ID to assess simple fractional exact-date ratio matches separately. The same ten-session evidence gates apply. A passing event is labelled `fractional_price_basis_candidate`, not an approved integer event. The manifest reports status counts by rational ratio; economic identity (split, bonus issue, rights, spin-off or another distribution) remains unresolved. No prices, volume, or quarantine are changed by this audit.

## Fractional approval and small-ratio follow-up

The 720 candidates in registry `28e74121870b0a73a31834931f035cf28fbf8dd68328461de34a0219d2037e77` were approved for price basis only. Record using the approval CLI with `--fractional --expected-count 720`. Economic review remains required.

The 243 exact 21:20 and 11:10 matches receive a separate `--small-fraction` audit. The original continuity gate demanded a log-distance margin of 0.10, which is mathematically unattainable for these ratios (log ratios about 0.049 and 0.095). The alternate test retains the exact-date/ratio checks, ten-session OHLC scale and adjusted-close factor steps, requires an absolute Yahoo event log return no greater than 0.10, and keeps simultaneous cash events blocked. It writes a separate candidate policy and never releases quarantine. The 41 same-day cash cases need separate cash/share reconciliation.

## Small-ratio approval and irregular ratios

The 175 candidates in registry `0d459c14be48fab1ca4e0548e76dc8a4d5ddd507cbfb8d14f634474c8562897a` were approved for price basis only; record with `approve_split_basis_v3 --small-fraction --expected-count 175`. The audit's `--irregular` mode screens the 373 exact-date ratio matches classified as irregular, using the same independent OHLC and adjusted-close factor gates and absolute event-return guard as the small-ratio audit. It retains exact source ratios and summarizes ratio bands, excludes simultaneous cash events, and emits a separate candidate status. Legal/economic identity, including rights, spin-offs and other distributions, is not inferred from a price-basis match. No quarantine release or production eligibility follows from screening.

## Irregular approval and unmatched population

The 239 irregular candidates in registry `f4da644501d73ed5d3db4185d244cb790a19bb35b346ee3ca3a42e46aaa654e5` were approved for price basis only. Record using `approve_split_basis_v3 --irregular --expected-count 239`. The `audit_unmatched_split_cases_v3` module joins pinned screening and reconciliation outputs by asset and Yahoo event date, checks source hashes and classifies the 290 unmatched Yahoo events without counting EODHD-only events as Yahoo splits. It distinguishes date, ratio, duplicate, overlap and historical coverage exceptions and writes an immutable detailed event table and reason distribution. No data corrections or quarantine release occur.

## Yahoo-only implicit price evidence

The `--yahoo-only` mode screens all 224 Yahoo-only events within the overlapping history. It verifies source hashes and exact Yahoo ratios, excludes simultaneous and nearby cash events and conflicting EODHD share events, requires matching ten-session OHLC scales with a ratio-sized discontinuity and matching adjusted-close factor steps, and retains the event-return guard. Equal provider price bases or ratios too small to distinguish from unity are not corroboration. Successful findings are separately labelled `yahoo_only_implicit_price_basis_candidate`; they establish price basis only, not the legal event or an independently reported split. No corrections or quarantine release are performed.

## Consolidated remaining cases

The 15 Yahoo-only implicit price-basis candidates (registry `3fa671a72c842ead2b6fcdc98e81891f4018146832012d716db087c294bd2fe7`) are approved for price basis only via `--yahoo-only --expected-count 15`. The remaining-case audit consumes all five basis runs in chronological order, the unmatched inventory, the original screening and reconciliation evidence and the pinned financial integration bundle. It verifies the immutable approval objects, checks source hashes and identities, deduplicates the targeted re-screenings and requires exactly 787 residual Yahoo cases. It classifies their latest reasons and matches prior evidence by canonical asset, exact date, within four calendar days or explicit event window. Hash mismatches are distinguished from source-pinned matches. Provider cash and older diagnostic observations are context, not approvals. No prices or quarantine are changed.

## Combined cash/share and prior factor review

The consolidated inventory selects the union of 226 cash/share cases and nine exact prior registry matches (six overlap), for 229 events. The new audit checks original Yahoo and EODHD receipts, reads exact documented factors from the pinned integration bundle, and distinguishes verified share ratios, reciprocal price coefficients, disagreements and provisional evidence. For simultaneous cash events, the EODHD original dividend must agree with Yahoo cash after conversion to the post-event share basis; OHLC ratio and each provider's adjustment-factor step must independently agree. Nearby cash, ambiguous or missing cash, currency disagreement, and factors too small to distinguish remain held. The prior-factor-only cases are separately reported and are not implicitly approved. Outputs are candidate-only and do not change prices or quarantine.
