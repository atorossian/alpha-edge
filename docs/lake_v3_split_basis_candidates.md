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

## Deterministic residual reconciliation

The reusable `reconcile_corporate_actions_v3` module consumes the entire pinned residual inventory, not hardcoded ticker/date lists. Each asset is read once, and every applicable hypothesis is evaluated: pre/post cash denomination, direct cross-provider factor confirmation replacing the continuity signal, bounded common-session windows with an unchanged event predecessor and no dropped corporate action, and near-date price-transition evidence. Observed OHLC ratios and reciprocals are reported for unresolved mismatches but never authorize conflicting source actions. Existing verified factors are retained as evidence, rather than used to bypass checks. Missing evidence and multiple passing economic hypotheses remain held. Per-asset immutable S3 checkpoints are reused only for the identical input fingerprint and policy. One final candidate registry and an event-level audit of all attempts are written; no approval, API research, price mutation or quarantine release is performed. Categories requiring new issuer evidence or wider complex-action reconstruction remain explicit unresolved cases.

## Deterministic residual diagnostics

The 26 candidates from the first deterministic run were approved for price basis only (registry `2f704abd659309d6f889287b1118d74e4cddd222968363a7ae94133a89311a8a`). The `audit_deterministic_residual_v3` command verifies exact candidate identities, hashes and chosen evidence against the deterministic event table, then records a separate immutable approval and diagnoses the residual 761 cases. It reports mutually exclusive prioritized patterns and nonexclusive evidence flags (adjusted-cash agreement, observed ratio, quote-unit hypotheses, matched price/factor steps and prior factors), plus counts by Yahoo symbol suffix and asset. Suffixes describe provider listings, not validated exchange mappings. Evidence flags do not establish economic event identity or authorize corrections. No source or production data is changed.

## Deterministic v2 follow-up

Use `reconcile_corporate_actions_v3 --residual-diagnostics-run <run>` to select only the residual population after the prior immutable approval. The engine verifies the partition and source pins, loads the consolidated session-calendar evidence, and evaluates five- and eight-session windows where available. All expected observations and action dates must be present. It tests per-field/per-session scales, each share and cash factor separately and cumulative steps, independently reported adjusted cash, tighter tolerances for small signals, and cross-provider adjusted returns including large returns. Large returns and unresolved original cash basis are explicitly flagged, not approved as economic events. Four-decimal rounding compatibility is diagnostic only; no rounding exception weakens the gate. Existing missing-share reconstruction (including cash and cumulative missing factors) is attached as context without reapplying it. Ambiguous or noncorroborated actions remain held. No production change or quarantine release occurs.

## V2 candidate review before further approval

The v2 run produced 165 price-basis candidates and 596 residual events. The `audit_v2_reconciliation_review_v3` command verifies the pinned registry against the event table without writing an approval. It splits candidates into large-return, original-cash-basis, both and neither review groups (target-event and wider-window flags distinguished), reports per-field and advanced failure evidence for the full residual population, and produces an issuer-evidence queue for the 137 cases lacking independent confirmation. No new thresholds, source corrections or production changes are authorized by this review.

## V3 field-specific evidence

The 165 V2 candidates are approved for price basis only using the pinned review run, retaining the 95 large-return flags. The residual engine accepts subsequent approval chains and validates the exact disjoint partition; `--field-review` adds a close-only daily factor test after the full-field tests fail. A successful close-only chain is stored in a separate partial candidate registry, never the complete price-basis approval registry. Other OHLC fields and volume remain unapproved. Cash-sequence diagnostics show reported and factor-implied cash by session without inserting a dividend or moving its ex-date. Rounding compatibility remains diagnostic, not a reason to relax a gate. No source, production or quarantine changes occur.

## Bounded issuer investigation pilot

The application-hosted `investigate_corporate_actions_v3` command uses the OpenAI Responses API with web search and strict structured output. It reads the pinned deterministic event table and selects unresolved public ticker/date/ratio cases in an asset-diverse order. No S3 keys, raw licensed bars, account data or secrets are sent to the model. The API key is read only on execution from AWS SSM `/alpha-edge/market-data/corporate-actions/api-key`. A dry run writes nothing and calls no OpenAI API. Explicit limits bound the number of requests, built-in search calls and output tokens; search input tokens are still billable and this is not a hard dollar cap. Conditional per-case claims and immutable checkpoints prevent automatic repeated paid requests after an uncertain interruption. The evidence report retains response IDs, usage, retrieved URLs and proposed facts; arithmetic and date checks are deterministic, while source authenticity, legal identity, economic interpretation and price-basis validation remain review-gated. No proposed source is treated as confirmed solely because the model labels it authoritative. No changes to Bronze, Silver production or quarantine are made. Start with ten cases and assess actual evidence before increasing scope.
