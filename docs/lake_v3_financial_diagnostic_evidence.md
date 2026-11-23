# AEA-94 — durable corporate-action diagnostic evidence

The S3-only diagnostic uses the pinned universe snapshot and verified Bronze
receipts. It writes no local market-data files. Each asset is assessed against
the explicit currency registry and optional historical quarantine registry.
The report contains complete source coverage, dividend hypothesis findings,
and quarantine findings. Every row is traceable to a Bronze key and SHA-256.
Each artifact is checksum-addressed, with a completion manifest written last
after read-back verification. An incomplete run has no manifest. The default
artifact format is Parquet; JSON is available for small review pilots.

## Duplicate and spurious dividends

For each event, let `P` be the previous session raw close and `D` the
currency-normalized cash dividend. Compare the observed Yahoo ratio change
to `1-D/P`, `1-2D/P`, `(1-D/P)^2`, and `1-D/(2P)`. The absolute tolerance is
one basis point. The latter three are hypotheses, not issuer evidence.
Identical payments within seven calendar days are also flagged, even if the
Yahoo ratio supports both, because consistent provider arithmetic is not
independent proof of the underlying event. These events extend the historical
quarantine until reviewed. Same-day split/dividend cases and invalid factors
remain non-evaluable. None of these flags automatically suppresses a payment.

The strict v2 duplicate correction registry holds the exact source asset, source hash,
source date and value, retained event date and value, currency, independently
documented issuer ex-date, amount and currency, evidence URL, reason, status,
and (for approval) reviewer. Its immutable content-addressed
versions can be explicitly pinned in the Silver financial evidence builder.
The source and retained amounts must be equal, the retained event must match
issuer documentation, and both distinct events must exist exactly once in
the pinned Bronze evidence. The candidate values must also be unchanged and
positive. Differing amounts or absent retained events fail closed. The prior
v1 registry remains historical evidence only and is not accepted by v2.
Only approved entries set the candidate normalized dividend to zero; original
source actions remain unchanged. No source price or adjusted close is changed.
Quarantine is still assessed from the original source evidence until a separate
price reconstruction and financial-validation stage has been completed.

## Verified nine-asset pilot (2026-11-22)

S3 bucket `alpha-edge-algo`, committed manifest:

`market/lake_v3/silver/financial_diagnostics/v1/runs/aea94-dividend-evidence-pilot-20261122-json/manifest.json`

Manifest SHA-256 `33a8bcb44f5920f58467e35f9b57a21bcd2ca51361a0e58d79f84fe9a1c81412`.
The pilot holds 60,717 source observations, 297 reported dividends, 21 reported
splits, 38 dividend findings and 3,761 quarantine findings. S3 read-back hashes
were verified for all three JSON artifacts before committing the manifest.
The pilot is selected coverage, not a full-universe diagnostic.

Pinned quarantine registry `c6e3faad73e3c364710829bb1c3aef78dc090a56792fd84caee3388f05b9c425`.
Proposed duplicate registry `7adb0935d0e9b7925b3c7f78806e3301a3387d368c76dd63afbc96e8e3028f33`.
Both are immutable S3 objects. Neither is implicitly activated. The historical
v1 proposal included Recordati and Wendel; the latter does not meet the strict
equal-amount criterion and remains unresolved.

The approved strict v2 registry is
`b61ca1f64a5a8c1966bd56b8829914884d646a22c47f66d238c438cb44241b32`,
under `market/lake_v3/config/dividend_duplicate_registries/v2/`.
It contains only Recordati 2005-04-22 (suppress EUR 0.11), retaining the
issuer-confirmed EUR 0.11 event on 2005-04-25. Both events were verified against
the exact Bronze checksum and the currency-corrected candidate; the separate
validation receipt is stored at
`market/lake_v3/silver/financial_diagnostics/v1/duplicate_validation/78ab41e0ff23d7f84fbbff442201c33b95574e5282fd9d0d1a97435adae05c88/receipt.json`.
Both S3 objects were read back and checksum-verified. The registry is available
for explicit pinning, not implicitly active or a production price correction.

## Full nearby-equal dividend review (2026-11-22)

The consolidated-universe diagnostic `aea94-full-universe-20261122` identified
59 nearby-equal pairs. The completed review records 25 approved duplicate
suppressions (including 20 previously approved), eight verified legitimate
distribution pairs for EVT, and 26 unresolved pairs. Unresolved source events
are preserved; ambiguity does not authorize suppression.

Pin strict registry version
`234ac8670302a1ea95121898f6f98228ca93e8862c87814a21d447ba97a23a8e`.
The full pair-by-pair review, original source pointers, reasons and validation
receipt are stored at
`market/lake_v3/silver/financial_diagnostics/v1/duplicate_validation/2ab840de431b507bdebdfb1868beb948e3d4e49e633cc41954e4e9eccae6aed8/receipt.json`.
Both registry and receipt were read back and checksum-verified. The five new
suppressions (GNTX, BAP, twice ICL, and PUM.DE) and eight legitimate EVT pairs
were checked against exact Bronze bytes and the pinned currency registry.

The optional `legitimate_pairs` registry field resolves only an exact nearby-equal
finding, with pinned source checksum, dates, currency, unchanged source amounts,
and independent distribution references. Both cash events remain positive;
the original finding is retained as verified, while other factor or price
quarantine reasons are unaffected. Older strict v2 registries remain readable.
No Bronze prices, adjusted closes, or Gold tables are changed by this review.

## TR share-basis reconstruction pilot

The TR price diagnostic is pinned at SHA-256
`34cd1ba25b4fdb7cc9256ae7dd815fa3d6c7489ceb9abae809762efee81e339f`.
Nine issuer-confirmed 3% stock distributions have zero source split values on
candidate ex-dates; opening gaps and contemporaneous issuer closing-price
ranges support the missing basis factors. Dates remain candidate evidence.

`alpha_edge.market.build_share_basis_candidate_v3` is a separate, explicitly
pinned Silver candidate, not a production financial stage. It checks the
Bronze and evidence hashes, validates all expected source events and applies
only missing subsequent ratios to OHLC and ordinary candidate cash dividends.
Source OHLC, adjusted close, dividends, splits and volume are retained. A
same-day cash observation on a reconstructed share event is not automatically
treated as post-share cash; it blocks the candidate total return and the
backward adjusted-price chain. Price-only returns are separate. The September
1992 duplicate pair also remains unresolved. Observations before 1992-09-22
are retained but untransformed; earlier stock/cash classification requires
additional investigation. No production eligibility is granted.

Pin TR candidate registry
`7e92fd0bc688a42de59baad938eaa5fadc12e71573889ac40f98dbdced7bbbae`.
Run `python -m alpha_edge.market.build_share_basis_candidate_v3 --registry-version
7e92fd0bc688a42de59baad938eaa5fadc12e71573889ac40f98dbdced7bbbae
--run-id tr-share-basis-pilot-20261122` for a dry run; add `--execute` only to
publish the isolated candidate to `market/lake_v3/silver/share_basis_candidates/v1/`.
The manifest is committed last, with content-addressed output and source inputs.

The second TR candidate extends scope to 1992-03-11, adds the missing 1992
March 1.03 share factor and ten explicit candidate cash reconciliations,
including the nine previously blocked simultaneous stock/cash events.
Each uses the documented original declared cash on the pre-share entitlement
basis, divided by the recorded and missing subsequent share factors including
the same-day event. The originally reported cash is retained unchanged.
The September 1992 pair is still blocked: secondary histories conflict, and
the four-business-day pre-1995 ex-date convention points to September 24 if
the secondary September 30 record date is correct. Do not suppress either
without adequate date evidence. Earlier 1980-1992 history remains out of scope
because additional missing stock and cash events have not been reconciled.
The updated candidate registry is
`386a7a050e9a201322982ac0775f260d3b5f70a778a8f774d7ba425928c2e502`.
In-memory replay on the exact Bronze object yielded 10 reconstructed factors,
10 cash reconciliations, 8,730 candidate total returns, and 8,596 continuous
candidate adjusted prices, starting 1992-09-24. No production approval.

## Explicit dividend-event reconstruction pilot (RMS.PA and ML.PA)

The approved strict duplicate registry now has 28 entries and eight legitimate
pairs: `145570abe413f8f9ac3cc352423afa15465d25b08f585b0ae5d1cca168ad41c9`.
VIV.PA and NOC retained/suppressed events passed a checksum-pinned replay.

`alpha_edge.market.build_dividend_event_candidate_v3` handles a reviewed window
with unsupported cash events and a matching amount already present in Bronze.
It verifies the exact source checksum, currency, complete cash-event window,
target session and subsequent recorded share ratios before reconstructing one
candidate event. Share ratios and price coefficients are separate types;
provisional coefficients cannot support a verified basis. Source observations
are retained, no price or return transformation is made, and the resulting
candidate is never production eligible.

Pin registry `a52772b152e3a0ecd957706626a5ff97871f6fbe505d1eb86a91ec7e3cd15824`.
Hermes: source 20/23 May 2008 EUR1.17 each and 5 June EUR1.00; candidate
6 June EUR1.00 on verified basis. Michelin: source 9/14 May EUR0.314028
each and 20 May EUR0.386496; candidate 20 May EUR0.386496 on provisional
basis, pending independent confirmation of the 2010 rights coefficient.
The issuer confirms original EUR1.60; the 2022 4:1 split is verified, and the
implied 0.96624 coefficient is not an approved independent factor.

Run `python -m alpha_edge.market.build_dividend_event_candidate_v3
--registry-version a52772b152e3a0ecd957706626a5ff97871f6fbe505d1eb86a91ec7e3cd15824
--run-id rms-ml-event-pilot-20261122` (add `--execute` for immutable candidate
publication). The two unresolved pairs remain in the diagnostic until final
financial reconciliation; the candidate cannot release quarantine or replace
the authoritative histories.

## European historical rights-issue coefficient reconciliation (2026-11-23)

All 13 reviewed mismatch windows replayed against five checksum-verified Bronze
histories: 26 original positive observations and 13 candidate positive events.
The historical provider basis was independently reconstructed from the official
rights terms and the ratio of the Bronze adjusted cum-close to the historical
unadjusted reference quote. Volkswagen's reference quote is reconstructed by
inverting the published theoretical ex-rights formula, not an independently
retrieved exchange print. Veolia requires both the 2007 and 2021 rights issues.
These price coefficients reconcile cash on the provider basis; they are not
additional stock splits to apply to returns.

Eleven windows retain the exact stored legitimate cash. Two additionally replace
the incorrectly reported cash with the issuer's declared cash excluding a
shareholder-specific tax credit: NEX.PA 2004-06-07 (0.267775 to 0.178516) and
RI.PA 2004-05-25 (0.896007 to 0.499877). The explicit issuer-cash correction
policy requires a verified basis, pinned positive source target and evidence;
the default exact retained-observation requirement is unchanged.

Evidence review: `96908a60c3fbc263d28d25491a5c411981c916b38a6364a3157901e790a2d0e9`.
Reconciliation registry: `7839957a5c68d65d52bcd92a54f871fdd1ffcdc5dafca3ed81af32e27de3d647`.
The partition is 40 cash/factor candidates, 35 event-reconstruction candidates
and 90 unresolved (165 total). All are candidate only; Bronze and production
prices remain unchanged.

## Sigma/ALFA historical dividend reconciliation (2026-11-23)

The Sigma/ALFA group exposed a currency divergence across historical dividends:
the provider's implied cash is approximately 5.402% of its reported cash.
Issuer and exchange records establish USD declarations and historical MXN cash
equivalents. The 2012 10-for-1 split is already reflected in prices. The provider
price coefficient before the Axtel distribution in May 2023 is approximately
0.91356983, independently pinned by the 2020-12-11 closing quote and corroborated
by historical issuer year-end quotes and the tick grid of the flagged sessions.
This is a price-basis normalization, not an additional split.

18 same-session cash/factor candidates were checksum-replayed; the 2018-02-06
source event is separately reconstructed on the independently documented
2018-03-07 ex-date (MXN 0.569977 on the provider price basis). The single-event
date-correction policy requires an explicit pinned positive source event,
verified basis, empty target observation and independent evidence. Historical
FX published after the ex-date remains retrospective-only and cannot be used
for point-in-time return construction. Noncash spin-offs are separately treated.

Review: `b8e74e68026405ad838f321042be6afa2a046e8492b0f870f9b44e2e118363db`.
Registry: `2b029229c01a88db9b9ae181072ce29b726f5aa6571ed262be534ebc8f439b52`.
Six historical Sigma cases remain held (2005, 2006, 2007, 2013, 2016 and 2025).
The new partition is 58 cash/factor candidates, 36 event reconstruction
candidates and 71 unresolved. Bronze and production prices are unchanged.

## Reproduce and scale

Use `python -m alpha_edge.market.audit_financial_universe_v3` with explicit
`--source-fingerprint`, `--bronze-run-id`, `--run-id`,
`--dividend-currency-registry-version`, optional
`--historical-quarantine-registry-version`, optional
`--dividend-duplicate-registry-version`, and either `--only ASSET_ID ...` or
`--all`. The latter assesses the complete included universe for the exact
pinned source snapshot and Bronze run; any absent or corrupt receipt fails
closed before publication. Alternatively, supply `--consolidated-index-version`
instead of the fingerprint and Bronze run. The audit then verifies the index,
exact active universe coverage, each indexed Silver manifest and its referenced
Bronze receipt and source bytes. This supports the consolidated universe across
multiple Bronze runs. `--workers` controls bounded parallel source reads.
Without `--execute` there are no S3 writes. The proposal registry is validated
against the source events but not applied by this diagnostic.

No Bronze or Gold prices were modified, and no artifact is production eligible.
