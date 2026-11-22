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
