# Silver adjusted-price candidate

The integrated candidate reads the consolidated session index and a separate
checksum-pinned financial integration registry. The latter references earlier
currency, duplicate, quarantine, cash/factor, reconstructed event, missing
session, share-basis, combined cash/stock and spin-off registries. Correction
amounts, source hashes and evidence stay in S3, not in the code repository.

Run the builder with explicit versions, an immutable run ID and canonical asset
selectors (or `--all`). Example, with the environment-specific versions supplied
by the operator:

```bash
poetry run python -m alpha_edge.market.build_silver_adjusted_v3 \
  --consolidated-index-version "$CONSOLIDATED_VERSION" \
  --bundle-version "$FINANCIAL_BUNDLE_VERSION" \
  --run-id adjusted-candidate-review \
  --all --workers 4
```

Without `--execute`, this performs read-only reconstruction and prints summary
statistics. Adding `--execute` publishes immutable per-asset Parquet objects and
a completion manifest only if all selected source discrepancy identities are
accounted for. All output is candidate-only. Neither Bronze, legacy nor Gold
is changed, and no quarantine is released.

Raw columns stay unchanged. `candidate_*` fields contain reconstructed local
currency OHLC, cash, volume, price/total/reference returns and factors.
Provider-reported splits are not applied again. A separately evidenced missing
share factor changes the historical price and volume basis reciprocally.
Ordinary dividends and noncash spin-off reference adjustments do not alter
volume. Estimated volumes and retrospective/provisional evidence are flagged.
FX translation to USD is outside this build.

The actual total-return chain breaks at missing sessions, invalid observations,
or an unvalued distributed security. The reference-value return is separate and
must not be substituted for actual shareholder return. The latest continuous
chain is `candidate_adjusted_close`; disconnected earlier segments are exposed
as `candidate_segment_adjusted_close` with `candidate_segment_id` and must not
be concatenated as a continuous series. Open/high/low use the same corresponding
price coefficient as close. Existing source-based quarantine is retained for
subsequent independent adjudication.

## Approved split price-basis impact

The optional `--split-basis-approvals` and `--deterministic-basis-approvals` arguments load immutable approved price-basis registries, validate their counts, unique identities and source hashes, and attach the exact matched decisions to the Silver candidate. The separate counterfactual return eligibility view removes only the split-price-basis blocker from the historical prefix; other reasons, unavailable returns and invalid sessions are retained. Existing historical quarantine and production eligibility are unchanged. All economic review requirements and large-return dates remain flagged; the view is a potential availability measure, not a production release. The manifest reports the before/after and incremental return observations. Close-only and AI investigation registries are intentionally not consumed by this build.

## Consolidated residual dispositions

`--residual-version`, `--expected-residual-count` and `--eodhd-run-id` enable
the approved residual decision inventory in the same builder. The pinned
registry must target the exact consolidated index and existing financial
integration bundle. The original manual approval objects are checksum
verified and their event identities matched. All selected decisions must be
represented, including intentionally held cases. The earlier close-only
research registry is not an approval and cannot be passed directly.

The candidate reuses explicitly executable asset price reconstruction plans,
rejects overlapping price or cash repairs, and retains independent volume,
cash and boundary checks. The approved treasury-share delivery duplicate is
handled only on its source-pinned dates and amount. Research-only actions
whose price reconstruction specification is incomplete remain held. The
close-only daily factor-chain audit is repeated against the pinned provider
and calendar objects; its local price returns are separate diagnostic fields,
not a replacement for total returns or proof of full OHLC quality. Full-price
basis checks additionally distinguish already-normalized prices from an
independent share-scale change. No decision clears historical quarantine.

The builder validates unchanged source observations, exact decision coverage,
positive finite segmented adjusted prices and return-chain consistency. The
manifest distinguishes applied repairs, reverified price bases and remaining
financial blockers. These checks are candidate invariants rather than final
financial acceptance. Execute only after reviewing a dry run; the same inputs
and a new immutable run ID publish to S3 with read-after-write verification.

The `--execution-plan-version` option replaces all individual registry/version
arguments. It reads a checksum-pinned S3 plan containing the approved counts
and exact input versions (3,014 existing price-basis approvals and 596 residual
decisions in the November 27, 2026 plan). For the full-universe dry run:

```bash
poetry run python -m alpha_edge.market.build_silver_adjusted_v3 \
  --bucket alpha-edge-algo --region eu-west-1 \
  --execution-plan-version 9e111f4b6be9a2d81c19001ad1dfe6ca24542f0f976dd5f9abb781ba556043a5 \
  --run-id aea94-consolidated-financial-20261127 --all --workers 4
```

The plan is candidate-only; it is not a financial acceptance manifest. Add
`--execute` only after the dry run is reviewed. No provider download is needed.
