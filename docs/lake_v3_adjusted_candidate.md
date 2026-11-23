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
