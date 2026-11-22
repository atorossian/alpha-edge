# AEA-94 — historical quarantine (candidate evidence)

The strict policy does not repair Yahoo prices, certify independently verified
share bases, or publish Gold. It is opt-in on the existing immutable Silver
financial evidence builder (`--strict-historical-quarantine`).

For a cash dividend already expressed on the price share basis, compare
`(AdjClose / Close)[t-1] / (AdjClose / Close)[t]` against
`1 - dividend[t] / Close[t-1]`. Absolute tolerance is one basis point.
Use the approved currency-normalized amount when a pinned registry is supplied.
A consistent factor does not establish independent correctness.

Unresolved backward-adjusting events quarantine all observations through the
event date, inclusive. This includes reported splits whose price basis is
unverified, missing actions recorded in an independent evidence registry,
unexplained ratio changes, nonpositive or missing adjusted closes, and
non-evaluable or inconsistent dividend factors. Same-session splits and
dividends require explicit basis evidence. Invalid raw closes block their own
session and both adjacent returns. The first observation after any quarantined
span cannot produce a return. Session-calendar completeness and FX timing
remain separate mandatory checks: candidate status is not production status.

The optional `--historical-quarantine-registry-version` is a content-addressed
S3 JSON object under `market/lake_v3/config/historical_quarantine_registries/v1/<sha256>/registry.json`.
Its schema is `alpha_edge_historical_quarantine_registry_v1`, with an `entries`
list of objects containing `asset_id`, `provider_symbol`, `bronze_sha256`,
`effective_date`, `reason` and `evidence_url`. A changed source checksum fails
closed. Evidence dates need not be trading sessions but must lie within the
source history. No registry is implicitly activated.

The share-basis calculation accepts explicit decimal split ratios (new shares
per old share) and separately verified price-basis factors, including rights
issue coefficients. It does not infer these from Yahoo, automatically apply
them to a security, or treat a rights issue as a change in share count. The
historical evidence cases require separately reviewed inputs before repair.

The two additional candidate outputs are `historical_quarantine_sessions` and
`historical_quarantine_findings`; the manifest pins policy and registry version.
Without `--execute`, the builder performs no S3 writes. With it, only immutable
financial candidate evidence is written; no Bronze or Gold is modified.
