# AEA-94 — Historical dividend currency correction

The historical price currency does not establish the original currency of a
dividend. A legacy currency value can have been presented in an EUR-denominated
series without conversion, or can already be expressed on a later share basis.
Do not mass-convert all pre-2002 observations or infer currency from dividend
yield alone. The European Commission's fixed conversion rates are the policy
reference: https://ec.europa.eu/eurostat/cache/metadata/en/ert_h_esms.htm.

`dividend_currency_registry_v3` supports ESP, ITL and FRF as a first cohort.
Each record requires canonical asset ID, provider symbol, effective source
session, exact Bronze SHA-256, recorded source dividend, independently documented
declared amount and currency, explicit share-basis multiplier, supporting
evidence, correction ID and status. The issuer's declared amount multiplied by
the independent share-basis multiplier must reconcile to the provider amount
within the documented 1e-7 relative tolerance (provider rounding only).

The normalized candidate is the recorded source amount divided by the fixed
legacy-currency units per EUR, preserving the source share basis. An approved
row requires an explicitly attributed reviewer. A proposed row exposes a preview
but leaves `normalized_value` unchanged. `not_corrected` means original source
evidence, not that the dividend has been independently accepted.

The registry is a canonical JSON document with schema
`alpha_edge_dividend_currency_registry_v1`, `rate_reference`, `units_per_eur`
and `entries`. Use the `publish_registry` helper only with explicit authorization
(`execute=True`); the default is read-only and returns the content SHA and S3
key. No local registry input is required. Every version is immutable under
`market/lake_v3/config/dividend_currency_registries/v1/<sha>/registry.json`.
The consumer must specify the exact version using
`build_silver_financial_v3 --dividend-currency-registry-version <sha>`. It
retains the original source actions and emits a separate
`currency_corrected_actions` candidate dataset; its run manifest pins the
registry version. No price, return, split or Gold output is changed.

The current stage cannot certify an ex-date solely from an issuer's payment
date. Reconcile timing independently before financial certification. Other
currency issues (e.g. USD declared distributions for MXN-denominated shares),
capital distributions, missing events and provider revisions need separate
policies. Do not automatically reapply an approved correction against a new
Bronze checksum; re-review its evidence and emit a new registry version.

## First review cohort (2026-11-22, proposed only)

Nine event amounts from the SHA-verified `bronze-recovery-20261117` histories
reconcile to regulatory/issuer evidence and produce positive candidate factors.
They are **not yet approved, published or used in returns**. The exact
proposed registry SHA-256 is
`1617104602570e7e47a980b9b53712f5e362e6bcdf14d8afe11e98758d4e0174`.

| Provider | Source sessions | Documentary evidence | Share basis |
| --- | --- | --- | --- |
| ANA.MC | 2000-01-17, 2000-07-03, 2001-01-15 | [CNMV interim](https://www.cnmv.es/webservices/verdocumento/ver?e=H3GvDo4HZ8XsdPm09A1MkB%2F4NXo3a%2FZf%2BL1UYBcVzRF1eq1W238ebQFWOKTuGvYU), [complementary](https://www.cnmv.es/webservices/verdocumento/ver?e=LPdbBGZugs%2FDaoF2a6iFnXoVWDfesgkpo6eZDnmbdpma3phc%2FKDdGHMm7I9fO0VE), [2001 interim](https://www.cnmv.es/portal/hr/resultadobusquedahr?desde=01%2F07%2F1988&division=1&hasta=12%2F11%2F2013&ihr=2&lang=es&nif=A08001851&page=40) | Source amount matches issuer |
| VID.MC | 2000-02-15, 2000-06-26, 2001-02-15, 2001-06-25 | [CNMV 2000 interim](https://www.cnmv.es/webservices/verdocumento/ver?e=iOjrOr79jhD7%2FFOpEi1Dhh%2F4NXo3a%2FZf%2BL1UYBcVzRF1eq1W238ebQFWOKTuGvYU), [2000 complementary](https://www.cnmv.es/webservices/verdocumento/ver?e=ULIksWPmpENQRRGfMHcG0h%2F4NXo3a%2FZf%2BL1UYBcVzRF1eq1W238ebQFWOKTuGvYU), [2001 history](https://internet.cnmv.es/Portal/hr/resultadobusquedahr?division=3&ihr=2&lang=eu&nif=A01004324&page=38), [2001 complementary](https://www.cnmv.es/webservices/verdocumento/ver?e=dAtw8p26V4M3i2XKB4wpjHoVWDfesgkpo6eZDnmbdpma3phc%2FKDdGHMm7I9fO0VE) | [Nine later 1-for-20 bonus issues](https://www.vidrala.com/en/investors/shareholder-centre/share-capital/) |
| BZU.MI | 2000-07-10 | [Statutory annual report 2000](https://www.buzzi.com/documents/20143/276716/Statutory%20Annual%20Report%202000.pdf/1eb14c71-2d79-57ff-860e-83180333261f) (ordinary shares, amount confirmed; date still needs corroboration) | Source amount matches issuer |
| REC.MI | 2001-05-07 | [Annual report 2000](https://d2e3isjppdvvam.cloudfront.net/wp-content/uploads/2023/06/27104119/annual-interim-ar00-1.pdf) (declared amount confirmed; date still needs corroboration) | [2005 four-for-one split](https://recordati.com/wp-content/uploads/2023/06/annual-interim-3_trim2005-1.pdf) |

Other early Recordati, Acea, Wendel, UniCredit and Banca Mediolanum events
remain open. The latter two show share-basis discrepancies or insufficient
historical lineage; no blanket conversion is authorized.
