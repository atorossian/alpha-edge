poetry run python src/alpha_edge/operations/record_cashflow.py \
  --as-of 2026-02-21 \
  --ts-utc "2026-02-21T09:23:51Z" \
  --type DEPOSIT \
  --amount 3037.11 \
  --currency USD \
  --note "deposit received"

poetry run python src/alpha_edge/operations/record_cashflow.py \
  --as-of 2026-01-25 \
  --ts-utc "2026-01-25T23:08:49Z" \
  --type WITHDRAWAL \
  --amount 64.83 \
  --currency USD \
  --note "withdrawal completed"

poetry run python src/alpha_edge/operations/record_dividend.py \
  --as-of 2024-03-14 \
  --ticker AU  \
  --asset-id EQHceb1b9c7f9a61b80 \
  --amount 16.05 \
  --currency USD \
  --withholding-tax 0.00 \
  --universe-path ./data/universe/universe.csv \
  --strict-universe


poetry run python src/alpha_edge/operations/record_dividend.py \
  --mode edit \
  --dividend-id 20240314-460d2e491e \
  --shares-held 28 \
  --dividend-per-share 0.19 \
  --source quantfury \
  --pay-date 2024-03-14

poetry run python src/alpha_edge/operations/record_dividend.py \
  --mode record \
  --as-of 2026-02-23 \
  --ticker NOC \
  --asset-id EQH5b14711eb121cbfc \
  --amount 13.86 \
  --currency USD \
  --withholding-tax 0.00 \
  --shares-held 6 \
  --dividend-per-share 2.31 \
  --pay-date 2026-02-23 \
  --source quantfury \
  --universe-path ./data/universe/universe.csv \
  --strict-universe \
  --strict-math

poetry run python src/alpha_edge/operations/record_dividend.py --mode migrate