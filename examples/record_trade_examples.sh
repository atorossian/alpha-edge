  #!/usr/bin/env bash

total_value=10035.59
price=0.2772

# calculate quantity only once
quantity=$(awk -v v="$total_value" -v p="$price" 'BEGIN { printf "%.8f", v / p }')

echo "Open quantity: $quantity"

poetry run python src/alpha_edge/operations/record_trade.py \
  --as-of 2026-03-04 \
  --ts-utc "2026-03-04T14:35:17Z" \
  --ticker ADA-USD \
  --asset-id CRYPTO:ADA-USD \
  --side BUY \
  --quantity "$quantity" \
  --price "$price" \
  --currency USD \
  --action-tag open \
  --quantity-unit coins \
  --value "$total_value" \
  --universe-path ./data/universe/universe.csv \
  --strict-universe


quantity=122.76705431
close_price=83.27

close_value=$(awk -v q="$quantity" -v p="$close_price" 'BEGIN { printf "%.2f", q * p }')

echo "Close value: $close_value"

poetry run python src/alpha_edge/operations/record_trade.py \
  --as-of 2026-03-04 \
  --ts-utc "2026-03-04T08:15:02Z" \
  --ticker SOL-USD \
  --asset-id CRYPTO:SOL-USD \
  --side BUY \
  --quantity "$quantity" \
  --price "$close_price" \
  --currency USD \
  --action-tag close \
  --quantity-unit coins \
  --value "$close_value" \
  --reported-pnl "$(awk -v cv="$close_value" -v ov="$total_value" 'BEGIN { printf "%.2f", cv - ov }')" \
  --universe-path ./data/universe/universe.csv \
  --strict-universe


poetry run python src/alpha_edge/operations/record_trade.py \
  --as-of 2026-03-04 \
  --ts-utc "2026-03-04T14:43:56Z" \
  --ticker SPY \
  --asset-id EQH7f9e4349dabafdd9 \
  --side BUY \
  --quantity 44 \
  --price 680.35 \
  --currency USD \
  --action-tag close \
  --quantity-unit shares \
  --value 30119.00

poetry run python src/alpha_edge/operations/rebuild_ledger.py

poetry run python src/alpha_edge/operations/apply_portfolio_choice.py promote-approved --candidate-id  "20260205-201004"


poetry run python src/alpha_edge/operations/backfill_trades_asset_id.py \
  --universe-path ./data/universe/universe.csv \
  --overrides-path ./data/universe/universe_overrides.csv \
  --overrides-key target_row_id \
  --dry-run \
  --mode strict

 poetry run python src/alpha_edge/jobs/run_warehouse_backfill.py \
   --start 2020-05-20 \
   --end 2026-03-09 \
   --account-id main \
   --ledger-prices-mode asof \
   --build-dim-assets  \
   --universe-path ./data/universe/universe.csv \
   --use-checkpoints \
   --write-checkpoints \
   --stop-on-error \
   --checkpoint-policy month_end \
   --force-ledger \
   --force-warehouse