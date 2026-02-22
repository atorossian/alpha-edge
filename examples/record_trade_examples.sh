total_value=5000.00
price=0.2832
result=$(awk "BEGIN { printf \"%.8f\", $total_value / $price }")

poetry run python src/alpha_edge/operations/record_trade.py \
  --as-of 2026-02-18 \
  --ts-utc "2026-02-18T10:49:39Z" \
  --ticker ADA-USD \
  --side SELL \
  --quantity $result \
  --price $price \
  --currency USD \
  --action-tag close \
  --quantity-unit ada \
  --value $total_value \
  --universe-path ./data/universe/universe.csv \
  --strict-universe


poetry run python src/alpha_edge/operations/record_trade.py \
  --as-of 2026-02-17 \
  --ts-utc "2026-02-17T14:42:10Z" \
  --ticker CASY \
  --side BUY \
  --quantity 8 \
  --price 669.40 \
  --currency USD \
  --action-tag open \
  --quantity-unit shares \
  --choice-id "2026-02-17-Q-8063d472" \
  --portfolio-run-id "20260205-201004" \
  --value 5355.20

poetry run python src/alpha_edge/operations/rebuild_ledger.py

poetry run python src/alpha_edge/operations/apply_portfolio_choice.py promote-approved --candidate-id  "20260205-201004"


poetry run python src/alpha_edge/operations/backfill_trades_asset_id.py \
  --universe-path ./data/universe/universe.csv \
  --overrides-path ./data/universe/universe_overrides.csv \
  --overrides-key target_row_id \
  --dry-run \
  --mode strict