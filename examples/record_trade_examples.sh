total_value=521.02
price=1.4422
result=$(awk "BEGIN { printf \"%.8f\", $total_value / $price }")

poetry run python src/alpha_edge/operations/record_trade.py \
  --as-of 2026-01-26 \
  --ts-utc "2026-01-26T14:07:26Z" \
  --ticker SUI-USD \
  --side BUY \
  --quantity $result \
  --price $price \
  --currency USD \
  --action-tag close \
  --quantity-unit sui \
  --value $total_value


poetry run python src/alpha_edge/operations/record_trade.py \
  --as-of 2026-02-03 \
  --ts-utc "2026-02-03T14:37:54Z" \
  --ticker GLD \
  --side BUY \
  --quantity 6 \
  --price 451.00 \
  --currency USD \
  --action-tag add \
  --quantity-unit shares \
  --choice-id "2026-01-28-f825420d" \
  --portfolio-run-id "20260127-174609" \
  --value 2706.00