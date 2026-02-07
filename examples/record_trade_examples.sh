total_value=5000.00
price=63414.68
result=$(awk "BEGIN { printf \"%.8f\", $total_value / $price }")

poetry run python src/alpha_edge/operations/record_trade.py \
  --as-of 2026-02-06 \
  --ts-utc "2026-02-06T00:56:45Z" \
  --ticker BTC-USD \
  --side BUY \
  --quantity $result \
  --price $price \
  --currency USD \
  --action-tag open \
  --quantity-unit btc \
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