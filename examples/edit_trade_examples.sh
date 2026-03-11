poetry run python src/alpha_edge/operations/edit_trades_asset_id.py \
  --mode plan \
  --dt 2026-02-24 \
  --universe-path ./data/universe/universe.csv \
  --out-csv ./data/trade_asset_id_plan_2026-02-24.csv


poetry run python src/alpha_edge/operations/edit_trades_asset_id.py \
  --mode apply \
  --plan-csv ./data/trade_asset_id_plan_2026-02-24.csv \
  --dry-run

poetry run python src/alpha_edge/operations/edit_trades_asset_id.py \
  --mode apply \
  --plan-csv ./data/trade_asset_id_plan_2026-02-24.csv


poetry run python src/alpha_edge/operations/record_trade.py \
  --mode edit \
  --trade-id 20260226-1c2cbde750 \
  --old-as-of 2026-02-26 \
  --action-tag close \
  --side BUY \
  --dry-run