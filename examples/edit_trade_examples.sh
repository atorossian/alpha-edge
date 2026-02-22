poetry run python src/alpha_edge/operations/edit_trades_asset_id.py \
  --mode plan \
  --dt 2023-07-18 \
  --universe-path ./data/universe/universe.csv \
  --out-csv ./data/trade_asset_id_plan_2023-07-18.csv


poetry run python src/alpha_edge/operations/edit_trades_asset_id.py \
  --mode apply \
  --plan-csv ./data/trade_asset_id_plan_2023-07-18.csv \
  --dry-run

poetry run python src/alpha_edge/operations/edit_trades_asset_id.py \
  --mode apply \
  --plan-csv ./data/trade_asset_id_plan_2023-07-18.csv
