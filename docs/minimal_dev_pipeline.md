Minimal dev pipeline

Use one known date and one tiny trade set. Do not start with a huge backfill.

Recommended smoke date:

export AE_ENV=dev
export AE_DT=2024-01-31
export AE_ACCOUNT=main
0. Confirm runtime resolution
poetry run python - <<'PY'
from alpha_edge.core.runtime import load_runtime_config
cfg = load_runtime_config("dev")
print(cfg)
PY

Expected:

env='dev'
bucket='alpha-edge-algo'
region='eu-west-1'
engine_root='dev/engine/v1'
market_root='dev/market'
warehouse_root='dev/engine/v1/warehouse'
is_prod=False

If this is wrong, stop.

1. Import or record a tiny dev trade set

Use either record_trade.py directly or a small CSV through bulk_import_trades.py.

Example CSV:

as_of,ts_utc,ticker,side,quantity,price,currency,action_tag,quantity_unit,note
2024-01-31,2024-01-31T15:30:00Z,SPY,BUY,1,480,USD,open,shares,dev smoke test

Then import:

poetry run python -m alpha_edge.operations.bulk_import_trades \
  --env dev \
  --csv ./data/dev_smoke_trades.csv \
  --dry-run

If dry run is clean:

poetry run python -m alpha_edge.operations.bulk_import_trades \
  --env dev \
  --csv ./data/dev_smoke_trades.csv

No --confirm-prod-write in dev.

2. Rebuild dev ledger

First dry run:

poetry run python -m alpha_edge.operations.rebuild_ledger \
  --env dev \
  --account-id main \
  --start 2024-01-31 \
  --end 2024-01-31 \
  --as-of 2024-01-31 \
  --prices-mode asof \
  --dry-run

Then actual write:

poetry run python -m alpha_edge.operations.rebuild_ledger \
  --env dev \
  --account-id main \
  --start 2024-01-31 \
  --end 2024-01-31 \
  --as-of 2024-01-31 \
  --prices-mode asof

Expected outputs under:

s3://alpha-edge-algo/dev/engine/v1/ledger/dt=2024-01-31/positions.json
s3://alpha-edge-algo/dev/engine/v1/ledger/dt=2024-01-31/pnl.json
s3://alpha-edge-algo/dev/engine/v1/ledger/positions/latest.json
s3://alpha-edge-algo/dev/engine/v1/ledger/pnl/latest.json
3. Build warehouse for that date

Dry run:

poetry run python -m alpha_edge.warehouse.build_warehouse \
  --env dev \
  --dt 2024-01-31 \
  --account-id main \
  --dry-run

Actual write:

poetry run python -m alpha_edge.warehouse.build_warehouse \
  --env dev \
  --dt 2024-01-31 \
  --account-id main

Expected outputs under:

s3://alpha-edge-algo/dev/engine/v1/warehouse/fct_trades/v=1/dt=2024-01-31/part-00000.parquet
s3://alpha-edge-algo/dev/engine/v1/warehouse/fct_positions_daily/v=1/dt=2024-01-31/part-00000.parquet
s3://alpha-edge-algo/dev/engine/v1/warehouse/fct_account_pnl_daily/v=1/dt=2024-01-31/part-00000.parquet

The daily report stats table may be skipped at this point because the daily report has not been generated yet. That is fine.

4. Run daily report in dev

This depends on run_daily_report.py now correctly accepting/using env.

The clean target should be something like:

poetry run python -m alpha_edge.jobs.run_daily_report \
  --env dev \
  --as-of 2024-01-31

But your current main() may not yet have a proper CLI parser. If not, the immediate next patch should be adding CLI args to run_daily_report.py:

--env
--confirm-prod-write
--as-of
--backtest-run-id
--no-write
--no-latest
--equity-override

For now, if main() is still hardcoded, do not trust it as a clean smoke test. Patch the CLI first.

5. Rebuild warehouse again after daily report

After the daily report exists, rerun warehouse for the same date:

poetry run python -m alpha_edge.warehouse.build_warehouse \
  --env dev \
  --dt 2024-01-31 \
  --account-id main

This time, fct_daily_report_stats should no longer be skipped.

Expected additional output:

s3://alpha-edge-algo/dev/engine/v1/warehouse/fct_daily_report_stats/v=1/dt=2024-01-31/part-00000.parquet
6. Run portfolio search in dev

Same warning: only do this once run_portfolio_search.py has been patched to accept --env.

Target command should be:

poetry run python -m alpha_edge.jobs.run_portfolio_search \
  --env dev \
  --as-of 2024-01-31 \
  --equity0 1000 \
  --goals 1200,1500,2000 \
  --main-goal 1500 \
  --dry-run

If the script does not support this yet, then run_portfolio_search.py is the next CLI patch target. Right now it still has hardcoded live defaults in main().