# optimizer_data_demo.py (or a notebook)

from alpha_edge.universe.universe import load_universe
from alpha_edge.core.data_loader import load_closes_from_folder
from alpha_edge.market.stats_engine import compute_daily_returns, compute_asset_stats, compute_corr_matrix

# 1) Load universe config
universe = load_universe("data/universe.csv")
universe_tickers = list(universe.keys())

# 2) Load prices for those tickers
closes = load_closes_from_folder("data/prices", universe_tickers)

# 3) Compute returns
returns = compute_daily_returns(closes)

# 4) Asset-level stats
asset_stats = compute_asset_stats(returns)
print("Per-asset stats:")
print(asset_stats.sort_values("sharpe", ascending=False).head(10))

# 5) Correlation matrix
corr = compute_corr_matrix(returns)
print("\nCorrelation to QQQ (if present):")
if "QQQ" in corr.columns:
    print(corr["QQQ"].sort_values())
