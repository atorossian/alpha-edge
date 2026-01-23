from alpha_edge.universe.universe import load_universe
from alpha_edge.core.data_loader import load_closes_from_folder
from alpha_edge.market.stats_engine import compute_daily_returns, compute_lw_cov_df
from alpha_edge.portfolio.optimizer_engine import evaluate_portfolio_candidate
from alpha_edge import paths

# 1) Universe & prices
universe = load_universe(paths.universe_dir() / "universe.csv")
tickers = list(universe.keys())

closes = load_closes_from_folder(paths.prices_dir(), tickers)
returns = compute_daily_returns(closes)

# 2) Ledoit–Wolf covariance
lw_cov = compute_lw_cov_df(returns)

# 3) Define a candidate portfolio (weights)
candidate_weights = {
    "SPY": 0.25,
    "INDA": 0.15,
    "XLE": 0.15,
    "GLD": 0.15,
    "TLT": 0.15,
    "BTC-USD": 0.05,
    "ETH-USD": 0.10,
}

equity0 = 650.79
notional = 5128.39  # e.g. your current book

metrics = evaluate_portfolio_candidate(
    returns=returns,
    weights=candidate_weights,
    equity0=equity0,
    notional=notional,
    lw_cov=lw_cov,
    days=252,
    n_paths=20000,
    lambda_ruin=0.5,
    gamma_vol=0.0,   # later you can >0 to penalize LW vol
)

print("Score:", metrics.score)
print("P(hit 2000):", metrics.p_hit_2000_1y)
print("P(ruin):", metrics.ruin_prob_1y)
print("LW ann vol:", metrics.ann_vol_lw)
