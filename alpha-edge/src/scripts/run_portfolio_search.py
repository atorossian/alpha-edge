# run_search.py
from storage.data_storage import load_universe
from storage.data_loader import load_closes_from_partitions
from storage.preprocess_prices import align_and_clean_closes
from report_engine.stats_engine import compute_daily_returns, compute_lw_cov_df
from optimization.portfolio_search import random_search_portfolios

universe = load_universe("data/universe.csv")
universe_tickers = list(universe.keys())

closes_raw = load_closes_from_partitions(
    root="data/prices",
    start="2015-01-01",
    end="2025-12-06",
    tickers=universe_tickers,
)

closes_clean, kept_tickers = align_and_clean_closes(
    closes_raw,
    max_ffill_days=5,
    min_history_days=252,
    max_missing_frac=0.05,
)

returns = compute_daily_returns(closes_clean)
lw_cov = compute_lw_cov_df(returns)

equity0 = 650.79
notional = 5128.39  # or whatever your current book is

results = random_search_portfolios(
    returns=returns,
    universe={t: universe[t] for t in kept_tickers},
    lw_cov=lw_cov,
    equity0=equity0,
    notional=notional,
    n_candidates=300,
    lambda_ruin=0.5,
    gamma_vol=0.0,
    max_assets=10,
    min_assets=5,
    ruin_cap=0.30,
)

for i, m in enumerate(results[:3], start=1):
    print(f"\n=== Candidate #{i} ===")
    print("Score:", m.score)
    print("P(hit 2000):", m.p_hit_2000_1y)
    print("P(ruin):", m.ruin_prob_1y)
    print("Weights:", m.weights)
