# backtest/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

@dataclass
class BacktestConfig:
    bucket: str = "alpha-edge-algo"
    region: str = "eu-west-1"

    # data / cache
    returns_wide_cache_path: str = "s3://alpha-edge-algo/market/cache/v1/returns_wide_min5y.parquet"
    min_years_cache: float = 5.0

    # as-of sampling constraints (trading days)
    warmup_days: int = 252 * 5
    min_forward_days: int = 252

    # simulation controls
    eval_freq: str = "M"          # monthly evaluation points ("W" weekly also ok)
    close_to_close: bool = True

    # goals
    goals: tuple[float, ...] = (1500.0, 2000.0, 3000.0)
    goal_confirm_steps: int = 1
    goal_buffer: float = 0.0

    # benchmark = Option B equal-weight proxy basket
    proxy_tickers: Sequence[str] = ("VT", "SPY", "QQQ", "IWM", "TLT", "HYG", "GLD")

    # backtest behavior
    seed: int | None = 42
    initial_equity: float = 1200.0

    # leverage policy (delegated to your leverage_from_hmm)
    risk_appetite: float = 0.6
    hard_cap: float = 12.0

    # reoptimize thresholds (same defaults as your function)
    max_score_drop: float = 0.15
    max_p_main_drop: float = 0.10
    min_alpha: float = -0.05
