from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Literal, Optional
import pandas as pd
import numpy as np

Goals3 = Tuple[float, float, float]

@dataclass(frozen=True)
class EvalMetrics:
    # Portfolio definition (optimizer-friendly)
    weights: Dict[str, float]

    # context (helps comparability + health checks)
    goals: Goals3
    main_goal: float

    # unlevered stats (sample-based)
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float

    # LW-based risk
    ann_vol_lw: float

    # VaR / CVaR (1d, 95%)
    var_95: float
    cvar_95: float

    # leveraged MC
    ruin_prob_1y: float
    p_hit_goal_1_1y: float
    p_hit_goal_2_1y: float
    p_hit_goal_3_1y: float
    med_t_goal_1_days: float | None
    med_t_goal_2_days: float | None
    med_t_goal_3_days: float | None

    # ending equity distribution
    ending_equity_p5: float
    ending_equity_p25: float
    ending_equity_p50: float
    ending_equity_p75: float
    ending_equity_p95: float

    # scalar score for optimization
    score: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    as_of: pd.Timestamp
    total_notional: float
    equity: float
    leverage: float
    positions_table: List[Dict[str, Any]]


@dataclass(frozen=True)
class PortfolioReport:
    snapshot: PortfolioSnapshot
    eval: EvalMetrics

@dataclass
class PortfolioCandidateMetrics:
    weights: Dict[str, float]

    # unlevered stats (sample-based)
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float

    # LW-based risk
    ann_vol_lw: float

    # VaR / CVaR (1d, 95%)
    var_95: float
    cvar_95: float

    # leveraged MC
    ruin_prob_1y: float
    p_hit_goal_1_1y: float
    p_hit_goal_2_1y: float
    p_hit_goal_3_1y: float
    med_t_goal_1_days: float | None
    med_t_goal_2_days: float | None
    med_t_goal_3_days: float | None

    # ending equity distribution
    ending_equity_p5: float
    ending_equity_p25: float
    ending_equity_p50: float
    ending_equity_p75: float
    ending_equity_p95: float

    # scalar score for optimization
    score: float

@dataclass
class ScoreConfig:
    lambda_ruin: float = 0.5
    lambda_cvar: float = 0.0
    lambda_mdd: float = 0.0
    lambda_conc: float = 0.0
    lambda_corr: float = 0.0
    lambda_time: float = 0.0

    # optional caps/targets
    ruin_cap: float | None = 0.10  # NEW
    cvar_cap: float = 0.03
    mdd_cap: float = 0.25
    hhi_cap: float = 0.15
    corr_cap: float = 0.60
    time_cap_days: int = 126

    # FFT bands...
    fft_bands_days: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (2.0, 20.0),
        (20.0, 60.0),
        (60.0, 250.0),
    )

    hf_ratio_cap: float = 0.55
    spec_entropy_cap: float = 1.05
    freq_overlap_cap: float = 0.60

    lambda_hf_ratio: float = 0.10
    lambda_freq_overlap: float = 0.05
    lambda_spec_entropy: float = 0.00



@dataclass
class Position:
    ticker: str
    quantity: float
    entry_price: float | None = None  # optional
    exit_price: float | None = None   # optional
    currency: str = "USD"

@dataclass
class PortfolioMetrics:
    # snapshot
    total_notional: float
    equity: float
    leverage: float
    positions_table: List[Dict[str, Any]]

    # unlevered stats
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float

    # LW-based risk
    ann_vol_lw: float

    # risk metrics
    var_95: float
    cvar_95: float

    # leveraged MC
    ruin_prob_1y: float
    p_hit_goal_1_1y: float
    p_hit_goal_2_1y: float
    p_hit_goal_3_1y: float
    med_t_goal_1_days: float | None
    med_t_goal_2_days: float | None
    med_t_goal_3_days: float | None

    # distribution of outcomes at 1y
    ending_equity_p5: float
    ending_equity_p25: float
    ending_equity_p50: float
    ending_equity_p75: float
    ending_equity_p95: float

    # scalar score for optimization
    score: float

@dataclass
class PortfolioHealth:
    date: pd.Timestamp
    score: float
    main_goal: float
    p_hit_main_goal: float
    ruin_prob: float
    ann_return: float
    sharpe: float

    # headline legacy metric (keep)
    alpha_vs_bench: float

    # alpha report blob (optional)
    alpha_report_json: str | None = None

    # ---- 1y diagnostics (optional) ----
    alpha_1y: float | None = None          # CAPM alpha annualized (preferred)
    beta_1y: float | None = None           # CAPM beta
    r2_1y: float | None = None             # CAPM R^2
    info_ratio_1y: float | None = None     # IR
    tracking_error_1y: float | None = None # TE annualized
    excess_return_1y: float | None = None  # mean(port-bench)*252

    # ---- 3m diagnostics (optional) ----
    alpha_3m: float | None = None
    beta_3m: float | None = None
    r2_3m: float | None = None
    info_ratio_3m: float | None = None
    tracking_error_3m: float | None = None
    excess_return_3m: float | None = None


@dataclass(frozen=True)
class PCAModel:
    tickers: list[str]
    mu: np.ndarray            # (N,)
    loadings: np.ndarray      # (N, K)
    factor_returns: np.ndarray  # (T, K)
    resid: np.ndarray         # (T, N)


@dataclass(frozen=True)
class DiscreteAllocation:
    shares: Dict[str, float]
    target_value: Dict[str, float]
    realized_value: Dict[str, float]
    realized_weights: Dict[str, float]
    total_spent: float
    cash_left: float

@dataclass(frozen=True)
class StabilityEnergyConfig:
    alpha_cdar: float = 0.95
    breach_dd: float = 0.25  # d*
    lambda_mdd: float = 1.0
    lambda_cdar: float = 1.2
    lambda_ttr: float = 0.7
    lambda_breach: float = 1.5
    lambda_underwater: float = 0.5

@dataclass(frozen=True)
class StabilityReport:
    energy: float

    # components (all normalized ~[0,1])
    mdd_mean: float
    cdar_alpha: float
    ttr_mean_norm: float
    p_breach: float
    underwater_mean: float



TradeSide = Literal["BUY", "SELL"]
TradeAction = Literal["open", "close", "add", "reduce"]
QuantityUnit = Literal["shares", "contracts", "coins"]  # adjust as you use them

@dataclass(frozen=True)
class Trade:
    trade_id: str
    as_of: str                 # YYYY-MM-DD
    ts_utc: str 
    ticker: str
    side: TradeSide
    quantity: float
    price: float               # REQUIRED
    currency: str = "USD"
    asset_id: Optional[str] = None   
    # NEW: Quantfury semantics (persist to JSON)
    action_tag: Optional[TradeAction] = None          # open/close/add/reduce
    quantity_unit: Optional[QuantityUnit] = None      # shares/contracts/coins
    value: Optional[float] = None                     # notional (esp for contracts)
    reported_pnl: Optional[float] = None              # close PnL when available

    # linkage / metadata
    choice_id: Optional[str] = None
    portfolio_run_id: Optional[str] = None
    note: Optional[str] = None
