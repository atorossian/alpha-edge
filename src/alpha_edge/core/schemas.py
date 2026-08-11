from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Literal, Optional, Deque
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

    # path-stability diagnostics from leveraged MC equity paths
    stability_energy: float | None = None
    path_mdd_mean: float | None = None
    cdar_95: float | None = None
    p_dd_breach: float | None = None
    underwater_mean: float | None = None
    ttr_mean_days: float | None = None

    # path-stability diagnostics from leveraged MC equity paths
    stability_energy: float | None = None
    path_mdd_mean: float | None = None
    cdar_95: float | None = None
    p_dd_breach: float | None = None
    underwater_mean: float | None = None
    ttr_mean_days: float | None = None


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

    # path-stability diagnostics from leveraged MC equity paths
    stability_energy: float | None = None
    path_mdd_mean: float | None = None
    cdar_95: float | None = None
    p_dd_breach: float | None = None
    underwater_mean: float | None = None
    ttr_mean_days: float | None = None

@dataclass
class ScoreConfig:
    lambda_ruin: float = 0.5
    lambda_cvar: float = 0.0
    lambda_mdd: float = 0.0
    lambda_conc: float = 0.0
    lambda_corr: float = 0.0
    lambda_time: float = 0.0

    # optional caps/targets
    ruin_cap: float | None = 0.10
    cvar_cap: float = 0.03
    mdd_cap: float = 0.25
    hhi_cap: float = 0.15
    corr_cap: float = 0.60
    time_cap_days: int = 126

    # FFT bands / spectral penalties
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

    # Path-stability caps/targets. These are measured from leveraged MC equity paths.
    # Lower is better for all fields. Caps are expressed in normalized units except days.
    stability_alpha_cdar: float = 0.95
    stability_breach_dd: float = 0.25
    stability_energy_cap: float = 0.80
    path_mdd_mean_cap: float = 0.30
    cdar_95_cap: float = 0.45
    p_dd_breach_cap: float = 0.25
    underwater_mean_cap: float = 0.55
    ttr_cap_days: int = 126

    # Path-stability penalty weights. These make the live score prefer smoother
    # executable equity paths, not only high goal-hit probability.
    lambda_stability_energy: float = 0.25
    lambda_path_mdd_mean: float = 0.25
    lambda_cdar_95: float = 0.35
    lambda_p_dd_breach: float = 0.40
    lambda_underwater: float = 0.10
    lambda_ttr: float = 0.10


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

    # path-stability diagnostics from leveraged MC equity paths
    stability_energy: float | None = None
    path_mdd_mean: float | None = None
    cdar_95: float | None = None
    p_dd_breach: float | None = None
    underwater_mean: float | None = None
    ttr_mean_days: float | None = None

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
QuantityUnit = Literal["shares", "contracts", "coins", "ounces"]

PositionEffect = Literal[
    "open_long",
    "open_short",
    "add_long",
    "add_short",
    "reduce_long",
    "reduce_short",
    "close_long",
    "close_short",
]

AssetClass = Literal["stock", "crypto", "forex", "futures", "commodity", "unknown"]


@dataclass(frozen=True)
class Trade:
    trade_id: str
    as_of: str
    ts_utc: str
    ticker: str
    side: TradeSide
    quantity: float
    price: float
    currency: str = "USD"
    asset_id: Optional[str] = None

    action_tag: Optional[TradeAction] = None
    quantity_unit: Optional[QuantityUnit] = None
    value: Optional[float] = None
    reported_pnl: Optional[float] = None

    choice_id: Optional[str] = None
    portfolio_run_id: Optional[str] = None
    note: Optional[str] = None

    # --- Trade calculation engine v1 ---
    asset_class: Optional[AssetClass] = None
    position_effect: Optional[PositionEffect] = None

    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    open_value: Optional[float] = None
    close_value: Optional[float] = None

    calculated_pnl: Optional[float] = None
    pnl_diff: Optional[float] = None
    pnl_source: Optional[str] = None  # calculated / broker_reported / manual_override / unavailable

    # --- Triple-barrier-style risk contract ---
    risk_model: Optional[str] = None
    stop_loss_price: Optional[float] = None
    target_profit_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    target_profit_pct: Optional[float] = None
    risk_amount: Optional[float] = None
    target_profit_amount: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

    max_holding_days: Optional[int] = None
    expiry_date: Optional[str] = None
    time_barrier_method: Optional[str] = None

    barrier_status: Optional[str] = None  # active / closed / stop_loss_hit / target_profit_hit / time_exit_due
    barrier_hit_at_utc: Optional[str] = None
    barrier_hit_price: Optional[float] = None
    barrier_hit_type: Optional[str] = None

    calculation_method: Optional[str] = None
    calculation_inputs: Optional[Dict[str, Any]] = None
    calculation_warnings: Optional[list[str]] = None

@dataclass(frozen=True)
class RuntimeConfig:
    env: str
    bucket: str
    region: str
    engine_root: str
    market_root: str
    warehouse_root: str
    is_prod: bool = False

@dataclass
class TradeRow:
    trade_id: str
    as_of: str
    ts_utc: str
    asset_id: str
    ticker: str
    side: str
    action_tag: str
    quantity: float
    price: float
    currency: str
    quantity_unit: Optional[str]
    value: Optional[float]
    reported_pnl: Optional[float]
    s3_key: str


@dataclass
class OpenQtyLot:
    trade_id: str
    as_of: str
    ts_utc: str
    asset_id: str
    ticker: str
    side: str
    action_tag: str
    quantity_open: float
    quantity_remaining: float
    price: float
    value: Optional[float]
    quantity_unit: Optional[str]


@dataclass
class OpenValueLot:
    trade_id: str
    as_of: str
    ts_utc: str
    asset_id: str
    ticker: str
    side: str
    action_tag: str
    value_open: float
    value_remaining: float
    price: float
    quantity: Optional[float]
    quantity_unit: Optional[str]

@dataclass
class RepairRow:
    trade_id: str
    as_of: str
    ts_utc: str
    asset_id: str
    ticker: str
    side: str
    action_tag: str
    old_quantity: Optional[float]
    new_quantity: Optional[float]
    old_price: Optional[float]
    new_price: Optional[float]
    old_value: Optional[float]
    new_value: Optional[float]
    old_quantity_unit: Optional[str]
    new_quantity_unit: Optional[str]
    old_reported_pnl: Optional[float]
    new_reported_pnl: Optional[float]
    quantity_policy: Optional[str]
    repair_action: str
    repair_reason: str
    command: str
    s3_key: str

@dataclass
class Dividend:
    dividend_id: str
    as_of: str
    ts_utc: str
    account_id: str
    asset_id: str
    ticker: Optional[str]
    amount: float
    currency: str
    note: Optional[str] = None
    shares_held: Optional[float] = None
    dividend_per_share: Optional[float] = None
    ex_date: Optional[str] = None
    record_date: Optional[str] = None
    pay_date: Optional[str] = None
    gross_amount: Optional[float] = None
    source: Optional[str] = None

@dataclass
class Cashflow:
    cashflow_id: str
    as_of: str
    ts_utc: str
    account_id: str
    type: str
    amount: float
    currency: str
    note: Optional[str] = None

@dataclass
class PositionLotAvg:
    asset_id: str
    ticker: str
    quantity: float
    avg_cost: float
    currency: str = "USD"


@dataclass
class PositionView:
    asset_id: str
    ticker: str
    quantity: float
    avg_cost: float
    last_price: float | None
    market_value: float | None
    cost_value: float
    unrealized_pnl: float | None
    currency: str = "USD"


@dataclass
class DerivativePositionView:
    asset_id: str
    ticker: str
    side: str
    open_notional_usd: float
    avg_entry_price: float
    currency: str = "USD"


@dataclass
class PnLSummary:
    as_of: str
    trade_count: int
    tickers_spot: int
    tickers_derivatives: int
    realized_pnl: float
    unrealized_pnl_spot: float
    dividends_pnl_usd: float
    net_cashflow_usd: float
    total_pnl_usd: float
    equity_usd: float


@dataclass
class LedgerState:
    spot_long_lots: Dict[str, Deque[OpenQtyLot]]
    spot_short_lots: Dict[str, Deque[OpenQtyLot]]
    notional_long_lots: Dict[str, Deque[OpenValueLot]]
    notional_short_lots: Dict[str, Deque[OpenValueLot]]
    ticker_by_asset: Dict[str, str]
    ccy_by_asset: Dict[str, str]
    realized_pnl_usd: float
    net_cashflow_usd: float
    dividends_pnl_usd: float
    trade_count: int


@dataclass(frozen=True)
class SplitEvent:
    ticker: str
    effective_date: str
    factor: float

@dataclass(frozen=True)
class UniverseIndex:
    by_broker_ticker: Dict[str, List[dict]]


@dataclass(frozen=True)
class AuditEvent:
    audit_id: str
    run_id: str
    event_type: str          # create, modify, delete, rebuild, backfill, import, migrate, build_dataset
    entity_type: str         # trade, cashflow, dividend, ledger, market_data, warehouse, portfolio_choice
    entity_id: Optional[str]
    as_of: Optional[str]

    env: str
    bucket: str
    root: str

    source_script: str
    source_mode: Optional[str]
    created_at_utc: str

    status: str              # success, failed, dry_run
    reason: Optional[str]

    input_args: Dict[str, Any]
    output_keys: list[str]
    backup_keys: list[str]
    deleted_keys: list[str]

    before_payload: Optional[dict] = None
    after_payload: Optional[dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class IndicatorBuildConfig:
    bucket: str
    region: str
    market_root: str

    universe_csv: str
    excluded_csv: Optional[str] = None
    filter_to_active_universe: bool = True

    mode: str = "full"
    start: str = "2010-01-01"
    end: Optional[str] = None

    lookback_calendar_days: int = 500
    replacement_calendar_days: int = 60
    skip_unchanged_assets: bool = True

    max_assets: Optional[int] = None
    min_obs: int = 60
    annualization_days: int = 252

    max_workers: int = 8
    progress_every: int = 100

    indicators_prefix: str = "market/indicators/v1"
    latest_snapshot_name: str = "latest_indicators"

    dry_run: bool = False

    def validate(self) -> None:
        self.bucket = str(self.bucket).strip()
        self.region = str(self.region).strip()
        self.market_root = str(self.market_root).strip("/")
        self.universe_csv = str(self.universe_csv)
        self.indicators_prefix = str(self.indicators_prefix).strip("/")
        self.latest_snapshot_name = str(self.latest_snapshot_name).strip()

        if self.excluded_csv is not None:
            self.excluded_csv = str(self.excluded_csv)

        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"full", "incremental"}:
            raise ValueError("mode must be 'full' or 'incremental'.")

        if self.lookback_calendar_days < 1:
            raise ValueError("lookback_calendar_days must be >= 1.")

        if self.replacement_calendar_days < 0:
            raise ValueError("replacement_calendar_days must be >= 0.")

        if self.replacement_calendar_days > self.lookback_calendar_days:
            raise ValueError(
                "replacement_calendar_days cannot exceed "
                "lookback_calendar_days."
            )

        if self.min_obs < 1:
            raise ValueError("min_obs must be >= 1.")

        if self.annualization_days < 1:
            raise ValueError("annualization_days must be >= 1.")

        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1.")

        if self.progress_every < 1:
            raise ValueError("progress_every must be >= 1.")


@dataclass
class CacheConfig:
    bucket: str = "alpha-edge-algo"
    region: str = "eu-west-1"
    market_root: str = "market"

    cache_prefix: Optional[str] = None

    mode: str = "full"
    min_years: float = 5.0
    start: str = "2010-01-01"
    end: Optional[str] = None
    min_obs: int = 252 * 5
    dtype: str = "float32"
    force: bool = False

    replacement_calendar_days: int = 30
    max_new_assets_full_build: int = 100

    progress_every: int = 100
    max_workers: int = 8
    s3_max_concurrency: int = 8
    strict_window: bool = True

    universe_csv: str = "data/universe/universe.csv"
    excluded_csv: Optional[str] = "data/universe/asset_excluded.csv"
    filter_to_active_universe: bool = True

    dry_run: bool = False

    def validate(self) -> None:
        self.bucket = str(self.bucket).strip()
        self.region = str(self.region).strip()
        self.market_root = str(self.market_root).strip("/")

        if self.cache_prefix is None:
            self.cache_prefix = f"{self.market_root}/cache/v1"
        else:
            self.cache_prefix = str(self.cache_prefix).strip("/")

        self.mode = str(self.mode).strip().lower()
        if self.mode not in {"full", "incremental"}:
            raise ValueError("mode must be 'full' or 'incremental'.")

        if self.min_years <= 0:
            raise ValueError("min_years must be > 0.")

        if self.min_obs < 1:
            raise ValueError("min_obs must be >= 1.")

        if self.replacement_calendar_days < 0:
            raise ValueError(
                "replacement_calendar_days must be >= 0."
            )

        if self.max_new_assets_full_build < 0:
            raise ValueError(
                "max_new_assets_full_build must be >= 0."
            )

        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1.")

        if self.s3_max_concurrency < 1:
            raise ValueError(
                "s3_max_concurrency must be >= 1."
            )

        if self.progress_every < 1:
            raise ValueError("progress_every must be >= 1.")

        self.universe_csv = str(self.universe_csv)

        if self.excluded_csv is not None:
            self.excluded_csv = str(self.excluded_csv)


RiskGrade = Literal["A", "B", "C", "D", "F"]
ThresholdMode = Literal["absolute", "fraction_of_initial"]
LeverageMode = Literal["none", "gross_exposure", "notional_multiplier"]


def _validate_positive(name: str, x: float) -> float:
    x = float(x)
    if not x > 0.0:
        raise ValueError(f"{name} must be > 0")
    return x


def _validate_non_negative(name: str, x: float) -> float:
    x = float(x)
    if x < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return x


def _validate_probability_like(name: str, x: float) -> float:
    """
    Used for inputs expressed as fractions.

    Examples:
        0.30 = 30%
        0.95 = 95%
    """
    x = float(x)
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return x


def _validate_horizon_days(name: str, x: int) -> int:
    x = int(x)
    if x <= 0:
        raise ValueError(f"{name} must be > 0")
    return x


@dataclass(frozen=True)
class RuinConfig:
    """
    Defines the ruin event.

    In Alpha Edge, ruin does not necessarily mean portfolio value reaches zero.
    It means the account crosses a minimum acceptable equity floor.

    Examples:
        threshold_mode="absolute", threshold_value=10_000
            Ruin occurs when equity <= 10,000.

        threshold_mode="fraction_of_initial", threshold_value=0.50
            Ruin occurs when equity <= 50% of initial capital.
    """

    enabled: bool = True
    threshold_mode: ThresholdMode = "fraction_of_initial"
    threshold_value: float = 0.50

    def validate(self) -> "RuinConfig":
        _validate_positive("ruin.threshold_value", self.threshold_value)

        if self.threshold_mode == "fraction_of_initial":
            if self.threshold_value > 1.0:
                raise ValueError(
                    "ruin.threshold_value must be <= 1 when threshold_mode='fraction_of_initial'"
                )

        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DrawdownBreachConfig:
    """
    Defines a drawdown breach event.

    drawdown_limit_pct is expressed as a positive fraction.

    Example:
        drawdown_limit_pct=0.30 means a -30% drawdown breach.
    """

    enabled: bool = True
    drawdown_limit_pct: float = 0.30

    def validate(self) -> "DrawdownBreachConfig":
        _validate_probability_like("drawdown.drawdown_limit_pct", self.drawdown_limit_pct)

        if self.drawdown_limit_pct <= 0.0:
            raise ValueError("drawdown.drawdown_limit_pct must be > 0")

        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalConfig:
    """
    Defines portfolio goal-reaching events.

    goal_value is absolute portfolio equity.

    Example:
        goal_value=50_000 means the path reaches the goal once equity >= 50,000.
    """

    enabled: bool = True
    goal_value: Optional[float] = None

    def validate(self) -> "GoalConfig":
        if self.enabled and self.goal_value is None:
            raise ValueError("goal.goal_value is required when goal.enabled=True")

        if self.goal_value is not None:
            _validate_positive("goal.goal_value", self.goal_value)

        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecoveryConfig:
    """
    Defines recovery analysis.

    A recovery event asks whether a portfolio path recovers from a drawdown.

    recovery_level can be interpreted as the level that must be recovered to.
    By default, we use 1.0, meaning recovery back to prior peak.
    """

    enabled: bool = True
    recovery_level: float = 1.0

    def validate(self) -> "RecoveryConfig":
        _validate_positive("recovery.recovery_level", self.recovery_level)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SurvivalConfig:
    """
    Defines time-to-event and survival-curve settings.

    horizons_days controls the points where survival probabilities are reported.

    Example:
        horizons_days=[21, 63, 126, 252, 756]
        approximately 1 month, 3 months, 6 months, 1 year, 3 years.
    """

    enabled: bool = True
    horizons_days: list[int] = field(default_factory=lambda: [21, 63, 126, 252, 756])

    def validate(self) -> "SurvivalConfig":
        if self.enabled and not self.horizons_days:
            raise ValueError("survival.horizons_days cannot be empty when survival.enabled=True")

        for h in self.horizons_days:
            _validate_horizon_days("survival.horizons_days item", h)

        if sorted(self.horizons_days) != self.horizons_days:
            raise ValueError("survival.horizons_days must be sorted ascending")

        if len(set(self.horizons_days)) != len(self.horizons_days):
            raise ValueError("survival.horizons_days cannot contain duplicates")

        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapitalAdequacyConfig:
    """
    Defines actuarial capital adequacy and solvency settings.

    target_ruin_probability:
        Maximum acceptable ruin probability.

    target_drawdown_breach_probability:
        Maximum acceptable probability of breaching the drawdown limit.

    min_solvent_capital_ratio:
        Minimum desired ratio of available capital to required capital.

    max_allowed_leverage:
        Hard cap for leverage recommendations.

    leverage_mode:
        Documents how leverage should be interpreted by the caller.
    """

    enabled: bool = True

    target_ruin_probability: float = 0.05
    target_drawdown_breach_probability: float = 0.20

    min_solvent_capital_ratio: float = 1.00

    current_leverage: float = 1.00
    max_allowed_leverage: float = 2.00
    leverage_mode: LeverageMode = "gross_exposure"

    def validate(self) -> "CapitalAdequacyConfig":
        _validate_probability_like(
            "target_ruin_probability",
            self.target_ruin_probability,
        )
        _validate_probability_like(
            "target_drawdown_breach_probability",
            self.target_drawdown_breach_probability,
        )
        _validate_positive(
            "min_solvent_capital_ratio",
            self.min_solvent_capital_ratio,
        )
        _validate_positive(
            "current_leverage",
            self.current_leverage,
        )
        _validate_positive(
            "max_allowed_leverage",
            self.max_allowed_leverage,
        )

        if self.leverage_mode not in {"none", "gross_exposure", "notional_multiplier"}:
            raise ValueError(
                "capital_adequacy.leverage_mode must be one of: "
                "none, gross_exposure, notional_multiplier"
            )

        # Important:
        # current_leverage is observed state.
        # max_allowed_leverage is policy state.
        # current_leverage may exceed max_allowed_leverage and should be reported
        # as a diagnostic risk, not rejected as invalid config.

        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActuarialRiskConfig:
    """
    Top-level configuration for the actuarial risk module.

    This config does not generate simulations. It evaluates already-produced
    portfolio equity paths.

    Expected future input shape:
        equity_paths: 2D array-like
            rows    = simulation paths
            columns = time steps
    """

    initial_value: float
    horizon_days: int
    trading_days_per_year: int = 252

    ruin: RuinConfig = field(default_factory=RuinConfig)
    drawdown: DrawdownBreachConfig = field(default_factory=DrawdownBreachConfig)
    goal: GoalConfig = field(default_factory=lambda: GoalConfig(enabled=False, goal_value=None))
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    survival: SurvivalConfig = field(default_factory=SurvivalConfig)
    capital_adequacy: CapitalAdequacyConfig = field(default_factory=CapitalAdequacyConfig)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "ActuarialRiskConfig":
        _validate_positive("initial_value", self.initial_value)
        _validate_horizon_days("horizon_days", self.horizon_days)
        _validate_horizon_days("trading_days_per_year", self.trading_days_per_year)

        self.ruin.validate()
        self.drawdown.validate()
        self.goal.validate()
        self.recovery.validate()
        self.survival.validate()
        self.capital_adequacy.validate()

        max_horizon = max(self.survival.horizons_days) if self.survival.horizons_days else 0
        if self.survival.enabled and max_horizon > self.horizon_days:
            raise ValueError(
                "survival.horizons_days cannot exceed config.horizon_days. "
                f"max survival horizon={max_horizon}, horizon_days={self.horizon_days}"
            )

        return self

    def ruin_threshold_value(self) -> float:
        """
        Resolves the actual portfolio equity value at which ruin occurs.
        """
        if self.ruin.threshold_mode == "absolute":
            return float(self.ruin.threshold_value)

        if self.ruin.threshold_mode == "fraction_of_initial":
            return float(self.initial_value) * float(self.ruin.threshold_value)

        raise ValueError(f"Unsupported ruin threshold mode: {self.ruin.threshold_mode!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SurvivalCurvePoint:
    """
    One point in a survival curve.

    survival_probability:
        Probability that the path has not hit the event by horizon_days.

    event_probability:
        Probability that the event has happened by horizon_days.
    """

    horizon_days: int
    survival_probability: Optional[float]
    event_probability: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActuarialRiskResult:
    """
    Structured result returned by the actuarial risk engine.

    This is intentionally broad. Step 1 only defines the contract.
    Step 2 and later will populate these fields.
    """

    initial_value: float
    horizon_days: int
    n_paths: int

    ruin_threshold: Optional[float] = None
    ruin_probability: Optional[float] = None
    expected_time_to_ruin_days: Optional[float] = None
    median_time_to_ruin_days: Optional[float] = None

    drawdown_limit_pct: Optional[float] = None
    drawdown_breach_probability: Optional[float] = None
    expected_max_drawdown: Optional[float] = None
    median_max_drawdown: Optional[float] = None
    cvar_max_drawdown_95: Optional[float] = None

    goal_value: Optional[float] = None
    goal_probability: Optional[float] = None
    median_time_to_goal_days: Optional[float] = None
    probability_goal_before_ruin: Optional[float] = None

    recovery_probability: Optional[float] = None
    median_recovery_time_days: Optional[float] = None

    capital_required: Optional[float] = None
    capital_buffer_gap: Optional[float] = None
    solvency_ratio: Optional[float] = None
    safe_leverage_estimate: Optional[float] = None

    survival_curve: list[SurvivalCurvePoint] = field(default_factory=list)

    risk_grade: Optional[RiskGrade] = None
    warnings: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "ActuarialRiskResult":
        _validate_positive("result.initial_value", self.initial_value)
        _validate_horizon_days("result.horizon_days", self.horizon_days)

        if self.n_paths <= 0:
            raise ValueError("result.n_paths must be > 0")

        for name in [
            "ruin_probability",
            "drawdown_breach_probability",
            "goal_probability",
            "probability_goal_before_ruin",
            "recovery_probability",
        ]:
            value = getattr(self, name)
            if value is not None:
                _validate_probability_like(f"result.{name}", float(value))

        for point in self.survival_curve:
            _validate_horizon_days("result.survival_curve.horizon_days", point.horizon_days)

            if point.survival_probability is not None:
                _validate_probability_like(
                    "result.survival_curve.survival_probability",
                    point.survival_probability,
                )

            if point.event_probability is not None:
                _validate_probability_like(
                    "result.survival_curve.event_probability",
                    point.event_probability,
                )

        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)

        # dataclasses.asdict already converts nested SurvivalCurvePoint instances,
        # but keeping this explicit makes the serialized contract obvious.
        d["survival_curve"] = [p.to_dict() for p in self.survival_curve]

        return d

@dataclass(frozen=True)
class ActuarialDiagnosticReport:
    """
    Human-readable and machine-readable diagnostic summary of an actuarial risk result.

    This object is intentionally separate from ActuarialRiskResult.

    ActuarialRiskResult is the full quantitative output.
    ActuarialDiagnosticReport is the reporting/diagnostic layer that can be attached
    to portfolio-search outputs, daily reports, or future warehouse rows.
    """

    portfolio_id: Optional[str]
    run_id: Optional[str]
    source: str

    verdict: Literal["pass", "warn", "fail"]

    risk_grade: Optional[RiskGrade]
    risk_flags: list[str]
    warnings: list[str]

    headline_metrics: Dict[str, Optional[float]]
    detail_metrics: Dict[str, Optional[float]]

    result: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "ActuarialDiagnosticReport":
        if self.verdict not in {"pass", "warn", "fail"}:
            raise ValueError("diagnostic_report.verdict must be one of: pass, warn, fail")

        if not self.source:
            raise ValueError("diagnostic_report.source is required")

        if not isinstance(self.risk_flags, list):
            raise ValueError("diagnostic_report.risk_flags must be a list")

        if not isinstance(self.warnings, list):
            raise ValueError("diagnostic_report.warnings must be a list")

        if not isinstance(self.headline_metrics, dict):
            raise ValueError("diagnostic_report.headline_metrics must be a dict")

        if not isinstance(self.detail_metrics, dict):
            raise ValueError("diagnostic_report.detail_metrics must be a dict")

        if not isinstance(self.result, dict):
            raise ValueError("diagnostic_report.result must be a dict")

        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ActuarialDiagnosticBatchReport:
    """
    Batch-level wrapper for actuarial diagnostic reports.

    This is intended for portfolio-search diagnostic persistence.

    It stores:
      - one manifest-like summary,
      - many individual diagnostic reports,
      - flattened summary rows for CSV/reporting use.
    """

    run_id: Optional[str]
    source: str
    n_reports: int
    reports: list[Dict[str, Any]]
    summary_rows: list[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "ActuarialDiagnosticBatchReport":
        if not self.source:
            raise ValueError("batch_report.source is required")

        if self.n_reports < 0:
            raise ValueError("batch_report.n_reports must be >= 0")

        if self.n_reports != len(self.reports):
            raise ValueError("batch_report.n_reports must match len(reports)")

        if self.n_reports != len(self.summary_rows):
            raise ValueError("batch_report.n_reports must match len(summary_rows)")

        if not isinstance(self.reports, list):
            raise ValueError("batch_report.reports must be a list")

        if not isinstance(self.summary_rows, list):
            raise ValueError("batch_report.summary_rows must be a list")

        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class PortfolioEvaluationResult:
    """
    Extended portfolio evaluation result used when downstream diagnostics need
    raw simulated equity paths.

    EvalMetrics remains the lightweight standard output used by GA/search.

    This wrapper should be used only when path-level diagnostics are needed,
    for example actuarial diagnostics on the final selected executable portfolio.
    """

    metrics: EvalMetrics
    equity_paths: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "PortfolioEvaluationResult":
        if self.metrics is None:
            raise ValueError("PortfolioEvaluationResult.metrics is required")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": asdict(self.metrics),
            "equity_paths": self.equity_paths,
            "metadata": dict(self.metadata),
        }


TransitionRecommendation = Literal[
    "HOLD",
    "LOCAL_OPTIMIZATION_RECOMMENDED",
    "FULL_SEARCH_REQUIRED",
    "SHADOW_PORTFOLIO_ACTIVE",
    "SHADOW_PORTFOLIO_ACCEPTED",
    "DEFENSIVE_MODE_REQUIRED",
]


@dataclass
class PortfolioTransitionConfig:
    min_health_score: float = 60.0
    health_drop_trigger: float = 15.0
    min_grade: str = "B"
    max_ruin_probability: float = 0.10
    max_drawdown_limit: float = 0.30
    full_search_refresh_days: int = 20
    shadow_confirmation_days: int = 3
    min_shadow_health_advantage: float = 5.0
    max_daily_turnover: float = 0.10
    min_local_improvement: float = 3.0

    # Regime-aware transition logic
    regime_change_requires_full_search: bool = True
    min_regime_confidence_for_full_search: float = 0.60


@dataclass
class CurrentPortfolioState:
    as_of: str
    portfolio_id: Optional[str]
    health_score: Optional[float]
    grade: Optional[str]
    optimizer_score: Optional[float]
    ruin_probability: Optional[float]
    max_drawdown: Optional[float]
    avg_max_drawdown: Optional[float]
    cdar_95: Optional[float]
    volatility: Optional[float]
    annual_return: Optional[float]
    hhi: Optional[float]
    correlation: Optional[float]

    # Current regime state
    regime: Optional[str]
    previous_regime: Optional[str] = None
    regime_changed: bool = False
    regime_confidence: Optional[float] = None

    original_health_score: Optional[float] = None
    days_since_full_search: Optional[int] = None
    local_optimizer_failed_days: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionAssessment:
    as_of: str
    recommendation: TransitionRecommendation
    reason: str
    current_state: CurrentPortfolioState
    full_search_required: bool
    local_optimization_allowed: bool
    shadow_portfolio_required: bool
    delta_execution_allowed: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


LocalOptimizerRecommendation = Literal[
    "HOLD",
    "LOCAL_REBALANCE_RECOMMENDED",
    "LOCAL_OPTIMIZATION_REJECTED",
]


@dataclass
class LocalTransitionOptimizerConfig:
    random_seed: int = 123

    # Existing annealing parameters from portfolio_search.py
    anneal_steps: int = 100
    temp_start: float = 0.50
    temp_end: float = 0.03
    n_paths_current: int = 5000
    n_paths_init: int = 2000
    n_paths_final: int = 8000
    path_source: str = "pca"
    pca_k: int | None = 5
    block_size: tuple[int, int] = (8, 12)

    # Portfolio shape constraints
    max_assets: int = 10
    min_assets: int = 5
    weight_mode: str = "long_short"

    # Decision thresholds
    max_turnover: float = 0.10
    min_score_improvement: float = 0.02
    min_health_improvement: float = 3.0

@dataclass
class LocalTransitionCandidate:
    weights: dict[str, float]
    score: float
    health_score: Optional[float]
    turnover: float
    score_improvement: float
    health_improvement: Optional[float]
    delta_weights: dict[str, float]
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalTransitionOptimizerResult:
    as_of: str
    recommendation: LocalOptimizerRecommendation
    reason: str
    current_weights: dict[str, float]
    current_score: float
    current_health_score: Optional[float]
    best_candidate: Optional[LocalTransitionCandidate]
    candidates_evaluated: int
    candidates_accepted_by_turnover: int
    config: LocalTransitionOptimizerConfig
    diagnostics: dict[str, Any] = field(default_factory=dict)

ShadowPortfolioRecommendation = Literal[
    "KEEP_CURRENT",
    "SHADOW_ACTIVE",
    "SHADOW_ACCEPTED",
    "SHADOW_REJECTED",
]


@dataclass
class ShadowPortfolioConfig:
    min_health_advantage: float = 5.0
    min_score_advantage: float = 0.02
    max_turnover_to_accept: float = 0.35
    confirmation_days: int = 3
    immediate_accept_health_advantage: float = 10.0
    immediate_accept_score_advantage: float = 0.05


@dataclass
class ShadowPortfolioState:
    shadow_id: str
    as_of: str
    source_run_id: Optional[str]
    source_run_key: Optional[str]
    status: str

    current_health_score: Optional[float]
    shadow_health_score: Optional[float]
    health_advantage: Optional[float]

    current_score: Optional[float]
    shadow_score: Optional[float]
    score_advantage: Optional[float]

    turnover: Optional[float]
    days_active: int = 1
    days_dominating: int = 0

    current_weights: dict[str, float] = field(default_factory=dict)
    shadow_weights: dict[str, float] = field(default_factory=dict)
    delta_weights: dict[str, float] = field(default_factory=dict)

    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowPortfolioAssessment:
    as_of: str
    recommendation: ShadowPortfolioRecommendation
    reason: str
    state: ShadowPortfolioState
    config: ShadowPortfolioConfig
    diagnostics: dict[str, Any] = field(default_factory=dict)


TransitionExecutionRecommendation = Literal[
    "NO_TRADE",
    "TRADE_RECOMMENDED",
    "TRADE_BLOCKED",
]


TradeDirection = Literal[
    "BUY",
    "SELL",
    "HOLD",
]


@dataclass
class TransitionExecutionConfig:
    max_total_turnover: float = 0.35
    max_daily_turnover: float = 0.10
    min_trade_value: float = 25.0
    min_trade_quantity: float = 0.0
    allow_partial_transition: bool = True


@dataclass
class TransitionTradeDelta:
    asset_id: str
    direction: TradeDirection

    current_quantity: float
    target_quantity: float
    delta_quantity: float

    price: Optional[float]
    current_value: float
    target_value: float
    delta_value: float

    current_weight: Optional[float] = None
    target_weight: Optional[float] = None
    delta_weight: Optional[float] = None

    reason: str = ""


@dataclass
class TransitionExecutionPlan:
    as_of: str
    recommendation: TransitionExecutionRecommendation
    reason: str
    source: str

    notional: float
    total_turnover: float
    daily_turnover_used: float
    blocked_turnover: float

    target_allocation: DiscreteAllocation

    trades: list[TransitionTradeDelta] = field(default_factory=list)
    blocked_trades: list[TransitionTradeDelta] = field(default_factory=list)

    config: TransitionExecutionConfig = field(default_factory=TransitionExecutionConfig)
    diagnostics: dict[str, Any] = field(default_factory=dict)