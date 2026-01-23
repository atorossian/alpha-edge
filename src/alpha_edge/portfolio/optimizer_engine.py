# optimizer_engine.py
from __future__ import annotations

from typing import Dict, Sequence
import numpy as np
import pandas as pd

from alpha_edge.core.schemas import EvalMetrics, Goals3, ScoreConfig, StabilityEnergyConfig, StabilityReport
from alpha_edge.portfolio.stability_energy import compute_stability_report
from alpha_edge.market.mc_engine import simulate_leveraged_paths_vectorized
from alpha_edge.market.factor_engine import fit_pca_model, sample_portfolio_returns_paths_pca
from alpha_edge.market.stats_engine import compute_lw_cov_df


def _lw_cov_subset(returns: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    sub = returns[tickers].dropna(how="any")
    if sub.shape[1] < 2:
        v = float(sub.var(ddof=1).iloc[0]) if sub.shape[1] == 1 else 0.0
        return np.array([[v]], dtype=np.float64)
    cov_df = compute_lw_cov_df(sub)
    return cov_df.loc[tickers, tickers].values.astype(np.float64, copy=False)


# --- FFT / spectral helpers -------------------------------------------------
# (unchanged)
def _fft_band_powers_from_returns(
    r: np.ndarray,
    bands_days: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (2.0, 20.0),
        (20.0, 60.0),
        (60.0, 250.0),
    ),
    eps: float = 1e-12,
) -> dict:
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 64:
        return {"hf": 0.0, "mf": 0.0, "lf": 0.0, "entropy": 0.0}

    x = r - r.mean()

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0)

    P = (X.real * X.real + X.imag * X.imag)

    freqs = freqs[1:]
    P = P[1:]

    if P.size == 0 or not np.isfinite(P).any():
        return {"hf": 0.0, "mf": 0.0, "lf": 0.0, "entropy": 0.0}

    with np.errstate(divide="ignore", invalid="ignore"):
        periods = 1.0 / freqs

    total = float(np.nansum(P))
    if not np.isfinite(total) or total <= eps:
        return {"hf": 0.0, "mf": 0.0, "lf": 0.0, "entropy": 0.0}

    (hf_min, hf_max), (mf_min, mf_max), (lf_min, lf_max) = bands_days

    hf_mask = (periods >= hf_min) & (periods < hf_max)
    mf_mask = (periods >= mf_min) & (periods < mf_max)
    lf_mask = (periods >= lf_min) & (periods <= lf_max)

    hf = float(np.nansum(P[hf_mask]) / total)
    mf = float(np.nansum(P[mf_mask]) / total)
    lf = float(np.nansum(P[lf_mask]) / total)

    band_sum = hf + mf + lf
    if not np.isfinite(band_sum) or band_sum <= eps:
        return {"hf": 0.0, "mf": 0.0, "lf": 0.0, "entropy": 0.0}

    hf /= band_sum
    mf /= band_sum
    lf /= band_sum

    p = np.array([hf, mf, lf], dtype=np.float64)
    entropy = float(-np.sum(p * np.log(p + eps)))

    return {"hf": hf, "mf": mf, "lf": lf, "entropy": entropy}


def _spectral_profiles_df(
    asset_rets: pd.DataFrame,
    bands_days: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (2.0, 20.0),
        (20.0, 60.0),
        (60.0, 250.0),
    ),
) -> pd.DataFrame:
    rows = {}
    for t in asset_rets.columns:
        rows[t] = _fft_band_powers_from_returns(asset_rets[t].to_numpy(), bands_days=bands_days)
    df = pd.DataFrame.from_dict(rows, orient="index")
    for c in ("hf", "mf", "lf", "entropy"):
        if c not in df.columns:
            df[c] = 0.0
    return df


def _frequency_overlap_penalty(w: np.ndarray, V: np.ndarray) -> float:
    if V.shape[0] < 2:
        return 0.0
    S = V @ V.T
    total = float(w @ S @ w)
    diag = float(np.sum((w * w) * np.diag(S)))
    val = total - diag
    return float(val) if np.isfinite(val) else 0.0


def compute_penalties(
    w: np.ndarray,
    stats: dict,
    mc: dict,
    asset_rets: pd.DataFrame,
    goals: list[float],
    cfg: ScoreConfig | None = None,
    spec_df_full: pd.DataFrame | None = None,
) -> dict:
    if cfg is None:
        cfg = ScoreConfig()

    # IMPORTANT: for long/short we penalize concentration on GROSS weights
    w_abs = np.abs(w)
    hhi = float(np.sum(w_abs**2))

    corr = asset_rets.corr().values
    n = corr.shape[0]
    avg_corr = float((corr[np.triu_indices(n, 1)]).mean()) if n > 1 else 0.0

    cvar95 = float(stats["cvar_95"])
    mdd = float(stats["max_drawdown"])

    g2 = float(goals[1])
    med_t = mc["goal_stats"].get(g2, {}).get("median_time_days", None)

    pen: dict[str, float] = {}

    pen["hhi"] = max(0.0, hhi - cfg.hhi_cap)
    pen["avg_corr"] = max(0.0, avg_corr - cfg.corr_cap)
    pen["cvar"] = max(0.0, (-cvar95) - cfg.cvar_cap)
    pen["mdd"] = max(0.0, (-mdd) - cfg.mdd_cap)

    if med_t is None:
        pen["time"] = 1.0
    else:
        pen["time"] = max(0.0, (float(med_t) - cfg.time_cap_days) / 252.0)

    pen["hf_ratio"] = 0.0
    pen["spec_entropy"] = 0.0
    pen["freq_overlap"] = 0.0

    try:
        if spec_df_full is not None:
            spec_df = spec_df_full.loc[list(asset_rets.columns)]
        else:
            spec_df = _spectral_profiles_df(asset_rets, bands_days=cfg.fft_bands_days)

        if len(spec_df) > 0:
            # use ABS weights for spectral composition penalties too
            hf_ratio = float(np.sum(w_abs * spec_df["hf"].to_numpy(dtype=np.float64)))
            ent = float(np.sum(w_abs * spec_df["entropy"].to_numpy(dtype=np.float64)))

            pen["hf_ratio"] = max(0.0, hf_ratio - cfg.hf_ratio_cap)
            pen["spec_entropy"] = max(0.0, ent - cfg.spec_entropy_cap)

            V = spec_df[["hf", "mf", "lf"]].to_numpy(dtype=np.float64)
            overlap = _frequency_overlap_penalty(w_abs, V)
            pen["freq_overlap"] = max(0.0, overlap - cfg.freq_overlap_cap)

    except Exception:
        pass

    return pen


def score_candidate(p_hit_main: float, ruin: float, penalties: dict, cfg: ScoreConfig | None = None) -> float:
    if cfg is None:
        cfg = ScoreConfig()

    return (
        p_hit_main
        - cfg.lambda_ruin * ruin
        - cfg.lambda_cvar * penalties["cvar"]
        - cfg.lambda_mdd * penalties["mdd"]
        - cfg.lambda_conc * penalties["hhi"]
        - cfg.lambda_corr * penalties["avg_corr"]
        - cfg.lambda_time * penalties["time"]
        - cfg.lambda_hf_ratio * penalties.get("hf_ratio", 0.0)
        - cfg.lambda_freq_overlap * penalties.get("freq_overlap", 0.0)
        - cfg.lambda_spec_entropy * penalties.get("spec_entropy", 0.0)
    )


def _normalize_weights_gross(weights: Dict[str, float], tickers: Sequence[str]) -> np.ndarray:
    w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=np.float64)
    s = float(np.sum(np.abs(w)))
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Gross weight sum <= 0 after aligning (need some exposure)")
    return w / s


# ---------------------------------------------------------------------------
# NEW: array-based evaluator used by GA workers to avoid duplicating returns_wide
# ---------------------------------------------------------------------------

def evaluate_portfolio_from_arrays(
    *,
    rets_assets: np.ndarray,
    tickers: Sequence[str],
    weights: Dict[str, float],
    equity0: float,
    notional: float,
    goals: tuple[float, float, float],
    main_goal: float,
    score_config: ScoreConfig,
    mc_seed: int | None,
    spec_rows: np.ndarray | None = None,
    days: int = 252,
    n_paths: int = 5000,
    block_size: int | tuple[int, int] | None = (8, 12),
    weight_mode: str = "long_only"
) -> EvalMetrics:

    k = len(tickers)
    if k == 0:
        raise ValueError("Empty tickers for candidate")

    X = np.asarray(rets_assets, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != k:
        raise ValueError(f"rets_assets must be (T,k) with k={k}, got {X.shape}")

    w_vec = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=np.float64)
    w_vec = np.where(np.isfinite(w_vec), w_vec, 0.0)

    if weight_mode == "gross_signed":
        gross = float(np.sum(np.abs(w_vec)))
        if not np.isfinite(gross) or gross <= 0:
            raise ValueError("Gross weight exposure <= 0")
        w_vec = w_vec / gross
    else:
        s = float(np.sum(w_vec))
        if not np.isfinite(s) or s <= 0:
            raise ValueError("Weights must sum to a positive number")
        w_vec = w_vec / s
    # ---- portfolio returns ----
    port_rets = X @ w_vec
    port_rets = port_rets[np.isfinite(port_rets)]
    if port_rets.size < 50:
        raise ValueError("Not enough data for candidate")

    daily_mean = float(np.mean(port_rets))
    daily_std = float(np.std(port_rets, ddof=1)) if port_rets.size > 1 else 0.0

    ann_return = float(daily_mean * 252.0)
    ann_vol = float(daily_std * np.sqrt(252.0))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")

    neg = port_rets[port_rets < 0]
    downside_std = float(np.std(neg, ddof=1)) if neg.size > 1 else 0.0
    ann_down = float(downside_std * np.sqrt(252.0))
    sortino = float(ann_return / ann_down) if ann_down > 0 else float("nan")

    cum = np.cumprod(1.0 + port_rets)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    max_dd = float(np.min(dd)) if dd.size else 0.0

    var_95 = float(np.percentile(port_rets, 5))
    cvar_95 = float(np.mean(port_rets[port_rets <= var_95])) if np.any(port_rets <= var_95) else float("nan")

    # ---- Ledoit-Wolf annual vol (match evaluate_portfolio behavior) ----
    ann_vol_lw = float("nan")
    try:
        # local import to keep module import-time light for workers
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(X)
        cov = lw.covariance_.astype(np.float64, copy=False)
        port_var_daily = float(w_vec @ cov @ w_vec)
        ann_vol_lw = float(np.sqrt(max(port_var_daily, 0.0) * 252.0))
    except Exception:
        ann_vol_lw = float("nan")

    # ---- MC ----
    mc = simulate_leveraged_paths_vectorized(
        port_rets=port_rets.astype(np.float32, copy=False),
        precomputed_r=None,
        notional=float(notional),
        equity0=float(equity0),
        goals=[float(goals[0]), float(goals[1]), float(goals[2])],
        n_days=int(days),
        n_paths=int(n_paths),
        seed=mc_seed,
        block_size=block_size,
        return_paths=False,
    )

    ruin = float(mc["ruin_prob"])
    p_main = float(mc["p_hit"].get(float(main_goal), 0.0))

    # ---- penalties/scoring (use the SAME machinery) ----
    # build a small DataFrame view ONLY for penalty routines that need asset corr
    # (k is small, so this is cheap and keeps logic identical)
    asset_rets_df = pd.DataFrame(X, columns=list(tickers))

    stats_for_pen = {"cvar_95": cvar_95, "max_drawdown": max_dd}

    penalties = compute_penalties(
        w=w_vec,  # already normalized in chosen mode
        stats=stats_for_pen,
        mc=mc,
        asset_rets=asset_rets_df,
        goals=[float(goals[0]), float(goals[1]), float(goals[2])],
        cfg=score_config,
        spec_df_full=None,  # we use spec_rows below for speed
    )

    # override spectral penalties with spec_rows fast path if provided
    if spec_rows is not None and spec_rows.shape[0] == k and spec_rows.shape[1] >= 4:
        cfg = score_config
        w_abs = np.abs(w_vec)

        hf = spec_rows[:, 0].astype(np.float64, copy=False)
        mf = spec_rows[:, 1].astype(np.float64, copy=False)
        lf = spec_rows[:, 2].astype(np.float64, copy=False)
        ent = spec_rows[:, 3].astype(np.float64, copy=False)

        hf_ratio = float(np.sum(w_abs * hf))
        ent_w = float(np.sum(w_abs * ent))

        penalties["hf_ratio"] = max(0.0, hf_ratio - cfg.hf_ratio_cap)
        penalties["spec_entropy"] = max(0.0, ent_w - cfg.spec_entropy_cap)

        V = np.stack([hf, mf, lf], axis=1)
        S = V @ V.T
        total = float(w_abs @ S @ w_abs)
        diag = float(np.sum((w_abs * w_abs) * np.diag(S)))
        overlap = total - diag
        if np.isfinite(overlap):
            penalties["freq_overlap"] = max(0.0, float(overlap) - cfg.freq_overlap_cap)

    score = score_candidate(p_main, ruin, penalties, score_config)

    g1, g2, g3 = float(goals[0]), float(goals[1]), float(goals[2])

    return EvalMetrics(
        weights=dict(weights),
        goals=(g1, g2, g3),
        main_goal=float(main_goal),

        ann_return=float(ann_return),
        ann_vol=float(ann_vol),
        ann_vol_lw=float(ann_vol_lw),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(max_dd),

        var_95=float(var_95),
        cvar_95=float(cvar_95),

        ruin_prob_1y=float(ruin),
        p_hit_goal_1_1y=float(mc["p_hit"].get(g1, 0.0)),
        p_hit_goal_2_1y=float(mc["p_hit"].get(g2, 0.0)),
        p_hit_goal_3_1y=float(mc["p_hit"].get(g3, 0.0)),
        med_t_goal_1_days=mc["goal_stats"][g1]["median_time_days"],
        med_t_goal_2_days=mc["goal_stats"][g2]["median_time_days"],
        med_t_goal_3_days=mc["goal_stats"][g3]["median_time_days"],

        ending_equity_p5=float(mc["end_p5"]),
        ending_equity_p25=float(mc["end_p25"]),
        ending_equity_p50=float(mc["end_p50"]),
        ending_equity_p75=float(mc["end_p75"]),
        ending_equity_p95=float(mc["end_p95"]),

        score=float(score),
    )


# ---------------------------------------------------------------------------
# Existing API
# ---------------------------------------------------------------------------

def compute_stability_for_candidate(
    *,
    returns: pd.DataFrame,
    weights: Dict[str, float],
    equity0: float,
    notional: float,
    goals: list[float] | Goals3,
    days: int = 252,
    n_paths: int = 20000,
    mc_seed: int | None = None,
    path_source: str = "bootstrap",
    pca_k: int = 5,
    block_size: int | tuple[int, int] | None = (8, 12),
    stability_cfg: StabilityEnergyConfig | None = None,
) -> StabilityReport:
    if stability_cfg is None:
        stability_cfg = StabilityEnergyConfig()

    tickers = [t for t in weights.keys() if t in returns.columns]
    if not tickers:
        raise ValueError("No overlap between weights and returns columns")

    w_vec = _normalize_weights_gross(weights, tickers)

    rets_assets = returns[tickers].dropna(how="any")
    port_rets = (rets_assets.to_numpy(dtype=np.float64) @ w_vec).astype(np.float64, copy=False)
    port_rets = port_rets[np.isfinite(port_rets)]
    if port_rets.size < 50:
        raise ValueError("Not enough data for stability eval")

    if path_source == "bootstrap":
        mc = simulate_leveraged_paths_vectorized(
            port_rets=port_rets.astype(np.float64, copy=False),
            notional=float(notional),
            equity0=float(equity0),
            goals=[float(g) for g in goals],
            n_days=int(days),
            n_paths=int(n_paths),
            seed=mc_seed,
            block_size=block_size,
            return_paths=True,
        )

    elif path_source == "pca":
        pca_model = fit_pca_model(rets_assets, k=int(pca_k))

        R_port_paths = sample_portfolio_returns_paths_pca(
            model=pca_model,
            weights=weights,
            n_paths=n_paths,
            n_days=days,
            seed=mc_seed,
            block_size=block_size,
        )

        mc = simulate_leveraged_paths_vectorized(
            port_rets=None,
            precomputed_r=R_port_paths,
            notional=float(notional),
            equity0=float(equity0),
            goals=[float(g) for g in goals],
            n_days=int(days),
            n_paths=int(n_paths),
            seed=mc_seed,
            block_size=None,
            return_paths=True,
        )

    else:
        raise ValueError(f"Unknown path_source={path_source!r}")

    rep = compute_stability_report(
        mc["equity_paths"],
        n_days=days,
        cfg=stability_cfg,
    )
    return rep


def evaluate_portfolio_candidate(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    equity0: float,
    notional: float,
    goals: list[float],
    main_goal: float,
    lw_cov: pd.DataFrame | None = None,
    days: int = 252,
    n_paths: int = 20000,
    score_config: ScoreConfig | None = None,
    mc_seed: int | None = None,
    path_source: str = "bootstrap",
    pca_k: int = 5,
    block_size: int | tuple[int, int] | None = (8, 12),
    spec_df_full: pd.DataFrame | None = None,
) -> EvalMetrics:
    return evaluate_portfolio(
        returns=returns,
        weights=weights,
        equity0=equity0,
        notional=notional,
        goals=goals,
        main_goal=main_goal,
        lw_cov=lw_cov,
        days=days,
        n_paths=n_paths,
        score_config=score_config,
        mc_seed=mc_seed,
        block_size=block_size,
        path_source=path_source,
        pca_k=pca_k,
        spec_df_full=spec_df_full,
    )


def evaluate_portfolio(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    equity0: float,
    notional: float,
    goals: list[float] | Goals3,
    main_goal: float,
    lw_cov: pd.DataFrame | None = None,
    days: int = 252,
    n_paths: int = 20000,
    score_config: ScoreConfig | None = None,
    mc_seed: int | None = None,
    block_size: int | tuple[int, int] | None = (8, 12),
    path_source: str = "bootstrap",
    pca_k: int = 5,
    spec_df_full: pd.DataFrame | None = None,
    weight_mode: str = "long_only",   # <<<<<< DEFAULT CHANGED
) -> EvalMetrics:

    goals = tuple(float(g) for g in goals)
    if len(goals) != 3:
        raise ValueError("For now, exactly 3 goals are required.")
    main_goal = float(main_goal)

    if score_config is None:
        score_config = ScoreConfig()

    tickers = [t for t in weights.keys() if t in returns.columns]
    if not tickers:
        raise ValueError("No overlap between weights and returns columns")

    w_vec = np.array([float(weights[t]) for t in tickers], dtype=np.float64)
    w_vec = np.where(np.isfinite(w_vec), w_vec, 0.0)

    if weight_mode == "gross_signed":
        gross = float(np.sum(np.abs(w_vec)))
        if not np.isfinite(gross) or gross <= 0:
            raise ValueError("Gross weight exposure <= 0 (check positions/prices)")
        w_vec = w_vec / gross
    else:
        s = float(np.sum(w_vec))
        if not np.isfinite(s) or s <= 0:
            raise ValueError("Weights must sum to a positive number")
        w_vec = w_vec / s

    rets_assets = returns[tickers].dropna(how="any")
    port_rets = (rets_assets.to_numpy(dtype=np.float64) @ w_vec).astype(np.float64, copy=False)
    port_rets = port_rets[np.isfinite(port_rets)]

    if port_rets.size < 50:
        raise ValueError("Not enough data for this portfolio candidate")

    daily_mean = float(np.mean(port_rets))
    daily_std = float(np.std(port_rets, ddof=1)) if port_rets.size > 1 else 0.0

    ann_return = float(daily_mean * 252.0)
    ann_vol = float(daily_std * np.sqrt(252.0))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")

    neg = port_rets[port_rets < 0]
    downside_std = float(np.std(neg, ddof=1)) if neg.size > 1 else 0.0
    ann_down = float(downside_std * np.sqrt(252.0))
    sortino = float(ann_return / ann_down) if ann_down > 0 else float("nan")

    cum = np.cumprod(1.0 + port_rets)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    max_dd = float(np.min(dd)) if dd.size else 0.0

    var_95 = float(np.percentile(port_rets, 5))
    cvar_95 = float(np.mean(port_rets[port_rets <= var_95])) if np.any(port_rets <= var_95) else float("nan")

    ann_vol_lw = float("nan")
    try:
        if lw_cov is None:
            cov_sub = _lw_cov_subset(returns, tickers)
            w_lw = w_vec
        else:
            lw_tickers = [t for t in tickers if t in lw_cov.columns]
            if len(lw_tickers) >= 1:
                w_lw = _normalize_weights_gross(weights, lw_tickers)
                cov_sub = lw_cov.loc[lw_tickers, lw_tickers].values.astype(np.float64, copy=False)
            else:
                cov_sub = None

        if cov_sub is not None and cov_sub.size > 0:
            port_var_daily = float(w_lw @ cov_sub @ w_lw)
            ann_vol_lw = float(np.sqrt(max(port_var_daily, 0.0) * 252.0))
    except Exception:
        ann_vol_lw = float("nan")

    if path_source == "bootstrap":
        mc = simulate_leveraged_paths_vectorized(
            port_rets=port_rets.astype(np.float64, copy=False),
            notional=float(notional),
            equity0=float(equity0),
            goals=list(goals),
            n_days=int(days),
            n_paths=int(n_paths),
            seed=mc_seed,
            block_size=block_size,
            return_paths=False,
        )

    elif path_source == "pca":
        pca_model = fit_pca_model(rets_assets, k=int(pca_k))

        R_port_paths = sample_portfolio_returns_paths_pca(
            model=pca_model,
            weights=weights,
            n_paths=n_paths,
            n_days=days,
            seed=mc_seed,
            block_size=block_size,
        )

        mc = simulate_leveraged_paths_vectorized(
            port_rets=None,
            precomputed_r=R_port_paths,
            notional=float(notional),
            equity0=float(equity0),
            goals=list(goals),
            n_days=int(days),
            n_paths=int(n_paths),
            seed=mc_seed,
            block_size=None,
            return_paths=False,
        )

    else:
        raise ValueError(f"Unknown path_source={path_source!r} (use 'bootstrap' or 'pca')")

    ruin = float(mc["ruin_prob"])
    p_main = float(mc["p_hit"].get(main_goal, 0.0))

    stats_for_pen = {"cvar_95": cvar_95, "max_drawdown": max_dd}

    penalties = compute_penalties(
        w=w_vec,
        stats=stats_for_pen,
        mc=mc,
        asset_rets=rets_assets,
        goals=list(goals),
        cfg=score_config,
        spec_df_full=spec_df_full,
    )

    score = score_candidate(p_main, ruin, penalties, score_config)

    g1, g2, g3 = goals

    return EvalMetrics(
        weights=dict(weights),
        goals=goals,
        main_goal=main_goal,

        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,

        ann_vol_lw=ann_vol_lw,

        var_95=var_95,
        cvar_95=cvar_95,

        ruin_prob_1y=ruin,
        p_hit_goal_1_1y=float(mc["p_hit"].get(g1, 0.0)),
        p_hit_goal_2_1y=float(mc["p_hit"].get(g2, 0.0)),
        p_hit_goal_3_1y=float(mc["p_hit"].get(g3, 0.0)),
        med_t_goal_1_days=mc["goal_stats"][g1]["median_time_days"],
        med_t_goal_2_days=mc["goal_stats"][g2]["median_time_days"],
        med_t_goal_3_days=mc["goal_stats"][g3]["median_time_days"],

        ending_equity_p5=float(mc["end_p5"]),
        ending_equity_p25=float(mc["end_p25"]),
        ending_equity_p50=float(mc["end_p50"]),
        ending_equity_p75=float(mc["end_p75"]),
        ending_equity_p95=float(mc["end_p95"]),

        score=float(score),
    )
