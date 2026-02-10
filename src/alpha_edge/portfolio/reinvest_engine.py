# alpha_edge/portfolio/reinvest_engine.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from alpha_edge.universe.universe import Asset
from alpha_edge.core.schemas import ScoreConfig, Position
from alpha_edge.portfolio.portfolio_search import (
    sample_random_weights,
    crossover_weights,
    mutate_weights,
)
from alpha_edge.portfolio.optimizer_engine import _spectral_profiles_df, evaluate_portfolio_from_arrays
from alpha_edge import paths


def _gross_notional_from_qty(*, qty: dict[str, float], px: pd.Series) -> float:
    s = 0.0
    for t, q in qty.items():
        p = float(px.get(t, np.nan))
        if np.isfinite(p) and p > 0 and np.isfinite(q):
            s += abs(float(q) * p)
    return float(s)


def _load_universe_include1(*, returns_cols: set[str]) -> dict[str, Asset]:
    u = pd.read_csv(paths.universe_dir() / "universe.csv")
    if "include" in u.columns:
        u = u[u["include"].fillna(1).astype(int) == 1].copy()

    # normalize ticker column
    if "ticker" not in u.columns:
        u["ticker"] = u["asset_id"]
    u["ticker"] = u["ticker"].astype(str).str.upper().str.strip()

    universe: dict[str, Asset] = {}
    for _, row in u.iterrows():
        t = str(row["ticker"]).upper().strip()
        if not t or t not in returns_cols:
            continue
        universe[t] = Asset(
            ticker=t,
            yahoo_ticker=row.get("yahoo_ticker"),
            name=row.get("name"),
            asset_class=row.get("asset_class"),
            role=row.get("role"),
            region=row.get("region"),
            max_weight=float(row.get("max_weight", 1.0) or 1.0),
            min_weight=float(row.get("min_weight", 0.0) or 0.0),
            include=True,
        )
    return universe


def _infer_core_from_asset_tp_plan(
    *,
    asset_tp_plan_df: pd.DataFrame | None,
    positions_qty: dict[str, float],
) -> set[str]:
    pos_tickers = {str(t).upper().strip() for t in positions_qty.keys()}

    if asset_tp_plan_df is None or asset_tp_plan_df.empty or "ticker" not in asset_tp_plan_df.columns:
        # if no TP plan, everything is core (frozen)
        return set(pos_tickers)

    df = asset_tp_plan_df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # exp_gross_reduce == 0 => not traded by TP
    if "exp_gross_reduce" in df.columns:
        reduce0 = df["exp_gross_reduce"].fillna(0.0).astype(float).abs() < 1e-9
        core = set(df.loc[reduce0, "ticker"].tolist())
    else:
        # if plan lacks column, be conservative: core=positions tickers not explicitly eligible/traded
        core = set(df["ticker"].tolist())

    # tickers in positions but not in plan => treat as core (not traded)
    in_plan = set(df["ticker"].tolist())
    core |= (pos_tickers - in_plan)
    return core


def reinvest_leftover_with_frozen_core(
    *,
    as_of: str,
    returns_wide: pd.DataFrame,            # daily returns matrix (tickers columns)
    exec_prices_usd: pd.Series,            # execution prices, indexed by ticker
    equity: float,
    gross_target: float,                   # equity * lev_target
    positions: dict[str, Position],        # for reference only (optional)
    positions_qty_after_tp: dict[str, float],
    asset_tp_plan_df: pd.DataFrame | None,
    score_cfg: ScoreConfig,
    goals: list[float],
    main_goal: float,
    # knobs
    max_assets_total: int = 10,
    min_assets_sleeve: int = 2,
    pop_size: int = 60,
    generations: int = 25,
    elite_frac: float = 0.15,
    n_paths_init: int = 4000,
    n_paths_final: int = 20000,
    block_size: int | tuple[int, int] | None = (8, 12),
    min_trade_usd: float = 25.0,
    seed: int = 123,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Reinvest leftover notional into a sleeve while freezing core quantities.
    Returns updated qty map + meta.
    """
    rng = np.random.default_rng(int(seed))

    # --- prices alignment ---
    px = pd.to_numeric(exec_prices_usd, errors="coerce").replace([np.inf, -np.inf], np.nan)
    px.index = px.index.astype(str).str.upper().str.strip()

    qty0 = {str(t).upper().strip(): float(q) for t, q in positions_qty_after_tp.items()}
    core = _infer_core_from_asset_tp_plan(asset_tp_plan_df=asset_tp_plan_df, positions_qty=qty0)

    gross_after_tp = _gross_notional_from_qty(qty=qty0, px=px)
    leftover = float(max(0.0, float(gross_target) - float(gross_after_tp)))

    meta: dict[str, Any] = {
        "as_of": as_of,
        "gross_target": float(gross_target),
        "gross_after_tp": float(gross_after_tp),
        "leftover": float(leftover),
        "core_size": int(len(core)),
    }

    if not np.isfinite(leftover) or leftover < float(min_trade_usd):
        meta["status"] = "skip_leftover_too_small"
        return dict(qty0), meta

    # --- returns matrix sanity ---
    R = returns_wide.copy()
    R.columns = [str(c).upper().strip() for c in R.columns]
    R = R.dropna(how="all")
    if R.shape[0] < 252:
        meta["status"] = "skip_not_enough_returns_history"
        return dict(qty0), meta

    # --- build sleeve universe (include==1, in returns columns, and not in core) ---
    universe_all = _load_universe_include1(returns_cols=set(R.columns))
    sleeve_universe = {t: a for t, a in universe_all.items() if t not in core}

    # how many sleeve names allowed (respect total max_assets)
    core_active = []
    for t in core:
        p = float(px.get(t, np.nan))
        q = float(qty0.get(t, 0.0))
        if np.isfinite(p) and p > 0 and np.isfinite(q) and abs(q * p) >= float(min_trade_usd):
            core_active.append(t)
    max_assets_sleeve = int(max(0, max_assets_total - len(core_active)))

    meta["core_active"] = core_active
    meta["max_assets_sleeve"] = max_assets_sleeve
    meta["sleeve_universe_size"] = int(len(sleeve_universe))

    if max_assets_sleeve < max(1, int(min_assets_sleeve)):
        meta["status"] = "skip_no_room_for_sleeve"
        return dict(qty0), meta

    if len(sleeve_universe) < max(5, int(min_assets_sleeve)):
        meta["status"] = "skip_sleeve_universe_too_small"
        return dict(qty0), meta

    # --- spectral rows cache for speed (optional) ---
    # we compute on sleeve returns only, then slice per candidate tickers
    returns_sleeve_clean = R[list(sleeve_universe.keys())].dropna(how="all")
    spec_df_full = _spectral_profiles_df(
        returns_sleeve_clean.fillna(0.0),
        bands_days=score_cfg.fft_bands_days,
    )

    def _build_full_weights_from_sleeve(weights_sleeve: dict[str, float]) -> dict[str, float]:
        """
        Gross-signed weights relative to gross_target:
        w[t] = signed_exposure_usd / gross_target

        Core: fixed via qty0 * px (can be long/short)
        Sleeve: long-only weights scaled by leftover/gross_target
        """
        sleeve_scale = float(leftover) / float(gross_target)

        weights_full: dict[str, float] = {}

        # core weights from qty*px
        for t in core:
            p = float(px.get(t, np.nan))
            q = float(qty0.get(t, 0.0))
            if not (np.isfinite(p) and p > 0 and np.isfinite(q)) or abs(q * p) < float(min_trade_usd):
                continue
            weights_full[t] = float((q * p) / float(gross_target))

        # sleeve weights (positive, long-only)
        for t, w in (weights_sleeve or {}).items():
            tt = str(t).upper().strip()
            if tt not in sleeve_universe:
                continue
            if not np.isfinite(w) or float(w) <= 0:
                continue
            weights_full[tt] = float(weights_full.get(tt, 0.0) + float(w) * sleeve_scale)

        return weights_full


    # --- helper: evaluate a sleeve candidate by merging with fixed core ---
    def eval_sleeve(weights_sleeve: dict[str, float], *, n_paths: int) -> Any | None:
        weights_full = _build_full_weights_from_sleeve(weights_sleeve)

        tickers_full = [t for t in weights_full.keys() if t in R.columns]
        if len(tickers_full) < 2:
            return None

        X = R[tickers_full].to_numpy(dtype=np.float32, copy=False)

        # spec rows
        spec_rows = None
        try:
            spec_rows = spec_df_full.loc[tickers_full, ["hf", "mf", "lf", "entropy"]].to_numpy(
                dtype=np.float32, copy=False
            )
        except Exception:
            spec_rows = None

        try:
            m = evaluate_portfolio_from_arrays(
                rets_assets=X,
                tickers=tickers_full,
                weights=weights_full,
                equity0=float(equity),
                notional=float(gross_target),
                goals=(float(goals[0]), float(goals[1]), float(goals[2])),
                main_goal=float(main_goal),
                score_config=score_cfg,
                mc_seed=int(rng.integers(0, 2**31 - 1)),
                spec_rows=spec_rows,
                n_paths=int(n_paths),
                days=252,
                block_size=block_size,
                weight_mode="gross_signed",
            )
            return m
        except Exception:
            return None


    # baseline: evaluate current (core only + current sleeve exposures) for “must improve”
    # We treat current sleeve as "no reinvest": i.e., invest leftover in cash -> not modeled.
    # So baseline = current portfolio AFTER TP, scaled to gross_target? No: baseline should be actual after-TP,
    # but you compare to candidate AFTER reinvest. We'll compare vs after-TP portfolio score.
    # Build weights_full_after_tp (only positions_qty_after_tp / gross_after_tp) evaluated at notional=gross_after_tp.
    # However, your evaluator expects notional consistent with weights normalization. We'll do a fair compare:
    # evaluate "do nothing" at notional=gross_target by keeping weights from current exposures / gross_target and leaving leftover as cash (ignored).
    baseline_sleeve = {}  # no reinvest
    baseline_m = eval_sleeve(baseline_sleeve, n_paths=max(4000, n_paths_init))
    baseline_score = float(baseline_m.score) if baseline_m is not None else -1e18
    meta["baseline_score"] = baseline_score

    # --- GA over sleeve weights (long-only) ---
    n_elite = max(1, int(pop_size * elite_frac))
    min_assets = int(min_assets_sleeve)
    max_assets = int(max_assets_sleeve)

    # init population
    pop_weights: list[dict[str, float]] = []
    pop_metrics: list[Any] = []

    while len(pop_metrics) < pop_size:
        w = sample_random_weights(
            sleeve_universe,
            max_assets=max_assets,
            min_assets=min_assets,
            rng=rng,
            weight_mode="long_only",
        )
        m = eval_sleeve(w, n_paths=int(n_paths_init))
        if m is None:
            continue
        pop_weights.append(w)
        pop_metrics.append(m)

    # evolve
    for gen in range(generations):
        # sort by score
        idx = np.argsort([-float(m.score) for m in pop_metrics]).tolist()
        pop_weights = [pop_weights[i] for i in idx]
        pop_metrics = [pop_metrics[i] for i in idx]

        elites_w = pop_weights[:n_elite]
        elites_m = pop_metrics[:n_elite]

        # children
        children_w: list[dict[str, float]] = []
        while len(children_w) < (pop_size - n_elite):
            pa, pb = rng.choice(elites_w, size=2, replace=True)
            cw = crossover_weights(pa, pb, max_assets=max_assets, rng=rng, weight_mode="long_only")
            cw = mutate_weights(
                cw,
                universe=sleeve_universe,
                max_assets=max_assets,
                min_assets=min_assets,
                rng=rng,
                sigma=0.12,
                replace_prob=0.20,
                weight_mode="long_only",
            )
            children_w.append(cw)

        # eval children with increasing n_paths
        x = gen / max(1, generations - 1)
        n_paths = int(n_paths_init + (n_paths_final - n_paths_init) * (x * x))

        children_m: list[Any] = []
        for w in children_w:
            m = eval_sleeve(w, n_paths=n_paths)
            if m is not None:
                children_m.append(m)
            if len(children_m) >= len(children_w):
                break

        # build next pop
        pop_weights = elites_w + children_w[: max(0, pop_size - n_elite)]
        pop_metrics = elites_m + children_m[: max(0, pop_size - n_elite)]

        best = pop_metrics[0]
        meta[f"gen_{gen+1}"] = {"best_score": float(best.score), "best_sharpe": float(best.sharpe)}

    # final best
    idx = np.argsort([-float(m.score) for m in pop_metrics]).tolist()
    best_w = pop_weights[idx[0]]
    best_m = pop_metrics[idx[0]]

    meta["best_weights_sleeve"] = dict(best_w)
    meta["weights_target"] = dict(_build_full_weights_from_sleeve(best_w))  # optional but useful for logs
    meta["core"] = sorted(list(core))
    meta["core_active"] = list(core_active)


    best_score = float(best_m.score)
    meta["best_score"] = best_score
    meta["improvement"] = float(best_score - baseline_score)
    meta["best_metrics"] = asdict(best_m)

    # enforce improvement
    if not np.isfinite(best_score) or best_score <= baseline_score + 1e-6:
        meta["status"] = "no_improvement"
        return dict(qty0), meta

    # --- convert sleeve allocation (leftover dollars) to target quantities ---
    qty_out = dict(qty0)

    for t, w in best_w.items():
        if float(w) <= 0:
            continue
        p = float(px.get(t, np.nan))
        if not (np.isfinite(p) and p > 0):
            continue

        dollars = float(w) * float(leftover)
        if dollars < float(min_trade_usd):
            continue

        dq = float(dollars / p)
        qty_out[t] = float(qty_out.get(t, 0.0) + dq)

    meta["status"] = "applied"
    return qty_out, meta
