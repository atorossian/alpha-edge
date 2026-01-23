# run_portfolio_search.py  (S3-only I/O)

from __future__ import annotations

from dataclasses import asdict
import datetime as dt

import numpy as np
import pandas as pd

from alpha_edge.universe.universe import load_universe, Asset
from alpha_edge.core.schemas import ScoreConfig, StabilityEnergyConfig, StabilityReport
from alpha_edge.portfolio.optimizer_engine import compute_stability_for_candidate
from alpha_edge.market.build_returns_wide_cache import build_returns_wide_cache, CacheConfig

from alpha_edge.market.regime_leverage import leverage_from_hmm
from alpha_edge.market.regime_filter import RegimeFilterState
from alpha_edge.portfolio.portfolio_search import evolve_portfolios_ga, refine_portfolio_annealing
from alpha_edge.portfolio.execution_engine import weights_to_discrete_shares
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.data_loader import (
    s3_init,
    s3_load_latest_json,
    s3_load_latest_report_score,
    s3_write_json_event,
    s3_write_parquet_partition,
    parse_positions_obj,
    clean_returns_matrix,
)


ENGINE_BUCKET = "alpha-edge-algo"
ENGINE_REGION = "eu-west-1"
ENGINE_ROOT_PREFIX = "engine/v1"


def print_portfolio_metrics(m, goals=None):
    if goals is None:
        goals = getattr(m, "goals", (800.0, 1200.0, 2000.0))
    g1, g2, g3 = [float(g) for g in goals]
    print("\n=== Best Portfolio ===")
    print(f"Score:               {m.score:.4f}")
    print(f"P(ruin, 1y):         {m.ruin_prob_1y:.2%}")
    print(f"Ann return:          {m.ann_return:.2%}")
    print(f"Ann vol (sample):    {m.ann_vol:.2%}")
    print(f"Ann vol (LW):        {m.ann_vol_lw:.2%}")
    print(f"Sharpe Ratio:        {m.sharpe:.2f}")
    print(f"Sortino Ratio:       {m.sortino:.2f}")
    print(f"Max drawdown:        {m.max_drawdown:.2%}")
    print(f"VaR 95%:             {m.var_95:.2%}")
    print(f"CVaR 95%:            {m.cvar_95:.2%}")
    print(f"P(hit {g1:.0f}, 1y):     {m.p_hit_goal_1_1y:.2%}")
    print(f"P(hit {g2:.0f}, 1y):     {m.p_hit_goal_2_1y:.2%}")
    print(f"P(hit {g3:.0f}, 1y):     {m.p_hit_goal_3_1y:.2%}")
    print(f"Median t({g1:.0f} days): {m.med_t_goal_1_days}")
    print(f"Median t({g2:.0f} days): {m.med_t_goal_2_days}")
    print(f"Median t({g3:.0f} days): {m.med_t_goal_3_days}")
    print(f"Ending eq p5:        {m.ending_equity_p5:.2f}")
    print(f"Ending eq p25:       {m.ending_equity_p25:.2f}")
    print(f"Ending eq p50:       {m.ending_equity_p50:.2f}")
    print(f"Ending eq p75:       {m.ending_equity_p75:.2f}")
    print(f"Ending eq p95:       {m.ending_equity_p95:.2f}")

    print("\nWeights:")
    for t, w in sorted(m.weights.items(), key=lambda x: -x[1]):
        print(f"  {t:8s}  {w:.2%}")


def format_stability_report(
    rep: StabilityReport,
    *,
    days: int = 252,
    cfg: StabilityEnergyConfig | None = None,
) -> str:
    if cfg is None:
        cfg = StabilityEnergyConfig()

    # Convert normalized back to readable units
    mdd = rep.mdd_mean * 100.0
    cdar = rep.cdar_alpha * 100.0
    ttr_days = rep.ttr_mean_norm * float(days)
    uw = rep.underwater_mean * 100.0
    breach = rep.p_breach * 100.0

    return (
        "Stability (lower is better)\n"
        f"  Energy:            {rep.energy:.4f}\n"
        f"  Avg MDD:           {mdd:.1f}%\n"
        f"  CDaR@{int(cfg.alpha_cdar*100)}:        {cdar:.1f}%\n"
        f"  Avg TTR:           {ttr_days:.0f} days\n"
        f"  P(MDD ≥ {cfg.breach_dd:.0%}):   {breach:.1f}%\n"
        f"  Underwater time:   {uw:.1f}% of days\n"
    )


def evalmetrics_to_row(m) -> dict:
    goals = getattr(m, "goals", (800.0, 1200.0, 2000.0))
    g1, g2, g3 = [float(g) for g in goals]
    return dict(
        score=float(m.score),
        ann_return=float(m.ann_return),
        ann_vol=float(m.ann_vol),
        ann_vol_lw=float(m.ann_vol_lw),
        sharpe=float(m.sharpe),
        sortino=float(m.sortino),
        max_drawdown=float(m.max_drawdown),
        var_95=float(m.var_95),
        cvar_95=float(m.cvar_95),
        ruin_prob_1y=float(m.ruin_prob_1y),
        p_hit_goal_1_1y=float(m.p_hit_goal_1_1y),
        p_hit_goal_2_1y=float(m.p_hit_goal_2_1y),
        p_hit_goal_3_1y=float(m.p_hit_goal_3_1y),
        med_t_goal_1_days=m.med_t_goal_1_days,
        med_t_goal_2_days=m.med_t_goal_2_days,
        med_t_goal_3_days=m.med_t_goal_3_days,
        end_p5=float(m.ending_equity_p5),
        end_p25=float(m.ending_equity_p25),
        end_p50=float(m.ending_equity_p50),
        end_p75=float(m.ending_equity_p75),
        end_p95=float(m.ending_equity_p95),
        goal_1=g1,
        goal_2=g2,
        goal_3=g3,
    )


def weights_to_rows(weights: dict[str, float], *, tag: str) -> list[dict]:
    return [{"tag": tag, "ticker": t, "weight": float(w)} for t, w in weights.items()]


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float(default)


def main():
    run_dt = pd.Timestamp(dt.date.today()).normalize()
    as_of_run_date = run_dt.strftime("%Y-%m-%d")
    # ---- S3 clients / stores ----
    s3 = s3_init(ENGINE_REGION)
    market = MarketStore(bucket=ENGINE_BUCKET)

    # keep hardcoded
    equity0 = 1295.12
    GOALS = (1500.0, 2000.0, 3000.0)
    MAIN_GOAL = 2000.0

    # ---------- Load latest inputs (S3-only) ----------
    raw_positions = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="inputs/positions"
    )
    if not raw_positions:
        raise RuntimeError("Missing S3 latest positions. Expected engine/v1/inputs/positions/latest.json")
    positions = parse_positions_obj(raw_positions)  # dict[str, Position]

    raw_score_cfg = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="configs/score_config"
    )
    if not raw_score_cfg:
        raise RuntimeError("Missing S3 latest score_config. Expected engine/v1/configs/score_config/latest.json")
    score_cfg = ScoreConfig(**raw_score_cfg)

    # Universe (load as dataframe so we can key by asset_id)
    u = pd.read_csv("data/universe/universe.csv")
    u = u[u.get("include", 1).fillna(1).astype(int) == 1].copy()

    u["asset_id"] = u["asset_id"].astype(str).str.strip()
    u["ticker"] = u.get("ticker", u["asset_id"]).astype(str).str.upper().str.strip()

    asset_to_ticker = dict(zip(u["asset_id"], u["ticker"]))
    ticker_to_asset = {v: k for k, v in asset_to_ticker.items()}

    # ---------- Load snapshots from S3 (market) ----------
    latest_prices_df = market.read_latest_prices_snapshot()
    if latest_prices_df.empty:
        raise RuntimeError("Missing latest_prices snapshot in S3. Run ingest_market_data.py first.")

    latest_prices_df["asset_id"] = latest_prices_df["asset_id"].astype(str).str.strip()
    latest_prices_df["adj_close_usd"] = pd.to_numeric(latest_prices_df["adj_close_usd"], errors="coerce")

    # prices keyed by asset_id (NOT ticker)
    px_map = (
        latest_prices_df.set_index("asset_id")["adj_close_usd"]
        .dropna()
        .to_dict()
    )

    # ---------- Load MARKET regime snapshot and set leverage ----------
    # Primary path: read engine/v1/regimes/market_hmm/latest.json and use leverage_recommendation.leverage
    # Fallback: compute from hmm payload if leverage_recommendation missing (backward compat)
    hmm_payload_wrapped = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="regimes/market_hmm"
    ) or {}

    # --- MARKET date (data as-of) ---
    as_of_market_date = hmm_payload_wrapped.get("as_of") if isinstance(hmm_payload_wrapped, dict) else None
    as_of_market_date = str(as_of_market_date) if as_of_market_date else as_of_run_date

    # Always define lev_rec to avoid UnboundLocalError later
    lev_rec: dict = {}

    lr = (hmm_payload_wrapped.get("leverage_recommendation") or {}) if isinstance(hmm_payload_wrapped, dict) else {}

    if isinstance(lr, dict) and lr.get("leverage") is not None:
        # Fast path: use already computed leverage recommendation from compute_market_regime.py
        lev_rec = lr
        target_leverage = float(lr["leverage"])

        # Make prints robust even if fields missing
        lev_rec.setdefault("mode", "stored")
        lev_rec.setdefault("confidence", np.nan)
        lev_rec.setdefault("chosen_label", (lr.get("chosen_label") or lr.get("label") or lr.get("label_commit")))

    else:
        # Fallback: compute from hmm payload (keeps backward compat)
        hmm_res = hmm_payload_wrapped.get("hmm") if isinstance(hmm_payload_wrapped, dict) else None
        if hmm_res is None:
            hmm_res = hmm_payload_wrapped if isinstance(hmm_payload_wrapped, dict) else {}

        st_raw = market.read_regime_filter_state() or {}
        st = RegimeFilterState(
            last_date=st_raw.get("last_date"),
            chosen_label=st_raw.get("chosen_label"),
            days_in_regime=int(st_raw.get("days_in_regime", 0) or 0),
            probs_smoothed=st_raw.get("probs_smoothed"),
        )

        lev_rec = leverage_from_hmm(
            hmm_res or {},
            default=1.0,
            risk_appetite=0.6,
            low_confidence_floor=0.2,
            hard_cap=12.0,
            filter_state=st,
            as_of=as_of_market_date,
            filter_alpha=0.20,
            min_hold_days=3,
            min_prob_to_switch=0.60,
            min_margin_to_switch=0.12,
        )

        if isinstance(lev_rec.get("filter_state"), dict):
            market.write_regime_filter_state(lev_rec["filter_state"])

        target_leverage = float(lev_rec.get("leverage", 1.0))

    # ---------- Target notional from hardcoded equity + leverage ----------
    notional = float(equity0) * float(target_leverage)
    if not np.isfinite(notional) or notional <= 0:
        raise RuntimeError(f"Invalid target notional={notional} from equity0={equity0} and lev={target_leverage}")

    # Keep current portfolio notional too (for reference only)
    current_gross_notional = 0.0
    missing_px = []

    for p in positions.values():
        t = str(p.ticker).upper().strip()
        aid = ticker_to_asset.get(t)

        if not aid or aid not in px_map:
            missing_px.append(t)
            continue

        px = float(px_map[aid])
        if not np.isfinite(px) or px <= 0:
            missing_px.append(t)
            continue

        exp = float(p.quantity) * px
        current_gross_notional += abs(exp)

    # Safe print formatting (confidence may be missing/NaN)
    conf = _safe_float(lev_rec.get("confidence"), default=np.nan)
    conf_s = "n/a" if not np.isfinite(conf) else f"{conf:.2f}"

    print(
        f"[dates] as_of_market_date={as_of_market_date} | as_of_run_date={as_of_run_date}\n"
        f"[capital] equity0={equity0:.2f} USD | "
        f"regime={lev_rec.get('chosen_label')} ({lev_rec.get('mode')}, conf={conf_s}) | "
        f"target_leverage={target_leverage:.2f}x -> target_notional={notional:.2f} USD"
    )

    if current_gross_notional > 0:
        print(f"[capital] current_positions_notional≈{current_gross_notional:.2f} USD (reference only)")
    if missing_px:
        print(f"[capital][warn] missing prices for {len(missing_px)} tickers (sample: {missing_px[:10]})")

    # ---------- Load returns wide cache ----------
    cache_cfg = CacheConfig(bucket=ENGINE_BUCKET, min_years=5.0)
    build_returns_wide_cache(cache_cfg)
    returns_path = f"s3://{ENGINE_BUCKET}/{cache_cfg.cache_prefix}/returns_wide_min{int(cache_cfg.min_years)}y.parquet"
    returns_wide = pd.read_parquet(returns_path, engine="pyarrow").sort_index()
    returns_wide, diag = clean_returns_matrix(returns_wide)

    # map asset_id -> ticker (if cache columns are asset_ids)
    returns_wide = returns_wide.rename(columns=lambda c: asset_to_ticker.get(c, c))

    # ---- Align universe to RETURNS (ticker-based for GA) ----
    returns_wide = returns_wide.rename(columns=lambda c: asset_to_ticker.get(c, c))

    # Build universe keyed by TICKER with Asset objects
    universe = {}
    for _, row in u.iterrows():
        t = row["ticker"]
        if t in returns_wide.columns:
            universe[t] = Asset(
                ticker=row["ticker"],
                yahoo_ticker=row["yahoo_ticker"],
                name=row["name"],
                asset_class=row["asset_class"],
                role=row["role"],
                region=row["region"],
                max_weight=float(row.get("max_weight", 1.0)),
                min_weight=float(row.get("min_weight", 0.0)),
                include=True,
            )

    if len(universe) < 10:
        raise RuntimeError(f"Universe after returns cleaning too small: {len(universe)}. diag={diag}")

    # ---------- Search ----------
    ga_params = dict(
        pop_size=80,
        generations=50,
        elite_frac=0.1,
        max_assets=10,
        min_assets=5,
        n_paths_init=5000,
        n_paths_final=20000,
        path_source="bootstrap",
        pca_k=3,
        block_size=(8, 12),
    )

    print("\n=== Running Genetic Algorithm Portfolio Search ===")
    ga_pop, ga_archive = evolve_portfolios_ga(
        returns=returns_wide,
        universe=universe,
        lw_cov=None,
        equity0=equity0,
        notional=notional,
        goals=GOALS,
        main_goal=MAIN_GOAL,
        score_config=score_cfg,
        return_archive=True,
        weight_mode="long_short",
        **ga_params,
    )

    best_ga = ga_pop[0]

    # ---------- Stability rerank on GA archive ----------
    topK = ga_archive[:200]  # tune K
    st_cfg = StabilityEnergyConfig(
        alpha_cdar=0.95,
        breach_dd=0.25,
        lambda_mdd=1.0,
        lambda_cdar=1.2,
        lambda_ttr=0.7,
        lambda_breach=1.5,
        lambda_underwater=0.5,
    )

    rng = np.random.default_rng(123)
    st_ranked = []

    for m in topK:
        rep = compute_stability_for_candidate(
            returns=returns_wide,
            weights=m.weights,
            equity0=equity0,
            notional=notional,
            goals=GOALS,
            days=252,
            n_paths=20000,
            mc_seed=int(rng.integers(0, 2**31 - 1)),
            path_source="bootstrap",   # OR "pca" if you want consistency with anneal later
            pca_k=5,
            block_size=(8, 12),
            stability_cfg=st_cfg,
        )
        st_ranked.append((m, rep))

    st_ranked.sort(key=lambda x: x[1].energy)
    best_by_stability, best_st_rep = st_ranked[0]

    print("\n=== Stability rerank (top 200 archive) ===")
    print(format_stability_report(best_st_rep, days=252, cfg=st_cfg))

    anneal_params = dict(
        max_assets=10,
        min_assets=5,
        n_steps=200,
        temp_start=1.0,
        temp_end=0.05,
        n_paths_init=3000,
        n_paths_final=20000,
        path_source="pca",
        pca_k=5,
        block_size=(8, 12),
    )

    best_refined = refine_portfolio_annealing(
        base_metrics=best_by_stability,
        returns=returns_wide,
        universe=universe,
        lw_cov=None,
        equity0=equity0,
        notional=notional,
        goals=GOALS,
        main_goal=MAIN_GOAL,
        score_config=score_cfg,
        weight_mode="long_short",
        **anneal_params,
    )

    # ---------- Discretize into shares (post-processing only) ----------
    weights_ticker = {asset_to_ticker.get(aid, aid): w for aid, w in best_refined.weights.items()}
    prices_ticker = {asset_to_ticker.get(aid, aid): px for aid, px in px_map.items()}

    alloc = weights_to_discrete_shares(
        weights=weights_ticker,
        prices=prices_ticker,
        notional=float(notional),
        min_units=1.0,
        min_weight=0.01,
        crypto_decimals=8,
    )

    # ---------- Persist to S3 (engine artifacts; append-only) ----------
    run_id = f"{run_dt.strftime('%Y%m%d')}-{pd.Timestamp.utcnow().strftime('%H%M%S')}"
    last_score = s3_load_latest_report_score(s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX)

    s3_write_json_event(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
        table="portfolio_search/runs",
        dt=run_dt,
        filename=f"run_{run_id}.json",
        payload={
            "run_id": run_id,
            "as_of": as_of_market_date,
            "meta": {
                "as_of_market_date": as_of_market_date,
                "as_of_run_date": as_of_run_date,
                "hmm_snapshot_as_of": as_of_market_date,
            },
            "inputs": {
                "equity0": float(equity0),
                "target_leverage": float(target_leverage),
                "target_notional": float(notional),
                "current_positions_notional": float(current_gross_notional),
                "current_leverage_real": float(current_gross_notional / float(equity0)) if float(equity0) > 0 else float("inf"),
                "goals": list(GOALS),
                "main_goal": MAIN_GOAL,
                "last_daily_report_score": last_score,
                "positions": {t: asdict(p) for t, p in positions.items()},
                "score_config": asdict(score_cfg),
                "universe_size": len(universe),
                "returns_clean_diag": diag,
                "regime": lev_rec,  # ALWAYS defined now
                "market_data": {
                    "bucket": ENGINE_BUCKET,
                    "returns_cache": "market/cache/v1/returns_wide_min5y.parquet",
                    "latest_prices_snapshot": "market/snapshots/v1/latest_prices.parquet",
                },
            },
            "params": {"ga": ga_params, "anneal": anneal_params},
            "outputs": {
                "best_ga": asdict(best_ga),
                "best_refined": asdict(best_refined),
                "discrete_allocation": {
                    "cash_left": float(alloc.cash_left),
                    "shares": {k: float(v) for k, v in alloc.shares.items()},
                    "realized_weights": {k: float(v) for k, v in alloc.realized_weights.items()},
                },
            },
        },
    )

    # candidates leaderboard (parquet)
    top_n = min(50, len(ga_pop))
    df_top = pd.DataFrame([evalmetrics_to_row(m) for m in ga_pop[:top_n]])
    df_top.insert(0, "run_id", run_id)

    s3_write_parquet_partition(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
        table="portfolio_search/candidates",
        dt=run_dt,
        filename=f"top_{top_n}_{run_id}.parquet",
        df=df_top,
    )

    # weights table (parquet)
    rows = []
    rows += weights_to_rows(best_refined.weights, tag="best_refined_weights")
    rows += weights_to_rows(alloc.realized_weights, tag="realized_weights")
    df_w = pd.DataFrame(rows)
    df_w.insert(0, "run_id", run_id)

    s3_write_parquet_partition(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
        table="portfolio_search/weights",
        dt=run_dt,
        filename=f"weights_{run_id}.parquet",
        df=df_w,
    )

    # ---------- Print ----------
    print("\n=== GA best ===")
    print(best_ga.score, best_ga.p_hit_goal_3_1y, best_ga.ruin_prob_1y, best_ga.weights)

    print("\n=== After annealing ===")
    print(best_refined.score, best_refined.p_hit_goal_3_1y, best_refined.ruin_prob_1y, best_refined.weights)
    print_portfolio_metrics(best_refined, goals=GOALS)

    print(f"\nTarget notional (USD): {notional:.2f}")

    print("\nCash left:", alloc.cash_left)
    print("\nShares:")
    for t, w in sorted(alloc.shares.items(), key=lambda x: -x[1]):
        print(f"  {t:8s}  {w:.2f}")

    print("\nRealized weights:")
    for t, w in sorted(alloc.realized_weights.items(), key=lambda x: -x[1]):
        print(f"  {t:8s}  {w:.2%}")

    print(f"\n[S3] Saved portfolio search run_id={run_id}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
