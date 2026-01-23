# run_score_weights_tuning.py  (S3-only I/O; universe filter matches portfolio search)
from __future__ import annotations

import datetime as dt
from dataclasses import asdict

import numpy as np
import pandas as pd

from alpha_edge.universe.universe import load_universe
from alpha_edge.portfolio.portfolio_search import evolve_portfolios_ga
from alpha_edge.tuning.tune_score_weights_optimize import tune_lambdas_by_optimization
from alpha_edge.market.regime_leverage import leverage_from_hmm

from alpha_edge.core.data_loader import (
    s3_init,
    s3_load_latest_json,
    s3_write_json_event,
    parse_positions_obj,
    clean_returns_matrix,   # MUST match portfolio search config
)
from alpha_edge import paths


ENGINE_BUCKET = "alpha-edge-algo"
ENGINE_REGION = "eu-west-1"
ENGINE_ROOT_PREFIX = "engine/v1"

RETURNS_WIDE_CACHE_PATH = "s3://alpha-edge-algo/market/cache/v1/returns_wide_min5y.parquet"


def main():
    today = pd.Timestamp(dt.date.today())

    # ---- S3 clients ----
    s3 = s3_init(ENGINE_REGION)

    # Universe (file-based for now)
    universe_all = load_universe(paths.universe_dir() / "universe.csv")

    # ---------- Load latest positions (S3-only) ----------
    raw_positions = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="inputs/positions"
    )
    if not raw_positions:
        raise RuntimeError("Missing S3 latest positions. Expected engine/v1/inputs/positions/latest.json")
    positions = parse_positions_obj(raw_positions)

    # ---------- Regime -> leverage -> notional ----------
    equity0 = 934.13  # keep hardcoded as you want

    hmm_payload_wrapped = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="regimes/hmm"
    )
    hmm_res = None
    if isinstance(hmm_payload_wrapped, dict):
        hmm_res = hmm_payload_wrapped.get("hmm") or hmm_payload_wrapped

    lev_rec = leverage_from_hmm(
        hmm_res or {},
        default=1.0,
        risk_appetite=0.6,
        low_confidence_floor=0.2,
        hard_cap=12.0,
    )
    target_leverage = 7.0# float(lev_rec.get("leverage", 1.0))
    notional = float(equity0) * float(target_leverage)

    if not np.isfinite(notional) or notional <= 0:
        raise RuntimeError(f"Invalid notional={notional} from equity0={equity0} lev={target_leverage}")

    print(
        f"[tuning] equity0={equity0:.2f} USD | "
        f"regime={lev_rec.get('chosen_label')} ({lev_rec.get('mode')}, conf={lev_rec.get('confidence'):.2f}) | "
        f"lev={target_leverage:.2f}x -> notional={notional:.2f}"
    )

    # ---------- Load returns wide cache (S3) ----------
    returns_wide = pd.read_parquet(RETURNS_WIDE_CACHE_PATH, engine="pyarrow").sort_index()

    # IMPORTANT: match portfolio search criteria EXACTLY
    returns_wide, diag = clean_returns_matrix(
        returns_wide,
        min_history_days=252 * 2,
        max_nan_frac=0.2,
        min_vol=1e-6,
    )

    # Restrict universe exactly like portfolio search
    universe = {t: a for t, a in universe_all.items() if t in returns_wide.columns}
    if len(universe) < 10:
        raise RuntimeError(f"Universe too small after returns cleaning: {len(universe)}. diag={diag}")

    # ---------- GA archive for candidate pool ----------
    # NOTE: this relies on the GA version that returns (population, archive) when return_archive=True
    ga_pop, ga_archive = evolve_portfolios_ga(
        returns=returns_wide,
        universe=universe,
        lw_cov=None,
        equity0=equity0,
        notional=notional,
        pop_size=80,
        generations=50,
        goals=(800.0, 1200.0, 2000.0),
        main_goal=2000.0,
        score_config=None,  # GA will create default ScoreConfig internally
        elite_frac=0.2,
        max_assets=10,
        min_assets=5,
        n_paths_init=5000,
        n_paths_final=20000,
        path_source="bootstrap",
        pca_k=3,
        block_size=(8, 12),
        return_archive=True,
        archive_limit=50000,
        # IMPORTANT: do NOT pass any discrete-repair knobs in rollback mode
        # hard_filter_ruin=False is the default in your current GA code
    )

    candidate_weights = [m.weights for m in ga_archive[:2000]]
    if not candidate_weights:
        raise RuntimeError("GA archive is empty; cannot tune lambdas.")

    # ---------- Tune lambdas ----------
    best_cfg, info = tune_lambdas_by_optimization(
        returns=returns_wide,
        lw_cov=None,
        candidate_pool=candidate_weights,
        equity0=equity0,
        notional=notional,
        goals=(800.0, 1200.0, 2000.0),
        main_goal=2000.0,
        n_trials=40,
        pool_sample_size=500,
        shortlist_size=60,
        n_paths_train=6000,
        n_paths_valid=20000,
        ruin_cap=0.25,
        alpha_ruin=0.5,
        train_frac=0.7,
    )

    print(best_cfg)
    print(info)

    # ---------- Persist tuned config to S3 (append-only + latest pointer) ----------
    s3_write_json_event(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
        table="configs/score_config",
        dt=today,
        filename="score_config.json",
        payload=asdict(best_cfg),
        update_latest=True,
    )

    run_id = f"{today.strftime('%Y%m%d')}-{pd.Timestamp.utcnow().strftime('%H%M%S')}"
    s3_write_json_event(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
        table="score_tuning/runs",
        dt=today,
        filename=f"tuning_{run_id}.json",
        payload={
            "run_id": run_id,
            "as_of": today.strftime("%Y-%m-%d"),
            "inputs": {
                "equity0": float(equity0),
                "target_leverage": float(target_leverage),
                "notional": float(notional),
                "positions_n": len(positions),
                "universe_n": len(universe),
                "returns_cache": "market/cache/v1/returns_wide_min5y.parquet",
                "returns_clean_diag": diag,
                "cleaning": {
                    "min_history_days": 252 * 2,
                    "max_nan_frac": 0.2,
                    "min_vol": 1e-6,
                },
                "regime": lev_rec,
            },
            "params": {
                "n_trials": 40,
                "pool_sample_size": 500,
                "shortlist_size": 60,
                "n_paths_train": 6000,
                "n_paths_valid": 20000,
                "ruin_cap": 0.25,
                "alpha_ruin": 0.5,
                "train_frac": 0.7,
            },
            "best_cfg": asdict(best_cfg),
            "info": info,
        },
    )

    print("\n[S3] Saved tuned score_config to engine/v1/configs/score_config/latest.json")


if __name__ == "__main__":
    main()
