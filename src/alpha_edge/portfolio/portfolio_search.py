# portfolio_search.py
from __future__ import annotations

from typing import Dict, List, Tuple
import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

import time
from alpha_edge.universe.universe import Asset
from alpha_edge.core.schemas import EvalMetrics, ScoreConfig
from alpha_edge.portfolio.optimizer_engine import _spectral_profiles_df, evaluate_portfolio_from_arrays


# ----------------------------
# Fingerprint / archive / diversity diagnostics
# ----------------------------

def _weights_fingerprint(weights: Dict[str, float], decimals: int = 6) -> Tuple[Tuple[str, float], ...]:
    """
    Stable fingerprint for deduplication.
    - sort tickers
    - round weights to avoid tiny float differences
    - keep signed weights so long/short portfolios do not collide with long-only
    """
    items = tuple(
        sorted(
            (str(t).upper().strip(), round(float(w), decimals))
            for t, w in weights.items()
            if np.isfinite(float(w)) and abs(float(w)) > 0.0
        )
    )
    return items


def _weight_l1_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Signed L1 distance over the union of assets. Higher means more structurally different."""
    keys = set(str(k).upper().strip() for k in (a or {}).keys())
    keys |= set(str(k).upper().strip() for k in (b or {}).keys())
    if not keys:
        return 0.0
    dist = 0.0
    for k in keys:
        dist += abs(float((a or {}).get(k, 0.0) or 0.0) - float((b or {}).get(k, 0.0) or 0.0))
    return float(dist)


def _asset_overlap_ratio(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Jaccard overlap of active asset sets. Higher means more similar."""
    sa = {str(k).upper().strip() for k, v in (a or {}).items() if abs(float(v)) > 1e-12}
    sb = {str(k).upper().strip() for k, v in (b or {}).items() if abs(float(v)) > 1e-12}
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return float(len(sa & sb) / len(union))


def _schedule_value(
    *,
    gen_idx: int,
    generations: int,
    start: float,
    end: float,
    power: float = 1.5,
) -> float:
    """Anneal a value from start to end across generations."""
    if generations <= 1:
        return float(end)
    x = float(gen_idx) / float(max(1, generations - 1))
    x = min(1.0, max(0.0, x))
    return float(end + (float(start) - float(end)) * (1.0 - x) ** float(power))


def _portfolio_diversity_summary(population: list[EvalMetrics]) -> dict:
    """Lightweight generation-level exploration diagnostics."""
    candidates = [m for m in (population or []) if getattr(m, "weights", None)]
    if not candidates:
        return {
            "population_size": 0,
            "unique_assets_used": 0,
            "avg_assets_per_candidate": 0.0,
            "avg_pairwise_asset_overlap": 0.0,
            "avg_pairwise_weight_l1_distance": 0.0,
            "top_asset_frequency": [],
            "avg_net_exposure": 0.0,
            "avg_short_gross": 0.0,
        }

    asset_counts: dict[str, int] = {}
    asset_counts_per_candidate: list[int] = []
    net_exposures: list[float] = []
    short_grosses: list[float] = []

    for m in candidates:
        weights = {str(k).upper().strip(): float(v) for k, v in m.weights.items() if abs(float(v)) > 1e-12}
        asset_counts_per_candidate.append(len(weights))
        net_exposures.append(float(sum(weights.values())))
        short_grosses.append(float(sum(-v for v in weights.values() if v < 0.0)))
        for t in weights:
            asset_counts[t] = asset_counts.get(t, 0) + 1

    overlaps: list[float] = []
    distances: list[float] = []
    max_pairs = 500
    pair_count = 0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            overlaps.append(_asset_overlap_ratio(candidates[i].weights, candidates[j].weights))
            distances.append(_weight_l1_distance(candidates[i].weights, candidates[j].weights))
            pair_count += 1
            if pair_count >= max_pairs:
                break
        if pair_count >= max_pairs:
            break

    n = float(len(candidates))
    top_assets = sorted(asset_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

    return {
        "population_size": int(len(candidates)),
        "unique_assets_used": int(len(asset_counts)),
        "avg_assets_per_candidate": float(np.mean(asset_counts_per_candidate)) if asset_counts_per_candidate else 0.0,
        "avg_pairwise_asset_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "avg_pairwise_weight_l1_distance": float(np.mean(distances)) if distances else 0.0,
        "top_asset_frequency": [
            {"ticker": str(t), "count": int(c), "frequency": float(c / n)} for t, c in top_assets
        ],
        "avg_net_exposure": float(np.mean(net_exposures)) if net_exposures else 0.0,
        "avg_short_gross": float(np.mean(short_grosses)) if short_grosses else 0.0,
    }


def _archive_add(
    archive: Dict[Tuple[Tuple[str, float], ...], EvalMetrics],
    m: EvalMetrics,
    *,
    decimals: int = 6,
    archive_limit: int | None = None,
    diversity_min_l1: float = 0.0,
    diversity_check_top_k: int = 250,
) -> None:
    """
    Add candidate to archive with optional diversity-aware retention.

    Exact duplicates are still deduplicated by fingerprint. When diversity_min_l1
    is enabled, a new candidate that is very close to an existing high-score
    archived candidate only replaces it if the new score is better.
    """
    fp = _weights_fingerprint(m.weights, decimals=decimals)
    cur = archive.get(fp)
    if cur is not None:
        if m.score > cur.score:
            archive[fp] = m
        return

    if diversity_min_l1 > 0.0 and archive:
        top_existing = sorted(archive.items(), key=lambda kv: kv[1].score, reverse=True)[: int(diversity_check_top_k)]
        nearest_fp = None
        nearest_m = None
        nearest_dist = float("inf")
        for ex_fp, ex_m in top_existing:
            d = _weight_l1_distance(m.weights, ex_m.weights)
            if d < nearest_dist:
                nearest_dist = d
                nearest_fp = ex_fp
                nearest_m = ex_m

        if nearest_m is not None and nearest_dist < float(diversity_min_l1):
            if m.score > nearest_m.score and nearest_fp is not None:
                del archive[nearest_fp]
                archive[fp] = m
            return

    archive[fp] = m

    if archive_limit is not None and len(archive) > archive_limit:
        if len(archive) > int(archive_limit * 1.10):
            keep = sorted(archive.values(), key=lambda x: x.score, reverse=True)[:archive_limit]
            archive.clear()
            for mm in keep:
                archive[_weights_fingerprint(mm.weights, decimals=decimals)] = mm


# ----------------------------
# Long/short normalization helpers (search-space constraints)
# ----------------------------

def normalize_gross_signed(
    w: dict[str, float],
    *,
    net_min: float = 0.20,
    net_max: float = 1.00,
    short_budget: float = 0.30,
    eps: float = 1e-12,
) -> dict[str, float] | None:
    """
    Normalize by gross exposure: sum(abs(w)) = 1

    Enforce:
      - net exposure: sum(w) in [net_min, net_max]
      - short budget: sum(-w for w<0) <= short_budget
    """
    w = {t: float(x) for t, x in w.items() if abs(float(x)) > eps}
    if not w:
        return None

    gross = float(sum(abs(x) for x in w.values()))
    if gross <= eps or not np.isfinite(gross):
        return None

    w = {t: x / gross for t, x in w.items()}  # gross=1

    net = float(sum(w.values()))
    short_gross = float(sum(-x for x in w.values() if x < 0.0))

    if (net < net_min) or (net > net_max):
        return None
    if short_gross > short_budget:
        return None

    return w


# Search-space defaults for long/short
LS_NET_MIN = 0.20
LS_NET_MAX = 1.00
LS_SHORT_BUDGET = 0.30
LS_MAX_TRIES = 200


# ----------------------------
# Weight sampling / crossover / mutation (mode-aware)
# ----------------------------

def sample_random_weights(
    universe: Dict[str, Asset],
    max_assets: int | None = None,
    min_assets: int = 5,
    rng: np.random.Generator | None = None,
    *,
    weight_mode: str = "long_only",
) -> Dict[str, float]:
    """
    Sample candidate weights in the requested search space.
      - long_only: positive weights sum to 1
      - long_short: signed weights with gross=1 + constraints (net band, short budget)
    """
    if rng is None:
        rng = np.random.default_rng()


    tickers = list(universe.keys())
    n_total = len(tickers)

    if max_assets is None or max_assets > n_total:
        max_assets = n_total

    if weight_mode == "long_only":
        k = int(rng.integers(low=min_assets, high=max_assets + 1))
        chosen = list(rng.choice(tickers, size=k, replace=False))

        alpha = np.ones(k)
        raw_w = rng.dirichlet(alpha).astype(np.float64)

        weights: Dict[str, float] = {}
        for i, t in enumerate(chosen):
            w = float(raw_w[i])
            max_w = float(universe[t].max_weight or 1.0)
            weights[t] = min(w, max_w)

        total = float(sum(weights.values()))
        if total <= 0:
            eq = 1.0 / float(k)
            return {t: eq for t in chosen}

        return {t: w / total for t, w in weights.items()}

    # --- long_short ---
    for _ in range(LS_MAX_TRIES):
        k = int(rng.integers(low=min_assets, high=max_assets + 1))
        chosen = list(rng.choice(tickers, size=k, replace=False))

        mag = rng.dirichlet(np.ones(k)).astype(np.float64)

        p_short = min(0.45, max(0.05, LS_SHORT_BUDGET))  # heuristic
        signs = np.where(rng.random(k) < p_short, -1.0, 1.0).astype(np.float64)

        w_raw = (mag * signs).astype(np.float64)

        w: Dict[str, float] = {}
        for i, t in enumerate(chosen):
            max_w = float(universe[t].max_weight or 1.0)
            w[t] = float(np.clip(w_raw[i], -max_w, max_w))

        w2 = normalize_gross_signed(
            w,
            net_min=LS_NET_MIN,
            net_max=LS_NET_MAX,
            short_budget=LS_SHORT_BUDGET,
        )
        if w2 is not None:
            return w2

    # fallback: long-only equal weight
    eq = 1.0 / float(min_assets)
    chosen = list(rng.choice(tickers, size=min_assets, replace=False))
    return {t: eq for t in chosen}


def crossover_weights(
    w_a: Dict[str, float],
    w_b: Dict[str, float],
    max_assets: int,
    rng: np.random.Generator,
    *,
    weight_mode: str = "long_only",
) -> Dict[str, float]:

    if weight_mode == "long_only":
        tickers = list(set(w_a.keys()) | set(w_b.keys()))
        child: Dict[str, float] = {}

        for t in tickers:
            wa = float(w_a.get(t, 0.0))
            wb = float(w_b.get(t, 0.0))
            base = 0.5 * (wa + wb)
            if base <= 0:
                continue
            noise = float(rng.normal(0.0, 0.05 * base))
            w = max(base + noise, 0.0)
            if w > 0:
                child[t] = float(w)

        if not child:
            child = dict(w_a)

        child = dict(sorted(child.items(), key=lambda kv: kv[1], reverse=True)[:max_assets])
        s = float(sum(child.values()))
        return {t: v / s for t, v in child.items()} if s > 0 else dict(w_a)

    # --- long_short ---
    tickers = list(set(w_a.keys()) | set(w_b.keys()))
    child: Dict[str, float] = {}

    for t in tickers:
        wa = float(w_a.get(t, 0.0))
        wb = float(w_b.get(t, 0.0))
        base = 0.5 * (wa + wb)

        scale = max(0.02, 0.10 * abs(base) + 0.02)  # allow sign flips when small
        noise = float(rng.normal(0.0, scale))
        w = base + noise

        if abs(w) > 1e-6:
            child[t] = float(w)

    if not child:
        child = dict(w_a)

    child = dict(sorted(child.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_assets])

    out = normalize_gross_signed(
        child,
        net_min=LS_NET_MIN,
        net_max=LS_NET_MAX,
        short_budget=LS_SHORT_BUDGET,
    )
    return out if out is not None else dict(w_a)


def mutate_weights(
    weights: Dict[str, float],
    universe: Dict[str, Asset],
    max_assets: int,
    min_assets: int,
    rng: np.random.Generator,
    sigma: float = 0.10,
    replace_prob: float = 0.1,
    *,
    weight_mode: str = "long_only",
) -> Dict[str, float]:

    if weight_mode == "long_only":
        w = {t: float(x) for t, x in weights.items() if float(x) > 0}

        for t in list(w.keys()):
            factor = float(rng.normal(1.0, sigma))
            w[t] = max(float(w[t]) * factor, 0.0)

        if rng.random() < replace_prob:
            if w:
                drop = rng.choice(list(w.keys()))
                w.pop(drop, None)

            available = [t for t in universe.keys() if t not in w]
            if available:
                new_t = rng.choice(available)
                w[str(new_t)] = 1e-3

        if len(w) < min_assets:
            available = [t for t in universe.keys() if t not in w]
            if available:
                needed = min(min_assets - len(w), len(available))
                new_ts = rng.choice(available, size=needed, replace=False)
                for t in new_ts:
                    w[str(t)] = 1e-3

        if len(w) > max_assets:
            w = dict(sorted(w.items(), key=lambda kv: kv[1], reverse=True)[:max_assets])

        s = float(sum(w.values()))
        if s <= 0:
            chosen = rng.choice(list(universe.keys()), size=max_assets, replace=False)
            eq = 1.0 / float(len(chosen))
            return {str(t): eq for t in chosen}

        return {t: v / s for t, v in w.items()}

    # --- long_short ---
    w = {t: float(x) for t, x in weights.items()}

    # multiplicative noise preserves sign
    for t in list(w.keys()):
        factor = float(rng.normal(1.0, sigma))
        w[t] = float(w[t] * factor)
        if abs(w[t]) < 1e-8:
            w.pop(t, None)

    # occasional sign flip
    if w and (rng.random() < 0.10):
        tflip = rng.choice(list(w.keys()))
        w[tflip] = -float(w[tflip])

    # replace: drop one, add one
    if rng.random() < replace_prob:
        if w:
            drop = rng.choice(list(w.keys()))
            w.pop(drop, None)

        available = [t for t in universe.keys() if t not in w]
        if available:
            new_t = rng.choice(available)
            sign = -1.0 if rng.random() < min(0.45, max(0.05, LS_SHORT_BUDGET)) else 1.0
            w[str(new_t)] = float(sign * 1e-3)

    if len(w) < min_assets:
        available = [t for t in universe.keys() if t not in w]
        if available:
            needed = min(min_assets - len(w), len(available))
            new_ts = rng.choice(available, size=needed, replace=False)
            for t in new_ts:
                sign = -1.0 if rng.random() < min(0.45, max(0.05, LS_SHORT_BUDGET)) else 1.0
                w[str(t)] = float(sign * 1e-3)

    if len(w) > max_assets:
        w = dict(sorted(w.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_assets])

    for t in list(w.keys()):
        max_w = float(universe[t].max_weight or 1.0) if t in universe else 1.0
        w[t] = float(np.clip(w[t], -max_w, max_w))

    out = normalize_gross_signed(
        w,
        net_min=LS_NET_MIN,
        net_max=LS_NET_MAX,
        short_budget=LS_SHORT_BUDGET,
    )
    if out is not None:
        return out

    # fallback: keep original
    out0 = normalize_gross_signed(
        dict(weights),
        net_min=LS_NET_MIN,
        net_max=LS_NET_MAX,
        short_budget=LS_SHORT_BUDGET,
    )
    return out0 if out0 is not None else dict(weights)


# ----------------------------
# Worker eval
# ----------------------------

def _eval_candidate_sliced(args):
    try:
        (
            rets_assets,
            tickers,
            weights,
            equity0,
            notional,
            goals,
            main_goal,
            score_config,
            mc_seed,
            n_paths,
            days,
            block_size,
            spec_rows,
            weight_mode,
        ) = args

        return evaluate_portfolio_from_arrays(
            rets_assets=rets_assets,
            tickers=tickers,
            weights=weights,
            equity0=equity0,
            notional=notional,
            goals=(float(goals[0]), float(goals[1]), float(goals[2])),
            main_goal=float(main_goal),
            score_config=score_config,
            mc_seed=mc_seed,
            spec_rows=spec_rows,
            n_paths=int(n_paths),
            days=int(days),
            block_size=block_size,
            weight_mode=weight_mode,   # <<<<<< pass-through
        )
    except Exception as e:
        # optional: print once in a while, but keep it light
        return None

# ----------------------------
# GA search
# ----------------------------

def evolve_portfolios_ga(
    returns: pd.DataFrame,
    universe: Dict[str, Asset],
    lw_cov: pd.DataFrame | None,
    equity0: float,
    notional: float,
    goals: list[float] = (800.0, 1200.0, 2000.0),
    main_goal: float = 2000.0,
    score_config: ScoreConfig | None = None,
    pop_size: int = 80,
    generations: int = 20,
    elite_frac: float = 0.2,
    max_assets: int = 10,
    min_assets: int = 5,
    n_paths_init: int = 3000,
    n_paths_final: int = 20000,
    rng: np.random.Generator | None = None,
    path_source: str = "bootstrap",
    pca_k: int | None = 5,
    block_size: int | tuple[int, int] | None = (8, 12),
    *,
    weight_mode: str = "long_only",
    return_archive: bool = False,
    return_diagnostics: bool = False,
    archive_limit: int | None = 50000,
    archive_fp_decimals: int = 6,
    mutation_sigma_start: float = 0.30,
    mutation_sigma_end: float = 0.05,
    replace_prob_start: float = 0.40,
    replace_prob_end: float = 0.05,
    immigrant_rate_start: float = 0.20,
    immigrant_rate_end: float = 0.03,
    exploration_power: float = 1.5,
    archive_diversity_min_l1: float = 0.15,
    archive_diversity_check_top_k: int = 250,
) -> List[EvalMetrics] | Tuple[List[EvalMetrics], List[EvalMetrics]] | Tuple[List[EvalMetrics], List[EvalMetrics], dict]:

    if rng is None:
        rng = np.random.default_rng()

    goals = [float(g) for g in goals]
    if len(goals) != 3:
        raise ValueError("For now evolve_portfolios_ga expects exactly 3 goals.")
    if score_config is None:
        score_config = ScoreConfig()

    # Restrict to universe columns (will KeyError if mismatch — which is good)
    returns_u = returns[list(universe.keys())]
    returns_clean = returns_u.dropna(how="all")

    spec_df_full = _spectral_profiles_df(
            returns_clean.fillna(0.0),
            bands_days=score_config.fft_bands_days
        )

    # >>> IMPORTANT: use threads on Windows to avoid ProcessPool pickling/spawn stalls
    max_workers = max(2, min(16, (os.cpu_count() or 4)))

    archive: Dict[Tuple[Tuple[str, float], ...], EvalMetrics] = {}
    days = 252

    ruin_cap_strict = float(score_config.ruin_cap if score_config.ruin_cap is not None else 0.10)
    lev = float(notional) / float(equity0) if float(equity0) > 0 else 1.0
    ruin_cap_init = min(0.30, max(0.18, ruin_cap_strict + 0.015 * max(0.0, lev - 1.0)))

    def ruin_cap_for_gen(gen_idx: int) -> float:
        if generations <= 1:
            return ruin_cap_strict
        x = gen_idx / float(generations - 1)
        return float(ruin_cap_strict + (ruin_cap_init - ruin_cap_strict) * (1.0 - x) ** 2)

    elite_strict_after = 0.6

    def _build_tasks(weights_list: list[dict[str, float]], *, n_paths: int) -> list[tuple]:
        tasks: list[tuple] = []
        for w in weights_list:
            tickers = [t for t in w.keys() if t in returns.columns]
            if not tickers:
                continue

            # small arrays per candidate; threads avoid pickling them
            rets_assets = returns[tickers].to_numpy(dtype=np.float32, copy=False)

            spec_rows = None
            if spec_df_full is not None:
                try:
                    spec_rows = spec_df_full.loc[tickers, ["hf", "mf", "lf", "entropy"]].to_numpy(dtype=np.float32, copy=False)
                except Exception:
                    spec_rows = None

            seed_i = int(rng.integers(0, 2**31 - 1))

            tasks.append((
                rets_assets,
                tickers,
                w,
                float(equity0),
                float(notional),
                goals,
                float(main_goal),
                score_config,
                seed_i,
                int(n_paths),
                int(days),
                block_size,
                spec_rows,
                weight_mode,
            ))
        return tasks

    # helper: map tasks with threads and yield results
    def _map_tasks(ex, tasks: list[tuple], *, chunksize: int = 1):
        # ThreadPoolExecutor doesn't support chunksize; we just submit
        futs = [ex.submit(_eval_candidate_sliced, t) for t in tasks]
        for fut in futs:
            try:
                yield fut.result()
            except Exception:
                yield None

    # -----------------------
    # RUN
    # -----------------------
    t0 = time.time()
    last_log = t0
    tasks_total = 0
    eval_ok = 0
    accepted = 0
    rejected_ruin = 0
    eval_failed = 0

    population: List[EvalMetrics] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        while len(population) < pop_size:

            batch = []
            need = pop_size - len(population)

            # --- guard: batch construction shouldn't hang ---
            t_batch0 = time.time()
            while len(batch) < need:
                w = sample_random_weights(
                    universe,
                    max_assets=max_assets,
                    min_assets=min_assets,
                    rng=rng,
                    weight_mode=weight_mode,
                )
                batch.append(w)

                if time.time() - t_batch0 > 10:
                    print(f"[GA:init][warn] slow batch build: {len(batch)}/{need}", flush=True)
                    t_batch0 = time.time()

            tasks = _build_tasks(batch, n_paths=n_paths_init)
            tasks_total += len(tasks)

            # submit tasks
            futs = [ex.submit(_eval_candidate_sliced, t) for t in tasks]

            # IMPORTANT: consume as they complete (not submission order)
            for fut in as_completed(futs):
                try:
                    metrics = fut.result(timeout=120)  # <- adjust, but keep a timeout
                except TimeoutError:
                    eval_failed += 1
                    continue
                except Exception:
                    eval_failed += 1
                    continue

                if metrics is None:
                    eval_failed += 1
                    continue

                eval_ok += 1
                if metrics.ruin_prob_1y > ruin_cap_init:
                    rejected_ruin += 1
                    continue

                population.append(metrics)
                accepted += 1
                if len(population) >= pop_size:
                    break

            now = time.time()
            if now - last_log > 5:
                last_log = now
                print(
                    f"[GA:init] pop={len(population)}/{pop_size} "
                    f"tasks={tasks_total} ok={eval_ok} fail={eval_failed} "
                    f"acc={accepted} rej_ruin={rejected_ruin} "
                    f"ruin_cap_init={ruin_cap_init:.3f} "
                    f"elapsed={now-t0:.1f}s",
                    flush=True
                )

        # ---------- generations ----------
        n_elite = max(1, int(pop_size * elite_frac))
        generation_diagnostics: list[dict] = []

        for gen in range(generations):
            n_paths = int(n_paths_init + (n_paths_final - n_paths_init) * (gen / max(1, generations - 1)))
            population.sort(key=lambda m: m.score, reverse=True)

            sigma_gen = _schedule_value(
                gen_idx=gen,
                generations=generations,
                start=float(mutation_sigma_start),
                end=float(mutation_sigma_end),
                power=float(exploration_power),
            )
            replace_prob_gen = _schedule_value(
                gen_idx=gen,
                generations=generations,
                start=float(replace_prob_start),
                end=float(replace_prob_end),
                power=float(exploration_power),
            )
            immigrant_rate_gen = _schedule_value(
                gen_idx=gen,
                generations=generations,
                start=float(immigrant_rate_start),
                end=float(immigrant_rate_end),
                power=float(exploration_power),
            )

            if generations > 1 and (gen / float(generations - 1)) >= elite_strict_after:
                feasible = [m for m in population if m.ruin_prob_1y <= ruin_cap_strict]
                elites = feasible[:n_elite] if len(feasible) >= n_elite else population[:n_elite]
            else:
                elites = population[:n_elite]

            new_population = elites.copy()

            for em in elites:
                _archive_add(
                    archive,
                    em,
                    decimals=archive_fp_decimals,
                    archive_limit=archive_limit,
                    diversity_min_l1=float(archive_diversity_min_l1),
                    diversity_check_top_k=int(archive_diversity_check_top_k),
                )

            remaining_slots = max(0, pop_size - len(new_population))
            n_immigrants = min(remaining_slots, max(0, int(round(pop_size * float(immigrant_rate_gen)))))
            n_children = max(0, remaining_slots - n_immigrants)

            children: list[dict[str, float]] = []
            parent_pool = population[: max(pop_size // 2, n_elite)]
            while len(children) < n_children:
                try:
                    replace_parents = len(parent_pool) < 2
                    parents = rng.choice(parent_pool, size=2, replace=replace_parents)
                    p_a, p_b = parents[0], parents[1]

                    child_w = crossover_weights(
                        p_a.weights,
                        p_b.weights,
                        max_assets=max_assets,
                        rng=rng,
                        weight_mode=weight_mode,
                    )
                    child_w = mutate_weights(
                        child_w,
                        universe=universe,
                        max_assets=max_assets,
                        min_assets=min_assets,
                        rng=rng,
                        sigma=float(sigma_gen),
                        replace_prob=float(replace_prob_gen),
                        weight_mode=weight_mode,
                    )
                    children.append(child_w)
                except Exception:
                    continue

            immigrants: list[dict[str, float]] = []
            for _ in range(n_immigrants):
                try:
                    immigrants.append(
                        sample_random_weights(
                            universe,
                            max_assets=max_assets,
                            min_assets=min_assets,
                            rng=rng,
                            weight_mode=weight_mode,
                        )
                    )
                except Exception:
                    continue

            tasks = _build_tasks(children + immigrants, n_paths=n_paths)
            cap_gen = ruin_cap_for_gen(gen)
            gen_eval_ok = 0
            gen_eval_failed = 0
            gen_rejected_ruin = 0

            for metrics in _map_tasks(ex, tasks):
                if metrics is None:
                    gen_eval_failed += 1
                    continue

                gen_eval_ok += 1
                _archive_add(
                    archive,
                    metrics,
                    decimals=archive_fp_decimals,
                    archive_limit=archive_limit,
                    diversity_min_l1=float(archive_diversity_min_l1),
                    diversity_check_top_k=int(archive_diversity_check_top_k),
                )

                if metrics.ruin_prob_1y > cap_gen:
                    gen_rejected_ruin += 1
                    continue

                new_population.append(metrics)
                if len(new_population) >= pop_size:
                    break

            # If the generation was too heavily filtered, top up with prior best candidates.
            # This avoids parent-pool collapse in the next generation while keeping the run moving.
            if len(new_population) < max(2, n_elite) and population:
                for m in population:
                    if m not in new_population:
                        new_population.append(m)
                    if len(new_population) >= max(2, n_elite):
                        break

            population = new_population
            population.sort(key=lambda m: m.score, reverse=True)

            best = population[0]
            diversity = _portfolio_diversity_summary(population)
            generation_diagnostics.append(
                {
                    "generation": int(gen + 1),
                    "generation_index": int(gen),
                    "n_paths": int(n_paths),
                    "ruin_cap": float(cap_gen),
                    "mutation_sigma": float(sigma_gen),
                    "replace_prob": float(replace_prob_gen),
                    "immigrant_rate": float(immigrant_rate_gen),
                    "immigrants_requested": int(n_immigrants),
                    "children_requested": int(n_children),
                    "eval_ok": int(gen_eval_ok),
                    "eval_failed": int(gen_eval_failed),
                    "rejected_ruin": int(gen_rejected_ruin),
                    "accepted_population_size": int(len(population)),
                    "archive_size": int(len(archive)),
                    "best_score": float(best.score),
                    "best_ruin_prob_1y": float(best.ruin_prob_1y),
                    "diversity": diversity,
                }
            )

            g1, g2, g3 = goals
            print(
                f"Gen {gen+1}/{generations} | best score={best.score:.4f} "
                f"P({g1:.0f})={best.p_hit_goal_1_1y:.2%} "
                f"P({g2:.0f})={best.p_hit_goal_2_1y:.2%} "
                f"P({g3:.0f})={best.p_hit_goal_3_1y:.2%} "
                f"ruin={best.ruin_prob_1y:.2%} "
                f"sigma={sigma_gen:.3f} replace={replace_prob_gen:.3f} immigrants={n_immigrants} "
                f"unique_assets={diversity.get('unique_assets_used', 0)}",
                flush=True,
            )

    population.sort(key=lambda m: m.score, reverse=True)
    archive_sorted = sorted(archive.values(), key=lambda m: m.score, reverse=True)

    diagnostics = {
        "schema_version": "ga_exploration_diagnostics_v1",
        "config": {
            "mutation_sigma_start": float(mutation_sigma_start),
            "mutation_sigma_end": float(mutation_sigma_end),
            "replace_prob_start": float(replace_prob_start),
            "replace_prob_end": float(replace_prob_end),
            "immigrant_rate_start": float(immigrant_rate_start),
            "immigrant_rate_end": float(immigrant_rate_end),
            "exploration_power": float(exploration_power),
            "archive_diversity_min_l1": float(archive_diversity_min_l1),
            "archive_diversity_check_top_k": int(archive_diversity_check_top_k),
        },
        "init": {
            "tasks_total": int(tasks_total),
            "eval_ok": int(eval_ok),
            "eval_failed": int(eval_failed),
            "accepted": int(accepted),
            "rejected_ruin": int(rejected_ruin),
            "ruin_cap_init": float(ruin_cap_init),
        },
        "generations": generation_diagnostics,
        "final_population": _portfolio_diversity_summary(population),
        "archive": {
            "size": int(len(archive_sorted)),
            "diversity": _portfolio_diversity_summary(archive_sorted[: min(250, len(archive_sorted))]),
        },
    }

    if not return_archive:
        return population

    if return_diagnostics:
        return population, archive_sorted, diagnostics

    return population, archive_sorted



# ----------------------------
# Annealing refinement (mode-aware)
# ----------------------------

def refine_portfolio_annealing(
    base_metrics: EvalMetrics,
    returns: pd.DataFrame,
    universe: Dict[str, Asset],
    lw_cov: pd.DataFrame | None,
    equity0: float,
    notional: float,
    goals: list[float] = (600.0, 800.0, 2000.0),
    main_goal: float = 800.0,
    score_config: ScoreConfig | None = None,
    max_assets: int = 10,
    min_assets: int = 5,
    n_steps: int = 200,
    temp_start: float = 1.0,
    temp_end: float = 0.05,
    n_paths_init: int = 3000,
    n_paths_final: int = 20000,
    rng: np.random.Generator | None = None,
    path_source: str = "bootstrap",
    pca_k: int | None = 5,
    block_size: int | tuple[int, int] | None = (8, 12),
    *,
    weight_mode: str = "long_only",   # <<<<<< NEW
) -> EvalMetrics:
    if rng is None:
        rng = np.random.default_rng()

    goals = [float(g) for g in goals]
    if len(goals) != 3:
        raise ValueError("For now refine_portfolio_annealing expects exactly 3 goals.")

    if score_config is None:
        score_config = ScoreConfig()

    returns_u = returns[list(universe.keys())]
    returns_clean = returns_u.dropna(how="all")
    spec_df_full = _spectral_profiles_df(returns_clean.fillna(0.0), bands_days=score_config.fft_bands_days)

    current = base_metrics
    best = base_metrics

    for step in range(n_steps):
        T = temp_start * (temp_end / temp_start) ** (step / max(1, n_steps - 1))
        x = step / max(1, n_steps - 1)
        x = x * x
        n_paths_step = int(n_paths_init + (n_paths_final - n_paths_init) * x)
        mc_seed = int(rng.integers(0, 2**31 - 1))

        try:
            cand_w = mutate_weights(
                current.weights,
                universe=universe,
                max_assets=max_assets,
                min_assets=min_assets,
                rng=rng,
                sigma=0.05,
                replace_prob=0.05,
                weight_mode=weight_mode,
            )

            tickers = [t for t in cand_w.keys() if t in returns.columns]
            if not tickers:
                continue

            X = returns[tickers].to_numpy(dtype=np.float32, copy=False)

            try:
                spec_rows = spec_df_full.loc[tickers, ["hf", "mf", "lf", "entropy"]].to_numpy(
                    dtype=np.float32, copy=False
                )
            except Exception:
                spec_rows = None

            cand = evaluate_portfolio_from_arrays(
                rets_assets=X,
                tickers=tickers,
                weights=cand_w,
                equity0=float(equity0),
                notional=float(notional),
                goals=(float(goals[0]), float(goals[1]), float(goals[2])),
                main_goal=float(main_goal),
                score_config=score_config,
                mc_seed=mc_seed,
                spec_rows=spec_rows,
                n_paths=int(n_paths_step),
                days=252,
                block_size=block_size,
                weight_mode=weight_mode
            )

        except Exception:
            continue

        delta = cand.score - current.score

        if delta >= 0:
            current = cand
            if cand.score > best.score:
                best = cand
        else:
            accept_prob = np.exp(delta / max(T, 1e-8))
            if rng.random() < accept_prob:
                current = cand

    return best


def evaluate_weights_for_search(
    *,
    returns: pd.DataFrame,
    weights: dict[str, float],
    equity0: float,
    notional: float,
    goals: list[float],
    main_goal: float,
    score_config: ScoreConfig | None = None,
    n_paths: int = 5000,
    mc_seed: int | None = 123,
    block_size: int | tuple[int, int] | None = (8, 12),
    weight_mode: str = "long_short",
) -> EvalMetrics:
    """
    Evaluate an existing weight dictionary using the same array-based evaluator
    used by GA workers and annealing.

    This is intended for:
      - current live portfolio evaluation
      - local transition optimizer baseline
      - shadow/current comparisons

    It keeps Milestone 17 aligned with the portfolio search engine.
    """
    if score_config is None:
        score_config = ScoreConfig()

    weights_n = {
        str(k): float(v)
        for k, v in (weights or {}).items()
        if str(k) in returns.columns and np.isfinite(float(v)) and abs(float(v)) > 1e-12
    }

    if not weights_n:
        raise ValueError("No valid weights overlap returns columns.")

    tickers = list(weights_n.keys())

    returns_clean = returns[tickers].dropna(how="all")
    if returns_clean.empty:
        raise ValueError("No returns rows available for selected weights.")

    spec_df_full = _spectral_profiles_df(
        returns[tickers].fillna(0.0),
        bands_days=score_config.fft_bands_days,
    )

    try:
        spec_rows = spec_df_full.loc[tickers, ["hf", "mf", "lf", "entropy"]].to_numpy(
            dtype=np.float32,
            copy=False,
        )
    except Exception:
        spec_rows = None

    X = returns[tickers].to_numpy(dtype=np.float32, copy=False)

    return evaluate_portfolio_from_arrays(
        rets_assets=X,
        tickers=tickers,
        weights=weights_n,
        equity0=float(equity0),
        notional=float(notional),
        goals=(float(goals[0]), float(goals[1]), float(goals[2])),
        main_goal=float(main_goal),
        score_config=score_config,
        mc_seed=mc_seed,
        spec_rows=spec_rows,
        n_paths=int(n_paths),
        days=252,
        block_size=block_size,
        weight_mode=weight_mode,
    )
