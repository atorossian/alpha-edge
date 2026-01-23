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
# Fingerprint / archive
# ----------------------------

def _weights_fingerprint(weights: Dict[str, float], decimals: int = 6) -> Tuple[Tuple[str, float], ...]:
    """
    Stable fingerprint for deduplication.
    - sort tickers
    - round weights to avoid tiny float differences
    - keep signed weights (so long_short portfolios don't collide with long_only)
    """
    items = tuple(
        sorted(
            (t, round(float(w), decimals))
            for t, w in weights.items()
            if abs(float(w)) > 0.0
        )
    )
    return items


def _archive_add(
    archive: Dict[Tuple[Tuple[str, float], ...], EvalMetrics],
    m: EvalMetrics,
    *,
    decimals: int = 6,
    archive_limit: int | None = None,
) -> None:
    fp = _weights_fingerprint(m.weights, decimals=decimals)
    cur = archive.get(fp)
    if (cur is None) or (m.score > cur.score):
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
    archive_limit: int | None = 50000,
    archive_fp_decimals: int = 6,
) -> List[EvalMetrics] | Tuple[List[EvalMetrics], List[EvalMetrics]]:

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

        for gen in range(generations):
            n_paths = int(n_paths_init + (n_paths_final - n_paths_init) * (gen / max(1, generations - 1)))
            population.sort(key=lambda m: m.score, reverse=True)

            if generations > 1 and (gen / float(generations - 1)) >= elite_strict_after:
                feasible = [m for m in population if m.ruin_prob_1y <= ruin_cap_strict]
                elites = feasible[:n_elite] if len(feasible) >= n_elite else population[:n_elite]
            else:
                elites = population[:n_elite]

            new_population = elites.copy()

            for em in elites:
                _archive_add(archive, em, decimals=archive_fp_decimals, archive_limit=archive_limit)

            children: list[dict[str, float]] = []
            while len(children) < (pop_size - len(new_population)):
                try:
                    parents = rng.choice(population[: max(pop_size // 2, n_elite)], size=2, replace=False)
                    p_a, p_b = parents[0], parents[1]

                    child_w = crossover_weights(
                        p_a.weights, p_b.weights,
                        max_assets=max_assets,
                        rng=rng,
                        weight_mode=weight_mode
                    )
                    child_w = mutate_weights(
                        child_w,
                        universe=universe,
                        max_assets=max_assets,
                        min_assets=min_assets,
                        rng=rng,
                        sigma=0.15,
                        replace_prob=0.25,
                        weight_mode=weight_mode,
                    )
                    children.append(child_w)
                except Exception:
                    continue

            tasks = _build_tasks(children, n_paths=n_paths)
            cap_gen = ruin_cap_for_gen(gen)

            for metrics in _map_tasks(ex, tasks):
                if metrics is None:
                    continue

                _archive_add(archive, metrics, decimals=archive_fp_decimals, archive_limit=archive_limit)

                if metrics.ruin_prob_1y > cap_gen:
                    continue

                new_population.append(metrics)
                if len(new_population) >= pop_size:
                    break

            population = new_population

            best = population[0]
            g1, g2, g3 = goals
            print(
                f"Gen {gen+1}/{generations} | best score={best.score:.4f} "
                f"P({g1:.0f})={best.p_hit_goal_1_1y:.2%} "
                f"P({g2:.0f})={best.p_hit_goal_2_1y:.2%} "
                f"P({g3:.0f})={best.p_hit_goal_3_1y:.2%} "
                f"ruin={best.ruin_prob_1y:.2%}",
                flush=True
            )

    population.sort(key=lambda m: m.score, reverse=True)

    if not return_archive:
        return population

    archive_sorted = sorted(archive.values(), key=lambda m: m.score, reverse=True)
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
