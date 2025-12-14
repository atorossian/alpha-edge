# portfolio_search.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from storage.data_storage import Asset  # the dataclass we defined earlier
from optimizer_engine import evaluate_portfolio_candidate, PortfolioCandidateMetrics


def sample_random_weights(
    universe: Dict[str, Asset],
    max_assets: int | None = None,
    min_assets: int = 5,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    """
    Sample a random weight vector over the given universe.
    Respects per-asset max_weight from Asset, but not min_weight (you can add later).
    """
    if rng is None:
        rng = np.random.default_rng()

    tickers = list(universe.keys())
    n_total = len(tickers)

    if max_assets is None or max_assets > n_total:
        max_assets = n_total

    # 1) choose how many assets will be non-zero
    k = rng.integers(low=min_assets, high=max_assets + 1)
    chosen = list(rng.choice(tickers, size=k, replace=False))

    # 2) Dirichlet random weights over chosen assets
    alpha = np.ones(k)
    raw_w = rng.dirichlet(alpha)  # sums to 1

    # 3) Apply per-asset max_weight & renormalize
    weights = {}
    for i, t in enumerate(chosen):
        w = float(raw_w[i])
        max_w = float(universe[t].max_weight or 1.0)
        weights[t] = min(w, max_w)

    # If everything got clamped too hard, renormalize
    total = sum(weights.values())
    if total <= 0:
        # fallback: equal weights over chosen
        equal_w = 1.0 / k
        weights = {t: equal_w for t in chosen}
    else:
        weights = {t: w / total for t, w in weights.items()}

    return weights

def random_search_portfolios(
    returns: pd.DataFrame,
    universe: Dict[str, Asset],
    lw_cov: pd.DataFrame,
    equity0: float,
    notional: float,
    n_candidates: int = 200,
    lambda_ruin: float = 0.5,
    gamma_vol: float = 0.0,
    max_assets: int | None = None,
    min_assets: int = 5,
    ruin_cap: float | None = None,
    rng: np.random.Generator | None = None,
) -> List[PortfolioCandidateMetrics]:
    """
    Perform random search over portfolio weights.
    - Sample n_candidates random weight vectors.
    - Evaluate each via leveraged MC + LW vol.
    - Optionally discard portfolios with ruin_prob > ruin_cap.
    - Return a list of PortfolioCandidateMetrics sorted by score (desc).
    """
    if rng is None:
        rng = np.random.default_rng()

    results: List[PortfolioCandidateMetrics] = []

    for i in range(n_candidates):
        try:
            weights = sample_random_weights(
                universe=universe,
                max_assets=max_assets,
                min_assets=min_assets,
                rng=rng,
            )

            metrics = evaluate_portfolio_candidate(
                returns=returns,
                weights=weights,
                equity0=equity0,
                notional=notional,
                lw_cov=lw_cov,
                days=252,
                n_paths=20000,
                lambda_ruin=lambda_ruin,
                gamma_vol=gamma_vol,
            )

            if ruin_cap is not None and metrics.ruin_prob_1y > ruin_cap:
                continue

            results.append(metrics)

        except ValueError:
            # e.g. not enough data for chosen subset; just skip
            continue

    # sort by score descending
    results.sort(key=lambda m: m.score, reverse=True)
    return results

def crossover_weights(
    w_a: Dict[str, float],
    w_b: Dict[str, float],
    max_assets: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Simple blend crossover: union of tickers, weights approx avg of parents.
    """
    tickers = list(set(w_a.keys()) | set(w_b.keys()))
    child = {}

    for t in tickers:
        wa = w_a.get(t, 0.0)
        wb = w_b.get(t, 0.0)
        # average with tiny noise
        base = 0.5 * (wa + wb)
        if base <= 0:
            continue
        noise = rng.normal(0.0, 0.05 * base)
        w = max(base + noise, 0.0)
        if w > 0:
            child[t] = w

    if not child:
        # fallback: one of the parents
        return dict(w_a)

    # keep only top max_assets tickers
    child = dict(sorted(child.items(), key=lambda kv: kv[1], reverse=True)[:max_assets])

    # renormalize
    total = sum(child.values())
    if total <= 0:
        return dict(w_a)
    child = {t: w / total for t, w in child.items()}
    return child


def mutate_weights(
    weights: Dict[str, float],
    universe: Dict[str, Asset],
    max_assets: int,
    min_assets: int,
    rng: np.random.Generator,
    sigma: float = 0.10,
    replace_prob: float = 0.1,
) -> Dict[str, float]:
    """
    Small Gaussian noise on weights + occasional asset replacement.
    """
    w = dict(weights)

    # 1) perturb existing weights
    for t in list(w.keys()):
        factor = rng.normal(1.0, sigma)
        w[t] = max(w[t] * factor, 0.0)

    # 2) occasional replacement of one asset
    if rng.random() < replace_prob:
        if w:
            drop = rng.choice(list(w.keys()))
            w.pop(drop, None)

        # add a random new asset from universe
        available = [t for t in universe.keys() if t not in w]
        if available:
            new_t = rng.choice(available)
            w[new_t] = 1e-3  # tiny seed weight

    # ensure at least min_assets
    if len(w) < min_assets:
        available = [t for t in universe.keys() if t not in w]
        if available:
            needed = min(min_assets - len(w), len(available))
            new_ts = rng.choice(available, size=needed, replace=False)
            for t in new_ts:
                w[t] = 1e-3

    # limit to max_assets
    if len(w) > max_assets:
        w = dict(sorted(w.items(), key=lambda kv: kv[1], reverse=True)[:max_assets])

    # renormalize
    total = sum(w.values())
    if total <= 0:
        # fallback: equal weights over a random subset
        chosen = rng.choice(list(universe.keys()), size=max_assets, replace=False)
        eq = 1.0 / len(chosen)
        return {t: eq for t in chosen}

    w = {t: wv / total for t, wv in w.items()}
    return w

def evolve_portfolios_ga(
    returns: pd.DataFrame,
    universe: Dict[str, Asset],
    lw_cov: pd.DataFrame,
    equity0: float,
    notional: float,
    *,
    pop_size: int = 80,
    generations: int = 20,
    elite_frac: float = 0.2,
    lambda_ruin: float = 0.5,
    gamma_vol: float = 0.0,
    max_assets: int = 10,
    min_assets: int = 5,
    ruin_cap: float | None = 0.30,
    n_paths_init: int = 3000,
    n_paths_final: int = 20000,
    rng: np.random.Generator | None = None,
) -> List[PortfolioCandidateMetrics]:
    """
    Genetic algorithm over portfolios using your existing evaluate_portfolio_candidate.
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- init population ---
    population: List[PortfolioCandidateMetrics] = []
    while len(population) < pop_size:
        try:
            from portfolio_search import sample_random_weights  # or adjust import
        except ImportError:
            # if sample_random_weights is in same file, just call it directly
            pass

        try:
            weights = sample_random_weights(
                universe=universe,
                max_assets=max_assets,
                min_assets=min_assets,
                rng=rng,
            )
            metrics = evaluate_portfolio_candidate(
                returns=returns,
                weights=weights,
                equity0=equity0,
                notional=notional,
                lw_cov=lw_cov,
                days=252,
                n_paths=n_paths_init,
                lambda_ruin=lambda_ruin,
                gamma_vol=gamma_vol,
            )
            if ruin_cap is not None and metrics.ruin_prob_1y > ruin_cap:
                continue
            population.append(metrics)
        except Exception:
            continue

    n_elite = max(1, int(pop_size * elite_frac))

    for gen in range(generations):
        # gradually increase paths
        n_paths = int(
            n_paths_init
            + (n_paths_final - n_paths_init) * (gen / max(1, generations - 1))
        )

        # sort by score
        population.sort(key=lambda m: m.score, reverse=True)
        elites = population[:n_elite]

        new_population: List[PortfolioCandidateMetrics] = elites.copy()

        # fill rest of population
        while len(new_population) < pop_size:
            try:
                # parent selection: tournament over a few randoms
                parents = rng.choice(population[: max(pop_size // 2, n_elite)], size=2, replace=False)
                p_a, p_b = parents[0], parents[1]

                child_w = crossover_weights(p_a.weights, p_b.weights, max_assets=max_assets, rng=rng)
                child_w = mutate_weights(
                    child_w,
                    universe=universe,
                    max_assets=max_assets,
                    min_assets=min_assets,
                    rng=rng,
                    sigma=0.10,
                    replace_prob=0.15,
                )

                metrics = evaluate_portfolio_candidate(
                    returns=returns,
                    weights=child_w,
                    equity0=equity0,
                    notional=notional,
                    lw_cov=lw_cov,
                    days=252,
                    n_paths=n_paths,
                    lambda_ruin=lambda_ruin,
                    gamma_vol=gamma_vol,
                )

                if ruin_cap is not None and metrics.ruin_prob_1y > ruin_cap:
                    continue

                new_population.append(metrics)

            except Exception:
                continue

        population = new_population

        # optional: print quick progress
        best = population[0]
        print(
            f"Gen {gen+1}/{generations} | best score={best.score:.4f} "
            f"P2000={best.p_hit_2000_1y:.2%} ruin={best.ruin_prob_1y:.2%}"
        )

    # final sort
    population.sort(key=lambda m: m.score, reverse=True)
    return population
