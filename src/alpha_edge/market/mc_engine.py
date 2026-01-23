# mc_engine.py
from __future__ import annotations

from typing import Any
import numpy as np


def simulate_leveraged_paths_vectorized(
    port_rets: np.ndarray | None,   # 1D history (N,) IF bootstrapping
    equity0: float,
    notional: float,               # GROSS notional (USD)
    goals: list[float],
    n_paths: int = 12000,
    n_days: int = 252,
    seed: int | None = None,
    idx: np.ndarray | None = None,  # optional explicit indices (n_paths, n_days)
    block_size: int | tuple[int, int] | None = (8, 12),
    clamp_ruined_to_zero: bool = True,
    return_paths: bool = False,
    precomputed_r: np.ndarray | None = None,  # shape (n_paths, n_days)
    mmr: float = 0.0,               # maintenance margin ratio on gross notional
) -> dict[str, Any]:
    """
    Fixed-notional leveraged MC.

    Equity evolves with constant gross exposure:
        E_{t+1} = E_t + notional * r_t

    If mmr > 0, liquidate when:
        E <= mmr * notional

    This is stable for long/short and does not suffer from the "mmr*L>1" trap.

    Output keys:
      ruin_prob, p_hit, goal_stats, end_p5..end_p95 (+ equity_paths optional)
    """
    rng = np.random.default_rng(seed)

    goals_arr = np.asarray([float(g) for g in goals], dtype=np.float32)
    if goals_arr.size != 3:
        raise ValueError("For now, exactly 3 goals are required.")
    G = int(goals_arr.size)

    goals_sorted = sorted([float(g) for g in goals_arr.tolist()])
    goal_to_idx = {float(g): i for i, g in enumerate(goals_arr.tolist())}

    # --- returns source setup ---
    stationary = False
    cur_idx = None
    p_restart = None
    port_rets_arr = None
    N = None

    if precomputed_r is not None:
        R = np.asarray(precomputed_r)
        if R.shape != (n_paths, n_days):
            raise ValueError(f"precomputed_r must have shape {(n_paths, n_days)}")
        if R.dtype != np.float32:
            R = R.astype(np.float32, copy=False)
    else:
        if idx is not None:
            idx = np.asarray(idx)
            if idx.shape != (n_paths, n_days):
                raise ValueError(f"idx must have shape {(n_paths, n_days)}")
        port_rets_arr = np.asarray(port_rets, dtype=np.float32)
        if port_rets_arr.ndim != 1 or port_rets_arr.size < 2:
            raise ValueError("port_rets must be 1D array with enough history")
        N = int(port_rets_arr.size)

        stationary = isinstance(block_size, tuple)
        if stationary:
            bmin, bmax = int(block_size[0]), int(block_size[1])
            if bmin < 1 or bmax < bmin:
                raise ValueError("block_size tuple must be (min>=1, max>=min)")
            avg_block = 0.5 * (bmin + bmax)
            p_restart = 1.0 / avg_block
            cur_idx = rng.integers(0, N, size=n_paths, dtype=np.int64)

        if (block_size is not None) and isinstance(block_size, int):
            B = int(block_size)
            if B < 1:
                raise ValueError("block_size int must be >= 1")
            block_pos = rng.integers(0, max(1, N - B), size=n_paths, dtype=np.int64)
            block_off = np.zeros(n_paths, dtype=np.int64)
        else:
            block_pos = None
            block_off = None

    # --- state ---
    E = np.full(n_paths, np.float32(equity0), dtype=np.float32)
    ruined = np.zeros(n_paths, dtype=bool)

    hit_any = np.zeros((n_paths, G), dtype=bool)
    hit_time = np.full((n_paths, G), -1, dtype=np.int32)

    if return_paths:
        E_paths = np.empty((n_paths, n_days + 1), dtype=np.float32)
        E_paths[:, 0] = E

    notional_f = np.float32(notional)
    margin_floor = np.float32(mmr) * notional_f if float(mmr) > 0 else np.float32(0.0)

    # --- time loop ---
    for t in range(n_days):
        alive = ~ruined

        if precomputed_r is not None:
            r_t = R[:, t]

        elif idx is not None:
            r_t = port_rets_arr[idx[:, t]].astype(np.float32, copy=False)

        elif stationary:
            if t > 0:
                restart = (rng.random(n_paths) < p_restart) | (cur_idx >= N - 1)
                new_draws = rng.integers(0, N, size=n_paths, dtype=np.int64)
                cur_idx = np.where(restart, new_draws, cur_idx + 1)
            r_t = port_rets_arr[cur_idx]

        elif (block_size is None) or (block_pos is None):
            draw = rng.integers(0, N, size=n_paths, dtype=np.int64)
            r_t = port_rets_arr[draw]

        else:
            B = int(block_size)
            need_new = block_off >= B
            if np.any(need_new):
                block_pos[need_new] = rng.integers(0, max(1, N - B), size=int(np.sum(need_new)), dtype=np.int64)
                block_off[need_new] = 0
            draw = block_pos + block_off
            draw = np.minimum(draw, N - 1)
            r_t = port_rets_arr[draw]
            block_off += 1

        pnl = notional_f * r_t
        E[alive] = E[alive] + pnl[alive]

        newly_ruined = alive & (E <= margin_floor)
        if newly_ruined.any():
            ruined[newly_ruined] = True
            if clamp_ruined_to_zero:
                E[newly_ruined] = np.float32(0.0)

        reached = E[:, None] >= goals_arr[None, :]
        newly_hit = reached & (~hit_any)
        if newly_hit.any():
            hit_time[newly_hit] = t + 1
            hit_any[newly_hit] = True

        if return_paths:
            E_paths[:, t + 1] = E

    ruin_prob = float(np.mean(ruined))

    p_hit = {g: float(np.mean(hit_any[:, goal_to_idx[g]])) for g in goals_sorted}

    goal_stats = {}
    for g in goals_sorted:
        j = goal_to_idx[g]
        if p_hit[g] > 0.0:
            times = hit_time[:, j]
            goal_stats[g] = {"median_time_days": float(np.nanmedian(times[hit_any[:, j]]))}
        else:
            goal_stats[g] = {"median_time_days": None}

    end_eq = E.astype(np.float64, copy=False)
    p5, p25, p50, p75, p95 = np.percentile(end_eq, [5, 25, 50, 75, 95])

    out = dict(
        ruin_prob=ruin_prob,
        p_hit=p_hit,
        goal_stats=goal_stats,
        end_p5=float(p5),
        end_p25=float(p25),
        end_p50=float(p50),
        end_p75=float(p75),
        end_p95=float(p95),
    )
    if return_paths:
        out["equity_paths"] = E_paths
    return out
