# stability_energy.py
from __future__ import annotations

import numpy as np
from alpha_edge.core.schemas import StabilityEnergyConfig, StabilityReport


def compute_stability_report(
    equity_paths: np.ndarray,  # (n_paths, n_days+1)
    *,
    n_days: int,
    cfg: StabilityEnergyConfig,
) -> StabilityReport:
    E = np.asarray(equity_paths, dtype=np.float64)
    if E.ndim != 2 or E.shape[1] != (n_days + 1):
        raise ValueError(f"equity_paths must have shape (n_paths, {n_days+1})")

    # running peaks
    peaks = np.maximum.accumulate(E, axis=1)
    peaks_safe = np.maximum(peaks, 1e-12)

    # drawdown series in [0,1]
    dd = 1.0 - (E / peaks_safe)

    # per-path max drawdown
    mdd = np.max(dd, axis=1)
    trough_idx = np.argmax(dd, axis=1)  # argmax drawdown time

    # underwater fraction
    underwater = (E < peaks).mean(axis=1)

    # breach probability
    p_breach = float(np.mean(mdd >= float(cfg.breach_dd)))

    # CDaR on MDD distribution
    alpha = float(cfg.alpha_cdar)
    q = float(np.quantile(mdd, alpha))
    tail = mdd[mdd >= q]
    cdar = float(tail.mean()) if tail.size else float(mdd.mean())

    # Time-to-recovery (TTR)
    P = E.shape[0]
    ttr = np.empty(P, dtype=np.float64)

    for i in range(P):
        t0 = int(trough_idx[i])
        if t0 <= 0:
            ttr[i] = 0.0
            continue

        peak_time = int(np.argmax(E[i, : t0 + 1]))
        peak_eq = float(E[i, peak_time])

        after = E[i, t0 + 1 :]
        rec = np.where(after >= peak_eq)[0]
        if rec.size == 0:
            ttr[i] = float(n_days)  # no recovery within horizon
        else:
            ttr[i] = float((t0 + 1 + rec[0]) - peak_time)

    # normalized components (~[0,1])
    mdd_mean = float(mdd.mean())
    ttr_mean_norm = float(np.mean(ttr) / max(1, n_days))
    underwater_mean = float(np.mean(underwater))

    # energy
    energy = float(
        cfg.lambda_mdd * mdd_mean
        + cfg.lambda_cdar * cdar
        + cfg.lambda_ttr * ttr_mean_norm
        + cfg.lambda_breach * p_breach
        + cfg.lambda_underwater * underwater_mean
    )

    return StabilityReport(
        energy=energy,
        mdd_mean=mdd_mean,
        cdar_alpha=float(cdar),
        ttr_mean_norm=ttr_mean_norm,
        p_breach=p_breach,
        underwater_mean=underwater_mean,
    )
