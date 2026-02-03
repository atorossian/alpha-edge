# alpha_report.py
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------
# Alpha / regression utilities
# -----------------------------
def _to_series(x: Any, name: str) -> pd.Series:
    if x is None:
        return pd.Series(dtype="float64", name=name)
    if isinstance(x, pd.Series):
        s = x.copy()
        s.name = name
        return s
    return pd.Series(x, dtype="float64", name=name)


def _align_returns(port_rets: pd.Series, bench_rets: pd.Series) -> tuple[pd.Series, pd.Series]:
    p = _to_series(port_rets, "port").replace([np.inf, -np.inf], np.nan).dropna()
    b = _to_series(bench_rets, "bench").replace([np.inf, -np.inf], np.nan).dropna()

    if p.empty or b.empty:
        return p, b

    df = pd.concat([p, b], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype="float64", name="port"), pd.Series(dtype="float64", name="bench")

    return df["port"], df["bench"]


def _slice_window(s: pd.Series, n: int) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype="float64")
    return s.iloc[-n:] if len(s) > n else s


def _capm_alpha_beta(
    port_rets: pd.Series,
    bench_rets: pd.Series,
    *,
    rf_daily: float = 0.0,
    ann_factor: int = 252,
    min_n: int = 40,
) -> dict[str, float]:
    """
    CAPM regression on excess returns:
      (Rp - Rf) = alpha + beta * (Rb - Rf) + eps

    Returns:
      alpha_ann, beta, r2, resid_vol_ann, t_alpha, n
    """
    p, b = _align_returns(port_rets, bench_rets)
    n = len(p)
    if n < min_n:
        return {
            "n": float(n),
            "alpha_ann": float("nan"),
            "beta": float("nan"),
            "r2": float("nan"),
            "resid_vol_ann": float("nan"),
            "t_alpha": float("nan"),
        }

    y = (p - rf_daily).to_numpy(dtype="float64")
    x = (b - rf_daily).to_numpy(dtype="float64")

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    sxx = float(np.sum((x - x_mean) ** 2))
    if not np.isfinite(sxx) or sxx <= 0:
        return {
            "n": float(n),
            "alpha_ann": float("nan"),
            "beta": float("nan"),
            "r2": float("nan"),
            "resid_vol_ann": float("nan"),
            "t_alpha": float("nan"),
        }

    beta = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    alpha_daily = float(y_mean - beta * x_mean)

    y_hat = alpha_daily + beta * x
    resid = y - y_hat

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)

    resid_std = float(np.std(resid, ddof=2)) if n > 2 else float("nan")
    resid_vol_ann = float(resid_std * math.sqrt(ann_factor)) if np.isfinite(resid_std) else float("nan")

    s2 = float(ss_res / max(n - 2, 1))
    s_alpha = float(math.sqrt(s2 * (1.0 / n + (x_mean ** 2) / sxx))) if n > 2 else float("nan")
    t_alpha = float(alpha_daily / s_alpha) if (np.isfinite(s_alpha) and s_alpha > 0) else float("nan")

    return {
        "n": float(n),
        "alpha_ann": float(alpha_daily * ann_factor),
        "beta": float(beta),
        "r2": float(r2),
        "resid_vol_ann": float(resid_vol_ann),
        "t_alpha": float(t_alpha),
    }


def _info_ratio(
    port_rets: pd.Series,
    bench_rets: pd.Series,
    *,
    ann_factor: int = 252,
    min_n: int = 40,
) -> dict[str, float]:
    """
    Active return / tracking error.
      active_return_ann = E[Rp - Rb] * 252
      tracking_error_ann = std(Rp - Rb) * sqrt(252)
      info_ratio = active_return_ann / tracking_error_ann
    """
    p, b = _align_returns(port_rets, bench_rets)
    n = len(p)
    if n < min_n:
        return {
            "n": float(n),
            "active_return_ann": float("nan"),
            "tracking_error_ann": float("nan"),
            "info_ratio": float("nan"),
        }

    a = (p - b).to_numpy(dtype="float64")
    mu = float(np.mean(a))
    te = float(np.std(a, ddof=1))

    active_ann = float(mu * ann_factor)
    te_ann = float(te * math.sqrt(ann_factor)) if (np.isfinite(te) and te > 0) else float("nan")
    ir = float(active_ann / te_ann) if (np.isfinite(te_ann) and te_ann > 0) else float("nan")

    return {
        "n": float(n),
        "active_return_ann": float(active_ann),
        "tracking_error_ann": float(te_ann),
        "info_ratio": float(ir),
    }


def _regime_alpha_report(
    port_rets: pd.Series,
    bench_rets: pd.Series,
    regime_labels: pd.Series,
    *,
    ann_factor: int = 252,
    min_n: int = 40,
) -> dict[str, Any]:
    """
    Regime-conditioned: for each label, compute
      - excess_return_ann (mean Rp-Rb)
      - tracking_error_ann, info_ratio
      - CAPM alpha/beta/r2 (if enough samples)
    """
    p, b = _align_returns(port_rets, bench_rets)
    if p.empty:
        return {"ok": False, "reason": "no_returns"}

    lab = _to_series(regime_labels, "regime")
    df = pd.concat([p.rename("p"), b.rename("b"), lab.rename("lab")], axis=1).dropna()
    if df.empty:
        return {"ok": False, "reason": "no_alignment"}

    out: dict[str, Any] = {"ok": True, "by_label": {}}

    for lbl, g in df.groupby("lab"):
        g = g.dropna()
        if len(g) < max(10, min_n // 4):
            continue

        a = (g["p"] - g["b"]).to_numpy(dtype="float64")
        excess_ann = float(np.mean(a) * ann_factor)
        te_ann = float(np.std(a, ddof=1) * math.sqrt(ann_factor)) if len(a) > 1 else float("nan")
        ir = float(excess_ann / te_ann) if (np.isfinite(te_ann) and te_ann > 0) else float("nan")

        capm = _capm_alpha_beta(g["p"], g["b"], rf_daily=0.0, ann_factor=ann_factor, min_n=min_n) if len(g) >= min_n else None

        out["by_label"][str(lbl)] = {
            "n": int(len(g)),
            "excess_return_ann": excess_ann,
            "tracking_error_ann": te_ann,
            "info_ratio": ir,
            "capm": capm,
        }

    return out


def compute_alpha_report(
    *,
    port_rets: pd.Series | None,
    bench_rets: pd.Series | None,
    regime_labels: pd.Series | None = None,
    ann_factor: int = 252,
) -> dict[str, Any]:
    """
    Rolling (1y/3m) + optional regime conditioning.

    IMPORTANT naming:
      - excess_return_ann = annualized mean(Rp - Rb)
      - capm.alpha_ann    = CAPM alpha (intercept), annualized  <-- THIS is “alpha”
    """
    p = _to_series(port_rets, "port")
    b = _to_series(bench_rets, "bench")

    if p.empty or b.empty:
        return {"ok": False, "reason": "missing_returns"}

    p, b = _align_returns(p, b)
    if p.empty:
        return {"ok": False, "reason": "no_overlap"}

    p_1y, b_1y = _slice_window(p, 252), _slice_window(b, 252)
    p_3m, b_3m = _slice_window(p, 63), _slice_window(b, 63)

    capm_1y = _capm_alpha_beta(p_1y, b_1y, rf_daily=0.0, ann_factor=ann_factor, min_n=40)
    capm_3m = _capm_alpha_beta(p_3m, b_3m, rf_daily=0.0, ann_factor=ann_factor, min_n=30)

    info_1y = _info_ratio(p_1y, b_1y, ann_factor=ann_factor, min_n=40)
    info_3m = _info_ratio(p_3m, b_3m, ann_factor=ann_factor, min_n=30)

    excess_1y = float((p_1y.mean() - b_1y.mean()) * ann_factor) if len(p_1y) > 0 else float("nan")
    excess_3m = float((p_3m.mean() - b_3m.mean()) * ann_factor) if len(p_3m) > 0 else float("nan")

    out: dict[str, Any] = {
        "ok": True,
        "n_total": int(len(p)),
        "windows": {
            "1y": {"n": int(len(p_1y)), "excess_return_ann": float(excess_1y), "capm": capm_1y, "info": info_1y},
            "3m": {"n": int(len(p_3m)), "excess_return_ann": float(excess_3m), "capm": capm_3m, "info": info_3m},
        },
    }

    if regime_labels is not None:
        out["regimes"] = _regime_alpha_report(p, b, _to_series(regime_labels, "regime"), ann_factor=ann_factor, min_n=40)

    return out


def format_alpha_report(alpha_report: dict[str, Any]) -> str:
    """
    Short, human-readable alpha summary.
    Uses “alpha” ONLY for CAPM alpha.
    """
    if not alpha_report or not alpha_report.get("ok"):
        return f"[alpha] unavailable: {alpha_report.get('reason', 'unknown')}"

    w1 = (alpha_report.get("windows") or {}).get("1y") or {}
    w3 = (alpha_report.get("windows") or {}).get("3m") or {}

    def f(x: Any) -> str:
        try:
            x = float(x)
            return "nan" if not np.isfinite(x) else f"{x:.4f}"
        except Exception:
            return "nan"

    capm1 = w1.get("capm", {}) or {}
    capm3 = w3.get("capm", {}) or {}
    info1 = w1.get("info", {}) or {}
    info3 = w3.get("info", {}) or {}

    lines = []
    lines.append(
        "[alpha] rolling 1y: "
        f"capm_alpha={f(capm1.get('alpha_ann'))} "
        f"beta={f(capm1.get('beta'))} "
        f"r2={f(capm1.get('r2'))} "
        f"IR={f(info1.get('info_ratio'))} "
        f"TE={f(info1.get('tracking_error_ann'))} "
        f"excess={f(w1.get('excess_return_ann'))}"
    )
    lines.append(
        "[alpha] rolling 3m: "
        f"capm_alpha={f(capm3.get('alpha_ann'))} "
        f"beta={f(capm3.get('beta'))} "
        f"r2={f(capm3.get('r2'))} "
        f"IR={f(info3.get('info_ratio'))} "
        f"TE={f(info3.get('tracking_error_ann'))} "
        f"excess={f(w3.get('excess_return_ann'))}"
    )

    regimes = alpha_report.get("regimes")
    if regimes and regimes.get("ok") and regimes.get("by_label"):
        lines.append("[alpha][regimes] conditional (excess / capm_alpha if enough samples):")
        for lbl, d in regimes["by_label"].items():
            capm = d.get("capm") or {}
            lines.append(
                f"  - {lbl}: n={int(d.get('n', 0))} "
                f"excess={f(d.get('excess_return_ann'))} "
                f"IR={f(d.get('info_ratio'))} "
                f"capm_alpha={f(capm.get('alpha_ann'))} "
                f"beta={f(capm.get('beta'))}"
            )

    return "\n".join(lines)
