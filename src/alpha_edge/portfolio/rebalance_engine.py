# rebalance_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


DEFAULT_CRYPTO_DECIMALS = 8


def is_crypto_ticker(ticker: str) -> bool:
    """
    Your universe uses Yahoo-style crypto tickers like BTC-USD, ETH-USD, SUI-USD, etc.
    Adjust if you have other conventions.
    """
    t = str(ticker).upper().strip()
    return t.endswith("-USD") and len(t) > 4


def trunc_toward_zero(x: float) -> int:
    """
    Integer truncation toward zero:  3.9 -> 3,  -3.9 -> -3
    """
    return int(np.trunc(x))


# ----------------------------
# Rebalance state + policy
# ----------------------------

@dataclass(frozen=True)
class RebalanceState:
    """
    Persisted state to support time/equity-band triggers.
    Stored in S3 as a small JSON payload.
    """
    last_rebalance_date: str | None = None  # "YYYY-MM-DD" in MARKET as-of date terms
    last_rebalance_equity: float | None = None


@dataclass(frozen=True)
class RebalanceDecision:
    should_rebalance: bool
    reasons: list[str]
    leverage_real: float
    leverage_target: float
    drift_ratio: float  # L_real / L_target


@dataclass(frozen=True)
class RescalePlan:
    as_of: str                       # market date "YYYY-MM-DD"
    equity: float
    recommended_leverage: float

    target_gross_notional: float
    max_notional_cap: float | None

    # outputs
    targets: pd.DataFrame            # per ticker target quantities / exposures
    used_gross_notional: float
    leftover_notional: float

    # debug
    gross_current: float
    leverage_current: float


def compute_gross_notional_from_positions(
    *,
    positions_qty: Dict[str, float],
    prices_usd: pd.Series,
) -> float:
    px = pd.to_numeric(prices_usd, errors="coerce").replace([np.inf, -np.inf], np.nan)

    gross = 0.0
    for t, q in positions_qty.items():
        tt = str(t).upper().strip()
        if not tt:
            continue
        p = float(px.get(tt, np.nan))
        if not np.isfinite(p) or p <= 0:
            continue
        gross += abs(float(q) * p)
    return float(gross)


def should_rebalance(
    *,
    as_of: str,
    equity: float,
    gross_notional: float,
    recommended_leverage: float,
    state: RebalanceState | None,
    # --- policy knobs ---
    drift_threshold: float = 0.15,         # 15% drift allowed (hard)
    min_days_between: int = 3,             # avoid churn
    time_rule_days: int | None = 30,       # periodic rebalance (e.g. monthly). None disables.
    equity_band: float | None = None,      # e.g. 0.10 for +/-10% equity vs last rebalance. None disables.
) -> RebalanceDecision:
    """
    Decides whether to rebalance, based on:
      - Leverage drift: |L_real/L_target - 1| >= drift_threshold
      - Time rule: days since last rebalance >= time_rule_days
      - Equity band: equity moved +/- equity_band since last rebalance

    Notes:
      - Uses gross notional / equity for realized leverage
      - Enforces a minimum cool-down in days
    """
    eq = float(equity)
    gross = float(gross_notional)
    L_t = float(recommended_leverage)

    if eq <= 0 or L_t <= 0:
        return RebalanceDecision(
            should_rebalance=False,
            reasons=["invalid_equity_or_target_leverage"],
            leverage_real=float("nan"),
            leverage_target=L_t,
            drift_ratio=float("nan"),
        )

    L_real = gross / eq
    drift_ratio = (L_real / L_t) if L_t > 0 else float("inf")
    drift = abs(drift_ratio - 1.0)

    reasons: list[str] = []

    # cooldown / min days between
    last_dt = None
    if state and state.last_rebalance_date:
        last_dt = pd.Timestamp(state.last_rebalance_date)
    now_dt = pd.Timestamp(as_of)

    if last_dt is not None and min_days_between is not None and int(min_days_between) > 0:
        days_since = int((now_dt - last_dt).days)
        if days_since < int(min_days_between):
            # within cooldown: only allow if extreme drift
            extreme = drift >= max(2.0 * float(drift_threshold), 0.30)
            if not extreme:
                return RebalanceDecision(
                    should_rebalance=False,
                    reasons=[f"cooldown_{days_since}d<{int(min_days_between)}d"],
                    leverage_real=float(L_real),
                    leverage_target=float(L_t),
                    drift_ratio=float(drift_ratio),
                )
            reasons.append("cooldown_override_extreme_drift")

    # leverage drift trigger
    if np.isfinite(drift) and drift >= float(drift_threshold):
        reasons.append(f"leverage_drift_{drift:.3f}>={float(drift_threshold):.3f}")

    # time trigger
    if time_rule_days is not None and last_dt is not None:
        days_since = int((now_dt - last_dt).days)
        if days_since >= int(time_rule_days):
            reasons.append(f"time_{days_since}d>={int(time_rule_days)}d")
    elif time_rule_days is not None and last_dt is None:
        # no state yet => allow time rule to trigger first time
        reasons.append("time_no_state")

    # equity band trigger
    if equity_band is not None:
        if state is None or state.last_rebalance_equity is None or not np.isfinite(float(state.last_rebalance_equity)):
            reasons.append("equity_band_no_state")
        else:
            base = float(state.last_rebalance_equity)
            if base > 0:
                rel = (eq / base) - 1.0
                if abs(rel) >= float(equity_band):
                    reasons.append(f"equity_move_{rel:+.3f}>={float(equity_band):.3f}")

    return RebalanceDecision(
        should_rebalance=(len(reasons) > 0),
        reasons=reasons,
        leverage_real=float(L_real),
        leverage_target=float(L_t),
        drift_ratio=float(drift_ratio),
    )


def compute_target_gross_notional(
    *,
    equity: float,
    recommended_leverage: float,
    max_notional_cap: float | None = None,
) -> float:
    """
    Target gross notional to deploy given leverage and optional hard cap.

    New world (no goal ladder):
      target = equity * recommended_leverage
      if max_notional_cap is provided -> min(target, cap)
    """
    eq = float(equity)
    lev = float(recommended_leverage)
    if eq <= 0 or lev <= 0:
        return 0.0

    desired = eq * lev
    if max_notional_cap is not None and float(max_notional_cap) > 0:
        return float(min(desired, float(max_notional_cap)))
    return float(desired)


def rescale_positions_to_target_gross(
    *,
    positions_qty: Dict[str, float],
    prices_usd: pd.Series,
    target_gross_notional: float,
    crypto_decimals: int = DEFAULT_CRYPTO_DECIMALS,
    greedy_fill: bool = True,
) -> tuple[pd.DataFrame, float, float, float]:
    """
    Rescales current positions to a new gross notional while keeping the same signed weights
    (i.e., same long/short composition).

    Returns:
      df_targets, used_gross, leftover_gross, gross_now
    """
    tgt = float(target_gross_notional)
    if tgt <= 0:
        raise ValueError("target_gross_notional must be > 0")

    px = pd.to_numeric(prices_usd, errors="coerce").replace([np.inf, -np.inf], np.nan)

    rows: list[dict[str, Any]] = []
    gross_now = 0.0

    # build current exposures
    for t, q in positions_qty.items():
        tt = str(t).upper().strip()
        if not tt:
            continue
        p = float(px.get(tt, np.nan))
        if not np.isfinite(p) or p <= 0:
            continue
        qf = float(q)
        exp = p * qf  # signed exposure
        gross_now += abs(exp)

        rows.append(
            dict(
                ticker=tt,
                price=p,
                qty_current=qf,
                exp_current=exp,
            )
        )

    if not rows:
        raise ValueError("No valid positions with finite prices to rescale")

    gross_now = float(gross_now)
    if gross_now <= 0:
        raise ValueError("Current gross exposure is <= 0; cannot rescale")

    df = pd.DataFrame(rows).set_index("ticker", drop=False)

    # signed weights based on current exposures, normalized by gross (supports shorts)
    df["w_signed"] = df["exp_current"] / gross_now
    df["w_abs"] = df["exp_current"].abs() / gross_now

    # target exposures (signed) under new gross notional
    df["exp_target"] = df["w_signed"] * tgt
    df["qty_target_raw"] = df["exp_target"] / df["price"]

    # rounding rules:
    # - crypto: allow decimals
    # - non-crypto: integer shares, trunc toward zero so we do NOT exceed gross due to rounding
    qty_target = []
    for _, r in df.iterrows():
        tt = str(r["ticker"])
        q_raw = float(r["qty_target_raw"])
        if is_crypto_ticker(tt):
            q_adj = float(np.round(q_raw, crypto_decimals))
        else:
            q_adj = float(trunc_toward_zero(q_raw))
        qty_target.append(q_adj)

    df["qty_target"] = qty_target
    df["exp_target_rounded"] = df["qty_target"] * df["price"]

    used = float(df["exp_target_rounded"].abs().sum())
    leftover = float(max(0.0, tgt - used))

    # optional greedy fill for integer-share assets to use leftover notional
    if greedy_fill and leftover > 0:
        int_mask = ~df["ticker"].map(is_crypto_ticker)
        order = df.loc[int_mask].sort_values("w_abs", ascending=False).index.tolist()

        if order:
            prices_map = df["price"].to_dict()
            # sign follows desired exposure sign (w_signed)
            sign_map = np.sign(df["exp_target"]).replace(0.0, 1.0).to_dict()

            min_price = float(np.nanmin(df.loc[int_mask, "price"].to_numpy()))
            max_iter = 200000
            it = 0

            while leftover >= min_price and it < max_iter:
                progressed = False
                for t in order:
                    p = float(prices_map[t])
                    if not np.isfinite(p) or p <= 0:
                        continue
                    if leftover < p:
                        continue

                    sgn = float(sign_map.get(t, 1.0))
                    df.at[t, "qty_target"] = float(df.at[t, "qty_target"]) + sgn * 1.0
                    df.at[t, "exp_target_rounded"] = float(df.at[t, "qty_target"]) * p

                    used += p
                    leftover = float(max(0.0, tgt - used))
                    progressed = True

                    if leftover < min_price:
                        break

                if not progressed:
                    break
                it += 1

    # deltas
    df["delta_qty"] = df["qty_target"] - df["qty_current"]
    df["delta_exp"] = df["exp_target_rounded"] - df["exp_current"]

    used_final = float(df["exp_target_rounded"].abs().sum())
    leftover_final = float(max(0.0, tgt - used_final))

    return df.reset_index(drop=True), used_final, leftover_final, gross_now


def build_rescale_plan(
    *,
    as_of: str,
    equity: float,
    recommended_leverage: float,
    positions_qty: Dict[str, float],
    prices_usd: pd.Series,
    max_notional_cap: float | None = None,
    crypto_decimals: int = DEFAULT_CRYPTO_DECIMALS,
) -> RescalePlan:
    """
    Creates a rescale plan that:
      - targets gross exposure = equity * recommended_leverage (optionally capped)
      - keeps current signed exposure mix (long/short) via signed weights on gross

    IMPORTANT:
      - This does NOT decide whether to rebalance; call should_rebalance() first.
    """
    tgt = compute_target_gross_notional(
        equity=float(equity),
        recommended_leverage=float(recommended_leverage),
        max_notional_cap=max_notional_cap,
    )
    if tgt <= 0:
        raise ValueError("Target gross notional computed as <= 0 (equity/leverage/cap)")

    df_targets, used, leftover, gross_now = rescale_positions_to_target_gross(
        positions_qty=positions_qty,
        prices_usd=prices_usd,
        target_gross_notional=tgt,
        crypto_decimals=crypto_decimals,
        greedy_fill=True,
    )

    lev_now = (gross_now / float(equity)) if float(equity) > 0 else float("inf")

    return RescalePlan(
        as_of=str(as_of),
        equity=float(equity),
        recommended_leverage=float(recommended_leverage),
        target_gross_notional=float(tgt),
        max_notional_cap=(None if max_notional_cap is None else float(max_notional_cap)),
        targets=df_targets,
        used_gross_notional=float(used),
        leftover_notional=float(leftover),
        gross_current=float(gross_now),
        leverage_current=float(lev_now),
    )
