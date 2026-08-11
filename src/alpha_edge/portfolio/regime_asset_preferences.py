from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


REGIME_LABELS_4 = [
    "CALM_BULL",
    "CHOPPY_BULL",
    "CHOPPY_BEAR",
    "STRESS_BEAR",
    "MIXED",
]


@dataclass(frozen=True)
class RegimeAssetPreferenceConfig:
    min_obs: int = 60
    annualization_days: int = 252
    min_abs_weight: float = 1e-8

    # Cross-sectional score weights.
    return_weight: float = 1.0
    vol_penalty: float = 0.50
    downside_penalty: float = 0.50
    drawdown_penalty: float = 0.25

    # Bucket thresholds by cross-sectional quantile.
    strong_quantile: float = 0.70
    weak_quantile: float = 0.30


@dataclass(frozen=True)
class RegimeAssetPreference:
    asset_id: str
    regime: str
    obs: int
    ann_return: float | None
    ann_vol: float | None
    downside_vol: float | None
    max_drawdown: float | None
    sharpe_like: float | None
    preference_score: float | None
    rank: int | None = None
    bucket: str = "UNKNOWN"
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioRegimeFit:
    regime: str
    weighted_preference_score: float | None
    strong_asset_weight: float
    weak_asset_weight: float
    unknown_asset_weight: float
    neutral_asset_weight: float
    asset_count: int
    covered_asset_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _clean_regime_label(x: Any) -> str:
    s = str(x or "").strip().upper()
    if not s or s in {"NONE", "NAN", "NULL"}:
        return "MIXED"
    return s


def _to_day_index(idx: Any) -> pd.DatetimeIndex:
    out = pd.to_datetime(idx, errors="coerce")
    out = pd.DatetimeIndex(out)
    if out.tz is not None:
        out = out.tz_convert(None)
    return out.normalize()


def _max_drawdown(returns: pd.Series) -> float | None:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return None

    wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1.0

    if dd.empty:
        return None

    return float(dd.min())


def _safe_ann_return(returns: pd.Series, annualization_days: int) -> float | None:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return None
    return float(float(r.mean()) * float(annualization_days))


def _safe_ann_vol(returns: pd.Series, annualization_days: int) -> float | None:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.shape[0] < 2:
        return None
    return float(float(r.std(ddof=1)) * np.sqrt(float(annualization_days)))


def _safe_downside_vol(returns: pd.Series, annualization_days: int) -> float | None:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    downside = r[r < 0.0]
    if downside.shape[0] < 2:
        return 0.0
    return float(float(downside.std(ddof=1)) * np.sqrt(float(annualization_days)))


def _standardize_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = float(x.mean(skipna=True))
    sigma = float(x.std(skipna=True, ddof=0))

    if not np.isfinite(sigma) or sigma <= 1e-12:
        return pd.Series(0.0, index=s.index)

    return (x - mu) / sigma


def normalize_regime_history(regime_history: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize regime history into a two-column frame: date, regime.

    Supported inputs:
      - date + regime
      - date + label_or_mixed
      - date + label
      - date + label_commit
      - as_of + regime/label_or_mixed/label/label_commit

    This function assumes the provided regime history is already point-in-time
    safe. It does not fetch or infer future regimes.
    """
    if regime_history is None or regime_history.empty:
        return pd.DataFrame(columns=["date", "regime"])

    df = regime_history.copy()

    if "date" not in df.columns:
        if "as_of" in df.columns:
            df["date"] = df["as_of"]
        else:
            raise RuntimeError("regime_history must contain either 'date' or 'as_of'.")

    if "regime" not in df.columns:
        if "label_or_mixed" in df.columns:
            df["regime"] = df["label_or_mixed"]
        elif "label" in df.columns:
            df["regime"] = df["label"]
        elif "label_commit" in df.columns:
            df["regime"] = df["label_commit"]
        else:
            raise RuntimeError(
                "regime_history must contain one of: regime, label_or_mixed, label, label_commit."
            )

    out = df[["date", "regime"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["date"] = out["date"].dt.tz_localize(None).dt.normalize()
    out["regime"] = out["regime"].map(_clean_regime_label)

    out = (
        out.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return out


def align_returns_to_regime(
    *,
    returns_wide: pd.DataFrame,
    regime_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join wide asset returns with one regime label per date.

    returns_wide:
        index = dates
        columns = asset_ids

    regime_history:
        normalized to date + regime
    """
    if returns_wide is None or returns_wide.empty:
        raise RuntimeError("returns_wide is empty.")

    regimes = normalize_regime_history(regime_history)
    if regimes.empty:
        raise RuntimeError("regime_history is empty after normalization.")

    rets = returns_wide.copy()
    rets.index = _to_day_index(rets.index)
    rets = rets.sort_index()
    rets = rets.apply(pd.to_numeric, errors="coerce")

    rets_df = rets.reset_index()
    date_col = rets_df.columns[0]
    rets_df = rets_df.rename(columns={date_col: "date"})
    rets_df["date"] = pd.to_datetime(rets_df["date"], errors="coerce").dt.normalize()

    merged = rets_df.merge(regimes, on="date", how="inner")
    merged = merged.dropna(subset=["date", "regime"]).sort_values("date")

    if merged.empty:
        raise RuntimeError("No overlapping dates between returns_wide and regime_history.")

    return merged.reset_index(drop=True)


def compute_asset_regime_preferences(
    *,
    returns_wide: pd.DataFrame,
    regime_history: pd.DataFrame,
    regime: str,
    cfg: RegimeAssetPreferenceConfig | None = None,
) -> dict[str, RegimeAssetPreference]:
    """
    Compute asset preference scores conditioned on a specific market regime.

    This is point-in-time safe if the supplied returns_wide and regime_history
    are point-in-time safe. The function performs no I/O and does not refit the
    market HMM.
    """
    cfg = cfg or RegimeAssetPreferenceConfig()
    regime_clean = _clean_regime_label(regime)

    merged = align_returns_to_regime(
        returns_wide=returns_wide,
        regime_history=regime_history,
    )

    regime_rows = merged[merged["regime"] == regime_clean].copy()
    if regime_rows.empty:
        return {}

    asset_cols = [c for c in regime_rows.columns if c not in {"date", "regime"}]
    rows: list[dict[str, Any]] = []

    for asset_id in asset_cols:
        s = pd.to_numeric(regime_rows[asset_id], errors="coerce").dropna()
        obs = int(s.shape[0])

        if obs < int(cfg.min_obs):
            rows.append(
                {
                    "asset_id": str(asset_id),
                    "regime": regime_clean,
                    "obs": obs,
                    "ann_return": None,
                    "ann_vol": None,
                    "downside_vol": None,
                    "max_drawdown": None,
                    "sharpe_like": None,
                    "raw_score": None,
                    "preference_score": None,
                    "rank": None,
                    "bucket": "UNKNOWN",
                }
            )
            continue

        ann_return = _safe_ann_return(s, cfg.annualization_days)
        ann_vol = _safe_ann_vol(s, cfg.annualization_days)
        downside_vol = _safe_downside_vol(s, cfg.annualization_days)
        max_dd = _max_drawdown(s)

        sharpe_like = None
        if ann_vol is not None and ann_vol > 1e-12 and ann_return is not None:
            sharpe_like = float(ann_return / ann_vol)

        rows.append(
            {
                "asset_id": str(asset_id),
                "regime": regime_clean,
                "obs": obs,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "downside_vol": downside_vol,
                "max_drawdown": max_dd,
                "sharpe_like": sharpe_like,
                "raw_score": None,
                "preference_score": None,
                "rank": None,
                "bucket": "UNKNOWN",
            }
        )

    score_df = pd.DataFrame(rows)

    valid = score_df[
        score_df["ann_return"].notna()
        & score_df["ann_vol"].notna()
        & score_df["downside_vol"].notna()
        & score_df["max_drawdown"].notna()
    ].copy()

    if not valid.empty:
        z_return = _standardize_series(valid["ann_return"])
        z_vol = _standardize_series(valid["ann_vol"])
        z_downside = _standardize_series(valid["downside_vol"])
        z_drawdown = _standardize_series(valid["max_drawdown"].abs())

        valid["raw_score"] = (
            float(cfg.return_weight) * z_return
            - float(cfg.vol_penalty) * z_vol
            - float(cfg.downside_penalty) * z_downside
            - float(cfg.drawdown_penalty) * z_drawdown
        )

        min_score = float(valid["raw_score"].min())
        max_score = float(valid["raw_score"].max())

        if np.isfinite(max_score - min_score) and (max_score - min_score) > 1e-12:
            valid["preference_score"] = (
                (valid["raw_score"] - min_score) / (max_score - min_score)
            )
        else:
            valid["preference_score"] = 0.5

        valid = valid.sort_values(
            ["preference_score", "ann_return", "asset_id"],
            ascending=[False, False, True],
            kind="stable",
        ).copy()
        valid["rank"] = range(1, valid.shape[0] + 1)

        strong_cut = float(valid["preference_score"].quantile(float(cfg.strong_quantile)))
        weak_cut = float(valid["preference_score"].quantile(float(cfg.weak_quantile)))

        valid["bucket"] = "NEUTRAL"
        valid.loc[valid["preference_score"] >= strong_cut, "bucket"] = "STRONG"
        valid.loc[valid["preference_score"] <= weak_cut, "bucket"] = "WEAK"

        score_df = score_df.drop(
            columns=["raw_score", "preference_score", "rank", "bucket"],
            errors="ignore",
        )
        score_df = score_df.merge(
            valid[["asset_id", "raw_score", "preference_score", "rank", "bucket"]],
            on="asset_id",
            how="left",
        )

    out: dict[str, RegimeAssetPreference] = {}

    for _, row in score_df.iterrows():
        asset_id = str(row["asset_id"])

        def val(name: str) -> Any:
            return row.get(name)

        out[asset_id] = RegimeAssetPreference(
            asset_id=asset_id,
            regime=str(row["regime"]),
            obs=int(row["obs"]),
            ann_return=None if pd.isna(val("ann_return")) else float(val("ann_return")),
            ann_vol=None if pd.isna(val("ann_vol")) else float(val("ann_vol")),
            downside_vol=None if pd.isna(val("downside_vol")) else float(val("downside_vol")),
            max_drawdown=None if pd.isna(val("max_drawdown")) else float(val("max_drawdown")),
            sharpe_like=None if pd.isna(val("sharpe_like")) else float(val("sharpe_like")),
            preference_score=(
                None if pd.isna(val("preference_score")) else float(val("preference_score"))
            ),
            rank=None if pd.isna(val("rank")) else int(val("rank")),
            bucket="UNKNOWN" if pd.isna(val("bucket")) else str(val("bucket")),
            diagnostics={
                "min_obs": int(cfg.min_obs),
                "annualization_days": int(cfg.annualization_days),
            },
        )

    return out


def assess_portfolio_regime_fit(
    *,
    weights: dict[str, float],
    preferences: dict[str, RegimeAssetPreference],
    regime: str,
    cfg: RegimeAssetPreferenceConfig | None = None,
) -> PortfolioRegimeFit:
    """
    Score portfolio exposure to assets preferred in the current regime.

    Uses absolute weights so it works for long/short portfolios. In the current
    spot-only phase, this behaves like ordinary positive weight exposure.
    """
    cfg = cfg or RegimeAssetPreferenceConfig()
    regime_clean = _clean_regime_label(regime)

    cleaned_weights: dict[str, float] = {}

    for asset_id, raw_weight in (weights or {}).items():
        try:
            weight = float(raw_weight)
        except Exception:
            continue

        if not np.isfinite(weight) or abs(weight) <= float(cfg.min_abs_weight):
            continue

        cleaned_weights[str(asset_id)] = weight

    if not cleaned_weights:
        return PortfolioRegimeFit(
            regime=regime_clean,
            weighted_preference_score=None,
            strong_asset_weight=0.0,
            weak_asset_weight=0.0,
            unknown_asset_weight=0.0,
            neutral_asset_weight=0.0,
            asset_count=0,
            covered_asset_count=0,
            diagnostics={"reason": "empty_weights"},
        )

    gross = float(sum(abs(w) for w in cleaned_weights.values()))
    if gross <= 1e-12:
        return PortfolioRegimeFit(
            regime=regime_clean,
            weighted_preference_score=None,
            strong_asset_weight=0.0,
            weak_asset_weight=0.0,
            unknown_asset_weight=0.0,
            neutral_asset_weight=0.0,
            asset_count=len(cleaned_weights),
            covered_asset_count=0,
            diagnostics={"reason": "zero_gross_weight"},
        )

    weighted_score_num = 0.0
    weighted_score_den = 0.0

    strong_weight = 0.0
    weak_weight = 0.0
    unknown_weight = 0.0
    neutral_weight = 0.0
    asset_rows: list[dict[str, Any]] = []

    for asset_id, raw_w in cleaned_weights.items():
        abs_w = abs(float(raw_w)) / gross
        pref = preferences.get(str(asset_id))

        if pref is None:
            unknown_weight += abs_w
            bucket = "UNKNOWN"
            score = None
            rank = None
            obs = None
        else:
            bucket = str(pref.bucket or "UNKNOWN")
            score = pref.preference_score
            rank = pref.rank
            obs = pref.obs

            if score is None:
                unknown_weight += abs_w
            else:
                weighted_score_num += abs_w * float(score)
                weighted_score_den += abs_w

            if bucket == "STRONG":
                strong_weight += abs_w
            elif bucket == "WEAK":
                weak_weight += abs_w
            elif bucket == "NEUTRAL":
                neutral_weight += abs_w
            else:
                unknown_weight += abs_w

        asset_rows.append(
            {
                "asset_id": asset_id,
                "weight": float(raw_w),
                "abs_weight_normalized": float(abs_w),
                "bucket": bucket,
                "preference_score": score,
                "rank": rank,
                "obs": obs,
            }
        )

    weighted_preference_score = (
        None if weighted_score_den <= 1e-12 else float(weighted_score_num / weighted_score_den)
    )

    covered_asset_count = int(
        sum(
            1
            for asset_id in cleaned_weights
            if asset_id in preferences and preferences[asset_id].preference_score is not None
        )
    )

    asset_rows = sorted(
        asset_rows,
        key=lambda x: abs(float(x["abs_weight_normalized"])),
        reverse=True,
    )

    return PortfolioRegimeFit(
        regime=regime_clean,
        weighted_preference_score=weighted_preference_score,
        strong_asset_weight=float(strong_weight),
        weak_asset_weight=float(weak_weight),
        unknown_asset_weight=float(unknown_weight),
        neutral_asset_weight=float(neutral_weight),
        asset_count=int(len(cleaned_weights)),
        covered_asset_count=covered_asset_count,
        diagnostics={
            "assets": asset_rows,
            "gross_weight": gross,
            "coverage_ratio": (
                float(covered_asset_count / len(cleaned_weights)) if cleaned_weights else 0.0
            ),
        },
    )


def regime_fit_advantage(
    *,
    current_fit: PortfolioRegimeFit,
    candidate_fit: PortfolioRegimeFit,
) -> dict[str, Any]:
    cur_score = current_fit.weighted_preference_score
    cand_score = candidate_fit.weighted_preference_score

    if cur_score is None or cand_score is None:
        score_advantage = None
    else:
        score_advantage = float(cand_score - cur_score)

    return {
        "current": asdict(current_fit),
        "candidate": asdict(candidate_fit),
        "preference_score_advantage": score_advantage,
        "strong_asset_weight_advantage": float(
            candidate_fit.strong_asset_weight - current_fit.strong_asset_weight
        ),
        "weak_asset_weight_reduction": float(
            current_fit.weak_asset_weight - candidate_fit.weak_asset_weight
        ),
        "unknown_asset_weight_change": float(
            candidate_fit.unknown_asset_weight - current_fit.unknown_asset_weight
        ),
    }


def build_portfolio_regime_fit_comparison(
    *,
    returns_wide: pd.DataFrame,
    regime_history: pd.DataFrame,
    regime: str,
    current_weights: dict[str, float],
    candidate_weights: dict[str, float],
    candidate_name: str = "candidate",
    cfg: RegimeAssetPreferenceConfig | None = None,
) -> dict[str, Any]:
    """
    Build diagnostic-only regime-fit comparison for transition workflows.

    This is intentionally pure. Runners are responsible for loading the PIT-safe
    regime_history and returns_wide slices before calling this function.
    """
    cfg = cfg or RegimeAssetPreferenceConfig()
    regime_clean = _clean_regime_label(regime)

    preferences = compute_asset_regime_preferences(
        returns_wide=returns_wide,
        regime_history=regime_history,
        regime=regime_clean,
        cfg=cfg,
    )

    if not preferences:
        return {
            "status": "unavailable",
            "reason": f"No asset preferences available for regime={regime_clean}",
            "regime": regime_clean,
            "candidate_name": str(candidate_name),
            "config": asdict(cfg),
        }

    current_fit = assess_portfolio_regime_fit(
        weights=current_weights,
        preferences=preferences,
        regime=regime_clean,
        cfg=cfg,
    )

    candidate_fit = assess_portfolio_regime_fit(
        weights=candidate_weights,
        preferences=preferences,
        regime=regime_clean,
        cfg=cfg,
    )

    comparison = regime_fit_advantage(
        current_fit=current_fit,
        candidate_fit=candidate_fit,
    )

    # Rename the generic candidate block for clarity in persisted diagnostics.
    candidate_block = comparison.pop("candidate")

    top_preferences = preferences_to_frame(preferences).head(20)
    bottom_preferences = preferences_to_frame(preferences).tail(20)

    return {
        "status": "success",
        "regime": regime_clean,
        "candidate_name": str(candidate_name),
        "preference_asset_count": int(len(preferences)),
        "current": comparison.pop("current"),
        str(candidate_name): candidate_block,
        "comparison": comparison,
        "top_preferred_assets": top_preferences[
            ["asset_id", "preference_score", "rank", "bucket", "ann_return", "ann_vol", "obs"]
        ].to_dict(orient="records") if not top_preferences.empty else [],
        "bottom_preferred_assets": bottom_preferences[
            ["asset_id", "preference_score", "rank", "bucket", "ann_return", "ann_vol", "obs"]
        ].to_dict(orient="records") if not bottom_preferences.empty else [],
        "config": asdict(cfg),
    }


def preferences_to_frame(preferences: dict[str, RegimeAssetPreference]) -> pd.DataFrame:
    if not preferences:
        return pd.DataFrame()

    return pd.DataFrame([asdict(v) for v in preferences.values()]).sort_values(
        ["rank", "asset_id"],
        na_position="last",
    ).reset_index(drop=True)
