from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run

import argparse
import datetime as dt
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from alpha_edge.core.data_loader import (
    s3_init,
    s3_load_latest_json,
    s3_write_json_event,
)
from alpha_edge.core.runtime import (
    RuntimeConfig,
    load_runtime_config,
    require_prod_confirmation,
)
from alpha_edge.core.schemas import (
    CurrentPortfolioState,
    PortfolioTransitionConfig,
)
from alpha_edge.portfolio.transition_engine import assess_transition


DEFAULT_ENGINE_BUCKET = "alpha-edge-algo"
DEFAULT_ENGINE_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT_PREFIX = "engine/v1"

TRANSITION_ASSESSMENT_TABLE = "portfolio_transition/assessment"


def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_ENGINE_BUCKET)).strip()


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_ENGINE_REGION)).strip()


def cfg_engine_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "engine_root", DEFAULT_ENGINE_ROOT_PREFIX)).strip("/")


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _safe_str(x: Any) -> str | None:
    if x is None:
        return None
    text = str(x).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _parse_date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value).tz_localize(None).normalize()
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


def _extract_health_score(raw_health: dict[str, Any]) -> float | None:
    """
    Supports both:
      - health/latest.json from daily report
      - final executable health payloads if reused later
    """
    if not isinstance(raw_health, dict):
        return None

    for key in ["health_score", "score"]:
        v = _safe_float(raw_health.get(key))
        if v is not None:
            return v

    health = raw_health.get("health")
    if isinstance(health, dict):
        return _safe_float(health.get("health_score"))

    return None


def _health_grade_from_score(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    if score >= 50:
        return "D"
    return "F"


def _extract_current_regime(raw_hmm: dict[str, Any] | None, raw_rescale_state: dict[str, Any] | None) -> tuple[str | None, float | None]:
    """
    First use market_rescale_state.label if available because daily_report writes it
    as a compact regime state. Fall back to regimes/hmm payload.
    """
    raw_rescale_state = raw_rescale_state or {}
    raw_hmm = raw_hmm or {}

    label = _safe_str(raw_rescale_state.get("label"))
    if label:
        return label, None

    lev_rec = raw_hmm.get("leverage_recommendation")
    if isinstance(lev_rec, dict):
        label = (
            _safe_str(lev_rec.get("chosen_label"))
            or _safe_str(lev_rec.get("label"))
            or _safe_str(lev_rec.get("regime"))
        )
        confidence = _safe_float(lev_rec.get("confidence"))
        if label:
            return label, confidence

    hmm = raw_hmm.get("hmm")
    if isinstance(hmm, dict):
        label = (
            _safe_str(hmm.get("chosen_label"))
            or _safe_str(hmm.get("label"))
            or _safe_str(hmm.get("regime"))
        )
        confidence = _safe_float(hmm.get("confidence"))
        if label:
            return label, confidence

    return None, None


def _load_previous_transition_regime(
    *,
    s3,
    bucket: str,
    root_prefix: str,
) -> str | None:
    latest = s3_load_latest_json(
        s3,
        bucket=bucket,
        root_prefix=root_prefix,
        table=TRANSITION_ASSESSMENT_TABLE,
    )
    if not isinstance(latest, dict):
        return None

    diagnostics = latest.get("diagnostics") or {}
    if not isinstance(diagnostics, dict):
        return None

    regime = diagnostics.get("regime") or {}
    if not isinstance(regime, dict):
        return None

    return _safe_str(regime.get("current_regime"))


def _days_since_latest_portfolio_search(
    *,
    s3,
    bucket: str,
    root_prefix: str,
    as_of_ts: pd.Timestamp,
    lookback_days: int = 120,
) -> int | None:
    """
    Lightweight discovery of latest portfolio_search/runs dt partition.

    This does not need to parse every run. It only checks whether any run JSON
    exists in recent dated partitions.
    """
    for offset in range(0, int(lookback_days) + 1):
        d = as_of_ts - pd.Timedelta(days=offset)
        dt_str = d.strftime("%Y-%m-%d")
        prefix = f"{root_prefix.strip('/')}/portfolio_search/runs/dt={dt_str}/"

        resp = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=1,
        )

        if resp.get("Contents"):
            return int(offset)

    return None


def build_current_portfolio_state(
    *,
    s3,
    bucket: str,
    root_prefix: str,
    as_of: str,
) -> CurrentPortfolioState:
    as_of_ts = pd.Timestamp(as_of).tz_localize(None).normalize()
    as_of_date = as_of_ts.strftime("%Y-%m-%d")

    raw_health = (
        s3_load_latest_json(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="health",
        )
        or {}
    )

    raw_hmm = (
        s3_load_latest_json(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="regimes/hmm",
        )
        or {}
    )

    raw_rescale_state = (
        s3_load_latest_json(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table="regimes/market_rescale_state",
        )
        or {}
    )

    current_regime, regime_confidence = _extract_current_regime(
        raw_hmm=raw_hmm,
        raw_rescale_state=raw_rescale_state,
    )

    previous_regime = _load_previous_transition_regime(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
    )

    regime_changed = (
        previous_regime is not None
        and current_regime is not None
        and str(previous_regime) != str(current_regime)
    )

    health_score = _extract_health_score(raw_health)
    grade = _safe_str(raw_health.get("health_grade")) or _health_grade_from_score(health_score)

    days_since_full_search = _days_since_latest_portfolio_search(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
        as_of_ts=as_of_ts,
    )

    # Daily health schema currently exposes score, p_hit_main_goal, ruin_prob,
    # ann_return, sharpe, alpha diagnostics. Some risk fields are not persisted
    # in health/latest.json yet, so they remain None for now.
    return CurrentPortfolioState(
        as_of=as_of_date,
        portfolio_id=None,
        health_score=health_score,
        grade=grade,
        optimizer_score=_safe_float(raw_health.get("score")),
        ruin_probability=_safe_float(raw_health.get("ruin_prob")),
        max_drawdown=None,
        avg_max_drawdown=None,
        cdar_95=None,
        volatility=None,
        annual_return=_safe_float(raw_health.get("ann_return")),
        hhi=None,
        correlation=None,
        regime=current_regime,
        previous_regime=previous_regime,
        regime_changed=bool(regime_changed),
        regime_confidence=regime_confidence,
        original_health_score=None,
        days_since_full_search=days_since_full_search,
        local_optimizer_failed_days=0,
        metadata={
            "health_source": f"{root_prefix}/health/latest.json",
            "regime_source": f"{root_prefix}/regimes/hmm/latest.json",
            "market_rescale_source": f"{root_prefix}/regimes/market_rescale_state/latest.json",
            "raw_health_date": str(raw_health.get("date") or raw_health.get("as_of") or ""),
            "raw_hmm_as_of": str(raw_hmm.get("as_of") or ""),
            "raw_rescale_as_of": str(raw_rescale_state.get("as_of") or ""),
        },
    )


def run_transition_assessment(
    *,
    cfg: RuntimeConfig,
    as_of: str | None = None,
    write_outputs: bool = True,
    update_latest: bool = True,
    confirm_prod_write: bool = False,
    min_health_score: float = 60.0,
    health_drop_trigger: float = 15.0,
    min_grade: str = "B",
    max_ruin_probability: float = 0.10,
    max_drawdown_limit: float = 0.30,
    full_search_refresh_days: int = 20,
    shadow_confirmation_days: int = 3,
    min_shadow_health_advantage: float = 5.0,
    max_daily_turnover: float = 0.10,
    min_local_improvement: float = 3.0,
    regime_change_requires_full_search: bool = True,
    min_regime_confidence_for_full_search: float = 0.60,
) -> dict[str, Any]:
    if write_outputs:
        require_prod_confirmation(cfg, bool(confirm_prod_write))

    bucket = cfg_bucket(cfg)
    region = cfg_region(cfg)
    root_prefix = cfg_engine_root(cfg)

    as_of_ts = pd.Timestamp(as_of or dt.date.today()).tz_localize(None).normalize()
    as_of_date = as_of_ts.strftime("%Y-%m-%d")

    s3 = s3_init(region)

    state = build_current_portfolio_state(
        s3=s3,
        bucket=bucket,
        root_prefix=root_prefix,
        as_of=as_of_date,
    )

    transition_cfg = PortfolioTransitionConfig(
        min_health_score=float(min_health_score),
        health_drop_trigger=float(health_drop_trigger),
        min_grade=str(min_grade),
        max_ruin_probability=float(max_ruin_probability),
        max_drawdown_limit=float(max_drawdown_limit),
        full_search_refresh_days=int(full_search_refresh_days),
        shadow_confirmation_days=int(shadow_confirmation_days),
        min_shadow_health_advantage=float(min_shadow_health_advantage),
        max_daily_turnover=float(max_daily_turnover),
        min_local_improvement=float(min_local_improvement),
        regime_change_requires_full_search=bool(regime_change_requires_full_search),
        min_regime_confidence_for_full_search=float(min_regime_confidence_for_full_search),
    )

    assessment = assess_transition(
        state=state,
        cfg=transition_cfg,
    )

    payload = {
        "schema_version": "portfolio_transition_assessment_v1",
        "as_of": as_of_date,
        "recommendation": assessment.recommendation,
        "reason": assessment.reason,
        "full_search_required": bool(assessment.full_search_required),
        "local_optimization_allowed": bool(assessment.local_optimization_allowed),
        "shadow_portfolio_required": bool(assessment.shadow_portfolio_required),
        "delta_execution_allowed": bool(assessment.delta_execution_allowed),
        "current_state": asdict(assessment.current_state),
        "config": asdict(transition_cfg),
        "diagnostics": dict(assessment.diagnostics or {}),
    }

    print("\n=== PORTFOLIO TRANSITION ASSESSMENT ===")
    print(f"env:                         {getattr(cfg, 'env', 'unknown')}")
    print(f"bucket:                      {bucket}")
    print(f"root_prefix:                 {root_prefix}")
    print(f"as_of:                       {as_of_date}")
    print(f"recommendation:              {payload['recommendation']}")
    print(f"full_search_required:        {payload['full_search_required']}")
    print(f"local_optimization_allowed:  {payload['local_optimization_allowed']}")
    print(f"shadow_portfolio_required:   {payload['shadow_portfolio_required']}")
    print(f"reason:                      {payload['reason']}")
    print(f"triggers:                    {payload['diagnostics'].get('triggers')}")

    if write_outputs:
        s3_write_json_event(
            s3,
            bucket=bucket,
            root_prefix=root_prefix,
            table=TRANSITION_ASSESSMENT_TABLE,
            dt=as_of_ts,
            filename="transition_assessment.json",
            payload=payload,
            update_latest=update_latest,
        )

        print(
            f"\n[S3] Saved transition assessment to "
            f"s3://{bucket}/{root_prefix}/{TRANSITION_ASSESSMENT_TABLE}/dt={as_of_date}/"
        )

    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run portfolio transition assessment for the Alpha Edge morning routine."
    )

    p.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    p.add_argument("--as-of", default=None, help="Assessment date YYYY-MM-DD. Default: today.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-prod-write", action="store_true")

    p.add_argument("--min-health-score", type=float, default=60.0)
    p.add_argument("--health-drop-trigger", type=float, default=15.0)
    p.add_argument("--min-grade", default="B")
    p.add_argument("--max-ruin-probability", type=float, default=0.10)
    p.add_argument("--max-drawdown-limit", type=float, default=0.30)
    p.add_argument("--full-search-refresh-days", type=int, default=20)
    p.add_argument("--shadow-confirmation-days", type=int, default=3)
    p.add_argument("--min-shadow-health-advantage", type=float, default=5.0)
    p.add_argument("--max-daily-turnover", type=float, default=0.10)
    p.add_argument("--min-local-improvement", type=float, default=3.0)

    p.add_argument(
        "--disable-regime-change-trigger",
        action="store_true",
        help="Disable full-search/shadow trigger on market regime change.",
    )
    p.add_argument("--min-regime-confidence-for-full-search", type=float, default=0.60)

    return p.parse_args()


def _main_impl(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_runtime_config(args.env)

    return run_transition_assessment(
        cfg=cfg,
        as_of=args.as_of,
        write_outputs=not bool(args.dry_run),
        update_latest=True,
        confirm_prod_write=bool(args.confirm_prod_write),
        min_health_score=float(args.min_health_score),
        health_drop_trigger=float(args.health_drop_trigger),
        min_grade=str(args.min_grade),
        max_ruin_probability=float(args.max_ruin_probability),
        max_drawdown_limit=float(args.max_drawdown_limit),
        full_search_refresh_days=int(args.full_search_refresh_days),
        shadow_confirmation_days=int(args.shadow_confirmation_days),
        min_shadow_health_advantage=float(args.min_shadow_health_advantage),
        max_daily_turnover=float(args.max_daily_turnover),
        min_local_improvement=float(args.min_local_improvement),
        regime_change_requires_full_search=not bool(args.disable_regime_change_trigger),
        min_regime_confidence_for_full_search=float(args.min_regime_confidence_for_full_search),
    )


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = bool(getattr(args, "dry_run", False))

    with capture_script_run(
        cfg=cfg,
        script_name="run_transition_assessment.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            payload = _main_impl(args)

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="create",
                entity_type="portfolio_transition_assessment",
                entity_id=str(payload.get("as_of")),
                as_of=str(payload.get("as_of")),
                source_script="run_transition_assessment.py",
                source_mode="transition_assessment",
                status=("dry_run" if is_dry_run else "success"),
                input_args=vars(args),
                output_keys=[] if is_dry_run else [
                    f"{cfg_engine_root(cfg)}/{TRANSITION_ASSESSMENT_TABLE}/dt={payload.get('as_of')}/transition_assessment.json",
                    f"{cfg_engine_root(cfg)}/{TRANSITION_ASSESSMENT_TABLE}/latest.json",
                ],
                metadata={
                    "recommendation": payload.get("recommendation"),
                    "full_search_required": payload.get("full_search_required"),
                    "shadow_portfolio_required": payload.get("shadow_portfolio_required"),
                    "triggers": (payload.get("diagnostics") or {}).get("triggers"),
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)

        except Exception as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="create",
                entity_type="portfolio_transition_assessment",
                entity_id=None,
                as_of=str(getattr(args, "as_of", "") or ""),
                source_script="run_transition_assessment.py",
                source_mode="transition_assessment",
                status="failed",
                input_args=vars(args),
                metadata={
                    "tier": "transition_assessment",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise


if __name__ == "__main__":
    main()