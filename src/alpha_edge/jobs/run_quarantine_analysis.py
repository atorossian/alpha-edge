# src/alpha_edge/jobs/run_quarantine_analysis.py
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.core.schemas import ScoreConfig, Position
from alpha_edge.core.data_loader import (
    clean_returns_matrix,
    s3_get_json,
    s3_init,
    s3_load_latest_json,
    s3_write_json_event,
)
from alpha_edge.core.market_store import MarketStore
from alpha_edge.jobs.run_daily_report import _load_closes_usd_from_ohlcv  # reuse trusted loader
from alpha_edge.market.build_returns_wide_cache import CacheConfig, build_returns_wide_cache
from alpha_edge.portfolio.report_engine import build_portfolio_report


ENGINE_BUCKET = "alpha-edge-algo"
ENGINE_REGION = "eu-west-1"
ENGINE_ROOT_PREFIX = "engine/v1"

RETURNS_WIDE_CACHE_PATH = "s3://alpha-edge-algo/market/cache/v1/returns_wide_min5y.parquet"

# Quarantine tables (evaluation-owned)
QUAR_EVALS_TABLE = "quarantine/evals"      # dt partitions + per-candidate latest.json
QUAR_SUMMARY_TABLE = "quarantine/summary"  # dt partitions + latest.json
QUAR_CAND_TABLE = "quarantine/candidates"  # candidate_id=.../latest.json (persistent baseline/streaks)
QUAR_REPORTS_TABLE = "quarantine/reports"  # dt partitions

# Auto-discovery source (input to quarantine evaluation)
PORTFOLIO_RUNS_TABLE = "portfolio_search/runs"


def _resolve_root_prefix(*, backtest_run_id: str | None) -> str:
    if backtest_run_id:
        return f"{ENGINE_ROOT_PREFIX}/backtests/{backtest_run_id}"
    return ENGINE_ROOT_PREFIX


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _s3_list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) or []:
            k = obj.get("Key", "")
            if k:
                keys.append(k)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def _s3_put_text(s3, *, bucket: str, key: str, text: str) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=(text or "").encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )


def _s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    import json

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _dt_prefix(root_prefix: str, table: str, dt_str: str) -> str:
    return f"{root_prefix.strip('/')}/{table.strip('/')}/dt={dt_str}/"


def _candidate_latest_key(root_prefix: str, table: str, cid: str) -> str:
    return f"{root_prefix.strip('/')}/{table.strip('/')}/candidate_id={cid}/latest.json"


def _metric(eval_dict: dict, *keys: str) -> float | None:
    for k in keys:
        if k in eval_dict:
            v = _safe_float(eval_dict.get(k))
            if v is not None:
                return v
    return None


def _s3_get_json_or_none(s3, *, bucket: str, key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    body = obj["Body"].read()
    import json

    try:
        out = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    return out if isinstance(out, dict) else None


def _discover_candidates_from_portfolio_runs(
    *,
    s3,
    bucket: str,
    root_prefix: str,
    as_of_ts: pd.Timestamp,
    lookback_days: int = 10,
) -> list[dict]:
    """
    Discover candidates from portfolio_search/runs by reading outputs.discrete_allocation.shares.
    candidate_id = run_id (stable).
    """
    out: list[dict] = []
    start_ts = (as_of_ts - pd.Timedelta(days=int(lookback_days))).normalize()
    days = pd.date_range(start_ts, as_of_ts, freq="D")

    for d in days:
        dt_str = d.strftime("%Y-%m-%d")
        prefix = _dt_prefix(root_prefix, PORTFOLIO_RUNS_TABLE, dt_str)
        keys = [k for k in _s3_list_keys(s3, bucket=bucket, prefix=prefix) if k.lower().endswith(".json")]

        for k in sorted(keys):
            try:
                payload = s3_get_json(s3, bucket=bucket, key=k) or {}
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            run_id = str(payload.get("run_id") or "").strip()
            if not run_id:
                run_id = (k.split("/")[-1] or "").replace(".json", "").strip()
                if not run_id:
                    continue

            outputs = payload.get("outputs") or {}
            if not isinstance(outputs, dict):
                continue
            disc = outputs.get("discrete_allocation") or {}
            if not isinstance(disc, dict):
                continue
            shares = disc.get("shares")
            if not isinstance(shares, dict) or not shares:
                continue

            shares_n: dict[str, float] = {}
            for t, q in shares.items():
                tt = str(t).upper().strip()
                qf = _safe_float(q)
                if tt and qf is not None and abs(qf) > 0:
                    shares_n[tt] = float(qf)
            if len(shares_n) < 2:
                continue

            out.append(
                {
                    "candidate_id": run_id,
                    "run_key": k,
                    "shares": shares_n,
                    "source": {
                        "table": PORTFOLIO_RUNS_TABLE,
                        "run_key": k,
                        "run_id": run_id,
                        "run_as_of": payload.get("as_of"),
                    },
                }
            )

    # de-dup by candidate_id; keep latest run_key lexicographically
    dedup: dict[str, dict] = {}
    for rec in out:
        cid = rec["candidate_id"]
        if cid not in dedup or str(rec.get("run_key", "")) > str(dedup[cid].get("run_key", "")):
            dedup[cid] = rec

    return list(dedup.values())


def _discover_pending_candidates_from_state(
    *,
    s3,
    bucket: str,
    root_prefix: str,
) -> list[dict]:
    """
    Discover candidates from quarantine/candidates that are still pending.
    This allows evaluation to continue even when outside lookback window.
    """
    prefix = f"{root_prefix.strip('/')}/{QUAR_CAND_TABLE.strip('/')}/candidate_id="
    keys = _s3_list_keys(s3, bucket=bucket, prefix=prefix)
    latest_keys = [k for k in keys if k.endswith("/latest.json")]

    out: list[dict] = []
    for k in latest_keys:
        cand_state = _s3_get_json_or_none(s3, bucket=bucket, key=k) or {}
        if not isinstance(cand_state, dict):
            continue

        cid = str(cand_state.get("candidate_id") or "").strip()
        if not cid:
            try:
                cid = k.split("candidate_id=")[-1].split("/")[0].strip()
            except Exception:
                cid = ""
        if not cid:
            continue

        q = cand_state.get("quarantine") or {}
        if not isinstance(q, dict):
            q = {}

        status = str(q.get("status") or "PENDING").upper()
        if status in ("APPROVED", "REJECTED", "EXPIRED"):
            continue

        shares = cand_state.get("shares") or {}
        if not isinstance(shares, dict) or len(shares) < 2:
            continue

        shares_n: dict[str, float] = {}
        for t, qv in shares.items():
            tt = str(t).upper().strip()
            qf = _safe_float(qv)
            if tt and qf is not None and abs(qf) > 0:
                shares_n[tt] = float(qf)
        if len(shares_n) < 2:
            continue

        out.append(
            {
                "candidate_id": cid,
                "run_key": None,
                "shares": shares_n,
                "source": cand_state.get("source") or {"table": QUAR_CAND_TABLE, "candidate_state_key": k},
                "from_state_key": k,
            }
        )

    return out


def _signed_gross_weights_from_shares(*, shares: dict[str, float], prices_usd: pd.Series) -> dict[str, float]:
    px = pd.to_numeric(prices_usd, errors="coerce").replace([np.inf, -np.inf], np.nan)
    exp: dict[str, float] = {}
    gross = 0.0
    for t, q in shares.items():
        if t not in px.index:
            continue
        p = _safe_float(px.loc[t])
        if p is None or p <= 0:
            continue
        e = float(p) * float(q)
        exp[t] = e
        gross += abs(e)
    if gross <= 0:
        return {}
    return {t: float(exp[t] / gross) for t in exp}


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float | None:
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return None
    va = np.array([float(a.get(k, 0.0)) for k in keys], dtype=np.float64)
    vb = np.array([float(b.get(k, 0.0)) for k in keys], dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 0 or nb <= 0:
        return None
    return float(np.dot(va, vb) / (na * nb))


def _l1_drift(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a.keys()) | set(b.keys())
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def _severity_rules(
    *,
    score_today: float | None,
    score_ref: float | None,
    decay_rate: float | None,
    metric_flags: list[str],
    age_days: int,
    consecutive_amber: int,
    consecutive_red: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    if score_today is None or score_ref is None:
        return "AMBER", ["missing_score_ref_or_today"]

    score_decay = float(score_today - score_ref)
    if score_decay <= -0.15:
        reasons.append("score_drop_gt_0.15")
        return "RED", reasons + metric_flags

    if decay_rate is not None and decay_rate <= -0.01 and age_days >= 5:
        reasons.append("decay_rate_lt_-0.01_per_day")
        return "RED", reasons + metric_flags

    if any(f.startswith("ruin") or f.startswith("mdd") for f in metric_flags):
        reasons.append("risk_metric_gap")
        return "AMBER", reasons + metric_flags

    if score_decay <= -0.08:
        reasons.append("score_drop_gt_0.08")
        return "AMBER", reasons + metric_flags

    if decay_rate is not None and decay_rate <= -0.005:
        reasons.append("decay_rate_lt_-0.005_per_day")
        return "AMBER", reasons + metric_flags

    if consecutive_red >= 2:
        return "RED", ["red_streak"] + metric_flags
    if consecutive_amber >= 3:
        return "AMBER", ["amber_streak"] + metric_flags

    return "GREEN", metric_flags


def _print_quarantine_candidate_report(cid: str, rec: dict) -> None:
    d = rec.get("degradation") or {}
    sev = d.get("severity")
    age = d.get("age_days")
    s_today = d.get("score_today")
    s_ref = d.get("score_ref")
    decay = d.get("score_decay")
    rate = d.get("score_decay_rate")
    reg_ch = d.get("regime_changed")
    reasons = d.get("reasons") or []

    print("\n" + "─" * 60)
    print(f"Quarantine Candidate: {cid}")
    print("─" * 60)
    print(f"Status: {rec.get('status')} | Severity: {sev} | Age: {age}d")
    if s_today is not None and s_ref is not None and decay is not None and rate is not None:
        print(f"Score:  today={s_today:.4f}  ref={s_ref:.4f}  decay={decay:+.4f}  rate={rate:+.4f}/day")
    else:
        print(f"Score:  today={s_today} ref={s_ref} decay={decay} rate={rate}")
    print(f"Regime: {d.get('baseline_regime')} -> {d.get('current_regime')}  changed={reg_ch}")
    print("Reasons:", ", ".join(reasons) if reasons else "none")


def run_quarantine_analysis_asof(
    *,
    as_of: str,
    backtest_run_id: str | None = None,
    write_outputs: bool = True,
    update_latest: bool = True,
    min_quarantine_days: int = 5,
    approve_requires_green_days: int = 5,
    ttl_days: int = 12,
    lookback_days: int = 10,
    min_entry_score: float = 0.75,
) -> dict:
    root_prefix = _resolve_root_prefix(backtest_run_id=backtest_run_id)
    mode = "backtest" if backtest_run_id else "live"

    as_of_ts = pd.Timestamp(as_of).tz_localize(None).normalize()
    as_of_date = as_of_ts.strftime("%Y-%m-%d")
    run_dt = pd.Timestamp(dt.date.today()).normalize() if mode == "live" else as_of_ts

    s3 = s3_init(ENGINE_REGION)
    _market = MarketStore(bucket="alpha-edge-algo")  # kept for compatibility with your infra

    raw_score_cfg = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=root_prefix, table="configs/score_config"
    )
    if not raw_score_cfg:
        raise RuntimeError("Missing S3 latest score_config.")
    score_cfg = ScoreConfig(**raw_score_cfg)

    # 1) recent candidates from portfolio runs
    recent = _discover_candidates_from_portfolio_runs(
        s3=s3,
        bucket=ENGINE_BUCKET,
        root_prefix=root_prefix,
        as_of_ts=as_of_ts,
        lookback_days=int(lookback_days),
    )

    # 2) pending candidates already persisted in quarantine/candidates
    pending_from_state = _discover_pending_candidates_from_state(
        s3=s3,
        bucket=ENGINE_BUCKET,
        root_prefix=root_prefix,
    )

    # union + dedup (prefer recent run record if both exist)
    union: dict[str, dict] = {}
    for rec in pending_from_state:
        cid = str(rec.get("candidate_id") or "").strip()
        if cid:
            union[cid] = rec
    for rec in recent:
        cid = str(rec.get("candidate_id") or "").strip()
        if cid:
            union[cid] = rec  # overwrite with recent

    discovered = list(union.values())

    if not discovered:
        out = {
            "as_of": as_of_date,
            "status": "no_candidates",
            "approved": [],
            "pending": [],
            "rejected": [],
            "expired": [],
            "counts": {"approved": 0, "pending": 0, "rejected": 0, "expired": 0},
            "meta": {"lookback_days": int(lookback_days), "ttl_days": int(ttl_days)},
        }
        if write_outputs:
            s3_write_json_event(
                s3,
                bucket=ENGINE_BUCKET,
                root_prefix=root_prefix,
                table=QUAR_SUMMARY_TABLE,
                dt=run_dt,
                filename="summary.json",
                payload=out,
                update_latest=update_latest,
            )
        return out

    market_hmm = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="regimes/market_hmm"
    ) or {}
    cur_regime_label = str((market_hmm.get("label_commit") or market_hmm.get("label") or "UNKNOWN"))

    # Keep returns cache warm/consistent (as your system expects)
    cache_cfg = CacheConfig(bucket=ENGINE_BUCKET, min_years=float(5.0))
    build_returns_wide_cache(cache_cfg)
    returns_wide = pd.read_parquet(RETURNS_WIDE_CACHE_PATH, engine="pyarrow").sort_index()
    returns_wide, _ = clean_returns_matrix(returns_wide)
    returns_wide.index = pd.to_datetime(returns_wide.index, errors="coerce").tz_localize(None).normalize()
    _ = returns_wide.loc[returns_wide.index <= as_of_ts]

    resolved: list[dict] = []
    tickers_union: set[str] = set()

    for rec in discovered:
        cid = str(rec.get("candidate_id") or "").strip()
        shares = rec.get("shares") or {}
        if not cid or not isinstance(shares, dict) or len(shares) < 2:
            continue
        shares_n = {str(t).upper().strip(): float(q) for t, q in shares.items()}
        tickers_union |= set(shares_n.keys())
        resolved.append({"candidate_id": cid, "shares": shares_n, "source": rec.get("source") or {}})

    if not resolved:
        out = {
            "as_of": as_of_date,
            "status": "no_valid_candidates",
            "approved": [],
            "pending": [],
            "rejected": [],
            "expired": [],
            "counts": {"approved": 0, "pending": 0, "rejected": 0, "expired": 0},
            "meta": {"lookback_days": int(lookback_days), "ttl_days": int(ttl_days)},
        }
        if write_outputs:
            s3_write_json_event(
                s3,
                bucket=ENGINE_BUCKET,
                root_prefix=root_prefix,
                table=QUAR_SUMMARY_TABLE,
                dt=run_dt,
                filename="summary.json",
                payload=out,
                update_latest=update_latest,
            )
        return out

    START_HISTORY = "2015-01-01"
    closes_all = _load_closes_usd_from_ohlcv(
        tickers=sorted(tickers_union),
        start=START_HISTORY,
        end=as_of_date,
    )
    prices_usd = pd.to_numeric(closes_all.iloc[-1], errors="coerce").replace([np.inf, -np.inf], np.nan)

    approved: list[str] = []
    pending: list[str] = []
    rejected: list[str] = []
    expired: list[str] = []

    for item in resolved:
        cid = item["candidate_id"]
        shares = item["shares"]

        cand_state_key = _candidate_latest_key(root_prefix, QUAR_CAND_TABLE, cid)
        cand_state = _s3_get_json_or_none(s3, bucket=ENGINE_BUCKET, key=cand_state_key) or {}

        q = cand_state.get("quarantine") or {}
        if not isinstance(q, dict):
            q = {}

        status_existing = str(q.get("status") or "").upper()
        if status_existing in ("APPROVED", "REJECTED", "EXPIRED"):
            continue

        is_new = (not isinstance(q.get("baseline_eval"), dict))

        cand_state.setdefault("candidate_id", cid)
        cand_state.setdefault("source", item.get("source") or {})

        # Persist shares always (so we can evaluate outside lookback)
        cand_state["shares"] = {str(t).upper().strip(): float(qv) for t, qv in shares.items()}

        # start_as_of fallback chain: q.start_as_of -> cand_state.last_seen_as_of -> as_of_date
        start_as_of = str(q.get("start_as_of") or "").strip() or None
        if start_as_of is None:
            start_as_of = str(cand_state.get("last_seen_as_of") or "").strip() or None
        if start_as_of is None:
            start_as_of = as_of_date

        age_days = int((as_of_ts - pd.Timestamp(start_as_of).tz_localize(None).normalize()).days)
        age_days = max(0, age_days)

        equity_ref = _safe_float(cand_state.get("equity_ref"))
        if equity_ref is None:
            gross_tmp = 0.0
            for t, qv in shares.items():
                if t not in prices_usd.index:
                    continue
                p = _safe_float(prices_usd.loc[t])
                if p is None or p <= 0:
                    continue
                gross_tmp += abs(float(p) * float(qv))
            equity_ref = float(gross_tmp / 5.0) if gross_tmp > 0 else 10000.0

        goals = cand_state.get("goals") or [7500.0, 10000.0, 12500.0]
        if not isinstance(goals, list) or not goals:
            goals = [7500.0, 10000.0, 12500.0]
        main_goal = _safe_float(cand_state.get("main_goal")) or 10000.0

        # Evaluate candidate via same report engine
        try:
            positions = {
                t: Position(ticker=t, quantity=float(qv), entry_price=None, currency="USD")
                for t, qv in shares.items()
            }
            closes = closes_all[list(positions.keys())].copy()

            report = build_portfolio_report(
                closes=closes,
                positions=positions,
                equity=float(equity_ref),
                goals=list(goals),
                main_goal=float(main_goal),
                score_config=score_cfg,
                prices_usd=prices_usd,
            )
            eval_today = asdict(report.eval)
        except Exception as e:
            eval_today = {"error": f"{type(e).__name__}: {e}"}

        score_today = _metric(eval_today, "score")

        # Entry gate for brand new candidates
        if is_new and (score_today is None or float(score_today) < float(min_entry_score)):
            status = "REJECTED"
            sev = "RED"
            reasons = [f"below_entry_score({score_today}->{min_entry_score})"]

            q = {
                "start_as_of": as_of_date,
                "baseline_eval": {},  # not accepted
                "baseline_regime_label": cur_regime_label,
                "streak_amber": 0,
                "streak_red": 0,
                "streak_green": 0,
                "status": status,
                "last_eval_as_of": as_of_date,
                "last_regime_label": cur_regime_label,
                "degradation": {
                    "as_of": as_of_date,
                    "start_as_of": as_of_date,
                    "age_days": 0,
                    "score_today": score_today,
                    "score_ref": None,
                    "score_decay": None,
                    "score_decay_rate": None,
                    "half_life_days": None,
                    "w_drift_l1": 0.0,
                    "cosine": None,
                    "regime_changed": False,
                    "baseline_regime": cur_regime_label,
                    "current_regime": cur_regime_label,
                    "severity": sev,
                    "reasons": reasons,
                },
            }

            cand_state["quarantine"] = q
            cand_state["last_seen_as_of"] = as_of_date

            eval_rec = {
                "as_of": as_of_date,
                "candidate_id": cid,
                "status": status,
                "severity": sev,
                "degradation": q["degradation"],
                "eval": eval_today,
                "baseline_eval": q.get("baseline_eval"),
                "source": item.get("source") or {},
            }

            rejected.append(cid)
            _print_quarantine_candidate_report(cid=cid, rec=eval_rec)

            if write_outputs:
                latest_eval_key = _candidate_latest_key(root_prefix, QUAR_EVALS_TABLE, cid)
                _s3_put_json(s3, bucket=ENGINE_BUCKET, key=latest_eval_key, payload=eval_rec)

                s3_write_json_event(
                    s3,
                    bucket=ENGINE_BUCKET,
                    root_prefix=root_prefix,
                    table=QUAR_EVALS_TABLE,
                    dt=run_dt,
                    filename=f"eval_{cid}.json",
                    payload=eval_rec,
                    update_latest=False,
                )

                _s3_put_json(s3, bucket=ENGINE_BUCKET, key=cand_state_key, payload=cand_state)

            continue

        # baseline init (accepted candidates only)
        if "baseline_eval" not in q or not isinstance(q.get("baseline_eval"), dict):
            q["start_as_of"] = start_as_of
            q["baseline_eval"] = dict(eval_today) if isinstance(eval_today, dict) else {}
            q["baseline_regime_label"] = cur_regime_label
            q["streak_amber"] = 0
            q["streak_red"] = 0
            q["streak_green"] = 0
            q["status"] = "PENDING"

        base_eval = q.get("baseline_eval") or {}
        if not isinstance(base_eval, dict):
            base_eval = {}

        score_ref = _metric(base_eval, "score")
        if score_ref is None:
            score_ref = score_today

        score_decay = None if (score_today is None or score_ref is None) else float(score_today - score_ref)
        decay_rate = None
        if score_decay is not None:
            decay_rate = float(score_decay / max(1, age_days))

        half_life_days = None
        if score_today is not None and score_ref is not None and decay_rate is not None and decay_rate < 0:
            target = 0.5 * float(score_ref)
            t_h = (float(target) - float(score_today)) / float(decay_rate)
            if np.isfinite(t_h) and t_h > 0:
                half_life_days = float(t_h)

        metric_flags: list[str] = []
        ruin_today = _metric(eval_today, "ruin_prob_1y", "ruin_probability", "ruin_prob")
        ruin_ref = _metric(base_eval, "ruin_prob_1y", "ruin_probability", "ruin_prob")
        if ruin_today is not None and ruin_ref is not None and ruin_today > ruin_ref + 0.01:
            metric_flags.append(f"ruin_up({ruin_ref:.3f}->{ruin_today:.3f})")

        mdd_today = _metric(eval_today, "max_drawdown", "mdd")
        mdd_ref = _metric(base_eval, "max_drawdown", "mdd")
        if mdd_today is not None and mdd_ref is not None and float(mdd_today) < float(mdd_ref) - 0.05:
            metric_flags.append(f"mdd_worse({mdd_ref:.3f}->{mdd_today:.3f})")

        ann_today = _metric(eval_today, "ann_return", "annual_return")
        ann_ref = _metric(base_eval, "ann_return", "annual_return")
        if ann_today is not None and ann_ref is not None and ann_today < ann_ref - 0.08:
            metric_flags.append(f"ann_down({ann_ref:.3f}->{ann_today:.3f})")

        # drift metrics currently compare target to itself (placeholder hook)
        w_target = _signed_gross_weights_from_shares(shares=shares, prices_usd=prices_usd)
        w_today = w_target
        w_drift_l1 = _l1_drift(w_today, w_target)
        cosine = _cosine_sim(w_today, w_target)

        base_reg = str(q.get("baseline_regime_label") or "UNKNOWN")
        regime_changed = (base_reg != cur_regime_label)

        sev, reasons = _severity_rules(
            score_today=score_today,
            score_ref=score_ref,
            decay_rate=decay_rate,
            metric_flags=metric_flags,
            age_days=age_days,
            consecutive_amber=int(q.get("streak_amber", 0) or 0),
            consecutive_red=int(q.get("streak_red", 0) or 0),
        )
        if regime_changed and age_days >= 3 and sev != "GREEN":
            reasons = (reasons or []) + ["regime_changed"]

        # streaks
        if sev == "GREEN":
            q["streak_green"] = int(q.get("streak_green", 0) or 0) + 1
            q["streak_amber"] = 0
            q["streak_red"] = 0
        elif sev == "AMBER":
            q["streak_green"] = 0
            q["streak_amber"] = int(q.get("streak_amber", 0) or 0) + 1
            q["streak_red"] = 0
        else:
            q["streak_green"] = 0
            q["streak_amber"] = 0
            q["streak_red"] = int(q.get("streak_red", 0) or 0) + 1

        status = str(q.get("status") or "PENDING").upper()
        if status not in ("PENDING", "APPROVED", "REJECTED", "EXPIRED"):
            status = "PENDING"

        # Hard reject if it falls below entry threshold while still pending
        if status == "PENDING" and (score_today is None or float(score_today) < float(min_entry_score)):
            status = "REJECTED"
            sev = "RED"
            reasons = (reasons or []) + [f"fell_below_entry_score({score_today}->{min_entry_score})"]

        # terminal rules
        if status == "PENDING" and age_days > int(ttl_days):
            status = "EXPIRED"

        if status == "PENDING":
            if sev == "RED" and int(q.get("streak_red", 0) or 0) >= 2:
                status = "REJECTED"

        if status == "PENDING":
            if age_days >= int(min_quarantine_days) and int(q.get("streak_green", 0) or 0) >= int(approve_requires_green_days):
                status = "APPROVED"

        q["status"] = status
        q["last_eval_as_of"] = as_of_date
        q["last_regime_label"] = cur_regime_label

        q["degradation"] = {
            "as_of": as_of_date,
            "start_as_of": str(q.get("start_as_of")),
            "age_days": int(age_days),
            "score_today": score_today,
            "score_ref": score_ref,
            "score_decay": score_decay,
            "score_decay_rate": decay_rate,
            "half_life_days": half_life_days,
            "w_drift_l1": float(w_drift_l1),
            "cosine": cosine,
            "regime_changed": bool(regime_changed),
            "baseline_regime": base_reg,
            "current_regime": cur_regime_label,
            "severity": sev,
            "reasons": reasons,
        }

        cand_state["quarantine"] = q
        cand_state["last_seen_as_of"] = as_of_date

        eval_rec = {
            "as_of": as_of_date,
            "candidate_id": cid,
            "status": status,
            "severity": sev,
            "degradation": q["degradation"],
            "eval": eval_today,
            "baseline_eval": q.get("baseline_eval"),
            "source": item.get("source") or {},
        }

        if status == "APPROVED":
            approved.append(cid)
        elif status == "REJECTED":
            rejected.append(cid)
        elif status == "EXPIRED":
            expired.append(cid)
        else:
            pending.append(cid)

        _print_quarantine_candidate_report(cid=cid, rec=eval_rec)

        if write_outputs:
            # text report
            lines: list[str] = []
            d = eval_rec.get("degradation") or {}
            lines.append(f"Quarantine Candidate: {cid}")
            lines.append(f"as_of: {as_of_date}")
            lines.append(f"status: {status} | severity: {sev} | age_days: {d.get('age_days')}")
            lines.append("")
            lines.append(f"score_today: {d.get('score_today')}")
            lines.append(f"score_ref:   {d.get('score_ref')}")
            lines.append(f"score_decay: {d.get('score_decay')}")
            lines.append(f"decay_rate:  {d.get('score_decay_rate')}")
            lines.append("")
            lines.append(
                f"baseline_regime: {d.get('baseline_regime')}  current_regime: {d.get('current_regime')}  changed={d.get('regime_changed')}"
            )
            lines.append("reasons: " + (", ".join(d.get("reasons") or []) or "none"))
            lines.append("")
            lines.append("eval_today:")
            for k, v in sorted((eval_today or {}).items()):
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("baseline_eval:")
            base_eval2 = q.get("baseline_eval") or {}
            if isinstance(base_eval2, dict):
                for k, v in sorted(base_eval2.items()):
                    lines.append(f"  {k}: {v}")

            report_key = f"{root_prefix.strip('/')}/{QUAR_REPORTS_TABLE}/dt={run_dt.strftime('%Y-%m-%d')}/report_{cid}.txt"
            _s3_put_text(s3, bucket=ENGINE_BUCKET, key=report_key, text="\n".join(lines))

            # candidate latest eval
            latest_eval_key = _candidate_latest_key(root_prefix, QUAR_EVALS_TABLE, cid)
            _s3_put_json(s3, bucket=ENGINE_BUCKET, key=latest_eval_key, payload=eval_rec)

            # per-day eval record
            s3_write_json_event(
                s3,
                bucket=ENGINE_BUCKET,
                root_prefix=root_prefix,
                table=QUAR_EVALS_TABLE,
                dt=run_dt,
                filename=f"eval_{cid}.json",
                payload=eval_rec,
                update_latest=False,
            )

            # persist candidate state latest
            _s3_put_json(s3, bucket=ENGINE_BUCKET, key=cand_state_key, payload=cand_state)

    summary = {
        "as_of": as_of_date,
        "status": "ok",
        "approved": sorted(set(approved)),
        "pending": sorted(set(pending)),
        "rejected": sorted(set(rejected)),
        "expired": sorted(set(expired)),
        "counts": {
            "approved": int(len(set(approved))),
            "pending": int(len(set(pending))),
            "rejected": int(len(set(rejected))),
            "expired": int(len(set(expired))),
        },
        "meta": {
            "mode": mode,
            "market_regime_label": cur_regime_label,
            "n_union": int(len(discovered)),
            "n_resolved": int(len(resolved)),
            "source_table": PORTFOLIO_RUNS_TABLE,
            "lookback_days": int(lookback_days),
            "min_quarantine_days": int(min_quarantine_days),
            "approve_requires_green_days": int(approve_requires_green_days),
            "ttl_days": int(ttl_days),
            "min_entry_score": float(min_entry_score),
        },
    }

    if write_outputs:
        s3_write_json_event(
            s3,
            bucket=ENGINE_BUCKET,
            root_prefix=root_prefix,
            table=QUAR_SUMMARY_TABLE,
            dt=run_dt,
            filename="summary.json",
            payload=summary,
            update_latest=update_latest,
        )

    return summary


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=str, default=None)
    ap.add_argument("--backtest-run-id", type=str, default=None)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-latest", action="store_true")

    ap.add_argument("--min-days", type=int, default=5)
    ap.add_argument("--approve-green-days", type=int, default=5)
    ap.add_argument("--ttl-days", type=int, default=12)
    ap.add_argument("--lookback-days", type=int, default=10)
    ap.add_argument("--min-entry-score", type=float, default=0.75)

    return ap.parse_args()


def main():
    args = parse_args()
    as_of = args.as_of or pd.Timestamp(dt.date.today()).strftime("%Y-%m-%d")

    out = run_quarantine_analysis_asof(
        as_of=as_of,
        backtest_run_id=args.backtest_run_id,
        write_outputs=(not bool(args.no_write)),
        update_latest=(not bool(args.no_latest)),
        min_quarantine_days=int(args.min_days),
        approve_requires_green_days=int(args.approve_green_days),
        ttl_days=int(args.ttl_days),
        lookback_days=int(args.lookback_days),
        min_entry_score=float(args.min_entry_score),
    )

    print("\n=== QUARANTINE SUMMARY ===")
    print(f"as_of:     {out.get('as_of')}")
    print(f"approved:  {out.get('counts', {}).get('approved', 0)}")
    print(f"pending:   {out.get('counts', {}).get('pending', 0)}")
    print(f"rejected:  {out.get('counts', {}).get('rejected', 0)}")
    print(f"expired:   {out.get('counts', {}).get('expired', 0)}")
    if out.get("approved"):
        print("approved_ids:", ", ".join(out["approved"][:20]))


if __name__ == "__main__":
    main()
