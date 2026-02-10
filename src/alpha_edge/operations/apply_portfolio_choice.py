from __future__ import annotations

import argparse
import datetime as dt
import json
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict

import boto3
import pandas as pd

from alpha_edge.core.data_loader import s3_get_json

BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"

PORTFOLIO_RUNS_TABLE = "portfolio_search/runs"
CHOICES_TABLE = "portfolio_choices"
TARGETS_TABLE = "targets"

# NEW: state + history for quarantine
CHOICE_STATE_TABLE = "portfolio_choice_state"
CHOICE_HISTORY_TABLE = "portfolio_choice_history"


# -------------------------
# S3 helpers
# -------------------------
def s3_client(region: str = REGION):
    return boto3.client("s3", region_name=region)


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def s3_list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []) or []:
            keys.append(it["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def s3_latest_dt(s3, *, bucket: str, table_prefix: str) -> Optional[str]:
    prefix = table_prefix.rstrip("/") + "/dt="
    keys = s3_list_keys(s3, bucket=bucket, prefix=prefix)
    dts = set()
    for k in keys:
        parts = k.split("/")
        for p in parts:
            if p.startswith("dt=") and len(p) == len("dt=YYYY-MM-DD"):
                dts.add(p.replace("dt=", ""))
    if not dts:
        return None
    return sorted(dts)[-1]


def s3_latest_run_key(s3, *, bucket: str, runs_table_prefix: str) -> str:
    latest_dt = s3_latest_dt(s3, bucket=bucket, table_prefix=runs_table_prefix)
    if not latest_dt:
        raise RuntimeError(f"No dt partitions found under s3://{bucket}/{runs_table_prefix}/dt=YYYY-MM-DD/")

    dt_prefix = f"{runs_table_prefix.rstrip('/')}/dt={latest_dt}/"
    keys = s3_list_keys(s3, bucket=bucket, prefix=dt_prefix)
    json_keys = [k for k in keys if k.endswith(".json")]
    if not json_keys:
        raise RuntimeError(f"No JSON files found under s3://{bucket}/{dt_prefix}")

    run_keys = [k for k in json_keys if k.split("/")[-1].startswith("run_")]
    pick_from = run_keys if run_keys else json_keys
    return sorted(pick_from)[-1]


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def dt_key(table: str, dt_str: str, filename: str) -> str:
    return engine_key(table, f"dt={dt_str}", filename)


# -------------------------
# Choice state schema
# -------------------------
@dataclass
class TargetsRef:
    table: str
    dt: str
    filename: str

    def to_key(self) -> str:
        return dt_key(self.table, self.dt, self.filename)


@dataclass
class PickedFrom:
    portfolio_search_run_key: str
    run_id: str
    run_as_of: str
    variant: str


@dataclass
class PortfolioSlot:
    # ACTIVE or CANDIDATE slot
    choice_id: str
    as_of: str
    picked_from: PickedFrom
    targets_ref: TargetsRef
    # metrics at selection time (baseline)
    baseline: Dict[str, Any]


@dataclass
class QuarantineConfig:
    min_quarantine_days: int = 5
    # score degradation thresholds (tune as you like)
    max_score_drop_abs: float = 0.08         # candidate_score - baseline_score >= -0.08
    max_score_drop_frac: float = 0.12        # candidate_score / baseline_score >= 0.88
    min_samples_to_decide: int = 3           # need at least N daily points before we can reject

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChoiceState:
    as_of: str
    active: Optional[PortfolioSlot]
    candidate: Optional[PortfolioSlot]
    quarantine: Dict[str, Any]  # config + running status/meta

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "active": None if self.active is None else asdict(self.active),
            "candidate": None if self.candidate is None else asdict(self.candidate),
            "quarantine": dict(self.quarantine or {}),
        }


def _load_choice_state(s3) -> ChoiceState:
    key = engine_key(CHOICE_STATE_TABLE, "latest.json")
    raw = s3_get_json(s3, bucket=BUCKET, key=key) or {}
    qc = raw.get("quarantine") or {}
    # seed defaults
    cfg = QuarantineConfig()
    qc = {
        "config": {**cfg.to_dict(), **(qc.get("config") or {})},
        "status": qc.get("status") or "ok",
        "notes": qc.get("notes") or [],
        "candidate_days": int(qc.get("candidate_days", 0) or 0),
        "candidate_points": int(qc.get("candidate_points", 0) or 0),
        "last_update": qc.get("last_update"),
        "last_decision": qc.get("last_decision"),
    }

    def parse_slot(x: dict | None) -> Optional[PortfolioSlot]:
        if not isinstance(x, dict):
            return None
        try:
            pf = x.get("picked_from") or {}
            tr = x.get("targets_ref") or {}
            return PortfolioSlot(
                choice_id=str(x.get("choice_id")),
                as_of=str(x.get("as_of")),
                picked_from=PickedFrom(
                    portfolio_search_run_key=str(pf.get("portfolio_search_run_key")),
                    run_id=str(pf.get("run_id")),
                    run_as_of=str(pf.get("run_as_of")),
                    variant=str(pf.get("variant")),
                ),
                targets_ref=TargetsRef(
                    table=str(tr.get("table")),
                    dt=str(tr.get("dt")),
                    filename=str(tr.get("filename")),
                ),
                baseline=dict(x.get("baseline") or {}),
            )
        except Exception:
            return None

    return ChoiceState(
        as_of=str(raw.get("as_of") or pd.Timestamp(dt.date.today()).strftime("%Y-%m-%d")),
        active=parse_slot(raw.get("active")),
        candidate=parse_slot(raw.get("candidate")),
        quarantine=qc,
    )


def _save_choice_state(s3, *, dt_str: str, state: ChoiceState, update_latest: bool = True) -> None:
    payload = state.to_dict()
    key = dt_key(CHOICE_STATE_TABLE, dt_str, "state.json")
    s3_put_json(s3, bucket=BUCKET, key=key, payload=payload)
    if update_latest:
        s3_put_json(s3, bucket=BUCKET, key=engine_key(CHOICE_STATE_TABLE, "latest.json"), payload=payload)


def _append_choice_history(s3, *, dt_str: str, event: dict, update_latest: bool = False) -> None:
    # history is partitioned; latest is optional
    eid = event.get("event_id") or uuid.uuid4().hex[:10]
    key = dt_key(CHOICE_HISTORY_TABLE, dt_str, f"event_{dt_str}_{eid}.json")
    s3_put_json(s3, bucket=BUCKET, key=key, payload=event)
    if update_latest:
        s3_put_json(s3, bucket=BUCKET, key=engine_key(CHOICE_HISTORY_TABLE, "latest.json"), payload=event)


# -------------------------
# Core behavior
# -------------------------
def _extract_shares_from_run_payload(run_payload: dict) -> dict[str, float]:
    outputs = (run_payload.get("outputs") or {})
    disc = (outputs.get("discrete_allocation") or {})
    shares = disc.get("shares") or {}
    if not isinstance(shares, dict) or not shares:
        raise RuntimeError("portfolio search run missing outputs.discrete_allocation.shares")

    out = {}
    for t, q in shares.items():
        try:
            qq = float(q)
        except Exception:
            continue
        # keep non-zero shares (allow future short support if needed)
        if not (pd.isna(qq) or qq == 0.0):
            out[str(t)] = qq
    if not out:
        raise RuntimeError("portfolio search run has no non-zero shares")
    return out


def apply_candidate_from_latest_search(*, dry_run: bool = False) -> None:
    """
    Registers the latest portfolio search output as a CANDIDATE under quarantine.
    DOES NOT modify targets/latest.json (so ACTIVE keeps trading).
    """
    s3 = s3_client(REGION)

    runs_prefix = engine_key(PORTFOLIO_RUNS_TABLE)
    run_key = s3_latest_run_key(s3, bucket=BUCKET, runs_table_prefix=runs_prefix)
    run_payload = s3_get_json(s3, bucket=BUCKET, key=run_key) or {}

    run_id = run_payload.get("run_id")
    run_as_of = run_payload.get("as_of")
    shares = _extract_shares_from_run_payload(run_payload)

    if not run_id or not run_as_of:
        raise RuntimeError("Latest portfolio search run payload missing run_id or as_of")

    today = pd.Timestamp(dt.date.today())
    dt_str = today.strftime("%Y-%m-%d")
    choice_id = f"{dt_str}-{uuid.uuid4().hex[:8]}"

    # Candidate targets payload (intent only, no prices)
    targets_payload = {
        "as_of": dt_str,
        "choice_id": choice_id,
        "mode": "CANDIDATE_QUARANTINE",
        "source": {
            "portfolio_search_run_key": run_key,
            "run_id": run_id,
            "run_as_of": run_as_of,
            "variant": "outputs.discrete_allocation.shares",
        },
        "targets": {
            # keep sign; execution later can decide policy
            "shares": {str(t): float(q) for t, q in shares.items()},
        },
    }

    # Choice record payload (for audit / trace)
    choice_payload = {
        "choice_id": choice_id,
        "as_of": dt_str,
        "status": "CANDIDATE",
        "picked_from": {
            "portfolio_search_run_key": run_key,
            "run_id": run_id,
            "run_as_of": run_as_of,
            "variant": "outputs.discrete_allocation.shares",
        },
        "targets_ref": {
            "table": TARGETS_TABLE,
            "dt": dt_str,
            "filename": f"targets_{choice_id}.json",
        },
        "note": "Candidate stored for quarantine. ACTIVE portfolio continues trading until promotion.",
    }

    targets_key = dt_key(TARGETS_TABLE, dt_str, f"targets_{choice_id}.json")
    choice_key = dt_key(CHOICES_TABLE, dt_str, f"choice_{choice_id}.json")

    # Load / update state
    state = _load_choice_state(s3)

    # Candidate baseline metrics: you can enrich later (score, market regime, etc.)
    baseline = {
        "baseline_score": float((run_payload.get("outputs") or {}).get("score", None))
        if isinstance((run_payload.get("outputs") or {}).get("score", None), (int, float))
        else None,
        "run_as_of": str(run_as_of),
        "registered_at": dt_str,
    }

    candidate_slot = PortfolioSlot(
        choice_id=choice_id,
        as_of=dt_str,
        picked_from=PickedFrom(
            portfolio_search_run_key=run_key,
            run_id=str(run_id),
            run_as_of=str(run_as_of),
            variant="outputs.discrete_allocation.shares",
        ),
        targets_ref=TargetsRef(table=TARGETS_TABLE, dt=dt_str, filename=f"targets_{choice_id}.json"),
        baseline=baseline,
    )

    # Reset quarantine counters when a new candidate is registered
    qc = state.quarantine or {}
    qc["status"] = "candidate_registered"
    qc["candidate_days"] = 0
    qc["candidate_points"] = 0
    qc["last_update"] = dt_str
    qc["last_decision"] = None
    state.candidate = candidate_slot
    state.as_of = dt_str
    state.quarantine = qc

    print("\n=== APPLY PORTFOLIO CANDIDATE (QUARANTINE) ===")
    print(f"Run key:    s3://{BUCKET}/{run_key}")
    print(f"Run id:     {run_id}")
    print(f"Apply dt:   {dt_str}")
    print(f"Choice id:  {choice_id}")
    print(f"Targets n:  {len(targets_payload['targets']['shares'])}")
    if state.active is not None:
        print(f"ACTIVE stays: choice_id={state.active.choice_id} as_of={state.active.as_of}")
    else:
        print("ACTIVE stays: <none> (no active portfolio yet)")
    print("")

    if dry_run:
        print("[DRY RUN] Would write:")
        print(f"  s3://{BUCKET}/{targets_key}")
        print(f"  s3://{BUCKET}/{choice_key}")
        print(f"  s3://{BUCKET}/{engine_key(CHOICE_STATE_TABLE,'latest.json')}")
        print("NOTE: targets/latest.json is NOT updated in quarantine.")
        return

    s3_put_json(s3, bucket=BUCKET, key=targets_key, payload=targets_payload)
    s3_put_json(s3, bucket=BUCKET, key=choice_key, payload=choice_payload)
    # do NOT touch targets/latest.json or choices/latest.json yet
    _save_choice_state(s3, dt_str=dt_str, state=state, update_latest=True)

    _append_choice_history(
        s3,
        dt_str=dt_str,
        event={
            "event_id": uuid.uuid4().hex[:10],
            "as_of": dt_str,
            "type": "candidate_registered",
            "candidate_choice_id": choice_id,
            "active_choice_id": (state.active.choice_id if state.active else None),
            "run_key": run_key,
            "run_id": run_id,
            "targets_key": targets_key,
        },
        update_latest=False,
    )

    print("[OK] Wrote candidate targets:")
    print(f"  s3://{BUCKET}/{targets_key}")
    print("[OK] Wrote candidate choice record:")
    print(f"  s3://{BUCKET}/{choice_key}")
    print("[OK] Updated choice state latest:")
    print(f"  s3://{BUCKET}/{engine_key(CHOICE_STATE_TABLE,'latest.json')}")
    print("")


def _should_reject(*, baseline_score: float | None, candidate_scores: list[float], cfg: dict) -> tuple[bool, str]:
    # need enough samples to reject
    min_samples = int(cfg.get("min_samples_to_decide", 3) or 3)
    if len(candidate_scores) < min_samples:
        return False, "not_enough_samples"

    # if baseline missing, reject only on absolute floor? for now: never reject without baseline
    if baseline_score is None or not pd.notna(baseline_score):
        return False, "baseline_missing"

    last = float(candidate_scores[-1])
    drop_abs = last - float(baseline_score)
    if drop_abs < -float(cfg.get("max_score_drop_abs", 0.08)):
        return True, f"score_drop_abs={drop_abs:.4f}"

    # frac check
    frac = last / float(baseline_score) if float(baseline_score) != 0 else 0.0
    if frac < (1.0 - float(cfg.get("max_score_drop_frac", 0.12))):
        return True, f"score_drop_frac={frac:.4f}"

    return False, "ok"


def _should_promote(*, candidate_days: int, cfg: dict, baseline_score: float | None, candidate_scores: list[float]) -> tuple[bool, str]:
    min_days = int(cfg.get("min_quarantine_days", 5) or 5)
    if candidate_days < min_days:
        return False, f"need_days={min_days} have={candidate_days}"

    # If we have baseline, require the last point not to be “too degraded”
    if baseline_score is not None and pd.notna(baseline_score) and candidate_scores:
        last = float(candidate_scores[-1])
        drop_abs = last - float(baseline_score)
        if drop_abs < -float(cfg.get("max_score_drop_abs", 0.08)):
            return False, "fails_on_promotion_score_abs"
        frac = last / float(baseline_score) if float(baseline_score) != 0 else 0.0
        if frac < (1.0 - float(cfg.get("max_score_drop_frac", 0.12))):
            return False, "fails_on_promotion_score_frac"

    return True, "ok"


def update_quarantine(
    *,
    as_of: str,
    candidate_score: float,
    extra_metrics: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Call this once per daily report run (after you compute score/health/etc) to:
      - record candidate metrics
      - decide reject / promote
    While in quarantine, ACTIVE targets remain in force.
    On promotion, targets/latest.json switches to candidate targets.

    Returns decision dict for logging.
    """
    s3 = s3_client(REGION)
    dt_str = str(pd.Timestamp(as_of).tz_localize(None).strftime("%Y-%m-%d"))

    state = _load_choice_state(s3)
    cfg = (state.quarantine.get("config") or {}) if state.quarantine else QuarantineConfig().to_dict()

    if state.candidate is None:
        return {"as_of": dt_str, "status": "no_candidate", "action": "none"}

    # record history point
    point = {
        "as_of": dt_str,
        "candidate_choice_id": state.candidate.choice_id,
        "active_choice_id": (state.active.choice_id if state.active else None),
        "candidate_score": float(candidate_score),
        "extra": dict(extra_metrics or {}),
    }

    # Load candidate history for this candidate (we keep it simple: store points as separate events)
    # We'll track counters in state; you can also query CHOICE_HISTORY_TABLE in Athena later.
    baseline_score = state.candidate.baseline.get("baseline_score")
    try:
        baseline_score = None if baseline_score is None else float(baseline_score)
    except Exception:
        baseline_score = None

    # Update counters
    qc = state.quarantine or {}
    qc["candidate_days"] = int(qc.get("candidate_days", 0) or 0) + 1
    qc["candidate_points"] = int(qc.get("candidate_points", 0) or 0) + 1
    qc["last_update"] = dt_str

    # For decisioning we only need last few scores; keep a tiny rolling list in state for speed.
    roll = qc.get("score_roll") or []
    if not isinstance(roll, list):
        roll = []
    roll.append(float(candidate_score))
    # keep last 20
    roll = roll[-20:]
    qc["score_roll"] = roll

    # Decide reject?
    reject, reject_reason = _should_reject(
        baseline_score=baseline_score,
        candidate_scores=roll,
        cfg=cfg,
    )

    # Decide promote?
    promote, promote_reason = _should_promote(
        candidate_days=int(qc.get("candidate_days", 0) or 0),
        cfg=cfg,
        baseline_score=baseline_score,
        candidate_scores=roll,
    )

    decision = {
        "as_of": dt_str,
        "candidate_choice_id": state.candidate.choice_id,
        "active_choice_id": (state.active.choice_id if state.active else None),
        "candidate_days": int(qc.get("candidate_days", 0) or 0),
        "candidate_score": float(candidate_score),
        "baseline_score": baseline_score,
        "action": "none",
        "reason": None,
    }

    # precedence: reject beats promote (fail-fast)
    if reject:
        decision["action"] = "reject"
        decision["reason"] = reject_reason
        qc["status"] = "candidate_rejected"
        qc["last_decision"] = {"as_of": dt_str, "action": "reject", "reason": reject_reason}

        # record history
        event = {
            "event_id": uuid.uuid4().hex[:10],
            "as_of": dt_str,
            "type": "candidate_rejected",
            "candidate_choice_id": state.candidate.choice_id,
            "active_choice_id": (state.active.choice_id if state.active else None),
            "reason": reject_reason,
            "point": point,
            "config": cfg,
        }

        if dry_run:
            return {**decision, "dry_run": True}

        _append_choice_history(s3, dt_str=dt_str, event=event, update_latest=False)

        # drop candidate; keep active unchanged
        state.candidate = None
        state.as_of = dt_str
        state.quarantine = qc
        _save_choice_state(s3, dt_str=dt_str, state=state, update_latest=True)

        return decision

    if promote:
        decision["action"] = "promote"
        decision["reason"] = promote_reason
        qc["status"] = "candidate_promoted"
        qc["last_decision"] = {"as_of": dt_str, "action": "promote", "reason": promote_reason}

        # promotion event
        event = {
            "event_id": uuid.uuid4().hex[:10],
            "as_of": dt_str,
            "type": "candidate_promoted",
            "candidate_choice_id": state.candidate.choice_id,
            "previous_active_choice_id": (state.active.choice_id if state.active else None),
            "new_active_choice_id": state.candidate.choice_id,
            "reason": promote_reason,
            "point": point,
            "config": cfg,
            "targets_key": state.candidate.targets_ref.to_key(),
        }

        # Build "ACTIVE choice record" and update latest pointers
        new_choice_id = state.candidate.choice_id
        new_targets_key = state.candidate.targets_ref.to_key()

        # A "choice" record for active promotion (audit)
        choice_payload = {
            "choice_id": new_choice_id,
            "as_of": dt_str,
            "status": "ACTIVE",
            "picked_from": asdict(state.candidate.picked_from),
            "targets_ref": asdict(state.candidate.targets_ref),
            "note": "Candidate passed quarantine and is now ACTIVE. targets/latest.json updated.",
        }

        choice_key = dt_key(CHOICES_TABLE, dt_str, f"choice_{new_choice_id}_ACTIVE.json")
        choice_latest_key = engine_key(CHOICES_TABLE, "latest.json")

        targets_latest_key = engine_key(TARGETS_TABLE, "latest.json")

        if dry_run:
            return {
                **decision,
                "dry_run": True,
                "would_update": {
                    "targets_latest": f"s3://{BUCKET}/{targets_latest_key} -> {new_targets_key}",
                    "choices_latest": f"s3://{BUCKET}/{choice_latest_key} -> {choice_key}",
                },
            }

        # Write promotion artifacts
        _append_choice_history(s3, dt_str=dt_str, event=event, update_latest=False)
        s3_put_json(s3, bucket=BUCKET, key=choice_key, payload=choice_payload)
        s3_put_json(s3, bucket=BUCKET, key=choice_latest_key, payload=choice_payload)

        # IMPORTANT: now switch execution targets
        targets_payload = s3_get_json(s3, bucket=BUCKET, key=new_targets_key) or {}
        s3_put_json(s3, bucket=BUCKET, key=targets_latest_key, payload=targets_payload)

        # Update state: candidate becomes active
        state.active = state.candidate
        state.candidate = None
        state.as_of = dt_str
        state.quarantine = qc
        _save_choice_state(s3, dt_str=dt_str, state=state, update_latest=True)

        return decision

    # no decision: just store point + updated state counters
    qc["status"] = "candidate_monitoring"
    qc["last_decision"] = {"as_of": dt_str, "action": "monitor", "reason": None}
    state.quarantine = qc
    state.as_of = dt_str

    event = {
        "event_id": uuid.uuid4().hex[:10],
        "as_of": dt_str,
        "type": "candidate_daily_point",
        "candidate_choice_id": state.candidate.choice_id,
        "active_choice_id": (state.active.choice_id if state.active else None),
        "point": point,
        "config": cfg,
    }

    if dry_run:
        return {**decision, "action": "monitor", "dry_run": True}

    _append_choice_history(s3, dt_str=dt_str, event=event, update_latest=False)
    _save_choice_state(s3, dt_str=dt_str, state=state, update_latest=True)

    decision["action"] = "monitor"
    decision["reason"] = None
    return decision


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_c = sub.add_parser("apply-candidate")
    ap_c.add_argument("--dry-run", action="store_true")

    ap_u = sub.add_parser("update-quarantine")
    ap_u.add_argument("--as-of", required=False, default=None)
    ap_u.add_argument("--candidate-score", required=True, type=float)
    ap_u.add_argument("--dry-run", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()

    if args.cmd == "apply-candidate":
        apply_candidate_from_latest_search(dry_run=bool(args.dry_run))
        return

    if args.cmd == "update-quarantine":
        as_of = args.as_of or pd.Timestamp(dt.date.today()).strftime("%Y-%m-%d")
        decision = update_quarantine(
            as_of=as_of,
            candidate_score=float(args.candidate_score),
            extra_metrics=None,
            dry_run=bool(args.dry_run),
        )
        print(json.dumps(decision, indent=2))
        return


if __name__ == "__main__":
    main()
