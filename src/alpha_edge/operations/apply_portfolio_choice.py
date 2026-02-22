from __future__ import annotations

import argparse
import datetime as dt
import json
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.core.data_loader import s3_get_json

BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"

CHOICES_TABLE = "portfolio_choices"
TARGETS_TABLE = "targets"

# Audit state/history (promotion-only use)
CHOICE_STATE_TABLE = "portfolio_choice_state"
CHOICE_HISTORY_TABLE = "portfolio_choice_history"

# Quarantine outputs (canonical source for approval)
QUAR_SUMMARY_TABLE = "quarantine/summary"
QUAR_CAND_TABLE = "quarantine/candidates"


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


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def dt_key(table: str, dt_str: str, filename: str) -> str:
    return engine_key(table, f"dt={dt_str}", filename)


def _candidate_latest_key(candidate_id: str) -> str:
    cid = str(candidate_id).strip()
    return engine_key(QUAR_CAND_TABLE, f"candidate_id={cid}", "latest.json")


def _s3_get_json_or_none(s3, *, bucket: str, key: str) -> dict | None:
    """
    Safe JSON getter: returns None if missing (NoSuchKey/404), raises otherwise.
    """
    try:
        return s3_get_json(s3, bucket=bucket, key=key)
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


# -------------------------
# Choice state schema (promotion audit)
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
    quarantine_summary_key: str
    quarantine_candidate_key: str
    candidate_id: str
    variant: str


@dataclass
class PortfolioSlot:
    choice_id: str
    as_of: str
    picked_from: PickedFrom
    targets_ref: TargetsRef
    baseline: Dict[str, Any]


@dataclass
class ChoiceState:
    as_of: str
    active: Optional[PortfolioSlot]

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "active": None if self.active is None else asdict(self.active),
        }


def _load_choice_state(s3) -> ChoiceState:
    """
    Bootstrap-safe: if latest.json doesn't exist yet, start with empty state.
    """
    key = engine_key(CHOICE_STATE_TABLE, "latest.json")

    raw = _s3_get_json_or_none(s3, bucket=BUCKET, key=key) or {}
    if not isinstance(raw, dict):
        raw = {}

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
                    quarantine_summary_key=str(pf.get("quarantine_summary_key")),
                    quarantine_candidate_key=str(pf.get("quarantine_candidate_key")),
                    candidate_id=str(pf.get("candidate_id")),
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
    )


def _save_choice_state(s3, *, dt_str: str, state: ChoiceState, update_latest: bool = True) -> None:
    payload = state.to_dict()
    key = dt_key(CHOICE_STATE_TABLE, dt_str, "state.json")
    s3_put_json(s3, bucket=BUCKET, key=key, payload=payload)
    if update_latest:
        s3_put_json(s3, bucket=BUCKET, key=engine_key(CHOICE_STATE_TABLE, "latest.json"), payload=payload)


def _append_choice_history(s3, *, dt_str: str, event: dict) -> None:
    eid = event.get("event_id") or uuid.uuid4().hex[:10]
    key = dt_key(CHOICE_HISTORY_TABLE, dt_str, f"event_{dt_str}_{eid}.json")
    s3_put_json(s3, bucket=BUCKET, key=key, payload=event)


# -------------------------
# Promotion (ONLY responsibility of this file)
# -------------------------
def _pick_approved_candidate_id(summary: dict, preferred: str | None) -> str:
    if preferred:
        return str(preferred).strip()

    approved = summary.get("approved")
    if isinstance(approved, list) and approved:
        return str(approved[0]).strip()

    raise RuntimeError("No approved candidates found in quarantine summary (approved list empty).")


def _extract_shares_from_candidate_state(cand_state: dict) -> dict[str, float]:
    shares = cand_state.get("shares")
    if not isinstance(shares, dict) or not shares:
        raise RuntimeError("quarantine candidate latest.json missing 'shares' dict")

    out: dict[str, float] = {}
    for t, q in shares.items():
        tt = str(t).upper().strip()
        try:
            qq = float(q)
        except Exception:
            continue
        if not tt or pd.isna(qq) or qq == 0.0:
            continue
        out[tt] = float(qq)

    if len(out) < 2:
        raise RuntimeError("quarantine candidate has <2 non-zero shares; refusing to promote")

    return out


def promote_approved(
    *,
    candidate_id: str | None = None,
    as_of: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    PROMOTION ONLY:
      - reads quarantine/summary/latest.json
      - reads quarantine/candidates/candidate_id=<cid>/latest.json
      - writes targets/latest.json (activates)
      - writes audit records + choice_state active slot
    """
    s3 = s3_client(REGION)

    dt_str = str(pd.Timestamp(as_of or dt.date.today()).tz_localize(None).strftime("%Y-%m-%d"))

    summary_key = engine_key(QUAR_SUMMARY_TABLE, "latest.json")
    summary = _s3_get_json_or_none(s3, bucket=BUCKET, key=summary_key) or {}
    if not isinstance(summary, dict):
        raise RuntimeError(f"Invalid quarantine summary at s3://{BUCKET}/{summary_key}")

    cid = _pick_approved_candidate_id(summary, candidate_id)

    cand_key = _candidate_latest_key(cid)
    cand_state = _s3_get_json_or_none(s3, bucket=BUCKET, key=cand_key) or {}
    if not isinstance(cand_state, dict) or not cand_state:
        raise RuntimeError(f"Candidate state not found at s3://{BUCKET}/{cand_key}")

    q = cand_state.get("quarantine") or {}
    if isinstance(q, dict):
        status = str(q.get("status") or "").upper()
        if status != "APPROVED":
            raise RuntimeError(f"Candidate {cid} is not APPROVED (status={status}). Refusing to promote.")

    shares = _extract_shares_from_candidate_state(cand_state)

    choice_id = f"{dt_str}-Q-{uuid.uuid4().hex[:8]}"

    targets_payload = {
        "as_of": dt_str,
        "choice_id": choice_id,
        "mode": "ACTIVE_FROM_QUARANTINE",
        "source": {
            "quarantine_summary_key": summary_key,
            "quarantine_candidate_key": cand_key,
            "candidate_id": cid,
            "variant": "quarantine/candidates/.../shares",
        },
        "targets": {"shares": {t: float(qty) for t, qty in shares.items()}},
    }

    targets_key = dt_key(TARGETS_TABLE, dt_str, f"targets_{choice_id}.json")
    targets_latest_key = engine_key(TARGETS_TABLE, "latest.json")

    choice_payload = {
        "choice_id": choice_id,
        "as_of": dt_str,
        "status": "ACTIVE",
        "picked_from": {
            "quarantine_summary_key": summary_key,
            "quarantine_candidate_key": cand_key,
            "candidate_id": cid,
            "variant": "quarantine/approved_candidate",
        },
        "targets_ref": {"table": TARGETS_TABLE, "dt": dt_str, "filename": f"targets_{choice_id}.json"},
        "note": "Promoted APPROVED quarantine candidate to ACTIVE. targets/latest.json updated.",
    }

    choice_key = dt_key(CHOICES_TABLE, dt_str, f"choice_{choice_id}_ACTIVE.json")
    choice_latest_key = engine_key(CHOICES_TABLE, "latest.json")

    # update audit choice_state (bootstrap-safe now)
    state = _load_choice_state(s3)
    active_slot = PortfolioSlot(
        choice_id=choice_id,
        as_of=dt_str,
        picked_from=PickedFrom(
            quarantine_summary_key=summary_key,
            quarantine_candidate_key=cand_key,
            candidate_id=cid,
            variant="quarantine/approved_candidate",
        ),
        targets_ref=TargetsRef(table=TARGETS_TABLE, dt=dt_str, filename=f"targets_{choice_id}.json"),
        baseline={
            "quarantine_as_of": summary.get("as_of"),
            "promoted_at": dt_str,
            "candidate_status": (q.get("status") if isinstance(q, dict) else None),
            "baseline_eval": (q.get("baseline_eval") if isinstance(q, dict) else None),
            "degradation": (q.get("degradation") if isinstance(q, dict) else None),
        },
    )
    state.active = active_slot
    state.as_of = dt_str

    print("\n=== PROMOTE APPROVED QUARANTINE CANDIDATE ===")
    print(f"Apply dt:         {dt_str}")
    print(f"Candidate id:     {cid}")
    print(f"Summary key:      s3://{BUCKET}/{summary_key}")
    print(f"Candidate key:    s3://{BUCKET}/{cand_key}")
    print(f"Choice id:        {choice_id}")
    print(f"Targets n:        {len(shares)}")
    print("")

    if dry_run:
        print("[DRY RUN] Would write:")
        print(f"  s3://{BUCKET}/{targets_key}")
        print(f"  s3://{BUCKET}/{targets_latest_key}  (activate)")
        print(f"  s3://{BUCKET}/{choice_key}")
        print(f"  s3://{BUCKET}/{choice_latest_key}")
        print(f"  s3://{BUCKET}/{engine_key(CHOICE_STATE_TABLE,'latest.json')}")
        print(f"  s3://{BUCKET}/{engine_key(CHOICE_HISTORY_TABLE,'dt=...')}/event_...json")
        return

    # Write targets + activate
    s3_put_json(s3, bucket=BUCKET, key=targets_key, payload=targets_payload)
    s3_put_json(s3, bucket=BUCKET, key=targets_latest_key, payload=targets_payload)

    # Write choice record + latest pointer
    s3_put_json(s3, bucket=BUCKET, key=choice_key, payload=choice_payload)
    s3_put_json(s3, bucket=BUCKET, key=choice_latest_key, payload=choice_payload)

    # Update state + history
    _save_choice_state(s3, dt_str=dt_str, state=state, update_latest=True)

    _append_choice_history(
        s3,
        dt_str=dt_str,
        event={
            "event_id": uuid.uuid4().hex[:10],
            "as_of": dt_str,
            "type": "promoted_from_quarantine",
            "choice_id": choice_id,
            "candidate_id": cid,
            "quarantine_summary_key": summary_key,
            "quarantine_candidate_key": cand_key,
            "targets_key": targets_key,
            "targets_latest_key": targets_latest_key,
            "choice_key": choice_key,
        },
    )

    print("[OK] Activated targets/latest.json.")
    print(f"  s3://{BUCKET}/{targets_latest_key}")
    print("")


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_p = sub.add_parser("promote-approved")
    ap_p.add_argument("--candidate-id", type=str, default=None)
    ap_p.add_argument("--as-of", type=str, default=None)
    ap_p.add_argument("--dry-run", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()

    if args.cmd == "promote-approved":
        promote_approved(
            candidate_id=(args.candidate_id if args.candidate_id else None),
            as_of=(args.as_of if args.as_of else None),
            dry_run=bool(args.dry_run),
        )
        return


if __name__ == "__main__":
    main()
