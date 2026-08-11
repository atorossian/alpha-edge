from __future__ import annotations

import argparse
import datetime as dt
import json
import uuid
from typing import Any, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT = "engine/v1"

CHOICES_TABLE = "portfolio_choices"
TARGETS_TABLE = "targets"

CHOICE_STATE_TABLE = "portfolio_choice_state"
CHOICE_HISTORY_TABLE = "portfolio_choice_history"

QUAR_SUMMARY_TABLE = "quarantine/summary"
QUAR_CAND_TABLE = "quarantine/candidates"


# -------------------------
# Runtime helpers
# -------------------------
def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET))


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION))


def cfg_engine_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "engine_root", DEFAULT_ENGINE_ROOT)).strip("/")


def cfg_env(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "env", "dev"))


# -------------------------
# S3 helpers
# -------------------------
def s3_client(region: str):
    return boto3.client("s3", region_name=region)


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def s3_get_json(s3, *, bucket: str, key: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def engine_key(cfg: RuntimeConfig, *parts: str) -> str:
    return "/".join([cfg_engine_root(cfg)] + [p.strip("/") for p in parts])


def dt_key(cfg: RuntimeConfig, table: str, dt_str: str, filename: str) -> str:
    return engine_key(cfg, table, f"dt={dt_str}", filename)


def _candidate_latest_key(cfg: RuntimeConfig, candidate_id: str) -> str:
    cid = str(candidate_id).strip()
    return engine_key(cfg, QUAR_CAND_TABLE, f"candidate_id={cid}", "latest.json")


def _s3_get_json_or_none(s3, *, bucket: str, key: str) -> dict | None:
    try:
        return s3_get_json(s3, bucket=bucket, key=key)
    except ClientError as e:
        code = (e.response.get("Error") or {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


# -------------------------
# Choice state helpers
# -------------------------
def _targets_ref_to_key(cfg: RuntimeConfig, targets_ref: dict[str, Any]) -> str:
    return dt_key(
        cfg,
        str(targets_ref["table"]),
        str(targets_ref["dt"]),
        str(targets_ref["filename"]),
    )


def _parse_choice_slot(x: dict | None) -> Optional[dict[str, Any]]:
    if not isinstance(x, dict):
        return None

    try:
        pf = x.get("picked_from") or {}
        tr = x.get("targets_ref") or {}

        return {
            "choice_id": str(x.get("choice_id")),
            "as_of": str(x.get("as_of")),
            "picked_from": {
                "quarantine_summary_key": str(pf.get("quarantine_summary_key")),
                "quarantine_candidate_key": str(pf.get("quarantine_candidate_key")),
                "candidate_id": str(pf.get("candidate_id")),
                "variant": str(pf.get("variant")),
            },
            "targets_ref": {
                "table": str(tr.get("table")),
                "dt": str(tr.get("dt")),
                "filename": str(tr.get("filename")),
            },
            "baseline": dict(x.get("baseline") or {}),
        }
    except Exception:
        return None


def _load_choice_state(s3, *, cfg: RuntimeConfig) -> dict[str, Any]:
    bucket = cfg_bucket(cfg)
    key = engine_key(cfg, CHOICE_STATE_TABLE, "latest.json")

    raw = _s3_get_json_or_none(s3, bucket=bucket, key=key) or {}
    if not isinstance(raw, dict):
        raw = {}

    return {
        "as_of": str(raw.get("as_of") or pd.Timestamp(dt.date.today()).strftime("%Y-%m-%d")),
        "active": _parse_choice_slot(raw.get("active")),
    }


def _save_choice_state(
    s3,
    *,
    cfg: RuntimeConfig,
    dt_str: str,
    state: dict[str, Any],
    update_latest: bool = True,
) -> None:
    bucket = cfg_bucket(cfg)

    key = dt_key(cfg, CHOICE_STATE_TABLE, dt_str, "state.json")
    s3_put_json(s3, bucket=bucket, key=key, payload=state)

    if update_latest:
        latest_key = engine_key(cfg, CHOICE_STATE_TABLE, "latest.json")
        s3_put_json(s3, bucket=bucket, key=latest_key, payload=state)


def _append_choice_history(
    s3,
    *,
    cfg: RuntimeConfig,
    dt_str: str,
    event: dict[str, Any],
) -> None:
    bucket = cfg_bucket(cfg)

    eid = event.get("event_id") or uuid.uuid4().hex[:10]
    key = dt_key(cfg, CHOICE_HISTORY_TABLE, dt_str, f"event_{dt_str}_{eid}.json")
    s3_put_json(s3, bucket=bucket, key=key, payload=event)


# -------------------------
# Promotion
# -------------------------
def _pick_approved_candidate_id(summary: dict, preferred: str | None) -> str:
    if preferred:
        cid = str(preferred).strip()
        if not cid:
            raise RuntimeError("--candidate-id was provided but empty.")
        return cid

    approved = summary.get("approved")
    if isinstance(approved, list) and approved:
        cid = str(approved[0]).strip()
        if cid:
            return cid

    raise RuntimeError("No approved candidates found in quarantine summary; approved list is empty.")


def _extract_shares_from_candidate_state(cand_state: dict) -> dict[str, float]:
    shares = cand_state.get("shares")
    if not isinstance(shares, dict) or not shares:
        raise RuntimeError("quarantine candidate latest.json missing non-empty 'shares' dict.")

    out: dict[str, float] = {}

    for t, q in shares.items():
        ticker = str(t).upper().strip()
        try:
            qty = float(q)
        except Exception:
            continue

        if not ticker or pd.isna(qty) or qty == 0.0:
            continue

        out[ticker] = float(qty)

    if len(out) < 2:
        raise RuntimeError("quarantine candidate has <2 non-zero shares; refusing to promote.")

    return out


def promote_approved(
    *,
    cfg: RuntimeConfig,
    candidate_id: str | None = None,
    as_of: str | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
    reason: str | None = None,
    input_args: dict[str, Any] | None = None,
) -> None:
    """
    PROMOTION ONLY:
      - reads quarantine/summary/latest.json
      - reads quarantine/candidates/candidate_id=<cid>/latest.json
      - writes targets/latest.json
      - writes choice audit records and active choice state
    """
    bucket = cfg_bucket(cfg)
    s3 = s3_client(cfg_region(cfg))

    dt_str = str(pd.Timestamp(as_of or dt.date.today()).tz_localize(None).strftime("%Y-%m-%d"))

    summary_key = engine_key(cfg, QUAR_SUMMARY_TABLE, "latest.json")
    summary = _s3_get_json_or_none(s3, bucket=bucket, key=summary_key) or {}
    if not isinstance(summary, dict):
        raise RuntimeError(f"Invalid quarantine summary at s3://{bucket}/{summary_key}")

    cid = _pick_approved_candidate_id(summary, candidate_id)

    cand_key = _candidate_latest_key(cfg, cid)
    cand_state = _s3_get_json_or_none(s3, bucket=bucket, key=cand_key) or {}
    if not isinstance(cand_state, dict) or not cand_state:
        raise RuntimeError(f"Candidate state not found at s3://{bucket}/{cand_key}")

    quarantine_meta = cand_state.get("quarantine") or {}
    if isinstance(quarantine_meta, dict):
        status = str(quarantine_meta.get("status") or "").upper()
        if status != "APPROVED":
            raise RuntimeError(f"Candidate {cid} is not APPROVED; status={status!r}. Refusing to promote.")

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
        "targets": {
            "shares": {ticker: float(qty) for ticker, qty in shares.items()},
        },
    }

    targets_filename = f"targets_{choice_id}.json"
    targets_key = dt_key(cfg, TARGETS_TABLE, dt_str, targets_filename)
    targets_latest_key = engine_key(cfg, TARGETS_TABLE, "latest.json")

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
        "targets_ref": {
            "table": TARGETS_TABLE,
            "dt": dt_str,
            "filename": targets_filename,
        },
        "note": "Promoted APPROVED quarantine candidate to ACTIVE. targets/latest.json updated.",
    }

    choice_key = dt_key(cfg, CHOICES_TABLE, dt_str, f"choice_{choice_id}_ACTIVE.json")
    choice_latest_key = engine_key(cfg, CHOICES_TABLE, "latest.json")

    before_state = _load_choice_state(s3, cfg=cfg)
    state = dict(before_state)
    state["as_of"] = dt_str
    state["active"] = {
        "choice_id": choice_id,
        "as_of": dt_str,
        "picked_from": {
            "quarantine_summary_key": summary_key,
            "quarantine_candidate_key": cand_key,
            "candidate_id": cid,
            "variant": "quarantine/approved_candidate",
        },
        "targets_ref": {
            "table": TARGETS_TABLE,
            "dt": dt_str,
            "filename": targets_filename,
        },
        "baseline": {
            "quarantine_as_of": summary.get("as_of"),
            "promoted_at": dt_str,
            "candidate_status": quarantine_meta.get("status") if isinstance(quarantine_meta, dict) else None,
            "baseline_eval": quarantine_meta.get("baseline_eval") if isinstance(quarantine_meta, dict) else None,
            "degradation": quarantine_meta.get("degradation") if isinstance(quarantine_meta, dict) else None,
        },
    }

    print("\n=== PROMOTE APPROVED QUARANTINE CANDIDATE ===")
    print(f"env:              {cfg_env(cfg)}")
    print(f"bucket:           {bucket}")
    print(f"engine_root:      {cfg_engine_root(cfg)}")
    print(f"apply dt:         {dt_str}")
    print(f"candidate id:     {cid}")
    print(f"summary key:      s3://{bucket}/{summary_key}")
    print(f"candidate key:    s3://{bucket}/{cand_key}")
    print(f"choice id:        {choice_id}")
    print(f"targets n:        {len(shares)}")
    print("")

    output_keys = [
        targets_key,
        targets_latest_key,
        choice_key,
        choice_latest_key,
        dt_key(cfg, CHOICE_STATE_TABLE, dt_str, "state.json"),
        engine_key(cfg, CHOICE_STATE_TABLE, "latest.json"),
        engine_key(cfg, CHOICE_HISTORY_TABLE, f"dt={dt_str}"),
    ]

    if dry_run:
        print("[DRY RUN] Would write:")
        for out_key in output_keys:
            print(f"  s3://{bucket}/{out_key}")

        audit_event = build_audit_event(
            cfg=cfg,
            run_id=run_id,
            event_type="promote",
            entity_type="portfolio_choice",
            entity_id=choice_id,
            as_of=dt_str,
            source_script="apply_portfolio_choice.py",
            source_mode="promote-approved",
            status="dry_run",
            reason=reason,
            input_args=input_args or {},
            output_keys=output_keys,
            before_payload=before_state,
            after_payload={
                "targets": targets_payload,
                "choice": choice_payload,
                "state": state,
            },
            metadata={
                "candidate_id": cid,
                "targets_n": len(shares),
                "summary_key": summary_key,
                "candidate_key": cand_key,
            },
        )
        write_audit_event(cfg=cfg, event=audit_event, dry_run=True)
        return

    s3_put_json(s3, bucket=bucket, key=targets_key, payload=targets_payload)
    s3_put_json(s3, bucket=bucket, key=targets_latest_key, payload=targets_payload)

    s3_put_json(s3, bucket=bucket, key=choice_key, payload=choice_payload)
    s3_put_json(s3, bucket=bucket, key=choice_latest_key, payload=choice_payload)

    _save_choice_state(s3, cfg=cfg, dt_str=dt_str, state=state, update_latest=True)

    _append_choice_history(
        s3,
        cfg=cfg,
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

    audit_event = build_audit_event(
        cfg=cfg,
        run_id=run_id,
        event_type="promote",
        entity_type="portfolio_choice",
        entity_id=choice_id,
        as_of=dt_str,
        source_script="apply_portfolio_choice.py",
        source_mode="promote-approved",
        status="success",
        reason=reason,
        input_args=input_args or {},
        output_keys=output_keys,
        before_payload=before_state,
        after_payload={
            "targets": targets_payload,
            "choice": choice_payload,
            "state": state,
        },
        metadata={
            "candidate_id": cid,
            "targets_n": len(shares),
            "summary_key": summary_key,
            "candidate_key": cand_key,
        },
    )
    write_audit_event(cfg=cfg, event=audit_event, dry_run=False)

    print("[OK] Activated targets/latest.json.")
    print(f"  s3://{bucket}/{targets_latest_key}")
    print("")


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Portfolio choice promotion tools.")
    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_p = sub.add_parser("promote-approved")
    ap_p.add_argument("--candidate-id", type=str, default=None)
    ap_p.add_argument("--as-of", type=str, default=None)
    ap_p.add_argument("--dry-run", action="store_true")
    ap_p.add_argument("--reason", default=None, help="Optional reason stored in the audit event.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(args.env)

    if args.cmd == "promote-approved":
        if not bool(args.dry_run):
            require_prod_confirmation(cfg, bool(args.confirm_prod_write))

        with capture_script_run(
            cfg=cfg,
            script_name="apply_portfolio_choice.py",
            input_args=vars(args),
            dry_run=bool(args.dry_run),
        ) as run_id:
            promote_approved(
                cfg=cfg,
                candidate_id=(args.candidate_id if args.candidate_id else None),
                as_of=(args.as_of if args.as_of else None),
                dry_run=bool(args.dry_run),
                run_id=run_id,
                reason=(args.reason if args.reason else None),
                input_args=vars(args),
            )
        return


if __name__ == "__main__":
    main()