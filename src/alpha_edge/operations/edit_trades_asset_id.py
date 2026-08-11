from __future__ import annotations

import argparse
import json
from typing import List, Optional, Tuple

import boto3
import pandas as pd

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import UniverseIndex


TRADES_TABLE = "trades"


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


def engine_key(cfg: RuntimeConfig, *parts: str) -> str:
    return "/".join([cfg.engine_root.strip("/")] + [p.strip("/") for p in parts])


def audit_prefix(cfg: RuntimeConfig) -> str:
    return engine_key(cfg, "trades_audit")


def s3_list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)

        for it in resp.get("Contents", []):
            keys.append(it["Key"])

        if not resp.get("IsTruncated"):
            break

        token = resp.get("NextContinuationToken")

    return keys


def s3_get_json(s3, *, bucket: str, key: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return json.loads(body.decode("utf-8"))


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def s3_copy_object(s3, *, bucket: str, src_key: str, dst_key: str) -> None:
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        ContentType="application/json",
        MetadataDirective="COPY",
    )


# ----------------------------
# Universe mapping
# ----------------------------
def load_universe_index(universe_path: str) -> UniverseIndex:
    df = pd.read_csv(universe_path)

    if "asset_id" not in df.columns or "broker_ticker" not in df.columns:
        raise RuntimeError("universe.csv must contain columns: asset_id, broker_ticker")

    df = df.copy()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["broker_ticker"] = df["broker_ticker"].astype(str).str.upper().str.strip()

    m: dict[str, list[dict]] = {}

    for _, r in df.iterrows():
        bt = str(r["broker_ticker"])
        rec = {
            "asset_id": str(r["asset_id"]),
            "row_id": r.get("row_id") if "row_id" in df.columns else None,
            "yahoo_ticker": r.get("yahoo_ticker") if "yahoo_ticker" in df.columns else None,
            "name": r.get("name") if "name" in df.columns else None,
            "include": r.get("include") if "include" in df.columns else None,
        }
        m.setdefault(bt, []).append(rec)

    return UniverseIndex(by_broker_ticker=m)


def candidates_for_ticker(idx: UniverseIndex, ticker: str) -> List[dict]:
    t = str(ticker).upper().strip()
    return idx.by_broker_ticker.get(t, [])


# ----------------------------
# Trade scanning
# ----------------------------
def list_trade_keys_for_dt(s3, *, cfg: RuntimeConfig, dt: str) -> list[str]:
    prefix = engine_key(cfg, TRADES_TABLE, f"dt={dt}")
    keys = s3_list_keys(s3, bucket=cfg.bucket, prefix=prefix)
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]
    return sorted(keys)


def list_trade_keys_all(s3, *, cfg: RuntimeConfig) -> list[str]:
    prefix = engine_key(cfg, TRADES_TABLE, "dt=")
    keys = s3_list_keys(s3, bucket=cfg.bucket, prefix=prefix)
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]
    return sorted(keys)


def is_missing_or_ambiguous_asset_id(trade: dict, idx: UniverseIndex) -> Tuple[bool, List[dict]]:
    tkr = str(trade.get("ticker") or "").upper().strip()
    if not tkr:
        return True, []

    cands = candidates_for_ticker(idx, tkr)

    if not cands:
        return True, []

    cur = trade.get("asset_id", None)
    cur = None if cur is None else str(cur).strip()

    if len(cands) > 1:
        return True, cands

    if len(cands) == 1 and not cur:
        return True, cands

    if len(cands) == 1 and cur and cur != str(cands[0]["asset_id"]):
        return True, cands

    return False, cands


# ----------------------------
# Plan + Apply
# ----------------------------
def build_plan_csv(
    *,
    cfg: RuntimeConfig,
    universe_path: str,
    out_csv: str,
    dt: Optional[str],
) -> dict:
    s3 = s3_client(cfg)
    idx = load_universe_index(universe_path)

    keys = list_trade_keys_for_dt(s3, cfg=cfg, dt=dt) if dt else list_trade_keys_all(s3, cfg=cfg)

    rows: list[dict] = []

    for k in keys:
        tr = s3_get_json(s3, bucket=cfg.bucket, key=k)
        if not isinstance(tr, dict):
            continue

        flag, cands = is_missing_or_ambiguous_asset_id(tr, idx)
        if not flag:
            continue

        tkr = str(tr.get("ticker") or "").upper().strip()
        cur = tr.get("asset_id", None)
        cur = None if cur is None else str(cur).strip()

        rows.append(
            {
                "s3_key": k,
                "trade_id": tr.get("trade_id"),
                "as_of": tr.get("as_of"),
                "ts_utc": tr.get("ts_utc"),
                "ticker": tkr,
                "side": tr.get("side"),
                "quantity": tr.get("quantity"),
                "price": tr.get("price"),
                "currency": tr.get("currency"),
                "current_asset_id": cur,
                "candidates_asset_id": "|".join([str(x["asset_id"]) for x in cands]) if cands else "",
                "candidates_yahoo_ticker": "|".join([str(x.get("yahoo_ticker") or "") for x in cands]) if cands else "",
                "candidates_name": "|".join([str(x.get("name") or "") for x in cands]) if cands else "",
                "resolution_asset_id": "",
                "resolution_note": "",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"[OK] wrote plan: {out_csv} rows={len(df)}")
    if len(df) > 0:
        print("Next: fill resolution_asset_id, then run with --mode apply")

    return {
        "rows_scanned": len(keys),
        "rows_in_plan": int(len(df)),
        "out_csv": str(out_csv),
        "dt": dt,
    }


def apply_plan_csv(
    *,
    cfg: RuntimeConfig,
    plan_csv: str,
    dry_run: bool,
    write_backups: bool,
    run_id: str,
    reason: Optional[str],
    input_args: dict,
) -> dict:
    s3 = s3_client(cfg)
    df = pd.read_csv(plan_csv)

    required = {"s3_key", "resolution_asset_id"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Plan CSV missing required columns: {missing}")

    n_total = 0
    n_updated = 0
    n_skipped = 0

    for _, r in df.iterrows():
        n_total += 1

        key = str(r["s3_key"]).strip()
        new_aid = r.get("resolution_asset_id", None)
        new_aid = None if pd.isna(new_aid) else str(new_aid).strip()

        if not key or not new_aid:
            n_skipped += 1
            continue

        tr = s3_get_json(s3, bucket=cfg.bucket, key=key)
        if not isinstance(tr, dict):
            n_skipped += 1
            continue

        old_aid = tr.get("asset_id", None)
        old_aid = None if old_aid is None else str(old_aid).strip()

        if old_aid == new_aid:
            n_skipped += 1
            continue

        before_payload = dict(tr)
        backup_key = None

        if write_backups:
            backup_key = f"{audit_prefix(cfg)}/{key.replace('/', '__')}"
            if dry_run:
                print(f"[DRY RUN] backup copy s3://{cfg.bucket}/{key} -> s3://{cfg.bucket}/{backup_key}")
            else:
                s3_copy_object(s3, bucket=cfg.bucket, src_key=key, dst_key=backup_key)

        tr["asset_id"] = new_aid
        after_payload = dict(tr)

        if dry_run:
            print(f"[DRY RUN] update {key}: asset_id {old_aid} -> {new_aid}")
        else:
            s3_put_json(s3, bucket=cfg.bucket, key=key, payload=tr)
            print(f"[OK] updated {key}: asset_id {old_aid} -> {new_aid}")

        audit_event = build_audit_event(
            cfg=cfg,
            run_id=run_id,
            event_type="modify",
            entity_type="trade",
            entity_id=tr.get("trade_id"),
            as_of=tr.get("as_of"),
            source_script="edit_trades_asset_id.py",
            source_mode="apply",
            status=("dry_run" if dry_run else "success"),
            reason=reason,
            input_args=input_args,
            output_keys=[key],
            backup_keys=[backup_key] if backup_key else [],
            before_payload=before_payload,
            after_payload=after_payload,
            metadata={
                "plan_csv": plan_csv,
                "field": "asset_id",
                "old_asset_id": old_aid,
                "new_asset_id": new_aid,
            },
        )
        write_audit_event(cfg=cfg, event=audit_event, dry_run=dry_run)

        n_updated += 1

    print("")
    print("=== APPLY SUMMARY ===")
    print(f"env:          {cfg.env}")
    print(f"bucket:       {cfg.bucket}")
    print(f"engine_root:  {cfg.engine_root}")
    print(f"rows_in_plan: {n_total}")
    print(f"updated:      {n_updated}")
    print(f"skipped:      {n_skipped}")

    return {
        "rows_in_plan": int(n_total),
        "updated": int(n_updated),
        "skipped": int(n_skipped),
        "plan_csv": str(plan_csv),
    }


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan/apply edits to trade JSONs in S3.")

    ap.add_argument("--mode", required=True, choices=["plan", "apply"])
    ap.add_argument("--universe-path", required=False, help="Local universe.csv path. Required for plan.")
    ap.add_argument("--dt", default=None, help="Optional single dt=YYYY-MM-DD to scope scanning.")
    ap.add_argument("--out-csv", default="./data/trade_asset_id_plan.csv")
    ap.add_argument("--plan-csv", default="./data/trade_asset_id_plan.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--reason", default=None, help="Required for real apply writes; stored in audit events.")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)

    if args.mode == "apply" and not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))
        if not args.reason:
            raise ValueError("--reason is required for real apply writes")

    with capture_script_run(
        cfg=cfg,
        script_name="edit_trades_asset_id.py",
        input_args=vars(args),
        dry_run=bool(args.dry_run),
    ) as run_id:
        print(f"[runtime] env={cfg.env} bucket={cfg.bucket} engine_root={cfg.engine_root}")

        if args.mode == "plan":
            if not args.universe_path:
                raise SystemExit("--universe-path is required in plan mode")

            meta = build_plan_csv(
                cfg=cfg,
                universe_path=args.universe_path,
                out_csv=args.out_csv,
                dt=args.dt,
            )
            audit_event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_plan",
                entity_type="trade_asset_id_repair_plan",
                entity_id=None,
                as_of=args.dt,
                source_script="edit_trades_asset_id.py",
                source_mode="plan",
                status=("dry_run" if args.dry_run else "success"),
                reason=args.reason,
                input_args=vars(args),
                output_keys=[str(args.out_csv)],
                metadata=meta,
            )
            write_audit_event(cfg=cfg, event=audit_event, dry_run=bool(args.dry_run))
            return

        apply_plan_csv(
            cfg=cfg,
            plan_csv=args.plan_csv,
            dry_run=bool(args.dry_run),
            write_backups=(not bool(args.no_backup)),
            run_id=run_id,
            reason=args.reason,
            input_args=vars(args),
        )


if __name__ == "__main__":
    main()