from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import boto3
import pandas as pd

BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"
TRADES_TABLE = "trades"

AUDIT_PREFIX = f"{ENGINE_ROOT}/trades_audit"  # backups land here


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(region: str = REGION):
    return boto3.client("s3", region_name=region)


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def s3_list_keys(s3, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix)
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
@dataclass(frozen=True)
class UniverseIndex:
    # ticker (broker_ticker) -> list of (asset_id, row_id, yahoo_ticker, name)
    by_broker_ticker: Dict[str, List[dict]]


def load_universe_index(universe_path: str) -> UniverseIndex:
    df = pd.read_csv(universe_path)

    if "asset_id" not in df.columns or "broker_ticker" not in df.columns:
        raise RuntimeError("universe.csv must contain columns: asset_id, broker_ticker")

    df = df.copy()
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["broker_ticker"] = df["broker_ticker"].astype(str).str.upper().str.strip()

    # optional columns
    for c in ["row_id", "yahoo_ticker", "name", "include"]:
        if c in df.columns:
            df[c] = df[c]

    m: Dict[str, List[dict]] = {}
    for _, r in df.iterrows():
        bt = str(r["broker_ticker"])
        rec = {
            "asset_id": str(r["asset_id"]),
            "row_id": (None if "row_id" not in df.columns else r.get("row_id")),
            "yahoo_ticker": (None if "yahoo_ticker" not in df.columns else r.get("yahoo_ticker")),
            "name": (None if "name" not in df.columns else r.get("name")),
            "include": (None if "include" not in df.columns else r.get("include")),
        }
        m.setdefault(bt, []).append(rec)

    return UniverseIndex(by_broker_ticker=m)


def candidates_for_ticker(idx: UniverseIndex, ticker: str) -> List[dict]:
    t = str(ticker).upper().strip()
    return idx.by_broker_ticker.get(t, [])


# ----------------------------
# Trade scanning
# ----------------------------
def list_trade_keys_for_dt(s3, *, dt: str) -> list[str]:
    prefix = engine_key(TRADES_TABLE, f"dt={dt}")
    keys = s3_list_keys(s3, bucket=BUCKET, prefix=prefix)
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]
    return sorted(keys)


def list_trade_keys_all(s3) -> list[str]:
    prefix = engine_key(TRADES_TABLE, "dt=")
    keys = s3_list_keys(s3, bucket=BUCKET, prefix=prefix)
    keys = [k for k in keys if k.endswith(".json") and "/trade_" in k]
    return sorted(keys)


def is_missing_or_ambiguous_asset_id(trade: dict, idx: UniverseIndex) -> Tuple[bool, List[dict]]:
    tkr = str(trade.get("ticker") or "").upper().strip()
    if not tkr:
        return True, []

    cands = candidates_for_ticker(idx, tkr)
    # missing universe mapping is also "problematic" for editing workflows
    if not cands:
        return True, []

    cur = trade.get("asset_id", None)
    cur = None if cur is None else str(cur).strip()

    # ambiguous mapping if >1 candidate
    if len(cands) > 1:
        return True, cands

    # if exactly 1 candidate but asset_id missing => fixable
    if len(cands) == 1 and not cur:
        return True, cands

    # if has asset_id but doesn't match the single candidate => fixable
    if len(cands) == 1 and cur and cur != str(cands[0]["asset_id"]):
        return True, cands

    return False, cands


# ----------------------------
# Plan + Apply
# ----------------------------
def build_plan_csv(
    *,
    universe_path: str,
    out_csv: str,
    dt: Optional[str],
) -> None:
    s3 = s3_client(REGION)
    idx = load_universe_index(universe_path)

    keys = list_trade_keys_for_dt(s3, dt=dt) if dt else list_trade_keys_all(s3)

    rows: list[dict] = []
    for k in keys:
        tr = s3_get_json(s3, bucket=BUCKET, key=k)
        if not isinstance(tr, dict):
            continue

        flag, cands = is_missing_or_ambiguous_asset_id(tr, idx)
        if not flag:
            continue

        tkr = str(tr.get("ticker") or "").upper().strip()
        cur = tr.get("asset_id", None)
        cur = None if cur is None else str(cur).strip()

        # format candidates
        cand_ids = "|".join([str(x["asset_id"]) for x in cands]) if cands else ""
        cand_names = "|".join([str(x.get("name") or "") for x in cands]) if cands else ""
        cand_yahoo = "|".join([str(x.get("yahoo_ticker") or "") for x in cands]) if cands else ""

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
                "candidates_asset_id": cand_ids,
                "candidates_yahoo_ticker": cand_yahoo,
                "candidates_name": cand_names,
                "resolution_asset_id": "",   # YOU FILL
                "resolution_note": "",       # YOU FILL
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote plan: {out_csv} rows={len(df)}")
    if len(df) > 0:
        print("Next: fill resolution_asset_id, then run with --mode apply")


def apply_plan_csv(
    *,
    plan_csv: str,
    dry_run: bool,
    write_backups: bool,
) -> None:
    s3 = s3_client(REGION)
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

        # read trade
        tr = s3_get_json(s3, bucket=BUCKET, key=key)
        if not isinstance(tr, dict):
            n_skipped += 1
            continue

        old_aid = tr.get("asset_id", None)
        old_aid = None if old_aid is None else str(old_aid).strip()

        if old_aid == new_aid:
            n_skipped += 1
            continue

        # backup
        if write_backups:
            backup_key = f"{AUDIT_PREFIX}/{key.replace('/', '__')}"
            if dry_run:
                print(f"[DRY RUN] backup copy s3://{BUCKET}/{key} -> s3://{BUCKET}/{backup_key}")
            else:
                s3_copy_object(s3, bucket=BUCKET, src_key=key, dst_key=backup_key)

        # update
        tr["asset_id"] = new_aid

        if dry_run:
            print(f"[DRY RUN] update {key}: asset_id {old_aid} -> {new_aid}")
        else:
            s3_put_json(s3, bucket=BUCKET, key=key, payload=tr)
            print(f"[OK] updated {key}: asset_id {old_aid} -> {new_aid}")
        n_updated += 1

    print("")
    print("=== APPLY SUMMARY ===")
    print(f"rows_in_plan: {n_total}")
    print(f"updated:      {n_updated}")
    print(f"skipped:      {n_skipped}")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan/apply edits to trade JSONs in S3 (fix asset_id ambiguity).")

    ap.add_argument("--mode", required=True, choices=["plan", "apply"])
    ap.add_argument("--universe-path", required=False, help="Local universe.csv path (required for plan).")
    ap.add_argument("--dt", default=None, help="Optional single dt=YYYY-MM-DD to scope scanning (plan mode).")
    ap.add_argument("--out-csv", default="./data/trade_asset_id_plan.csv", help="Plan CSV path (plan mode).")
    ap.add_argument("--plan-csv", default="./data/trade_asset_id_plan.csv", help="Plan CSV path (apply mode).")
    ap.add_argument("--dry-run", action="store_true", help="Do not write to S3 (apply mode).")
    ap.add_argument("--no-backup", action="store_true", help="Disable backup copies (apply mode).")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "plan":
        if not args.universe_path:
            raise SystemExit("--universe-path is required in plan mode")
        build_plan_csv(
            universe_path=args.universe_path,
            out_csv=args.out_csv,
            dt=args.dt,
        )
        return

    # apply
    apply_plan_csv(
        plan_csv=args.plan_csv,
        dry_run=bool(args.dry_run),
        write_backups=(not bool(args.no_backup)),
    )


if __name__ == "__main__":
    main()
