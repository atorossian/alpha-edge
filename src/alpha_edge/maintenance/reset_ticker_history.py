# reset_ticker_history.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import boto3
import pandas as pd

from alpha_edge import paths
from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_MARKET_ROOT = "market"


# ----------------------------
# Runtime helpers
# ----------------------------
def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET)).strip()


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION)).strip()


def cfg_market_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "market_root", DEFAULT_MARKET_ROOT)).strip("/")


def cfg_env(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "env", "dev")).strip()


def make_market_store(cfg: RuntimeConfig) -> MarketStore:
    try:
        return MarketStore(
            bucket=cfg_bucket(cfg),
            region=cfg_region(cfg),
            base_prefix=cfg_market_root(cfg),
        )
    except TypeError:
        return MarketStore(
            bucket=cfg_bucket(cfg),
            region=cfg_region(cfg),
        )


# ----------------------------
# S3 helpers
# ----------------------------
def s3_get_json_or_empty(s3, *, bucket: str, key: str) -> dict:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        payload = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )


def list_prefix_keys(
    *,
    bucket: str,
    prefix: str,
    region: str,
) -> list[str]:
    s3 = boto3.client("s3", region_name=region)

    keys: list[str] = []
    token = None

    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) or []:
            k = obj.get("Key")
            if isinstance(k, str) and k:
                keys.append(k)

        if not resp.get("IsTruncated"):
            break

        token = resp.get("NextContinuationToken")

    return keys


def delete_keys(
    *,
    bucket: str,
    keys: list[str],
    region: str,
    dry_run: bool,
    print_limit: int = 80,
) -> int:
    keys = sorted(set(keys))

    if not keys:
        return 0

    if dry_run:
        for k in keys[: int(print_limit)]:
            print(f"[DRY RUN] would delete s3://{bucket}/{k}")
        if len(keys) > int(print_limit):
            print(f"[DRY RUN] ... and {len(keys) - int(print_limit)} more")
        return len(keys)

    s3 = boto3.client("s3", region_name=region)

    deleted = 0
    for i in range(0, len(keys), 1000):
        chunk = keys[i : i + 1000]
        resp = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
        )
        deleted += len(chunk)

        errors = resp.get("Errors") or []
        if errors:
            print("[WARN] delete errors, showing up to 20:")
            for e in errors[:20]:
                print(f"  - {e}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")

    return deleted


# ----------------------------
# Resolution helpers
# ----------------------------
def load_universe_df(universe_csv: str | Path) -> pd.DataFrame:
    p = Path(universe_csv)
    if not p.exists():
        raise RuntimeError(f"Universe CSV not found: {p}")

    u = pd.read_csv(p)
    if u.empty:
        raise RuntimeError(f"Universe CSV is empty: {p}")

    if "asset_id" not in u.columns:
        raise RuntimeError(f"Universe CSV missing asset_id column: {p}")

    for c in ["asset_id", "ticker", "broker_ticker", "yahoo_ticker"]:
        if c in u.columns:
            u[c] = u[c].astype(str).str.upper().str.strip()

    if "include" in u.columns:
        u["include"] = pd.to_numeric(u["include"], errors="coerce").fillna(1).astype(int)
    else:
        u["include"] = 1

    return u


def resolve_asset_id(
    *,
    ticker: str | None,
    asset_id: str | None,
    universe_csv: str | Path,
) -> tuple[str, str | None]:
    if asset_id:
        aid = str(asset_id).strip()
        if not aid:
            raise ValueError("--asset-id cannot be empty")
        return aid, None

    if not ticker:
        raise ValueError("Provide either --asset-id or --ticker")

    t = str(ticker).upper().strip()
    if not t:
        raise ValueError("--ticker cannot be empty")

    u = load_universe_df(universe_csv)

    mask = pd.Series(False, index=u.index)
    for col in ["ticker", "broker_ticker", "yahoo_ticker"]:
        if col in u.columns:
            mask = mask | (u[col].astype(str).str.upper().str.strip() == t)

    matches = u.loc[mask].copy()

    if matches.empty:
        raise RuntimeError(f"Could not resolve ticker={t!r} to asset_id using {universe_csv}")

    # Prefer included row if duplicates exist.
    matches = matches.sort_values("include", ascending=False)

    if matches["asset_id"].nunique() > 1:
        sample = matches[["asset_id", "ticker", "broker_ticker", "yahoo_ticker", "include"]].head(10)
        raise RuntimeError(
            f"Ambiguous ticker={t!r}; matched multiple asset_ids. "
            f"Use --asset-id explicitly.\n{sample.to_string(index=False)}"
        )

    aid = str(matches["asset_id"].iloc[0]).strip()
    display_ticker = None
    if "ticker" in matches.columns:
        display_ticker = str(matches["ticker"].iloc[0]).upper().strip()

    return aid, display_ticker


def market_prefixes_for_asset(*, market_root: str, asset_id: str) -> list[str]:
    root = str(market_root).strip("/")
    aid = str(asset_id).strip()

    return [
        f"{root}/ohlcv_usd/v1/asset_id={aid}/",
        f"{root}/returns_usd/v1/asset_id={aid}/",
        f"{root}/manifests/ohlcv_usd/asset_id={aid}/",
        f"{root}/manifests/returns_usd/asset_id={aid}/",
    ]


def state_keys_for_market_root(*, market_root: str) -> list[str]:
    root = str(market_root).strip("/")

    return [
        f"{root}/state/last_date_by_asset.json",
        f"{root}/state/last_date_by_ticker.json",          # legacy
        f"{root}/state/provider_symbol_by_asset.json",
        f"{root}/state/provider_symbol_by_ticker.json",    # legacy
    ]


def remove_asset_from_state_payload(
    payload: dict,
    *,
    asset_id: str,
    ticker: str | None,
) -> tuple[dict, bool]:
    """
    Removes both asset_id and ticker keys if present.
    This supports both old ticker-keyed and current asset_id-keyed state files.
    """
    out = dict(payload or {})
    changed = False

    candidates = {str(asset_id).strip()}
    if ticker:
        candidates.add(str(ticker).upper().strip())

    for k in list(out.keys()):
        if str(k).strip() in candidates or str(k).upper().strip() in candidates:
            out.pop(k, None)
            changed = True

    return out, changed


def print_safety_banner(
    *,
    cfg: RuntimeConfig,
    bucket: str,
    region: str,
    market_root: str,
    asset_id: str,
    ticker: str | None,
    prefixes: list[str],
    state_keys: list[str],
    dry_run: bool,
) -> None:
    print("\n" + "=" * 88)
    print("MAINTENANCE: RESET MARKET HISTORY FOR ONE ASSET")
    print("=" * 88)
    print("This deletes OHLCV/returns partitions and removes ingest state for one asset.")
    print("Default mode is dry-run. Deletion requires --execute --confirm.")
    print("")
    print(f"env:         {cfg_env(cfg)}")
    print(f"bucket:      {bucket}")
    print(f"region:      {region}")
    print(f"market_root: {market_root}")
    print(f"asset_id:    {asset_id}")
    print(f"ticker:      {ticker}")
    print(f"dry_run:     {dry_run}")
    print("")
    print("prefixes scanned/deleted:")
    for p in prefixes:
        print(f"  - s3://{bucket}/{p}")
    print("")
    print("state files checked:")
    for k in state_keys:
        print(f"  - s3://{bucket}/{k}")
    print("=" * 88 + "\n")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Delete market partitions and reset ingestion state for one asset."
    )

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument("--bucket", default=None, help="Override runtime bucket. Usually avoid this.")
    ap.add_argument("--region", default=None, help="Override runtime region. Usually avoid this.")
    ap.add_argument("--market-root", default=None, help="Override runtime market root. Usually avoid this.")

    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--ticker", default=None)
    target.add_argument("--asset-id", default=None)

    ap.add_argument(
        "--universe-csv",
        default=str(paths.universe_dir() / "universe.csv"),
        help="Used only when resolving --ticker to asset_id.",
    )

    ap.add_argument("--execute", action="store_true", help="Actually delete and mutate state. Default is dry-run.")
    ap.add_argument("--confirm", action="store_true", help="Required together with --execute.")
    ap.add_argument("--dry-run", action="store_true", help="Compatibility flag. Default is dry-run anyway.")

    ap.add_argument("--reason", default=None, help="Required for real maintenance writes/deletes. Stored in audit trail.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if bool(args.execute) and bool(args.dry_run):
        raise SystemExit("Conflicting flags: use either --execute or --dry-run, not both.")

    cfg = load_runtime_config(args.env)

    bucket = str(args.bucket or cfg_bucket(cfg)).strip()
    region = str(args.region or cfg_region(cfg)).strip()
    market_root = str(args.market_root or cfg_market_root(cfg)).strip("/")

    dry_run = not bool(args.execute)
    do_write = bool(args.execute)

    if do_write and not bool(args.confirm):
        raise SystemExit("Actual reset requires both --execute and --confirm.")

    if do_write:
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    asset_id, resolved_ticker = resolve_asset_id(
        ticker=args.ticker,
        asset_id=args.asset_id,
        universe_csv=args.universe_csv,
    )
    ticker = str(args.ticker).upper().strip() if args.ticker else resolved_ticker

    prefixes = market_prefixes_for_asset(market_root=market_root, asset_id=asset_id)
    state_keys = state_keys_for_market_root(market_root=market_root)

    print_safety_banner(
        cfg=cfg,
        bucket=bucket,
        region=region,
        market_root=market_root,
        asset_id=asset_id,
        ticker=ticker,
        prefixes=prefixes,
        state_keys=state_keys,
        dry_run=dry_run,
    )

    # Constructing store is not strictly necessary for deletion, but it validates runtime MarketStore compatibility.
    _ = make_market_store(cfg)

    keys_to_delete: list[str] = []
    for prefix in prefixes:
        keys = list_prefix_keys(bucket=bucket, prefix=prefix, region=region)
        keys_to_delete.extend(keys)
        print(f"[SCAN] s3://{bucket}/{prefix} objects={len(keys)}")

    keys_to_delete = sorted(set(keys_to_delete))
    print("")
    print(f"[TOTAL] matched objects: {len(keys_to_delete)}")

    deleted = delete_keys(
        bucket=bucket,
        keys=keys_to_delete,
        region=region,
        dry_run=dry_run,
    )

    print(f"[INFO] objects {'planned' if dry_run else 'deleted'}: {deleted}")

    s3 = boto3.client("s3", region_name=region)

    changed_state_files = 0
    for state_key in state_keys:
        payload = s3_get_json_or_empty(s3, bucket=bucket, key=state_key)
        if not payload:
            print(f"[STATE] missing/empty: s3://{bucket}/{state_key}")
            continue

        new_payload, changed = remove_asset_from_state_payload(
            payload,
            asset_id=asset_id,
            ticker=ticker,
        )

        if dry_run:
            print(
                f"[DRY RUN] state {'would change' if changed else 'unchanged'}: "
                f"s3://{bucket}/{state_key}"
            )
            if changed:
                changed_state_files += 1
            continue

        if changed:
            s3_put_json(s3, bucket=bucket, key=state_key, payload=new_payload)
            changed_state_files += 1
            print(f"[OK] updated state: s3://{bucket}/{state_key}")
        else:
            print(f"[STATE] unchanged: s3://{bucket}/{state_key}")

    print("")
    print("--- summary ---")
    print(f"env={cfg_env(cfg)}")
    print(f"bucket={bucket}")
    print(f"market_root={market_root}")
    print(f"asset_id={asset_id}")
    print(f"ticker={ticker}")
    print(f"objects_matched={len(keys_to_delete)}")
    print(f"objects_deleted_or_planned={deleted}")
    print(f"state_files_changed_or_planned={changed_state_files}")

    if dry_run:
        print("\n[DRY RUN DONE] No deletion/state mutation executed. Rerun with --execute --confirm to apply.")
    else:
        print("\n[RESET DONE]")



# ----------------------------
# Audit/logging entrypoint wrapper
# ----------------------------
def main_with_audit() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = not bool(getattr(args, "execute", False))
    if bool(getattr(args, "execute", False)) and not getattr(args, "reason", None):
        raise SystemExit("--reason is required when using --execute for reset_ticker_history.")

    captured_delete_keys: list[str] = []
    captured_state_keys: list[str] = []
    original_delete_keys = globals()["delete_keys"]
    original_s3_put_json = globals()["s3_put_json"]

    def _audited_delete_keys(*, bucket: str, keys: list[str], region: str, dry_run: bool) -> int:
        captured_delete_keys.extend([str(k) for k in keys])
        return original_delete_keys(bucket=bucket, keys=keys, region=region, dry_run=dry_run)

    def _audited_s3_put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
        captured_state_keys.append(str(key))
        return original_s3_put_json(s3, bucket=bucket, key=key, payload=payload)

    with capture_script_run(cfg=cfg, script_name="reset_ticker_history.py", input_args=vars(args), dry_run=is_dry_run) as run_id:
        globals()["delete_keys"] = _audited_delete_keys
        globals()["s3_put_json"] = _audited_s3_put_json
        try:
            main()
            planned = sorted(set(captured_delete_keys))
            changed = sorted(set(captured_state_keys))
            event = build_audit_event(
                cfg=cfg, run_id=run_id, event_type="delete", entity_type="asset_market_history_reset",
                entity_id=str(getattr(args, "asset_id", None) or getattr(args, "ticker", None) or ""), as_of=None,
                source_script="reset_ticker_history.py", source_mode="reset_ticker_history",
                status=("dry_run" if is_dry_run else "success"), reason=getattr(args, "reason", None),
                input_args=vars(args), output_keys=changed, deleted_keys=([] if is_dry_run else planned),
                metadata={"tier": "tier_2", "payload_policy": "large_dataset_metadata_only", "planned_deleted_keys": planned,
                          "planned_deleted_count": len(planned), "changed_state_keys": changed, "changed_state_count": len(changed),
                          "ticker": getattr(args, "ticker", None), "asset_id": getattr(args, "asset_id", None),
                          "execute": bool(getattr(args, "execute", False))},
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
        except Exception as exc:
            planned = sorted(set(captured_delete_keys))
            changed = sorted(set(captured_state_keys))
            event = build_audit_event(
                cfg=cfg, run_id=run_id, event_type="delete", entity_type="asset_market_history_reset",
                entity_id=str(getattr(args, "asset_id", None) or getattr(args, "ticker", None) or ""), as_of=None,
                source_script="reset_ticker_history.py", source_mode="reset_ticker_history", status="failed",
                reason=getattr(args, "reason", None), input_args=vars(args),
                metadata={"tier": "tier_2", "payload_policy": "large_dataset_metadata_only", "planned_deleted_keys": planned,
                          "planned_deleted_count": len(planned), "changed_state_keys": changed, "changed_state_count": len(changed),
                          "execute": bool(getattr(args, "execute", False))}, error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise
        finally:
            globals()["delete_keys"] = original_delete_keys
            globals()["s3_put_json"] = original_s3_put_json


if __name__ == "__main__":
    main_with_audit()