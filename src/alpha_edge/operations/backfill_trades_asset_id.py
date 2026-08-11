from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple

import boto3
import pandas as pd

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT = "engine/v1"
TRADES_TABLE = "trades"

QTY_DECIMALS = 8
VALUE_DECIMALS = 2
QTY_TOL = 1e-8
VALUE_TOL = 0.01
INTEGER_QTY_TOL = 1e-6

FRACTIONAL_EQUITY_TICKERS = {
    "AI.PA",
}
FRACTIONAL_EQUITY_ASSET_IDS = {
    # "EQHxxxxxxxxxxxxxxxxxxx",
}


# ----------------------------
# Runtime helpers
# ----------------------------
def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET))


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION))


def cfg_engine_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "engine_root", DEFAULT_ENGINE_ROOT)).strip("/")


def cfg_env(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "env", "dev"))


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(region: str):
    return boto3.client("s3", region_name=region)


def engine_key(cfg: RuntimeConfig, *parts: str) -> str:
    return "/".join([cfg_engine_root(cfg)] + [p.strip("/") for p in parts])


def dt_key(cfg: RuntimeConfig, table: str, dt_str: str, filename: str) -> str:
    return engine_key(cfg, table, f"dt={dt_str}", filename)


def s3_get_json(s3, *, bucket: str, key: str) -> dict:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return json.loads(body.decode("utf-8"))


def s3_get_json_optional(s3, *, bucket: str, key: str) -> Optional[dict]:
    try:
        return s3_get_json(s3, bucket=bucket, key=key)
    except Exception:
        return None


# ----------------------------
# Helpers
# ----------------------------
def _round_qty(x: float, decimals: int = QTY_DECIMALS) -> float:
    return round(float(x), decimals)


def _round_value(x: float, decimals: int = VALUE_DECIMALS) -> float:
    return round(float(x), decimals)


def _is_finite_positive(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v) and v > 0.0
    except Exception:
        return False


def _normalize_quantity_unit(x: Optional[str], ticker: str) -> Optional[str]:
    if x is None:
        return None

    s = str(x).strip().lower()
    if s == "":
        return None

    unit_map = {
        "share": "shares",
        "shares": "shares",
        "contract": "contracts",
        "contracts": "contracts",
        "coin": "coins",
        "coins": "coins",
        "ounce": "ounces",
        "ounces": "ounces",
        "btc": "coins",
        "eth": "coins",
        "sol": "coins",
        "ada": "coins",
        "xrp": "coins",
        "dot": "coins",
        "ltc": "coins",
        "bnb": "coins",
        "avax": "coins",
        "link": "coins",
        "matic": "coins",
        "atom": "coins",
        "near": "coins",
        "uni": "coins",
        "aave": "coins",
        "trx": "coins",
        "etc": "coins",
        "doge": "coins",
        "hbar": "coins",
        "sui": "coins",
        "dash": "coins",
        "bch": "coins",
        "qtum": "coins",
        "apt": "coins",
        "arb": "coins",
        "inj": "coins",
        "mana": "coins",
        "neo": "coins",
        "render": "coins",
        "derivative": "derivative",
        "derivatives": "derivative",
    }

    out = unit_map.get(s, s)

    t = str(ticker).upper().strip().replace("/", "-")
    if "-" in t:
        base = t.split("-", 1)[0]
        crypto_bases = {
            "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "LTC", "BNB",
            "AVAX", "LINK", "MATIC", "ATOM", "NEAR", "UNI", "AAVE",
            "TRX", "ETC", "DOGE", "HBAR", "SUI", "DASH", "BCH",
            "QTUM", "APT", "ARB", "INJ", "MANA", "NEO", "RENDER",
        }
        if base in crypto_bases:
            return "coins"

    return out


def _infer_quantity_from_value_price(value: float, price: float) -> float:
    if not _is_finite_positive(value):
        raise ValueError(f"Cannot infer quantity: invalid value={value!r}")
    if not _is_finite_positive(price):
        raise ValueError(f"Cannot infer quantity: invalid price={price!r}")
    return float(value) / float(price)


def _is_crypto_pair(ticker: str) -> bool:
    t = str(ticker).upper().strip().replace("/", "-")
    if "-" not in t:
        return False

    base, quote = t.split("-", 1)
    crypto_bases = {
        "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "LTC", "BNB",
        "AVAX", "LINK", "MATIC", "ATOM", "NEAR", "UNI", "AAVE",
        "TRX", "ETC", "DOGE", "HBAR", "SUI", "DASH", "BCH",
        "QTUM", "APT", "ARB", "INJ", "MANA", "NEO", "RENDER",
    }
    crypto_quotes = {"USD", "USDT", "USDC", "EUR"}
    return base in crypto_bases and quote in crypto_quotes


def _quantity_policy(*, ticker: str, asset_id: str, quantity_unit: Optional[str]) -> str:
    t = str(ticker).upper().strip()
    aid = str(asset_id or "").strip()
    unit = str(quantity_unit or "").strip().lower()

    if aid in FRACTIONAL_EQUITY_ASSET_IDS or t in FRACTIONAL_EQUITY_TICKERS:
        return "fractional"

    if _is_crypto_pair(t):
        return "fractional"

    if unit in {"derivative", "derivatives", "contracts", "coins"}:
        return "fractional"

    return "integer"


def _normalize_repaired_quantity(*, raw_qty: float, policy: str) -> tuple[float, bool, str]:
    if not math.isfinite(raw_qty) or raw_qty <= 0.0:
        return raw_qty, False, "raw quantity is invalid"

    if policy == "fractional":
        return _round_qty(raw_qty), True, "fractional quantity allowed"

    nearest = round(raw_qty)
    if abs(raw_qty - nearest) <= INTEGER_QTY_TOL:
        return float(nearest), True, "integer quantity snapped from value / price"

    return _round_qty(raw_qty), False, "quantity is materially fractional for integer-only instrument"


# ----------------------------
# Repair row model
# ----------------------------
@dataclass
class RepairPlanRow:
    trade_id: str
    as_of: str
    ticker: str
    asset_id: str
    side: str
    action_tag: str
    status_from_audit: str
    issue_code: str
    s3_key: str

    old_quantity: Optional[float]
    new_quantity: Optional[float]

    old_price: Optional[float]
    new_price: Optional[float]

    old_value: Optional[float]
    new_value: Optional[float]

    old_quantity_unit: Optional[str]
    new_quantity_unit: Optional[str]

    old_reported_pnl: Optional[float]
    new_reported_pnl: Optional[float]

    quantity_policy: Optional[str]
    repair_action: str
    repair_reason: str
    command: str


# ----------------------------
# Load audit + original trade
# ----------------------------
def _load_original_trade(
    s3,
    *,
    cfg: RuntimeConfig,
    trade_id: str,
    as_of: str,
    s3_key: Optional[str],
) -> dict:
    bucket = cfg_bucket(cfg)

    if s3_key:
        obj = s3_get_json_optional(s3, bucket=bucket, key=s3_key)
        if isinstance(obj, dict):
            return obj

    key = dt_key(cfg, TRADES_TABLE, str(as_of), f"trade_{trade_id}.json")
    obj = s3_get_json_optional(s3, bucket=bucket, key=key)
    if isinstance(obj, dict):
        return obj

    raise RuntimeError(f"Could not load original trade trade_id={trade_id} as_of={as_of} s3_key={s3_key}")


def _build_edit_command_args(
    row: RepairPlanRow,
    *,
    python_entrypoint: str,
    dry_run: bool,
    cfg: RuntimeConfig,
    confirm_prod_write: bool,
) -> List[str]:
    parts = [
        "poetry",
        "run",
        "python",
        python_entrypoint,
        "--mode",
        "edit",
        "--trade-id",
        str(row.trade_id),
        "--old-as-of",
        str(row.as_of),
        "--env",
        cfg_env(cfg),
    ]

    if confirm_prod_write:
        parts.append("--confirm-prod-write")

    if row.new_quantity is not None and (
        row.old_quantity is None or abs(float(row.new_quantity) - float(row.old_quantity)) > QTY_TOL
    ):
        parts.extend(["--quantity", str(row.new_quantity)])

    if row.new_price is not None and row.old_price != row.new_price:
        parts.extend(["--price", str(row.new_price)])

    if row.new_value is not None and (
        row.old_value is None or abs(float(row.new_value) - float(row.old_value)) > VALUE_TOL
    ):
        parts.extend(["--value", str(row.new_value)])

    if row.new_quantity_unit is not None and row.old_quantity_unit != row.new_quantity_unit:
        parts.extend(["--quantity-unit", str(row.new_quantity_unit)])

    if row.new_reported_pnl is not None and row.old_reported_pnl != row.new_reported_pnl:
        parts.extend(["--reported-pnl", str(row.new_reported_pnl)])

    if dry_run:
        parts.append("--dry-run")

    return parts


def _command_args_to_text(args: List[str]) -> str:
    def q(x: str) -> str:
        if any(ch in x for ch in [' ', '"', "'"]):
            return '"' + x.replace('"', '\\"') + '"'
        return x

    return " ".join(q(x) for x in args)


def _manual_row(
    *,
    trade_id: str,
    as_of: str,
    ticker: str,
    asset_id: str,
    side: str,
    action_tag: str,
    status: str,
    issue_code: str,
    s3_key: str,
    reason: str,
    old_quantity: Optional[float] = None,
    new_quantity: Optional[float] = None,
    old_price: Optional[float] = None,
    new_price: Optional[float] = None,
    old_value: Optional[float] = None,
    new_value: Optional[float] = None,
    old_quantity_unit: Optional[str] = None,
    new_quantity_unit: Optional[str] = None,
    old_reported_pnl: Optional[float] = None,
    new_reported_pnl: Optional[float] = None,
    quantity_policy: Optional[str] = None,
) -> RepairPlanRow:
    return RepairPlanRow(
        trade_id=trade_id,
        as_of=as_of,
        ticker=ticker,
        asset_id=asset_id,
        side=side,
        action_tag=action_tag,
        status_from_audit=status,
        issue_code=issue_code,
        s3_key=s3_key,
        old_quantity=old_quantity,
        new_quantity=new_quantity,
        old_price=old_price,
        new_price=new_price,
        old_value=old_value,
        new_value=new_value,
        old_quantity_unit=old_quantity_unit,
        new_quantity_unit=new_quantity_unit,
        old_reported_pnl=old_reported_pnl,
        new_reported_pnl=new_reported_pnl,
        quantity_policy=quantity_policy,
        repair_action="MANUAL_REVIEW",
        repair_reason=reason,
        command="",
    )


# ----------------------------
# Core repair-plan / execution logic
# ----------------------------
def build_repair_plan(
    *,
    cfg: RuntimeConfig,
    audit_rows_csv: str,
    out_auto_csv: str,
    out_manual_csv: str,
    python_entrypoint: str,
    only_status: Tuple[str, ...] = ("REFactor",),
    recompute_reported_pnl: bool = False,
    execute: bool = False,
    dry_run: bool = False,
    stop_on_error: bool = False,
    confirm_prod_write: bool = False,
) -> None:
    s3 = s3_client(cfg_region(cfg))

    audit_df = pd.read_csv(audit_rows_csv)
    if audit_df.empty:
        pd.DataFrame().to_csv(out_auto_csv, index=False)
        pd.DataFrame().to_csv(out_manual_csv, index=False)
        print("[OK] Empty audit file. No repair plan generated.")
        return

    audit_df = audit_df.where(pd.notna(audit_df), None)

    auto_rows: List[RepairPlanRow] = []
    manual_rows: List[RepairPlanRow] = []

    exec_ok = 0
    exec_fail = 0
    exec_skipped = 0

    for _, a in audit_df.iterrows():
        status = str(a.get("status") or "")
        if status not in only_status:
            continue

        trade_id = str(a.get("trade_id") or "").strip()
        as_of = str(a.get("as_of") or "").strip()
        s3_key = None if a.get("s3_key") is None else str(a.get("s3_key")).strip()
        issue_code = str(a.get("issue_code") or "")
        ticker = str(a.get("ticker") or "").upper().strip()
        asset_id = str(a.get("asset_id") or "").strip()
        side = str(a.get("side") or "").upper().strip()
        action_tag = str(a.get("action_tag") or "").lower().strip()

        if not trade_id or not as_of:
            manual_rows.append(
                _manual_row(
                    trade_id=trade_id,
                    as_of=as_of,
                    ticker=ticker,
                    asset_id=asset_id,
                    side=side,
                    action_tag=action_tag,
                    status=status,
                    issue_code=issue_code,
                    s3_key=s3_key or "",
                    reason="Missing trade_id or as_of in audit row.",
                )
            )
            exec_skipped += 1
            continue

        try:
            obj = _load_original_trade(
                s3,
                cfg=cfg,
                trade_id=trade_id,
                as_of=as_of,
                s3_key=s3_key,
            )
        except Exception as e:
            manual_rows.append(
                _manual_row(
                    trade_id=trade_id,
                    as_of=as_of,
                    ticker=ticker,
                    asset_id=asset_id,
                    side=side,
                    action_tag=action_tag,
                    status=status,
                    issue_code=issue_code,
                    s3_key=s3_key or "",
                    reason=f"Could not load original trade: {e}",
                )
            )
            exec_skipped += 1
            continue

        old_qty = None if obj.get("quantity") is None else float(obj.get("quantity"))
        old_px = None if obj.get("price") is None else float(obj.get("price"))
        old_val = None if obj.get("value") is None else float(obj.get("value"))
        old_unit = None if obj.get("quantity_unit") is None else str(obj.get("quantity_unit"))
        old_rpnl = None if obj.get("reported_pnl") is None else float(obj.get("reported_pnl"))

        if not ticker:
            ticker = str(obj.get("ticker") or "").upper().strip()
        if not asset_id:
            asset_id = str(obj.get("asset_id") or "").strip()
        if not side:
            side = str(obj.get("side") or "").upper().strip()
        if not action_tag:
            action_tag = str(obj.get("action_tag") or "").lower().strip()

        new_qty = old_qty
        new_px = old_px
        new_val = old_val
        new_unit = _normalize_quantity_unit(old_unit, ticker=ticker)
        new_rpnl = old_rpnl

        repair_reason_parts: List[str] = []
        quantity_policy = _quantity_policy(ticker=ticker, asset_id=asset_id, quantity_unit=new_unit)

        if _is_finite_positive(old_val) and _is_finite_positive(old_px):
            raw_qty = _infer_quantity_from_value_price(float(old_val), float(old_px))
            normalized_qty, auto_ok, qty_detail = _normalize_repaired_quantity(
                raw_qty=raw_qty,
                policy=quantity_policy,
            )

            if not auto_ok:
                manual_rows.append(
                    _manual_row(
                        trade_id=trade_id,
                        as_of=as_of,
                        ticker=ticker,
                        asset_id=asset_id,
                        side=side,
                        action_tag=action_tag,
                        status=status,
                        issue_code=issue_code,
                        s3_key=s3_key or "",
                        old_quantity=old_qty,
                        new_quantity=normalized_qty,
                        old_price=old_px,
                        new_price=old_px,
                        old_value=old_val,
                        new_value=old_val,
                        old_quantity_unit=old_unit,
                        new_quantity_unit=new_unit,
                        old_reported_pnl=old_rpnl,
                        new_reported_pnl=old_rpnl,
                        quantity_policy=quantity_policy,
                        reason=qty_detail,
                    )
                )
                exec_skipped += 1
                continue

            if old_qty is None or abs(float(normalized_qty) - float(old_qty)) > QTY_TOL:
                new_qty = normalized_qty
                repair_reason_parts.append("quantity := value / price")
            else:
                new_qty = old_qty

            new_val = _round_value(float(new_qty) * float(old_px))
            if old_val is None or abs(float(new_val) - float(old_val)) > VALUE_TOL:
                repair_reason_parts.append("value := quantity * price")

        else:
            manual_rows.append(
                _manual_row(
                    trade_id=trade_id,
                    as_of=as_of,
                    ticker=ticker,
                    asset_id=asset_id,
                    side=side,
                    action_tag=action_tag,
                    status=status,
                    issue_code=issue_code,
                    s3_key=s3_key or "",
                    old_quantity=old_qty,
                    new_quantity=None,
                    old_price=old_px,
                    new_price=old_px,
                    old_value=old_val,
                    new_value=old_val,
                    old_quantity_unit=old_unit,
                    new_quantity_unit=new_unit,
                    old_reported_pnl=old_rpnl,
                    new_reported_pnl=old_rpnl,
                    quantity_policy=quantity_policy,
                    reason="Cannot infer quantity because value and/or price are missing/invalid.",
                )
            )
            exec_skipped += 1
            continue

        if recompute_reported_pnl:
            # Intentionally disabled: reported_pnl should come from broker/import logic, not this quantity repair.
            pass

        changed = False

        if old_qty is None or new_qty is None:
            changed = True
        elif abs(float(new_qty) - float(old_qty)) > QTY_TOL:
            changed = True

        if new_unit != old_unit:
            changed = True

        if new_val is not None and old_val is not None:
            if abs(float(new_val) - float(old_val)) > VALUE_TOL:
                changed = True
        elif new_val != old_val:
            changed = True

        row = RepairPlanRow(
            trade_id=trade_id,
            as_of=as_of,
            ticker=ticker,
            asset_id=asset_id,
            side=side,
            action_tag=action_tag,
            status_from_audit=status,
            issue_code=issue_code,
            s3_key=s3_key or dt_key(cfg, TRADES_TABLE, as_of, f"trade_{trade_id}.json"),
            old_quantity=old_qty,
            new_quantity=new_qty,
            old_price=old_px,
            new_price=new_px,
            old_value=old_val,
            new_value=new_val,
            old_quantity_unit=old_unit,
            new_quantity_unit=new_unit,
            old_reported_pnl=old_rpnl,
            new_reported_pnl=new_rpnl,
            quantity_policy=quantity_policy,
            repair_action="AUTO_PATCH" if changed else "NO_CHANGE",
            repair_reason="; ".join(repair_reason_parts) if repair_reason_parts else "no effective patch required",
            command="",
        )

        cmd_args: List[str] = []
        if changed:
            cmd_args = _build_edit_command_args(
                row,
                python_entrypoint=python_entrypoint,
                dry_run=dry_run,
                cfg=cfg,
                confirm_prod_write=confirm_prod_write,
            )
            row.command = _command_args_to_text(cmd_args)

        auto_rows.append(row)

        if execute and changed:
            print("\n=== EXECUTE REPAIR ===")
            print(row.command)
            try:
                subprocess.run(cmd_args, check=True)
                exec_ok += 1
            except subprocess.CalledProcessError as e:
                exec_fail += 1
                row.repair_action = "EXEC_FAILED"
                row.repair_reason = f"{row.repair_reason}; execution failed rc={e.returncode}"

                if stop_on_error:
                    pd.DataFrame([asdict(r) for r in auto_rows]).to_csv(out_auto_csv, index=False)
                    pd.DataFrame([asdict(r) for r in manual_rows]).to_csv(out_manual_csv, index=False)
                    raise

    pd.DataFrame([asdict(r) for r in auto_rows]).to_csv(out_auto_csv, index=False)
    pd.DataFrame([asdict(r) for r in manual_rows]).to_csv(out_manual_csv, index=False)

    print("\n=== TRADE REPAIR PLAN ===")
    print(f"env:                {cfg_env(cfg)}")
    print(f"bucket:             {cfg_bucket(cfg)}")
    print(f"engine_root:        {cfg_engine_root(cfg)}")
    print(f"audit_rows_csv:     {audit_rows_csv}")
    print(f"auto_rows:          {len(auto_rows)}")
    print(f"manual_rows:        {len(manual_rows)}")
    print(f"auto_csv:           {out_auto_csv}")
    print(f"manual_csv:         {out_manual_csv}")
    print(f"execute:            {execute}")
    print(f"dry_run:            {dry_run}")

    if execute:
        print(f"exec_ok:            {exec_ok}")
        print(f"exec_fail:          {exec_fail}")
        print(f"exec_skipped:       {exec_skipped}")

    print("")


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build deterministic repair plan for broken trade quantities and optionally execute edits."
    )

    ap.add_argument("--audit-rows-csv", required=True, help="Path to audit_crypto_quantity_rows.csv")
    ap.add_argument("--out-auto-csv", default="./data/trade_repair_plan_auto.csv")
    ap.add_argument("--out-manual-csv", default="./data/trade_repair_plan_manual.csv")
    ap.add_argument(
        "--python-entrypoint",
        default="src/alpha_edge/operations/record_trade.py",
        help="Path used in generated poetry edit commands.",
    )
    ap.add_argument(
        "--statuses",
        default="REFactor",
        help='Comma-separated audit statuses to include. Default: "REFactor"',
    )
    ap.add_argument("--execute", action="store_true", help="Actually execute the generated edit commands.")
    ap.add_argument("--dry-run", action="store_true", help="Pass --dry-run to each generated edit command.")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop immediately if one executed command fails.")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    args = ap.parse_args()

    cfg = load_runtime_config(args.env)

    if bool(args.execute) and not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    with capture_script_run(
        cfg=cfg,
        script_name="backfill_trades_asset_id.py",
        input_args=vars(args),
        dry_run=bool(args.dry_run),
    ) as run_id:
        statuses = tuple(x.strip() for x in str(args.statuses).split(",") if x.strip())

        build_repair_plan(
            cfg=cfg,
            audit_rows_csv=str(args.audit_rows_csv),
            out_auto_csv=str(args.out_auto_csv),
            out_manual_csv=str(args.out_manual_csv),
            python_entrypoint=str(args.python_entrypoint),
            only_status=statuses,
            recompute_reported_pnl=False,
            execute=bool(args.execute),
            dry_run=bool(args.dry_run),
            stop_on_error=bool(args.stop_on_error),
            confirm_prod_write=bool(args.confirm_prod_write),
        )

        audit_event = build_audit_event(
            cfg=cfg,
            run_id=run_id,
            event_type=("execute_repair_plan" if args.execute else "build_plan"),
            entity_type="trade_repair_plan",
            entity_id=None,
            as_of=None,
            source_script="backfill_trades_asset_id.py",
            source_mode=("execute" if args.execute else "plan"),
            status=("dry_run" if args.dry_run else "success"),
            reason=None,
            input_args=vars(args),
            output_keys=[str(args.out_auto_csv), str(args.out_manual_csv)],
            metadata={
                "audit_rows_csv": str(args.audit_rows_csv),
                "statuses": list(statuses),
                "execute": bool(args.execute),
                "note": "Individual trade edits are audited by record_trade.py when executed.",
            },
        )
        write_audit_event(cfg=cfg, event=audit_event, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()