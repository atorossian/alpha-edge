from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd


BUCKET = "alpha-edge-algo"
REGION = "eu-west-1"
ENGINE_ROOT = "engine/v1"
TRADES_TABLE = "trades"

QTY_DECIMALS = 8
VALUE_DECIMALS = 2
QTY_TOL = 1e-8
VALUE_TOL = 0.01
INTEGER_QTY_TOL = 1e-6

# Allow fractional equity quantities only for explicit exceptions
# Prefer asset_id over ticker when possible.
FRACTIONAL_EQUITY_TICKERS = {
    "AI.PA",   # Air Liquide split case
}
FRACTIONAL_EQUITY_ASSET_IDS = {
    # "EQHxxxxxxxxxxxxxxxxxxx",
}


# ----------------------------
# S3 helpers
# ----------------------------
def s3_client(region: str = REGION):
    return boto3.client("s3", region_name=region)


def engine_key(*parts: str) -> str:
    return "/".join([ENGINE_ROOT.strip("/")] + [p.strip("/") for p in parts])


def dt_key(table: str, dt_str: str, filename: str) -> str:
    return engine_key(table, f"dt={dt_str}", filename)


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
        "derivative": "derivative",
        "derivatives": "derivative",
    }
    out = unit_map.get(s, s)

    t = str(ticker).upper().strip()
    if "-" in t:
        base = t.split("-", 1)[0]
        if base in {
            "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "LTC", "BNB",
            "AVAX", "LINK", "MATIC", "ATOM", "NEAR", "UNI", "AAVE",
            "TRX", "ETC", "DOGE", "HBAR", "SUI", "DASH", "BCH"
        }:
            if out in {base.lower(), base.upper().lower(), s}:
                return "coins"

    return out


def _infer_quantity_from_value_price(value: float, price: float) -> float:
    if not _is_finite_positive(value):
        raise ValueError(f"Cannot infer quantity: invalid value={value!r}")
    if not _is_finite_positive(price):
        raise ValueError(f"Cannot infer quantity: invalid price={price!r}")
    return float(value) / float(price)


def _is_crypto_pair(ticker: str) -> bool:
    t = str(ticker).upper().strip()
    if "-" not in t:
        return False
    base, quote = t.split("-", 1)
    crypto_bases = {
        "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "LTC", "BNB",
        "AVAX", "LINK", "MATIC", "ATOM", "NEAR", "UNI", "AAVE",
        "TRX", "ETC", "DOGE", "HBAR", "SUI", "DASH", "BCH"
    }
    crypto_quotes = {"USD", "USDT", "USDC", "EUR"}
    return base in crypto_bases and quote in crypto_quotes


def _quantity_policy(*, ticker: str, asset_id: str, quantity_unit: Optional[str]) -> str:
    """
    Returns:
      - 'fractional' for crypto / derivatives / explicit exceptions
      - 'integer' for normal equities / ETFs / stocks
    """
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


def _normalize_repaired_quantity(
    *,
    raw_qty: float,
    policy: str,
) -> tuple[float, bool, str]:
    """
    Returns:
      (normalized_qty, auto_repair_ok, detail)

    - fractional: round to 8 decimals
    - integer: snap to integer only if raw qty is very close to an integer,
      otherwise mark as manual review
    """
    if not math.isfinite(raw_qty) or raw_qty <= 0.0:
        return (raw_qty, False, "raw quantity is invalid")

    if policy == "fractional":
        return (_round_qty(raw_qty), True, "fractional quantity allowed")

    nearest = round(raw_qty)
    if abs(raw_qty - nearest) <= INTEGER_QTY_TOL:
        return (float(nearest), True, "integer quantity snapped from value / price")

    return (_round_qty(raw_qty), False, "quantity is materially fractional for integer-only instrument")


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
    trade_id: str,
    as_of: str,
    s3_key: Optional[str],
) -> dict:
    if s3_key:
        obj = s3_get_json_optional(s3, bucket=BUCKET, key=s3_key)
        if isinstance(obj, dict):
            return obj

    key = dt_key(TRADES_TABLE, str(as_of), f"trade_{trade_id}.json")
    obj = s3_get_json_optional(s3, bucket=BUCKET, key=key)
    if isinstance(obj, dict):
        return obj

    raise RuntimeError(f"Could not load original trade trade_id={trade_id} as_of={as_of} s3_key={s3_key}")


def _build_edit_command_args(row: RepairPlanRow, python_entrypoint: str, dry_run: bool) -> List[str]:
    parts = [
        "poetry",
        "run",
        "python",
        python_entrypoint,
        "--mode", "edit",
        "--trade-id", str(row.trade_id),
        "--old-as-of", str(row.as_of),
    ]

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


# ----------------------------
# Core repair-plan / execution logic
# ----------------------------
def build_repair_plan(
    *,
    audit_rows_csv: str,
    out_auto_csv: str,
    out_manual_csv: str,
    python_entrypoint: str,
    only_status: Tuple[str, ...] = ("REFactor",),
    recompute_reported_pnl: bool = False,
    execute: bool = False,
    dry_run: bool = False,
    stop_on_error: bool = False,
) -> None:
    s3 = s3_client(REGION)

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
        s3_key = None if a.get("s3_key") is None else str(a.get("s3_key"))
        issue_code = str(a.get("issue_code") or "")
        ticker = str(a.get("ticker") or "").upper().strip()
        asset_id = str(a.get("asset_id") or "").strip()
        side = str(a.get("side") or "").upper().strip()
        action_tag = str(a.get("action_tag") or "").lower().strip()

        if not trade_id or not as_of:
            manual_rows.append(
                RepairPlanRow(
                    trade_id=trade_id,
                    as_of=as_of,
                    ticker=ticker,
                    asset_id=asset_id,
                    side=side,
                    action_tag=action_tag,
                    status_from_audit=status,
                    issue_code=issue_code,
                    s3_key=s3_key or "",
                    old_quantity=None,
                    new_quantity=None,
                    old_price=None,
                    new_price=None,
                    old_value=None,
                    new_value=None,
                    old_quantity_unit=None,
                    new_quantity_unit=None,
                    old_reported_pnl=None,
                    new_reported_pnl=None,
                    quantity_policy=None,
                    repair_action="MANUAL_REVIEW",
                    repair_reason="Missing trade_id or as_of in audit row.",
                    command="",
                )
            )
            exec_skipped += 1
            continue

        try:
            obj = _load_original_trade(s3, trade_id=trade_id, as_of=as_of, s3_key=s3_key)
        except Exception as e:
            manual_rows.append(
                RepairPlanRow(
                    trade_id=trade_id,
                    as_of=as_of,
                    ticker=ticker,
                    asset_id=asset_id,
                    side=side,
                    action_tag=action_tag,
                    status_from_audit=status,
                    issue_code=issue_code,
                    s3_key=s3_key or "",
                    old_quantity=None,
                    new_quantity=None,
                    old_price=None,
                    new_price=None,
                    old_value=None,
                    new_value=None,
                    old_quantity_unit=None,
                    new_quantity_unit=None,
                    old_reported_pnl=None,
                    new_reported_pnl=None,
                    quantity_policy=None,
                    repair_action="MANUAL_REVIEW",
                    repair_reason=f"Could not load original trade: {e}",
                    command="",
                )
            )
            exec_skipped += 1
            continue

        old_qty = None if obj.get("quantity") is None else float(obj.get("quantity"))
        old_px = None if obj.get("price") is None else float(obj.get("price"))
        old_val = None if obj.get("value") is None else float(obj.get("value"))
        old_unit = None if obj.get("quantity_unit") is None else str(obj.get("quantity_unit"))
        old_rpnl = None if obj.get("reported_pnl") is None else float(obj.get("reported_pnl"))

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
                    RepairPlanRow(
                        trade_id=trade_id,
                        as_of=as_of,
                        ticker=ticker,
                        asset_id=asset_id,
                        side=side,
                        action_tag=action_tag,
                        status_from_audit=status,
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
                        repair_action="MANUAL_REVIEW",
                        repair_reason=qty_detail,
                        command="",
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
                RepairPlanRow(
                    trade_id=trade_id,
                    as_of=as_of,
                    ticker=ticker,
                    asset_id=asset_id,
                    side=side,
                    action_tag=action_tag,
                    status_from_audit=status,
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
                    repair_action="MANUAL_REVIEW",
                    repair_reason="Cannot infer quantity because value and/or price are missing/invalid.",
                    command="",
                )
            )
            exec_skipped += 1
            continue

        if recompute_reported_pnl:
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
            s3_key=s3_key or dt_key(TRADES_TABLE, as_of, f"trade_{trade_id}.json"),
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

        if changed:
            cmd_args = _build_edit_command_args(row, python_entrypoint=python_entrypoint, dry_run=dry_run)
            row.command = _command_args_to_text(cmd_args)
        else:
            row.command = ""

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
                    auto_df = pd.DataFrame([asdict(r) for r in auto_rows])
                    manual_df = pd.DataFrame([asdict(r) for r in manual_rows])
                    auto_df.to_csv(out_auto_csv, index=False)
                    manual_df.to_csv(out_manual_csv, index=False)
                    raise

    auto_df = pd.DataFrame([asdict(r) for r in auto_rows])
    manual_df = pd.DataFrame([asdict(r) for r in manual_rows])

    auto_df.to_csv(out_auto_csv, index=False)
    manual_df.to_csv(out_manual_csv, index=False)

    print("\n=== TRADE REPAIR PLAN ===")
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
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the generated edit commands.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to each generated edit command.",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one executed command fails.",
    )
    args = ap.parse_args()

    statuses = tuple(x.strip() for x in str(args.statuses).split(",") if x.strip())

    build_repair_plan(
        audit_rows_csv=str(args.audit_rows_csv),
        out_auto_csv=str(args.out_auto_csv),
        out_manual_csv=str(args.out_manual_csv),
        python_entrypoint=str(args.python_entrypoint),
        only_status=statuses,
        recompute_reported_pnl=False,
        execute=bool(args.execute),
        dry_run=bool(args.dry_run),
        stop_on_error=bool(args.stop_on_error),
    )


if __name__ == "__main__":
    main()