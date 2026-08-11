# bulk_import_trades.py
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation
from alpha_edge.operations.record_trade import record_trade, edit_trade


def _is_blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in ("none", "nan")


def _maybe_str(x) -> Optional[str]:
    return None if _is_blank(x) else str(x).strip()

def _maybe_float(x) -> Optional[float]:
    if _is_blank(x):
        return None

    value = float(x)

    if pd.isna(value):
        return None

    return value

def _maybe_int(x) -> Optional[int]:
    value = _maybe_float(x)
    return None if value is None else int(value)

def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x

    if _is_blank(x):
        return False

    value = str(x).strip().lower()

    if value in {"1", "true", "yes", "y"}:
        return True

    if value in {"0", "false", "no", "n"}:
        return False

    raise ValueError(f"Invalid boolean value: {x!r}")

def _float_or_default(x, default: float) -> float:
    value = _maybe_float(x)
    return float(default if value is None else value)


def _int_or_default(x, default: int) -> int:
    value = _maybe_int(x)
    return int(default if value is None else value)

def normalize_ticker_like_yahoo(raw: str) -> str:
    s = str(raw).strip().upper()

    if "/" in s:
        base, quote = [x.strip() for x in s.split("/", 1)]
        if base and quote:
            return f"{base}-{quote}"

    return s


def _normalize_side(x: str) -> str:
    s = str(x).upper().strip()
    if s not in ("BUY", "SELL"):
        raise ValueError(f"Invalid side={x!r}")
    return s


def _normalize_as_of(x: str) -> str:
    return pd.Timestamp(x).date().strftime("%Y-%m-%d")


def make_trade_id_deterministic(
    *,
    as_of: str,
    ts_utc: Optional[str],
    ticker: str,
    side: str,
    action_tag: str,
    quantity: Optional[float],
    price: Optional[float],
    value: Optional[float],
    currency: str,
) -> str:
    ts = "" if ts_utc is None else str(ts_utc).strip()

    def _fmt(x: Optional[float]) -> str:
        return "" if x is None else f"{float(x):.10f}"

    raw = "|".join(
        [
            str(as_of),
            ts,
            str(ticker),
            str(side),
            str(action_tag),
            _fmt(quantity),
            _fmt(price),
            _fmt(value),
            str(currency),
        ]
    )

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{as_of.replace('-', '')}-{digest}"


def import_csv(
    *,
    cfg: RuntimeConfig,
    csv_path: str,
    dry_run: bool = False,
    limit: int | None = None,
    out_dir: str = "bulk_logs",
    print_every: int = 50,
    max_retries: int = 3,
    retry_sleep_sec: float = 1.0,
    universe_path: Optional[str] = None,
    universe_key: Optional[str] = None,
    strict_universe: bool = False,
    run_id: Optional[str] = None,
    input_args: Optional[Dict[str, Any]] = None,
    indicators_snapshot_key: Optional[str] = None,
    indicators_root_prefix: Optional[str] = None,
) -> None:
    df = pd.read_csv(csv_path)

    required = {
        "as_of",
        "ticker",
        "side",
        "action_tag",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    optional_defaults = {
        "currency": "USD",
        "ts_utc": None,
        "note": None,
        "choice_id": None,
        "portfolio_run_id": None,
        "quantity": None,
        "price": None,
        "value": None,
        "quantity_unit": None,
        "reported_pnl": None,
        "trade_id": None,
        "asset_id": None,
        "open_value": None,
        "entry_price": None,
        "asset_class": None,
        "risk_model": "fixed_by_asset_class",
        "indicator_mode": "auto",
        "stop_loss_pct": None,
        "target_profit_pct": None,
        "max_holding_days": None,
        "disable_risk_contract": False,
        "atr_stop_multiplier": 2.0,
        "atr_target_multiplier": 4.0,
        "volatility_stop_multiplier": 1.0,
        "volatility_target_multiplier": 1.8,
        "reward_multiple": 2.0,
        "min_stop_pct": 0.02,
        "max_stop_pct": 0.25,
        "max_target_pct": 0.95,
        "max_indicator_staleness_days": 10,
    }

    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    # ----------------------------
    # Normalize dataframe columns
    # ----------------------------

    numeric_columns = {
        "quantity",
        "price",
        "value",
        "reported_pnl",
        "open_value",
        "entry_price",
        "stop_loss_pct",
        "target_profit_pct",
        "max_holding_days",
        "atr_stop_multiplier",
        "atr_target_multiplier",
        "volatility_stop_multiplier",
        "volatility_target_multiplier",
        "reward_multiple",
        "min_stop_pct",
        "max_stop_pct",
        "max_target_pct",
        "max_indicator_staleness_days",
    }

    optional_string_columns = {
        "ts_utc",
        "note",
        "choice_id",
        "portfolio_run_id",
        "action_tag",
        "quantity_unit",
        "asset_id",
        "asset_class",
        "risk_model",
        "indicator_mode",
    }

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in optional_string_columns:
        df[col] = df[col].apply(_maybe_str)

    df["as_of"] = df["as_of"].apply(_normalize_as_of)

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .apply(normalize_ticker_like_yahoo)
    )

    df["side"] = df["side"].apply(_normalize_side)

    df["currency"] = (
        df["currency"]
        .apply(_maybe_str)
        .fillna("USD")
        .str.upper()
        .str.strip()
    )

    df["action_tag"] = (
        df["action_tag"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["quantity_unit"] = (
        df["quantity_unit"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["asset_class"] = (
        df["asset_class"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["risk_model"] = (
        df["risk_model"]
        .fillna("fixed_by_asset_class")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["indicator_mode"] = (
        df["indicator_mode"]
        .fillna("auto")
        .astype(str)
        .str.strip()
        .str.lower()
    )


    valid_actions = {"open", "close", "add", "reduce"}

    invalid_actions = df.loc[
        ~df["action_tag"].isin(valid_actions),
        "action_tag",
    ]

    if not invalid_actions.empty:
        invalid_values = sorted(
            invalid_actions.dropna().astype(str).unique().tolist()
        )

        raise ValueError(
            "CSV contains invalid action_tag values: "
            f"{invalid_values}"
        )
    
    if df["action_tag"].isna().any():
        bad_rows = df.index[df["action_tag"].isna()].tolist()

        raise ValueError(
            "CSV contains empty action_tag values at rows: "
            f"{bad_rows[:20]}"
        )

    valid_risk_models = {
        "fixed_by_asset_class",
        "atr_based",
        "volatility_based",
        "hybrid",
    }

    invalid_risk_models = df.loc[
        ~df["risk_model"].isin(valid_risk_models),
        "risk_model",
    ]

    if not invalid_risk_models.empty:
        invalid_values = sorted(
            invalid_risk_models.dropna().astype(str).unique().tolist()
        )

        raise ValueError(
            "CSV contains invalid risk_model values: "
            f"{invalid_values}"
        )

    valid_indicator_modes = {
        "auto",
        "latest",
        "point_in_time",
    }

    invalid_indicator_modes = df.loc[
        ~df["indicator_mode"].isin(valid_indicator_modes),
        "indicator_mode",
    ]

    if not invalid_indicator_modes.empty:
        invalid_values = sorted(
            invalid_indicator_modes.dropna().astype(str).unique().tolist()
        )

        raise ValueError(
            "CSV contains invalid indicator_mode values: "
            f"{invalid_values}"
        )

    def _fill_trade_id(row) -> str:
        if not _is_blank(row.get("trade_id")):
            return str(row["trade_id"]).strip()

        return make_trade_id_deterministic(
            as_of=str(row["as_of"]),
            ts_utc=row["ts_utc"],
            ticker=str(row["ticker"]),
            side=str(row["side"]),
            action_tag=str(row["action_tag"]),
            quantity=_maybe_float(row["quantity"]),
            price=_maybe_float(row["price"]),
            value=_maybe_float(row["value"]),
            currency=str(row["currency"]) or "USD",
        )

    df["trade_id"] = df.apply(_fill_trade_id, axis=1)

    dup = int(df["trade_id"].duplicated().sum())
    if dup > 0:
        raise ValueError(
            f"CSV contains {dup} duplicate trade_id values. "
            "Fix upstream ID generation or include a stable broker/export trade id."
        )

    n = len(df) if limit is None else min(len(df), int(limit))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    errors_path = Path(out_dir) / "failed_rows.csv"
    ok_path = Path(out_dir) / "uploaded_rows.csv"

    err_fields = [
        "row_idx",
        "trade_id",
        "as_of",
        "ticker",
        "asset_id",
        "side",
        "quantity",
        "price",
        "currency",
        "ts_utc",
        "action_tag",
        "quantity_unit",
        "value",
        "reported_pnl",
        "choice_id",
        "portfolio_run_id",
        "note",
        "error",
    ]

    ok_fields = [
        "row_idx",
        "trade_id",
        "as_of",
        "ticker",
        "asset_id",
        "side",
        "quantity",
        "price",
        "currency",
        "ts_utc",
        "action_tag",
        "quantity_unit",
        "value",
        "reported_pnl",
        "choice_id",
        "portfolio_run_id",
        "note",
    ]

    ok = 0
    failed = 0

    print(
        f"[bulk] env={cfg.env} bucket={cfg.bucket} engine_root={cfg.engine_root} "
        f"csv={csv_path} rows={len(df)} importing={n} dry_run={dry_run}"
    )
    print(f"[bulk] writing logs to: {errors_path} and {ok_path}")
    print("[bulk] trade_id is guaranteed: CSV-provided or deterministic.")

    with open(errors_path, "w", newline="", encoding="utf-8") as errors_f, open(
        ok_path, "w", newline="", encoding="utf-8"
    ) as ok_f:
        err_writer = csv.DictWriter(errors_f, fieldnames=err_fields)
        ok_writer = csv.DictWriter(ok_f, fieldnames=ok_fields)
        err_writer.writeheader()
        ok_writer.writeheader()

        for i in range(n):
            row = df.iloc[i]

            payload = {
                "row_idx": i,
                "trade_id": str(row["trade_id"]),
                "as_of": str(row["as_of"]),
                "ticker": str(row["ticker"]),
                "asset_id": row["asset_id"],
                "side": str(row["side"]),
                "action_tag": row["action_tag"],

                "quantity": _maybe_float(row["quantity"]),
                "price": _maybe_float(row["price"]),
                "value": _maybe_float(row["value"]),

                "currency": (
                    str(row["currency"]).upper().strip()
                    if pd.notna(row["currency"])
                    else "USD"
                ),
                "ts_utc": row["ts_utc"],
                "choice_id": row["choice_id"],
                "portfolio_run_id": row["portfolio_run_id"],
                "note": row["note"],

                "quantity_unit": row["quantity_unit"],
                "reported_pnl": _maybe_float(row["reported_pnl"]),
                "open_value": _maybe_float(row["open_value"]),
                "entry_price": _maybe_float(row["entry_price"]),
                "risk_model": (
                    _maybe_str(row["risk_model"])
                    or "fixed_by_asset_class"
                ),
                "indicator_mode": (
                    _maybe_str(row["indicator_mode"])
                    or "auto"
                ),
                "asset_class": _maybe_str(row["asset_class"]),

                "stop_loss_pct": _maybe_float(
                    row["stop_loss_pct"]
                ),
                "target_profit_pct": _maybe_float(
                    row["target_profit_pct"]
                ),
                "max_holding_days": _maybe_int(
                    row["max_holding_days"]
                ),
                "disable_risk_contract": _as_bool(
                    row["disable_risk_contract"]
                ),
                "atr_stop_multiplier": _float_or_default(
                    row["atr_stop_multiplier"],
                    2.0,
                ),
                "atr_target_multiplier": _float_or_default(
                    row["atr_target_multiplier"],
                    4.0,
                ),
                "volatility_stop_multiplier": _float_or_default(
                    row["volatility_stop_multiplier"],
                    1.0,
                ),
                "volatility_target_multiplier": _float_or_default(
                    row["volatility_target_multiplier"],
                    1.8,
                ),
                "reward_multiple": _float_or_default(
                    row["reward_multiple"],
                    2.0,
                ),
                "min_stop_pct": _float_or_default(
                    row["min_stop_pct"],
                    0.02,
                ),
                "max_stop_pct": _float_or_default(
                    row["max_stop_pct"],
                    0.25,
                ),
                "max_target_pct": _float_or_default(
                    row["max_target_pct"],
                    0.95,
                ),
                "max_indicator_staleness_days": _int_or_default(
                    row["max_indicator_staleness_days"],
                    10,
                ),
            }

            try:
                if payload["side"] not in ("BUY", "SELL"):
                    raise ValueError(f"Invalid side={payload['side']}")
                provided_economics = sum(
                    payload[name] is not None
                    for name in ["quantity", "price", "value"]
                )

                if provided_economics < 2:
                    raise ValueError(
                        "Each trade requires at least two of quantity, price, value."
                    )

                for name in ["quantity", "price", "value"]:
                    field_value = payload[name]

                    if field_value is not None and field_value <= 0:
                        raise ValueError(
                            f"{name} must be > 0 when provided."
                        )

                last_exc: Exception | None = None

                for attempt in range(max_retries + 1):
                    try:
                        record_trade(
                            cfg=cfg,
                            as_of=payload["as_of"],
                            ticker=payload["ticker"],
                            side=payload["side"],
                            action_tag=payload["action_tag"],

                            quantity=payload["quantity"],
                            price=payload["price"],
                            value=payload["value"],

                            currency=payload["currency"],
                            trade_id=payload["trade_id"],
                            ts_utc=payload["ts_utc"],
                            asset_id=payload["asset_id"],

                            universe_path=universe_path,
                            universe_key=universe_key,
                            strict_universe=strict_universe,

                            quantity_unit=payload["quantity_unit"],
                            reported_pnl=payload["reported_pnl"],
                            open_value=payload["open_value"],
                            entry_price=payload["entry_price"],
                            asset_class=payload["asset_class"],

                            risk_model=payload["risk_model"],
                            indicator_mode=payload["indicator_mode"],
                            indicators_snapshot_key=indicators_snapshot_key,
                            indicators_root_prefix=indicators_root_prefix,

                            stop_loss_pct=payload["stop_loss_pct"],
                            target_profit_pct=payload["target_profit_pct"],
                            max_holding_days=payload["max_holding_days"],
                            disable_risk_contract=(
                                payload["disable_risk_contract"]
                            ),

                            max_indicator_staleness_days=(
                                payload["max_indicator_staleness_days"]
                            ),
                            atr_stop_multiplier=payload[
                                "atr_stop_multiplier"
                            ],
                            atr_target_multiplier=payload[
                                "atr_target_multiplier"
                            ],
                            volatility_stop_multiplier=payload[
                                "volatility_stop_multiplier"
                            ],
                            volatility_target_multiplier=payload[
                                "volatility_target_multiplier"
                            ],
                            reward_multiple=payload["reward_multiple"],
                            min_stop_pct=payload["min_stop_pct"],
                            max_stop_pct=payload["max_stop_pct"],
                            max_target_pct=payload["max_target_pct"],

                            choice_id=payload["choice_id"],
                            portfolio_run_id=payload["portfolio_run_id"],
                            note=payload["note"],

                            dry_run=dry_run,
                            run_id=run_id,
                            source_script="bulk_import_trades.py",
                            source_mode="bulk_record_row",
                            input_args={
                                **dict(input_args or {}),
                                "row_idx": i,
                                "trade_id": payload["trade_id"],
                            },
                            reason="bulk_import_trades",
                        )
                        last_exc = None
                        break

                    except Exception as e:
                        last_exc = e
                        if attempt < max_retries:
                            time.sleep(retry_sleep_sec * (attempt + 1))

                if last_exc is not None:
                    raise last_exc

                ok_writer.writerow({k: payload.get(k) for k in ok_fields})
                ok_f.flush()
                ok += 1

            except Exception as e:
                failed += 1
                payload["error"] = f"{type(e).__name__}: {e}"
                err_writer.writerow({k: payload.get(k) for k in err_fields})
                errors_f.flush()

            if (i + 1) % print_every == 0:
                print(f"[bulk] progress {i + 1}/{n} ok={ok} failed={failed}")

    audit = build_audit_event(
        cfg=cfg,
        run_id=run_id,
        event_type="import",
        entity_type="trades_batch",
        entity_id=str(Path(csv_path).name),
        as_of=None,
        source_script="bulk_import_trades.py",
        source_mode="bulk_import",
        status=("dry_run" if dry_run else "success"),
        reason="bulk import trades from CSV",
        input_args=input_args,
        output_keys=[],
        metadata={
            "csv_path": str(csv_path),
            "rows_in_csv": int(len(df)),
            "rows_requested": int(n),
            "rows_uploaded": int(ok),
            "rows_failed": int(failed),
            "uploaded_rows_log": str(ok_path),
            "failed_rows_log": str(errors_path),
        },
    )
    write_audit_event(cfg=cfg, event=audit, dry_run=bool(dry_run))

    print(f"[bulk] done ok={ok} failed={failed}")
    print(f"[bulk] failures saved to {errors_path}")



# NOTE: edit mode supports ts_utc patches for broker timestamp corrections.
def edit_csv(
    *,
    cfg: RuntimeConfig,
    csv_path: str,
    dry_run: bool = False,
    limit: int | None = None,
    out_dir: str = "bulk_edit_logs",
    print_every: int = 50,
    max_retries: int = 3,
    retry_sleep_sec: float = 1.0,
    run_id: Optional[str] = None,
    input_args: Optional[Dict[str, Any]] = None,
    no_backup: bool = False,
) -> None:
    df = pd.read_csv(csv_path)

    required = {"trade_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    optional_defaults = {
        "old_as_of": None,
        "new_as_of": None,
        "ts_utc": None,
        "quantity": None,
        "price": None,
        "value": None,
        "reported_pnl": None,
        "currency": None,
        "note": None,
        "reason": "bulk_edit_trades",
        "note_append": None,
        "fields_to_update": None,
    }

    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    numeric_columns = {"quantity", "price", "value", "reported_pnl"}
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n = len(df) if limit is None else min(len(df), int(limit))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    errors_path = Path(out_dir) / "failed_rows.csv"
    ok_path = Path(out_dir) / "edited_rows.csv"

    ok_fields = [
        "row_idx",
        "trade_id",
        "old_as_of",
        "new_as_of",
        "ts_utc",
        "quantity",
        "price",
        "value",
        "reported_pnl",
        "currency",
        "note",
        "reason",
        "fields_to_update",
    ]
    err_fields = ok_fields + ["error"]

    ok = 0
    failed = 0

    print(
        f"[bulk-edit] env={cfg.env} bucket={cfg.bucket} engine_root={cfg.engine_root} "
        f"csv={csv_path} rows={len(df)} editing={n} dry_run={dry_run}"
    )
    print(f"[bulk-edit] writing logs to: {errors_path} and {ok_path}")

    with open(errors_path, "w", newline="", encoding="utf-8") as errors_f, open(
        ok_path, "w", newline="", encoding="utf-8"
    ) as ok_f:
        err_writer = csv.DictWriter(errors_f, fieldnames=err_fields)
        ok_writer = csv.DictWriter(ok_f, fieldnames=ok_fields)
        err_writer.writeheader()
        ok_writer.writeheader()

        for i in range(n):
            row = df.iloc[i]

            payload = {
                "row_idx": i,
                "trade_id": str(row["trade_id"]).strip(),
                "old_as_of": _maybe_str(row["old_as_of"]),
                "new_as_of": _maybe_str(row["new_as_of"]),
                "ts_utc": _maybe_str(row["ts_utc"]),
                "quantity": _maybe_float(row["quantity"]),
                "price": _maybe_float(row["price"]),
                "value": _maybe_float(row["value"]),
                "reported_pnl": _maybe_float(row["reported_pnl"]),
                "currency": _maybe_str(row["currency"]),
                "note": _maybe_str(row["note"]),
                "reason": _maybe_str(row["reason"]) or "bulk_edit_trades",
                "fields_to_update": _maybe_str(row["fields_to_update"]),
            }

            try:
                if not payload["trade_id"]:
                    raise ValueError("trade_id is required.")

                patch: dict[str, Any] = {}

                for name in ["ts_utc", "quantity", "price", "value", "reported_pnl", "currency", "note"]:
                    if payload[name] is not None:
                        patch[name] = payload[name]

                if not patch and payload["new_as_of"] is None:
                    raise ValueError("Nothing to edit for this row.")

                last_exc: Exception | None = None

                for attempt in range(max_retries + 1):
                    try:
                        edit_trade(
                            cfg=cfg,
                            trade_id=payload["trade_id"],
                            old_as_of=payload["old_as_of"],
                            new_as_of=payload["new_as_of"],
                            patch=patch,
                            dry_run=dry_run,
                            write_backup=(not bool(no_backup)),
                            run_id=run_id,
                            input_args={
                                **dict(input_args or {}),
                                "row_idx": i,
                                "trade_id": payload["trade_id"],
                                "patch_keys": sorted(patch.keys()),
                            },
                            reason=payload["reason"],
                        )
                        last_exc = None
                        break

                    except Exception as e:
                        last_exc = e
                        if attempt < max_retries:
                            time.sleep(retry_sleep_sec * (attempt + 1))

                if last_exc is not None:
                    raise last_exc

                ok_writer.writerow({k: payload.get(k) for k in ok_fields})
                ok_f.flush()
                ok += 1

            except Exception as e:
                failed += 1
                payload["error"] = f"{type(e).__name__}: {e}"
                err_writer.writerow({k: payload.get(k) for k in err_fields})
                errors_f.flush()

            if (i + 1) % print_every == 0:
                print(f"[bulk-edit] progress {i + 1}/{n} ok={ok} failed={failed}")

    audit = build_audit_event(
        cfg=cfg,
        run_id=run_id,
        event_type="modify",
        entity_type="trades_batch",
        entity_id=str(Path(csv_path).name),
        as_of=None,
        source_script="bulk_import_trades.py",
        source_mode="bulk_edit",
        status=("dry_run" if dry_run else "success"),
        reason="bulk edit trades from CSV",
        input_args=input_args,
        output_keys=[],
        metadata={
            "csv_path": str(csv_path),
            "rows_in_csv": int(len(df)),
            "rows_requested": int(n),
            "rows_edited": int(ok),
            "rows_failed": int(failed),
            "edited_rows_log": str(ok_path),
            "failed_rows_log": str(errors_path),
        },
    )
    write_audit_event(cfg=cfg, event=audit, dry_run=bool(dry_run))

    print(f"[bulk-edit] done ok={ok} failed={failed}")
    print(f"[bulk-edit] failures saved to {errors_path}")



def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk import or bulk edit trades from CSV.")

    ap.add_argument("--mode", choices=["import", "edit"], default="import")
    ap.add_argument("--csv", required=True, help="Path to CSV file.")
    ap.add_argument("--dry-run", action="store_true", help="Validate and print S3 keys but do not write.")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N rows.")
    ap.add_argument("--out-dir", default="bulk_logs", help="Output directory for logs.")
    ap.add_argument("--print-every", type=int, default=50, help="Progress print frequency.")
    ap.add_argument("--max-retries", type=int, default=3, help="Retries per row on failure.")
    ap.add_argument("--retry-sleep-sec", type=float, default=1.0, help="Base sleep seconds between retries.")

    ap.add_argument("--universe-path", default=None, help="Local universe.csv path for asset_id resolution.")
    ap.add_argument("--universe-key", default=None, help="S3 universe.csv key for asset_id resolution.")
    ap.add_argument("--strict-universe", action="store_true", help="Fail rows where asset_id cannot be resolved.")

    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")
    ap.add_argument("--no-backup", action="store_true", help="Edit mode only. Disable per-trade backups; not recommended.")
    ap.add_argument(
        "--indicators-snapshot-key",
        default=None,
    )
    ap.add_argument(
        "--indicators-root-prefix",
        default=None,
    )

    args = ap.parse_args()

    cfg = load_runtime_config(args.env)

    if not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    input_args = vars(args)

    with capture_script_run(
        cfg=cfg,
        script_name="bulk_import_trades.py",
        input_args=input_args,
        dry_run=bool(args.dry_run),
    ) as run_id:
        if args.mode == "edit":
            edit_csv(
                cfg=cfg,
                csv_path=args.csv,
                dry_run=bool(args.dry_run),
                limit=args.limit,
                out_dir=str(args.out_dir),
                print_every=int(args.print_every),
                max_retries=int(args.max_retries),
                retry_sleep_sec=float(args.retry_sleep_sec),
                run_id=run_id,
                input_args=input_args,
                no_backup=bool(args.no_backup),
            )
            return

        import_csv(
            cfg=cfg,
            csv_path=args.csv,
            dry_run=bool(args.dry_run),
            limit=args.limit,
            out_dir=str(args.out_dir),
            print_every=int(args.print_every),
            max_retries=int(args.max_retries),
            retry_sleep_sec=float(args.retry_sleep_sec),
            universe_path=args.universe_path,
            universe_key=args.universe_key,
            strict_universe=bool(args.strict_universe),
            run_id=run_id,
            input_args=input_args,
            indicators_snapshot_key=args.indicators_snapshot_key,
            indicators_root_prefix=args.indicators_root_prefix,
        )


if __name__ == "__main__":
    main()
