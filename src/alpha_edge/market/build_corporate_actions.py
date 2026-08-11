from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run

import argparse
from dataclasses import asdict
from typing import Any

import pandas as pd
import yfinance as yf

from alpha_edge import paths
from alpha_edge.core.market_store import MarketStore
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import CorporateActionRow


DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"


# ----------------------------
# Runtime helpers
# ----------------------------
def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_BUCKET))


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_REGION))


def cfg_env(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "env", "dev"))


# ----------------------------
# Helpers
# ----------------------------
def _now_utc_iso() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_yahoo_symbol(sym: str) -> str:
    return str(sym or "").strip().upper()


def asdict_safe(obj: Any) -> dict:
    try:
        return asdict(obj)
    except TypeError:
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        raise TypeError(f"Cannot convert object to dict: {type(obj)}")


def _load_universe(universe_path: str) -> pd.DataFrame:
    u = pd.read_csv(universe_path)

    if "asset_id" not in u.columns:
        raise RuntimeError("Universe CSV must include 'asset_id'.")
    if "ticker" not in u.columns and "broker_ticker" not in u.columns:
        raise RuntimeError("Universe CSV must include 'ticker' or 'broker_ticker'.")

    u = u.copy()
    u["asset_id"] = u["asset_id"].astype(str).str.strip()

    if "ticker" in u.columns:
        u["ticker"] = u["ticker"].astype(str).str.upper().str.strip()
    else:
        u["ticker"] = u["broker_ticker"].astype(str).str.upper().str.strip()

    if "yahoo_ticker" in u.columns:
        u["yahoo_ticker"] = u["yahoo_ticker"].astype(str).str.strip()
        u["yahoo_ticker"] = u["yahoo_ticker"].replace(
            {
                "": None,
                "NAN": None,
                "nan": None,
                "None": None,
                "NONE": None,
            }
        )
    else:
        u["yahoo_ticker"] = None

    u["yahoo_ticker_norm"] = u["yahoo_ticker"].fillna(u["ticker"]).astype(str).map(_normalize_yahoo_symbol)

    if "include" in u.columns:
        u["include"] = pd.to_numeric(u["include"], errors="coerce").fillna(0).astype(int)
        u = u[u["include"] == 1].copy()

    u = u[(u["asset_id"] != "") & (u["ticker"] != "")].copy()
    u = u.drop_duplicates(subset=["asset_id"], keep="last").reset_index(drop=True)

    return u


def _fetch_splits_for_symbol(yahoo_ticker: str) -> pd.Series:
    t = yf.Ticker(str(yahoo_ticker).strip())
    s = t.splits

    if s is None:
        return pd.Series(dtype="float64")

    if not isinstance(s, pd.Series):
        try:
            s = pd.Series(s)
        except Exception:
            return pd.Series(dtype="float64")

    if s.empty:
        return pd.Series(dtype="float64")

    s = s.dropna().astype("float64")
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()

    return s


def _build_rows_for_asset(asset_id: str, ticker: str, yahoo_ticker: str) -> list[dict]:
    detected_at = _now_utc_iso()
    rows: list[dict] = []

    try:
        splits = _fetch_splits_for_symbol(yahoo_ticker)
    except Exception as e:
        return [
            {
                "asset_id": str(asset_id).strip(),
                "ticker": str(ticker).upper().strip(),
                "yahoo_ticker": str(yahoo_ticker).strip(),
                "effective_date": None,
                "action_type": "ERROR",
                "split_factor": None,
                "source": "yfinance",
                "source_action_id": None,
                "detected_at_utc": detected_at,
                "notes": f"split_fetch_error: {type(e).__name__}: {e}",
            }
        ]

    for idx, factor in splits.items():
        eff = pd.Timestamp(idx).date().isoformat()
        factor_f = float(factor)

        rows.append(
            asdict_safe(
                CorporateActionRow(
                    asset_id=str(asset_id).strip(),
                    ticker=str(ticker).upper().strip(),
                    yahoo_ticker=str(yahoo_ticker).strip(),
                    effective_date=eff,
                    action_type="SPLIT",
                    split_factor=factor_f,
                    source="yfinance",
                    source_action_id=f"{str(yahoo_ticker).strip()}::{eff}::SPLIT",
                    detected_at_utc=detected_at,
                    notes=None,
                )
            )
        )

    return rows


def build_corporate_actions_df(
    *,
    universe_path: str,
    asset_id: str | None = None,
    ticker: str | None = None,
    yahoo_ticker: str | None = None,
) -> pd.DataFrame:
    u = _load_universe(universe_path)

    if asset_id:
        u = u[u["asset_id"] == str(asset_id).strip()].copy()

    if ticker:
        t = str(ticker).upper().strip()
        u = u[u["ticker"] == t].copy()

    if yahoo_ticker:
        yt = str(yahoo_ticker).strip().upper()
        u = u[u["yahoo_ticker_norm"] == yt].copy()

    rows: list[dict] = []

    for _, r in u.iterrows():
        rows.extend(
            _build_rows_for_asset(
                asset_id=str(r["asset_id"]).strip(),
                ticker=str(r["ticker"]).upper().strip(),
                yahoo_ticker=str(r["yahoo_ticker_norm"]).strip(),
            )
        )

    df = pd.DataFrame(rows)

    columns = [
        "asset_id",
        "ticker",
        "yahoo_ticker",
        "effective_date",
        "action_type",
        "split_factor",
        "source",
        "source_action_id",
        "detected_at_utc",
        "notes",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    # Keep only real corporate actions in the canonical table.
    df = df[df["action_type"] == "SPLIT"].copy()

    if df.empty:
        return pd.DataFrame(columns=columns)

    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["yahoo_ticker"] = df["yahoo_ticker"].astype(str).str.strip()
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce").dt.date
    df["split_factor"] = pd.to_numeric(df["split_factor"], errors="coerce")

    df = df.dropna(subset=["asset_id", "ticker", "effective_date", "split_factor"])
    df = df.drop_duplicates(subset=["asset_id", "effective_date", "action_type"], keep="last")
    df = df.sort_values(["asset_id", "effective_date"], kind="stable").reset_index(drop=True)

    return df[columns]


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build corporate actions parquet storage from Yahoo splits.")

    ap.add_argument("--bucket", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--universe-path", default=str(paths.universe_dir() / "universe.csv"))

    ap.add_argument("--asset-id", default=None)
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--yahoo-ticker", default=None)

    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    return ap.parse_args()


def _main_impl() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)

    bucket = str(args.bucket or cfg_bucket(cfg))
    region = str(args.region or cfg_region(cfg))

    if not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    store = MarketStore(bucket=bucket, region=region)

    df = build_corporate_actions_df(
        universe_path=str(args.universe_path),
        asset_id=args.asset_id,
        ticker=args.ticker,
        yahoo_ticker=args.yahoo_ticker,
    )

    print("\n=== BUILD CORPORATE ACTIONS ===")
    print(f"env={cfg_env(cfg)}")
    print(f"rows={len(df)}")
    print(f"bucket={bucket}")
    print(f"region={region}")
    print(f"prefix={store.corporate_actions_prefix}")

    if df.empty:
        print("[OK] no corporate actions found.")
        return

    if args.dry_run:
        sample = df.head(20).copy()
        print(sample.to_string(index=False))
        return

    written = store.write_corporate_actions_partitioned(df)

    print(f"[OK] wrote partitions={len(written)}")
    for k in written[:20]:
        print(f"  s3://{bucket}/{k}")
    if len(written) > 20:
        print(f"  ... ({len(written) - 20} more)")


# ----------------------------
# Audit/logging entrypoint wrapper
# ----------------------------
def _tier1_audit_is_dry_run(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "dry_run", False) or getattr(args, "no_write", False))


def main_with_audit() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = _tier1_audit_is_dry_run(args)

    with capture_script_run(
        cfg=cfg,
        script_name="build_corporate_actions.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            _main_impl()

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="corporate_actions",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="build_corporate_actions.py",
                source_mode="corporate_actions",
                status=("dry_run" if is_dry_run else "success"),
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                    "note": "Tier 1 audit event is entrypoint-level. Detailed output keys/row counts are available in the script log stdout and script-specific metadata where emitted by the script.",
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
        except Exception as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="corporate_actions",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="build_corporate_actions.py",
                source_mode="corporate_actions",
                status="failed",
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise


def main() -> None:
    main_with_audit()


if __name__ == "__main__":
    main()
