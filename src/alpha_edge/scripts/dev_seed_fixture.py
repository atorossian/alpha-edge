# scripts/dev_seed_fixture.py
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import boto3
import pandas as pd

from alpha_edge import paths
from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import ScoreConfig, RuntimeConfig


DEFAULT_UNIVERSE = paths.universe_dir() / "universe.csv"

WRITTEN_S3_KEYS: list[str] = []

SAMPLE_TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "TLT",
    "VCIT",
    "GLD",
    "AAPL",
    "MSFT",
    "KO",
    "VT",
]


def s3_client(cfg: RuntimeConfig):
    return boto3.client("s3", region_name=cfg.region)


def join_key(*parts: str) -> str:
    return "/".join([str(p).strip("/") for p in parts if p is not None and str(p).strip("/") != ""])


def engine_key(cfg: RuntimeConfig, *parts: str) -> str:
    return join_key(cfg.engine_root, *parts)


def dt_key(cfg: RuntimeConfig, table: str, dt_str: str, filename: str) -> str:
    return engine_key(cfg, table, f"dt={dt_str}", filename)


def put_json(s3, *, bucket: str, key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def write_json_event(
    s3,
    *,
    cfg: RuntimeConfig,
    table: str,
    dt_str: str,
    filename: str,
    payload: dict,
    update_latest: bool = False,
) -> None:
    key = dt_key(cfg, table, dt_str, filename)
    put_json(s3, bucket=cfg.bucket, key=key, payload=payload)
    WRITTEN_S3_KEYS.append(key)

    if update_latest:
        latest_key = engine_key(cfg, table, "latest.json")
        put_json(s3, bucket=cfg.bucket, key=latest_key, payload=payload)
        WRITTEN_S3_KEYS.append(latest_key)

    print(f"[S3] wrote s3://{cfg.bucket}/{key}")
    if update_latest:
        print(f"[S3] updated s3://{cfg.bucket}/{engine_key(cfg, table, 'latest.json')}")


def _id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def load_asset_map(universe_path: str | Path) -> dict[str, dict[str, Any]]:
    u = pd.read_csv(universe_path)
    required = {"ticker", "asset_id"}
    missing = required - set(u.columns)
    if missing:
        raise RuntimeError(f"Universe missing required columns: {sorted(missing)}")

    u = u.copy()
    u["ticker"] = u["ticker"].astype(str).str.upper().str.strip()
    u["asset_id"] = u["asset_id"].astype(str).str.strip()

    if "include" in u.columns:
        u["include"] = pd.to_numeric(u["include"], errors="coerce").fillna(1).astype(int)
    else:
        u["include"] = 1

    u = u[(u["ticker"] != "") & (u["asset_id"] != "")].copy()
    u = u.sort_values(["ticker", "include"], ascending=[True, False])
    u = u.drop_duplicates(subset=["ticker"], keep="first")

    out: dict[str, dict[str, Any]] = {}
    for _, r in u.iterrows():
        t = str(r["ticker"]).upper().strip()
        out[t] = {
            "ticker": t,
            "asset_id": str(r["asset_id"]).strip(),
            "currency": str(r.get("currency") or "USD").upper().strip(),
            "name": r.get("name"),
        }

    return out


def make_trade(
    *,
    ticker: str,
    asset_id: str,
    as_of: str,
    side: str,
    quantity: float,
    price: float,
    action_tag: str,
    note: str,
    currency: str = "USD",
    ts_suffix: str = "15:30:00Z",
) -> dict:
    ts_utc = f"{as_of}T{ts_suffix}"
    trade_id = _id("trade", as_of, ticker, side, quantity, price, action_tag, note)

    return {
        "trade_id": trade_id,
        "as_of": as_of,
        "ts_utc": ts_utc,
        "asset_id": asset_id,
        "ticker": ticker,
        "side": side,
        "quantity": float(quantity),
        "price": float(price),
        "currency": currency,
        "action_tag": action_tag,
        "quantity_unit": "shares",
        "value": None,
        "reported_pnl": None,
        "choice_id": None,
        "portfolio_run_id": None,
        "note": note,
    }


def make_cashflow(
    *,
    as_of: str,
    amount: float,
    kind: str,
    note: str,
    currency: str = "USD",
) -> dict:
    ts_utc = f"{as_of}T08:00:00Z"
    cashflow_id = _id("cashflow", as_of, amount, kind, note)

    return {
        "cashflow_id": cashflow_id,
        "as_of": as_of,
        "ts_utc": ts_utc,
        "type": kind,
        "amount": float(amount),
        "currency": currency,
        "note": note,
    }


def make_dividend(
    *,
    ticker: str,
    asset_id: str,
    as_of: str,
    amount: float,
    shares_held: float,
    dividend_per_share: float,
    note: str,
    currency: str = "USD",
) -> dict:
    ts_utc = f"{as_of}T12:00:00Z"
    dividend_id = _id("dividend", as_of, ticker, amount, shares_held, dividend_per_share, note)

    return {
        "dividend_id": dividend_id,
        "as_of": as_of,
        "ts_utc": ts_utc,
        "asset_id": asset_id,
        "ticker": ticker,
        "amount": float(amount),
        "currency": currency,
        "shares_held": float(shares_held),
        "dividend_per_share": float(dividend_per_share),
        "gross_amount": float(amount),
        "withholding_tax": 0.0,
        "ex_date": as_of,
        "record_date": as_of,
        "pay_date": as_of,
        "source": "dev_fixture",
        "note": note,
    }


def write_dev_universe_sample(*, universe_path: str | Path, out_path: str | Path) -> None:
    u = pd.read_csv(universe_path)
    if "ticker" not in u.columns:
        raise RuntimeError("Universe missing ticker column.")

    u = u.copy()
    u["ticker"] = u["ticker"].astype(str).str.upper().str.strip()
    sample = u[u["ticker"].isin(SAMPLE_TICKERS)].copy()

    missing = sorted(set(SAMPLE_TICKERS) - set(sample["ticker"].tolist()))
    if missing:
        raise RuntimeError(f"Missing sample tickers in universe: {missing}")

    if "include" in sample.columns:
        sample["include"] = 1

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    print(f"[LOCAL] wrote sample universe rows={len(sample)} -> {out_path}")


def seed_score_config(s3, *, cfg: RuntimeConfig, dt_str: str) -> None:
    score_cfg = ScoreConfig()

    write_json_event(
        s3,
        cfg=cfg,
        table="configs/score_config",
        dt_str=dt_str,
        filename="score_config.json",
        payload=asdict(score_cfg),
        update_latest=True,
    )


def seed_cashflows(s3, *, cfg: RuntimeConfig) -> None:
    cashflows = [
        make_cashflow(
            as_of="2024-01-02",
            amount=2500.0,
            kind="DEPOSIT",
            note="dev fixture initial deposit",
        ),
        make_cashflow(
            as_of="2024-01-25",
            amount=250.0,
            kind="WITHDRAWAL",
            note="dev fixture withdrawal",
        ),
    ]

    for cf in cashflows:
        write_json_event(
            s3,
            cfg=cfg,
            table="cashflows",
            dt_str=str(cf["as_of"]),
            filename=f"{cf['cashflow_id']}.json",
            payload=cf,
            update_latest=False,
        )


def seed_trades(s3, *, cfg: RuntimeConfig, asset_map: dict[str, dict[str, Any]]) -> None:
    def aid(t: str) -> str:
        return asset_map[t]["asset_id"]

    trades = [
        # SPY long open/add/reduce, remains open
        make_trade(
            ticker="SPY",
            asset_id=aid("SPY"),
            as_of="2024-01-03",
            side="BUY",
            quantity=2.0,
            price=470.0,
            action_tag="open",
            note="dev fixture SPY open",
        ),
        make_trade(
            ticker="SPY",
            asset_id=aid("SPY"),
            as_of="2024-01-10",
            side="BUY",
            quantity=1.0,
            price=475.0,
            action_tag="add",
            note="dev fixture SPY add",
        ),
        make_trade(
            ticker="SPY",
            asset_id=aid("SPY"),
            as_of="2024-01-24",
            side="SELL",
            quantity=1.0,
            price=485.0,
            action_tag="reduce",
            note="dev fixture SPY reduce",
        ),

        # QQQ open then full close
        make_trade(
            ticker="QQQ",
            asset_id=aid("QQQ"),
            as_of="2024-01-04",
            side="BUY",
            quantity=2.0,
            price=400.0,
            action_tag="open",
            note="dev fixture QQQ open",
        ),
        make_trade(
            ticker="QQQ",
            asset_id=aid("QQQ"),
            as_of="2024-01-26",
            side="SELL",
            quantity=2.0,
            price=420.0,
            action_tag="close",
            note="dev fixture QQQ close",
        ),

        # AAPL open remains open
        make_trade(
            ticker="AAPL",
            asset_id=aid("AAPL"),
            as_of="2024-01-05",
            side="BUY",
            quantity=5.0,
            price=180.0,
            action_tag="open",
            note="dev fixture AAPL open",
        ),

        # KO open remains open for dividend test
        make_trade(
            ticker="KO",
            asset_id=aid("KO"),
            as_of="2024-01-08",
            side="BUY",
            quantity=10.0,
            price=60.0,
            action_tag="open",
            note="dev fixture KO open",
        ),

        # GLD short open/reduce/close to test short path
        make_trade(
            ticker="GLD",
            asset_id=aid("GLD"),
            as_of="2024-01-09",
            side="SELL",
            quantity=2.0,
            price=180.0,
            action_tag="open",
            note="dev fixture GLD short open",
        ),
        make_trade(
            ticker="GLD",
            asset_id=aid("GLD"),
            as_of="2024-01-22",
            side="BUY",
            quantity=1.0,
            price=178.0,
            action_tag="reduce",
            note="dev fixture GLD short reduce",
        ),
        make_trade(
            ticker="GLD",
            asset_id=aid("GLD"),
            as_of="2024-01-29",
            side="BUY",
            quantity=1.0,
            price=176.0,
            action_tag="close",
            note="dev fixture GLD short close",
        ),
    ]

    for tr in trades:
        write_json_event(
            s3,
            cfg=cfg,
            table="trades",
            dt_str=str(tr["as_of"]),
            filename=f"{tr['trade_id']}.json",
            payload=tr,
            update_latest=False,
        )


def seed_dividends(s3, *, cfg: RuntimeConfig, asset_map: dict[str, dict[str, Any]]) -> None:
    # Artificial dev dividend. This is intentionally a fixture, not a claim that
    # this exact dividend happened on this date.
    dv = make_dividend(
        ticker="KO",
        asset_id=asset_map["KO"]["asset_id"],
        as_of="2024-01-19",
        amount=4.60,
        shares_held=10.0,
        dividend_per_share=0.46,
        note="dev fixture KO dividend",
    )

    write_json_event(
        s3,
        cfg=cfg,
        table="dividends",
        dt_str=str(dv["as_of"]),
        filename=f"{dv['dividend_id']}.json",
        payload=dv,
        update_latest=False,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Seed a minimal Alpha Edge dev fixture.")

    ap.add_argument("--env", default="dev", choices=["dev", "staging", "prod"])
    ap.add_argument("--confirm-prod-write", action="store_true")

    ap.add_argument("--universe-path", default=str(DEFAULT_UNIVERSE))
    ap.add_argument(
        "--sample-universe-out",
        default=str(paths.local_outputs_dir() / "dev" / "dev_universe_sample.csv"),
    )

    ap.add_argument("--seed-date", default="2024-01-31")
    ap.add_argument("--write-sample-universe", action="store_true")
    ap.add_argument("--write-s3-fixture", action="store_true")
    ap.add_argument("--reason", default=None, help="Optional reason recorded in audit events for S3 fixture writes.")

    return ap.parse_args()


def _main_impl(args: argparse.Namespace, cfg: RuntimeConfig) -> None:
    require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    print("\n=== DEV FIXTURE SEED ===")
    print(f"env:          {cfg.env}")
    print(f"bucket:       {cfg.bucket}")
    print(f"region:       {cfg.region}")
    print(f"engine_root:  {cfg.engine_root}")
    print(f"market_root:  {cfg.market_root}")
    print("")

    if args.write_sample_universe:
        write_dev_universe_sample(
            universe_path=args.universe_path,
            out_path=args.sample_universe_out,
        )

    if not args.write_s3_fixture:
        print("[OK] no S3 fixture requested.")
        return

    s3 = s3_client(cfg)
    asset_map = load_asset_map(args.sample_universe_out if Path(args.sample_universe_out).exists() else args.universe_path)

    for t in SAMPLE_TICKERS:
        if t not in asset_map:
            raise RuntimeError(f"Ticker {t} missing from asset map.")

    seed_score_config(s3, cfg=cfg, dt_str=str(args.seed_date))
    seed_cashflows(s3, cfg=cfg)
    seed_trades(s3, cfg=cfg, asset_map=asset_map)
    seed_dividends(s3, cfg=cfg, asset_map=asset_map)

    print("\n[OK] dev fixture seeded.")


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(args.env)

    with capture_script_run(
        cfg=cfg,
        script_name="scripts/dev_seed_fixture.py",
        input_args=vars(args),
        dry_run=False,
    ) as run_id:
        _main_impl(args, cfg)

        if args.write_s3_fixture:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="create",
                entity_type="dev_fixture",
                entity_id=str(args.seed_date),
                as_of=str(args.seed_date),
                source_script="scripts/dev_seed_fixture.py",
                source_mode="write_s3_fixture",
                status="success",
                reason=args.reason,
                input_args=vars(args),
                output_keys=WRITTEN_S3_KEYS,
                metadata={
                    "seed_date": args.seed_date,
                    "write_sample_universe": bool(args.write_sample_universe),
                    "sample_universe_out": str(args.sample_universe_out),
                    "universe_path": str(args.universe_path),
                    "output_key_count": len(WRITTEN_S3_KEYS),
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=False)


if __name__ == "__main__":
    main()