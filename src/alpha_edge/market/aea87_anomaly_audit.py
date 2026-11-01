"""Read-only AEA-87 price/return forensic report from active S3 manifests.

Run from src: poetry run python -m alpha_edge.market.aea87_anomaly_audit
This command never calls any S3 write method.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_edge.core.market_store import MarketStore
from alpha_edge import paths


def inspect_asset(prices: pd.DataFrame, returns: pd.DataFrame, asset_id: str,
                  jump_threshold: float = 0.5, mismatch_tolerance: float = 1e-5):
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    p = p[p["asset_id"].astype(str) == asset_id].dropna(subset=["date"])
    p["close_adjusted_usd"] = pd.to_numeric(p["close_adjusted_usd"], errors="coerce")
    p = p.sort_values("date")
    duplicate_dates = set(p.loc[p.duplicated("date", keep=False), "date"])
    p = p.drop_duplicates("date", keep="last")
    p["previous_price"] = p["close_adjusted_usd"].shift()
    p["recomputed_return"] = p["close_adjusted_usd"] / p["previous_price"] - 1
    r = returns.copy()
    if not r.empty and {"asset_id", "date"}.issubset(r.columns):
        r = r[r["asset_id"].astype(str) == asset_id].copy()
        r["date"] = pd.to_datetime(r["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        r = r.drop_duplicates("date", keep="last")
        col = next((c for c in ("ret_adj_close_usd", "ret_close_adjusted_usd") if c in r), None)
        if col:
            r = r[["date", col]].rename(columns={col: "stored_return"})
            r["stored_return"] = pd.to_numeric(r["stored_return"], errors="coerce")
            p = p.merge(r, on="date", how="left", validate="one_to_one")
    if "stored_return" not in p:
        p["stored_return"] = np.nan
    p["discrepancy"] = (p["stored_return"] - p["recomputed_return"]).abs()
    nonpositive = p["close_adjusted_usd"].le(0) | ~np.isfinite(p["close_adjusted_usd"])
    jump = p["recomputed_return"].abs().gt(jump_threshold)
    mismatch = p["discrepancy"].gt(mismatch_tolerance)
    flags = nonpositive | jump | mismatch | p["date"].isin(duplicate_dates)
    d = p.loc[flags, ["date", "close_adjusted_usd", "previous_price",
                       "recomputed_return", "stored_return", "discrepancy"]].copy()
    d.insert(0, "asset_id", asset_id)
    d["flag"] = [
        "|".join(x for x, yes in (
            ("invalid_price", bool(nonpositive.loc[idx])),
            ("large_return_review", bool(jump.loc[idx])),
            ("return_mismatch", bool(mismatch.loc[idx])),
            ("duplicate_price_date", bool(p.loc[idx, "date"] in duplicate_dates)),
        ) if yes)
        for idx in d.index
    ]
    # A threshold breach is a review candidate, NOT confirmed corruption.
    d["classification"] = "REVIEW_REQUIRED"
    d["severity"] = np.maximum(d["recomputed_return"].abs().fillna(0),
                                 d["discrepancy"].fillna(0))
    return d, p


def plot_asset(frame: pd.DataFrame, anomalies: pd.DataFrame, dest: Path, window=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = frame.sort_values("date")
    if frame.empty or anomalies.empty:
        return
    worst = anomalies.sort_values("severity", ascending=False).iloc[0]
    focal = worst["date"]
    local = frame[(frame["date"] >= focal - pd.Timedelta(days=window)) &
                  (frame["date"] <= focal + pd.Timedelta(days=window))]
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    axs[0].plot(frame["date"], frame["close_adjusted_usd"], lw=0.8)
    axs[0].set_title("Full adjusted USD price history")
    axs[1].plot(local["date"], local["close_adjusted_usd"], marker=".", lw=1)
    axs[1].axvline(focal, linestyle="--")
    axs[1].set_title("Price near flagged observation")
    axs[2].plot(local["date"], local["recomputed_return"], label="recalculated")
    axs[2].plot(local["date"], local["stored_return"], linestyle="--", label="stored")
    axs[2].axvline(focal, linestyle="--")
    axs[2].legend()
    axs[2].set_title("Simple daily returns (decimal)")
    fig.suptitle(str(worst["asset_id"]) + " | " + str(focal.date()))
    for ax in axs:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def run(args):
    store = MarketStore(bucket=args.bucket, region=args.region)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(args.universe)
    if "asset_id" not in universe.columns:
        raise ValueError("universe must include asset_id")
    if "include" in universe:
        universe = universe.loc[pd.to_numeric(universe["include"], errors="coerce").fillna(1) == 1]
    ids = sorted(set(universe["asset_id"].dropna().astype(str).str.strip()) - {""})
    if args.asset_ids:
        ids = [x.strip() for x in args.asset_ids.split(",") if x.strip()]
    if args.max_assets:
        ids = ids[:args.max_assets]
    print(f"[audit] {len(ids):,} assets; active S3 histories; READ ONLY", flush=True)

    def one(aid):
        prices = store.read_ohlcv_usd([aid], start=args.start, end=args.end)
        rets = store.read_returns_usd([aid], start=args.start, end=args.end)
        if not prices.empty and not {"date", "asset_id", "close_adjusted_usd"}.issubset(prices):
            raise ValueError("Price columns missing for " + aid)
        return inspect_asset(prices, rets, aid, args.threshold, args.mismatch_tolerance)

    findings, chart_choices, errors = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(one, aid): aid for aid in ids}
        for n, job in enumerate(as_completed(jobs), 1):
            aid = jobs[job]
            try:
                findings_df, full = job.result()
                if not findings_df.empty:
                    findings.append(findings_df)
                    chart_choices.append((float(findings_df["severity"].max()), aid, full, findings_df))
            except Exception as e:
                errors.append({"asset_id": aid, "error": str(e)})
            if n % 50 == 0 or n == len(ids):
                print(f"[audit] {n:,}/{len(ids):,} assets, flagged={len(chart_choices)}, errors={len(errors)}", flush=True)
    columns = ["asset_id", "date", "previous_price", "close_adjusted_usd",
               "recomputed_return", "stored_return", "discrepancy", "flag",
               "classification", "severity"]
    anomalies = (pd.concat(findings, ignore_index=True) if findings
                 else pd.DataFrame(columns=columns))
    anomalies = anomalies.sort_values("severity", ascending=False)
    anomalies.to_csv(output / "anomalies.csv", index=False)
    pd.DataFrame(errors, columns=["asset_id", "error"]).to_csv(output / "read_errors.csv", index=False)
    chart_dir = output / "plots"
    chart_dir.mkdir(exist_ok=True)
    selected = sorted(chart_choices, key=lambda x: x[0], reverse=True)[:args.plots]
    for i, (_, aid, frame, flagged) in enumerate(selected, 1):
        filename = f"{i:02d}_{''.join(c if c.isalnum() else '_' for c in aid)}.png"
        plot_asset(frame, flagged, chart_dir / filename, args.window)
    summary = {"assets_attempted": len(ids), "assets_flagged": len(chart_choices),
               "flagged_observations": len(anomalies), "read_errors": len(errors),
               "plots": len(selected), "output_dir": str(output),
               "classification_note": "Review candidates only; no automatic repairs or confirmed-corruption assertions"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} asset reads failed; see read_errors.csv")


def main():
    p = argparse.ArgumentParser(description="Read-only forensic audit of S3 market histories")
    p.add_argument("--universe", default=str(paths.universe_dir() / "universe.csv"))
    p.add_argument("--bucket", default="alpha-edge-algo")
    p.add_argument("--region", default="eu-west-1")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-11-01")
    p.add_argument("--asset-ids", default=None, help="Comma-separated asset IDs for focused audit")
    p.add_argument("--max-assets", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--mismatch-tolerance", type=float, default=1e-5)
    p.add_argument("--plots", type=int, default=15)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--output-dir", default=str(paths.data_dir() / "outputs" / "aea87_anomaly_audit"))
    args = p.parse_args()
    if not 1 <= args.workers <= 16 or args.plots < 0:
        p.error("workers must be 1..16 and plots must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
