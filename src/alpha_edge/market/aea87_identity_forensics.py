"""Read-only identity-focused forensic screening for active Alpha Edge S3 histories."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from alpha_edge import paths
from alpha_edge.core.market_store import MarketStore

PRICE = "close_adjusted_usd"
RETURN_TYPES = {"ret_adj_close_usd": "simple", "ret_close_adjusted_usd": "simple",
                "ret_log_close_adjusted_usd": "log"}


def normalize(prices, returns, asset_id, inception=None, jump=0.5, tolerance=1e-5):
    """Preserve absent prices separately; never equate a missing value to an invalid quote."""
    columns = ["asset_id", "date", "flag", "previous_price", PRICE,
               "recomputed_return", "stored_return", "return_type",
               "discrepancy", "classification", "severity"]
    if prices.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    p = prices.loc[prices["asset_id"].astype(str) == asset_id].copy()
    p["date"] = pd.to_datetime(p["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    p = p.dropna(subset=["date"]).sort_values("date")
    duplicate_dates = set(p.loc[p.duplicated("date", keep=False), "date"])
    p = p.drop_duplicates("date", keep="last").reset_index(drop=True)
    if PRICE not in p:
        p[PRICE] = np.nan
    p[PRICE] = pd.to_numeric(p[PRICE], errors="coerce")
    p["previous_price"] = p[PRICE].shift()
    valid = p[PRICE].gt(0) & p["previous_price"].gt(0)
    p["recomputed_return"] = (p[PRICE].div(p["previous_price"]) - 1).where(valid)
    p["recomputed_log_return"] = np.log(p[PRICE].div(p["previous_price"])).where(valid)
    p["stored_return"] = np.nan
    p["return_type"] = ""
    if not returns.empty and {"asset_id", "date"}.issubset(returns.columns):
        r = returns.loc[returns["asset_id"].astype(str) == asset_id].copy()
        r["date"] = pd.to_datetime(r["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
        r = r.drop_duplicates("date", keep="last")
        for name, typ in RETURN_TYPES.items():
            if name in r and pd.to_numeric(r[name], errors="coerce").notna().any():
                r["stored_return"] = pd.to_numeric(r[name], errors="coerce")
                r["return_type"] = typ
                break
        if "stored_return" in r:
            p = p.drop(columns=["stored_return", "return_type"]).merge(
                r[["date", "stored_return", "return_type"]], on="date", how="left", validate="one_to_one")
    target = p["recomputed_log_return"].where(p["return_type"].eq("log"),
                                             p["recomputed_return"])
    p["discrepancy"] = (p["stored_return"] - target).abs()
    missing = p[PRICE].isna()
    invalid = p[PRICE].notna() & (~np.isfinite(p[PRICE]) | p[PRICE].le(0))
    large = p["recomputed_return"].abs().gt(jump)
    mismatch = p["discrepancy"].gt(tolerance)
    early = p["date"].lt(pd.Timestamp(inception)) if inception else pd.Series(False, index=p.index)
    duplicate = p["date"].isin(duplicate_dates)
    flagged = invalid | large | mismatch | early | duplicate
    d = p.loc[flagged, ["date", "previous_price", PRICE, "recomputed_return",
                        "stored_return", "return_type", "discrepancy"]].copy()
    d.insert(0, "asset_id", asset_id)
    labels = [(invalid, "invalid_price"), (large, "large_move_review"),
              (mismatch, "return_mismatch"), (early, "pre_inception"),
              (duplicate, "duplicate_date")]
    d["flag"] = ["|".join(label for mask, label in labels if bool(mask.loc[i])) for i in d.index]
    d["classification"] = "REVIEW_REQUIRED"
    d["severity"] = d["recomputed_return"].abs().clip(upper=100).fillna(0) + d["discrepancy"].fillna(0)
    d.loc[early.loc[d.index], "severity"] += 200
    d = d[columns]
    p["missing_adjusted_price"] = missing
    return d, p


def load_boundaries(path):
    if not path:
        return {}
    frame = pd.read_csv(path, dtype=str).fillna("")
    if not {"asset_id", "inception_date"}.issubset(frame.columns):
        raise ValueError("inception CSV requires asset_id,inception_date")
    out = {}
    for row in frame.itertuples(index=False):
        if row.asset_id and row.inception_date:
            out[row.asset_id] = str(pd.Timestamp(row.inception_date).date())
    return out


def plot_case(frame, flags, output, inception=None, window=40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    focus = flags.sort_values("severity", ascending=False).iloc[0]["date"]
    local = frame[frame["date"].between(focus - pd.Timedelta(days=window),
                                         focus + pd.Timedelta(days=window))]
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    axs[0].plot(frame["date"], frame[PRICE])
    axs[0].set_title("Full adjusted USD price history")
    axs[1].plot(local["date"], local[PRICE], marker=".", lw=1)
    axs[1].set_title("Suspected identity boundary / price discontinuity")
    axs[2].plot(local["date"], local["recomputed_return"], label="recomputed simple")
    axs[2].plot(local["date"], local["stored_return"], label="stored (see return_type)", alpha=.7)
    axs[2].legend()
    for ax in axs:
        ax.axvline(focus, ls="--", alpha=.7)
        if inception:
            ax.axvline(pd.Timestamp(inception), color="red", ls=":", alpha=.75)
        ax.grid(alpha=.2)
    fig.suptitle(str(flags.iloc[0]["asset_id"]))
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    boundaries = load_boundaries(args.inception_csv)
    store = MarketStore(bucket=args.bucket, region=args.region)
    universe = pd.read_csv(args.universe)
    if "include" in universe:
        universe = universe[pd.to_numeric(universe["include"], errors="coerce").fillna(1).eq(1)]
    ids = sorted(set(universe["asset_id"].dropna().astype(str)) - {""})
    if args.asset_ids:
        ids = [s.strip() for s in args.asset_ids.split(",") if s.strip()]
    if args.max_assets:
        ids = ids[:args.max_assets]
    print(f"[forensics] {len(ids)} assets, READ ONLY", flush=True)
    def one(aid):
        price = store.read_ohlcv_usd([aid], start=args.start, end=args.end)
        returns = store.read_returns_usd([aid], start=args.start, end=args.end)
        return normalize(price, returns, aid, inception=boundaries.get(aid),
                         jump=args.threshold, tolerance=args.tolerance)
    findings, candidates, errors = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, aid): aid for aid in ids}
        for n, future in enumerate(as_completed(futures), 1):
            aid = futures[future]
            try:
                flags, frame = future.result()
                if not flags.empty:
                    findings.append(flags)
                    reason_counts = flags["flag"].str.split("|", regex=False).explode().value_counts()
                    identity = int(reason_counts.get("pre_inception", 0))
                    discontinuity = int(reason_counts.get("large_move_review", 0))
                    mismatches = int(reason_counts.get("return_mismatch", 0))
                    # Inception evidence outranks dramatic but uncorroborated moves.
                    score = identity * 100 + min(discontinuity, 30) * 5 + min(mismatches, 30)
                    candidates.append((score, aid, flags, frame))
            except Exception as exc:
                errors.append({"asset_id": aid, "error": str(exc)})
            if n % 50 == 0 or n == len(ids):
                print(f"[forensics] {n}/{len(ids)} done; candidates={len(candidates)} errors={len(errors)}", flush=True)
    cols = ["asset_id", "date", "flag", "previous_price", PRICE,
            "recomputed_return", "stored_return", "return_type", "discrepancy",
            "classification", "severity"]
    anomalies = pd.concat(findings, ignore_index=True) if findings else pd.DataFrame(columns=cols)
    anomalies.to_csv(output / "anomalies.csv", index=False)
    pd.DataFrame(errors, columns=["asset_id", "error"]).to_csv(output / "read_errors.csv", index=False)
    ranking = []
    for score, aid, flags, frame in sorted(candidates, key=lambda x: x[0], reverse=True):
        ranking.append({"asset_id": aid, "identity_priority_score": score,
                        "first_observation": str(frame["date"].min().date()),
                        "inception_date": boundaries.get(aid, ""),
                        "flagged_rows": len(flags),
                        "pre_inception_rows": int(flags["flag"].str.contains("pre_inception").sum()),
                        "return_mismatch_rows": int(flags["flag"].str.contains("return_mismatch").sum()),
                        "large_move_rows": int(flags["flag"].str.contains("large_move_review").sum())})
    pd.DataFrame(ranking, columns=["asset_id","identity_priority_score","first_observation",
                                   "inception_date","flagged_rows","pre_inception_rows",
                                   "return_mismatch_rows","large_move_rows"]).to_csv(output / "candidates.csv", index=False)
    chart_dir = output / "forensic_cases"
    chart_dir.mkdir(exist_ok=True)
    for i, (_, aid, flags, frame) in enumerate(sorted(candidates, key=lambda x: x[0], reverse=True)[:args.plots], 1):
        case = chart_dir / (f"{i:02d}_" + "".join(c if c.isalnum() else "_" for c in aid))
        case.mkdir(exist_ok=True)
        plot_case(frame, flags, case / "price_evidence.png", inception=boundaries.get(aid))
        (case / "evidence.json").write_text(json.dumps({
            "asset_id":aid, "validated_inception":boundaries.get(aid),
            "observed_first_date":str(frame["date"].min().date()),
            "flags":flags["flag"].value_counts().to_dict(),
            "status":"REVIEW_REQUIRED; no automated deletion or confirmation"
        }, indent=2))
    report = {"assets_scanned":len(ids),"candidate_assets":len(candidates),
              "flagged_observations":len(anomalies),"read_errors":len(errors),
              "plots":min(args.plots,len(candidates)),
              "inception_rules":len(boundaries),
              "production_modified":False}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise RuntimeError("Failed asset reads; consult read_errors.csv")


def main():
    parser = argparse.ArgumentParser(description="Read-only AEA87 identity forensics")
    parser.add_argument("--universe", default=str(paths.universe_dir() / "universe.csv"))
    parser.add_argument("--inception-csv", default=str(paths.universe_dir() / "aea87_validated_inceptions.csv"))
    parser.add_argument("--bucket", default="alpha-edge-algo")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-11-01")
    parser.add_argument("--asset-ids")
    parser.add_argument("--max-assets", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=.50)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--plots", type=int, default=15)
    parser.add_argument("--output-dir", default=str(paths.local_outputs_dir() / "aea87_forensics"))
    args = parser.parse_args()
    if not Path(args.inception_csv).exists():
        parser.error("Validated inceptions CSV not found: " + args.inception_csv)
    if args.workers < 1 or args.workers > 16 or args.plots < 0:
        parser.error("workers must be 1..16 and plots must be >=0")
    run(args)


if __name__ == "__main__":
    main()
