# run_universe_triage.py
import pandas as pd
from pathlib import Path

from alpha_edge.universe.universe_triage import triage_failures, write_triage_outputs_local
from alpha_edge.core.market_store import MarketStore
from alpha_edge import paths

def run_post_ingest_triage(
    *,
    store: MarketStore,
    as_of: str,
    universe_csv: str,
    overrides_csv: str,
    excluded_csv: str,
    mapping_changes: pd.DataFrame | None = None,
    mapping_validation: pd.DataFrame | None = None,
    verbose: bool = True,
    sample_n: int = 10,
    local_out_dir: str | None = None,
) -> None:
    # normalize mapping artifacts so outputs exist every day (even empty)
    if mapping_changes is None:
        mapping_changes = pd.DataFrame()
    if mapping_validation is None:
        mapping_validation = pd.DataFrame()

    failures_path = (
        f"s3://{store.bucket}/{store.base_prefix}/ingest_failures/"
        f"{store.version}/dt={as_of}/failures.parquet"
    )

    if verbose:
        print(f"[triage] as_of={as_of}")
        print(f"[triage] failures_path={failures_path}")
        print(f"[triage] universe_csv={universe_csv}")
        print(f"[triage] overrides_csv={overrides_csv} exists={Path(overrides_csv).exists()}")
        print(f"[triage] excluded_csv={excluded_csv} exists={Path(excluded_csv).exists()}")

    try:
        fails = pd.read_parquet(failures_path)
        if verbose:
            print(f"[triage] failures_loaded rows={len(fails)} cols={list(fails.columns)}")
    except Exception as e:
        fails = pd.DataFrame()
        if verbose:
            print(f"[triage] failures_load_failed -> treating as empty. err={str(e)[:200]}")

    # defaults (keep schema stable)
    triage_report = pd.DataFrame()
    sug_overrides = pd.DataFrame(
        columns=["ticker", "yahoo_ticker", "lock_yahoo_ticker", "exclude", "exclude_reason", "expected_name", "note"]
    )
    sug_exclusions = pd.DataFrame(columns=["ticker", "asset_class", "reason"])

    # Only do work if failures exist
    if fails is not None and not fails.empty:
        universe = pd.read_csv(universe_csv)
        overrides = pd.read_csv(overrides_csv) if Path(overrides_csv).exists() else pd.DataFrame()
        excluded = pd.read_csv(excluded_csv) if Path(excluded_csv).exists() else pd.DataFrame()

        if verbose:
            print(f"[triage] universe_loaded rows={len(universe)} include_col={'include' in universe.columns}")
            print(f"[triage] overrides_loaded rows={len(overrides)} cols={list(overrides.columns) if not overrides.empty else []}")
            print(f"[triage] excluded_loaded rows={len(excluded)} cols={list(excluded.columns) if not excluded.empty else []}")

            # quick look at failures
            cols = [c for c in ["ticker", "yahoo_ticker", "reason", "error"] if c in fails.columns]
            if cols:
                print("[triage] failures_sample:")
                print(fails[cols].head(sample_n).to_string(index=False))

        triage_report, sug_overrides, sug_exclusions = triage_failures(
            fails=fails,
            universe=universe,
            overrides=overrides,
            excluded=excluded,
            verbose=verbose,      # NEW
            sample_n=sample_n,    # NEW
        )

        if verbose:
            print(f"[triage] triage_report rows={len(triage_report)}")
            print(f"[triage] suggested_overrides rows={len(sug_overrides)}")
            print(f"[triage] suggested_exclusions rows={len(sug_exclusions)}")

            if not triage_report.empty and "classification" in triage_report.columns:
                print("[triage] classification_counts:")
                print(triage_report["classification"].value_counts(dropna=False).head(20).to_string())

            if not sug_overrides.empty:
                print("[triage] suggested_overrides_sample:")
                print(sug_overrides.head(sample_n).to_string(index=False))

            if not sug_exclusions.empty:
                print("[triage] suggested_exclusions_sample:")
                print(sug_exclusions.head(sample_n).to_string(index=False))

    else:
        if verbose:
            print("[triage] no failures for this dt -> writing empty triage outputs.")

    store.write_universe_triage_outputs(
        as_of=as_of,
        triage_report=triage_report,
        suggested_overrides=sug_overrides,
        suggested_exclusions=sug_exclusions,
        mapping_changes=mapping_changes,
        mapping_validation=mapping_validation,
    )

    # 5) also write local copies for debugging (ALWAYS useful)
    out_dir = paths.ensure_dir(paths.universe_dir()).as_posix()
    try:
        write_triage_outputs_local(
            out_dir=out_dir,
            triage_report=triage_report,
            suggested_overrides=sug_overrides,
            suggested_exclusions=sug_exclusions,
        )
        print(f"[triage][local] wrote csvs -> {out_dir}")
    except Exception as e:
        print(f"[triage][local][warn] could not write local triage csvs: {e}")


    if verbose:
        print("[triage] outputs_written (triage_report/suggested_overrides/suggested_exclusions + mapping artifacts)")
