# alpha_edge/cli/commands/pipeline_cmd.py
from __future__ import annotations

import typer

from alpha_edge.market.ingest_market_data import ingest as ingest_fn
from alpha_edge.market.build_returns_wide_cache import build_returns_wide_cache, CacheConfig
from alpha_edge.market.compute_market_regime import compute_market_regime as compute_market_regime_main
from alpha_edge.jobs.run_daily_report import main as run_daily_report_main
from alpha_edge.jobs.run_portfolio_search import main as run_portfolio_search_main
from alpha_edge.jobs.run_quarantine_analysis import main as run_quarantine_analysis_main
from alpha_edge import paths


app = typer.Typer(no_args_is_help=True, help="Daily pipelines (composed jobs).")


# -------------------------
# Plain entrypoints (NO typer.Option defaults)
# -------------------------

def morning_entry(
    *,
    bucket: str = "alpha-edge-algo",
    universe_csv: str = str(paths.universe_dir() / "universe.csv"),
    start_base: str = "2010-01-01",
    interval: str = "1d",
    max_tickers: int | None = None,
    force_refresh_csv: str | None = str(paths.universe_dir() / "ingest_force_refresh.csv"),
    max_workers: int = 4,
    yahoo_max_concurrency: int = 2,
    cache_min_years: float = 5.0,
    cache_force: bool = False,
    cache_dtype: str = "float32",
) -> None:
    print("\n=== PIPELINE: morning ===")

    print("\n[1/3] ingest_market_data")
    ingest_fn(
        bucket=bucket,
        universe_csv=universe_csv,
        start_base=start_base,
        interval=interval,
        max_tickers=max_tickers,
        force_refresh_csv=force_refresh_csv,
        max_workers=max_workers,
        yahoo_max_concurrency=yahoo_max_concurrency,
    )

    print("\n[2/3] build_returns_wide_cache")
    cfg = CacheConfig(
        bucket=bucket,
        min_years=cache_min_years,
        dtype=cache_dtype,
        force=cache_force,
    )
    build_returns_wide_cache(cfg)

    print("\n[3/3] compute_market_regime")
    compute_market_regime_main()

    print("\n[DONE] pipeline morning\n")

def close_entry() -> None:
    print("\n=== PIPELINE: close ===")
    print("\n[1/2] run_daily_report")

    run_daily_report_main()

    print("\n[2/2] run_quarantine_analysis")
    run_quarantine_analysis_main()

    print("\n[DONE] pipeline close\n")



def search_entry() -> None:
    print("\n=== PIPELINE: search ===")
    run_portfolio_search_main()
    print("\n[DONE] pipeline search\n")


# -------------------------
# Typer wrappers (parse CLI args properly)
# -------------------------

@app.command("morning")
def morning(
    bucket: str = typer.Option("alpha-edge-algo", help="S3 bucket."),
    universe_csv: str = typer.Option(str(paths.universe_dir() / "universe.csv"), help="Universe CSV path."),
    start_base: str = typer.Option("2010-01-01", help="Start date for ingestion."),
    interval: str = typer.Option("1d", help="Bar interval."),
    max_tickers: int | None = typer.Option(None, help="Limit tickers for testing."),
    force_refresh_csv: str | None = typer.Option(str(paths.universe_dir() / "ingest_force_refresh.csv"), help="Force refresh list."),
    max_workers: int = typer.Option(4, help="ThreadPool max workers."),
    yahoo_max_concurrency: int = typer.Option(2, help="Throttle yfinance concurrency."),
    cache_min_years: float = typer.Option(5.0, help="Returns cache min years."),
    cache_force: bool = typer.Option(False, help="Force rebuild returns cache."),
    cache_dtype: str = typer.Option("float32", help="Returns cache dtype."),
) -> None:
    return morning_entry(
        bucket=bucket,
        universe_csv=universe_csv,
        start_base=start_base,
        interval=interval,
        max_tickers=max_tickers,
        force_refresh_csv=force_refresh_csv,
        max_workers=max_workers,
        yahoo_max_concurrency=yahoo_max_concurrency,
        cache_min_years=cache_min_years,
        cache_force=cache_force,
        cache_dtype=cache_dtype,
    )


@app.command("search")
def search() -> None:
    return search_entry()


@app.command("close")
def close() -> None:
    return close_entry()


def main() -> None:
    # optional: if you ever want a single CLI: `poetry run alphaedge pipeline morning`
    app()
