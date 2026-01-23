from __future__ import annotations

import typer

from alpha_edge.market.ingest_market_data import ingest as ingest_fn


def ingest(
    bucket: str = typer.Option("alpha-edge-algo", help="S3 bucket."),
    universe_csv: str = typer.Option("data/universe/universe.csv", help="Universe CSV path."),
    start_base: str = typer.Option("2010-01-01", help="Start date for full history."),
    interval: str = typer.Option("1d", help="Bar interval (e.g., 1d)."),
    max_tickers: int | None = typer.Option(None, help="Limit tickers for testing."),
    force_refresh_csv: str | None = typer.Option("data/universe/ingest_force_refresh.csv", help="Force refresh list."),
    max_workers: int = typer.Option(4, help="ThreadPool max workers."),
    yahoo_max_concurrency: int = typer.Option(2, help="Throttle yfinance concurrency."),
) -> None:
    """
    Ingest market data: download OHLCV, compute returns, write snapshots & state, run post-ingest triage.
    """
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
