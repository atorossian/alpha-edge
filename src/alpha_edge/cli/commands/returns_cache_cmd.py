from __future__ import annotations

import typer

from alpha_edge.market.build_returns_wide_cache import build_returns_wide_cache, CacheConfig


def cache(
    bucket: str = typer.Option("alpha-edge-algo", help="S3 bucket."),
    min_years: float = typer.Option(5.0, help="Minimum years of data for asset inclusion."),
    start: str = typer.Option("2010-01-01", help="Cache start date."),
    end: str | None = typer.Option(None, help="Cache end date (default: today UTC)."),
    min_obs: int = typer.Option(252 * 5, help="Minimum observations."),
    dtype: str = typer.Option("float32", help="Parquet dtype for cache."),
    force: bool = typer.Option(False, help="Force rebuild even if up-to-date."),
    progress_every: int = typer.Option(100, help="Progress print frequency (assets)."),
) -> None:
    """
    Build returns_wide cache used by portfolio search.
    """
    cfg = CacheConfig(
        bucket=bucket,
        min_years=min_years,
        start=start,
        end=end,
        min_obs=min_obs,
        dtype=dtype,
        force=force,
        progress_every=progress_every,
    )
    build_returns_wide_cache(cfg)
