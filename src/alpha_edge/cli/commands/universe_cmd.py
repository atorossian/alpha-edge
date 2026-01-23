from __future__ import annotations

import typer

from alpha_edge.jobs.run_universe_update import main as universe_update_main
from alpha_edge.jobs.run_universe_triage import run_post_ingest_triage  # usually called by ingest
from alpha_edge.core.market_store import MarketStore

app = typer.Typer(no_args_is_help=True, help="Universe maintenance commands.")


@app.command("update")
def update(
    mode: str = typer.Option("patch", help="full|patch"),
) -> None:
    """
    Update universe.csv (full scrape+enrich or patch from overrides/exclusions).
    """
    # If your run_universe_update.py main() reads args from CLI already, keep as-is.
    # Otherwise, you’ll refactor it into main(mode=...) and wire it here.
    _ = mode
    universe_update_main()


@app.command("triage")
def triage(
    as_of: str = typer.Option(..., help="dt partition date YYYY-MM-DD for ingest failures."),
    universe_csv: str = typer.Option("data/universe/universe.csv", help="Universe CSV."),
    overrides_csv: str = typer.Option("data/universe/universe_overrides.csv", help="Overrides CSV."),
    excluded_csv: str = typer.Option("data/universe/asset_excluded.csv", help="Excluded CSV."),
    bucket: str = typer.Option("alpha-edge-algo", help="S3 bucket."),
) -> None:
    """
    Manually run triage on ingest failures for a given dt (normally ingest runs this).
    """
    store = MarketStore(bucket=bucket)
    run_post_ingest_triage(
        store=store,
        as_of=as_of,
        universe_csv=universe_csv,
        overrides_csv=overrides_csv,
        excluded_csv=excluded_csv,
        mapping_changes=None,
        mapping_validation=None,
    )
