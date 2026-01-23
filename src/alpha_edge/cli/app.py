from __future__ import annotations

import typer

from src.alpha_edge.cli.commands.ingest_cmd import (
    ingest_cmd,
    returns_cache_cmd,
    market_regime_cmd,
    daily_report_cmd,
    portfolio_search_cmd,
    universe_cmd,
)

app = typer.Typer(
    name="alpha-edge",
    help="Alpha Edge CLI: ingestion, caches, regimes, portfolio search, daily reports.",
    no_args_is_help=True,
)

# top-level commands
app.command("ingest")(ingest_cmd.ingest)
app.command("returns-cache")(returns_cache_cmd.cache)
app.command("market-regime")(market_regime_cmd.market_regime)
app.command("daily-report")(daily_report_cmd.daily_report)
app.command("portfolio-search")(portfolio_search_cmd.portfolio_search)

# grouped commands
app.add_typer(universe_cmd.app, name="universe")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
