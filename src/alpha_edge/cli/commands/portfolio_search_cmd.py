from __future__ import annotations

import typer

from alpha_edge.jobs.run_portfolio_search import main as portfolio_search_main


def portfolio_search() -> None:
    """
    Run portfolio search (GA + stability rerank + annealing), using latest market regime.
    """
    portfolio_search_main()
