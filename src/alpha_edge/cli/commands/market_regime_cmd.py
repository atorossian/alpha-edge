from __future__ import annotations

import typer

from alpha_edge.market.compute_market_regime import main as compute_market_regime_main


def market_regime(
    verbose: bool = typer.Option(True, help="Print regime summary."),
) -> None:
    """
    Compute market regime (HMM) from proxy basket and write engine/v1/regimes/market_hmm/...
    """
    # If your compute_market_regime.py supports args, wire them here.
    # For now: just call main() to keep minimal change.
    _ = verbose
    compute_market_regime_main()
