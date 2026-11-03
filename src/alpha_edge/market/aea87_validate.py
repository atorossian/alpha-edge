"""Canonical AEA-87 whole-universe historical integrity validation command.

Read-only against active market data; results saved directly to S3.
"""
from alpha_edge.market.aea87_universe_screen import main


if __name__ == "__main__":
    main()
