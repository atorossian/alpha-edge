from __future__ import annotations

import typer

from alpha_edge.jobs.run_daily_report import main as daily_report_main


def daily_report() -> None:
    """
    Run the daily report (after US market close).
    """
    daily_report_main()
