# src/alpha_edge/cli/commands/pipeline_cmd.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

import typer

from alpha_edge import paths
from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import load_runtime_config


app = typer.Typer(no_args_is_help=True, help="Alpha Edge composed pipelines.")


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    printable = " ".join(str(x) for x in cmd)
    print(f"\n$ {printable}")

    if dry_run:
        return

    subprocess.run(cmd, check=True)


def _python_module(module: str, args: Iterable[str | int | float | Path | None]) -> list[str]:
    out = [sys.executable, "-m", module]
    for a in args:
        if a is None:
            continue
        out.append(str(a))
    return out


def _flag(name: str, enabled: bool) -> list[str]:
    return [name] if enabled else []


def _opt(name: str, value) -> list[str]:
    if value is None:
        return []
    return [name, str(value)]


def _extract_cli_option(name: str, default: str | None = None) -> str | None:
    """Best-effort option extraction for console-script entrypoints before Typer parses args."""
    argv = list(sys.argv)
    for i, item in enumerate(argv):
        if item == name and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return default


def _cli_has_flag(name: str) -> bool:
    return name in set(sys.argv)


def _run_pipeline_entrypoint_with_audit(*, pipeline_name: str, callback) -> None:
    env = _extract_cli_option("--env", "dev") or "dev"
    dry_run = _cli_has_flag("--dry-run") or _cli_has_flag("--no-write")
    cfg = load_runtime_config(env)
    input_args = {
        "pipeline_name": pipeline_name,
        "argv": list(sys.argv),
        "env": env,
        "dry_run": dry_run,
    }

    with capture_script_run(
        cfg=cfg,
        script_name="pipeline_cmd.py",
        input_args=input_args,
        dry_run=bool(dry_run),
    ) as run_id:
        try:
            callback()
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="pipeline",
                entity_type="pipeline_orchestration",
                entity_id=pipeline_name,
                as_of=_extract_cli_option("--as-of") or _extract_cli_option("--run-dt"),
                source_script="pipeline_cmd.py",
                source_mode=pipeline_name,
                status=("dry_run" if dry_run else "success"),
                input_args=input_args,
                metadata={
                    "tier": "additional_orchestration_writer",
                    "payload_policy": "large_dataset_metadata_only",
                    "note": "Pipeline-level audit only. Child scripts are expected to emit detailed script logs and audit events for their own datalake mutations.",
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=bool(dry_run))
        except BaseException as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="pipeline",
                entity_type="pipeline_orchestration",
                entity_id=pipeline_name,
                as_of=_extract_cli_option("--as-of") or _extract_cli_option("--run-dt"),
                source_script="pipeline_cmd.py",
                source_mode=pipeline_name,
                status="failed",
                input_args=input_args,
                metadata={
                    "tier": "additional_orchestration_writer",
                    "payload_policy": "large_dataset_metadata_only",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=bool(dry_run))
            raise


# ---------------------------------------------------------------------
# Entry functions used by pyproject scripts
# ---------------------------------------------------------------------
def morning_entry() -> None:
    """
    Console-script entrypoint for:
      alphaedge-morning --env dev/prod ...
    """
    _run_pipeline_entrypoint_with_audit(pipeline_name="morning", callback=lambda: typer.run(morning))


def close_entry() -> None:
    """
    Console-script entrypoint for:
      alphaedge-after-close --env dev/prod ...
    """
    _run_pipeline_entrypoint_with_audit(pipeline_name="close", callback=lambda: typer.run(close))


def search_entry() -> None:
    """
    Console-script entrypoint for:
      alphaedge-portfolio-search --env dev/prod ...
    """
    _run_pipeline_entrypoint_with_audit(pipeline_name="search", callback=lambda: typer.run(search))


# ---------------------------------------------------------------------
# Typer commands
# ---------------------------------------------------------------------
@app.command("morning")
def morning(
    env: str = typer.Option("dev", "--env", help="Runtime env: dev/staging/prod."),
    universe_csv: str = typer.Option(
        str(paths.universe_dir() / "universe.csv"),
        "--universe-csv",
        help="Universe CSV path.",
    ),
    excluded_csv: str = typer.Option(
        str(paths.universe_dir() / "asset_excluded.csv"),
        "--excluded-csv",
        help="Excluded-assets CSV path.",
    ),
    start: str = typer.Option("2010-01-01", "--start", help="Market-data start date."),
    end: str | None = typer.Option(None, "--end", help="Market-data end date. Optional."),
    interval: str = typer.Option("1d", "--interval", help="Bar interval."),
    max_assets: int | None = typer.Option(
        None,
        "--max-assets",
        help="Limit assets across morning pipeline for testing.",
    ),
    max_workers: int = typer.Option(
        4,
        "--max-workers",
        help="Ingestion worker count.",
    ),
    indicator_max_workers: int = typer.Option(
        8,
        "--indicator-max-workers",
        help="Market-indicator worker count.",
    ),
    indicator_min_obs: int = typer.Option(
        60,
        "--indicator-min-obs",
        help="Minimum observations required to retain an asset in the indicator layer.",
    ),
    indicator_annualization_days: int = typer.Option(
        252,
        "--indicator-annualization-days",
        help="Annualization factor used for realized volatility.",
    ),
    indicator_progress_every: int = typer.Option(
        100,
        "--indicator-progress-every",
        help="Print indicator progress after this many completed assets.",
    ),
    indicator_build_mode: str = typer.Option(
        "incremental",
        "--indicator-build-mode",
        help="Indicator build mode: full or incremental.",
    ),
    indicator_lookback_calendar_days: int = typer.Option(
        500,
        "--indicator-lookback-calendar-days",
        help=("Calendar-day warm-up window used when incrementally recalculating indicators."),
    ),
    indicator_replacement_calendar_days: int = typer.Option(
        60,
        "--indicator-replacement-calendar-days",
        help=("Recent calendar-day overlap replaced during an incremental indicator build."),
    ),
    skip_indicators: bool = typer.Option(
        False,
        "--skip-indicators",
        help="Skip historical/latest market indicator construction.",
    ),
    yahoo_max_concurrency: int = typer.Option(
        2,
        "--yahoo-max-concurrency",
        help="Yahoo max concurrency.",
    ),
    yahoo_rate_per_sec: float = typer.Option(
        1.5,
        "--yahoo-rate-per-sec",
        help="Yahoo request rate.",
    ),
    cache_min_years: float = typer.Option(
        5.0,
        "--cache-min-years",
        help="Returns cache min years.",
    ),
    cache_force: bool = typer.Option(
        False,
        "--cache-force",
        help=("Force execution of the returns cache build in the selected cache mode, even if it appears up to date."),
    ),
    cache_build_mode: str = typer.Option(
        "incremental",
        "--cache-build-mode",
        help="Returns-wide cache build mode: full or incremental.",
    ),
    cache_replacement_calendar_days: int = typer.Option(
        30,
        "--cache-replacement-calendar-days",
        help=("Recent calendar-day overlap replaced during an incremental returns-wide cache update."),
    ),
    cache_max_new_assets_full_build: int = typer.Option(
        100,
        "--cache-max-new-assets-full-build",
        help=("Maximum number of new assets allowed to be full-backfilled during an incremental cache update."),
    ),
    no_triage: bool = typer.Option(
        False,
        "--no-triage",
        help="Skip ingest post-triage.",
    ),
    ignore_existing_state: bool = typer.Option(
        False,
        "--ignore-existing-state",
        help="Ignore ingest state.",
    ),
    no_write: bool = typer.Option(
        False,
        "--no-write",
        help="Run child processes without persistent writes where supported.",
    ),
    confirm_prod_write: bool = typer.Option(
        False,
        "--confirm-prod-write",
        help="Allow prod writes.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print child commands only; do not execute them.",
    ),
    run_transition: bool = typer.Option(
        False,
        "--run-transition",
        help=(
            "Run Milestone 17 portfolio transition checks after market regime. "
            "Disabled by default to keep the market-data morning routine unchanged."
        ),
    ),
    transition_as_of: str | None = typer.Option(
        None,
        "--transition-as-of",
        help="Transition as-of date YYYY-MM-DD. Defaults to child script default.",
    ),
    transition_equity0: float | None = typer.Option(
        None,
        "--transition-equity0",
        help="Equity0 used by local/shadow transition evaluation.",
    ),
    transition_notional: float | None = typer.Option(
        None,
        "--transition-notional",
        help="Optional notional passed to transition execution plan.",
    ),
    transition_goals: str = typer.Option(
        "7500,10000,12500",
        "--transition-goals",
        help="Comma-separated transition goals.",
    ),
    transition_main_goal: float = typer.Option(
        10000.0,
        "--transition-main-goal",
        help="Main goal used by local/shadow transition evaluation.",
    ),
    transition_prefer_local: bool = typer.Option(
        False,
        "--transition-prefer-local",
        help="Prefer local optimizer target over accepted shadow target in execution plan.",
    ),
    transition_skip_local: bool = typer.Option(
        False,
        "--transition-skip-local",
        help="Skip run_local_transition_optimizer.py.",
    ),
    transition_skip_shadow: bool = typer.Option(
        False,
        "--transition-skip-shadow",
        help="Skip run_shadow_portfolio_assessment.py.",
    ),
) -> None:
    print("\n=== PIPELINE: morning ===")

    if no_write and not dry_run:
        print("[WARN] --no-write does not apply to ingest_market_data.py. Use --dry-run to prevent ingestion writes.")
    # ---------------------------------------------------------
    # 1. Market ingestion
    #
    # Writes:
    #   - OHLCV USD
    #   - canonical returns_usd
    #   - latest price/return snapshots
    # ---------------------------------------------------------

    indicator_build_mode = str(indicator_build_mode).strip().lower()

    if indicator_build_mode not in {"full", "incremental"}:
        raise typer.BadParameter("--indicator-build-mode must be full or incremental.")

    if indicator_lookback_calendar_days < 1:
        raise typer.BadParameter("--indicator-lookback-calendar-days must be >= 1.")

    if indicator_replacement_calendar_days < 0:
        raise typer.BadParameter("--indicator-replacement-calendar-days must be >= 0.")

    if indicator_replacement_calendar_days > indicator_lookback_calendar_days:
        raise typer.BadParameter(
            "--indicator-replacement-calendar-days cannot exceed --indicator-lookback-calendar-days."
        )

    cache_build_mode = str(cache_build_mode).strip().lower()

    if cache_build_mode not in {"full", "incremental"}:
        raise typer.BadParameter("--cache-build-mode must be full or incremental.")

    if cache_replacement_calendar_days < 0:
        raise typer.BadParameter("--cache-replacement-calendar-days must be >= 0.")

    if cache_max_new_assets_full_build < 0:
        raise typer.BadParameter("--cache-max-new-assets-full-build must be >= 0.")

    ingest_args: list[str | int | float | Path | None] = [
        "--env",
        env,
        "--universe-path",
        universe_csv,
        "--start",
        start,
        *_opt("--end", end),
        "--interval",
        interval,
        *_opt("--max-assets", max_assets),
        "--max-workers",
        max_workers,
        "--yahoo-max-concurrency",
        yahoo_max_concurrency,
        "--yahoo-rate-per-sec",
        yahoo_rate_per_sec,
        *_flag("--no-triage", no_triage),
        *_flag("--ignore-existing-state", ignore_existing_state),
        *_flag("--confirm-prod-write", confirm_prod_write),
    ]

    _run(
        _python_module(
            "alpha_edge.market.ingest_market_data",
            ingest_args,
        ),
        dry_run=dry_run,
    )

    # ---------------------------------------------------------
    # 2. Historical and latest market indicators
    #
    # Depends on:
    #   - market/ohlcv_usd/v1
    #   - market/returns_usd/v1
    #
    # Writes:
    #   - market/indicators/v1/asset_id=.../year=...
    #   - market/snapshots/v1/latest_indicators.parquet
    #
    # The indicator script uses --dry-run rather than --no-write.
    # ---------------------------------------------------------
    if not skip_indicators:
        indicator_args: list[str | int | float | Path | None] = [
            "--env",
            env,
            "--universe-csv",
            universe_csv,
            "--excluded-csv",
            excluded_csv,
            "--mode",
            indicator_build_mode,
            "--start",
            start,
            *_opt("--end", end),
            "--lookback-calendar-days",
            indicator_lookback_calendar_days,
            "--replacement-calendar-days",
            indicator_replacement_calendar_days,
            *_opt("--max-assets", max_assets),
            "--min-obs",
            indicator_min_obs,
            "--annualization-days",
            indicator_annualization_days,
            "--max-workers",
            indicator_max_workers,
            "--progress-every",
            indicator_progress_every,
            *_flag("--dry-run", no_write),
            *_flag(
                "--confirm-prod-write",
                confirm_prod_write,
            ),
        ]

        _run(
            _python_module(
                "alpha_edge.market.build_market_indicators",
                indicator_args,
            ),
            dry_run=dry_run,
        )
    else:
        print("\n[SKIP] build_market_indicators")

    # ---------------------------------------------------------
    # 3. Wide returns cache
    #
    # Depends on canonical returns_usd.
    # Used by portfolio analytics and market regime.
    # ---------------------------------------------------------
    returns_args: list[str | int | float | Path | None] = [
        "--env",
        env,
        "--mode",
        cache_build_mode,
        "--start",
        start,
        *_opt("--end", end),
        "--min-years",
        cache_min_years,
        "--replacement-calendar-days",
        cache_replacement_calendar_days,
        "--max-new-assets-full-build",
        cache_max_new_assets_full_build,
        "--universe-csv",
        universe_csv,
        "--excluded-csv",
        excluded_csv,
        *_flag("--force", cache_force),
        *_flag("--no-write", no_write),
        *_flag("--confirm-prod-write", confirm_prod_write),
    ]

    _run(
        _python_module(
            "alpha_edge.market.build_returns_wide_cache",
            returns_args,
        ),
        dry_run=dry_run,
    )

    # ---------------------------------------------------------
    # 4. Market regime
    # ---------------------------------------------------------
    regime_args: list[str | int | float | Path | None] = [
        "--env",
        env,
        *_flag("--no-write", no_write),
        *_flag("--confirm-prod-write", confirm_prod_write),
    ]

    _run(
        _python_module(
            "alpha_edge.market.compute_market_regime",
            regime_args,
        ),
        dry_run=dry_run,
    )
    # ---------------------------------------------------------
    # 5. Portfolio transition routine, optional
    #
    # Depends on:
    #   - daily health/latest.json
    #   - market regime latest state
    #   - returns_wide cache
    #   - latest prices
    #   - dated ledger positions for execution planning
    #
    # Writes:
    #   - portfolio_transition/assessment
    #   - portfolio_transition/local_optimizer, if allowed
    #   - portfolio_transition/shadow, if allowed
    #   - portfolio_transition/execution_plan
    #
    # This is disabled by default because the core Milestone 10 morning
    # routine is market-data only.
    # ---------------------------------------------------------
    if run_transition:
        print("\n=== PIPELINE: morning / transition ===")

        transition_assessment_args: list[str | int | float | Path | None] = [
            "--env",
            env,
            *_opt("--as-of", transition_as_of),
            *_flag("--dry-run", no_write),
            *_flag("--confirm-prod-write", confirm_prod_write),
        ]

        _run(
            _python_module(
                "alpha_edge.jobs.run_transition_assessment",
                transition_assessment_args,
            ),
            dry_run=dry_run,
        )

        if transition_equity0 is None and not (transition_skip_local and transition_skip_shadow):
            print(
                "[equity] --transition-equity0 not supplied; child jobs will resolve current equity from ledger + latest prices."
            )

        if not transition_skip_local:
            local_transition_args: list[str | int | float | Path | None] = [
                "--env",
                env,
                *_opt("--as-of", transition_as_of),
                *_opt("--equity0", transition_equity0),
                *_opt("--notional", transition_notional),
                "--goals",
                transition_goals,
                "--main-goal",
                transition_main_goal,
                "--cache-min-years",
                int(cache_min_years),
                *_flag("--dry-run", no_write),
                *_flag("--confirm-prod-write", confirm_prod_write),
            ]

            _run(
                _python_module(
                    "alpha_edge.jobs.run_local_transition_optimizer",
                    local_transition_args,
                ),
                dry_run=dry_run,
            )
        else:
            print("\n[SKIP] run_local_transition_optimizer")

        if not transition_skip_shadow:
            shadow_transition_args: list[str | int | float | Path | None] = [
                "--env",
                env,
                *_opt("--as-of", transition_as_of),
                *_opt("--equity0", transition_equity0),
                *_opt("--notional", transition_notional),
                "--goals",
                transition_goals,
                "--main-goal",
                transition_main_goal,
                "--cache-min-years",
                int(cache_min_years),
                *_flag("--dry-run", no_write),
                *_flag("--confirm-prod-write", confirm_prod_write),
            ]

            _run(
                _python_module(
                    "alpha_edge.jobs.run_shadow_portfolio_assessment",
                    shadow_transition_args,
                ),
                dry_run=dry_run,
            )
        else:
            print("\n[SKIP] run_shadow_portfolio_assessment")

        transition_execution_args: list[str | int | float | Path | None] = [
            "--env",
            env,
            *_opt("--as-of", transition_as_of),
            *_opt("--notional", transition_notional),
            *_flag("--prefer-local", transition_prefer_local),
            *_flag("--dry-run", no_write),
            *_flag("--confirm-prod-write", confirm_prod_write),
        ]

        _run(
            _python_module(
                "alpha_edge.jobs.run_transition_execution_plan",
                transition_execution_args,
            ),
            dry_run=dry_run,
        )
    else:
        print("\n[SKIP] portfolio transition routine")

    print("\n[DONE] pipeline morning\n")


@app.command("close")
def close(
    env: str = typer.Option("dev", "--env", help="Runtime env: dev/staging/prod."),
    as_of: str | None = typer.Option(None, "--as-of", help="As-of date YYYY-MM-DD."),
    no_write: bool = typer.Option(False, "--no-write", help="Run without writing where supported."),
    confirm_prod_write: bool = typer.Option(False, "--confirm-prod-write", help="Allow prod writes."),
    quarantine_lookback_days: int = typer.Option(10, "--quarantine-lookback-days"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands only."),
) -> None:
    print("\n=== PIPELINE: close ===")

    daily_report_args: list[str | int | float | Path | None] = [
        "--env",
        env,
        *_opt("--as-of", as_of),
        *_flag("--no-write", no_write),
        *_flag("--confirm-prod-write", confirm_prod_write),
    ]

    _run(
        _python_module("alpha_edge.jobs.run_daily_report", daily_report_args),
        dry_run=dry_run,
    )

    quarantine_args: list[str | int | float | Path | None] = [
        "--env",
        env,
        *_opt("--as-of", as_of),
        "--lookback-days",
        quarantine_lookback_days,
        *_flag("--no-write", no_write),
        *_flag("--confirm-prod-write", confirm_prod_write),
    ]

    _run(
        _python_module("alpha_edge.jobs.run_quarantine_analysis", quarantine_args),
        dry_run=dry_run,
    )

    print("\n[DONE] pipeline close\n")


@app.command("search")
def search(
    env: str = typer.Option("dev", "--env", help="Runtime env: dev/staging/prod."),
    as_of: str | None = typer.Option(None, "--as-of", help="Market as-of date YYYY-MM-DD."),
    run_dt: str | None = typer.Option(None, "--run-dt", help="Run partition date YYYY-MM-DD."),
    universe_csv: str | None = typer.Option(None, "--universe-csv", help="Universe CSV path."),
    equity0: float | None = typer.Option(
        None,
        "--equity0",
        "--equity-override",
        help="Equity override. If omitted, child job resolves current equity from ledger + latest prices.",
    ),
    goals: str = typer.Option("10000,12500,15000", "--goals"),
    main_goal: float = typer.Option(10000.0, "--main-goal"),
    override_target_leverage: float | None = typer.Option(None, "--override-target-leverage"),
    no_market_hmm: bool = typer.Option(False, "--no-market-hmm"),
    no_write: bool = typer.Option(False, "--no-write"),
    confirm_prod_write: bool = typer.Option(False, "--confirm-prod-write"),
    pop_size: int = typer.Option(80, "--pop-size"),
    generations: int = typer.Option(50, "--generations"),
    n_paths_init: int = typer.Option(5000, "--n-paths-init"),
    n_paths_final: int = typer.Option(20000, "--n-paths-final"),
    skip_stability_rerank: bool = typer.Option(False, "--skip-stability-rerank"),
    anneal_steps: int = typer.Option(200, "--anneal-steps"),
    anneal_n_paths_init: int = typer.Option(3000, "--anneal-n-paths-init"),
    anneal_n_paths_final: int = typer.Option(20000, "--anneal-n-paths-final"),
    min_universe_size: int = typer.Option(10, "--min-universe-size"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands only."),
) -> None:
    print("\n=== PIPELINE: search ===")

    args: list[str | int | float | Path | None] = [
        "--env",
        env,
        *_opt("--as-of", as_of),
        *_opt("--run-dt", run_dt),
        *_opt("--universe-csv", universe_csv),
        *_opt("--equity0", equity0),
        "--goals",
        goals,
        "--main-goal",
        main_goal,
        *_opt("--override-target-leverage", override_target_leverage),
        *_flag("--no-market-hmm", no_market_hmm),
        *_flag("--no-write", no_write),
        *_flag("--confirm-prod-write", confirm_prod_write),
        "--pop-size",
        pop_size,
        "--generations",
        generations,
        "--n-paths-init",
        n_paths_init,
        "--n-paths-final",
        n_paths_final,
        *_flag("--skip-stability-rerank", skip_stability_rerank),
        "--anneal-steps",
        anneal_steps,
        "--anneal-n-paths-init",
        anneal_n_paths_init,
        "--anneal-n-paths-final",
        anneal_n_paths_final,
        "--min-universe-size",
        min_universe_size,
    ]

    _run(
        _python_module("alpha_edge.jobs.run_portfolio_search", args),
        dry_run=dry_run,
    )

    print("\n[DONE] pipeline search\n")


@app.command("dev-smoke")
def dev_smoke(
    as_of: str = typer.Option("2024-01-31", "--as-of"),
    universe_csv: str = typer.Option("data/outputs/dev/dev_universe_sample.csv", "--universe-csv"),
    equity0: float = typer.Option(1000.0, "--equity0"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print commands only."),
) -> None:
    """
    Minimal dev smoke pipeline.

    Assumes sample trades/cashflows/dividends and dev universe sample already exist.
    This command intentionally uses --env dev and small workloads.
    """
    print("\n=== PIPELINE: dev-smoke ===")

    _run(
        _python_module(
            "alpha_edge.market.ingest_market_data",
            [
                "--env",
                "dev",
                "--universe-path",
                universe_csv,
                "--start",
                "2015-01-01",
                "--end",
                "2024-02-05",
                "--ignore-existing-state",
                "--no-triage",
                "--max-workers",
                2,
                "--yahoo-max-concurrency",
                1,
                "--yahoo-rate-per-sec",
                0.8,
            ],
        ),
        dry_run=dry_run,
    )

    _run(
        _python_module(
            "alpha_edge.market.build_returns_wide_cache",
            [
                "--env",
                "dev",
                "--min-years",
                5,
                "--force",
            ],
        ),
        dry_run=dry_run,
    )

    _run(
        _python_module(
            "alpha_edge.market.compute_market_regime",
            [
                "--env",
                "dev",
            ],
        ),
        dry_run=dry_run,
    )

    _run(
        _python_module(
            "alpha_edge.jobs.rebuild_ledger",
            [
                "--env",
                "dev",
                "--as-of",
                as_of,
                "--end",
                as_of,
                "--account-id",
                "main",
                "--prices-mode",
                "asof",
            ],
        ),
        dry_run=dry_run,
    )

    _run(
        _python_module(
            "alpha_edge.warehouse.build_warehouse",
            [
                "--env",
                "dev",
                "--dt",
                as_of,
                "--account-id",
                "main",
            ],
        ),
        dry_run=dry_run,
    )

    _run(
        _python_module(
            "alpha_edge.jobs.run_daily_report",
            [
                "--env",
                "dev",
                "--as-of",
                as_of,
            ],
        ),
        dry_run=dry_run,
    )

    _run(
        _python_module(
            "alpha_edge.jobs.run_portfolio_search",
            [
                "--env",
                "dev",
                "--as-of",
                as_of,
                "--run-dt",
                as_of,
                "--equity0",
                equity0,
                "--goals",
                "1200,1500,2000",
                "--main-goal",
                "1500",
                "--override-target-leverage",
                "1.0",
                "--universe-csv",
                universe_csv,
                "--pop-size",
                "20",
                "--generations",
                "5",
                "--n-paths-init",
                "500",
                "--n-paths-final",
                "1000",
                "--skip-stability-rerank",
                "--anneal-steps",
                "20",
                "--anneal-n-paths-init",
                "500",
                "--anneal-n-paths-final",
                "1000",
                "--min-universe-size",
                "5",
            ],
        ),
        dry_run=dry_run,
    )

    _run(
        _python_module(
            "alpha_edge.jobs.run_quarantine_analysis",
            [
                "--env",
                "dev",
                "--as-of",
                as_of,
                "--lookback-days",
                "10",
            ],
        ),
        dry_run=dry_run,
    )

    print("\n[DONE] pipeline dev-smoke\n")


def main() -> None:
    app()
