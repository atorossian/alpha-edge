"""AEA-87 canonical full-universe integrity validation and audit entry point."""

from __future__ import annotations
import sys

from alpha_edge.core.audit import make_run_id, build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run
from alpha_edge.core.runtime import load_runtime_config
from alpha_edge.market.aea87_universe_screen import main as validate_universe


def main():
    run_id = make_run_id("aea87_validate")
    cfg = load_runtime_config("prod")
    with capture_script_run(cfg=cfg, script_name="aea87_validate", input_args={"argv": sys.argv[1:]}, run_id=run_id):
        try:
            report = validate_universe(run_id=run_id)
        except Exception as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="build_dataset",
                entity_type="market_data_validation",
                entity_id=None,
                as_of=None,
                source_script="aea87_validate",
                source_mode="read_only_screen",
                status="failed",
                input_args={"argv": sys.argv[1:]},
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event)
            raise
        event = build_audit_event(
            cfg=cfg,
            run_id=run_id,
            event_type="build_dataset",
            entity_type="market_data_validation",
            entity_id=None,
            as_of=None,
            source_script="aea87_validate",
            source_mode="read_only_screen",
            status="success",
            input_args={"argv": sys.argv[1:]},
            output_keys=[f"market/repairs/aea87/universe_screen/v1/runs/{run_id}/report.json"],
            metadata={
                "assets_scanned": report["scanned"],
                "classification_counts": report["classification_counts"],
                "production_market_data_modified": False,
            },
        )
        write_audit_event(cfg=cfg, event=event)


if __name__ == "__main__":
    main()
