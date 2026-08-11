from __future__ import annotations

import os
import subprocess
import sys

import pytest


MODULES = [
    "alpha_edge.operations.record_trade",
    "alpha_edge.operations.record_cashflow",
    "alpha_edge.operations.record_dividend",
    "alpha_edge.operations.rebuild_ledger",
]


@pytest.mark.parametrize("module", MODULES)
def test_critical_cli_help_loads(module: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env["AWS_EC2_METADATA_DISABLED"] = "true"

    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
