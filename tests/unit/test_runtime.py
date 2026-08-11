from __future__ import annotations

import pytest

from alpha_edge.core.runtime import (
    load_runtime_config,
    require_prod_confirmation,
    runtime_dt_key,
    runtime_engine_key,
)


def test_dev_runtime_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPHA_EDGE_ENV", raising=False)
    cfg = load_runtime_config("dev")

    assert cfg.is_prod is False
    assert cfg.engine_root == "dev/engine/v1"
    assert cfg.market_root == "dev/market"
    assert runtime_engine_key(cfg, "trades", "index.json") == "dev/engine/v1/trades/index.json"
    assert runtime_dt_key(cfg, "trades", "2026-06-14", "trade_1.json") == (
        "dev/engine/v1/trades/dt=2026-06-14/trade_1.json"
    )


def test_prod_write_requires_explicit_confirmation() -> None:
    cfg = load_runtime_config("prod")

    with pytest.raises(SystemExit, match="Refusing to write to prod"):
        require_prod_confirmation(cfg, confirm=False)

    require_prod_confirmation(cfg, confirm=True)
