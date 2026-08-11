from __future__ import annotations

import os
from typing import Literal

from alpha_edge.core.schemas import RuntimeConfig


EnvName = Literal["dev", "staging", "prod"]

DEFAULT_BUCKET = "alpha-edge-algo"
DEFAULT_REGION = "eu-west-1"


def _clean_env(env: str | None) -> EnvName:
    env_norm = (env or os.getenv("ALPHA_EDGE_ENV") or "dev").strip().lower()

    if env_norm not in {"dev", "staging", "prod"}:
        raise ValueError(f"Invalid env={env_norm!r}. Expected dev, staging, or prod.")

    return env_norm  # type: ignore[return-value]


def load_runtime_config(env: str | None = None) -> RuntimeConfig:
    """
    Central runtime/environment resolver.

    Prod keeps legacy paths.
    Dev/staging are isolated under separate prefixes in the same bucket by default.

    Overrides:
      ALPHA_EDGE_ENV
      ALPHA_EDGE_BUCKET
      ALPHA_EDGE_REGION
    """
    env_norm = _clean_env(env)

    bucket = os.getenv("ALPHA_EDGE_BUCKET", DEFAULT_BUCKET).strip()
    region = os.getenv("ALPHA_EDGE_REGION", DEFAULT_REGION).strip()

    if env_norm == "prod":
        return RuntimeConfig(
            env="prod",
            bucket=bucket,
            region=region,
            engine_root="engine/v1",
            market_root="market",
            warehouse_root="engine/v1/warehouse",
            is_prod=True,
        )

    return RuntimeConfig(
        env=env_norm,
        bucket=bucket,
        region=region,
        engine_root=f"{env_norm}/engine/v1",
        market_root=f"{env_norm}/market",
        warehouse_root=f"{env_norm}/engine/v1/warehouse",
        is_prod=False,
    )


def require_prod_confirmation(cfg: RuntimeConfig, confirm: bool) -> None:
    if cfg.is_prod and not confirm:
        raise SystemExit("Refusing to write to prod without --confirm-prod-write")


def runtime_engine_key(cfg: RuntimeConfig, *parts: str) -> str:
    return "/".join([cfg.engine_root.strip("/")] + [p.strip("/") for p in parts if str(p).strip("/")])


def runtime_market_key(cfg: RuntimeConfig, *parts: str) -> str:
    return "/".join([cfg.market_root.strip("/")] + [p.strip("/") for p in parts if str(p).strip("/")])


def runtime_warehouse_key(cfg: RuntimeConfig, table: str, *parts: str) -> str:
    return "/".join(
        [
            cfg.warehouse_root.strip("/"),
            table.strip("/"),
            "v=1",
            *[p.strip("/") for p in parts if str(p).strip("/")],
        ]
    )


def runtime_dt_key(cfg: RuntimeConfig, table: str, dt_str: str, filename: str) -> str:
    return runtime_engine_key(cfg, table, f"dt={dt_str}", filename)


def print_runtime_config(cfg: RuntimeConfig) -> None:
    print("\n=== RUNTIME CONFIG ===")
    print(f"env:            {cfg.env}")
    print(f"bucket:         {cfg.bucket}")
    print(f"region:         {cfg.region}")
    print(f"engine_root:    {cfg.engine_root}")
    print(f"market_root:    {cfg.market_root}")
    print(f"warehouse_root: {cfg.warehouse_root}")
    print(f"is_prod:        {cfg.is_prod}")
    print("")