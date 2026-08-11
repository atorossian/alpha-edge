# run_update_score_caps.py  (S3-only I/O)
from __future__ import annotations

from alpha_edge.core.audit import build_audit_event, write_audit_event
from alpha_edge.core.run_logging import capture_script_run

import argparse
import datetime as dt
from dataclasses import asdict, replace
from typing import Any, Dict, Tuple

import pandas as pd

from alpha_edge.core.data_loader import s3_init, s3_load_latest_json, s3_write_json_event
from alpha_edge.core.runtime import RuntimeConfig, load_runtime_config, require_prod_confirmation
from alpha_edge.core.schemas import ScoreConfig


DEFAULT_ENGINE_BUCKET = "alpha-edge-algo"
DEFAULT_ENGINE_REGION = "eu-west-1"
DEFAULT_ENGINE_ROOT_PREFIX = "engine/v1"


CAP_FIELDS = {
    "ruin_cap",
    "cvar_cap",
    "mdd_cap",
    "hhi_cap",
    "corr_cap",
    "time_cap_days",
    "hf_ratio_cap",
    "spec_entropy_cap",
    "freq_overlap_cap",
    "fft_bands_days",
}


def cfg_bucket(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "bucket", DEFAULT_ENGINE_BUCKET))


def cfg_region(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "region", DEFAULT_ENGINE_REGION))


def cfg_engine_root(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "engine_root", DEFAULT_ENGINE_ROOT_PREFIX)).strip("/")


def cfg_env(cfg: RuntimeConfig) -> str:
    return str(getattr(cfg, "env", "dev"))


def _parse_scalar(v: str) -> Any:
    v = v.strip()
    if v.lower() in {"none", "null"}:
        return None
    if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v


def _parse_fft_bands(s: str) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    raw = s.strip()
    if not raw:
        raise ValueError("fft_bands_days override is empty")

    parts = raw.split(";")
    if len(parts) != 3:
        raise ValueError("fft_bands_days must have exactly 3 bands separated by ';'")

    bands = []
    for p in parts:
        p = p.strip()
        if "," in p:
            a, b = p.split(",", 1)
        elif "-" in p:
            a, b = p.split("-", 1)
        else:
            raise ValueError(f"Bad band format {p!r}. Use 'a,b' or 'a-b'.")

        lo = float(a.strip())
        hi = float(b.strip())
        bands.append((lo, hi))

    return bands[0], bands[1], bands[2]


def _in_0_1_or_none(name: str, x: Any) -> None:
    if x is None:
        return
    fx = float(x)
    if not (0.0 <= fx <= 1.0):
        raise ValueError(f"{name} must be in [0,1] or None. Got {x}")


def _validate_caps(cfg: ScoreConfig) -> None:
    _in_0_1_or_none("ruin_cap", cfg.ruin_cap)
    _in_0_1_or_none("cvar_cap", cfg.cvar_cap)
    _in_0_1_or_none("mdd_cap", cfg.mdd_cap)
    _in_0_1_or_none("hhi_cap", cfg.hhi_cap)
    _in_0_1_or_none("corr_cap", cfg.corr_cap)
    _in_0_1_or_none("hf_ratio_cap", cfg.hf_ratio_cap)
    _in_0_1_or_none("freq_overlap_cap", cfg.freq_overlap_cap)

    if cfg.spec_entropy_cap is not None:
        se = float(cfg.spec_entropy_cap)
        if not (se > 0.0):
            raise ValueError(f"spec_entropy_cap must be > 0. Got {cfg.spec_entropy_cap}")

    if cfg.time_cap_days is not None and int(cfg.time_cap_days) <= 0:
        raise ValueError(f"time_cap_days must be > 0 or None. Got {cfg.time_cap_days}")

    bands = cfg.fft_bands_days
    if not isinstance(bands, tuple) or len(bands) != 3:
        raise ValueError("fft_bands_days must be a tuple of 3 (lo, hi) bands")

    prev_hi = None
    for i, (lo, hi) in enumerate(bands):
        lo = float(lo)
        hi = float(hi)

        if not (lo > 0 and hi > 0 and hi > lo):
            raise ValueError(f"fft_bands_days band {i} must satisfy 0 < lo < hi. Got {(lo, hi)}")

        if prev_hi is not None and lo < prev_hi:
            raise ValueError(
                "fft_bands_days bands must be non-overlapping and ordered. "
                f"Band {i} lo={lo} < previous hi={prev_hi}."
            )

        prev_hi = hi


def apply_cap_overrides(cfg: ScoreConfig, overrides: Dict[str, Any]) -> ScoreConfig:
    unknown = set(overrides.keys()) - CAP_FIELDS
    if unknown:
        raise ValueError(f"Unknown/forbidden fields in overrides: {sorted(unknown)}")

    updated = replace(cfg, **overrides)
    _validate_caps(updated)
    return updated


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Update ScoreConfig caps, including FFT caps/bands, and write to S3 latest.json."
    )

    p.add_argument("--bucket", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--engine-root", default=None)

    p.add_argument("--ruin_cap", default=None)
    p.add_argument("--cvar_cap", default=None)
    p.add_argument("--mdd_cap", default=None)
    p.add_argument("--hhi_cap", default=None)
    p.add_argument("--corr_cap", default=None)
    p.add_argument("--time_cap_days", default=None)

    p.add_argument("--hf_ratio_cap", default=None)
    p.add_argument("--spec_entropy_cap", default=None)
    p.add_argument("--freq_overlap_cap", default=None)

    p.add_argument(
        "--fft_bands_days",
        default=None,
        help='Format: "2,20;20,60;60,250" or "2-20;20-60;60-250".',
    )

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--env", default=None, choices=["dev", "staging", "prod"])
    p.add_argument("--confirm-prod-write", action="store_true")

    return p.parse_args()


def _main_impl() -> None:
    args = parse_args()

    cfg = load_runtime_config(args.env)

    bucket = str(args.bucket or cfg_bucket(cfg))
    region = str(args.region or cfg_region(cfg))
    engine_root = str(args.engine_root or cfg_engine_root(cfg)).strip("/")

    if not bool(args.dry_run):
        require_prod_confirmation(cfg, bool(args.confirm_prod_write))

    overrides: Dict[str, Any] = {}
    for k in CAP_FIELDS:
        v = getattr(args, k, None)
        if v is None:
            continue

        if k == "fft_bands_days":
            overrides[k] = _parse_fft_bands(v)
        else:
            overrides[k] = _parse_scalar(v)

    if not overrides:
        raise SystemExit("No overrides provided. Pass at least one --<field>=...")

    today = pd.Timestamp(dt.date.today())
    s3 = s3_init(region)

    raw = s3_load_latest_json(
        s3,
        bucket=bucket,
        root_prefix=engine_root,
        table="configs/score_config",
    )
    if not raw:
        raise RuntimeError(f"Missing s3://{bucket}/{engine_root}/configs/score_config/latest.json")

    current = ScoreConfig(**raw)
    updated = apply_cap_overrides(current, overrides)

    print("\n=== UPDATE SCORE CONFIG CAPS ===")
    print(f"env:         {cfg_env(cfg)}")
    print(f"bucket:      {bucket}")
    print(f"region:      {region}")
    print(f"engine_root: {engine_root}")
    print(f"dry_run:     {bool(args.dry_run)}")

    print("\n[score_config] requested overrides:")
    for k, v in overrides.items():
        print(f"  {k} = {v}")

    print("\n[score_config] updated values:")
    for k in sorted(CAP_FIELDS):
        print(f"  {k}: {getattr(updated, k)}")

    if args.dry_run:
        print("\n[DRY RUN] no S3 write performed.")
        print(f"[DRY RUN] would update s3://{bucket}/{engine_root}/configs/score_config/latest.json")
        return

    s3_write_json_event(
        s3,
        bucket=bucket,
        root_prefix=engine_root,
        table="configs/score_config",
        dt=today,
        filename="score_config.json",
        payload=asdict(updated),
        update_latest=True,
    )

    print(f"\n[S3] Updated s3://{bucket}/{engine_root}/configs/score_config/latest.json")


# ----------------------------
# Audit/logging entrypoint wrapper
# ----------------------------
def _tier1_audit_is_dry_run(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "dry_run", False) or getattr(args, "no_write", False))


def main_with_audit() -> None:
    args = parse_args()
    cfg = load_runtime_config(getattr(args, "env", None))
    is_dry_run = _tier1_audit_is_dry_run(args)

    with capture_script_run(
        cfg=cfg,
        script_name="run_update_score_caps.py",
        input_args=vars(args),
        dry_run=is_dry_run,
    ) as run_id:
        try:
            _main_impl()

            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="modify",
                entity_type="score_config",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="run_update_score_caps.py",
                source_mode="score_config",
                status=("dry_run" if is_dry_run else "success"),
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                    "note": "Tier 1 audit event is entrypoint-level. Detailed output keys/row counts are available in the script log stdout and script-specific metadata where emitted by the script.",
                },
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
        except Exception as exc:
            event = build_audit_event(
                cfg=cfg,
                run_id=run_id,
                event_type="modify",
                entity_type="score_config",
                entity_id=None,
                as_of=str(getattr(args, "as_of", None) or getattr(args, "dt", None) or getattr(args, "run_dt", None) or ""),
                source_script="run_update_score_caps.py",
                source_mode="score_config",
                status="failed",
                input_args=vars(args),
                metadata={
                    "tier": "tier_1",
                    "payload_policy": "large_dataset_metadata_only",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
            write_audit_event(cfg=cfg, event=event, dry_run=is_dry_run)
            raise


def main() -> None:
    main_with_audit()


if __name__ == "__main__":
    main()
