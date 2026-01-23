# run_update_score_caps.py  (S3-only I/O)
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import asdict, replace
from typing import Any, Dict, Tuple

import pandas as pd

from alpha_edge.core.schemas import ScoreConfig
from alpha_edge.core.data_loader import s3_init, s3_load_latest_json, s3_write_json_event

ENGINE_BUCKET = "alpha-edge-algo"
ENGINE_REGION = "eu-west-1"
ENGINE_ROOT_PREFIX = "engine/v1"

# Allowed overrides: "caps" + FFT band defs (NOT lambdas)
CAP_FIELDS = {
    "ruin_cap",
    "cvar_cap",
    "mdd_cap",
    "hhi_cap",
    "corr_cap",
    "time_cap_days",
    # FFT caps
    "hf_ratio_cap",
    "spec_entropy_cap",
    "freq_overlap_cap",
    # FFT bands
    "fft_bands_days",
}


def _parse_scalar(v: str) -> Any:
    v = v.strip()
    if v.lower() in {"none", "null"}:
        return None
    # int?
    if v.isdigit() or (v.startswith("-") and v[1:].isdigit()):
        return int(v)
    # float?
    try:
        return float(v)
    except ValueError:
        return v


def _parse_fft_bands(s: str) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Accepts either:
      - "2,20;20,60;60,250"
      - "2-20;20-60;60-250"
    Returns ((2.0,20.0),(20.0,60.0),(60.0,250.0))
    """
    raw = s.strip()
    if not raw:
        raise ValueError("fft_bands_days override is empty")

    bands = []
    parts = raw.split(";")
    if len(parts) != 3:
        raise ValueError("fft_bands_days must have exactly 3 bands separated by ';' (e.g. 2,20;20,60;60,250)")

    for p in parts:
        p = p.strip()
        if "," in p:
            a, b = p.split(",", 1)
        elif "-" in p:
            a, b = p.split("-", 1)
        else:
            raise ValueError(f"Bad band format '{p}'. Use 'a,b' or 'a-b'.")
        lo = float(a.strip())
        hi = float(b.strip())
        bands.append((lo, hi))

    return (bands[0], bands[1], bands[2])


def _in_0_1_or_none(name: str, x: Any) -> None:
    if x is None:
        return
    fx = float(x)
    if not (0.0 <= fx <= 1.0):
        raise ValueError(f"{name} must be in [0,1] or None. Got {x}")


def _validate_caps(cfg: ScoreConfig) -> None:
    # probability/ratio caps
    _in_0_1_or_none("ruin_cap", cfg.ruin_cap)
    _in_0_1_or_none("cvar_cap", cfg.cvar_cap)
    _in_0_1_or_none("mdd_cap", cfg.mdd_cap)
    _in_0_1_or_none("hhi_cap", cfg.hhi_cap)
    _in_0_1_or_none("corr_cap", cfg.corr_cap)

    _in_0_1_or_none("hf_ratio_cap", cfg.hf_ratio_cap)
    _in_0_1_or_none("freq_overlap_cap", cfg.freq_overlap_cap)

    # spectral entropy is not necessarily in [0,1] depending on your definition.
    # We'll just enforce finite + positive-ish.
    if cfg.spec_entropy_cap is not None:
        se = float(cfg.spec_entropy_cap)
        if not (se > 0.0):
            raise ValueError(f"spec_entropy_cap must be > 0. Got {cfg.spec_entropy_cap}")

    # time cap
    if cfg.time_cap_days is not None and int(cfg.time_cap_days) <= 0:
        raise ValueError(f"time_cap_days must be > 0 or None. Got {cfg.time_cap_days}")

    # fft bands validation
    bands = cfg.fft_bands_days
    if not isinstance(bands, tuple) or len(bands) != 3:
        raise ValueError("fft_bands_days must be a tuple of 3 (lo,hi) bands")

    prev_hi = None
    for i, (lo, hi) in enumerate(bands):
        lo = float(lo)
        hi = float(hi)
        if not (lo > 0 and hi > 0 and hi > lo):
            raise ValueError(f"fft_bands_days band {i} must satisfy 0 < lo < hi. Got {(lo, hi)}")
        if prev_hi is not None:
            # allow touching edges (e.g., 20-60 after 2-20)
            if lo < prev_hi:
                raise ValueError(
                    f"fft_bands_days bands must be non-overlapping and ordered. "
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


def main():
    p = argparse.ArgumentParser(description="Update ScoreConfig caps (including FFT caps/bands) and write to S3 latest.json")

    # classic caps
    p.add_argument("--ruin_cap", default=None)
    p.add_argument("--cvar_cap", default=None)
    p.add_argument("--mdd_cap", default=None)
    p.add_argument("--hhi_cap", default=None)
    p.add_argument("--corr_cap", default=None)
    p.add_argument("--time_cap_days", default=None)

    # FFT caps
    p.add_argument("--hf_ratio_cap", default=None)
    p.add_argument("--spec_entropy_cap", default=None)
    p.add_argument("--freq_overlap_cap", default=None)

    # FFT bands
    p.add_argument(
        "--fft_bands_days",
        default=None,
        help='Format: "2,20;20,60;60,250" (or 2-20;20-60;60-250)',
    )

    args = p.parse_args()

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
    s3 = s3_init(ENGINE_REGION)

    raw = s3_load_latest_json(
        s3, bucket=ENGINE_BUCKET, root_prefix=ENGINE_ROOT_PREFIX, table="configs/score_config"
    )
    if not raw:
        raise RuntimeError("Missing score_config/latest.json in S3")

    current = ScoreConfig(**raw)
    updated = apply_cap_overrides(current, overrides)

    print("[score_config] requested overrides:")
    for k, v in overrides.items():
        print(f"  {k} = {v}")

    print("\n[score_config] updated values:")
    for k in sorted(CAP_FIELDS):
        print(f"  {k}: {getattr(updated, k)}")

    s3_write_json_event(
        s3,
        bucket=ENGINE_BUCKET,
        root_prefix=ENGINE_ROOT_PREFIX,
        table="configs/score_config",
        dt=today,
        filename="score_config.json",
        payload=asdict(updated),
        update_latest=True,
    )

    print("\n[S3] Updated engine/v1/configs/score_config/latest.json")


if __name__ == "__main__":
    main()
