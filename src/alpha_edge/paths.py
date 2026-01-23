from __future__ import annotations

import os
from pathlib import Path

def project_root() -> Path:
    """
    Best-effort project root resolver.
    Priority:
      1) env ALPHA_EDGE_ROOT
      2) walk upwards from this file until pyproject.toml found
      3) fallback: current working directory
    """
    env = os.getenv("ALPHA_EDGE_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd().resolve()


def data_dir() -> Path:
    return project_root() / "data"


def universe_dir() -> Path:
    return data_dir() / "universe"


def local_outputs_dir() -> Path:
    """
    All debug/local artifacts go here.
    Keep it OUT of src/ so packaging doesn't matter.
    """
    return data_dir() / "outputs"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
