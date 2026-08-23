from __future__ import annotations

import numpy as np
import pytest

from alpha_edge.risk.actuarial.path_metrics import validate_equity_paths


def test_validate_equity_paths_accepts_sub_cent_initial_value_drift():
    initial_value = 28693.50675574092
    paths = np.array(
        [
            [28693.505859375, 28700.0, 28800.0],
            [28693.505859375, 28650.0, 28600.0],
        ],
        dtype=float,
    )

    out = validate_equity_paths(paths, initial_value=initial_value)

    assert out.shape == paths.shape


def test_validate_equity_paths_rejects_material_initial_value_drift():
    initial_value = 28693.50
    paths = np.array(
        [
            [28690.00, 28700.0, 28800.0],
            [28690.00, 28650.0, 28600.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="tolerance=0.01"):
        validate_equity_paths(paths, initial_value=initial_value)
