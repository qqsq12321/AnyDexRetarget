"""L20-specific checks for Pico 4 retargeting v2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anydexretarget.optimizer import (
    AdaptiveLPFilterV2,
    AdaptiveOptimizerAnalyticalV2,
)
from anydexretarget.retarget import Retargeter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PROJECT_ROOT / "example/config/adaptive/pico4/pico4_linker_l20_v2.yaml"
)


def _synthetic_open_hand() -> np.ndarray:
    keypoints = np.zeros((21, 3), dtype=np.float64)
    roots = {1: -2.0, 5: -1.5, 9: -0.5, 13: 0.5, 17: 1.5}
    for root_index, x_value in roots.items():
        for offset in range(4):
            keypoints[root_index + offset] = [x_value, 0.0, float(offset)]
    return keypoints


def test_pico_l20_v2_config_selects_v2_optimizer_and_filter() -> None:
    retargeter = Retargeter.from_yaml(str(CONFIG_PATH), hand_side="right")

    assert isinstance(retargeter.optimizer, AdaptiveOptimizerAnalyticalV2)
    assert isinstance(retargeter.lp_filter, AdaptiveLPFilterV2)
    assert retargeter.optimizer.direct_only_v2 is False


def test_l20_v2_single_finger_curl_drives_base_and_tip_channels() -> None:
    optimizer = Retargeter.from_yaml(
        str(CONFIG_PATH), hand_side="right"
    ).optimizer
    keypoints = _synthetic_open_hand()
    keypoints[6] = [-1.5, 0.0, 1.0]
    keypoints[7] = [-1.5, 1.0, 1.0]
    keypoints[8] = [-1.5, 1.0, 0.0]

    target = optimizer._compute_direct_targets_v2(keypoints)
    names = list(optimizer.robot.dof_joint_names)

    assert target[names.index("index_mcp_pitch")] > 1.0
    assert target[names.index("index_pip")] > 0.9
    assert np.isclose(target[names.index("middle_mcp_pitch")], 0.0)
    assert np.isclose(target[names.index("middle_pip")], 0.0)


def test_l20_v2_enables_all_four_pinch_partners() -> None:
    optimizer = Retargeter.from_yaml(
        str(CONFIG_PATH), hand_side="right"
    ).optimizer

    assert optimizer.contact_enabled_partners_v2 == (1, 2, 3, 4)
