"""Tests for robot-agnostic qpos-to-actuator calibration maps."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from output.real.joint_command_mapping import (
    map_joint_command,
    resolve_joint_command_mapping,
)


def test_piecewise_mapping_preserves_calibrated_midpoint() -> None:
    spec = {
        "input": [0.1, 0.7, 1.4],
        "output": [255, 128, 0],
    }

    assert map_joint_command(0.7, spec) == pytest.approx(128.0)
    assert map_joint_command(0.4, spec) == pytest.approx(191.5)


def test_same_algorithm_supports_different_robot_ranges() -> None:
    short_range = {"input": [0.0, 0.8], "output": [0.0, 1.0]}
    long_range = {"input": [-0.4, 1.2], "output": [0.0, 1.0]}

    assert map_joint_command(0.4, short_range) == pytest.approx(0.5)
    assert map_joint_command(0.4, long_range) == pytest.approx(0.5)


def test_side_mapping_merges_default_with_selected_hand() -> None:
    mapping = {
        "default": {
            "thumb_pitch": {"input": [0.0, 1.0], "output": [255, 0]},
        },
        "left": {
            "index_roll": {"input": [-1.0, 1.0], "output": [255, 0]},
        },
        "right": {
            "index_roll": {"input": [-1.0, 1.0], "output": [0, 255]},
        },
    }

    left = resolve_joint_command_mapping(mapping, "left")
    right = resolve_joint_command_mapping(mapping, "right")

    assert set(left) == {"thumb_pitch", "index_roll"}
    assert map_joint_command(1.0, left["index_roll"]) == pytest.approx(0.0)
    assert map_joint_command(1.0, right["index_roll"]) == pytest.approx(255.0)


def test_mapping_rejects_non_increasing_input_points() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        map_joint_command(
            0.5,
            {"input": [0.0, 0.5, 0.5], "output": [0.0, 0.5, 1.0]},
        )


def test_mapping_rejects_non_monotonic_output_points() -> None:
    with pytest.raises(ValueError, match="output points must be monotonic"):
        map_joint_command(
            0.5,
            {"input": [0.0, 0.5, 1.0], "output": [0.0, 1.0, 0.5]},
        )
