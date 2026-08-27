"""Robot-agnostic monotonic mapping from model joints to actuator commands."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def map_joint_command(value: float, spec: Mapping[str, object]) -> float:
    """Project one model joint through a monotonic piecewise-linear map."""
    input_points = np.asarray(spec.get("input"), dtype=np.float64)
    output_points = np.asarray(spec.get("output"), dtype=np.float64)
    if input_points.ndim != 1 or output_points.ndim != 1:
        raise ValueError("Joint command mapping points must be one-dimensional")
    if len(input_points) < 2 or len(input_points) != len(output_points):
        raise ValueError("Joint command mapping requires matching point arrays")
    if not np.all(np.isfinite(input_points)) or not np.all(np.isfinite(output_points)):
        raise ValueError("Joint command mapping points must be finite")
    if np.any(np.diff(input_points) <= 0.0):
        raise ValueError("Joint command mapping input points must be strictly increasing")
    output_steps = np.diff(output_points)
    if not (np.all(output_steps >= 0.0) or np.all(output_steps <= 0.0)):
        raise ValueError("Joint command mapping output points must be monotonic")
    if not np.isfinite(value):
        raise ValueError("Joint command input must be finite")
    return float(np.interp(float(value), input_points, output_points))


def resolve_joint_command_mapping(
    mapping: Mapping[str, object] | None,
    hand_side: str,
) -> dict[str, Mapping[str, object]]:
    """Merge shared/default calibration with the selected hand side."""
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise TypeError("Joint command mapping must be a dictionary")

    side_keys = {"default", "left", "right"}
    if not side_keys.intersection(mapping):
        return dict(mapping)

    hand_side = hand_side.lower()
    if hand_side not in {"left", "right"}:
        raise ValueError(f"Hand side must be left or right, got {hand_side}")
    shared = {key: value for key, value in mapping.items() if key not in side_keys}
    default = mapping.get("default", {})
    selected = mapping.get(hand_side, {})
    if not isinstance(default, Mapping) or not isinstance(selected, Mapping):
        raise TypeError("Side-aware joint command mappings must contain dictionaries")
    return {**shared, **default, **selected}


__all__ = ["map_joint_command", "resolve_joint_command_mapping"]
