"""Feature extraction for calibration-parameter prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
INPUT_TYPES = ["mediapipe", "noitom", "quest3", "avp", "pico4"]
HAND_SIDES = ["left", "right"]

MP_MCP_INDICES = [1, 5, 9, 13, 17]
MP_PIP_INDICES = [2, 6, 10, 14, 18]
MP_DIP_INDICES = [3, 7, 11, 15, 19]
MP_TIP_INDICES = [4, 8, 12, 16, 20]
MP_ORIGIN_IDX = 0


@dataclass(frozen=True)
class RobotGeometry:
    """Robot geometry features used by the model.

    Attributes:
        segment_lengths: (5, 4) origin/root/proximal/middle/distal lengths.
        tip_reaches: (5,) origin->tip distances.
        root_positions: optional (5, 3) finger root positions relative to origin.
    """

    segment_lengths: np.ndarray
    tip_reaches: np.ndarray
    root_positions: np.ndarray | None = None


def _safe_norm(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=axis)


def _one_hot(value: str, choices: Iterable[str]) -> np.ndarray:
    choices = list(choices)
    arr = np.zeros(len(choices), dtype=np.float64)
    if value in choices:
        arr[choices.index(value)] = 1.0
    return arr


def human_bone_lengths(keypoints_open: np.ndarray) -> np.ndarray:
    """Return per-frame human finger segment lengths, shape (T, 5, 4)."""
    kp = np.asarray(keypoints_open, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[1:] != (21, 3):
        raise ValueError(f"Expected keypoints_open shape (T, 21, 3), got {kp.shape}")

    wrist = kp[:, MP_ORIGIN_IDX]
    mcp = kp[:, MP_MCP_INDICES]
    pip = kp[:, MP_PIP_INDICES]
    dip = kp[:, MP_DIP_INDICES]
    tip = kp[:, MP_TIP_INDICES]
    return np.stack(
        [
            _safe_norm(mcp - wrist[:, None, :]),
            _safe_norm(pip - mcp),
            _safe_norm(dip - pip),
            _safe_norm(tip - dip),
        ],
        axis=-1,
    )


def human_tip_reaches(keypoints_open: np.ndarray) -> np.ndarray:
    """Return per-frame wrist->tip distances, shape (T, 5)."""
    kp = np.asarray(keypoints_open, dtype=np.float64)
    wrist = kp[:, MP_ORIGIN_IDX]
    tip = kp[:, MP_TIP_INDICES]
    return _safe_norm(tip - wrist[:, None, :])


def human_palm_features(keypoints_open: np.ndarray) -> np.ndarray:
    """Return per-frame palm width/spread features."""
    kp = np.asarray(keypoints_open, dtype=np.float64)
    mcp = kp[:, MP_MCP_INDICES]
    pairs = [(1, 2), (2, 3), (3, 4), (1, 4), (0, 1)]
    return np.stack([_safe_norm(mcp[:, a] - mcp[:, b]) for a, b in pairs], axis=-1)


def _median_std(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.concatenate([
        np.median(values, axis=0).reshape(-1),
        np.std(values, axis=0).reshape(-1),
    ])


def robot_geometry_from_optimizer(optimizer) -> RobotGeometry:
    """Compute robot geometry from an optimizer's neutral FK."""
    robot = optimizer.robot
    qpos = (
        optimizer.neutral_qpos.copy()
        if optimizer.neutral_qpos is not None
        else np.zeros(robot.model.nq, dtype=np.float64)
    )
    robot.compute_forward_kinematics(qpos)

    def point(name: str, offset=None) -> np.ndarray:
        pose = robot.get_link_pose(robot.get_link_index(name))
        pos = pose[:3, 3].copy()
        if offset is not None:
            pos += pose[:3, :3] @ np.asarray(offset, dtype=np.float64)
        return pos

    origin = point(optimizer.origin_link_name)
    nf = optimizer.num_fingers
    roots = np.zeros((5, 3), dtype=np.float64)
    segment_lengths = np.zeros((5, 4), dtype=np.float64)
    tip_reaches = np.zeros(5, dtype=np.float64)

    for i in range(nf):
        root = point(optimizer.link1_names[i])
        pip = point(optimizer.link3_names[i], optimizer.link3_offsets[i])
        dip = point(optimizer.link4_names[i], optimizer.link4_offsets[i])
        tip = point(optimizer.task_link_names[i], optimizer.task_offsets[i])
        roots[i] = root - origin
        segment_lengths[i] = [
            np.linalg.norm(root - origin),
            np.linalg.norm(pip - root),
            np.linalg.norm(dip - pip),
            np.linalg.norm(tip - dip),
        ]
        tip_reaches[i] = np.linalg.norm(tip - origin)

    return RobotGeometry(segment_lengths=segment_lengths, tip_reaches=tip_reaches, root_positions=roots)


def extract_features(
    keypoints_open: np.ndarray,
    robot_geometry: RobotGeometry,
    input_type: str,
    hand_side: str,
) -> np.ndarray:
    """Build a fixed-length feature vector for calibration prediction."""
    human_segments = human_bone_lengths(keypoints_open)
    human_reaches = human_tip_reaches(keypoints_open)
    human_palm = human_palm_features(keypoints_open)

    robot_parts = [
        np.asarray(robot_geometry.segment_lengths, dtype=np.float64).reshape(-1),
        np.asarray(robot_geometry.tip_reaches, dtype=np.float64).reshape(-1),
    ]
    if robot_geometry.root_positions is not None:
        roots = np.asarray(robot_geometry.root_positions, dtype=np.float64)
        robot_parts.append(roots.reshape(-1))
        robot_parts.append(np.array([
            np.linalg.norm(roots[1] - roots[2]),
            np.linalg.norm(roots[2] - roots[3]),
            np.linalg.norm(roots[3] - roots[4]),
            np.linalg.norm(roots[1] - roots[4]),
            np.linalg.norm(roots[0] - roots[1]),
        ], dtype=np.float64))

    return np.concatenate([
        _median_std(human_segments),
        _median_std(human_reaches),
        _median_std(human_palm),
        *robot_parts,
        _one_hot(input_type, INPUT_TYPES),
        _one_hot(hand_side, HAND_SIDES),
    ]).astype(np.float32)


def target_vector(segment_scaling: np.ndarray, pinch_scaling: float) -> np.ndarray:
    """Flatten target parameters to 21 values."""
    segment = np.asarray(segment_scaling, dtype=np.float64)
    if segment.shape != (5, 4):
        raise ValueError(f"Expected segment_scaling shape (5, 4), got {segment.shape}")
    return np.concatenate([segment.reshape(-1), [float(pinch_scaling)]]).astype(np.float32)


def split_prediction(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Split 21 model outputs into segment_scaling and pinch_scaling."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 21:
        raise ValueError(f"Expected 21 values, got {arr.size}")
    return arr[:20].reshape(5, 4), float(arr[20])
