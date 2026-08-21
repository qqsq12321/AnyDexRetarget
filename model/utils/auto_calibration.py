#!/usr/bin/env python3
"""Automatic calibration-label utilities for model dataset samples.

The functions here turn a saved open-hand sample into supervised labels for the
calibration model.  They intentionally operate on ``model/data/*.npz`` files and
never modify the repository's example/config YAML files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEGENERATE_SEGMENT_M = 5e-4

MP_MCP_INDICES = [1, 5, 9, 13, 17]
MP_PIP_INDICES = [2, 6, 10, 14, 18]
MP_DIP_INDICES = [3, 7, 11, 15, 19]
MP_TIP_INDICES = [4, 8, 12, 16, 20]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def human_segment_lengths(keypoints_open: np.ndarray) -> np.ndarray:
    """Return robust human open-hand segment lengths, shape ``(5, 4)``.

    Columns are ``wrist->MCP``, ``MCP->PIP``, ``PIP->DIP``, ``DIP->TIP``.
    The median over the capture sequence suppresses tracking spikes.
    """
    kp = np.asarray(keypoints_open, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[1:] != (21, 3):
        raise ValueError(f"Expected keypoints_open shape (T, 21, 3), got {kp.shape}")

    wrist = kp[:, 0]
    mcp = kp[:, MP_MCP_INDICES]
    pip = kp[:, MP_PIP_INDICES]
    dip = kp[:, MP_DIP_INDICES]
    tip = kp[:, MP_TIP_INDICES]
    per_frame = np.stack(
        [
            np.linalg.norm(mcp - wrist[:, None, :], axis=-1),
            np.linalg.norm(pip - mcp, axis=-1),
            np.linalg.norm(dip - pip, axis=-1),
            np.linalg.norm(tip - dip, axis=-1),
        ],
        axis=-1,
    )
    return np.median(per_frame, axis=0)


def _safe_ratio(robot_len: float, human_len: float, default: float = 1.0) -> float:
    if human_len > 1e-4 and robot_len > 1e-6:
        return float(robot_len / human_len)
    return float(default)


def auto_segment_scaling(
    keypoints_open: np.ndarray,
    robot_segment_lengths: np.ndarray,
    *,
    round_digits: int = 4,
) -> np.ndarray:
    """Compute geometry-calibration labels from one open-hand capture.

    Args:
        keypoints_open: ``(T, 21, 3)`` transformed open-hand keypoints.
        robot_segment_lengths: ``(5, 4)`` robot geometry lengths with columns
            ``root``, ``proximal``, ``middle``, ``distal``.
        round_digits: number of decimals used for dataset labels.

    Returns:
        ``(5, 4)`` segment_scaling label.
    """
    human = human_segment_lengths(keypoints_open)
    robot = np.asarray(robot_segment_lengths, dtype=np.float64)
    if robot.shape != (5, 4):
        raise ValueError(f"Expected robot_segment_lengths shape (5, 4), got {robot.shape}")

    result = np.ones((5, 4), dtype=np.float64)
    for i in range(5):
        result[i, 0] = _safe_ratio(robot[i, 0], human[i, 0])
        r0, r1, r2 = robot[i, 1], robot[i, 2], robot[i, 3]
        h0, h1, h2 = human[i, 1], human[i, 2], human[i, 3]
        if r0 < DEGENERATE_SEGMENT_M:
            shared = _safe_ratio(r1, h0 + h1)
            result[i, 1] = shared
            result[i, 2] = shared
            result[i, 3] = _safe_ratio(r2, h2)
        else:
            result[i, 1] = _safe_ratio(r0, h0)
            result[i, 2] = _safe_ratio(r1, h1)
            result[i, 3] = _safe_ratio(r2, h2)
    return np.round(result, round_digits).astype(np.float32)


def auto_pinch_scaling(
    keypoints_open: np.ndarray,
    robot_tip_reaches: np.ndarray,
    *,
    finger_index: int = 1,
    round_digits: int = 4,
) -> float:
    """Compute a simple geometry label for ``pinch_scaling``.

    This mirrors ``example/test/calibrate_pinch_scaling.py``: use an open-hand
    wrist->index-tip reach ratio as the uniform pinch scale.  ``finger_index`` is
    a MediaPipe/global finger index where 1 is index finger.
    """
    kp = np.asarray(keypoints_open, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[1:] != (21, 3):
        raise ValueError(f"Expected keypoints_open shape (T, 21, 3), got {kp.shape}")
    robot_tip_reaches = np.asarray(robot_tip_reaches, dtype=np.float64).reshape(-1)
    if finger_index >= robot_tip_reaches.size:
        raise ValueError(
            f"finger_index {finger_index} is outside robot_tip_reaches shape {robot_tip_reaches.shape}"
        )

    tip_idx = MP_TIP_INDICES[finger_index]
    reaches = np.linalg.norm(kp[:, tip_idx] - kp[:, 0], axis=-1)
    human_reach = float(np.median(reaches[np.isfinite(reaches)]))
    if human_reach <= 1e-6:
        return 1.0
    return round(float(robot_tip_reaches[finger_index] / human_reach), round_digits)


def update_sample_label(
    sample_path: str | Path,
    output_path: str | Path | None = None,
    *,
    pinch_scaling: float | None = None,
    auto_pinch: bool = True,
    pinch_finger_index: int = 1,
    label_source: str = "geometry_calibration",
    label_status: str = "calibrated",
    overwrite: bool = False,
) -> Path:
    """Write a copy of ``sample_path`` with geometry-calibration labels applied."""
    sample_path = resolve_project_path(sample_path)
    sample = np.load(sample_path, allow_pickle=True)
    if "keypoints_open" not in sample:
        raise KeyError(f"{sample_path} missing keypoints_open")
    segment_scaling = auto_segment_scaling(
        sample["keypoints_open"],
        sample["robot_segment_lengths"],
    )

    if pinch_scaling is None:
        if auto_pinch:
            if "robot_tip_reaches" not in sample:
                raise KeyError(f"{sample_path} missing robot_tip_reaches")
            pinch_scaling = auto_pinch_scaling(
                sample["keypoints_open"],
                sample["robot_tip_reaches"],
                finger_index=pinch_finger_index,
            )
        else:
            pinch_scaling = float(sample["pinch_scaling"]) if "pinch_scaling" in sample else 1.0

    if output_path is None:
        output_path = sample_path.with_name(f"{sample_path.stem}_{label_status}.npz")
    output_path = resolve_project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass overwrite=True or --overwrite")

    fields = {key: sample[key] for key in sample.files}
    fields.update({
        "segment_scaling": segment_scaling,
        "pinch_scaling": np.asarray(float(pinch_scaling), dtype=np.float32),
        "label_source": np.asarray(label_source),
        "label_status": np.asarray(label_status),
        "schema_version": np.asarray("calibration_sample_v1"),
    })
    np.savez_compressed(output_path, **fields)
    return output_path


def print_segment_scaling(segment_scaling: np.ndarray, pinch_scaling: float) -> None:
    print("segment_scaling:")
    for name, values in zip(FINGER_NAMES, segment_scaling):
        print(f"  {name}: [{', '.join(f'{float(v):.4f}' for v in values)}]")
    print(f"pinch_scaling: {float(pinch_scaling):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label model dataset samples from open-hand geometry")
    parser.add_argument("sample", help="Input .npz sample")
    parser.add_argument("--output", default=None, help="Output .npz; default adds _calibrated suffix")
    parser.add_argument("--pinch-scaling", type=float, default=None, help="Override pinch_scaling label")
    parser.add_argument("--keep-existing-pinch", action="store_true", help="Keep existing sample pinch_scaling instead of auto-labeling it")
    parser.add_argument("--pinch-finger-index", type=int, default=1, help="Finger index used for auto pinch label; default 1=index")
    parser.add_argument("--label-source", default="geometry_calibration")
    parser.add_argument("--label-status", default="calibrated")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-only", action="store_true", help="Print computed label without writing")
    args = parser.parse_args()

    sample_path = resolve_project_path(args.sample)
    sample = np.load(sample_path, allow_pickle=True)
    segment_scaling = auto_segment_scaling(sample["keypoints_open"], sample["robot_segment_lengths"])
    pinch_scaling = args.pinch_scaling
    if pinch_scaling is None:
        if args.keep_existing_pinch:
            pinch_scaling = float(sample["pinch_scaling"]) if "pinch_scaling" in sample else 1.0
        else:
            pinch_scaling = auto_pinch_scaling(
                sample["keypoints_open"],
                sample["robot_tip_reaches"],
                finger_index=args.pinch_finger_index,
            )
    print_segment_scaling(segment_scaling, pinch_scaling)

    if args.print_only:
        return

    out = update_sample_label(
        sample_path,
        args.output,
        pinch_scaling=pinch_scaling,
        auto_pinch=not args.keep_existing_pinch,
        pinch_finger_index=args.pinch_finger_index,
        label_source=args.label_source,
        label_status=args.label_status,
        overwrite=args.overwrite,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
