#!/usr/bin/env python3
"""Normalize calibration .npz files into the model dataset format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "model" / "data"


def _scalar_string(sample, key: str, default: str = "") -> str:
    if key not in sample:
        return default
    value = sample[key]
    try:
        return str(value.item())
    except Exception:
        return str(value)


def normalize_sample(
    source_path: Path,
    output_dir: Path,
    subject_id: str,
    label_source: str,
    label_status: str,
    overwrite: bool = False,
) -> Path:
    sample = np.load(source_path, allow_pickle=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_type = _scalar_string(sample, "robot_type", "unknown")
    input_type = _scalar_string(sample, "input_type", "unknown")
    hand_side = _scalar_string(sample, "hand_side", "unknown")
    stem = source_path.stem
    output_path = output_dir / f"{stem}_{subject_id}_{label_status}.npz"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite to replace")

    fields = {key: sample[key] for key in sample.files}
    fields.update({
        "subject_id": np.asarray(subject_id),
        "capture_id": np.asarray(stem),
        "label_source": np.asarray(label_source),
        "label_status": np.asarray(label_status),
        "schema_version": np.asarray("calibration_sample_v1"),
        "source_path": np.asarray(str(source_path)),
        "robot_type": np.asarray(robot_type),
        "input_type": np.asarray(input_type),
        "hand_side": np.asarray(hand_side),
    })
    np.savez_compressed(output_path, **fields)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize calibration samples into model/data")
    parser.add_argument("paths", nargs="+", help="Source .npz files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--subject-id", default="subject_000", help="Anonymous subject identifier")
    parser.add_argument("--label-source", default="current_yaml", choices=["current_yaml", "geometry_calibration", "functional_calibration", "manual_tuned"])
    parser.add_argument("--label-status", default="seed", choices=["seed", "raw", "calibrated", "verified"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    for raw_path in args.paths:
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        output_path = normalize_sample(
            source_path=source_path,
            output_dir=output_dir,
            subject_id=args.subject_id,
            label_source=args.label_source,
            label_status=args.label_status,
            overwrite=args.overwrite,
        )
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
