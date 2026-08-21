#!/usr/bin/env python3
"""Run debug_skeleton with checkpoint-predicted calibration parameters.

This wrapper reuses example/test/debug_skeleton.py for the actual visualization,
but intercepts Retargeter construction so the retargeter receives an in-memory
config with calibration values predicted from a model checkpoint.  No YAML file
is generated or modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = PROJECT_ROOT / "example"
TEST_ROOT = EXAMPLE_ROOT / "test"
MODEL_ROOT = PROJECT_ROOT / "model"
CONFIG_ROOT = MODEL_ROOT / "config"
for path in (PROJECT_ROOT, EXAMPLE_ROOT, TEST_ROOT, MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydexretarget import Retargeter
from utils.ckpt_calibration import (
    apply_prediction_to_config,
    predict_calibration,
    print_prediction,
    resolve_project_path,
)

ROBOT_NAME_MAP = {
    "shadow": "shadow_hand",
    "wuji": "wuji_hand",
    "allegro": "allegro_hand",
    "leap": "leap_hand",
    "inspire": "inspire_hand",
    "ability": "ability_hand",
    "svh": "svh_hand",
    "rohand": "rohand",
    "linkerhand_l21": "linkerhand_l21",
    "linker_l20": "linker_l20",
    "unitree_dex5": "unitree_dex5_hand",
    "sharpa": "sharpa_hand",
    "gaia": "gaia_hand20",
}

INPUT_TO_CONFIG_DIR = {
    "noitom": "noitom",
    "avp": "avp",
    "quest3": "quest3",
    "pico4": "pico4",
}


def resolve_debug_config(args) -> Path:
    if args.config:
        path = Path(args.config)
        if path.is_absolute():
            return path
        return resolve_project_path(path)
    robot_file = ROBOT_NAME_MAP.get(args.robot, args.robot)
    config_dir = INPUT_TO_CONFIG_DIR.get(args.input, "mediapipe")
    return resolve_project_path(args.config_root) / f"{args.optimizer}/{config_dir}/{config_dir}_{robot_file}.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug skeleton visualization with checkpoint-predicted calibration"
    )
    parser.add_argument("--sample", required=True, help="Calibration .npz sample used as model input")
    parser.add_argument("--checkpoint", required=True, help="Calibration MLP checkpoint")
    parser.add_argument("--print-only", action="store_true", help="Only print prediction, do not open viewer")

    # Keep these aligned with example/test/debug_skeleton.py so the wrapper can
    # resolve the same base config before forwarding all remaining args.
    parser.add_argument("--config", default=None, help="Config YAML path (overrides --robot and --optimizer)")
    parser.add_argument("--config-root", default=str(CONFIG_ROOT), help="Model config root, default: model/config")
    parser.add_argument("--optimizer", default="adaptive", choices=["adaptive", "vector"])
    parser.add_argument("--robot", default="leap", choices=list(ROBOT_NAME_MAP.keys()))
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--input", default="camera", choices=["camera", "video", "replay", "noitom", "realsense", "avp", "quest3", "pico4"])

    args, passthrough = parser.parse_known_args()

    sample_path = resolve_project_path(args.sample)
    checkpoint_path = resolve_project_path(args.checkpoint)
    config_path = resolve_debug_config(args)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    segment_scaling, pinch_scaling = predict_calibration(sample_path, checkpoint_path)
    print_prediction(segment_scaling, pinch_scaling)
    predicted_config = apply_prediction_to_config(base_config, segment_scaling, pinch_scaling)
    print("Using predicted calibration in memory; no YAML file is written.")

    if args.print_only:
        return

    original_from_yaml = Retargeter.from_yaml

    @classmethod
    def from_yaml_with_prediction(cls, yaml_path: str, hand_side: str = "right"):
        requested = Path(yaml_path).resolve()
        if requested == config_path.resolve():
            return cls.from_config(predicted_config, hand_side)
        return original_from_yaml(yaml_path, hand_side)

    Retargeter.from_yaml = from_yaml_with_prediction

    import debug_skeleton

    forwarded = [
        "debug_skeleton.py",
        "--optimizer", args.optimizer,
        "--robot", args.robot,
        "--hand", args.hand,
        "--input", args.input,
    ]
    if args.config:
        forwarded.extend(["--config", args.config])
    forwarded.extend(passthrough)

    old_argv = sys.argv
    try:
        sys.argv = forwarded
        debug_skeleton.main()
    finally:
        sys.argv = old_argv
        Retargeter.from_yaml = original_from_yaml


if __name__ == "__main__":
    main()
