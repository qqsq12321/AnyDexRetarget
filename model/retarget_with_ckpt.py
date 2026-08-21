#!/usr/bin/env python3
"""Run retargeting with calibration parameters predicted from a checkpoint.

This script keeps the normal teleop_sim retargeting path, but replaces the
YAML calibration fields with values predicted by model/checkpoints/*.pt before
constructing the Retargeter.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = PROJECT_ROOT / "example"
CONFIG_ROOT = MODEL_ROOT / "config"
for path in (PROJECT_ROOT, EXAMPLE_ROOT, MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import CalibrationDataset
from features import FINGER_NAMES, split_prediction
from network import CalibrationMLP, torch
from teleop_sim import (
    DEFAULT_PICO4_BROADCAST_PORT,
    DEFAULT_PICO4_PORT,
    DEFAULT_PICO4_RELAY_HOST,
    DEFAULT_PICO4_RELAY_PORT,
    run_teleop,
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
    "quest3": "quest3",
    "visionpro": "avp",
    "noitom": "noitom",
    "pico4": "pico4",
}


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_config_path(args) -> Path:
    if args.config:
        return _resolve_project_path(args.config)
    config_dir = INPUT_TO_CONFIG_DIR.get(args.input, "mediapipe")
    robot_file = ROBOT_NAME_MAP.get(args.robot, args.robot)
    return _resolve_project_path(args.config_root) / f"{args.optimizer}/{config_dir}/{config_dir}_{robot_file}.yaml"


def predict_calibration(sample_path: Path, checkpoint_path: Path) -> tuple[np.ndarray, float]:
    if torch is None:
        raise RuntimeError("PyTorch is required to load calibration checkpoints")

    dataset = CalibrationDataset([sample_path])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CalibrationMLP(int(ckpt["input_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = dataset.features.astype(np.float32)
    mean = ckpt.get("feature_mean")
    std = ckpt.get("feature_std")
    if mean is not None and std is not None:
        x = (x - mean) / std

    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32)).numpy()[0]
    return split_prediction(pred)


def apply_prediction_to_config(config: dict, segment_scaling: np.ndarray, pinch_scaling: float) -> dict:
    updated = copy.deepcopy(config)
    retarget = updated.setdefault("retarget", {})
    retarget["segment_scaling"] = {
        name: [round(float(v), 4) for v in values]
        for name, values in zip(FINGER_NAMES, segment_scaling)
    }
    retarget["pinch_scaling"] = round(float(pinch_scaling), 4)
    return updated


def print_prediction(segment_scaling: np.ndarray, pinch_scaling: float) -> None:
    print("Predicted calibration:")
    print("  segment_scaling:")
    for name, values in zip(FINGER_NAMES, segment_scaling):
        print(f"    {name}: [{', '.join(f'{float(v):.4f}' for v in values)}]")
    print(f"  pinch_scaling: {float(pinch_scaling):.4f}")


def write_generated_config(config: dict, output_config: str) -> Path:
    out = _resolve_project_path(output_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run teleop_sim with calibration parameters predicted from a model checkpoint"
    )
    parser.add_argument("--sample", required=True, help="Calibration .npz sample used as model input")
    parser.add_argument("--checkpoint", required=True, help="Calibration MLP checkpoint")
    parser.add_argument("--output-config", default=None, help="Optional path to save the generated YAML config")
    parser.add_argument("--print-only", action="store_true", help="Only print predicted parameters and generated config path")

    parser.add_argument("--config", default=None, help="Base YAML config; overrides --robot/--optimizer config resolution")
    parser.add_argument("--config-root", default=str(CONFIG_ROOT), help="Model config root, default: model/config")
    parser.add_argument("--optimizer", default="adaptive", choices=["adaptive", "vector"])
    parser.add_argument("--robot", default="linker_l20", choices=list(ROBOT_NAME_MAP.keys()))
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--input", default="video", choices=["visionpro", "quest3", "pico4", "noitom", "mediapipe_replay", "camera", "realsense", "video"])
    parser.add_argument("--video", default=None)
    parser.add_argument("--play", default=None, help="MediaPipe replay file")
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--video-depth-scale", type=float, default=1.25)
    parser.add_argument("--ip", default="192.168.50.127")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--protocol", default="udp", choices=["udp", "tcp"])
    parser.add_argument("--pico4-mode", default="relay", choices=["relay", "direct"])
    parser.add_argument("--pico4-relay-host", default=DEFAULT_PICO4_RELAY_HOST)
    parser.add_argument("--pico4-relay-port", type=int, default=DEFAULT_PICO4_RELAY_PORT)
    parser.add_argument("--pico4-port", type=int, default=DEFAULT_PICO4_PORT)
    parser.add_argument("--pico4-broadcast-port", type=int, default=DEFAULT_PICO4_BROADCAST_PORT)
    parser.add_argument("--noitom-local-ip", default="192.168.5.25")
    parser.add_argument("--noitom-local-port", type=int, default=8000)
    parser.add_argument("--noitom-server-ip", default="192.168.5.33")
    parser.add_argument("--noitom-server-port", type=int, default=9000)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-sim", default=None)
    parser.add_argument("--save-qpos", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--cam-azimuth", type=float, default=None)
    parser.add_argument("--cam-elevation", type=float, default=None)
    parser.add_argument("--cam-distance", type=float, default=None)
    args = parser.parse_args()

    input_type = args.input
    video_path = args.video or ""
    replay_path = args.play or ""
    if args.video:
        input_type = "video"
    if args.play:
        input_type = "mediapipe_replay"
    if input_type == "video" and not video_path:
        video_path = "data/right.mp4"

    sample_path = _resolve_project_path(args.sample)
    checkpoint_path = _resolve_project_path(args.checkpoint)
    base_config_path = resolve_config_path(args)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    with open(base_config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    segment_scaling, pinch_scaling = predict_calibration(sample_path, checkpoint_path)
    print_prediction(segment_scaling, pinch_scaling)
    generated_config = apply_prediction_to_config(base_config, segment_scaling, pinch_scaling)
    if args.output_config:
        generated_config_path = write_generated_config(generated_config, args.output_config)
        print(f"Saved generated config: {generated_config_path}")

    if args.print_only:
        return

    run_teleop(
        hand_side=args.hand,
        config_path=str(base_config_path.relative_to(EXAMPLE_ROOT) if base_config_path.is_relative_to(EXAMPLE_ROOT) else base_config_path),
        input_device_type=input_type,
        mediapipe_replay_path=replay_path,
        video_path=video_path,
        visionpro_ip=args.ip,
        quest3_port=args.port,
        quest3_protocol=args.protocol,
        pico4_mode=args.pico4_mode,
        pico4_relay_host=args.pico4_relay_host,
        pico4_relay_port=args.pico4_relay_port,
        pico4_port=args.pico4_port,
        pico4_broadcast_port=args.pico4_broadcast_port,
        noitom_local_ip=args.noitom_local_ip,
        noitom_local_port=args.noitom_local_port,
        noitom_server_ip=args.noitom_server_ip,
        noitom_server_port=args.noitom_server_port,
        playback_speed=args.speed,
        playback_loop=not args.no_loop,
        enable_recording=False,
        show_video=args.show_video,
        video_depth_scale=args.video_depth_scale,
        headless=args.headless or args.save_sim is not None,
        output_video_path=args.save_sim or "",
        output_qpos_path=args.save_qpos or "",
        render_width=args.width,
        render_height=args.height,
        max_frames=args.max_frames,
        camera_azimuth=args.cam_azimuth,
        camera_elevation=args.cam_elevation,
        camera_distance=args.cam_distance,
        config_override=generated_config,
    )


if __name__ == "__main__":
    main()
