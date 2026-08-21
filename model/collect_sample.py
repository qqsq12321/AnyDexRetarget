#!/usr/bin/env python3
"""Collect one training sample for calibration-parameter prediction.

This script does not calibrate. It records an open-hand keypoint sequence and
uses the current YAML/optimizer values as labels for model training.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "model"
CONFIG_ROOT = MODEL_ROOT / "config"
EXAMPLE_ROOT = PROJECT_ROOT / "example"
TEST_ROOT = EXAMPLE_ROOT / "test"
for path in (PROJECT_ROOT, EXAMPLE_ROOT, TEST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anydexretarget import Retargeter
from calibrate_scaling import (  # Reuse existing input plumbing.
    FINGER_NAMES,
    INPUT_TO_CONFIG_DIR,
    ROBOT_NAME_MAP,
    _pose_countdown,
    _transform_input_keypoints,
    _wait_for_capture_start,
    create_input_device,
)
from features import robot_geometry_from_optimizer


def _resolve_config(args) -> Path:
    robot_file = ROBOT_NAME_MAP.get(args.robot, args.robot)
    config_dir = INPUT_TO_CONFIG_DIR[args.input]
    return (Path(args.config_root) / f"adaptive/{config_dir}/{config_dir}_{robot_file}.yaml").resolve()


def _collect_open_keypoints(
    input_device,
    retargeter,
    hand: str,
    duration: float,
    sample_rate: float,
) -> np.ndarray:
    samples = []
    interval = 1.0 / sample_rate
    start = time.time()
    next_sample = start
    last_print = 0.0
    while time.time() - start < duration:
        now = time.time()
        if now < next_sample:
            time.sleep(min(next_sample - now, 0.002))
            continue
        next_sample += interval

        fingers_data = input_device.get_fingers_data()
        raw_kp = fingers_data[f"{hand}_fingers"]
        if np.allclose(raw_kp, 0):
            continue
        samples.append(_transform_input_keypoints(np.asarray(raw_kp), retargeter, hand))

        elapsed = time.time() - start
        if elapsed - last_print >= 1.0:
            last_print = elapsed
            print(
                f"  采集中... {elapsed:.0f}/{duration:.0f}s ({len(samples)} 帧, {sample_rate:g}Hz)",
                flush=True,
            )

    if not samples:
        raise RuntimeError("未收到有效张手数据，请检查输入设备。")
    return np.asarray(samples, dtype=np.float32)


def _segment_scaling_label(optimizer) -> np.ndarray:
    seg = getattr(optimizer, "segment_scaling_full", None)
    if seg is None:
        raise ValueError("optimizer does not expose segment_scaling_full")
    arr = np.ones((5, 4), dtype=np.float32)
    seg = np.asarray(seg, dtype=np.float32)
    n = min(arr.shape[0], seg.shape[0])
    arr[:n, :seg.shape[1]] = seg[:n, :4]
    return arr


def _save_sample(args, config_path: Path, keypoints: np.ndarray) -> Path:
    retargeter = Retargeter.from_yaml(str(config_path), args.hand)
    optimizer = retargeter.optimizer
    robot_file = ROBOT_NAME_MAP.get(args.robot, args.robot)
    robot_type = retargeter.config.get("robot", {}).get("type", robot_file)
    geometry = robot_geometry_from_optimizer(optimizer)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    parts = [stamp]
    if args.subject_id:
        parts.append(args.subject_id)
    if args.take_id:
        parts.append(args.take_id)
    parts.extend([args.input, robot_type, args.hand])
    output_path = output_dir / ("_".join(parts) + ".npz")
    np.savez_compressed(
        output_path,
        keypoints_open=keypoints,
        robot_type=np.asarray(robot_type),
        input_type=np.asarray(args.input),
        hand_side=np.asarray(args.hand),
        segment_scaling=_segment_scaling_label(optimizer),
        pinch_scaling=np.asarray(float(getattr(optimizer, "pinch_scaling", 1.0)), dtype=np.float32),
        robot_segment_lengths=np.asarray(geometry.segment_lengths, dtype=np.float32),
        robot_tip_reaches=np.asarray(geometry.tip_reaches, dtype=np.float32),
        robot_root_positions=np.asarray(geometry.root_positions, dtype=np.float32),
        config_path=np.asarray(str(config_path)),
        subject_id=np.asarray(args.subject_id or ""),
        take_id=np.asarray(args.take_id or ""),
        label_source=np.asarray("current_yaml"),
        label_status=np.asarray("raw"),
        schema_version=np.asarray("calibration_sample_v1"),
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect one calibration model training sample")
    parser.add_argument("--robot", default="wuji", choices=list(ROBOT_NAME_MAP.keys()))
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--input", default="mediapipe", choices=["mediapipe", "noitom", "quest3", "avp", "pico4"])
    parser.add_argument("--video", default=None, help="Video file for mediapipe input")
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--duration", type=float, default=3.0, help="张手采集时长（秒）")
    parser.add_argument("--sample-rate", type=float, default=30.0, help="采样频率 Hz，默认 30")
    parser.add_argument("--pose-delay", type=float, default=2.0, help="开始采集前调整姿势时间（秒）")
    parser.add_argument("--output-dir", default="model/data/raw")
    parser.add_argument("--config-root", default=str(CONFIG_ROOT), help="Model config root, default: model/config")
    parser.add_argument("--subject-id", default="", help="Anonymous subject id, e.g. subject_001")
    parser.add_argument("--take-id", default="", help="Take id, e.g. take_01")
    # Noitom
    parser.add_argument("--noitom-local-ip", default="0.0.0.0")
    parser.add_argument("--noitom-local-port", type=int, default=8000)
    parser.add_argument("--noitom-server-ip", default="192.168.5.33")
    parser.add_argument("--noitom-server-port", type=int, default=9000)
    # Quest3
    parser.add_argument("--quest3-port", type=int, default=9000)
    parser.add_argument("--quest3-protocol", default="udp", choices=["udp", "tcp"])
    # Pico4
    parser.add_argument("--pico4-mode", default="relay", choices=["relay", "direct"])
    parser.add_argument("--pico4-relay-host", default="127.0.0.1")
    parser.add_argument("--pico4-relay-port", type=int, default=63902)
    parser.add_argument("--pico4-port", type=int, default=63901)
    parser.add_argument("--pico4-broadcast-port", type=int, default=29888)
    # AVP
    parser.add_argument("--avp-ip", default="192.168.50.127")
    args = parser.parse_args()

    if args.duration <= 0 or args.pose_delay < 0:
        parser.error("--duration must be positive; --pose-delay must be >= 0")
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive")

    config_path = _resolve_config(args)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"配置: {config_path}")
    retargeter = Retargeter.from_yaml(str(config_path), args.hand)
    input_device = create_input_device(args)
    _wait_for_capture_start(
        input_device,
        "请自然伸直并张开所有手指。按 Enter/空格/s 后开始倒计时采集训练样本...",
    )
    _pose_countdown(input_device, args.pose_delay)
    keypoints = _collect_open_keypoints(
        input_device, retargeter, args.hand, args.duration, args.sample_rate
    )
    output_path = _save_sample(args, config_path, keypoints)
    print(f"已保存训练样本: {output_path}")
    print(f"  keypoints_open: {keypoints.shape}")


if __name__ == "__main__":
    main()
