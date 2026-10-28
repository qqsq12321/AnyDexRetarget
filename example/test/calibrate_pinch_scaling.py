"""Estimate retarget.pinch_scaling from an open-hand index reach capture.

pinch_scaling is a uniform wrist->tip scale used by the active pinch pair in
AdaptiveOptimizerAnalytical.  This script keeps the estimate simple: ask the
user to open the hand, compare human wrist->index-tip reach with the robot's
origin->index-tip reach, and write that ratio to YAML.

Example:
    python example/test/calibrate_pinch_scaling.py \
        --robot linker_l20 --input pico4 --hand right --write
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from anydexretarget import Retargeter
from calibrate_scaling import (  # Reuse the existing input-device plumbing.
    FINGER_NAMES,
    ROBOT_NAME_MAP,
    _resolve_config_paths,
    _transform_input_keypoints,
    add_calibration_input_args,
    create_input_device,
)
from input.cli import config_dir_for


def _backup_config(config_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = config_path.with_name(f"{config_path.name}.bak-{stamp}")
    shutil.copy2(config_path, backup_path)
    print(f"  已备份: {backup_path}")
    return backup_path


def _write_pinch_scaling(config_path: Path, value: float) -> None:
    _backup_config(config_path)
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config_path) as f:
        doc = yaml.load(f)
    doc.setdefault("retarget", {})["pinch_scaling"] = round(float(value), 4)
    with open(config_path, "w") as f:
        yaml.dump(doc, f)


def _wait_for_capture_start(input_device, prompt: str) -> None:
    print(prompt)
    try:
        input("按 Enter 开始采集...")
    except EOFError:
        pass


def _pose_countdown(input_device, delay: float) -> None:
    if delay <= 0:
        return
    start = time.monotonic()
    last_second = None
    while True:
        remaining = delay - (time.monotonic() - start)
        if remaining <= 0:
            break
        second = int(np.ceil(remaining))
        if second != last_second:
            print(f"  请调整姿势，{second} 秒后开始采集...", flush=True)
            last_second = second
        input_device.get_fingers_data()
        time.sleep(0.005)
    print("  开始采集。", flush=True)


def _robot_tip_reaches(optimizer) -> np.ndarray:
    qpos = (
        optimizer.neutral_qpos.copy()
        if optimizer.neutral_qpos is not None
        else np.zeros(optimizer.robot.model.nq, dtype=np.float64)
    )
    optimizer.robot.compute_forward_kinematics(qpos)
    origin = optimizer.robot.get_link_pose(
        optimizer.robot.get_link_index(optimizer.origin_link_name)
    )[:3, 3]
    reaches = []
    for name, offset in zip(optimizer.task_link_names, optimizer.task_offsets):
        link_id = optimizer.robot.get_link_index(name)
        pose = optimizer.robot.get_link_pose(link_id)
        tip = pose[:3, 3].copy()
        if offset is not None:
            tip += pose[:3, :3] @ np.asarray(offset, dtype=np.float64)
        reaches.append(float(np.linalg.norm(tip - origin)))
    return np.asarray(reaches, dtype=np.float64)


def _collect_open_index_reach(input_device, retargeter, hand: str, duration: float) -> dict | None:
    optimizer = retargeter.optimizer
    if optimizer.num_fingers < 2:
        raise ValueError("pinch_scaling calibration needs an index finger")

    mp_index = optimizer.mp_finger_indices[1]
    index_tip_idx = optimizer.MP_TIP_INDICES[mp_index]
    wrist_idx = optimizer.MP_ORIGIN_IDX

    reaches = []
    seen = 0
    start = time.time()
    last_print = 0.0
    while time.time() - start < duration:
        fingers_data = input_device.get_fingers_data()
        raw_kp = fingers_data[f"{hand}_fingers"]
        if np.allclose(raw_kp, 0):
            continue
        kp = _transform_input_keypoints(raw_kp, retargeter, hand)
        reach = float(np.linalg.norm(kp[index_tip_idx] - kp[wrist_idx]))
        seen += 1
        if np.isfinite(reach) and reach > 1e-6:
            reaches.append(reach)

        elapsed = time.time() - start
        if elapsed - last_print >= 1.0:
            last_print = elapsed
            print(
                f"  采集中... {elapsed:.0f}/{duration:.0f}s "
                f"(有效 {len(reaches)}/{seen} 帧)",
                flush=True,
            )

    if not reaches:
        return None
    return {
        "reach": float(np.median(np.asarray(reaches, dtype=np.float64))),
        "valid_frames": len(reaches),
        "seen_frames": seen,
    }


def _resolve_config(args) -> Path:
    robot_file = ROBOT_NAME_MAP[args.robot]
    adaptive_path, _ = _resolve_config_paths(args, robot_file)
    return adaptive_path.resolve()


def _resolve_config_batch(args) -> list[Path]:
    if args.all_robots:
        config_dir = EXAMPLE_ROOT / f"config/adaptive/{config_dir_for(args.input)}"
        return sorted(config_dir.glob("*.yaml"))
    return [_resolve_config(args)]


def _config_label(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate retarget.pinch_scaling from open-hand index reach",
    )
    parser.add_argument("--all-robots", action="store_true", help="采一次张手并批量写入当前 input 的所有 adaptive YAML")
    parser.add_argument("--robot", default="wuji", choices=list(ROBOT_NAME_MAP.keys()))
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--duration", type=float, default=3.0, help="张手采集时长（秒）")
    parser.add_argument("--pose-delay", type=float, default=2.0, help="开始采集前调整姿势时间（秒）")
    parser.add_argument("--write", action="store_true", help="写入推荐 pinch_scaling 到配置文件")
    parser.add_argument("--dry-run", action="store_true", help="只显示建议值，绝不写入")
    add_calibration_input_args(parser)
    args = parser.parse_args()

    if args.write and args.dry_run:
        parser.error("--write and --dry-run cannot be used together")
    if args.duration <= 0 or args.pose_delay < 0:
        parser.error("--duration must be positive; --pose-delay must be >= 0")

    config_paths = _resolve_config_batch(args)
    if not config_paths:
        raise FileNotFoundError(f"No adaptive YAML files found for input {args.input!r}")
    missing = [p for p in config_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Config not found: {missing[0]}")

    # Use the first config only to run the shared input preprocessing during capture.
    capture_config = config_paths[0]
    print(f"采集配置: {_config_label(capture_config)}")
    retargeter = Retargeter.from_yaml(str(capture_config), args.hand)
    optimizer = retargeter.optimizer
    input_device = create_input_device(args)
    _wait_for_capture_start(
        input_device,
        "\n请自然伸直并张开所有手指；脚本只使用食指 wrist→tip 长度计算 pinch_scaling。",
    )
    _pose_countdown(input_device, args.pose_delay)
    sample = _collect_open_index_reach(input_device, retargeter, args.hand, args.duration)
    if sample is None:
        print("\n没有有效张手数据，无法推荐 pinch_scaling。")
        return

    human_index_reach = sample["reach"]
    print(
        f"\n人手食指长度: human wrist→index_tip = {human_index_reach * 100:.2f}cm "
        f"(有效 {sample['valid_frames']}/{sample['seen_frames']} 帧)"
    )

    results = []
    for config_path in config_paths:
        retargeter_i = Retargeter.from_yaml(str(config_path), args.hand)
        optimizer_i = retargeter_i.optimizer
        if optimizer_i.num_fingers < 2:
            print(f"  跳过 {_config_label(config_path)}: 没有食指")
            continue
        robot_reaches = _robot_tip_reaches(optimizer_i)
        fi_names = [FINGER_NAMES[i] for i in optimizer_i.mp_finger_indices]
        robot_index_reach = float(robot_reaches[1])
        recommended = robot_index_reach / human_index_reach
        results.append((config_path, recommended, robot_index_reach, fi_names))

    print("\n推荐结果:")
    for config_path, recommended, robot_index_reach, _ in results:
        print(
            f"  {_config_label(config_path)}: pinch_scaling={recommended:.4f} "
            f"(robot index={robot_index_reach * 100:.2f}cm)"
        )

    if args.write:
        for config_path, recommended, _, _ in results:
            _write_pinch_scaling(config_path, recommended)
        print(f"\n已写入 {len(results)} 个配置。")
    elif not args.dry_run:
        answer = input(f"\n是否写入这 {len(results)} 个配置？[y/N] ").strip().lower()
        if answer in ("y", "yes"):
            for config_path, recommended, _, _ in results:
                _write_pinch_scaling(config_path, recommended)
            print(f"已写入 {len(results)} 个配置。")
        else:
            print("未写入。")


if __name__ == "__main__":
    main()
