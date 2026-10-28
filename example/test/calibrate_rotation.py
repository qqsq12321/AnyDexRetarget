#!/usr/bin/env python3
"""Calibrate ``mediapipe_rotation`` by aligning finger roots to the robot.

``mediapipe_rotation`` is the first step of the calibration chain and the only
one that had no tool: it rotates the transformed input so the hand faces the
same way as the robot.  Everything downstream (``segment_scaling``) assumes it
is already correct, and a wrong rotation cannot be compensated by scaling.

The finger roots (MCP, plus the thumb CMC) are rigid with respect to the wrist
no matter how the fingers bend, which makes them the right landmarks to fit.
``Retargeter._apply_rotation`` rotates about the wrist without translating, so
this solves the *no-translation* scaled Procrustes problem

    minimize over rotation R and scale s:   sum_i || s * R * h_i - r_i ||^2

where ``h_i`` is a human finger root relative to the wrist and ``r_i`` is the
robot's own finger root relative to its origin link.

Residuals are reported and no value is written automatically: when the two
hands differ in shape rather than orientation, the best-fit rotation is still
a poor description of the mismatch and should not be trusted blindly.

Examples:
    python test/calibrate_rotation.py --robot wuji --input pico4
    python test/calibrate_rotation.py --robot shadow --input pico4 --hand right
    python test/calibrate_rotation.py --robot gaia --input avp --avp-ip 192.168.5.32
    python test/calibrate_rotation.py --robot wuji --input video --file data/right.mp4
    python test/calibrate_rotation.py --robot wuji --input replay --file data/avp1.pkl --trust-pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
for search_path in (PROJECT_ROOT, EXAMPLE_ROOT, TEST_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from anydexretarget import Retargeter
from anydexretarget.mediapipe import apply_mediapipe_transformations

# Reuse the device factory and capture prompts rather than duplicating them.
from calibrate_scaling import (
    CALIBRATION_INPUTS,
    FINGER_NAMES,
    ROBOT_NAME_MAP,
    _pose_countdown,
    _robust_center,
    _wait_for_capture_start,
    add_calibration_input_args,
    create_input_device,
)
from input.cli import config_dir_for, input_file

MP_MCP_INDICES = [1, 5, 9, 13, 17]
M_TO_CM = 100.0

# A recording is fitted with the config of the device that produced it,
# which is read off the file name (e.g. data/avp1.pkl -> avp).
REPLAY_SOURCES = ("mediapipe", "noitom", "quest3", "avp", "pico4")


def fit_rotation(human_roots: np.ndarray, robot_roots: np.ndarray):
    """Scaled Procrustes without translation: find R, s with s*R*h ~= r.

    Args:
        human_roots: (N, 3) finger roots relative to the human wrist.
        robot_roots: (N, 3) finger roots relative to the robot origin link.

    Returns:
        rotation: (3, 3) matrix, applied as ``points @ rotation.T``.
        scale: best-fit uniform scale (diagnostic only; not a config value).
    """
    h = np.asarray(human_roots, dtype=np.float64)
    r = np.asarray(robot_roots, dtype=np.float64)
    if h.shape != r.shape or h.ndim != 2 or h.shape[1] != 3:
        raise ValueError(f"Shape mismatch: {h.shape} vs {r.shape}")
    if len(h) < 3:
        raise ValueError(f"Need at least 3 landmarks, got {len(h)}")

    covariance = h.T @ r
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, optimal_scale(h, r, rotation)


def optimal_scale(human_roots, robot_roots, rotation) -> float:
    """Best uniform scale for a *given* rotation.

    Comparing the fitted rotation against the configured one is only fair if
    each is given its own best scale; reusing one scale for both would make
    whichever rotation it was derived from look better than it is.
    """
    h = np.asarray(human_roots, dtype=np.float64)
    r = np.asarray(robot_roots, dtype=np.float64)
    denominator = float(np.sum(h * h))
    if denominator <= 1e-12:
        return 1.0
    return float(np.sum(r * (h @ np.asarray(rotation).T)) / denominator)


def residuals_cm(human_roots, robot_roots, rotation, scale):
    """Per-landmark distance left over after the fitted rotation and scale."""
    aligned = (np.asarray(human_roots) @ rotation.T) * scale
    return np.linalg.norm(aligned - np.asarray(robot_roots), axis=1) * M_TO_CM


def get_robot_finger_roots(optimizer):
    """Robot finger roots at the neutral pose, relative to the origin link."""
    robot = optimizer.robot
    qpos = (
        optimizer.neutral_qpos.copy()
        if optimizer.neutral_qpos is not None
        else np.zeros(robot.model.nq, dtype=np.float64)
    )
    robot.compute_forward_kinematics(qpos)
    origin = robot.get_link_pose(robot.get_link_index(optimizer.origin_link_name))[:3, 3]
    roots = []
    for name in optimizer.link1_names:
        roots.append(robot.get_link_pose(robot.get_link_index(name))[:3, 3] - origin)
    return np.asarray(roots, dtype=np.float64)


def collect_human_roots(input_device, hand, duration):
    """Median finger-root positions in the wrist frame, before any rotation."""
    samples = []
    start = time.time()
    last_print = 0.0
    while time.time() - start < duration:
        fingers_data = input_device.get_fingers_data()
        raw_kp = fingers_data[f"{hand}_fingers"]
        if np.allclose(raw_kp, 0):
            continue
        kp = apply_mediapipe_transformations(np.asarray(raw_kp), hand)
        samples.append(kp[MP_MCP_INDICES])

        elapsed = time.time() - start
        if elapsed - last_print >= 1.0:
            last_print = elapsed
            print(f"  采集中... {elapsed:.0f}/{duration:.0f}s  ({len(samples)} 帧)", flush=True)

    if not samples:
        return None, 0
    return _robust_center(np.asarray(samples)), len(samples)


def collect_human_roots_from_pkl(pkl_path: Path, hand: str, trust_pkl: bool, limit=None):
    """Collect transformed finger roots from a recorded hand-keypoint pickle."""
    if not trust_pkl:
        raise ValueError(
            "Refusing to load pickle without explicit trust. "
            "Use --trust-pkl only for files you fully trust."
        )
    with pkl_path.open("rb") as handle:
        frames = pickle.load(handle)
    if limit is not None:
        frames = list(frames)[:limit]

    samples = []
    for frame in frames:
        raw_kp = frame.get(f"{hand}_fingers") if isinstance(frame, dict) else None
        if raw_kp is None:
            continue
        raw_kp = np.asarray(raw_kp)
        if raw_kp.shape != (21, 3) or np.allclose(raw_kp, 0):
            continue
        samples.append(apply_mediapipe_transformations(raw_kp, hand)[MP_MCP_INDICES])

    if not samples:
        return None, 0
    return _robust_center(np.asarray(samples)), len(samples)


def gradient_warning(residual_vectors, finger_labels):
    """Flag a monotonic residual pattern, which rotation alone cannot remove.

    A systematic ramp across the four non-thumb fingers means the two hands
    disagree about how the roots are spread, not about orientation.  Reporting
    it stops the user from chasing the rotation values forever.
    """
    if len(residual_vectors) < 4:
        return None
    for axis, axis_name in enumerate("xyz"):
        column = residual_vectors[:, axis]
        deltas = np.diff(column)
        if not (np.all(deltas > 0) or np.all(deltas < 0)):
            continue
        spread = float(column.max() - column.min())
        if spread > 1.0:
            return (
                f"{axis_name} 方向残差呈单调梯度（{' → '.join(f'{v:+.2f}' for v in column)}，"
                f"跨度 {spread:.2f}cm）。这是两只手指根排布比例不同造成的，"
                f"旋转补不了 —— 继续调角度不会改善。"
            )
    return None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Calibrate mediapipe_rotation from finger-root alignment"
    )
    parser.add_argument("--robot", default="wuji", choices=sorted(ROBOT_NAME_MAP))
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--frames", type=int, default=None, help="Use only the first N replay frames")
    parser.add_argument("--trust-pkl", action="store_true", help="Confirm the replay pickle is trusted")
    parser.add_argument("--duration", type=float, default=5.0, help="Capture seconds")
    parser.add_argument("--pose-delay", type=float, default=2.0, help="Seconds to settle before capture")
    parser.add_argument(
        "--include-thumb", action="store_true",
        help="Include the thumb CMC in the fit (off by default: the human CMC "
             "and the robot thumb mount often differ by tens of degrees and "
             "drag the whole estimate off)",
    )
    add_calibration_input_args(parser, default="pico4", choices=(*CALIBRATION_INPUTS, "replay"))
    return parser


def main():
    args = build_parser().parse_args()

    robot_file = ROBOT_NAME_MAP.get(args.robot, args.robot)
    replay_path = None
    if args.input == "replay":
        replay_path = Path(input_file(args))
        if not replay_path.is_absolute():
            replay_path = EXAMPLE_ROOT / replay_path
        replay_path = replay_path.resolve()
        stem = replay_path.stem.lower()
        config_dir = next((name for name in REPLAY_SOURCES if name in stem), None)
        if config_dir is None:
            raise SystemExit(
                f"无法从 pkl 文件名推断原始输入源；文件名需包含 {'/'.join(REPLAY_SOURCES)} 之一"
            )
    else:
        config_dir = config_dir_for(args.input)

    config_file = EXAMPLE_ROOT / f"config/adaptive/{config_dir}/{config_dir}_{robot_file}.yaml"

    print(f"配置文件: {config_file}")
    input_label = f"{replay_path}（原始输入源: {config_dir}）" if replay_path else args.input
    print(f"机器人:   {robot_file}    手: {args.hand}    输入: {input_label}")

    retargeter = Retargeter.from_yaml(str(config_file), args.hand)
    optimizer = retargeter.optimizer
    robot_roots = get_robot_finger_roots(optimizer)

    # Line up the robot's fingers with their MediaPipe counterparts.
    mp_fingers = list(optimizer.mp_finger_indices)[: len(robot_roots)]
    keep = [i for i, fi in enumerate(mp_fingers) if args.include_thumb or fi != 0]
    if len(keep) < 3:
        raise SystemExit("可用地标少于 3 个，无法拟合旋转")
    labels = [FINGER_NAMES[mp_fingers[i]] for i in keep]

    if replay_path is not None:
        human_all, frames = collect_human_roots_from_pkl(
            replay_path, args.hand, args.trust_pkl, args.frames
        )
    else:
        device = create_input_device(args)
        _wait_for_capture_start(
            device,
            "请自然伸直并张开五指，掌心朝向与平时遥操作时一致。"
            f"按 Enter/空格/s 后有 {args.pose_delay:g} 秒调整，再采集 {args.duration:.0f} 秒...",
        )
        _pose_countdown(device, args.pose_delay)
        human_all, frames = collect_human_roots(device, args.hand, args.duration)

    if human_all is None:
        raise SystemExit("未收到有效数据，请检查输入设备。")
    print(f"采集完成，共 {frames} 帧")

    human = np.asarray([human_all[mp_fingers[i]] for i in keep], dtype=np.float64)
    robot = robot_roots[keep]

    rotation, scale = fit_rotation(human, robot)
    euler = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    res = residuals_cm(human, robot, rotation, scale)
    res_vectors = (human @ rotation.T) * scale - robot

    current = retargeter.rotation_xyz or {}
    current_matrix = Rotation.from_euler(
        "xyz",
        [current.get("x", 0.0), current.get("y", 0.0), current.get("z", 0.0)],
        degrees=True,
    ).as_matrix()
    current_res = residuals_cm(
        human, robot, current_matrix, optimal_scale(human, robot, current_matrix)
    )

    print(f"\n{'=' * 68}")
    print(f"拟合结果（{'含拇指' if args.include_thumb else '不含拇指'}，{len(keep)} 个指根）")
    print(f"{'=' * 68}")
    print(
        f"  当前配置: x={current.get('x', 0.0):7.2f}  "
        f"y={current.get('y', 0.0):7.2f}  z={current.get('z', 0.0):7.2f}"
        f"     残差 RMS {np.sqrt(np.mean(current_res ** 2)):.2f}cm"
    )
    print(
        f"  拟合最优: x={euler[0]:7.2f}  y={euler[1]:7.2f}  z={euler[2]:7.2f}"
        f"     残差 RMS {np.sqrt(np.mean(res ** 2)):.2f}cm"
    )
    print(f"  （拟合尺度 {scale:.3f}，仅作诊断，不写入配置）")

    print(f"\n  逐指残差（施加拟合旋转后，cm）:")
    for label, vec, dist in zip(labels, res_vectors * M_TO_CM, res):
        print(
            f"    {label:8s} [{vec[0]:+6.2f}, {vec[1]:+6.2f}, {vec[2]:+6.2f}]   |{dist:5.2f}|"
        )

    rms = float(np.sqrt(np.mean(res ** 2)))
    print()
    if rms < 0.5:
        print("  ✅ 残差小，拟合可信，建议直接采用。")
    elif rms < 1.5:
        print("  ⚠️  残差中等。建议采用后用 debug_skeleton.py 目视确认。")
    else:
        print(
            "  ❌ 残差偏大：两只手的指根排布不只是朝向不同，还有形状差异。\n"
            "     拟合值只能当起点，务必目视校核。"
        )

    warning = gradient_warning(res_vectors * M_TO_CM, labels)
    if warning:
        print(f"\n  诊断: {warning}")

    improvement = float(np.sqrt(np.mean(current_res ** 2))) - rms
    if improvement > 0.05:
        print(f"\n  相比当前配置，残差 RMS 改善 {improvement:.2f}cm")
    elif improvement < -0.05:
        print(f"\n  当前配置反而更好（拟合差 {-improvement:.2f}cm），建议保持不变。")
    else:
        print("\n  与当前配置基本等效，无需修改。")

    print(f"\n{'=' * 68}")
    print("将以下内容填入配置文件的 retarget: 段（本工具不会自动写入）:")
    print(f"{'=' * 68}")
    print("  mediapipe_rotation:")
    print(f"    x: {euler[0]:.1f}")
    print(f"    y: {euler[1]:.1f}")
    print(f"    z: {euler[2]:.1f}")
    print(f"{'=' * 68}")
    print("\n改完后跑 debug_skeleton.py 目视确认蓝色骨架朝向是否与机械手一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
