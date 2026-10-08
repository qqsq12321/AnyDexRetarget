"""Calibrate segment_scaling for any robot hand and input source.

Uses robust open-hand morphology calibration: robot link lengths are divided
by median human segment lengths from a natural open-hand capture.

Outputs recommended segment_scaling values and optionally writes them back to
both Adaptive and Vector config YAML files.

Supported inputs: realsense, video (both use mediapipe configs), noitom, quest3, avp, pico4

Usage:
    python test/calibrate_scaling.py --robot sharpa --input video --file data/right.mp4
    python test/calibrate_scaling.py --robot wuji --input realsense
    python test/calibrate_scaling.py --robot wuji --input noitom
    python test/calibrate_scaling.py --robot wuji --input quest3
    python test/calibrate_scaling.py --robot wuji --input avp
    python test/calibrate_scaling.py --robot wuji --input realsense --optimizer both --write
"""

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from anydexretarget import Retargeter
from anydexretarget.mediapipe import apply_mediapipe_transformations
from anydexretarget.optimizer.utils import CM_TO_M
from input.cli import add_input_args, config_dir_for, open_input

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# MediaPipe landmark indices, mirrored from BaseOptimizer.
MP_MCP_INDICES = [1, 5, 9, 13, 17]
MP_PIP_INDICES = [2, 6, 10, 14, 18]
MP_DIP_INDICES = [3, 7, 11, 15, 19]
MP_TIP_INDICES = [4, 8, 12, 16, 20]

ROBOT_NAME_MAP = {
    "shadow": "shadow_hand", "wuji": "wuji_hand", "allegro": "allegro_hand",
    "leap": "leap_hand", "inspire": "inspire_hand", "ability": "ability_hand",
    "svh": "svh_hand", "rohand": "rohand", "linkerhand_l21": "linkerhand_l21",
    "linker_l20": "linker_l20", "unitree_dex5": "unitree_dex5_hand", "sharpa": "sharpa_hand",
    "gaia": "gaia_hand20",
}

# Live sources only: calibration captures a held pose, not a recording.
CALIBRATION_INPUTS = ("realsense", "video", "avp", "quest3", "pico4", "noitom")

# MediaPipe task_kp → (finger, level): level 0=PIP, 1=DIP, 2=TIP
_KP_TO_FINGER_LEVEL = {
    2: ("thumb",  0), 3: ("thumb",  1), 4: ("thumb",  2),
    6: ("index",  0), 7: ("index",  1), 8: ("index",  2),
    10: ("middle", 0), 11: ("middle", 1), 12: ("middle", 2),
    14: ("ring",   0), 15: ("ring",   1), 16: ("ring",   2),
    18: ("pinky",  0), 19: ("pinky",  1), 20: ("pinky",  2),
}


def _resolve_config_paths(args, robot_file):
    """Return (adaptive_path, vector_path) as Path objects (may not exist)."""
    config_dir = config_dir_for(args.input)
    adaptive = EXAMPLE_ROOT / f"config/adaptive/{config_dir}/{config_dir}_{robot_file}.yaml"
    vector   = EXAMPLE_ROOT / f"config/vector/{config_dir}/{config_dir}_{robot_file}.yaml"
    return adaptive, vector


def _backup_config(config_path: Path):
    """Create a timestamped backup before changing a tuned configuration."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = config_path.with_name(f"{config_path.name}.bak-{stamp}")
    shutil.copy2(config_path, backup_path)
    print(f"  已备份: {backup_path}")
    return backup_path


def _write_adaptive(config_path: Path, result: dict, backup: bool = True):
    """Update retarget.segment_scaling in an adaptive config YAML."""
    if backup:
        _backup_config(config_path)
    ryaml = YAML()
    ryaml.preserve_quotes = True
    with open(config_path) as f:
        doc = ryaml.load(f)

    seg = doc["retarget"]["segment_scaling"]
    for fname, vals in result.items():
        if fname not in seg:
            continue
        # Calibration now produces the full [MCP, PIP, DIP, TIP] set because the
        # wrist->MCP span is scaled like any other.  Legacy three-value entries
        # are grown in place so the comment layout of the file survives.
        dst = seg[fname]
        while len(dst) < len(vals):
            dst.append(1.0)
        for i, v in enumerate(vals):
            dst[i] = v

    with open(config_path, "w") as f:
        ryaml.dump(doc, f)


def _write_vector(config_path: Path, result: dict, backup: bool = True):
    """Update key_vectors[*].scale in a vector config YAML."""
    if backup:
        _backup_config(config_path)
    ryaml = YAML()
    ryaml.preserve_quotes = True
    with open(config_path) as f:
        doc = ryaml.load(f)

    kv_list = doc["retarget"]["key_vectors"]
    for entry in kv_list:
        task_kp = int(entry.get("task_kp", -1))
        mapping = _KP_TO_FINGER_LEVEL.get(task_kp)
        if mapping is None:
            continue
        fname, level = mapping
        if fname in result:
            entry["scale"] = float(result[fname][level])

    with open(config_path, "w") as f:
        ryaml.dump(doc, f)


def write_configs(args, robot_file, result, cumulative_result, explicit_config_path=None):
    """Write calibration result to adaptive and/or vector config files.

    The two optimizers scale different things, so they take different numbers.
    ``result`` holds per-segment phalanx ratios for adaptive ``segment_scaling``;
    ``cumulative_result`` holds origin-to-joint ratios for the vector
    optimizer, whose ``key_vectors`` entries use ``origin_kp: 0`` and therefore
    still scale wrist-anchored vectors.

    Adaptive/vector paths are resolved automatically from --robot and --input,
    then written according to --optimizer.
    """
    optimizer_mode = getattr(args, "optimizer", "both")

    adaptive_path, vector_path = _resolve_config_paths(args, robot_file)
    targets = []
    if optimizer_mode in ("adaptive", "both") and adaptive_path.exists():
        targets.append(("adaptive", adaptive_path))
    if optimizer_mode in ("vector", "both") and vector_path.exists():
        targets.append(("vector", vector_path))

    if not targets:
        print("  未找到对应配置文件，跳过写入。")
        return

    for opt_type, path in targets:
        try:
            if opt_type == "adaptive":
                _write_adaptive(path, result, backup=True)
            else:
                _write_vector(path, cumulative_result, backup=True)
            print(f"  已写入 ({opt_type}): {path}")
        except Exception as e:
            print(f"  写入失败 ({path}): {e}")


def _write_single(config_path: Path, result: dict, cumulative_result: dict, args):
    """Write to a single explicitly-specified config file, auto-detecting type."""
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        opt_type_str = raw.get("optimizer", {}).get("type", "")
        if "KeyVector" in opt_type_str:
            _write_vector(config_path, cumulative_result, backup=True)
            print(f"  已写入 (vector): {config_path}")
        else:
            _write_adaptive(config_path, result, backup=True)
            print(f"  已写入 (adaptive): {config_path}")
    except Exception as e:
        print(f"  写入失败 ({config_path}): {e}")


def _get_link_pos(robot, link_name, offset=None):
    lid = robot.get_link_index(link_name)
    pose = robot.get_link_pose(lid)
    pos = pose[:3, 3].copy()
    if offset is not None:
        pos += pose[:3, :3] @ np.asarray(offset, dtype=np.float64)
    return pos


def get_robot_distances(optimizer):
    """Compute robot FK distances at neutral pose.

    Returns:
        cumulative: (pip_dists, dip_dists, tip_dists) — origin→link3/link4/tip
        segment: (seg_pip, seg_dip, seg_tip) — link1→link3, link3→link4, link4→tip

    The per-segment values are what ``segment_scaling`` calibrates against:
    FullHandVec grows each chain from the robot's own finger root, so the
    first segment is measured from link1 rather than from the origin.
    """
    robot = optimizer.robot
    nf = optimizer.num_fingers

    qpos = optimizer.neutral_qpos.copy() if optimizer.neutral_qpos is not None \
        else np.zeros(robot.model.nq)
    robot.compute_forward_kinematics(qpos)

    origin_pos = _get_link_pos(robot, optimizer.origin_link_name)

    def pos_for(link_names, offsets):
        result = []
        for fi in range(nf):
            if fi >= len(link_names):
                result.append(None)
                continue
            off = offsets[fi] if offsets is not None else None
            result.append(_get_link_pos(robot, link_names[fi], off))
        return result

    root_pos = pos_for(getattr(optimizer, 'link1_names', []), None)
    pip_pos = pos_for(
        getattr(optimizer, 'link3_names', []),
        getattr(optimizer, 'link3_offsets', None),
    )
    dip_pos = pos_for(
        getattr(optimizer, 'link4_names', []),
        getattr(optimizer, 'link4_offsets', None),
    )
    tip_pos = pos_for(
        optimizer.task_link_names,
        getattr(optimizer, 'task_offsets', None),
    )

    # Cumulative: origin → each link
    pip_dists = [float(np.linalg.norm(p - origin_pos)) if p is not None else None for p in pip_pos]
    dip_dists = [float(np.linalg.norm(p - origin_pos)) if p is not None else None for p in dip_pos]
    tip_dists = [float(np.linalg.norm(p - origin_pos)) if p is not None else None for p in tip_pos]

    # Per-segment: link1→link3, link3→link4, link4→tip
    seg_pip = []
    seg_dip = []
    seg_tip = []
    for fi in range(nf):
        if fi < len(root_pos) and root_pos[fi] is not None and pip_pos[fi] is not None:
            seg_pip.append(float(np.linalg.norm(pip_pos[fi] - root_pos[fi])))
        else:
            seg_pip.append(None)
        if pip_pos[fi] is not None and dip_pos[fi] is not None:
            seg_dip.append(float(np.linalg.norm(dip_pos[fi] - pip_pos[fi])))
        else:
            seg_dip.append(None)
        if dip_pos[fi] is not None and tip_pos[fi] is not None:
            seg_tip.append(float(np.linalg.norm(tip_pos[fi] - dip_pos[fi])))
        else:
            seg_tip.append(None)

    return (pip_dists, dip_dists, tip_dists), (seg_pip, seg_dip, seg_tip)


def _transform_input_keypoints(raw_kp, retargeter, hand):
    """Return the exact transformed keypoints passed into optimizer.solve()."""
    kp = apply_mediapipe_transformations(raw_kp, hand)
    if retargeter.rotation_xyz:
        kp = retargeter._apply_rotation(kp)
    return np.asarray(kp, dtype=np.float64)


def collect_median_keypoints(input_device, hand, duration, label="采集中"):
    """Median of the transformed keypoints over a capture, before rotation.

    Batch calibration reuses one capture across every robot, and each robot
    applies its own ``mediapipe_rotation``, so the rotation is deliberately
    left off here and applied per robot later.

    Returns:
        (median_keypoints, accepted_frames, seen_frames)
    """
    samples = []
    seen = 0
    start = time.time()
    last_print = 0.0
    while time.time() - start < duration:
        fingers_data = input_device.get_fingers_data()
        raw_kp = fingers_data[f"{hand}_fingers"]
        if np.allclose(raw_kp, 0):
            continue
        kp = apply_mediapipe_transformations(np.asarray(raw_kp), hand)
        seen += 1
        samples.append(kp)

        elapsed = time.time() - start
        if elapsed - last_print >= 1.0:
            last_print = elapsed
            print(
                f"  {label}... {elapsed:.0f}/{duration:.0f}s  "
                f"(有效 {len(samples)}/{seen} 帧)",
                flush=True,
            )

    if not samples:
        return None, 0, seen
    return _robust_center(np.asarray(samples)), len(samples), seen


DEGENERATE_SEGMENT_M = 5e-4  # 0.5 mm: below this a robot link has no length


def palm_scale_for_finger(robot_root_len, human_palm_len):
    """Scale that stretches the human wrist->MCP span to the robot's.

    FullHandVec scales this span like any other, so the palm-length mismatch
    is absorbed here instead of by the phalanx factors.
    """
    if human_palm_len > 1e-4 and robot_root_len > 1e-6:
        return round(robot_root_len / human_palm_len, 3)
    return 1.0


def segment_scales_for_finger(robot_segments, human_segments):
    """Three per-segment scales, merging phalanges when a robot link is absent.

    Inspire, Ability and ROHand only have two moving phalanges, so their
    ``link1_names`` and ``link3_names`` name the same link and the first robot
    segment has zero length.  Scaling the human proximal phalanx by zero would
    throw it away.  Because the chain accumulates,

        root + prox * s + mid * s  ==  root + (DIP - MCP) * s

    using one shared scale for the first two segments maps the human's whole
    proximal-plus-middle span onto the robot's single first link instead.

    Args:
        robot_segments: (root->link3, link3->link4, link4->tip) lengths in m.
        human_segments: (MCP->PIP, PIP->DIP, DIP->TIP) lengths in m.

    Returns:
        (scales, merged) where ``merged`` flags the two-phalanx fallback.
    """
    r0, r1, r2 = (float(v or 0.0) for v in robot_segments)
    h0, h1, h2 = (float(v or 0.0) for v in human_segments)

    def safe(robot_len, human_len, default=1.0):
        return round(robot_len / human_len, 3) if human_len > 1e-4 else default

    if r0 < DEGENERATE_SEGMENT_M:
        shared = safe(r1, h0 + h1)
        return [shared, shared, safe(r2, h2)], True
    return [safe(r0, h0), safe(r1, h1), safe(r2, h2)], False


def _phalanx_segments(kp, mp_finger_index):
    """Three phalanx vectors (MCP->PIP, PIP->DIP, DIP->TIP) of one finger."""
    mcp = kp[MP_MCP_INDICES[mp_finger_index]]
    pip = kp[MP_PIP_INDICES[mp_finger_index]]
    dip = kp[MP_DIP_INDICES[mp_finger_index]]
    tip = kp[MP_TIP_INDICES[mp_finger_index]]
    return np.stack([pip - mcp, dip - pip, tip - dip])


def _robust_center(values):
    """Robust scalar/vector center used to suppress tracking spikes."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return np.median(arr, axis=0)


def _wait_for_capture_start(input_device, message):
    """Preview input until Enter/space/s is pressed."""
    print(message)
    import select
    import sys as _sys
    while True:
        input_device.get_fingers_data()
        ready, _, _ = select.select([_sys.stdin], [], [], 0.0)
        if ready:
            _sys.stdin.readline()
            return
        try:
            import cv2
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), 13, ord(" ")):
                return
        except Exception:
            pass


def _pose_countdown(input_device, delay):
    """Keep refreshing the input and give the user time to form the pose."""
    delay = float(delay)
    if delay <= 0:
        return
    start = time.monotonic()
    last_second = None
    while True:
        elapsed = time.monotonic() - start
        remaining = delay - elapsed
        if remaining <= 0:
            break
        second = int(np.ceil(remaining))
        if second != last_second:
            print(f"  请调整姿势，{second} 秒后开始采集...", flush=True)
            last_second = second
        input_device.get_fingers_data()
        time.sleep(0.005)
    print("  开始采集。", flush=True)


def collect_human_distances(input_device, retargeter, hand, duration):
    """Collect input data for `duration` seconds and compute robust median distances.

    Returns:
        frames: number of valid frames
        cumulative: (pip_median, dip_median, tip_median) — wrist→PIP/DIP/TIP
        segment: (seg_pip, seg_dip, seg_tip) — MCP→PIP, PIP→DIP, DIP→TIP
        palm: wrist→MCP median per finger, the span segment_scaling[0] scales
    """
    optimizer = retargeter.optimizer
    nf = optimizer.num_fingers
    mcp_idx = [optimizer.MP_MCP_INDICES[i] for i in optimizer.mp_finger_indices]
    pip_idx = [optimizer.MP_PIP_INDICES[i] for i in optimizer.mp_finger_indices]
    dip_idx = [optimizer.MP_DIP_INDICES[i] for i in optimizer.mp_finger_indices]
    tip_idx = [optimizer.MP_TIP_INDICES[i] for i in optimizer.mp_finger_indices]

    pip_buf = [[] for _ in range(nf)]
    dip_buf = [[] for _ in range(nf)]
    tip_buf = [[] for _ in range(nf)]
    seg_pip_buf = [[] for _ in range(nf)]
    seg_dip_buf = [[] for _ in range(nf)]
    seg_tip_buf = [[] for _ in range(nf)]
    palm_buf = [[] for _ in range(nf)]

    start = time.time()
    frames = 0
    last_print = 0
    while time.time() - start < duration:
        fingers_data = input_device.get_fingers_data()
        raw_kp = fingers_data[f"{hand}_fingers"]
        if np.allclose(raw_kp, 0):
            continue

        kp = _transform_input_keypoints(raw_kp, retargeter, hand)

        wrist = kp[0]
        for i in range(nf):
            p_mcp = kp[mcp_idx[i]]
            p_pip = kp[pip_idx[i]]
            p_dip = kp[dip_idx[i]]
            p_tip = kp[tip_idx[i]]
            # Cumulative (diagnostics only)
            pip_buf[i].append(float(np.linalg.norm(p_pip - wrist)))
            dip_buf[i].append(float(np.linalg.norm(p_dip - wrist)))
            tip_buf[i].append(float(np.linalg.norm(p_tip - wrist)))
            # Per-segment lengths — these drive segment_scaling
            palm_buf[i].append(float(np.linalg.norm(p_mcp - wrist)))
            seg_pip_buf[i].append(float(np.linalg.norm(p_pip - p_mcp)))
            seg_dip_buf[i].append(float(np.linalg.norm(p_dip - p_pip)))
            seg_tip_buf[i].append(float(np.linalg.norm(p_tip - p_dip)))
        frames += 1

        elapsed = time.time() - start
        if elapsed - last_print >= 1.0:
            last_print = elapsed
            print(f"  采集中... {elapsed:.0f}/{duration:.0f}s  ({frames} 帧)", flush=True)

    if frames == 0:
        return frames, None, None, None

    def robust_list(bufs):
        return [float(_robust_center(b)) if b else 0.0 for b in bufs]

    cumulative = (robust_list(pip_buf), robust_list(dip_buf), robust_list(tip_buf))
    segment = (robust_list(seg_pip_buf), robust_list(seg_dip_buf), robust_list(seg_tip_buf))
    return frames, cumulative, segment, robust_list(palm_buf)


def calibrate_one_robot(args, robot_file, open_kp):
    """Derive segment_scaling for a single robot from shared human captures.

    ``open_kp`` holds un-rotated keypoints so every robot can apply its own
    ``mediapipe_rotation`` before measuring.

    Returns:
        (result, cumulative_result, fi_names, merged_fingers) or None when the
        robot has no config for this input source.
    """
    adaptive_path, _ = _resolve_config_paths(args, robot_file)
    if not adaptive_path.exists():
        return None

    retargeter = Retargeter.from_yaml(str(adaptive_path), args.hand)
    optimizer = retargeter.optimizer
    mp_fingers = list(optimizer.mp_finger_indices)
    fi_names = [FINGER_NAMES[i] for i in mp_fingers]

    def rotated(kp):
        return retargeter._apply_rotation(kp) if retargeter.rotation_xyz else kp

    kp_open = rotated(open_kp)
    (pip_robot, dip_robot, tip_robot), (rseg_pip, rseg_dip, rseg_tip) = \
        get_robot_distances(optimizer)

    def ratio(robot_d, human_d):
        if robot_d and human_d and human_d > 1e-4:
            return round(robot_d / human_d, 3)
        return 1.0

    wrist = kp_open[0]
    result = {}
    cumulative_result = {}
    merged_fingers = []
    robot_roots_m = optimizer.finger_root_vectors * CM_TO_M
    for i, fi in enumerate(mp_fingers):
        segs = np.linalg.norm(_phalanx_segments(kp_open, fi), axis=1)
        scales, merged = segment_scales_for_finger(
            (rseg_pip[i], rseg_dip[i], rseg_tip[i]), segs
        )
        palm = palm_scale_for_finger(
            float(np.linalg.norm(robot_roots_m[i])),
            float(np.linalg.norm(kp_open[MP_MCP_INDICES[fi]] - wrist)),
        )
        result[fi_names[i]] = [palm] + scales
        if merged:
            merged_fingers.append(fi_names[i])
        cumulative_result[fi_names[i]] = [
            ratio(pip_robot[i], float(np.linalg.norm(kp_open[MP_PIP_INDICES[fi]] - wrist))),
            ratio(dip_robot[i], float(np.linalg.norm(kp_open[MP_DIP_INDICES[fi]] - wrist))),
            ratio(tip_robot[i], float(np.linalg.norm(kp_open[MP_TIP_INDICES[fi]] - wrist))),
        ]

    return result, cumulative_result, fi_names, merged_fingers


def run_batch_calibration(args):
    """Capture the human hand once and calibrate every configured robot."""
    device = create_input_device(args)

    _wait_for_capture_start(
        device,
        "请自然伸直并张开所有手指。不要握拳；按 Enter/空格/s 后有 "
        f"{args.pose_delay:g} 秒调整姿势，再采集 {args.duration:.0f} 秒...",
    )
    _pose_countdown(device, args.pose_delay)
    open_kp, frames, _ = collect_median_keypoints(device, args.hand, args.duration)
    if open_kp is None:
        print("未收到有效数据，请检查输入设备。")
        return
    print(f"张手采集完成，共 {frames} 帧")

    print(f"\n{'='*68}")
    print(f"逐机器人标定（{args.input} 输入）")
    print(f"{'='*68}")

    written, skipped = [], []
    for short_name, robot_file in ROBOT_NAME_MAP.items():
        outcome = calibrate_one_robot(args, robot_file, open_kp)
        if outcome is None:
            skipped.append(short_name)
            continue
        result, cumulative_result, fi_names, merged_fingers = outcome

        print(f"\n--- {short_name} ({robot_file})")
        print(f"    {'手指':8s}  {'腕→MCP':>8s}  {'MCP→PIP':>8s}  {'PIP→DIP':>8s}  {'DIP→TIP':>8s}")
        for name in fi_names:
            v = result[name]
            print(f"    {name:8s}  " + "  ".join(f"{x:>8.3f}" for x in v))
        if merged_fingers:
            print(f"      注意: {', '.join(merged_fingers)} 只有两节指骨，"
                  f"已把人手近节+中节合并映射到机械手第一节（前两个系数相同）")

        if args.dry_run:
            continue
        if args.write:
            write_configs(args, robot_file, result, cumulative_result)
            written.append(short_name)

    print(f"\n{'='*68}")
    if args.dry_run:
        print("dry-run：仅显示建议值，未写入任何配置。")
    elif args.write:
        print(f"已写入 {len(written)} 个机器人: {', '.join(written)}")
    else:
        print("未指定 --write，未写入任何配置。加 --write 可批量落盘。")
    if skipped:
        print(f"跳过（该输入源无配置）: {', '.join(skipped)}")


def add_calibration_input_args(parser, default="realsense", choices=CALIBRATION_INPUTS):
    """Register the shared input options with the calibration defaults."""
    add_input_args(parser, default=default, choices=choices)
    # Calibration listens on every interface.
    parser.set_defaults(noitom_local_ip="0.0.0.0")


def create_input_device(args):
    """Create the calibration input device selected by --input."""
    if args.input == "realsense":
        # The pose has to be checked on screen, so RealSense always previews.
        args = argparse.Namespace(**{**vars(args), "show_video": True})
    return open_input(args, args.hand)


def fmt_cm(v):
    return f"{v*100:.2f}cm" if v else "   N/A"


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate segment_scaling for any robot hand and input source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--robot", default="wuji",
                        choices=list(ROBOT_NAME_MAP.keys()),
                        help="Robot hand type (default: wuji)")
    parser.add_argument("--all-robots", action="store_true",
                        help="Calibrate every robot for this input source from a "
                             "single hand capture. segment_scaling divides robot "
                             "link lengths by human phalanx lengths, and the human "
                             "side is rotation-invariant, so one capture serves all.")
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--duration", type=float, default=5.0,
                        help="采集时长（秒），默认 5")
    parser.add_argument("--pose-delay", type=float, default=2.0,
                        help="按 Enter/空格/s 后留给调整姿势的时间（秒），默认 2")
    # Write-back options
    parser.add_argument("--optimizer", default="both",
                        choices=["adaptive", "vector", "both"],
                        help="写入哪种配置文件（默认: both）")
    parser.add_argument("--write", action="store_true",
                        help="标定后直接写入配置文件，不询问")
    parser.add_argument("--dry-run", action="store_true",
                        help="只显示建议值，绝不写入配置文件")
    add_calibration_input_args(parser)
    args = parser.parse_args()
    if args.write and args.dry_run:
        parser.error("--write and --dry-run cannot be used together")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.pose_delay < 0:
        parser.error("--pose-delay must be >= 0")
    if args.all_robots:
        print(f"批量标定 | 输入源: {args.input} | 手: {args.hand}")
        run_batch_calibration(args)
        return

    # Resolve config path
    robot_file = ROBOT_NAME_MAP.get(args.robot, args.robot)
    config_dir = config_dir_for(args.input)
    config_path = f"config/adaptive/{config_dir}/{config_dir}_{robot_file}.yaml"
    config_file = EXAMPLE_ROOT / config_path

    print(f"配置文件: {config_file}")
    print(f"输入源:   {args.input}")
    print(f"机器人:   {robot_file}")
    if args.input == "pico4":
        if args.pico4_mode == "direct":
            print(f"Pico4模式: direct (tcp={args.pico4_port}, udp_broadcast={args.pico4_broadcast_port})")
        else:
            print(f"Pico4模式: relay ({args.pico4_relay_host}:{args.pico4_relay_port})")

    retargeter = Retargeter.from_yaml(str(config_file), args.hand)
    optimizer = retargeter.optimizer
    nf = optimizer.num_fingers
    fi_names = [FINGER_NAMES[i] for i in optimizer.mp_finger_indices]
    # ── Robot FK distances ───────────────────────────────────────────────────
    print("\n正在计算机器人 FK 距离（中性位姿）...")
    (pip_robot, dip_robot, tip_robot), (rseg_pip, rseg_dip, rseg_tip) = \
        get_robot_distances(optimizer)

    print(f"\n{'='*68}")
    print(f"机器人 FK 距离（中性位姿）")
    print(f"{'='*68}")
    print(f"  累计距离（origin→各关节）:")
    print(f"  {'手指':8s}  {'→link3':>8s}  {'→link4':>8s}  {'→tip':>8s}")
    for i in range(nf):
        print(f"  {fi_names[i]:8s}  {fmt_cm(pip_robot[i]):>8s}  "
              f"{fmt_cm(dip_robot[i]):>8s}  {fmt_cm(tip_robot[i]):>8s}")

    print(f"\n  逐段距离:")
    print(f"  {'手指':8s}  {'o→L3':>8s}  {'L3→L4':>8s}  {'L4→tip':>8s}")
    for i in range(nf):
        print(f"  {fi_names[i]:8s}  {fmt_cm(rseg_pip[i]):>8s}  "
              f"{fmt_cm(rseg_dip[i]):>8s}  {fmt_cm(rseg_tip[i]):>8s}")

    # ── Create input device ──────────────────────────────────────────────────
    input_device = create_input_device(args)

    # ── Preview & wait for user ──────────────────────────────────────────────
    print(f"\n{'='*68}")
    _wait_for_capture_start(
        input_device,
        "请自然伸直并张开所有手指。不要握拳；按 Enter/空格/s 后有 "
        f"{args.pose_delay:g} 秒调整姿势，再采集 {args.duration:.0f} 秒...",
    )
    _pose_countdown(input_device, args.pose_delay)
    print("开始采集张手姿态...")

    # ── Collect ──────────────────────────────────────────────────────────────
    frames, cumulative, segment, hpalm = collect_human_distances(
        input_device, retargeter, args.hand, args.duration
    )

    if frames == 0:
        print("未收到有效数据，请检查输入设备。")
        return

    pip_human, dip_human, tip_human = cumulative
    hseg_pip, hseg_dip, hseg_tip = segment

    print(f"张手采集完成，共 {frames} 帧")

    print(f"\n{'='*68}")
    print(f"人手输入距离（变换后中位数）")
    print(f"{'='*68}")
    print(f"  累计距离（wrist→各关节）:")
    print(f"  {'手指':8s}  {'→PIP':>8s}  {'→DIP':>8s}  {'→TIP':>8s}")
    for i in range(nf):
        print(f"  {fi_names[i]:8s}  {fmt_cm(pip_human[i]):>8s}  "
              f"{fmt_cm(dip_human[i]):>8s}  {fmt_cm(tip_human[i]):>8s}")

    print(f"\n  逐段距离:")
    print(f"  {'手指':8s}  {'w→PIP':>8s}  {'PIP→DIP':>8s}  {'DIP→TIP':>8s}")
    for i in range(nf):
        print(f"  {fi_names[i]:8s}  {fmt_cm(hseg_pip[i]):>8s}  "
              f"{fmt_cm(hseg_dip[i]):>8s}  {fmt_cm(hseg_tip[i]):>8s}")

    # ── Compute scaling ──────────────────────────────────────────────────────
    def ratio(robot_d, human_d):
        if robot_d and human_d and human_d > 1e-4:
            return round(robot_d / human_d, 3)
        return 1.0

    # segment_scaling scales every span of the chain independently, so each
    # factor is calibrated against its own robot link: the wrist->MCP factor
    # stretches the palm, the other three the phalanges.
    robot_roots_m = optimizer.finger_root_vectors * CM_TO_M
    open_result = {}
    for i, fname in enumerate(fi_names):
        scales, merged = segment_scales_for_finger(
            (rseg_pip[i], rseg_dip[i], rseg_tip[i]),
            (hseg_pip[i], hseg_dip[i], hseg_tip[i]),
        )
        palm = palm_scale_for_finger(
            float(np.linalg.norm(robot_roots_m[i])), float(hpalm[i])
        )
        open_result[fname] = [palm] + scales
        if merged:
            print(f"  注意: {fname} 只有两节指骨，已合并近节+中节（第 2、3 个系数相同）")

    result = {name: list(vals) for name, vals in open_result.items()}

    # Cumulative ratios are no longer what segment_scaling means, but they
    # still make a useful sanity check on overall reach.
    cumulative_result = {}
    for i, fname in enumerate(fi_names):
        cumulative_result[fname] = [
            ratio(pip_robot[i], pip_human[i]),
            ratio(dip_robot[i], dip_human[i]),
            ratio(tip_robot[i], tip_human[i]),
        ]

    print(f"\n{'='*68}")
    print(f"标定结果")
    print(f"{'='*68}")

    print("\n  segment_scaling（逐段骨长 ratio，用于配置文件）:")
    print(f"  {'手指':8s}  {'腕→MCP':>8s}  {'MCP→PIP':>8s}  {'PIP→DIP':>8s}  {'DIP→TIP':>8s}")
    for fname, vals in result.items():
        print(f"  {fname:8s}  " + "  ".join(f"{x:>8.3f}" for x in vals))

    print(f"\n  累计距离 ratio（仅供参考，不再用于配置）:")
    print(f"  {'手指':8s}  {'o→L3':>6s}  {'L3→L4':>6s}  {'L4→tip':>6s}")
    for fname, vals in cumulative_result.items():
        print(f"  {fname:8s}  {vals[0]:>6.3f}  {vals[1]:>6.3f}  {vals[2]:>6.3f}")

    print(f"\n复制以下内容到配置文件的 segment_scaling 部分:")
    print("  segment_scaling:")
    for fname, vals in result.items():
        print(f"    {fname}: {vals}")

    # ── Write back to config ─────────────────────────────────────────────────
    print(f"\n{'='*68}")
    if args.write:
        print("写入配置文件...")
        write_configs(args, robot_file, result, cumulative_result)
    else:
        # Show which files would be written
        adaptive_path, vector_path = _resolve_config_paths(args, robot_file)
        targets_info = []
        if args.optimizer in ("adaptive", "both") and adaptive_path.exists():
            targets_info.append(str(adaptive_path))
        if args.optimizer in ("vector", "both") and vector_path.exists():
            targets_info.append(str(vector_path))

        if targets_info:
            print("目标配置文件:")
            for p in targets_info:
                print(f"  {p}")
            if args.dry_run:
                print("dry-run：仅显示建议值，未写入任何配置。")
            else:
                answer = input("\n是否写入？[y=写入 / n=跳过]: ").strip().lower()
                if answer == "y":
                    write_configs(args, robot_file, result, cumulative_result)
                else:
                    print("已跳过写入。")
        else:
            print("未找到对应配置文件，跳过写入。")


if __name__ == "__main__":
    main()
