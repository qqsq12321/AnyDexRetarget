"""Debug visualization for KeyVectorOptimizer (vector retargeting).
vector（KeyVectorOptimizer）专用的骨架可视化，效果和 debug_skeleton.py 类似。

Shows three hand skeletons side-by-side in the MuJoCo viewer:
  - Blue:  Raw MediaPipe skeleton (after coordinate transform, before scaling)
  - Green: Vector optimizer's scaled target points (key_vectors * scaling_factor)
  - Red:   Robot FK skeleton (retargeting result)

There is no yellow pinch-scaling preview here: KeyVectorOptimizer has no
pinch_scaling/segment_scaling concept, just one global scaling_factor.

Usage:
    python debug_skeleton_vector.py --robot leap
    python debug_skeleton_vector.py --robot linker_l20 --input avp
    python debug_skeleton_vector.py --robot shadow --input video --file data/right.mp4
"""

import argparse
import sys
import time
import threading
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from anydexretarget import Retargeter
from anydexretarget.mediapipe import apply_mediapipe_transformations
from anydexretarget.optimizer.key_vector_optimizer import KeyVectorOptimizer
from input.cli import add_input_args, config_dir_for, open_input
from output.sim.mujoco_output import (
    ROBOT_HAND_CONFIGS,
    map_retarget_qpos,
    validate_mujoco_actuator_mapping,
)
from debug_skeleton import (
    draw_skeleton,
    get_robot_fk_skeleton,
    build_raw_skeleton,
    estimate_pinocchio_to_mujoco_transform,
)


def build_vector_scaled_skeleton(mediapipe_kp, optimizer):
    """Build the KeyVectorOptimizer's scaled target skeleton.

    Mirrors ``KeyVectorOptimizer._compute_target_vectors`` exactly: each
    ``key_vectors`` entry sets

        target[task_kp] = point[origin_kp] + (mp[task_kp] - mp[origin_kp]) * scaling_factor

    Connections are drawn directly from each vector's (origin_kp, task_kp)
    pair -- not from MediaPipe's "adjacent landmark" topology table, which
    does not represent what the optimizer actually constrains. A vector
    optimizer's origin_kp is typically the wrist (0) for every entry, so each
    drawn segment is literally "wrist -> target point", not a finger chain.

    Points never referenced as a ``task_kp`` (and not the wrist) are not part
    of any loss term, so they are left out of the returned skeleton.

    Returns:
        points: (M, 3) positions in meters (robot frame)
        connections: connections remapped to the returned points
    """
    robot = optimizer.robot
    origin_id = robot.get_link_index(optimizer.origin_link_name)
    origin_pos = robot.data.oMf[origin_id].translation.copy()

    wrist = mediapipe_kp[0]
    scaling_factor = float(getattr(optimizer, 'scaling_factor', 1.0))

    scaled_kp = np.zeros_like(mediapipe_kp)
    scaled_kp[0] = origin_pos

    covered = {0}
    vector_edges = []
    for origin_kp_idx, task_kp_idx in zip(optimizer._origin_kp_indices, optimizer._task_kp_indices):
        origin_kp_idx = int(origin_kp_idx)
        task_kp_idx = int(task_kp_idx)
        if origin_kp_idx == 0:
            scaled_kp[task_kp_idx] = origin_pos + (mediapipe_kp[task_kp_idx] - wrist) * scaling_factor
        else:
            # Chain off an already-scaled point, same as the optimizer would
            # if a vector's origin were itself another vector's task point.
            scaled_kp[task_kp_idx] = scaled_kp[origin_kp_idx] + (
                mediapipe_kp[task_kp_idx] - mediapipe_kp[origin_kp_idx]
            ) * scaling_factor
        covered.add(task_kp_idx)
        vector_edges.append((origin_kp_idx, task_kp_idx))

    active_indices = sorted(covered)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(active_indices)}
    connections = [(index_map[i], index_map[j]) for i, j in vector_edges]
    return scaled_kp[active_indices], connections


def main():
    parser = argparse.ArgumentParser(description="Debug skeleton visualization (vector/KeyVectorOptimizer)")
    parser.add_argument("--config", default=None, help="Config YAML path (overrides --robot)")
    parser.add_argument("--robot", default="leap",
        choices=["shadow", "wuji", "allegro", "leap",
                 "inspire", "ability", "svh", "rohand",
                 "linkerhand_l21", "linker_l20", "unitree_dex5", "sharpa", "gaia"],
                        help="Robot hand type (default: leap)")
    parser.add_argument("--hand", default="right", choices=["left", "right"])
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Robot hand transparency (0=invisible, 1=opaque, default: 0.25)")
    add_input_args(parser, default="camera")
    args = parser.parse_args()

    robot_name_map = {
        "shadow": "shadow_hand", "wuji": "wuji_hand", "allegro": "allegro_hand",
        "leap": "leap_hand", "inspire": "inspire_hand", "ability": "ability_hand",
        "svh": "svh_hand", "rohand": "rohand", "linkerhand_l21": "linkerhand_l21",
        "linker_l20": "linker_l20", "unitree_dex5": "unitree_dex5_hand", "sharpa": "sharpa_hand",
        "gaia": "gaia_hand20",
    }
    robot_file = robot_name_map.get(args.robot, args.robot)
    config_dir = config_dir_for(args.input)
    config_path = args.config if args.config else f"config/vector/{config_dir}/{config_dir}_{robot_file}.yaml"
    config_file = EXAMPLE_ROOT / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    opt_type_str = config.get('optimizer', {}).get('type', '')
    if "KeyVector" not in opt_type_str:
        parser.error(
            f"{config_file} is not a KeyVectorOptimizer config (optimizer.type={opt_type_str!r}). "
            "Use debug_skeleton.py for adaptive configs."
        )

    robot_type = config.get('robot', {}).get('type', 'shadow_hand')
    hand_cfg = ROBOT_HAND_CONFIGS.get(robot_type, {})
    model_path = hand_cfg["model_path"](args.hand)

    print(f"Robot: {robot_type}")
    print(f"Model: {model_path}")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    actuator_joint_names = validate_mujoco_actuator_mapping(model, hand_cfg)
    if actuator_joint_names:
        print("MuJoCo actuator joint order verified:")
        for actuator_id, joint_name in enumerate(actuator_joint_names):
            print(f"  ctrl[{actuator_id:02d}] -> {joint_name}")

    # Determine control mode (same logic as teleop_sim.py)
    qpos_servo_alpha = hand_cfg.get("qpos_servo_alpha")
    direct_qpos_mode = hand_cfg.get("direct_qpos", False)
    qpos_servo_mode = (qpos_servo_alpha is not None) and not direct_qpos_mode
    actuator_mode = (model.nu > 0) and not direct_qpos_mode and not qpos_servo_mode
    if not actuator_mode and not qpos_servo_mode:
        direct_qpos_mode = True
    target_len = model.nu if actuator_mode else model.nq

    # Initialize control
    if actuator_mode:
        for i in range(model.nu):
            if model.actuator_ctrllimited[i]:
                r = model.actuator_ctrlrange[i]
                data.ctrl[i] = (r[0] + r[1]) / 2
            else:
                data.ctrl[i] = 0.0
        for _ in range(100):
            mujoco.mj_step(model, data)
    else:
        mujoco.mj_forward(model, data)

    # Initialize retargeter
    retargeter = Retargeter.from_yaml(str(config_file), args.hand)
    optimizer = retargeter.optimizer
    assert isinstance(optimizer, KeyVectorOptimizer), (
        f"Expected KeyVectorOptimizer, got {type(optimizer).__name__}"
    )
    print(f"scaling_factor: {optimizer.scaling_factor}  num_vectors: {optimizer.num_vectors}")

    input_device = open_input(args, args.hand)

    print(f"Input: {args.input}")
    if args.input == "pico4":
        if args.pico4_mode == "direct":
            print(f"Pico4 mode: direct (tcp={args.pico4_port}, udp_broadcast={args.pico4_broadcast_port})")
        else:
            print(f"Pico4 mode: relay ({args.pico4_relay_host}:{args.pico4_relay_port})")
    if qpos_servo_mode:
        print(f"Control mode: qpos servo (alpha={qpos_servo_alpha})")
    elif actuator_mode:
        print("Control mode: actuator position")
    else:
        print("Control mode: direct qpos")

    # Make only the hand semi-transparent; keep the shared scene floor opaque.
    hand_geom_mask = model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
    model.geom_rgba[hand_geom_mask, 3] = args.alpha

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    cam_cfg = config.get('render', {}).get('camera', {})
    viewer.cam.azimuth = cam_cfg.get('azimuth', 135)
    viewer.cam.elevation = cam_cfg.get('elevation', -20)
    viewer.cam.distance = cam_cfg.get('distance', 0.5)
    viewer.cam.lookat[:] = cam_cfg.get('lookat', [0, 0, 0.05])

    # Colors (RGBA float32)
    COLOR_RAW = np.array([0.2, 0.4, 1.0, 0.6], dtype=np.float32)      # Blue: raw input
    COLOR_SCALED = np.array([0.2, 0.9, 0.3, 0.6], dtype=np.float32)   # Green: scaled target
    COLOR_FK = np.array([1.0, 0.2, 0.2, 0.8], dtype=np.float32)       # Red: robot FK

    # Offsets to separate skeletons (left-right)
    OFFSET_RAW = np.array([-0.20, 0, 0])    # Raw on the left
    OFFSET_SCALED = np.array([0, 0, 0])      # Scaled at center (overlaps with robot)
    OFFSET_FK = np.array([0, 0, 0])          # FK at center (robot model position)

    # Shared state
    latest_target = np.zeros(target_len, dtype=np.float32)
    latest_raw_mediapipe_kp = None
    latest_retarget_input_kp = None
    latest_qpos = None
    data_lock = threading.Lock()
    stop_event = threading.Event()

    def input_thread_fn():
        nonlocal latest_raw_mediapipe_kp, latest_retarget_input_kp, latest_qpos
        while not stop_event.is_set():
            try:
                fingers_data = input_device.get_fingers_data()
            except Exception:
                break

            raw_kp = fingers_data[f"{args.hand}_fingers"]
            if np.allclose(raw_kp, 0):
                time.sleep(0.005)
                continue

            raw_mediapipe_kp = apply_mediapipe_transformations(raw_kp, args.hand)
            if retargeter.rotation_xyz:
                raw_mediapipe_kp = retargeter._apply_rotation(raw_mediapipe_kp)

            qpos, verbose = retargeter.retarget_verbose(raw_kp, apply_filter=True)
            retarget_input_kp = verbose["mediapipe_kp"]

            target = map_retarget_qpos(
                qpos,
                hand_cfg,
                retargeter.optimizer.robot.dof_joint_names,
            )
            target = np.asarray(target, dtype=np.float32)

            with data_lock:
                n = min(len(target), target_len)
                latest_target[:n] = target[:n]
                latest_raw_mediapipe_kp = raw_mediapipe_kp.copy()
                latest_retarget_input_kp = retarget_input_kp.copy()
                latest_qpos = qpos.copy()

    input_thread = threading.Thread(target=input_thread_fn, daemon=True)

    frame_rot_matrix, frame_translation, frame_diagnostics = (
        estimate_pinocchio_to_mujoco_transform(optimizer, model, data)
    )
    print(
        "Skeleton frame alignment: "
        f"{frame_diagnostics['method']} "
        f"({len(frame_diagnostics['matched_names'])} matched links)"
    )
    if frame_diagnostics["rms_error"] is not None:
        print(
            "  neutral FK residual: "
            f"rms={frame_diagnostics['rms_error'] * 1000:.3f} mm, "
            f"max={frame_diagnostics['max_error'] * 1000:.3f} mm"
        )

    def align_points_to_mujoco(pts):
        return pts @ frame_rot_matrix.T + frame_translation

    print("=" * 60)
    print("Debug Skeleton Viewer (vector / KeyVectorOptimizer)")
    print("  Blue  = Raw MediaPipe (no scaling)")
    print("  Green = key_vectors target (scaling_factor applied)")
    print("  Red   = Robot FK (retargeting result)")
    print("=" * 60)

    try:
        input_thread.start()

        while viewer.is_running():
            with data_lock:
                target_copy = latest_target.copy()
                raw_mp_kp = (latest_raw_mediapipe_kp.copy()
                              if latest_raw_mediapipe_kp is not None else None)
                input_mp_kp = (latest_retarget_input_kp.copy()
                                if latest_retarget_input_kp is not None else None)
                qpos_copy = latest_qpos.copy() if latest_qpos is not None else None

            # Apply control
            if direct_qpos_mode:
                data.qpos[:target_len] = target_copy
                mujoco.mj_forward(model, data)
            elif qpos_servo_mode:
                data.qpos[:target_len] += float(qpos_servo_alpha) * (target_copy - data.qpos[:target_len])
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            elif actuator_mode:
                data.ctrl[:] = target_copy
                for _ in range(10):
                    mujoco.mj_step(model, data)

            # Draw debug skeletons
            viewer.user_scn.ngeom = 0  # clear previous frame

            if raw_mp_kp is not None and input_mp_kp is not None and qpos_copy is not None:
                # 1. Raw MediaPipe skeleton (blue) - offset to left
                raw_pts, raw_conns = build_raw_skeleton(raw_mp_kp, optimizer)
                draw_skeleton(viewer.user_scn, align_points_to_mujoco(raw_pts), raw_conns, COLOR_RAW,
                              radius=0.0015, offset=OFFSET_RAW)

                # 2. Vector optimizer's scaled target skeleton (green)
                scaled_pts, scaled_conns = build_vector_scaled_skeleton(input_mp_kp, optimizer)
                draw_skeleton(viewer.user_scn, align_points_to_mujoco(scaled_pts), scaled_conns, COLOR_SCALED,
                              radius=0.0015, offset=OFFSET_SCALED)

                # 3. Robot FK skeleton (red) - retargeting output
                fk_pts, fk_conns = get_robot_fk_skeleton(optimizer, qpos_copy)
                draw_skeleton(viewer.user_scn, align_points_to_mujoco(fk_pts), fk_conns, COLOR_FK,
                              radius=0.002, offset=OFFSET_FK)

            viewer.sync()
            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        viewer.close()
        print("Done.")


if __name__ == "__main__":
    main()
