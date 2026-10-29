"""Debug visualization: compare MediaPipe input, scaled target, and retargeted FK skeletons.
比较初始输入数据，经过scaling之后的，跟经过retargeted之后的骨架

Shows four hand skeletons side-by-side in the MuJoCo viewer:
  - Blue:   Raw MediaPipe skeleton (after coordinate transform, before scaling)
  - Yellow: Retarget input skeleton uniformly scaled by pinch_scaling
  - Green:  Full-hand target built by the optimizer's segment scaling
  - Red:    Robot FK skeleton (retargeting result)

Usage:
    python debug_skeleton.py --robot leap
    python debug_skeleton.py --robot leap --input video --file data/right.mp4
    python debug_skeleton.py --robot leap --input camera
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
from anydexretarget.optimizer.base_optimizer import BaseOptimizer
from input.cli import add_input_args, config_dir_for, open_input
from output.sim.mujoco_output import (
    ROBOT_HAND_CONFIGS,
    map_retarget_qpos,
    validate_mujoco_actuator_mapping,
)

# MediaPipe hand connections (pairs of landmark indices)
MP_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]

M_TO_CM = 100.0


def draw_skeleton(scn, points, connections, color, radius=0.001, offset=None):
    """Draw a hand skeleton using capsule geoms in the MuJoCo scene.

    Args:
        scn: mujoco.MjvScene (viewer.user_scn)
        points: (N, 3) landmark positions in meters
        connections: list of (i, j) index pairs
        color: (4,) RGBA color
        radius: capsule radius
        offset: (3,) offset to shift the entire skeleton
    """
    if offset is not None:
        points = points + offset

    for i, j in connections:
        if i >= len(points) or j >= len(points):
            continue
        p1 = points[i]
        p2 = points[j]

        if scn.ngeom >= scn.maxgeom:
            break

        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[radius, 0, 0],
            pos=(p1 + p2) / 2,
            mat=np.eye(3).flatten(),
            rgba=color,
        )
        # Set capsule endpoints
        mujoco.mjv_connector(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            width=radius,
            from_=p1,
            to=p2,
        )
        scn.ngeom += 1

    # Draw spheres at joint positions
    for idx in range(len(points)):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius * 2, 0, 0],
            pos=points[idx],
            mat=np.eye(3).flatten(),
            rgba=color,
        )
        scn.ngeom += 1


def get_robot_fk_skeleton(optimizer, qpos):
    """Get robot FK positions for visualization.

    Returns:
        points: list of (3,) positions
        connections: list of (i, j) pairs
    """
    robot = optimizer.robot
    robot.compute_forward_kinematics(qpos)

    def point_pos(link_name, offset=None):
        lid = robot.get_link_index(link_name)
        pose = robot.get_link_pose(lid)
        pos = pose[:3, 3]
        if offset is not None:
            pos = pos + pose[:3, :3] @ np.asarray(offset, dtype=np.float64)
        return pos

    # Get origin position
    origin_pos = point_pos(optimizer.origin_link_name)

    points = [origin_pos.copy()]  # index 0 = origin
    connections = []

    nf = optimizer.num_fingers
    for fi in range(nf):
        # Get link positions: link1, link3, link4, tip
        link_names = []
        if hasattr(optimizer, 'link1_names') and fi < len(optimizer.link1_names):
            link_names.append((optimizer.link1_names[fi], None))
        if hasattr(optimizer, 'link3_names') and fi < len(optimizer.link3_names):
            offset = optimizer.link3_offsets[fi] if hasattr(optimizer, 'link3_offsets') else None
            link_names.append((optimizer.link3_names[fi], offset))
        if hasattr(optimizer, 'link4_names') and fi < len(optimizer.link4_names):
            offset = optimizer.link4_offsets[fi] if hasattr(optimizer, 'link4_offsets') else None
            link_names.append((optimizer.link4_names[fi], offset))
        if fi < len(optimizer.task_link_names):
            offset = optimizer.task_offsets[fi] if hasattr(optimizer, 'task_offsets') else None
            link_names.append((optimizer.task_link_names[fi], offset))

        prev_idx = 0  # origin
        for lname, offset in link_names:
            pos = point_pos(lname, offset)
            cur_idx = len(points)
            points.append(pos.copy())
            connections.append((prev_idx, cur_idx))
            prev_idx = cur_idx

    return np.array(points), connections


def build_scaled_skeleton(mediapipe_kp, optimizer):
    """Build the optimizer's scaled full-hand target skeleton.

    For the adaptive optimizer, PIP/DIP/TIP positions are taken directly from
    ``_compute_full_hand_vectors`` so the green visualization uses exactly the
    same cumulative per-segment scaling and lateral scaling as retargeting.

    Returns:
        points: scaled keypoints in meters (robot frame)
        connections: connections remapped to the returned points
    """
    robot = optimizer.robot
    origin_id = robot.get_link_index(optimizer.origin_link_name)
    origin_pos = robot.data.oMf[origin_id].translation.copy()

    scaling = optimizer.scaling if hasattr(optimizer, 'scaling') else 1.0
    wrist = mediapipe_kp[0]

    scaled_kp = np.zeros_like(mediapipe_kp)
    scaled_kp[0] = origin_pos

    mp_finger_indices = (
        optimizer.mp_finger_indices
        if hasattr(optimizer, 'mp_finger_indices')
        else [0, 1, 2, 3, 4]
    )
    active_indices = set(range(len(mediapipe_kp)))

    if hasattr(optimizer, 'segment_scaling_full'):
        seg_full = np.asarray(optimizer.segment_scaling_full, dtype=np.float64)
        mcp_indices = [optimizer.MP_MCP_INDICES[fi] for fi in mp_finger_indices]
        pip_indices = [optimizer.MP_PIP_INDICES[fi] for fi in mp_finger_indices]
        dip_indices = [optimizer.MP_DIP_INDICES[fi] for fi in mp_finger_indices]
        tip_indices = [optimizer.MP_TIP_INDICES[fi] for fi in mp_finger_indices]

        # Use the optimizer's own target builder instead of duplicating its
        # cumulative MCP->PIP->DIP->TIP scaling logic here.  It returns
        # wrist-relative PIP/DIP/TIP vectors in centimeters.
        target_vectors_m = optimizer._compute_full_hand_vectors(
            mediapipe_kp,
            seg_full,
            getattr(optimizer, 'lateral_scaling', 1.0),
            getattr(optimizer, 'lateral_axis', 1),
        ) / M_TO_CM

        nf = len(mp_finger_indices)
        scaled_kp[pip_indices] = origin_pos + target_vectors_m[:nf]
        scaled_kp[dip_indices] = origin_pos + target_vectors_m[nf:2 * nf]
        scaled_kp[tip_indices] = origin_pos + target_vectors_m[2 * nf:3 * nf]

        # MCP is not an independent loss target, but reconstruct it with the
        # exact first step used by _compute_full_hand_vectors so the displayed
        # finger chain is geometrically consistent with its PIP/DIP/TIP targets.
        palm_vectors = mediapipe_kp[mcp_indices] - wrist
        lateral_scaling = float(getattr(optimizer, 'lateral_scaling', 1.0))
        lateral_axis = int(getattr(optimizer, 'lateral_axis', 1))
        if lateral_scaling != 1.0:
            palm_vectors = palm_vectors.copy()
            palm_vectors[:, lateral_axis] *= lateral_scaling
        scaled_kp[mcp_indices] = origin_pos + palm_vectors * seg_full[:nf, 0:1]

        # Four-finger robots do not consume MediaPipe's pinky landmarks.  Return
        # only optimizer-active points so zero-initialized inactive joints are
        # not rendered as part of the green skeleton.
        active_indices = {0, *mcp_indices, *pip_indices, *dip_indices, *tip_indices}
    else:
        for i in range(1, 21):
            scaled_kp[i] = origin_pos + (mediapipe_kp[i] - wrist) * scaling

    active_indices = sorted(active_indices)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(active_indices)}
    connections = [
        (index_map[i], index_map[j])
        for i, j in MP_CONNECTIONS
        if i in index_map and j in index_map
    ]
    return scaled_kp[active_indices], connections


def build_pinch_scaled_skeleton(mediapipe_kp, optimizer):
    """Build the whole-hand skeleton uniformly scaled by pinch_scaling."""
    robot = optimizer.robot
    origin_id = robot.get_link_index(optimizer.origin_link_name)
    origin_pos = robot.data.oMf[origin_id].translation.copy()

    wrist = mediapipe_kp[0]
    scale = float(getattr(optimizer, 'pinch_scaling', 1.0))
    scaled_kp = np.zeros_like(mediapipe_kp)
    scaled_kp[0] = origin_pos
    for i in range(1, 21):
        scaled_kp[i] = origin_pos + (mediapipe_kp[i] - wrist) * scale
    return scaled_kp, MP_CONNECTIONS


def estimate_pinocchio_to_mujoco_transform(optimizer, model, data):
    """Estimate the display transform from Pinocchio to MuJoCo coordinates.

    The URDF origin link and the MuJoCo root body do not always share a name or
    frame convention.  In particular, Ability uses URDF ``base`` but MuJoCo
    ``palm``; treating the latter as the former applies an erroneous extra
    90-degree Z rotation.  Fit one rigid transform from all same-named hand
    links instead of guessing from the root body.

    Returns:
        rotation: (3, 3), applied as ``points @ rotation.T``.
        translation: (3,), added after rotation.
        diagnostics: mapping with match count and RMS/max residuals.
    """
    robot = optimizer.robot
    zero_qpos = np.zeros(robot.model.nq, dtype=np.float64)
    robot.compute_forward_kinematics(zero_qpos)

    # The live ``data`` may already be in a driven pose: actuator-controlled
    # hands are moved to the midpoint of their control ranges before this
    # function is called.  Fitting a rigid frame between Pinocchio zero FK and
    # that articulated MuJoCo pose contaminates the frame estimate with finger
    # motion (Ability can be tilted by tens of degrees).  Always compare the
    # two models in their own neutral/default state instead.
    alignment_data = mujoco.MjData(model)
    mujoco.mj_forward(model, alignment_data)

    candidate_names = [optimizer.origin_link_name]
    for attr in ("link1_names", "link3_names", "link4_names", "task_link_names"):
        candidate_names.extend(getattr(optimizer, attr, []))

    pin_points = []
    mujoco_points = []
    matched_names = []
    for link_name in dict.fromkeys(candidate_names):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        if body_id < 0:
            continue
        try:
            link_id = robot.get_link_index(link_name)
        except Exception:
            continue
        pin_points.append(robot.get_link_pose(link_id)[:3, 3].copy())
        mujoco_points.append(alignment_data.xpos[body_id].copy())
        matched_names.append(link_name)

    if len(pin_points) >= 3:
        source = np.asarray(pin_points, dtype=np.float64)
        target = np.asarray(mujoco_points, dtype=np.float64)
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        source_zero = source - source_center
        target_zero = target - target_center

        # Kabsch fit for target ~= source @ rotation.T + translation.
        u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1] *= -1.0
            rotation = vt.T @ u.T
        translation = target_center - source_center @ rotation.T

        predicted = source @ rotation.T + translation
        residuals = np.linalg.norm(predicted - target, axis=1)
        diagnostics = {
            "method": "matched-link fit",
            "matched_names": matched_names,
            "rms_error": float(np.sqrt(np.mean(residuals ** 2))),
            "max_error": float(np.max(residuals)),
        }
        return rotation, translation, diagnostics

    # Compatibility fallback for models with fewer than three matching body
    # names.  Prefer a true same-name origin.  A differently named sole root is
    # used for translation only: its local orientation is not necessarily the
    # URDF origin orientation (the Ability bug fixed above).
    pin_origin_id = robot.get_link_index(optimizer.origin_link_name)
    pin_origin_pose = robot.get_link_pose(pin_origin_id)
    pin_origin_pos = pin_origin_pose[:3, 3].copy()
    pin_origin_rot = pin_origin_pose[:3, :3].copy()
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, optimizer.origin_link_name
    )
    if body_id >= 0:
        mujoco_origin_rot = alignment_data.xmat[body_id].reshape(3, 3).copy()
        rotation = mujoco_origin_rot @ pin_origin_rot.T
        translation = alignment_data.xpos[body_id].copy() - pin_origin_pos @ rotation.T
        method = "same-name origin"
    else:
        root_body_ids = [
            index
            for index in range(1, model.nbody)
            if model.body_parentid[index] == 0
        ]
        rotation = np.eye(3, dtype=np.float64)
        if len(root_body_ids) == 1:
            translation = alignment_data.xpos[root_body_ids[0]].copy() - pin_origin_pos
            method = "root translation only"
        else:
            translation = -pin_origin_pos
            method = "identity fallback"

    return rotation, translation, {
        "method": method,
        "matched_names": matched_names,
        "rms_error": None,
        "max_error": None,
    }

def build_raw_skeleton(mediapipe_kp, optimizer):
    """Build the raw MediaPipe skeleton (no scaling, just coordinate transform).

    Places it at the robot origin for comparison.
    """
    robot = optimizer.robot
    origin_id = robot.get_link_index(optimizer.origin_link_name)
    origin_pos = robot.data.oMf[origin_id].translation.copy()

    wrist = mediapipe_kp[0]
    raw_kp = np.zeros_like(mediapipe_kp)
    raw_kp[0] = origin_pos

    for i in range(1, 21):
        vec = mediapipe_kp[i] - wrist  # meters, no scaling
        raw_kp[i] = origin_pos + vec

    return raw_kp, MP_CONNECTIONS


def main():
    parser = argparse.ArgumentParser(description="Debug skeleton visualization")
    parser.add_argument("--config", default=None, help="Config YAML path (overrides --robot and --optimizer)")
    parser.add_argument("--optimizer", default="adaptive",
                        choices=["adaptive"],
                        help="Optimizer type: adaptive (default; vector visualization removed)")
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
    config_path = args.config if args.config else f"config/{args.optimizer}/{config_dir}/{config_dir}_{robot_file}.yaml"
    config_file = EXAMPLE_ROOT / config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

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
    COLOR_PINCH = np.array([1.0, 0.9, 0.1, 0.9], dtype=np.float32)     # Yellow: pinch_scaling preview
    COLOR_SCALED = np.array([0.2, 0.9, 0.3, 0.6], dtype=np.float32)    # Green: scaled input
    COLOR_FK = np.array([1.0, 0.2, 0.2, 0.8], dtype=np.float32)        # Red: robot FK

    # Offsets to separate skeletons (left-right)
    OFFSET_RAW = np.array([-0.30, 0, 0])     # Raw on the left
    OFFSET_PINCH = np.array([-0.15, 0, 0])   # Uniform pinch scale preview
    OFFSET_SCALED = np.array([0, 0, 0])       # Scaled at center (overlaps with robot)
    OFFSET_FK = np.array([0, 0, 0])           # FK at center (robot model position)

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

            # Raw blue skeleton: transformed/rotated Pico input before
            # robot-specific preprocessing.
            raw_mediapipe_kp = apply_mediapipe_transformations(raw_kp, args.hand)
            if retargeter.rotation_xyz:
                raw_mediapipe_kp = retargeter._apply_rotation(raw_mediapipe_kp)

            # Use exactly the same end-to-end path as teleop_sim.py.  The
            # verbose keypoints are the keypoints actually consumed by
            # optimizer.solve; the returned qpos also includes the configured
            # low-pass filter, just like normal teleoperation.
            qpos, verbose = retargeter.retarget_verbose(raw_kp, apply_filter=True)
            retarget_input_kp = verbose["mediapipe_kp"]

            # Map the full retarget output to the exact MuJoCo actuator/qpos
            # order. GEORT L20 is mapped by independent joint names.
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

    # Fit the visualization frame from corresponding URDF/MuJoCo hand links.
    # This avoids assuming that a differently named MuJoCo root body has the
    # same local frame as the URDF origin (Ability otherwise gains an extra
    # 90-degree Z rotation).
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
        """Transform URDF-frame skeleton points into the MuJoCo world frame."""
        return pts @ frame_rot_matrix.T + frame_translation

    print("=" * 60)
    print("Debug Skeleton Viewer")
    print("  Blue  = Raw MediaPipe (no scaling)")
    print("  Yellow= Retarget input uniformly scaled by pinch_scaling")
    print("  Green = Optimizer full-hand target from segment_scaling")
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

                # 2. Whole-hand uniform pinch scaling preview (yellow)
                pinch_scaled_pts, pinch_scaled_conns = build_pinch_scaled_skeleton(input_mp_kp, optimizer)
                draw_skeleton(viewer.user_scn, align_points_to_mujoco(pinch_scaled_pts), pinch_scaled_conns, COLOR_PINCH,
                              radius=0.0015, offset=OFFSET_PINCH)

                # 3. Optimizer full-hand target skeleton (green)
                scaled_pts, scaled_conns = build_scaled_skeleton(input_mp_kp, optimizer)
                draw_skeleton(viewer.user_scn, align_points_to_mujoco(scaled_pts), scaled_conns, COLOR_SCALED,
                              radius=0.0015, offset=OFFSET_SCALED)

                # 4. Robot FK skeleton (red) - retargeting output
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
