"""Teleoperation with MuJoCo Simulation.

Uses the Retargeter interface to map hand tracking input to robot hand joint
angles, visualized in MuJoCo simulation. Supports multiple robot hands.
"""

import argparse
import os
import pickle
import sys
import threading
import time
from pathlib import Path

if os.environ.get("MUJOCO_GL") is None and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anydexretarget import Retargeter
from input.cli import add_input_args, config_dir_for, open_input


from output.sim.mujoco_output import (
    ROBOT_HAND_CONFIGS,
    apply_qpos_to_mujoco,
    map_retarget_qpos,
    map_urdf_to_mujoco_menagerie,
    retarget_to_mujoco_target,
    validate_mujoco_actuator_mapping,
)

def _resolve_example_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _apply_camera_config(
    camera,
    cam_cfg: dict,
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    lookat: list[float] | None = None,
):
    camera.azimuth = cam_cfg.get("azimuth", 135) if azimuth is None else azimuth
    camera.elevation = cam_cfg.get("elevation", -20) if elevation is None else elevation
    camera.distance = cam_cfg.get("distance", 0.5) if distance is None else distance
    camera.lookat[:] = cam_cfg.get("lookat", [0, 0, 0.05]) if lookat is None else lookat


def run_teleop(
    input_args: argparse.Namespace,
    hand_side: str = "right",
    config_path: str = "config/mediapipe/mediapipe_shadow_hand.yaml",
    headless: bool = False,
    output_video_path: str = "",
    output_qpos_path: str = "",
    render_width: int = 1280,
    render_height: int = 720,
    max_frames: int | None = None,
    camera_azimuth: float | None = None,
    camera_elevation: float | None = None,
    camera_distance: float | None = None,
):
    """Run teleoperation with MuJoCo simulation.

    ``input_args`` holds the options registered by ``input.cli.add_input_args``.
    """
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"
    input_device_type = input_args.input
    playback_speed = input_args.speed
    playback_loop = not input_args.no_loop
    enable_recording = input_args.record

    config_file = Path(__file__).parent / config_path
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    robot_type = config.get("robot", {}).get("type", "shadow_hand")
    if robot_type not in ROBOT_HAND_CONFIGS:
        raise ValueError(f"Unknown robot type: {robot_type}. Supported: {list(ROBOT_HAND_CONFIGS.keys())}")

    hand_cfg = ROBOT_HAND_CONFIGS[robot_type]
    model_path = Path(hand_cfg["model_path"](hand_side))
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")

    print(f"  Robot: {robot_type}")
    print(f"  Model: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    actuator_joint_names = validate_mujoco_actuator_mapping(model, hand_cfg)
    if actuator_joint_names:
        print("  MuJoCo actuator joint order verified:")
        for actuator_id, joint_name in enumerate(actuator_joint_names):
            print(f"    ctrl[{actuator_id:02d}] -> {joint_name}")
    force_control_cfg = hand_cfg.get("force_control")
    qpos_servo_alpha = hand_cfg.get("qpos_servo_alpha")
    direct_qpos_mode = hand_cfg.get("direct_qpos", False)
    qpos_servo_mode = (qpos_servo_alpha is not None) and not direct_qpos_mode
    actuator_mode = (model.nu > 0) and not direct_qpos_mode and not qpos_servo_mode
    force_control_mode = (force_control_cfg is not None) and not actuator_mode and not direct_qpos_mode and not qpos_servo_mode
    target_len = model.nu if actuator_mode else model.nq

    max_offscreen_width = int(model.vis.global_.offwidth)
    max_offscreen_height = int(model.vis.global_.offheight)
    if render_width > max_offscreen_width or render_height > max_offscreen_height:
        print(f"  Clamp offscreen render size: {render_width}x{render_height} -> {min(render_width, max_offscreen_width)}x{min(render_height, max_offscreen_height)}")
        render_width = min(render_width, max_offscreen_width)
        render_height = min(render_height, max_offscreen_height)

    if actuator_mode:
        print("  Control mode: actuator position")
        for i in range(model.nu):
            if model.actuator_ctrllimited[i]:
                ctrl_range = model.actuator_ctrlrange[i]
                data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
            else:
                data.ctrl[i] = 0.0
        for _ in range(100):
            mujoco.mj_step(model, data)
    elif qpos_servo_mode:
        print("  Control mode: qpos servo")
        mujoco.mj_forward(model, data)
    elif force_control_mode:
        print("  Control mode: joint PD force")
        mujoco.mj_forward(model, data)
    else:
        print("  Control mode: direct qpos")
        direct_qpos_mode = True
        mujoco.mj_forward(model, data)

    cam_cfg = config.get("render", {}).get("camera", {})
    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data)
        _apply_camera_config(
            viewer.cam,
            cam_cfg,
            azimuth=camera_azimuth,
            elevation=camera_elevation,
            distance=camera_distance,
        )

    if input_device_type == "pico4":
        if input_args.pico4_mode == "relay":
            print(f"  Pico 4: relay mode {input_args.pico4_relay_host}:{input_args.pico4_relay_port}")
        else:
            print(
                f"  Pico 4: direct mode TCP {input_args.pico4_port}, "
                f"UDP broadcast {input_args.pico4_broadcast_port}"
            )

    input_device = open_input(input_args, hand_side)
    retargeter = Retargeter.from_yaml(str(config_file), hand_side)

    if input_device_type in ("replay", "video") and enable_recording:
        print("Note: Recording disabled in replay/video mode")
        enable_recording = False

    output_video = _resolve_example_path(output_video_path)
    output_qpos = _resolve_example_path(output_qpos_path)
    trajectory_log = [] if output_qpos is not None else None

    renderer = None
    render_cam = None
    video_writer = None
    if headless or output_video is not None:
        renderer = mujoco.Renderer(model, render_height, render_width)
        render_cam = mujoco.MjvCamera()
        render_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        _apply_camera_config(
            render_cam,
            cam_cfg,
            azimuth=camera_azimuth,
            elevation=camera_elevation,
            distance=camera_distance,
        )
        if output_video is not None:
            fps = 30.0
            if input_device_type == "video":
                fps = max(1.0, getattr(input_device, "fps", 30.0) * playback_speed)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (render_width, render_height))
            if not video_writer.isOpened():
                raise RuntimeError(f"Cannot open output video writer: {output_video}")

    input_data_log = [] if enable_recording else None
    start_time = time.time()
    synchronous_input = headless and input_device_type == "video"

    latest_target = np.zeros(target_len, dtype=np.float32)
    ctrl_lock = threading.Lock()
    ctrl_ready = False
    stop_event = threading.Event()
    input_frame_count = 0

    def maybe_log_input(fingers_data: dict):
        if enable_recording and input_data_log is not None:
            input_data_log.append({
                "t": time.time() - start_time,
                "left_fingers": fingers_data["left_fingers"].copy(),
                "right_fingers": fingers_data["right_fingers"].copy(),
            })

    def apply_target(target: np.ndarray | None):
        if target is None:
            return
        if direct_qpos_mode:
            data.qpos[:] = target
            mujoco.mj_forward(model, data)
        elif actuator_mode:
            data.ctrl[:] = target
        elif qpos_servo_mode or force_control_mode:
            latest_target[:] = target

    def input_thread_fn():
        nonlocal ctrl_ready, input_frame_count
        while not stop_event.is_set():
            try:
                fingers_data = input_device.get_fingers_data()
            except Exception:
                break
            target = retarget_to_mujoco_target(
                fingers_data=fingers_data,
                hand_side=hand_side,
                retargeter=retargeter,
                hand_cfg=hand_cfg,
                target_len=target_len,
            )
            if target is None:
                time.sleep(0.005)
                continue
            maybe_log_input(fingers_data)
            with ctrl_lock:
                latest_target[:] = target
                ctrl_ready = True
            input_frame_count += 1

    input_thread = None if synchronous_input else threading.Thread(target=input_thread_fn, daemon=True)

    try:
        print("Starting teleoperation...")
        print(f"  Config: {config_path}")
        print(f"  Hand: {hand_side}")
        print(f"  Input: {input_device_type}")
        print(f"  Headless: {'ON' if headless else 'OFF'}")
        if output_video is not None:
            print(f"  Save video: {output_video}")
        if output_qpos is not None:
            print(f"  Save qpos: {output_qpos}")
        print(f"  Recording: {'ON' if enable_recording else 'OFF'}")
        print("=" * 50)

        if input_thread is not None:
            input_thread.start()

        render_count = 0
        fps_start_time = time.time()
        sim_dt = model.opt.timestep
        if synchronous_input and not direct_qpos_mode:
            target_fps = max(1.0, getattr(input_device, "fps", 30.0) * playback_speed)
            n_substeps = max(1, int(round((1.0 / target_fps) / sim_dt)))
        else:
            n_substeps = 10
        render_interval = sim_dt * n_substeps

        while True:
            loop_start = time.time()

            if viewer is not None and not viewer.is_running():
                break

            if synchronous_input:
                fingers_data = input_device.get_fingers_data()
                target = retarget_to_mujoco_target(
                    fingers_data=fingers_data,
                    hand_side=hand_side,
                    retargeter=retargeter,
                    hand_cfg=hand_cfg,
                    target_len=target_len,
                )
                if getattr(input_device, "_finished", False) and target is None:
                    break
                maybe_log_input(fingers_data)
                if target is not None:
                    apply_target(target)
                    input_frame_count += 1
            else:
                with ctrl_lock:
                    if ctrl_ready:
                        apply_target(latest_target)

            if direct_qpos_mode:
                if not synchronous_input:
                    mujoco.mj_forward(model, data)
            elif actuator_mode:
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)
            elif qpos_servo_mode:
                data.qpos[:target_len] += float(qpos_servo_alpha) * (latest_target - data.qpos[:target_len])
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
            elif force_control_mode:
                kp = float(force_control_cfg.get("kp", 10.0))
                kd = float(force_control_cfg.get("kd", 1.0))
                max_force = float(force_control_cfg.get("max_force", 2.0))
                for _ in range(n_substeps):
                    q_err = latest_target - data.qpos[:target_len]
                    tau = kp * q_err - kd * data.qvel[:target_len]
                    tau = np.clip(tau, -max_force, max_force)
                    data.qfrc_applied[:] = 0.0
                    data.qfrc_applied[:target_len] = tau
                    mujoco.mj_step(model, data)
            else:
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()

            if trajectory_log is not None:
                trajectory_log.append({
                    "t": time.time() - start_time,
                    "target": np.array(data.qpos if direct_qpos_mode else (latest_target if (qpos_servo_mode or force_control_mode) else data.ctrl), copy=True),
                    "sim_qpos": np.array(data.qpos, copy=True),
                    "sim_ctrl": np.array(data.ctrl, copy=True) if model.nu > 0 else np.zeros(0, dtype=np.float32),
                })

            if renderer is not None and video_writer is not None:
                renderer.update_scene(data, camera=render_cam)
                frame_rgb = renderer.render()
                video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            render_count += 1
            if render_count % 200 == 0:
                elapsed = time.time() - fps_start_time
                render_fps = render_count / elapsed
                print(f"Render FPS: {render_fps:.1f}  |  Input FPS: {input_frame_count / elapsed:.1f}")

            if max_frames is not None and render_count >= max_frames:
                break

            if not headless:
                elapsed_this_frame = time.time() - loop_start
                sleep_time = render_interval - elapsed_this_frame
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if (
                not synchronous_input
                and input_device_type in ("video", "replay")
                and not playback_loop
                and getattr(input_device, "_finished", False)
            ):
                break

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        stop_event.set()
        if input_thread is not None:
            input_thread.join(timeout=2.0)
        if viewer is not None:
            viewer.close()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved simulation video to {output_video}")
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass
        if trajectory_log is not None and output_qpos is not None:
            with open(output_qpos, "wb") as f:
                pickle.dump(trajectory_log, f)
            print(f"Saved trajectory log with {len(trajectory_log)} frames to {output_qpos}")

    return input_data_log


def main():
    parser = argparse.ArgumentParser(
        description="Teleoperation with MuJoCo Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file (overrides --robot and --optimizer)")
    parser.add_argument("--optimizer", type=str, default="adaptive",
                        choices=["adaptive", "vector", "dexpilot"],
                        help="Optimizer type: adaptive (default), vector (KeyVectorOptimizer), or dexpilot (DexPilotOptimizer)")
    parser.add_argument("--robot", type=str, default="shadow",
                        choices=["shadow", "wuji", "allegro", "leap",
                                 "inspire", "ability", "svh", "rohand",
                                 "linkerhand_l21", "linker_l20", "unitree_dex5", "sharpa", "gaia"],
                        help="Robot hand type (default: shadow)")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"],
                        help="Hand side (default: right)")

    add_input_args(parser, default="video", recording=True)

    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI viewer")
    parser.add_argument("--save-sim", type=str, default=None, metavar="FILE",
                        help="Save MuJoCo simulation video to MP4")
    parser.add_argument("--save-qpos", type=str, default=None, metavar="FILE",
                        help="Save target/sim qpos trajectory to PKL")
    parser.add_argument("--width", type=int, default=1280,
                        help="Offscreen render width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Offscreen render height (default: 720)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum number of simulation/render frames")
    parser.add_argument("--cam-azimuth", type=float, default=None,
                        help="Override camera azimuth")
    parser.add_argument("--cam-elevation", type=float, default=None,
                        help="Override camera elevation")
    parser.add_argument("--cam-distance", type=float, default=None,
                        help="Override camera distance")

    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        robot_name_map = {
            "shadow": "shadow_hand", "wuji": "wuji_hand", "allegro": "allegro_hand",
            "leap": "leap_hand", "inspire": "inspire_hand", "ability": "ability_hand",
            "svh": "svh_hand", "rohand": "rohand", "linkerhand_l21": "linkerhand_l21",
            "linker_l20": "linker_l20", "unitree_dex5": "unitree_dex5_hand", "sharpa": "sharpa_hand",
            "gaia": "gaia_hand20",
        }
        config_dir = config_dir_for(args.input)
        robot_file = robot_name_map.get(args.robot, args.robot)
        config_path = f"config/{args.optimizer}/{config_dir}/{config_dir}_{robot_file}.yaml"

    log = run_teleop(
        input_args=args,
        hand_side=args.hand,
        config_path=config_path,
        headless=args.headless or args.save_sim is not None,
        output_video_path=args.save_sim or "",
        output_qpos_path=args.save_qpos or "",
        render_width=args.width,
        render_height=args.height,
        max_frames=args.max_frames,
        camera_azimuth=args.cam_azimuth,
        camera_elevation=args.cam_elevation,
        camera_distance=args.cam_distance,
    )

    if log is not None and len(log) > 0:
        if args.record_file:
            log_path = Path(args.record_file)
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_path = Path(__file__).parent / f"input_data_log_{timestamp}.pkl"
        with open(log_path, "wb") as f:
            pickle.dump(log, f)
        print(f"Saved input data log with {len(log)} entries to {log_path}")


if __name__ == "__main__":
    main()
