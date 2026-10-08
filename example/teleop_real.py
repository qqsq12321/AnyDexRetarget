"""Teleoperation with real robot hand hardware.

Uses the Retargeter interface to map hand tracking input to joint targets and
send them to real hardware via:
- wujihandpy (Wuji Hand)
- TCP socket bridge (Shadow Hand)
- direct serial protocol (Inspire RH56DFX)
- Gaia HandSDK (GaiaHand20 / Gaia20)
- RS485 Modbus (Linker L20)
"""

import argparse
import pickle
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anydexretarget import Retargeter
from input.cli import add_input_args, config_dir_for, open_input
from output.real import GaiaHand20Output, InspireSerialOutput, LinkerL20Output, ShadowTCPOutput, WujiOutput


# -------------------- Teleoperation --------------------

def run_teleop(
    input_args: argparse.Namespace,
    robot_type: str = "wuji",
    hand_side: str = "right",
    config_path: str = "config/mediapipe/mediapipe_wuji_hand.yaml",
    docker_ip: str = "localhost",
    docker_port: int = 5555,
    inspire_port: str = "/dev/ttyUSB0",
    inspire_baudrate: int = 115200,
    inspire_hand_id: int = 1,
    gaia_port: str = "/dev/ttyACM0",
    gaia_baudrate: int = 921600,
    gaia_use_slcan: bool = True,
    gaia_has_main_board: bool = True,
    gaia_speed: float = 1.0,
    gaia_use_broadcast: bool = True,
    gaia_lpf_level: int = 3,
    gaia_pip_scale: float = 1.5,
    gaia_thumb_pip_scale: float = 1.1,
    gaia_enable_delay: float = 1.0,
    gaia_command_hz: float = 100.0,
    gaia_zero_on_close: bool = False,
    l20_port: str = "/dev/ttyUSB0",
    l20_baudrate: int = 460800,
    l20_slave_id: int | None = None,
    l20_speed: int = 200,
    l20_current_limit: int = 200,
    l20_clear_faults: bool = False,
    l20_open_on_exit: bool = False,
    l20_print_registers: bool = False,
    l20_dry_run: bool = False,
):
    """Run teleoperation with real hardware.

    Input acquisition + retargeting runs in a background thread, while the main
    thread sends the latest joint target to hardware at a steady control rate.
    ``input_args`` holds the options registered by ``input.cli.add_input_args``.
    """
    hand_side = hand_side.lower()
    assert hand_side in {"right", "left"}, "hand_side must be 'right' or 'left'"
    input_device_type = input_args.input
    playback_loop = not input_args.no_loop
    enable_recording = input_args.record

    input_device = open_input(input_args, hand_side)

    config_file = Path(__file__).parent / config_path
    retargeter = Retargeter.from_yaml(str(config_file), hand_side)

    # Get joint names from retargeter for name-aware hardware mappings.
    joint_names = retargeter.optimizer.robot.dof_joint_names

    # Initialize hardware only after input/config validation. This avoids
    # leaving an enabled hand behind when an input dependency or YAML is bad.
    if robot_type == "wuji":
        output = WujiOutput()
    elif robot_type == "shadow":
        output = ShadowTCPOutput(docker_ip=docker_ip, port=docker_port)
    elif robot_type == "inspire":
        output = InspireSerialOutput(
            port_name=inspire_port,
            baudrate=inspire_baudrate,
            hand_id=inspire_hand_id,
        )
    elif robot_type == "gaia":
        output = GaiaHand20Output(
            port_name=gaia_port,
            hand_side=hand_side,
            baudrate=gaia_baudrate,
            use_slcan=gaia_use_slcan,
            has_main_board=gaia_has_main_board,
            speed=gaia_speed,
            use_broadcast=gaia_use_broadcast,
            lpf_level=gaia_lpf_level,
            pip_scale=gaia_pip_scale,
            thumb_pip_scale=gaia_thumb_pip_scale,
            enable_delay=gaia_enable_delay,
            command_hz=gaia_command_hz,
            zero_on_close=gaia_zero_on_close,
        )
    elif robot_type == "linker_l20":
        output = LinkerL20Output(
            port=l20_port,
            baudrate=l20_baudrate,
            slave_id=l20_slave_id if l20_slave_id is not None else (0x29 if hand_side == "right" else 0x2A),
            speed=l20_speed,
            current_limit=l20_current_limit,
            clear_faults=l20_clear_faults,
            open_on_exit=l20_open_on_exit,
            print_registers=l20_print_registers,
            dry_run=l20_dry_run,
        )
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

    if input_device_type in ("replay", "video") and enable_recording:
        print("Note: Recording disabled in replay/video mode")
        enable_recording = False

    input_data_log = [] if enable_recording else None
    start_time = time.time()

    latest_qpos = np.zeros(retargeter.num_joints, dtype=np.float32)
    qpos_lock = threading.Lock()
    qpos_ready = False
    stop_event = threading.Event()
    input_frame_count = 0
    control_frame_count = 0
    input_thread_error = None

    def input_thread_fn():
        nonlocal qpos_ready, input_frame_count, input_thread_error
        while not stop_event.is_set():
            try:
                fingers_data = input_device.get_fingers_data()
            except Exception as exc:
                input_thread_error = exc
                break

            fingers_pose = fingers_data[f"{hand_side}_fingers"]
            if np.allclose(fingers_pose, 0):
                if (
                    input_device_type in ("replay", "video")
                    and not playback_loop
                    and getattr(input_device, "_finished", False)
                ):
                    break
                time.sleep(0.01)
                continue

            if enable_recording and input_data_log is not None:
                input_data_log.append({
                    "t": time.time() - start_time,
                    "left_fingers": fingers_data["left_fingers"].copy(),
                    "right_fingers": fingers_data["right_fingers"].copy(),
                })

            qpos = retargeter.retarget(fingers_pose)
            with qpos_lock:
                latest_qpos[:] = qpos
                qpos_ready = True
            input_frame_count += 1

    input_thread = threading.Thread(target=input_thread_fn, daemon=True)

    try:
        print("Starting teleoperation...")
        print(f"  Robot: {robot_type}")
        print(f"  Config: {config_path}")
        print(f"  Hand: {hand_side}")
        print(f"  Input: {input_device_type}")
        print(f"  Recording: {'ON' if enable_recording else 'OFF'}")
        if input_device_type == "pico4":
            if input_args.pico4_mode == "direct":
                print(f"  Pico4 mode: direct (tcp={input_args.pico4_port}, "
                      f"udp_broadcast={input_args.pico4_broadcast_port})")
            else:
                print(f"  Pico4 mode: relay ({input_args.pico4_relay_host}:{input_args.pico4_relay_port})")
        if robot_type == "shadow":
            print(f"  Shadow bridge: {docker_ip}:{docker_port}")
        elif robot_type == "inspire":
            print(f"  Inspire serial: {inspire_port} @ {inspire_baudrate} (id={inspire_hand_id})")
        elif robot_type == "gaia":
            print(
                f"  Gaia HandSDK: {gaia_port} @ {gaia_baudrate} "
                f"(slcan={gaia_use_slcan}, main_board={gaia_has_main_board}, "
                f"command_hz={gaia_command_hz:g}, pip_scale={gaia_pip_scale:g}, "
                f"thumb_pip_scale={gaia_thumb_pip_scale:g})"
            )
        elif robot_type == "linker_l20":
            effective_l20_slave_id = l20_slave_id if l20_slave_id is not None else (0x29 if hand_side == "right" else 0x2A)
            print(
                f"  Linker L20 RS485: {l20_port} @ {l20_baudrate} "
                f"(slave_id={effective_l20_slave_id}, speed={l20_speed}, "
                f"current_limit={l20_current_limit}, dry_run={l20_dry_run})"
            )
        print("=" * 50)

        input_thread.start()

        control_hz = 100.0
        control_dt = 1.0 / control_hz
        fps_start_time = time.time()

        while True:
            loop_start = time.time()

            with qpos_lock:
                if qpos_ready:
                    qpos_to_send = latest_qpos.copy()
                else:
                    qpos_to_send = None

            if qpos_to_send is not None:
                output.send(qpos_to_send, joint_names)
                control_frame_count += 1

            if control_frame_count > 0 and control_frame_count % 100 == 0:
                elapsed = time.time() - fps_start_time
                control_fps = control_frame_count / elapsed
                input_fps = input_frame_count / elapsed if elapsed > 0 else 0.0
                print(f"Control FPS: {control_fps:.1f}  |  Input FPS: {input_fps:.1f}")

            if input_thread_error is not None and not input_thread.is_alive():
                raise RuntimeError("Input/retargeting thread stopped unexpectedly") from input_thread_error

            if (
                input_device_type in ("replay", "video")
                and not playback_loop
                and getattr(input_device, "_finished", False)
                and not input_thread.is_alive()
            ):
                break

            sleep_time = control_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        stop_event.set()
        input_thread.join(timeout=2.0)
        output.close()

    return input_data_log


def main():
    parser = argparse.ArgumentParser(
        description="Teleoperation with Real Robot Hand Hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Wuji Hand with replay data
  python teleop_real.py --robot wuji --input replay --file data/avp1.pkl

  # Shadow Hand via TCP to Docker ROS bridge
  python teleop_real.py --robot shadow --input replay --file data/avp1.pkl --docker-ip localhost --docker-port 5555

  # Live VisionPro input with Shadow Hand
  python teleop_real.py --robot shadow --input avp --avp-ip <your-vision-pro-ip>

  # Inspire Hand via direct serial control
  python teleop_real.py --robot inspire --input noitom --hand right --noitom-local-ip 192.168.5.25 --inspire-port /dev/ttyUSB0

  # Pico 4 direct mode with PC broadcast discovery
  python teleop_real.py --robot inspire --input pico4 --hand right --pico4-mode direct --inspire-port /dev/ttyUSB0

  # GaiaHand20 through HandSDK using the local Pico relay daemon
  python teleop_real.py --robot gaia --input pico4 --hand right --pico4-mode relay --gaia-port /dev/ttyACM0

  # Linker L20 through RS485 Modbus
  python teleop_real.py --robot linker_l20 --input pico4 --hand right --pico4-mode relay --l20-port /dev/ttyUSB0

  # Linker L20 dry-run register print without opening serial
  python teleop_real.py --robot linker_l20 --input replay --file data/avp1.pkl --l20-dry-run --l20-print-registers

  # GaiaHand20 without main board (direct serial)
  python teleop_real.py --robot gaia --input replay --file data/avp1.pkl --gaia-port /dev/ttyUSB0 --gaia-baudrate 230400 --no-gaia-use-slcan --no-gaia-has-main-board

  # RealSense camera input
  python teleop_real.py --robot shadow --input realsense

  # Video file input
  python teleop_real.py --robot shadow --input video --file data/right.mp4

  # Record input data while using VisionPro
  python teleop_real.py --input avp --record
        """,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file (overrides --robot and --optimizer)")
    parser.add_argument("--optimizer", type=str, default="adaptive",
                        choices=["adaptive", "vector", "dexpilot"],
                        help="Optimizer type: adaptive (default), vector (KeyVectorOptimizer), or dexpilot (DexPilotOptimizer)")
    parser.add_argument("--robot", type=str, default="wuji",
                        choices=["wuji", "shadow", "inspire", "gaia", "linker_l20"],
                        help="Robot hand type (default: wuji)")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"],
                        help="Hand side (default: right)")

    add_input_args(parser, default="replay", recording=True)

    # Shadow Hand TCP options
    parser.add_argument("--docker-ip", type=str, default="localhost",
                        help="Docker ROS bridge IP (default: localhost, for --robot shadow)")
    parser.add_argument("--docker-port", type=int, default=5555,
                        help="Docker ROS bridge port (default: 5555, for --robot shadow)")
    parser.add_argument("--inspire-port", type=str, default="/dev/ttyUSB0",
                        help="Inspire serial port (default: /dev/ttyUSB0, for --robot inspire)")
    parser.add_argument("--inspire-baudrate", type=int, default=115200,
                        help="Inspire serial baudrate (default: 115200, for --robot inspire)")
    parser.add_argument("--inspire-hand-id", type=int, default=1,
                        help="Inspire hand ID in serial protocol (default: 1, for --robot inspire)")

    # GaiaHand20 / HandSDK options
    parser.add_argument("--gaia-port", type=str, default="/dev/ttyACM0",
                        help="GaiaHand20 serial/SLCAN port (default: /dev/ttyACM0)")
    parser.add_argument("--gaia-baudrate", type=int, default=921600,
                        help="GaiaHand20 baudrate (default: 921600; use 230400 without main board)")
    parser.add_argument("--gaia-use-slcan", action=argparse.BooleanOptionalAction, default=True,
                        help="Use Gaia HandSDK SLCAN transport (default: enabled)")
    parser.add_argument("--gaia-has-main-board", action=argparse.BooleanOptionalAction, default=True,
                        help="Declare that the Gaia hand has a main board (default: enabled)")
    parser.add_argument("--gaia-speed", type=float, default=1.0,
                        help="Gaia movement speed in (0, 1] (default/maximum: 1.0)")
    parser.add_argument("--gaia-use-broadcast", action=argparse.BooleanOptionalAction, default=True,
                        help="Use HandSDK broadcast joint commands (default: enabled)")
    parser.add_argument("--gaia-lpf-level", type=int, choices=range(0, 6), default=3,
                        help="Gaia motor position LPF level, 0-5 (default: 3)")
    parser.add_argument("--gaia-pip-scale", type=float, default=1.5,
                        help="Gaia non-thumb PIP command scale (default: 1.5; DIP follows mechanically)")
    parser.add_argument("--gaia-thumb-pip-scale", type=float, default=1.1,
                        help="Gaia thumb PIP (joint_3) command scale (default: 1.1)")
    parser.add_argument("--gaia-enable-delay", type=float, default=1.0,
                        help="Seconds to wait after enabling Gaia motors (default: 1.0)")
    parser.add_argument("--gaia-command-hz", type=float, default=100.0,
                        help="Maximum Gaia command rate in Hz (default/effective maximum: 100)")
    parser.add_argument("--gaia-zero-on-close", action="store_true",
                        help="Return GaiaHand20 to zero before disabling it on exit")

    # Linker L20 RS485 options
    parser.add_argument("--l20-port", type=str, default="/dev/ttyUSB0",
                        help="Linker L20 serial/RS485 port (default: /dev/ttyUSB0)")
    parser.add_argument("--l20-baudrate", type=int, default=460800,
                        help="Linker L20 baudrate (default: 460800)")
    parser.add_argument("--l20-slave-id", type=lambda x: int(x, 0), default=None,
                        help="Linker L20 Modbus slave id (default: 0x29 right, 0x2A left)")
    parser.add_argument("--l20-speed", type=int, default=200,
                        help="Linker L20 speed register value, 0-255 (default: 200)")
    parser.add_argument("--l20-current-limit", type=int, default=200,
                        help="Linker L20 current-limit register value, 0-255 (default: 200)")
    parser.add_argument("--l20-clear-faults", action="store_true",
                        help="Clear Linker L20 fault registers during startup")
    parser.add_argument("--l20-open-on-exit", action="store_true",
                        help="Return Linker L20 to open-palm registers before closing")
    parser.add_argument("--l20-print-registers", action="store_true",
                        help="Print Linker L20 register commands")
    parser.add_argument("--l20-dry-run", action="store_true",
                        help="Compute and print Linker L20 registers without opening serial")


    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        robot_name_map = {
            "wuji": "wuji_hand",
            "shadow": "shadow_hand",
            "inspire": "inspire_hand",
            "gaia": "gaia_hand20",
            "linker_l20": "linker_l20",
        }
        config_dir = config_dir_for(args.input)
        robot_file = robot_name_map.get(args.robot, args.robot)
        config_path = f"config/{args.optimizer}/{config_dir}/{config_dir}_{robot_file}.yaml"

    log = run_teleop(
        input_args=args,
        robot_type=args.robot,
        hand_side=args.hand,
        config_path=config_path,
        docker_ip=args.docker_ip,
        docker_port=args.docker_port,
        inspire_port=args.inspire_port,
        inspire_baudrate=args.inspire_baudrate,
        inspire_hand_id=args.inspire_hand_id,
        gaia_port=args.gaia_port,
        gaia_baudrate=args.gaia_baudrate,
        gaia_use_slcan=args.gaia_use_slcan,
        gaia_has_main_board=args.gaia_has_main_board,
        gaia_speed=args.gaia_speed,
        gaia_use_broadcast=args.gaia_use_broadcast,
        gaia_lpf_level=args.gaia_lpf_level,
        gaia_pip_scale=args.gaia_pip_scale,
        gaia_thumb_pip_scale=args.gaia_thumb_pip_scale,
        gaia_enable_delay=args.gaia_enable_delay,
        gaia_command_hz=args.gaia_command_hz,
        gaia_zero_on_close=args.gaia_zero_on_close,
        l20_port=args.l20_port,
        l20_baudrate=args.l20_baudrate,
        l20_slave_id=args.l20_slave_id,
        l20_speed=args.l20_speed,
        l20_current_limit=args.l20_current_limit,
        l20_clear_faults=args.l20_clear_faults,
        l20_open_on_exit=args.l20_open_on_exit,
        l20_print_registers=args.l20_print_registers,
        l20_dry_run=args.l20_dry_run,
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
