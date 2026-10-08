"""跑到指定帧后打开 MuJoCo 交互窗口，手动截图。

用法:
    python test/view_frame.py --robot shadow --frame 40
    python test/view_frame.py --robot wuji --frame 215
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXAMPLE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from anydexretarget import Retargeter
from input.landmark_utils import landmarks_to_array, process_landmarks
from output.sim.mujoco_output import (
    ROBOT_HAND_CONFIGS,
    apply_qpos_to_mujoco,
    validate_mujoco_actuator_mapping,
)


def main():
    parser = argparse.ArgumentParser(description="跑到指定帧，打开 MuJoCo 窗口截图")
    parser.add_argument("--config", type=str, default=None, help="配置文件 (相对于 example/，覆盖 --robot)")
    parser.add_argument("--robot", type=str, default="shadow",
        choices=["shadow", "wuji", "allegro", "leap",
                 "inspire", "ability", "svh", "rohand",
                 "linkerhand_l21", "linker_l20", "unitree_dex5", "sharpa", "gaia"],
                        help="灵巧手类型 (默认: shadow)")
    parser.add_argument("--frame", type=int, required=True, help="目标帧号")
    parser.add_argument("--video", type=str, default=str(EXAMPLE_DIR / "data" / "right.mp4"))
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"])
    args = parser.parse_args()

    # 加载配置
    robot_name_map = {
        "shadow": "shadow_hand", "wuji": "wuji_hand", "allegro": "allegro_hand",
        "leap": "leap_hand", "inspire": "inspire_hand", "ability": "ability_hand",
        "svh": "svh_hand", "rohand": "rohand", "linkerhand_l21": "linkerhand_l21",
        "linker_l20": "linker_l20", "unitree_dex5": "unitree_dex5_hand",
        "sharpa": "sharpa_hand", "gaia": "gaia_hand20",
    }
    robot_file = robot_name_map.get(args.robot, args.robot)
    config_path = args.config if args.config else f"config/adaptive/mediapipe/mediapipe_{robot_file}.yaml"
    config_file = EXAMPLE_DIR / config_path
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    robot_type = config.get("robot", {}).get("type", "shadow_hand")
    hand_cfg = ROBOT_HAND_CONFIGS[robot_type]

    print(f"配置: {args.config}  机器人: {robot_type}  目标帧: {args.frame}")

    # 加载 Retargeter
    retargeter = Retargeter.from_yaml(str(config_file), args.hand)

    # 打开视频，连续跑到目标帧
    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    expected_label = "Left" if args.hand == "right" else "Right"
    last_valid_kp = None
    final_qpos = None

    print(f"连续跑帧 0 ~ {args.frame} ...")
    for fi in range(args.frame + 1):
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = mp_hands.process(rgb)
        chosen = None
        if det.multi_hand_landmarks and det.multi_handedness:
            for hlm, hcls in zip(det.multi_hand_landmarks, det.multi_handedness):
                if hcls.classification[0].label == expected_label:
                    chosen = hlm; break
            if chosen is None: chosen = det.multi_hand_landmarks[0]
        if chosen is not None:
            kp_raw = landmarks_to_array(chosen, w, h)
            last_valid_kp = process_landmarks(kp_raw)
        if last_valid_kp is not None:
            qpos, verbose = retargeter.retarget_verbose(last_valid_kp, apply_filter=True)
            final_qpos = qpos

    cap.release()
    mp_hands.close()

    if final_qpos is None:
        print("未检测到手部，退出"); return

    cost = verbose["cost"]
    print(f"帧 {args.frame}: loss={cost:.4f}")

    # 加载 MuJoCo 模型并应用 qpos
    model_path = hand_cfg["model_path"](args.hand)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    validate_mujoco_actuator_mapping(model, hand_cfg)
    apply_qpos_to_mujoco(
        model,
        data,
        final_qpos,
        hand_cfg,
        retargeter.optimizer.robot.dof_joint_names,
    )

    # 打开交互窗口
    print("MuJoCo 窗口已打开，调整视角后截图。关闭窗口退出。")
    viewer = mujoco.viewer.launch_passive(model, data)
    while viewer.is_running():
        viewer.sync()

if __name__ == "__main__":
    main()
