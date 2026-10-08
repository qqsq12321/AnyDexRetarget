"""Visualize MediaPipe skeleton with scaling applied.

Shows the original MANO-transformed skeleton alongside the scaled target
skeleton used by the optimizer, so you can see how scaling/segment_scaling
affect the target keypoints.

Usage:
    python visualize_scaling.py --robot allegro --input replay --file data/avp1.pkl --hand right
    python visualize_scaling.py --robot allegro --input video --file data/right.mp4 --hand right
    python visualize_scaling.py --robot wuji --input replay --file data/avp1.pkl --hand right
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from anydexretarget import Retargeter
from anydexretarget.mediapipe import apply_mediapipe_transformations
from anydexretarget.optimizer.base_optimizer import M_TO_CM
from input.cli import DEFAULT_FILES, input_file

# MediaPipe hand connections
MP_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]

MP_TIP_INDICES = [4, 8, 12, 16, 20]
MP_PIP_INDICES = [2, 6, 10, 14, 18]
MP_DIP_INDICES = [3, 7, 11, 15, 19]


def get_frames_from_pkl(pkl_path, hand_side):
    """Load frames from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    frames = []
    for frame in data:
        kp = frame.get(f'{hand_side}_fingers', np.zeros((21, 3)))
        if not np.allclose(kp, 0):
            frames.append(kp)
    return frames


def get_frames_from_video(video_path, hand_side):
    """Extract frames from video using MediaPipe."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            kp = np.array([[l.x, l.y, l.z] for l in lm.landmark])
            frames.append(kp)

    cap.release()
    hands.close()
    return frames


def apply_scaling_to_keypoints(kp_transformed, scaling, segment_scaling, mp_finger_indices):
    """Apply the same scaling the optimizer uses to the keypoints.

    Returns scaled keypoints for visualization.
    """
    wrist = kp_transformed[0]
    scaled_kp = kp_transformed.copy()

    # Scale tip vectors (same as _compute_tip_vectors)
    for fi_robot, fi_mp in enumerate(mp_finger_indices):
        tip_idx = MP_TIP_INDICES[fi_mp]
        vec = kp_transformed[tip_idx] - wrist
        scaled_kp[tip_idx] = wrist + vec * scaling

    # Scale PIP/DIP vectors (same as _compute_full_hand_vectors)
    for fi_robot, fi_mp in enumerate(mp_finger_indices):
        pip_idx = MP_PIP_INDICES[fi_mp]
        dip_idx = MP_DIP_INDICES[fi_mp]
        tip_idx = MP_TIP_INDICES[fi_mp]

        seg_s = segment_scaling[fi_robot]  # [PIP_scale, DIP_scale, TIP_scale]

        pip_vec = kp_transformed[pip_idx] - wrist
        dip_vec = kp_transformed[dip_idx] - wrist
        tip_vec = kp_transformed[tip_idx] - wrist

        scaled_kp[pip_idx] = wrist + pip_vec * seg_s[0]
        scaled_kp[dip_idx] = wrist + dip_vec * seg_s[1]
        scaled_kp[tip_idx] = wrist + tip_vec * seg_s[2]

    # Also scale MCP joints (index 1,5,9,13,17) proportionally
    mcp_indices = [1, 5, 9, 13, 17]
    for fi_mp, mcp_idx in enumerate(mcp_indices):
        if fi_mp in mp_finger_indices:
            fi_robot = mp_finger_indices.index(fi_mp)
            vec = kp_transformed[mcp_idx] - wrist
            scaled_kp[mcp_idx] = wrist + vec * segment_scaling[fi_robot][0]

    return scaled_kp


def visualize_matplotlib(frames_raw, config_path, hand_side):
    """Interactive 3D visualization using matplotlib."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    retarget_config = config.get('retarget', {})
    scaling = retarget_config.get('scaling', 1.0)

    robot_config = config.get('robot', {})
    robot_type = robot_config.get('type', 'shadow_hand')

    from anydexretarget.optimizer.base_optimizer import BaseOptimizer
    robot_defaults = BaseOptimizer.ROBOT_CONFIGS.get(robot_type, {})
    num_fingers = robot_defaults.get('num_fingers', 5)

    if num_fingers == 4:
        mp_finger_indices = [0, 1, 2, 3]
        finger_names = ['thumb', 'index', 'middle', 'ring']
    else:
        mp_finger_indices = [0, 1, 2, 3, 4]
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    seg_config = retarget_config.get('segment_scaling', {})
    segment_scaling = np.ones((num_fingers, 3))
    for i, name in enumerate(finger_names):
        if name in seg_config:
            s = seg_config[name]
            if len(s) == 3:
                segment_scaling[i] = s
            elif len(s) == 4:
                segment_scaling[i] = s[1:4]

    # Also create retargeter for FK visualization
    retargeter = Retargeter.from_yaml(str(config_path), hand_side)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f'{robot_type}', fontsize=12)

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    frame_idx = [0]

    def draw_skeleton(ax, kp_cm, title, color='blue', alpha=1.0):
        ax.cla()
        ax.set_title(title, fontsize=10)

        # Draw connections
        for i, j in MP_CONNECTIONS:
            ax.plot([kp_cm[i, 0], kp_cm[j, 0]],
                    [kp_cm[i, 1], kp_cm[j, 1]],
                    [kp_cm[i, 2], kp_cm[j, 2]],
                    color=color, linewidth=1.5, alpha=alpha)

        # Draw joints
        ax.scatter(kp_cm[:, 0], kp_cm[:, 1], kp_cm[:, 2],
                   c='red', s=20, alpha=alpha)

        # Highlight tips
        tips = kp_cm[MP_TIP_INDICES]
        ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2],
                   c='green', s=50, marker='^', alpha=alpha)

        # Wrist
        ax.scatter(kp_cm[0, 0], kp_cm[0, 1], kp_cm[0, 2],
                   c='black', s=80, marker='o')

        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')

        # Set equal aspect ratio
        max_range = np.max(np.abs(kp_cm)) * 1.2
        max_range = max(max_range, 5)
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

    def draw_fk(ax, qpos, retargeter):
        """Draw robot FK skeleton."""
        ax.cla()
        ax.set_title('Robot FK (after retarget)', fontsize=10)

        robot = retargeter.optimizer.robot
        robot.compute_forward_kinematics(qpos)

        origin_name = retargeter.optimizer.origin_link_name
        origin_pos = robot.get_link_pose(robot.get_link_index(origin_name))[:3, 3] * M_TO_CM

        tip_names = retargeter.optimizer.task_link_names
        link1_names = retargeter.optimizer.link1_names
        link3_names = retargeter.optimizer.link3_names
        link4_names = retargeter.optimizer.link4_names

        colors = ['purple', 'blue', 'green', 'orange', 'red']
        for fi in range(retargeter.optimizer.num_fingers):
            positions = [origin_pos]
            for link_list in [link1_names, link3_names, link4_names, tip_names]:
                name = link_list[fi]
                pos = robot.get_link_pose(robot.get_link_index(name))[:3, 3] * M_TO_CM
                positions.append(pos)

            positions = np.array(positions)
            c = colors[fi % len(colors)]
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                    color=c, linewidth=2, marker='o', markersize=4)

            # Label tip
            tip_pos = positions[-1]
            label = finger_names[fi] if fi < len(finger_names) else f'f{fi}'
            ax.text(tip_pos[0], tip_pos[1], tip_pos[2], f' {label}', fontsize=7)

        ax.scatter(*origin_pos, c='black', s=80, marker='o')

        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')

        max_range = 15
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

    def update(idx):
        raw_kp = frames_raw[idx]
        transformed = apply_mediapipe_transformations(raw_kp, hand_side)
        transformed_cm = transformed * M_TO_CM

        scaled = apply_scaling_to_keypoints(transformed, scaling, segment_scaling, mp_finger_indices)
        scaled_cm = scaled * M_TO_CM

        draw_skeleton(ax1, transformed_cm, f'Original (MANO)\nFrame {idx}/{len(frames_raw)}', color='blue')
        draw_skeleton(ax2, scaled_cm, f'Scaled (optimizer target)', color='darkred')

        # Retarget and draw FK
        qpos = retargeter.retarget(raw_kp, apply_filter=False)
        draw_fk(ax3, qpos, retargeter)

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == 'right':
            frame_idx[0] = min(frame_idx[0] + 1, len(frames_raw) - 1)
        elif event.key == 'left':
            frame_idx[0] = max(frame_idx[0] - 1, 0)
        elif event.key == 'up':
            frame_idx[0] = min(frame_idx[0] + 10, len(frames_raw) - 1)
        elif event.key == 'down':
            frame_idx[0] = max(frame_idx[0] - 10, 0)
        elif event.key == 'q':
            plt.close(fig)
            return
        update(frame_idx[0])

    fig.canvas.mpl_connect('key_press_event', on_key)
    update(0)

    print("Controls: Left/Right = prev/next frame, Up/Down = ±10 frames, Q = quit")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize scaling effect on hand skeleton')
    parser.add_argument('--config', type=str, default=None, help='YAML config file (overrides --robot)')
    parser.add_argument('--robot', type=str, default='allegro',
        choices=['shadow', 'wuji', 'allegro', 'leap',
                 'inspire', 'ability', 'svh', 'rohand',
                 'linkerhand_l21', 'linker_l20', 'unitree_dex5', 'sharpa', 'gaia'],
                        help='Robot hand type (default: allegro)')
    parser.add_argument('--hand', type=str, default='right', choices=['left', 'right'])
    # Offline only: frames are read straight from the file, not through a device.
    parser.add_argument('--input', type=str, default='video', choices=list(DEFAULT_FILES),
                        help='Input source (default: video)')
    parser.add_argument('--file', type=str, default=None, metavar='FILE',
                        help='Video / replay file relative to example/ (default: '
                             + ', '.join(f'{k}: {v}' for k, v in DEFAULT_FILES.items()) + ')')
    args = parser.parse_args()

    robot_name_map = {
        "shadow": "shadow_hand", "wuji": "wuji_hand", "allegro": "allegro_hand",
        "leap": "leap_hand", "inspire": "inspire_hand", "ability": "ability_hand",
        "svh": "svh_hand", "rohand": "rohand", "linkerhand_l21": "linkerhand_l21",
        "linker_l20": "linker_l20", "unitree_dex5": "unitree_dex5_hand", "sharpa": "sharpa_hand", "gaia": "gaia_hand20",
    }
    robot_file = robot_name_map.get(args.robot, args.robot)
    cfg = args.config if args.config else f"config/adaptive/mediapipe/mediapipe_{robot_file}.yaml"
    config_path = EXAMPLE_ROOT / cfg

    file_path = EXAMPLE_ROOT / input_file(args)
    if args.input == 'replay':
        print(f"Loading from {file_path}...")
        frames = get_frames_from_pkl(str(file_path), args.hand)
    else:
        print(f"Extracting from {file_path}...")
        frames = get_frames_from_video(str(file_path), args.hand)

    print(f"Loaded {len(frames)} frames")

    if not frames:
        print("No valid frames found!")
        return

    visualize_matplotlib(frames, str(config_path), args.hand)


if __name__ == '__main__':
    main()
