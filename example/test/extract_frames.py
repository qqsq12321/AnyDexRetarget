"""手动控制视频播放，抽取关键帧用于对比优化器性能。

快捷键:
    空格        播放 / 暂停
    D / →       下一帧
    A / ←       上一帧
    W / ↑       快进 10 帧
    S / ↓       快退 10 帧
    E           提取当前帧（保存图片 + landmarks）
    Q / ESC     退出

提取结果保存在 example/output/extracted_frames/ 下:
    frame_0123.png          原始帧图片（带骨架叠加）
    frame_0123_raw.png      原始帧图片（无骨架）
    extracted_landmarks.pkl  所有已提取帧的 landmarks 字典列表
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from input.landmark_utils import landmarks_to_array, process_landmarks, draw_skeleton


def main():
    parser = argparse.ArgumentParser(description="手动控制视频播放，提取关键帧")
    example_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--video", type=str,
                        default=str(example_dir / "data" / "right.mp4"),
                        help="视频文件路径")
    parser.add_argument("--hand", type=str, default="right", choices=["left", "right"],
                        help="手的方向 (default: right)")
    parser.add_argument("--output", type=str,
                        default=str(example_dir / "output" / "extracted_frames"),
                        help="提取帧的输出目录")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {args.video}")
    print(f"  分辨率: {frame_w}x{frame_h} @ {fps:.1f}fps, 共 {total_frames} 帧")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "extracted_landmarks.pkl"

    # 加载已有的提取记录
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            extracted = pickle.load(f)
        print(f"  已加载 {len(extracted)} 条提取记录")
    else:
        extracted = []
    extracted_indices = {e["frame_idx"] for e in extracted}

    # MediaPipe
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,  # 逐帧独立检测，更稳定
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    expected_label = "Left" if args.hand == "right" else "Right"

    win_name = "Frame Extractor"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    cur_frame = 0
    playing = False
    need_refresh = True

    def on_trackbar(val):
        nonlocal cur_frame, need_refresh
        cur_frame = val
        need_refresh = True

    cv2.createTrackbar("Frame", win_name, 0, max(total_frames - 1, 0), on_trackbar)

    print("\n--- 快捷键 ---")
    print("  空格: 播放/暂停   D/→: 下一帧   A/←: 上一帧")
    print("  W/↑: +10帧       S/↓: -10帧    E: 提取当前帧")
    print("  Q/ESC: 退出")
    print("-" * 30)

    while True:
        if playing:
            cur_frame = min(cur_frame + 1, total_frames - 1)
            if cur_frame >= total_frames - 1:
                playing = False
            need_refresh = True

        if need_refresh:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            ret, frame = cap.read()
            if not ret:
                print(f"  读取第 {cur_frame} 帧失败")
                need_refresh = False
                continue

            # MediaPipe 检测
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(rgb)

            display = frame.copy()
            raw_lm = None
            kp_processed = None

            if results.multi_hand_landmarks and results.multi_handedness:
                # 优先匹配目标手
                chosen = None
                for hand_lm, hand_cls in zip(results.multi_hand_landmarks, results.multi_handedness):
                    if hand_cls.classification[0].label == expected_label:
                        chosen = hand_lm
                        break
                if chosen is None:
                    chosen = results.multi_hand_landmarks[0]

                raw_lm = [(lm.x, lm.y) for lm in chosen.landmark]
                kp_raw = landmarks_to_array(chosen, frame_w, frame_h)
                kp_processed = process_landmarks(kp_raw)
                draw_skeleton(display, raw_lm)

            # 状态信息
            is_extracted = cur_frame in extracted_indices
            status = "PLAYING" if playing else "PAUSED"
            detected = "OK" if raw_lm else "NO HAND"
            mark = " [EXTRACTED]" if is_extracted else ""

            info_lines = [
                f"Frame {cur_frame}/{total_frames - 1} ({cur_frame / max(total_frames - 1, 1) * 100:.1f}%)  |  {status}  |  {detected}{mark}",
                f"Extracted: {len(extracted_indices)} frames  |  E=extract  SPACE=play/pause  Q=quit",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(display, line, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # 缩放显示
            scale = min(960 / display.shape[1], 720 / display.shape[0])
            if scale < 1.0:
                display = cv2.resize(display, None, fx=scale, fy=scale)

            cv2.imshow(win_name, display)
            cv2.setTrackbarPos("Frame", win_name, cur_frame)
            need_refresh = False

        wait_ms = int(1000 / fps) if playing else 30
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q") or key == 27:  # Q / ESC
            break
        elif key == ord(" "):  # 空格
            playing = not playing
            need_refresh = True
        elif key == ord("d") or key == 83:  # D / →
            playing = False
            cur_frame = min(cur_frame + 1, total_frames - 1)
            need_refresh = True
        elif key == ord("a") or key == 81:  # A / ←
            playing = False
            cur_frame = max(cur_frame - 1, 0)
            need_refresh = True
        elif key == ord("w") or key == 82:  # W / ↑
            playing = False
            cur_frame = min(cur_frame + 10, total_frames - 1)
            need_refresh = True
        elif key == ord("s") or key == 84:  # S / ↓
            playing = False
            cur_frame = max(cur_frame - 10, 0)
            need_refresh = True
        elif key == ord("e"):  # E = 提取
            if raw_lm is None or kp_processed is None:
                print(f"  [!] 第 {cur_frame} 帧未检测到手部，跳过")
                continue

            # 保存图片
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
            ret, raw_frame = cap.read()
            if ret:
                raw_path = out_dir / f"frame_{cur_frame:04d}_raw.png"
                cv2.imwrite(str(raw_path), raw_frame)

                vis_frame = raw_frame.copy()
                draw_skeleton(vis_frame, raw_lm)
                vis_path = out_dir / f"frame_{cur_frame:04d}.png"
                cv2.imwrite(str(vis_path), vis_frame)

                # 记录 landmarks
                entry = {
                    "frame_idx": cur_frame,
                    "landmarks_processed": kp_processed.copy(),
                    "landmarks_raw_pixel": np.array(raw_lm, dtype=np.float32),
                    "hand_side": args.hand,
                }

                if cur_frame in extracted_indices:
                    # 更新已有记录
                    extracted = [e for e in extracted if e["frame_idx"] != cur_frame]
                extracted.append(entry)
                extracted_indices.add(cur_frame)

                # 按帧号排序后保存
                extracted.sort(key=lambda e: e["frame_idx"])
                with open(pkl_path, "wb") as f:
                    pickle.dump(extracted, f)

                print(f"  [v] 提取第 {cur_frame} 帧 -> {vis_path.name}  (共 {len(extracted)} 帧)")
                need_refresh = True

    cap.release()
    mp_hands.close()
    cv2.destroyAllWindows()

    print(f"\n完成! 共提取 {len(extracted)} 帧，保存在 {out_dir}")
    if extracted:
        print(f"  landmarks 文件: {pkl_path}")
        print(f"  帧号列表: {[e['frame_idx'] for e in extracted]}")


if __name__ == "__main__":
    main()
