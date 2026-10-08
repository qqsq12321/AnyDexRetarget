"""Video file input device using MediaPipe Hands.

Reads an MP4/AVI video file, runs MediaPipe hand detection frame by frame,
and outputs hand landmarks in the standard (21, 3) format.

Usage:
    python teleop_sim.py --video data/right.mp4 --hand right
    python teleop_sim.py --video data/right.mp4 --hand right --show-video
"""

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from .landmark_utils import (
    HAND_CONNECTIONS,
    landmarks_to_array,
    process_landmarks,
)


class Video:
    """Read video file and extract hand landmarks via MediaPipe."""

    def __init__(
        self,
        video_path: str,
        hand_side: str = "right",
        show_video: bool = False,
        playback_speed: float = 1.0,
        loop: bool = True,
    ):
        self.hand_side = hand_side.lower()
        self.show_video = show_video
        self.playback_speed = playback_speed
        self.loop = loop

        video_path = Path(video_path)
        if not video_path.is_absolute():
            video_path = Path(__file__).resolve().parents[1] / video_path

        self.video_path = video_path
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # MediaPipe reports handedness from camera view (mirrored)
        self._expected_mp_label = "Left" if self.hand_side == "right" else "Right"

        self._last_valid_kp = None
        self._last_valid_raw = None
        self._frame_idx = 0
        self._finished = False

        print(f"Video loaded: {video_path}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height} @ {self.fps:.1f}fps")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Hand side: {self.hand_side}, playback_speed: {self.playback_speed}")

    def get_fingers_data(self) -> dict:
        empty = np.zeros((21, 3), dtype=np.float32)

        if self._finished:
            return {"left_fingers": empty, "right_fingers": empty}

        ret, frame = self._cap.read()
        if not ret:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_idx = 0
                ret, frame = self._cap.read()
                if not ret:
                    self._finished = True
                    return {"left_fingers": empty, "right_fingers": empty}
            else:
                self._finished = True
                return {"left_fingers": empty, "right_fingers": empty}

        self._frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)

        kp = None
        raw_lm = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_cls in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = hand_cls.classification[0].label
                if label == self._expected_mp_label:
                    kp = landmarks_to_array(hand_lm, self.frame_width, self.frame_height)
                    raw_lm = [(lm.x, lm.y) for lm in hand_lm.landmark]
                    break

            if kp is None and results.multi_hand_landmarks:
                kp = landmarks_to_array(results.multi_hand_landmarks[0], self.frame_width, self.frame_height)
                raw_lm = [(lm.x, lm.y) for lm in results.multi_hand_landmarks[0].landmark]

        if kp is not None:
            kp = process_landmarks(kp)
            self._last_valid_kp = kp
            self._last_valid_raw = raw_lm
        else:
            kp = self._last_valid_kp
            raw_lm = self._last_valid_raw

        if self.show_video:
            self._show_video_frame(frame, raw_lm)

        if kp is None:
            return {"left_fingers": empty, "right_fingers": empty}

        result = {"left_fingers": empty.copy(), "right_fingers": empty.copy()}
        result[f"{self.hand_side}_fingers"] = kp
        return result

    def _show_video_frame(self, frame: np.ndarray, raw_lm):
        display = frame.copy()
        if raw_lm is not None:
            h, w = display.shape[:2]
            pts = [(int(x * w), int(y * h)) for x, y in raw_lm]
            for s, e in HAND_CONNECTIONS:
                cv2.line(display, pts[s], pts[e], (0, 255, 0), 2)
            for i, pt in enumerate(pts):
                cv2.circle(display, pt, 4, (0, 0, 255), -1)

        # Show progress
        progress = self._frame_idx / max(self.total_frames, 1) * 100
        cv2.putText(display, f"Frame {self._frame_idx}/{self.total_frames} ({progress:.0f}%)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        scale = 480 / display.shape[0]
        display = cv2.resize(display, None, fx=scale, fy=scale)
        cv2.imshow("Video MediaPipe", display)
        cv2.waitKey(1)

    def __del__(self):
        if hasattr(self, "_cap"):
            self._cap.release()
        if hasattr(self, "mp_hands"):
            self.mp_hands.close()
        if hasattr(self, "show_video") and self.show_video:
            cv2.destroyAllWindows()
