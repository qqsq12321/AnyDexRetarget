"""RealSense camera input device for teleoperation.

Reads live RGB frames from an Intel RealSense camera, runs MediaPipe hand
detection in real time, and outputs hand landmarks in the standard (21, 3) format.

Usage:
    python teleop_sim.py --realsense --hand right
    python teleop_sim.py --realsense --hand right --show-video
"""

import cv2
import numpy as np
import mediapipe as mp

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from .landmark_utils import (
    HAND_CONNECTIONS,
    landmarks_to_array,
    process_landmarks,
)


class Realsense:
    """Read live RealSense RGB frames and extract hand landmarks via MediaPipe."""

    def __init__(
        self,
        hand_side: str = "right",
        show_video: bool = False,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        reference_wrist_to_mid_mcp: float = 0.09,
    ):
        if rs is None:
            raise ImportError("pyrealsense2 is not installed. Install it to use RealSense input.")
        self.hand_side = hand_side.lower()
        self.show_video = show_video
        self.frame_width = width
        self.frame_height = height
        self._reference_wrist_to_mid_mcp = reference_wrist_to_mid_mcp

        # Initialize RealSense pipeline (RGB stream only)
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(config)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # MediaPipe reports handedness from camera view (mirrored)
        self._expected_mp_label = "Left" if self.hand_side == "right" else "Right"

        # Cache last valid landmarks for continuity
        self._last_valid_kp = None
        self._last_valid_raw = None

        print(f"RealSense camera initialized")
        print(f"  Resolution: {width}x{height} @ {fps}fps")
        print(f"  Hand side: {self.hand_side}")

    def get_fingers_data(self) -> dict:
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        empty = np.zeros((21, 3), dtype=np.float32)
        if not color_frame:
            return {"left_fingers": empty, "right_fingers": empty}

        frame = np.asanyarray(color_frame.get_data())
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
            kp = process_landmarks(
                kp,
                reference_wrist_to_mid_mcp=self._reference_wrist_to_mid_mcp,
            )
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

        scale = 480 / display.shape[0]
        display = cv2.resize(display, None, fx=scale, fy=scale)
        cv2.putText(display, "RealSense Live",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("RealSense MediaPipe", display)
        cv2.waitKey(1)

    def __del__(self):
        if hasattr(self, "_pipeline"):
            self._pipeline.stop()
        if hasattr(self, "mp_hands"):
            self.mp_hands.close()
        if hasattr(self, "show_video") and self.show_video:
            cv2.destroyAllWindows()
