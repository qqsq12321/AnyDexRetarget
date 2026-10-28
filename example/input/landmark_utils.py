"""Shared constants and functions for MediaPipe hand landmark processing.

All example input devices and test scripts that process MediaPipe landmarks
should import from here to avoid code duplication.
"""

import cv2
import numpy as np

# Reference palm size: wrist to middle MCP distance (meters)
REFERENCE_WRIST_TO_MIDDLE_MCP = 0.092

# MediaPipe hand skeleton connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def landmarks_to_array(hand_landmarks, w: int, h: int) -> np.ndarray:
    """Convert MediaPipe hand landmarks to a (21, 3) pixel-space array.

    MediaPipe reports x and y normalized to the frame and z as relative depth
    on roughly the same scale as x, so all three axes use the frame width.

    Args:
        hand_landmarks: MediaPipe hand landmarks object.
        w: Frame width in pixels.
        h: Frame height in pixels.

    Returns:
        (21, 3) float32 array in pixel coordinates.
    """
    kp = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )
    kp[:, 0] *= w
    kp[:, 1] *= h
    kp[:, 2] *= w
    return kp


def process_landmarks(
    kp: np.ndarray,
    reference_wrist_to_mid_mcp: float = REFERENCE_WRIST_TO_MIDDLE_MCP,
) -> np.ndarray:
    """Full landmark post-processing: center on the wrist and normalize scale.

    Args:
        kp: (21, 3) raw keypoints in pixel space.
        reference_wrist_to_mid_mcp: Reference wrist-to-middle-MCP distance.

    Returns:
        (21, 3) processed keypoints.
    """
    kp = kp - kp[0:1, :]
    dist = np.linalg.norm(kp[9])
    if dist < 1e-6:
        return kp
    scale = reference_wrist_to_mid_mcp / dist
    return kp * scale


def draw_skeleton(frame: np.ndarray, raw_lm, connections=None, color=(0, 255, 0)):
    """Draw hand skeleton on a frame.

    Args:
        frame: BGR image (modified in place).
        raw_lm: List of (x, y) normalized landmark coordinates.
        connections: List of (i, j) index pairs (default: HAND_CONNECTIONS).
        color: Line color in BGR.
    """
    if connections is None:
        connections = HAND_CONNECTIONS
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in raw_lm]
    for s, e in connections:
        cv2.line(frame, pts[s], pts[e], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)
