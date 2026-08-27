"""Low-latency output filtering for retargeting v2."""

from __future__ import annotations

import numpy as np

from .utils import LPFilter


class AdaptiveLPFilterV2(LPFilter):
    """Filter micro-jitter while bypassing intentional joint motion."""

    def __init__(
        self,
        alpha: float,
        bypass_ratio: float,
        joint_ranges: np.ndarray,
        passthrough_mask: np.ndarray | None = None,
    ):
        super().__init__(alpha)
        joint_ranges = np.asarray(joint_ranges, dtype=np.float64)
        self.bypass_thresholds = np.maximum(joint_ranges, 1e-8) * float(
            bypass_ratio
        )
        if passthrough_mask is None:
            passthrough_mask = np.zeros(joint_ranges.shape, dtype=bool)
        self.passthrough_mask = np.asarray(passthrough_mask, dtype=bool)
        if self.passthrough_mask.shape != joint_ranges.shape:
            raise ValueError(
                "passthrough_mask must match the independent joint shape"
            )

    def next(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if not self.is_init:
            return super().next(x)

        delta = x - self.y
        bypass = (np.abs(delta) >= self.bypass_thresholds) | self.passthrough_mask
        self.y = np.where(bypass, x, self.y + self.alpha * delta)
        return self.y.copy()
