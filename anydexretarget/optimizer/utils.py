"""Utility classes and functions for hand retargeting optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


# Unit conversion: internal computations use cm
M_TO_CM = 100.0
CM_TO_M = 0.01


def huber_loss_np(x: np.ndarray, delta: float = 2.0) -> np.ndarray:
    """Huber loss function (smooth L1 loss)."""
    abs_x = np.abs(x)
    return np.where(
        abs_x <= delta,
        0.5 * x ** 2,
        delta * (abs_x - 0.5 * delta)
    )


def huber_loss_grad_np(x: np.ndarray, delta: float = 2.0) -> np.ndarray:
    """Gradient of Huber loss w.r.t. x (numpy version)."""
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, x, delta * np.sign(x))


def huber_norm_gradient_np(residual: np.ndarray, delta: float) -> np.ndarray:
    """Exact gradient of rho_delta(||e||), including e=0; no norm epsilon.

    In the quadratic region the gradient is e itself. At the Huber boundary
    either branch has the same gradient. delta must be finite and positive.
    """
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("Huber delta must be finite and positive")
    residual = np.asarray(residual, dtype=np.float64)
    norm = np.linalg.norm(residual, axis=-1, keepdims=True)
    factor = np.ones_like(norm)
    np.divide(delta, norm, out=factor, where=norm > delta)
    return residual * factor


def stabilized_normalize_with_jacobian(z: np.ndarray, epsilon: float = 1e-8):
    """Return z/(||z||+epsilon) and its exact derivative w.r.t. z.

    z can have batch dimensions; the final dimension is the vector dimension.
    At z=0 the derivative exists and is I/epsilon. This does not make a zero
    FK segment a physically meaningful direction; it only defines the function.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Normalization epsilon must be finite and positive")
    z = np.asarray(z, dtype=np.float64)
    n = np.linalg.norm(z, axis=-1, keepdims=True)
    denominator = n + epsilon
    unit = np.divide(z, n, out=np.zeros_like(z), where=n > 0)
    outer = unit[..., :, None] * unit[..., None, :]
    jacobian = (np.eye(z.shape[-1]) - (n / denominator)[..., None] * outer) / denominator[..., None]
    return z / denominator, jacobian


@dataclass
class TimingStats:
    """Timing statistics for optimizer performance analysis."""
    preprocess_ms: float = 0.0
    fk_ms: float = 0.0
    jacobian_ms: float = 0.0
    gradient_ms: float = 0.0
    nlopt_ms: float = 0.0
    total_ms: float = 0.0
    call_count: int = 0
    iter_counts: List[int] = field(default_factory=list)
    # Per-frame iteration losses: list of lists, each inner list is losses per iteration
    iter_losses: List[List[float]] = field(default_factory=list)
    # Current frame's iteration losses (temporary storage during optimization)
    _current_iter_losses: List[float] = field(default_factory=list)

    def reset(self):
        """Reset all timing statistics."""
        self.preprocess_ms = 0.0
        self.fk_ms = 0.0
        self.jacobian_ms = 0.0
        self.gradient_ms = 0.0
        self.nlopt_ms = 0.0
        self.total_ms = 0.0
        self.call_count = 0
        self.iter_counts = []
        self.iter_losses = []
        self._current_iter_losses = []

    def start_frame(self):
        """Start recording for a new frame."""
        self._current_iter_losses = []

    def record_iter_loss(self, loss: float):
        """Record loss for current iteration."""
        self._current_iter_losses.append(loss)

    def end_frame(self, num_evals: int):
        """End recording for current frame."""
        self.iter_counts.append(num_evals)
        self.iter_losses.append(self._current_iter_losses.copy())
        self._current_iter_losses = []

    def get_last_iter_losses(self) -> List[float]:
        """Get iteration losses for the last frame."""
        if self.iter_losses:
            return self.iter_losses[-1]
        return []

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'preprocess_ms': self.preprocess_ms,
            'fk_ms': self.fk_ms,
            'jacobian_ms': self.jacobian_ms,
            'gradient_ms': self.gradient_ms,
            'nlopt_ms': self.nlopt_ms,
            'total_ms': self.total_ms,
            'call_count': self.call_count,
        }

    def get_avg(self) -> Dict[str, float]:
        """Get average timing per call."""
        if self.call_count == 0:
            return self.to_dict()
        return {
            'preprocess_ms': self.preprocess_ms / self.call_count,
            'fk_ms': self.fk_ms / self.call_count,
            'jacobian_ms': self.jacobian_ms / self.call_count,
            'gradient_ms': self.gradient_ms / self.call_count,
            'nlopt_ms': self.nlopt_ms / self.call_count,
            'total_ms': self.total_ms / self.call_count,
            'call_count': self.call_count,
        }

    def get_iter_stats(self) -> Dict[str, float]:
        """Get iteration count statistics."""
        if not self.iter_counts:
            return {}
        arr = np.array(self.iter_counts)
        return {
            'min': int(np.min(arr)),
            'max': int(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'p90': float(np.percentile(arr, 90)),
            'p99': float(np.percentile(arr, 99)),
        }


class LPFilter:
    """Low-pass filter for smoothing joint positions."""

    def __init__(self, alpha: float):
        """Initialize filter.

        Args:
            alpha: Filter coefficient (0 < alpha <= 1).
                   Smaller = smoother but more latency.
        """
        alpha = float(alpha)
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("LPFilter alpha must be finite and in (0, 1]")
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x: np.ndarray) -> np.ndarray:
        """Apply filter to new value."""
        if not self.is_init:
            self.y = x.copy()
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        """Reset filter state."""
        self.y = None
        self.is_init = False
