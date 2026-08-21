"""Neural network for direct calibration-parameter prediction."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - lets non-training utilities import safely.
    torch = None
    nn = None

SEGMENT_MIN = 0.3
SEGMENT_MAX = 3.5
PINCH_MIN = 0.5
PINCH_MAX = 2.5


if nn is not None:
    class CalibrationMLP(nn.Module):
        """MLP that predicts 20 segment_scaling values and one pinch_scaling."""

        def __init__(self, input_dim: int, hidden=(256, 256, 128)):
            super().__init__()
            layers = []
            prev = int(input_dim)
            for width in hidden:
                layers.extend([nn.Linear(prev, int(width)), nn.ReLU()])
                prev = int(width)
            layers.append(nn.Linear(prev, 21))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            raw = self.net(x)
            scaled = torch.sigmoid(raw)
            segment = SEGMENT_MIN + scaled[..., :20] * (SEGMENT_MAX - SEGMENT_MIN)
            pinch = PINCH_MIN + scaled[..., 20:] * (PINCH_MAX - PINCH_MIN)
            return torch.cat([segment, pinch], dim=-1)
else:
    class CalibrationMLP:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to use CalibrationMLP")
