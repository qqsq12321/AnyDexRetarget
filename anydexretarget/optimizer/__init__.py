"""Optimizers for hand retargeting.

AdaptiveOptimizerAnalytical - Recommended optimizer using Huber loss + analytical gradients + NLopt SLSQP.
Uses adaptive blending between TipDirVec and FullHandVec based on pinch distance.

DexPilotOptimizer - DexPilot method with adaptive distance projection for stable grasping.

All parameters are read from YAML configuration files.
"""

from .base_optimizer import BaseOptimizer
from .utils import (
    LPFilter,
    TimingStats,
    M_TO_CM,
    CM_TO_M,
)
from .analytical_optimizer import AdaptiveOptimizerAnalytical
from .key_vector_optimizer import KeyVectorOptimizer
from .dexpilot_optimizer import DexPilotOptimizer


__all__ = [
    "BaseOptimizer",
    "AdaptiveOptimizerAnalytical",
    "KeyVectorOptimizer",
    "DexPilotOptimizer",
    "LPFilter",
    "TimingStats",
    "M_TO_CM",
    "CM_TO_M",
]
