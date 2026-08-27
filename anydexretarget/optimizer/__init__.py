"""Optimizers for hand retargeting.

AdaptiveOptimizerAnalytical - Recommended optimizer using Huber loss + analytical gradients + NLopt SLSQP.
Uses adaptive blending between TipDirVec and FullHandVec based on pinch distance.

All parameters are read from YAML configuration files.
"""

from .analytical_optimizer import AdaptiveOptimizerAnalytical
from .analytical_optimizer_v2 import AdaptiveOptimizerAnalyticalV2
from .base_optimizer import BaseOptimizer
from .filter_v2 import AdaptiveLPFilterV2
from .key_vector_optimizer import KeyVectorOptimizer
from .utils import CM_TO_M, M_TO_CM, LPFilter, TimingStats

__all__ = [
    "CM_TO_M",
    "M_TO_CM",
    "AdaptiveLPFilterV2",
    "AdaptiveOptimizerAnalytical",
    "AdaptiveOptimizerAnalyticalV2",
    "BaseOptimizer",
    "KeyVectorOptimizer",
    "LPFilter",
    "TimingStats",
]
