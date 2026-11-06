"""
Adaptive Spatial Fusion Package.
Part C of Problem Statement 1 (P1).
"""

from fusion.config import FusionConfig, FusionResult
from fusion.spatial_fusion import (
    apply_quality_awareness_guard,
    compute_smooth_candidate_weights,
    fuse_adaptive,
    fuse_hybrid,
    to_float32_rgb,
)

__all__ = [
    "FusionConfig",
    "FusionResult",
    "fuse_hybrid",
    "fuse_adaptive",
    "compute_smooth_candidate_weights",
    "apply_quality_awareness_guard",
    "to_float32_rgb",
]
