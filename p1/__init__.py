"""
P1 Low-Light Image Restoration and Enhancement Package.

Shared interface (frozen per docs/P1_ML_EXECUTION_PLAN.md):
    enhance(rgb_uint8: np.ndarray, config: dict | EnhancementConfig) -> EnhancementResult
"""

from .interface import (
    EnhancementConfig,
    EnhancementResult,
    compute_bytes_sha256,
    compute_config_hash,
     compute_file_sha256,
    enhance,
    load_oriented_rgb_uint8,
)

__all__ = [
    "EnhancementConfig",
    "EnhancementResult",
    "compute_bytes_sha256",
    "compute_config_hash",
    "compute_file_sha256",
    "enhance",
    "load_oriented_rgb_uint8",
]
