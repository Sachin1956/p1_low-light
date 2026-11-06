"""
Configuration and Schema for Part C: Adaptive Spatial Fusion & Quality Guard.
Adheres strictly to docs/P1_ML_EXECUTION_PLAN.md Section 4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class FusionConfig:
    """Immutable configuration for adaptive spatial fusion."""
    # Temperature for softmax normalization over candidate logits
    softmax_temperature: float = 1.0

    # Guided filtering parameters for spatial weight smoothing
    guided_radius: int = 15
    guided_eps: float = 1e-3

    # Logit sensitivity factors
    bright_threshold: float = 0.70
    dark_threshold: float = 0.35
    mid_center: float = 0.45

    # Quality awareness & clipping guard
    enable_quality_guard: bool = True
    clipping_ceiling: float = 0.98

    # Version tracking
    branch_version: str = "Part_C_v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_hash(self) -> str:
        canonical_json = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]


@dataclass
class FusionResult:
    """Standard result structure for adaptive hybrid fusion."""
    image: np.ndarray                    # HxWx3 uint8 sRGB hybrid output
    weights: np.ndarray                  # HxWx3 float32 spatial weights [W_o, W_c, W_d]
    diagnostics: Dict[str, np.ndarray]   # Aligned diagnostic maps
    params: Dict[str, Any]               # Runtime summary parameters
    elapsed_ms: float                    # Measured inference execution latency in ms
    config_hash: str                     # Config digest
