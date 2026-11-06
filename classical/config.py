"""
Configuration and Result schema for Classical Low-Light Enhancement.
Adheres strictly to the P1 ML Execution Plan and Agent Contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class PipelineMode(str, Enum):
    """Execution mode for the classical pipeline."""
    E0_BASELINE = "E0_BASELINE"          # Identity or fixed gamma
    E1_FIXED = "E1_FIXED"                # Fixed classical (Retinex + Guided + CLAHE)
    E2_ADAPTIVE = "E2_ADAPTIVE"          # Input-adaptive settings based on luminance and noise


@dataclass(frozen=True)
class ClassicalConfig:
    """Immutable configuration for classical low-light enhancement."""
    mode: PipelineMode = PipelineMode.E2_ADAPTIVE

    # E0 Baseline settings
    e0_gamma: float = 0.80

    # Illumination estimation & guided refinement (LIME-inspired proxy)
    illumination_refinement_radius: int = 15
    illumination_refinement_eps: float = 1e-3
    tau_floor: float = 0.02              # Epsilon floor to stabilize division
    fixed_gamma: float = 0.75            # Fixed gamma for E1
    fixed_gain_cap: float = 3.5          # Maximum amplification cap for E1

    # Denoising settings
    enable_denoise: bool = True
    fixed_denoise_strength: float = 0.025
    denoise_h_min: float = 0.010
    denoise_h_max: float = 0.050

    # CLAHE settings (Luminance channel only)
    enable_clahe: bool = True
    fixed_clahe_clip: float = 1.8
    clahe_grid_size: int = 8             # 8x8 contextual tiles

    # Highlight & structure protection
    enable_highlight_guard: bool = True
    highlight_threshold: float = 0.85
    highlight_softness: float = 0.10

    # Color & contrast correction
    enable_color_correction: bool = True
    saturation_factor: float = 1.10

    # Version tracking
    branch_version: str = "Part_A_v1.0"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d

    def compute_hash(self) -> str:
        """Deterministic SHA256 hex digest of the configuration."""
        canonical_json = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]


@dataclass
class EnhancementResult:
    """
    Shared interface standard output:
    enhance(rgb_uint8: HxWx3, config) -> EnhancementResult
    """
    image: np.ndarray                    # HxWx3 uint8 sRGB restored image
    maps: Dict[str, np.ndarray]          # Named HxW float32 feature/diagnostic maps
    params: Dict[str, Any]               # Actual selected runtime parameters
    elapsed_ms: float                    # Measured inference execution time in ms
    branch_version: str                  # Branch/code version identifier
    config_hash: str                     # SHA256 digest of runtime config
    checkpoint_hash: Optional[str] = None # None for Classical CV branch
