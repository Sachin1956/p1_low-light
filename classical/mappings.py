"""
Monotone Parameter Mappings for Adaptive Classical Low-Light Enhancement.
Implements the exact formulas defined in Gate 2 of docs/P1_PART_A_IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from classical.config import ClassicalConfig, PipelineMode
from classical.features import ImageAnalysisFeatures


def clip01(x: float) -> float:
    """Bounded clamp function c(x) = min(1.0, max(0.0, x))."""
    return min(1.0, max(0.0, float(x)))


@dataclass(frozen=True)
class SelectedParameters:
    """Actual resolved parameter bundle for the enhancement pipeline."""
    gamma: float
    gain_cap: float
    denoise_strength: float
    clahe_clip_limit: float
    applied_mode: str
    noise_adapted: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gamma": round(self.gamma, 4),
            "gain_cap": round(self.gain_cap, 4),
            "denoise_strength": round(self.denoise_strength, 4),
            "clahe_clip_limit": round(self.clahe_clip_limit, 4),
            "applied_mode": self.applied_mode,
            "noise_adapted": self.noise_adapted,
        }


def map_features_to_parameters(
    features: ImageAnalysisFeatures,
    config: ClassicalConfig,
) -> SelectedParameters:
    """
    Resolve operational parameters according to configuration mode.

    - In E0_BASELINE: fixed baseline gamma (e.g. 0.80) or identity.
    - In E1_FIXED: constant preregistered values for gamma, gain_cap, denoise, and CLAHE.
    - In E2_ADAPTIVE: concrete monotone mappings driven by input luminance, highlights, and noise.
    """
    if config.mode == PipelineMode.E0_BASELINE:
        return SelectedParameters(
            gamma=config.e0_gamma,
            gain_cap=1.0,
            denoise_strength=0.0,
            clahe_clip_limit=0.0,
            applied_mode="E0_BASELINE",
            noise_adapted=False,
        )

    if config.mode == PipelineMode.E1_FIXED:
        return SelectedParameters(
            gamma=config.fixed_gamma,
            gain_cap=config.fixed_gain_cap,
            denoise_strength=config.fixed_denoise_strength if config.enable_denoise else 0.0,
            clahe_clip_limit=config.fixed_clahe_clip if config.enable_clahe else 0.0,
            applied_mode="E1_FIXED",
            noise_adapted=False,
        )

    # Mode E2_ADAPTIVE:
    m = features.median_luminance
    h = features.highlight_fraction
    n = features.noise_estimate

    # 1. Dark lift factor d = c((0.55 - m) / 0.45) in [0, 1]
    d = clip01((0.55 - m) / 0.45)

    # 2. Retinex illumination gamma = 1.0 - 0.40 * d in [0.60, 1.00]
    gamma = 1.0 - 0.40 * d

    # 3. Maximum illumination gain cap = 2.0 + 3.0 * d * (1.0 - c(h / 0.10)) in [2.0, 5.0]
    gain_cap = 2.0 + 3.0 * d * (1.0 - clip01(h / 0.10))

    # 4. Noise adaptation (guarded by patch validity flag q >= 0.02)
    if features.noise_is_valid and config.enable_denoise:
        z = clip01((n - 0.01) / 0.05)
        denoise_strength = config.denoise_h_min + (config.denoise_h_max - config.denoise_h_min) * z
        clahe_clip = 2.2 - 0.8 * z
        noise_adapted = True
    else:
        # Fall back to E1 fixed values when flat shadow patch noise is invalid
        denoise_strength = config.fixed_denoise_strength if config.enable_denoise else 0.0
        clahe_clip = config.fixed_clahe_clip if config.enable_clahe else 0.0
        noise_adapted = False

    if not config.enable_clahe:
        clahe_clip = 0.0

    return SelectedParameters(
        gamma=gamma,
        gain_cap=gain_cap,
        denoise_strength=denoise_strength,
        clahe_clip_limit=clahe_clip,
        applied_mode="E2_ADAPTIVE",
        noise_adapted=noise_adapted,
    )
