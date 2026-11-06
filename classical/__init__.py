"""
Classical Low-Light Image Restoration and Enhancement Package.
Part A of Problem Statement 1 (P1).
"""

from classical.config import ClassicalConfig, EnhancementResult, PipelineMode
from classical.features import (
    ImageAnalysisFeatures,
    compute_diagnostic_luminance,
    compute_edge_gradient_map,
    compute_highlight_protection_mask,
    estimate_flat_shadow_noise,
    extract_features_and_maps,
)
from classical.filters import (
    apply_conservative_color_correction,
    apply_edge_preserving_denoise,
    apply_highlight_guard_blend,
    apply_luminance_clahe,
    apply_retinex_exposure_boost,
    decompose_and_refine_illumination,
    fast_guided_filter_gray,
)
from classical.mappings import SelectedParameters, clip01, map_features_to_parameters
from classical.pipeline import enhance, run_batch_inference, validate_input_image

__all__ = [
    "enhance",
    "run_batch_inference",
    "validate_input_image",
    "ClassicalConfig",
    "EnhancementResult",
    "PipelineMode",
    "ImageAnalysisFeatures",
    "SelectedParameters",
    "extract_features_and_maps",
    "map_features_to_parameters",
    "fast_guided_filter_gray",
    "decompose_and_refine_illumination",
    "apply_retinex_exposure_boost",
    "apply_edge_preserving_denoise",
    "apply_luminance_clahe",
    "apply_conservative_color_correction",
    "apply_highlight_guard_blend",
]
