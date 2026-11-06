"""
Evaluation Metrics Package for P1: Robust Low-Light Image Restoration and Enhancement.

Implements all mandatory P1 evaluation parameters in the exact order specified by the
Computer Vision Hackathon 2026 document:
    1. PSNR & SSIM (when ground-truth references are available)
    2. LPIPS for perceptual quality
    3. NIQE (Natural Image Quality Evaluator) no-reference quality metric
    4. Processing latency (ms and seconds)
    5. Noise suppression (Wavelet MAD shadow noise reduction) & Edge preservation (EPI)
"""

from .metrics import (
    EvaluationResult,
    LowLightEvaluator,
    compute_edge_preservation_index,
    compute_lpips,
    compute_niqe,
    compute_noise_suppression,
    compute_psnr_ssim,
    estimate_shadow_noise_mad,
    measure_latency,
    rgb_to_luminance,
    to_float_rgb,
)
from .report import (
    format_ablation_markdown,
    format_comparative_markdown,
    generate_eval_markdown_file,
)

__all__ = [
    "EvaluationResult",
    "LowLightEvaluator",
    "compute_psnr_ssim",
    "compute_lpips",
    "compute_niqe",
    "measure_latency",
    "estimate_shadow_noise_mad",
    "compute_noise_suppression",
    "compute_edge_preservation_index",
    "rgb_to_luminance",
    "to_float_rgb",
    "format_comparative_markdown",
    "format_ablation_markdown",
    "generate_eval_markdown_file",
]
