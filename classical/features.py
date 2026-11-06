"""
Feature Extraction and Analysis Maps for Classical Low-Light Enhancement.
Implements diagnostic luminance, highlight detection, edge estimation,
and flat-shadow patch noise estimation with explicit validity checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageAnalysisFeatures:
    """Extracted scalar metrics and diagnostic flags from input image."""
    median_luminance: float              # m: median(Y)
    highlight_fraction: float            # h: fraction(Y >= 0.90)
    noise_estimate: float                # n: flat-shadow noise proxy (MAD-based)
    shadow_patch_fraction: float         # q: fraction of valid flat shadow pixels
    noise_is_valid: bool                 # True if q >= 0.02 and sufficient patches
    mean_luminance: float                # mean(Y)
    dynamic_range: float                 # max(Y) - min(Y)


def compute_diagnostic_luminance(rgb_float: np.ndarray) -> np.ndarray:
    """
    Compute standard ITU-R BT.709 sRGB diagnostic luminance:
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    """
    if rgb_float.ndim != 3 or rgb_float.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB float array, got shape {rgb_float.shape}")
    r = rgb_float[..., 0]
    g = rgb_float[..., 1]
    b = rgb_float[..., 2]
    return np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0.0, 1.0).astype(np.float32)


def compute_edge_gradient_map(luminance: np.ndarray) -> np.ndarray:
    """
    Compute normalized Sobel edge gradient magnitude map in [0, 1].
    """
    gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    # Normalize by theoretical max for Sobel on [0, 1] (sqrt(4^2 + 4^2) = ~5.66)
    grad_norm = np.clip(grad_mag / 4.0, 0.0, 1.0).astype(np.float32)
    return grad_norm


def estimate_flat_shadow_noise(
    luminance: np.ndarray,
    edge_map: np.ndarray,
    min_patch_fraction: float = 0.02,
) -> Tuple[float, float, bool, np.ndarray]:
    """
    Estimate high-frequency noise proxy strictly on non-clipped, low-gradient shadow patches.
    Avoids mistaking texture or highlights for noise.

    Returns:
        noise_n: float in [0, 1]
        shadow_q: float fraction of valid pixels
        is_valid: bool (True if shadow_q >= min_patch_fraction)
        valid_mask: HxW bool array of valid flat shadow pixels
    """
    h, w = luminance.shape
    total_pixels = h * w

    # Non-clipped dark shadow region: [0.015, 0.35]
    shadow_mask = (luminance >= 0.015) & (luminance <= 0.35)

    # Flat patch condition: Sobel gradient below 0.05
    flat_mask = edge_map < 0.05

    valid_mask = shadow_mask & flat_mask
    valid_count = int(np.sum(valid_mask))
    shadow_q = float(valid_count / max(1, total_pixels))

    if shadow_q < min_patch_fraction or valid_count < 64:
        # Insufficient flat shadow region to compute a reliable noise statistic
        return 0.0, shadow_q, False, valid_mask

    # Extract high-pass residual via 3x3 Laplacian on valid pixels
    laplacian = cv2.Laplacian(luminance, cv2.CV_32F, ksize=3)
    valid_lap = laplacian[valid_mask]

    # Median Absolute Deviation (MAD) of the high-pass residual:
    # sigma_hat = median(|x - median(x)|) / 0.6745
    med = np.median(valid_lap)
    mad = np.median(np.abs(valid_lap - med))
    sigma_hat = float(mad / 0.6745)

    # Bound to [0.0, 1.0]
    noise_n = float(np.clip(sigma_hat, 0.0, 1.0))
    return noise_n, shadow_q, True, valid_mask


def compute_highlight_protection_mask(
    luminance: np.ndarray,
    threshold: float = 0.85,
    softness: float = 0.10,
) -> np.ndarray:
    """
    Smooth highlight protection mask in [0, 1].
    Values approach 1.0 in already bright/highlight regions (to protect against blowouts),
    and 0.0 in dark/mid-tone areas.
    """
    low_bound = max(0.0, threshold - softness)
    high_bound = min(1.0, threshold + softness)
    span = max(1e-5, high_bound - low_bound)
    mask = np.clip((luminance - low_bound) / span, 0.0, 1.0)
    # Smooth Hermite interpolation (smoothstep)
    smooth_mask = mask * mask * (3.0 - 2.0 * mask)
    return smooth_mask.astype(np.float32)


def extract_features_and_maps(
    rgb_float: np.ndarray,
    highlight_threshold: float = 0.85,
) -> Tuple[ImageAnalysisFeatures, Dict[str, np.ndarray]]:
    """
    Extract all diagnostic maps and summary features from an input float RGB image.
    All returned maps are aligned HxW float32 arrays in [0, 1].
    """
    # 1. Diagnostic luminance
    Y = compute_diagnostic_luminance(rgb_float)

    # 2. Raw illumination estimate: T0 = max(R, G, B)
    T0 = np.max(rgb_float, axis=-1).astype(np.float32)

    # 3. Edge gradient map
    edge_map = compute_edge_gradient_map(Y)

    # 4. Flat shadow noise estimate and validity mask
    noise_n, shadow_q, is_valid, valid_mask = estimate_flat_shadow_noise(Y, edge_map)

    # 5. Highlight protection mask
    highlight_mask = compute_highlight_protection_mask(Y, threshold=highlight_threshold)

    # 6. Scalar feature statistics
    m = float(np.median(Y))
    h_frac = float(np.mean(Y >= 0.90))
    mean_lum = float(np.mean(Y))
    dyn_range = float(np.max(Y) - np.min(Y))

    features = ImageAnalysisFeatures(
        median_luminance=m,
        highlight_fraction=h_frac,
        noise_estimate=noise_n,
        shadow_patch_fraction=shadow_q,
        noise_is_valid=is_valid,
        mean_luminance=mean_lum,
        dynamic_range=dyn_range,
    )

    maps: Dict[str, np.ndarray] = {
        "illumination_raw": T0,
        "luminance_diagnostic": Y,
        "edge_map": edge_map,
        "highlight_mask": highlight_mask,
        "flat_shadow_validity": valid_mask.astype(np.float32),
    }

    return features, maps
