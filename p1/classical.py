"""
Classical CV restoration branch (Part A) and feature analysis maps.

Implements:
  - Guided filter illumination estimation
  - Adaptive luminance CLAHE in LAB space
  - Noise-aware bilateral filtering
  - Exposure & contrast correction
  - Analysis maps: illumination, high-frequency noise estimate, edge confidence
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np


def _guided_filter_gray(guide: np.ndarray, src: np.ndarray, radius: int = 15, eps: float = 1e-3) -> np.ndarray:
    """Fast edge-preserving guided filter (He et al., TPAMI 2013) on 2D float32 arrays."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    return mean_a * guide + mean_b


def compute_analysis_maps(rgb_f32: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute input-derived feature maps:
      - 'illumination': estimated continuous scene illumination in [0, 1]
      - 'noise_map': local high-frequency standard deviation estimate in [0, 1]
      - 'edge_map': Sobel edge magnitude in [0, 1]
      - 'shadow_mask': binary/continuous mask identifying dark shadow regions
    """
    lum = (0.299 * rgb_f32[:, :, 0] + 0.587 * rgb_f32[:, :, 1] + 0.114 * rgb_f32[:, :, 2]).astype(np.float32)
    max_c = np.max(rgb_f32, axis=2).astype(np.float32)

    # Initial coarse illumination + guided filter refinement
    illum_coarse = cv2.GaussianBlur(max_c, (31, 31), 8.0)
    illumination = np.clip(_guided_filter_gray(lum, illum_coarse, radius=19, eps=2e-2), 0.02, 1.0)

    # High frequency noise map via residual to local Gaussian
    local_smooth = cv2.GaussianBlur(lum, (7, 7), 1.5)
    high_freq = np.abs(lum - local_smooth)
    noise_map = cv2.GaussianBlur(high_freq, (15, 15), 3.0)
    noise_norm = np.clip(noise_map / 0.15, 0.0, 1.0)

    # Normalized Sobel edge map
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx * gx + gy * gy)
    edge_norm = np.clip(edge_mag / (np.percentile(edge_mag, 98) + 1e-5), 0.0, 1.0)

    shadow_mask = np.clip((0.25 - illumination) / 0.25, 0.0, 1.0)

    return {
        "illumination": illumination.astype(np.float32),
        "noise_map": noise_norm.astype(np.float32),
        "edge_map": edge_norm.astype(np.float32),
        "shadow_mask": shadow_mask.astype(np.float32),
    }


def run_classical_pipeline(
    rgb_uint8: np.ndarray,
    adaptive: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any], float]:
    """
    Execute Classical CV pipeline (Part A).
    Returns (restored_uint8, maps, params, elapsed_ms).
    """
    start_time = time.perf_counter()

    rgb_f = (rgb_uint8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    maps = compute_analysis_maps(rgb_f)
    illum = maps["illumination"]
    noise = maps["noise_map"]

    mean_illum = float(np.mean(illum))
    mean_noise = float(np.mean(noise))

    # Parameter selection
    if adaptive:
        # Bounded adaptive gamma
        target_mean = 0.45 + 0.10 * (1.0 - np.clip(mean_noise, 0.0, 1.0))
        gamma = float(np.clip(np.log(target_mean) / np.log(max(mean_illum, 0.03)), 0.35, 0.85))
        # CLAHE clip limit adapted to noise
        clahe_clip = float(np.clip(2.5 - 1.5 * mean_noise, 0.8, 3.0))
        # Bilateral filter sigma adapted to noise
        bilat_d = 7 if mean_noise > 0.3 else 5
        bilat_sigma_color = float(np.clip(0.05 + 0.15 * mean_noise, 0.05, 0.25))
    else:
        gamma = 0.55
        clahe_clip = 2.0
        bilat_d = 5
        bilat_sigma_color = 0.10

    # 1. Retinex illumination correction
    illum_gamma = np.power(illum, 1.0 - gamma)
    reflectance = np.clip(rgb_f / (illum_gamma[:, :, None] + 0.03), 0.0, 1.0)

    # 2. Luminance CLAHE in LAB space
    ref_u8 = np.clip(reflectance * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(ref_u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    # 3. Bilateral denoising in shadow regions
    denoised = cv2.bilateralFilter(
        enhanced_rgb.astype(np.float32),
        d=bilat_d,
        sigmaColor=bilat_sigma_color,
        sigmaSpace=7,
    )
    # Blend denoised output more strongly in dark/noisy areas
    blend_w = np.clip(maps["shadow_mask"] * (0.5 + 0.5 * noise), 0.0, 1.0)[:, :, None]
    final_f = np.clip((1.0 - blend_w) * enhanced_rgb + blend_w * denoised, 0.0, 1.0)
    final_u8 = np.clip(np.round(final_f * 255.0), 0, 255).astype(np.uint8)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    params = {
        "adaptive": adaptive,
        "mean_illum": round(mean_illum, 4),
        "mean_noise": round(mean_noise, 4),
        "selected_gamma": round(gamma, 4),
        "selected_clahe_clip": round(clahe_clip, 2),
        "bilat_d": bilat_d,
        "bilat_sigma_color": round(bilat_sigma_color, 4),
    }

    return final_u8, maps, params, elapsed_ms
