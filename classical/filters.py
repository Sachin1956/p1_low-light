"""
Image processing operators for Classical Low-Light Enhancement.
Implements Guided Filter, Retinex Illumination Decomposition & Refinement,
Edge-Preserving Denoising, Luminance-only CLAHE, and Highlight Protection.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def fast_guided_filter_gray(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 15,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Fast O(N) guided filter implementation (He et al., ECCV 2010).
    Works on float32 2D single-channel arrays in [0, 1].

    If cv2.ximgproc is available, delegates to its optimized C++ implementation.
    Otherwise uses an exact separable box-filter NumPy/OpenCV implementation.
    """
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        return cv2.ximgproc.guidedFilter(
            guide=guide.astype(np.float32),
            src=src.astype(np.float32),
            radius=radius,
            eps=eps,
        )

    # Pure OpenCV boxFilter implementation
    guide_f = guide.astype(np.float32)
    src_f = src.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(guide_f, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(src_f, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_Ip = cv2.boxFilter(guide_f * src_f, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide_f * guide_f, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    q = mean_a * guide_f + mean_b
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def decompose_and_refine_illumination(
    rgb_float: np.ndarray,
    luminance_diagnostic: np.ndarray,
    radius: int = 15,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute initial illumination T0 = max_c(I_c) and refine via guided filtering
    using diagnostic luminance Y as the edge-preserving guidance.

    Returns:
        T0: Raw illumination map HxW in [0, 1]
        T: Refined illumination map HxW in [0, 1]
    """
    T0 = np.max(rgb_float, axis=-1).astype(np.float32)
    T_refined = fast_guided_filter_gray(
        guide=luminance_diagnostic,
        src=T0,
        radius=radius,
        eps=eps,
    )
    # Illumination must be positive and bounded below by a small floor
    T_refined = np.maximum(T_refined, 0.01).astype(np.float32)
    return T0, T_refined


def apply_retinex_exposure_boost(
    rgb_float: np.ndarray,
    refined_illumination: np.ndarray,
    gamma: float = 0.75,
    gain_cap: float = 3.5,
    tau_floor: float = 0.02,
) -> np.ndarray:
    """
    Apply bounded illumination compensation (LIME-inspired Retinex proxy):
    T' = max(T, tau)
    gain(x, y) = clip( (T')^(gamma - 1.0), 1.0, gain_cap )
    J_c = clip(I_c * gain, 0.0, 1.0)
    """
    T_prime = np.maximum(refined_illumination, tau_floor)
    # When gamma < 1.0 and T' < 1.0, (T')^(gamma - 1) > 1.0 (illumination lift)
    raw_gain = np.power(T_prime, gamma - 1.0)
    capped_gain = np.clip(raw_gain, 1.0, gain_cap).astype(np.float32)

    enhanced = rgb_float * capped_gain[..., np.newaxis]
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def apply_edge_preserving_denoise(
    rgb_float: np.ndarray,
    denoise_strength: float,
) -> np.ndarray:
    """
    Apply edge-preserving bilateral filtering to suppress shadow noise
    amplified during illumination expansion.
    """
    if denoise_strength <= 1e-4:
        return rgb_float

    # Convert to uint8 for OpenCV bilateral filter
    img_u8 = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)
    sigma_color = float(np.clip(denoise_strength * 255.0, 5.0, 60.0))
    sigma_space = 3.0

    denoised_u8 = cv2.bilateralFilter(
        img_u8,
        d=5,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
        borderType=cv2.BORDER_REFLECT,
    )
    return (denoised_u8.astype(np.float32) / 255.0)


def apply_luminance_clahe(
    rgb_float: np.ndarray,
    clip_limit: float = 1.8,
    grid_size: int = 8,
    highlight_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    strictly to the Luminance (L*) channel in CIELAB color space.
    Preserves chromaticity (a*, b*) to avoid artificial color casts.
    """
    if clip_limit <= 1e-4:
        return rgb_float

    img_u8 = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(grid_size), int(grid_size)),
    )
    l_clahe = clahe.apply(l_chan)

    # In highlight regions, blend back original L channel to avoid harsh clipping
    if highlight_mask is not None:
        l_orig_f = l_chan.astype(np.float32)
        l_clahe_f = l_clahe.astype(np.float32)
        l_final = (1.0 - highlight_mask) * l_clahe_f + highlight_mask * l_orig_f
        l_chan = np.clip(l_final, 0, 255).astype(np.uint8)
    else:
        l_chan = l_clahe

    lab_merged = cv2.merge([l_chan, a_chan, b_chan])
    rgb_out_u8 = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)
    return (rgb_out_u8.astype(np.float32) / 255.0)


def apply_conservative_color_correction(
    rgb_float: np.ndarray,
    saturation_factor: float = 1.10,
) -> np.ndarray:
    """
    Gentle saturation and contrast adjustment to restore natural vibrancy
    without introducing unnatural color drift.
    """
    if abs(saturation_factor - 1.0) < 1e-3:
        return rgb_float

    img_u8 = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Scale saturation
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_factor, 0.0, 255.0)
    hsv_u8 = np.clip(hsv, 0, 255).astype(np.uint8)
    rgb_corrected = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB)
    return (rgb_corrected.astype(np.float32) / 255.0)


def apply_highlight_guard_blend(
    input_rgb_float: np.ndarray,
    enhanced_rgb_float: np.ndarray,
    highlight_mask: np.ndarray,
) -> np.ndarray:
    """
    Soft-blend original input in naturally bright areas to prevent light blowout:
    I_final = (1 - M_highlight) * I_enhanced + M_highlight * I_input
    """
    mask_3d = highlight_mask[..., np.newaxis]
    fused = (1.0 - mask_3d) * enhanced_rgb_float + mask_3d * input_rgb_float
    return np.clip(fused, 0.0, 1.0).astype(np.float32)
