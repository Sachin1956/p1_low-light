"""
Adaptive Spatial Fusion and Quality Guard Module.
Part C of Problem Statement 1 (P1).

Combines:
  - Original input I_o (preserves well-exposed lights, lamps, and sky)
  - Classical candidate I_c (enhances local contrast, fine edges, and structure)
  - Deep learning candidate I_d (powerful in noisy darkness and deep shadows)

Using smooth, input-guided spatial logits normalized via softmax:
  W_o(x,y) + W_c(x,y) + W_d(x,y) = 1.0
  I_hybrid(x,y) = W_o(x,y)*I_o(x,y) + W_c(x,y)*I_c(x,y) + W_d(x,y)*I_d(x,y)
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from classical.filters import fast_guided_filter_gray
from fusion.config import FusionConfig, FusionResult


def to_float32_rgb(arr: np.ndarray) -> np.ndarray:
    """Normalize uint8 [0, 255] or float [0, 1] into contiguous float32 [0.0, 1.0]."""
    if arr is None:
        raise ValueError("Input array cannot be None.")
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.max() > 1.05:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def compute_smooth_candidate_weights(
    low_f: np.ndarray,
    config: FusionConfig,
    precomputed_maps: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute spatially variable, normalized scalar weights for:
    [W_o (Original), W_c (Classical), W_d (Deep)].

    Returns:
        weights: HxWx3 float32 array where sum(axis=2) == 1.0 everywhere.
        diagnostics: dict of diagnostic maps.
    """
    h, w, _ = low_f.shape

    # 1. Diagnostic luminance Y
    if precomputed_maps and "luminance_diagnostic" in precomputed_maps:
        Y = precomputed_maps["luminance_diagnostic"]
    else:
        Y = np.clip(0.2126 * low_f[..., 0] + 0.7152 * low_f[..., 1] + 0.0722 * low_f[..., 2], 0.0, 1.0)

    # 2. Edge / gradient map
    if precomputed_maps and "edge_map" in precomputed_maps:
        edge_map = precomputed_maps["edge_map"]
    else:
        gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
        edge_map = np.clip(np.sqrt(gx * gx + gy * gy) / 4.0, 0.0, 1.0)

    # 3. Calculate candidate logits
    # Logit for Original (O): strongly preferred in high luminance highlights (Y >= 0.70)
    # Penalized in deep darkness.
    logit_o = 4.0 * np.maximum(0.0, (Y - config.bright_threshold) / (1.0 - config.bright_threshold + 1e-4)) \
            - 2.5 * np.maximum(0.0, (config.dark_threshold - Y) / config.dark_threshold)

    # Logit for Deep (D): dominant in dark noisy regions (Y < 0.35)
    # Penalized in bright highlight regions to avoid deep learning color hallucinations.
    logit_d = 3.5 * np.maximum(0.0, (config.dark_threshold - Y) / config.dark_threshold) \
            - 3.0 * np.maximum(0.0, (Y - config.bright_threshold) / (1.0 - config.bright_threshold + 1e-4))

    # Logit for Classical (C): balanced contrast and edge recovery in midtones and high-frequency boundaries
    dist_mid = np.abs(Y - config.mid_center) / config.mid_center
    logit_c = 2.0 * np.maximum(0.0, 1.0 - dist_mid) + 2.5 * edge_map

    # 4. Input-guided spatial smoothing of logits (eliminates seams and boundary halos)
    r = config.guided_radius
    eps = config.guided_eps
    smooth_lo = fast_guided_filter_gray(Y, logit_o, radius=r, eps=eps)
    smooth_lc = fast_guided_filter_gray(Y, logit_c, radius=r, eps=eps)
    smooth_ld = fast_guided_filter_gray(Y, logit_d, radius=r, eps=eps)

    # 5. Numerically stable Softmax over candidates
    logits_stack = np.stack([smooth_lo, smooth_lc, smooth_ld], axis=-1) / max(1e-4, config.softmax_temperature)
    max_logit = np.max(logits_stack, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_stack - max_logit)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    weights = (exp_logits / np.maximum(1e-7, sum_exp)).astype(np.float32)

    diagnostics = {
        "luminance_diagnostic": Y,
        "edge_map": edge_map,
        "weight_original": weights[..., 0],
        "weight_classical": weights[..., 1],
        "weight_deep": weights[..., 2],
    }

    return weights, diagnostics


def apply_quality_awareness_guard(
    low_f: np.ndarray,
    fused_f: np.ndarray,
    clipping_ceiling: float = 0.98,
) -> np.ndarray:
    """
    Quality guard: detects artificial highlight blowouts introduced by enhancement.
    If fused output exceeds clipping ceiling while original was not near clipping,
    gently compress the highlights with a soft knee.
    """
    out = np.copy(fused_f)
    # Check if original was well below highlight threshold
    orig_not_bright = np.max(low_f, axis=-1) < 0.85

    # Identify clipped pixels that were NOT originally bright
    blowout_mask = (out > clipping_ceiling) & orig_not_bright[..., np.newaxis]

    if np.any(blowout_mask):
        # Soft-knee compression: knee at 0.90
        excess = out[blowout_mask] - 0.90
        compressed = 0.90 + excess / (1.0 + excess * 4.0)
        out[blowout_mask] = np.clip(compressed, 0.0, clipping_ceiling)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fuse_hybrid(
    low_rgb: np.ndarray,
    classical_rgb: np.ndarray,
    deep_rgb: np.ndarray,
    config: Optional[FusionConfig] = None,
    precomputed_maps: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Standard interface expected by evaluate.py:
    fuse_hybrid(low_rgb, classical_rgb, deep_rgb) -> hybrid_rgb (HxWx3 float32 or uint8).
    """
    res = fuse_adaptive(low_rgb, classical_rgb, deep_rgb, config, precomputed_maps)
    return res.image


def fuse_adaptive(
    low_rgb: np.ndarray,
    classical_rgb: np.ndarray,
    deep_rgb: np.ndarray,
    config: Optional[FusionConfig] = None,
    precomputed_maps: Optional[Dict[str, np.ndarray]] = None,
) -> FusionResult:
    """
    Full-featured adaptive spatial fusion with diagnostics and timing.
    """
    t_start = time.perf_counter()

    if config is None:
        config = FusionConfig()

    low_f = to_float32_rgb(low_rgb)
    classical_f = to_float32_rgb(classical_rgb)
    deep_f = to_float32_rgb(deep_rgb)

    if low_f.shape != classical_f.shape or low_f.shape != deep_f.shape:
        raise ValueError(
            f"Shape mismatch: low {low_f.shape}, classical {classical_f.shape}, deep {deep_f.shape}"
        )

    # Compute smooth candidate weights
    weights, diagnostics = compute_smooth_candidate_weights(
        low_f, config, precomputed_maps=precomputed_maps
    )

    # Spatial Fusion: I_hybrid = W_o * I_o + W_c * I_c + W_d * I_d
    w_o = weights[..., 0:1]
    w_c = weights[..., 1:2]
    w_d = weights[..., 2:3]

    fused_f = w_o * low_f + w_c * classical_f + w_d * deep_f

    # Apply quality guard
    if config.enable_quality_guard:
        fused_f = apply_quality_awareness_guard(
            low_f, fused_f, clipping_ceiling=config.clipping_ceiling
        )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # Output uint8 sRGB image
    out_u8 = np.clip(np.round(fused_f * 255.0), 0, 255).astype(np.uint8)

    params = {
        "mean_weight_original": float(np.mean(weights[..., 0])),
        "mean_weight_classical": float(np.mean(weights[..., 1])),
        "mean_weight_deep": float(np.mean(weights[..., 2])),
        "quality_guard_enabled": config.enable_quality_guard,
    }

    return FusionResult(
        image=out_u8,
        weights=weights,
        diagnostics=diagnostics,
        params=params,
        elapsed_ms=float(elapsed_ms),
        config_hash=config.compute_hash(),
    )
