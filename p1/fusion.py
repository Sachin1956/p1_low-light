"""
Adaptive Spatial Fusion and Quality Guard (Part C).

Implements:
  - Input-guided spatial logit fusion over Original (O), Classical (C), and Deep (D)
  - Softmax normalized weights (nonnegative, sum to 1.0)
  - Guided filter weight smoothing to eliminate boundary seams
  - Quality Guard: highlight clipping preservation & soft-knee blend back
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .classical import _guided_filter_gray


def adaptive_fusion(
    img_orig_u8: np.ndarray,
    img_class_u8: np.ndarray,
    img_deep_u8: np.ndarray,
    maps: Dict[str, np.ndarray],
    mode: str = "hybrid_guarded",
    quality_guard: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any], float]:
    """
    Spatially fuse Original (O), Classical (C), and Deep (D) images.
    Returns (fused_uint8, fusion_maps, params, elapsed_ms).
    """
    start_time = time.perf_counter()

    orig_f = img_orig_u8.astype(np.float32) / 255.0
    class_f = img_class_u8.astype(np.float32) / 255.0
    deep_f = img_deep_u8.astype(np.float32) / 255.0

    illum = maps["illumination"]
    noise = maps["noise_map"]
    edge = maps["edge_map"]

    h, w, _ = orig_f.shape

    if mode == "hybrid_5050":
        # Baseline trivial blend: 50% Classical + 50% Deep
        fused_f = 0.5 * class_f + 0.5 * deep_f
        w_orig = np.zeros((h, w), dtype=np.float32)
        w_class = np.full((h, w), 0.5, dtype=np.float32)
        w_deep = np.full((h, w), 0.5, dtype=np.float32)

    elif mode == "hybrid_illum":
        # Illumination-only guidance
        # Deep dominates in dark (1 - illum); original in highlights; classical in midtones
        logit_deep = (1.0 - illum) * 3.0
        logit_orig = np.clip((illum - 0.6) * 6.0, 0.0, 4.0)
        logit_class = np.full_like(illum, 1.0)

        logits = np.stack([logit_orig, logit_class, logit_deep], axis=-1)
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        weights = exp_l / (np.sum(exp_l, axis=-1, keepdims=True) + 1e-6)

        w_orig = weights[:, :, 0]
        w_class = weights[:, :, 1]
        w_deep = weights[:, :, 2]
        fused_f = (
            w_orig[:, :, None] * orig_f
            + w_class[:, :, None] * class_f
            + w_deep[:, :, None] * deep_f
        )

    else:
        # Full guidance: Illumination + Noise + Structure
        # 1. High noise + deep darkness -> trust Deep network
        # 2. Strong edges in moderate light -> trust Classical guided Retinex
        # 3. Already bright/highlight pixels -> preserve Original
        logit_deep = (1.0 - illum) * 2.5 + noise * 2.0 - edge * 0.5
        logit_class = 1.0 + edge * 2.0 - noise * 1.5
        logit_orig = np.clip((illum - 0.55) * 6.0, 0.0, 5.0)

        logits = np.stack([logit_orig, logit_class, logit_deep], axis=-1)
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        weights = exp_l / (np.sum(exp_l, axis=-1, keepdims=True) + 1e-6)

        guide_gray = (0.299 * orig_f[:, :, 0] + 0.587 * orig_f[:, :, 1] + 0.114 * orig_f[:, :, 2]).astype(np.float32)

        # Smooth weights with guided filter to prevent spatial seam artifacts
        w_orig = np.clip(_guided_filter_gray(guide_gray, weights[:, :, 0], radius=8, eps=1e-3), 0.0, 1.0)
        w_class = np.clip(_guided_filter_gray(guide_gray, weights[:, :, 1], radius=8, eps=1e-3), 0.0, 1.0)
        w_deep = np.clip(_guided_filter_gray(guide_gray, weights[:, :, 2], radius=8, eps=1e-3), 0.0, 1.0)

        # Re-normalize weights so sum == 1.0
        w_sum = w_orig + w_class + w_deep + 1e-6
        w_orig /= w_sum
        w_class /= w_sum
        w_deep /= w_sum

        fused_f = (
            w_orig[:, :, None] * orig_f
            + w_class[:, :, None] * class_f
            + w_deep[:, :, None] * deep_f
        )

    # Quality Guard: Highlight soft-knee protection
    if quality_guard:
        # Where original was already bright (> 0.75), prevent blown out highlights
        orig_max = np.max(orig_f, axis=2)
        highlight_mask = np.clip((orig_max - 0.70) / 0.25, 0.0, 1.0)[:, :, None]
        fused_f = (1.0 - highlight_mask * 0.8) * fused_f + (highlight_mask * 0.8) * orig_f

    fused_f = np.clip(fused_f, 0.0, 1.0)
    fused_u8 = np.clip(np.round(fused_f * 255.0), 0, 255).astype(np.uint8)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    fusion_maps = {
        "weight_orig": w_orig.astype(np.float32),
        "weight_classical": w_class.astype(np.float32),
        "weight_deep": w_deep.astype(np.float32),
    }

    params = {
        "fusion_mode": mode,
        "quality_guard": quality_guard,
        "mean_weight_orig": round(float(np.mean(w_orig)), 4),
        "mean_weight_class": round(float(np.mean(w_class)), 4),
        "mean_weight_deep": round(float(np.mean(w_deep)), 4),
    }

    return fused_u8, fusion_maps, params, elapsed_ms
