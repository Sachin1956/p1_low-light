"""
Master pipeline execution for enhance(rgb_uint8, config).
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from .classical import run_classical_pipeline
from .deep_adapter import run_deep_inference
from .fusion import adaptive_fusion
from .interface import EnhancementConfig, EnhancementResult


def run_enhancement_pipeline(
    rgb_uint8: np.ndarray,
    config: EnhancementConfig,
) -> EnhancementResult:
    """Dispatches requested mode and records full execution metadata."""
    total_start = time.perf_counter()
    mode = config.mode.lower()

    all_maps: Dict[str, np.ndarray] = {}
    params: Dict[str, Any] = {"requested_mode": mode}
    ckpt_hash = None

    if mode == "identity":
        out_u8 = rgb_uint8.copy()

    elif mode == "gamma":
        # Fixed gamma = 0.5 baseline (E0)
        f = (rgb_uint8.astype(np.float32) / 255.0).clip(0.0, 1.0)
        out_f = np.power(f, 0.5)
        out_u8 = np.clip(np.round(out_f * 255.0), 0, 255).astype(np.uint8)

    elif mode in ("classical_fixed", "classical_adaptive"):
        adaptive = (mode == "classical_adaptive")
        out_u8, all_maps, c_params, _ = run_classical_pipeline(rgb_uint8, adaptive=adaptive)
        params.update(c_params)

    elif mode == "deep":
        if not config.checkpoint_path:
            raise ValueError("mode='deep' requires a valid config.checkpoint_path.")
        out_u8, d_params, _, ckpt_hash = run_deep_inference(
            rgb_uint8,
            checkpoint_path=config.checkpoint_path,
            device=config.device,
            use_amp=config.use_amp,
            gamma_input=config.gamma_input,
        )
        params.update(d_params)

    elif mode.startswith("hybrid") or mode in ("hybrid_5050", "hybrid_illum", "hybrid_guided", "hybrid_guarded"):
        # Run Classical branch to get both restoration and feature maps
        c_u8, c_maps, c_params, _ = run_classical_pipeline(rgb_uint8, adaptive=config.classical_adaptive)
        all_maps.update(c_maps)

        # Run Deep branch
        d_u8, d_params, _, ckpt_hash = run_deep_inference(
            rgb_uint8,
            checkpoint_path=config.checkpoint_path or "weights/cidnet/CIDNETweights/lolv2/best_PSNR.pth",
            device=config.device,
            use_amp=config.use_amp,
            gamma_input=config.gamma_input,
        )

        # Run Adaptive Spatial Fusion
        out_u8, f_maps, f_params, _ = adaptive_fusion(
            img_orig_u8=rgb_uint8,
            img_class_u8=c_u8,
            img_deep_u8=d_u8,
            maps=all_maps,
            mode=mode,
            quality_guard=config.quality_guard,
        )
        all_maps.update(f_maps)
        params["classical_branch"] = c_params
        params["deep_branch"] = d_params
        params["fusion"] = f_params

    else:
        raise ValueError(f"Unknown enhancement mode: '{config.mode}'")

    total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

    return EnhancementResult(
        image=out_u8,
        maps=all_maps,
        params=params,
        elapsed_ms=round(total_elapsed_ms, 2),
        branch_version="p1-v1.0",
        config_hash=config.config_hash,
        checkpoint_hash=ckpt_hash,
    )
