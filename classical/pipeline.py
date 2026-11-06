"""
Classical Low-Light Enhancement Pipeline and Shared Interface.
Implements enhance(rgb_uint8, config) -> EnhancementResult adhering to P1 Contract.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from classical.config import ClassicalConfig, EnhancementResult, PipelineMode
from classical.features import extract_features_and_maps
from classical.filters import (
    apply_conservative_color_correction,
    apply_edge_preserving_denoise,
    apply_highlight_guard_blend,
    apply_luminance_clahe,
    apply_retinex_exposure_boost,
    decompose_and_refine_illumination,
)
from classical.mappings import map_features_to_parameters


def validate_input_image(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Validate input image structure, dimensionality, and channel count.
    Ensures sRGB HxWx3 uint8 array without NaN or Inf values.
    """
    if rgb_uint8 is None:
        raise ValueError("Input image cannot be None.")
    if not isinstance(rgb_uint8, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(rgb_uint8)}")
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {rgb_uint8.shape}")
    if rgb_uint8.size == 0 or rgb_uint8.shape[0] < 2 or rgb_uint8.shape[1] < 2:
        raise ValueError(f"Invalid image dimensions: {rgb_uint8.shape}")

    # Ensure uint8
    if rgb_uint8.dtype != np.uint8:
        if np.issubdtype(rgb_uint8.dtype, np.floating):
            # Check bounds and convert
            if np.isnan(rgb_uint8).any() or np.isinf(rgb_uint8).any():
                raise ValueError("Input image contains NaN or Inf values.")
            if rgb_uint8.max() <= 1.05 and rgb_uint8.min() >= -0.05:
                rgb_uint8 = np.clip(rgb_uint8 * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)
        else:
            rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(rgb_uint8)


def enhance(
    rgb_uint8: np.ndarray,
    config: Optional[ClassicalConfig] = None,
) -> EnhancementResult:
    """
    Shared interface execution boundary:
    enhance(rgb_uint8: HxWx3, config) -> EnhancementResult

    Inputs:
        rgb_uint8: HxWx3 uint8 array in sRGB color space.
        config: Optional ClassicalConfig (defaults to E2_ADAPTIVE).

    Returns:
        EnhancementResult containing:
            - image: restored HxWx3 uint8 sRGB array
            - maps: diagnostic HxW float32 maps (illumination, edge, noise, highlights)
            - params: selected runtime parameter values
            - elapsed_ms: end-to-end execution latency in milliseconds
            - branch_version: string identifier of code version
            - config_hash: SHA256 hex digest of runtime config
            - checkpoint_hash: None for classical CV
    """
    t_start = time.perf_counter()

    if config is None:
        config = ClassicalConfig(mode=PipelineMode.E2_ADAPTIVE)

    # 1. Input validation & float conversion
    img_u8 = validate_input_image(rgb_uint8)
    h, w, c = img_u8.shape
    img_f = (img_u8.astype(np.float32) / 255.0)

    # 2. Feature analysis and diagnostic map extraction
    features, maps = extract_features_and_maps(
        img_f,
        highlight_threshold=config.highlight_threshold,
    )

    # 3. Dynamic parameter mapping
    params = map_features_to_parameters(features, config)

    # 4. Pipeline Execution
    if config.mode == PipelineMode.E0_BASELINE:
        # Fixed baseline gamma lift: I_out = clip(I ^ gamma, 0, 1)
        gamma = params.gamma
        out_f = np.power(img_f, gamma)
        out_f = np.clip(out_f, 0.0, 1.0)
        refined_illum = maps["illumination_raw"]
    else:
        # Illumination Decomposition and Guided Refinement
        T0, T_refined = decompose_and_refine_illumination(
            img_f,
            luminance_diagnostic=maps["luminance_diagnostic"],
            radius=config.illumination_refinement_radius,
            eps=config.illumination_refinement_eps,
        )
        maps["illumination_refined"] = T_refined

        # Stage 1: Retinex Illumination Compensation (LIME-inspired)
        retinex_f = apply_retinex_exposure_boost(
            img_f,
            refined_illumination=T_refined,
            gamma=params.gamma,
            gain_cap=params.gain_cap,
            tau_floor=config.tau_floor,
        )

        # Stage 2: Edge-Preserving Denoising
        if config.enable_denoise and params.denoise_strength > 0:
            denoised_f = apply_edge_preserving_denoise(
                retinex_f,
                denoise_strength=params.denoise_strength,
            )
        else:
            denoised_f = retinex_f

        # Stage 3: Luminance-only CLAHE
        if config.enable_clahe and params.clahe_clip_limit > 0:
            clahe_f = apply_luminance_clahe(
                denoised_f,
                clip_limit=params.clahe_clip_limit,
                grid_size=config.clahe_grid_size,
                highlight_mask=maps["highlight_mask"],
            )
        else:
            clahe_f = denoised_f

        # Stage 4: Conservative Color & Contrast Enhancement
        if config.enable_color_correction:
            color_f = apply_conservative_color_correction(
                clahe_f,
                saturation_factor=config.saturation_factor,
            )
        else:
            color_f = clahe_f

        # Stage 5: Highlight Guard (soft blend with input in bright regions)
        if config.enable_highlight_guard:
            out_f = apply_highlight_guard_blend(
                input_rgb_float=img_f,
                enhanced_rgb_float=color_f,
                highlight_mask=maps["highlight_mask"],
            )
        else:
            out_f = color_f

    # 5. Output packaging
    out_u8 = np.clip(np.round(out_f * 255.0), 0, 255).astype(np.uint8)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return EnhancementResult(
        image=out_u8,
        maps=maps,
        params=params.to_dict(),
        elapsed_ms=float(elapsed_ms),
        branch_version=config.branch_version,
        config_hash=config.compute_hash(),
        checkpoint_hash=None,
    )


def run_batch_inference(
    manifest_records: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    config: Optional[ClassicalConfig] = None,
) -> Path:
    """
    Run deterministic batch inference over a list of image records.
    Writes one lossless PNG per image ID and a JSONL provenance record.

    Adheres strictly to the batch specification in docs/P1_ML_EXECUTION_PLAN.md.
    """
    out_dir = Path(output_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "inference_log.jsonl"

    if config is None:
        config = ClassicalConfig(mode=PipelineMode.E2_ADAPTIVE)

    records = []
    with jsonl_path.open("w", encoding="utf-8") as f_jsonl:
        for rec in manifest_records:
            image_id = str(rec["image_id"])
            input_path = Path(rec["input_path"])

            # Read image file bytes for SHA256
            input_bytes = input_path.read_bytes()
            input_sha256 = hashlib.sha256(input_bytes).hexdigest()

            # Decode image
            bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if bgr is None:
                err_rec = {
                    "image_id": image_id,
                    "input_sha256": input_sha256,
                    "output_sha256": None,
                    "branch_version": config.branch_version,
                    "config_hash": config.compute_hash(),
                    "checkpoint_hash": None,
                    "elapsed_ms": 0.0,
                    "error": f"Failed to read image at {input_path}",
                }
                f_jsonl.write(json.dumps(err_rec) + "\n")
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = enhance(rgb, config)

            # Save lossless PNG output
            out_filename = f"{image_id}.png"
            out_path = images_dir / out_filename
            bgr_out = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), bgr_out)

            out_bytes = out_path.read_bytes()
            output_sha256 = hashlib.sha256(out_bytes).hexdigest()

            log_entry = {
                "image_id": image_id,
                "input_sha256": input_sha256,
                "output_sha256": output_sha256,
                "branch_version": result.branch_version,
                "config_hash": result.config_hash,
                "checkpoint_hash": None,
                "elapsed_ms": result.elapsed_ms,
                "params": result.params,
                "error": None,
            }
            f_jsonl.write(json.dumps(log_entry) + "\n")
            records.append(log_entry)

    return jsonl_path
