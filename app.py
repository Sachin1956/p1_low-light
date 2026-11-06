"""
Problem Statement 1 (P1): Robust Low-Light Image Restoration and Enhancement.
Live Interactive Web Application (Gradio).

Features:
- Side-by-side comparison: Original, Classical-Only, Deep-Only (HVI-CIDNet), and Adaptive Hybrid (ARHR).
- Real-time diagnostic maps inspection: Illumination, Noise Validity, Edges, Highlight Protection, and Fusion Weights.
- Full parameter transparency and measured per-pipeline latency.
- Canonical presets from `data/handoff_fixtures/` plus arbitrary judge image upload.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

import classical
from classical.config import ClassicalConfig, PipelineMode
import fusion
from fusion.config import FusionConfig
import models
from models.inference import enhance_deep


def process_image(
    input_image: Optional[np.ndarray],
    preset_choice: Optional[str] = None,
    mode_selection: str = "Adaptive Hybrid",
    gamma_override: Optional[float] = None,
) -> Tuple[
    np.ndarray,  # Enhanced primary output
    np.ndarray,  # Classical output
    np.ndarray,  # Deep output
    np.ndarray,  # Hybrid output
    np.ndarray,  # Illumination map
    np.ndarray,  # Edge map
    np.ndarray,  # Highlight mask
    np.ndarray,  # Spatial weights visualization (RGB)
    str,         # Execution summary markdown
]:
    """Process an input image across Classical, Deep, and Hybrid pipelines."""
    if input_image is None:
        raise gr.Error("Please upload an image or choose one of the preset handoff fixtures.")

    t0_total = time.perf_counter()
    rgb = np.asarray(input_image, dtype=np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[..., :3]

    h, w, _ = rgb.shape

    # 1. Classical Branch Execution
    cfg_classical = ClassicalConfig(mode=PipelineMode.E2_ADAPTIVE)
    if gamma_override is not None and 0.4 <= gamma_override <= 1.0:
        cfg_classical = ClassicalConfig(mode=PipelineMode.E1_FIXED, fixed_gamma=gamma_override)

    res_c = classical.enhance(rgb, cfg_classical)
    img_classical = res_c.image
    lat_c = res_c.elapsed_ms

    # 2. Deep Branch Execution (HVI-CIDNet)
    t0_d = time.perf_counter()
    img_d_float = enhance_deep(rgb)
    img_deep = np.clip(np.round(img_d_float * 255.0), 0, 255).astype(np.uint8)
    lat_d = (time.perf_counter() - t0_d) * 1000.0

    # 3. Adaptive Spatial Fusion (Part C)
    t0_f = time.perf_counter()
    cfg_fusion = FusionConfig()
    res_hybrid = fusion.fuse_adaptive(rgb, img_classical, img_deep, cfg_fusion, precomputed_maps=res_c.maps)
    img_hybrid = res_hybrid.image
    lat_f = res_hybrid.elapsed_ms

    total_latency_ms = (time.perf_counter() - t0_total) * 1000.0

    # Diagnostic maps for visualization
    illum_map = (np.clip(res_c.maps["illumination_refined"], 0.0, 1.0) * 255.0).astype(np.uint8)
    edge_map = (np.clip(res_c.maps["edge_map"] * 3.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    highlight_mask = (np.clip(res_c.maps["highlight_mask"], 0.0, 1.0) * 255.0).astype(np.uint8)

    # Fusion weight visualization as RGB image (R=Original, G=Classical, B=Deep)
    weights_rgb = (np.clip(res_hybrid.weights, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Select primary display output
    if mode_selection == "Classical Only":
        primary_out = img_classical
    elif mode_selection == "Deep Learning Only":
        primary_out = img_deep
    else:
        primary_out = img_hybrid

    # Summary report
    p = res_c.params
    fw = res_hybrid.params
    summary_md = rf"""
### 📊 Measured Runtime Diagnostics & Parameters

| Metric / Parameter | Value |
|---|---|
| **Input Resolution** | `{w} × {h}` px |
| **Total Pipeline Latency** | **`{total_latency_ms:.1f} ms`** (`{total_latency_ms/1000:.3f} s`) |
| **Classical Execution** | `{lat_c:.1f} ms` |
| **Deep HVI-CIDNet Execution** | `{lat_d:.1f} ms` |
| **Adaptive Spatial Fusion** | `{lat_f:.1f} ms` |
| **Retinex Gamma ($\gamma$)** | `{p['gamma']:.3f}` |
| **Illumination Gain Cap** | `{p['gain_cap']:.2f}×` |
| **Bilateral Denoise Strength** | `{p['denoise_strength']:.4f}` (Noise adapted: `{p['noise_adapted']}`) |
| **Luminance CLAHE Clip Limit** | `{p['clahe_clip_limit']:.2f}` |
| **Original Weight $W_o$ (Highs)** | `{fw['mean_weight_original']*100:.1f}%` |
| **Classical Weight $W_c$ (Edges)**| `{fw['mean_weight_classical']*100:.1f}%` |
| **Deep Weight $W_d$ (Shadows)**   | `{fw['mean_weight_deep']*100:.1f}%` |
| **Quality Guard Active** | `{fw['quality_guard_enabled']}` |
"""

    return (
        primary_out,
        img_classical,
        img_deep,
        img_hybrid,
        illum_map,
        edge_map,
        highlight_mask,
        weights_rgb,
        summary_md,
    )


def create_demo() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )

    with gr.Blocks(title="P1: Robust Low-Light Image Restoration (ARHR)", theme=theme) as demo:
        gr.Markdown(
            """
            # 🌙 Problem Statement 1 (P1): Robust Low-Light Image Restoration & Enhancement
            ### Adaptive Retinex-Guided Hybrid Restoration (ARHR)
            **Team 01:** Rohit Ajit Bharadwaj (`1RUA24CSE0375`), Rohith P Hegde (`1RUA24CSE0376`), S Sachithananthan (`1RUA24CSE0384`), Saket M (`1RUA24CSE0395`)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(label="Input Low-Light Image", type="numpy")

                with gr.Accordion("⚙️ Pipeline Controls", open=False):
                    mode_sel = gr.Radio(
                        ["Adaptive Hybrid (Recommended)", "Classical Only", "Deep Learning Only"],
                        value="Adaptive Hybrid (Recommended)",
                        label="Primary View Selection",
                    )
                    gamma_slider = gr.Slider(
                        minimum=0.40,
                        maximum=1.00,
                        value=0.75,
                        step=0.05,
                        label="Manual Fixed Gamma Override (optional)",
                    )

                btn_enhance = gr.Button("⚡ Enhance Image", variant="primary", size="lg")

                gr.Markdown("### 📂 Canonical Handoff Presets")
                preset_paths = list(Path("data/handoff_fixtures").glob("*.png"))
                preset_paths = sorted([str(p) for p in preset_paths])
                if preset_paths:
                    gr.Examples(
                        examples=preset_paths,
                        inputs=input_img,
                        label="Click to load a benchmark test scene",
                    )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("🖼️ Restored Output Comparison"):
                        with gr.Row():
                            out_primary = gr.Image(label="Enhanced Output", interactive=False)
                        with gr.Row():
                            out_classical = gr.Image(label="Classical Branch Only", interactive=False)
                            out_deep = gr.Image(label="Deep Learning Only (HVI-CIDNet)", interactive=False)
                            out_hybrid = gr.Image(label="Adaptive Hybrid (ARHR)", interactive=False)

                    with gr.TabItem("🔍 Diagnostic & Analysis Maps"):
                        gr.Markdown("Real-time intermediate maps derived directly during the enhancement pass:")
                        with gr.Row():
                            out_illum = gr.Image(label="Refined Illumination Map (T)", interactive=False)
                            out_edge = gr.Image(label="Sobel Edge Map (E)", interactive=False)
                        with gr.Row():
                            out_highlight = gr.Image(label="Highlight Protection Mask (M_h)", interactive=False)
                            out_weights = gr.Image(
                                label="Spatial Fusion Map (R=Original, G=Classical, B=Deep)", interactive=False
                            )

                    with gr.TabItem("📈 Quantitative Telemetry"):
                        out_summary = gr.Markdown("Click **Enhance Image** to compute telemetry.")

        btn_enhance.click(
            fn=process_image,
            inputs=[input_img, gr.State(None), mode_sel, gamma_slider],
            outputs=[
                out_primary,
                out_classical,
                out_deep,
                out_hybrid,
                out_illum,
                out_edge,
                out_highlight,
                out_weights,
                out_summary,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
