"""P1 restoration demo with three outputs and input-derived diagnostics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

from p1.deep_adapter import load_cidnet_model
from p1.fusion import adaptive_fusion
from p1.interface import EnhancementConfig, enhance

_REPO_ROOT = Path(__file__).resolve().parent
_CKPT_PATH = _REPO_ROOT / "weights/kaggle_run_v3/p1_multidataset_output/best_model.pth"
_EVIDENCE = _REPO_ROOT / "outputs/submission"
MAX_LONG_SIDE = 768
MAX_PIXELS = 24_000_000


def _colorize_map(f32_map: np.ndarray, colormap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """Normalize float32 map in [0, 1] and apply false color for visual inspection."""
    u8 = np.clip(np.round(f32_map * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, colormap), cv2.COLOR_BGR2RGB)


def process_image(
    input_img: Optional[np.ndarray],
    fusion_mode: str,
    enable_quality_guard: bool,
    adaptive_classical: bool,
) -> Tuple[
    Optional[np.ndarray],  # Classical
    Optional[np.ndarray],  # Deep
    Optional[np.ndarray],  # Hybrid
    Optional[np.ndarray],  # Illum map
    Optional[np.ndarray],  # Noise map
    Optional[np.ndarray],  # Edge map
    Optional[np.ndarray],  # Fusion weights visualization
    str,                   # Diagnostics text
]:
    start = time.perf_counter()
    if input_img is None:
        return None, None, None, None, None, None, None, "Please upload or select an input image."

    if input_img.dtype != np.uint8 or input_img.ndim not in (2, 3):
        raise ValueError("Upload an 8-bit RGB, grayscale or RGBA image.")
    if min(input_img.shape[:2]) < 8 or np.prod(input_img.shape[:2]) > MAX_PIXELS:
        raise ValueError("Image dimensions must be at least 8×8 and at most 24 megapixels.")
    if fusion_mode not in {"hybrid_guarded", "hybrid_illum", "hybrid_5050"}:
        raise ValueError("Unknown fusion strategy.")

    # Validate shape & type (handles grayscale, RGBA, and standard sRGB)
    if input_img.ndim == 2:
        input_img = np.stack([input_img, input_img, input_img], axis=-1)
    elif input_img.ndim == 3 and input_img.shape[2] == 4:
        rgb = input_img[..., :3].astype(np.float32)
        alpha = (input_img[..., 3:4].astype(np.float32) / 255.0).clip(0.0, 1.0)
        input_img = np.clip(np.round(rgb * alpha), 0, 255).astype(np.uint8)
    elif input_img.ndim != 3 or input_img.shape[2] != 3:
        raise ValueError("Upload an RGB, grayscale or RGBA image.")

    orig_h, orig_w, _ = input_img.shape

    # Section 8 live mode resolution policy: max long side 768 preserving aspect ratio
    if max(orig_h, orig_w) > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / float(max(orig_h, orig_w))
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        if min(new_w, new_h) < 8:
            raise ValueError("Image is too narrow to process at the demo resolution.")
        input_img = cv2.resize(input_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    h, w, _ = input_img.shape
    input_img = np.require(input_img, requirements=["C", "W"])

    # 1. Classical-only run
    c_mode = "classical_adaptive" if adaptive_classical else "classical_fixed"
    cfg_c = EnhancementConfig(mode=c_mode, device="cpu")
    t0 = time.perf_counter()
    res_c = enhance(input_img, cfg_c)
    t_c = (time.perf_counter() - t0) * 1000.0

    # 2. Deep-only run
    cfg_d = EnhancementConfig(mode="deep", checkpoint_path=str(_CKPT_PATH), device="cpu")
    t0 = time.perf_counter()
    res_d = enhance(input_img, cfg_d)
    t_d = (time.perf_counter() - t0) * 1000.0

    # 3. Hybrid run (fuse computed Classical and Deep candidates directly)
    t0 = time.perf_counter()
    hybrid_u8, f_maps, _, _ = adaptive_fusion(
        input_img,
        res_c.image,
        res_d.image,
        res_c.maps,
        mode=fusion_mode,
        quality_guard=enable_quality_guard,
    )
    t_h = (time.perf_counter() - t0) * 1000.0

    illum_vis = _colorize_map(res_c.maps["illumination"], cv2.COLORMAP_INFERNO)
    noise_vis = _colorize_map(res_c.maps["noise_map"], cv2.COLORMAP_HOT)
    edge_vis = _colorize_map(res_c.maps["edge_map"], cv2.COLORMAP_MAGMA)

    # Colorize RGB weights map (R=Original, G=Classical, B=Deep)
    w_o = f_maps["weight_orig"]
    w_c = f_maps["weight_classical"]
    w_d = f_maps["weight_deep"]
    fusion_weights_rgb = np.stack([
        np.clip(w_o * 255.0, 0, 255).astype(np.uint8),
        np.clip(w_c * 255.0, 0, 255).astype(np.uint8),
        np.clip(w_d * 255.0, 0, 255).astype(np.uint8),
    ], axis=-1)

    res_info = f"{w}×{h} px"
    if (orig_w, orig_h) != (w, h):
        res_info += f" (resized from {orig_w}×{orig_h})"
    elapsed = (time.perf_counter() - start) * 1000

    diag = f"""### Execution Diagnostics
- **Resolution**: {res_info}
- **Classical Latency**: {t_c:.1f} ms (selected gamma: {res_c.params.get('selected_gamma', 'N/A')}, CLAHE clip: {res_c.params.get('selected_clahe_clip', 'N/A')})
- **Deep Model Latency**: {t_d:.1f} ms (Checkpoint: `{Path(str(_CKPT_PATH)).name}`, SHA: `{res_d.checkpoint_hash[:12] if res_d.checkpoint_hash else 'N/A'}`)
- **Hybrid Fusion Latency**: {t_h:.1f} ms
- **Server processing time**: {elapsed / 1000:.2f} s (upload, queue and browser display measured separately)
- **Mean pre-guard fusion weights**: Original {np.mean(w_o)*100:.1f}%, Classical {np.mean(w_c)*100:.1f}%, Deep {np.mean(w_d)*100:.1f}%
- **Quality Guard**: {'Active (Highlight protection soft-knee)' if enable_quality_guard else 'Disabled'}
"""
    return (
        res_c.image,
        res_d.image,
        hybrid_u8,
        illum_vis,
        noise_vis,
        edge_vis,
        fusion_weights_rgb,
        diag,
    )


def create_app() -> gr.Blocks:
    sample_dir = _REPO_ROOT / "data" / "handoff_fixtures"
    examples = [str(p) for p in sorted(sample_dir.glob("*.png"))]

    def submit(image, strategy, guard, adaptive):
        try:
            return process_image(image, strategy, guard, adaptive)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    def read_evidence(name):
        path = _EVIDENCE / name
        return path.read_text(encoding="utf-8") if path.is_file() else "Evaluation evidence is pending."

    with gr.Blocks(title="P1: Low-Light Restoration & Enhancement", delete_cache=(300, 3600), analytics_enabled=False) as demo:
        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown("# 🌙 P1: Robust Low-Light Image Restoration & Enhancement")
            gr.Markdown("**Adaptive Retinex–HVI Restoration (ARHR) System** | Hackathon P1 Submission")
            gr.Markdown("Upload a low-light photo to compare three methods. Use images you can share with the demo host. Uploads and outputs are cached temporarily.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Low-Light Image", type="numpy", image_mode=None, sources=["upload", "clipboard"])
                
                with gr.Accordion("Pipeline & Fusion Controls", open=True):
                    fusion_mode = gr.Dropdown(
                        label="Hybrid Fusion Strategy",
                        choices=[
                            ("Adaptive Guided Fusion + Quality Guard", "hybrid_guarded"),
                            ("Illumination Guidance Only", "hybrid_illum"),
                            ("Trivial 50/50 Baseline Blend", "hybrid_5050"),
                        ],
                        value="hybrid_guarded",
                    )
                    enable_guard = gr.Checkbox(label="Enable Quality Guard (Highlight Protection)", value=True)
                    adaptive_classical = gr.Checkbox(label="Adaptive Classical Parameters", value=True)
                
                run_btn = gr.Button("🚀 Restore & Enhance", variant="primary", size="lg")
                
                if examples:
                    gr.Examples(examples=examples, inputs=input_image, label="Synthetic demo fixtures")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("🖼️ Multi-Branch Comparative Restoration"):
                        with gr.Row():
                            out_classical = gr.Image(label="1. Classical Only (Retinex+CLAHE)")
                            out_deep = gr.Image(label="2. Deep Only (HVI-CIDNet Adapted)")
                        with gr.Row():
                            out_hybrid = gr.Image(label="3. Adaptive Hybrid (ARHR Proposed)", interactive=False)
                    
                    with gr.TabItem("🗺️ Intermediate Analysis & Fusion Maps"):
                        with gr.Row():
                            map_illum = gr.Image(label="Illumination Map (Guided Retinex)")
                            map_noise = gr.Image(label="Noise Estimate Map (High-Freq)")
                        with gr.Row():
                            map_edge = gr.Image(label="Structure Edge Map (Sobel)")
                            map_weights = gr.Image(label="Spatial Fusion Weights (R=Orig, G=Class, B=Deep)")
                        gr.Markdown("Maps come from this upload. Noise is a local high-frequency estimate. The RGB fusion map shows weights before the highlight guard.")

                    with gr.TabItem("Benchmark & limitations"):
                        benchmark = gr.Markdown(read_evidence("summary.md"))
                        failures = gr.Markdown(read_evidence("failures.md"))

                diagnostics_box = gr.Markdown("Click **Restore & Enhance** to process.")

        run_btn.click(
            fn=submit,
            inputs=[input_image, fusion_mode, enable_guard, adaptive_classical],
            outputs=[
                out_classical,
                out_deep,
                out_hybrid,
                map_illum,
                map_noise,
                map_edge,
                map_weights,
                diagnostics_box,
            ], api_name="restore", concurrency_limit=1,
        )
        demo.load(
            lambda: (read_evidence("summary.md"), read_evidence("failures.md")),
            outputs=[benchmark, failures], api_visibility="private",
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="P1 Low-Light Interactive Application")
    parser.add_argument("--share", action="store_true", help="Generate public shareable Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Local server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP")
    parser.add_argument("--max-file-size", default="15mb", help="Maximum size per upload")
    args = parser.parse_args()

    torch.set_num_threads(4)
    _, checkpoint_hash = load_cidnet_model(_CKPT_PATH, device="cpu")
    print(f"Loaded checkpoint SHA-256 {checkpoint_hash}", flush=True)
    app = create_app()
    print(f"Launching application on {args.host}:{args.port} (share={args.share})...", flush=True)
    app.launch(
        share=args.share, server_name=args.host, server_port=args.port,
        max_file_size=args.max_file_size,
        blocked_paths=[str(_REPO_ROOT / p) for p in (".git", ".venv", "weights", "data/submission_lolv2")],
    )
